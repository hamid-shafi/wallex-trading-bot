from dataclasses import asdict, dataclass
import math
from .model import Learner


@dataclass
class Position:
    quantity: float
    cost: float
    entry: float
    stop: float
    opened: int
    liquidating: bool = False
    dust_topups: int = 0


@dataclass
class Intent:
    symbol: str
    side: str
    quantity: float
    reason: str
    stop_fraction: float = 0.0
    decision_ts: int = 0


class Portfolio:
    def __init__(self, cash):
        self.initial = cash
        self.cash = cash
        self.positions = {}
        self.fees = 0.0
        self.turnover = 0.0
        self.realized = 0.0
        self.fills = []
        self.round_trips = []
        self.last_exit = {}

    def equity(self, prices):
        return self.cash + sum(p.quantity * prices[s] for s, p in self.positions.items())

    def apply(self, intent, quantity, price, fee, ts):
        """Fees here are quote fees. Live base-asset fees are normalized before calling."""
        s = intent.symbol
        if quantity <= 0 or price <= 0 or fee < 0:
            raise ValueError("invalid fill")
        notional = quantity*price
        pnl = 0.0
        if intent.side == "BUY":
            cleanup = intent.reason=="dust_consolidation_for_exit"
            if s in self.positions and not cleanup:
                raise ValueError("pyramiding disabled")
            if s in self.positions and self.positions[s].dust_topups>=1:
                raise ValueError("only one dust consolidation per position")
            if notional + fee > self.cash + 1e-6:
                raise ValueError("fill exceeds cash")
            self.cash -= notional + fee
            if s in self.positions:
                p = self.positions[s]
                p.quantity += quantity
                p.cost += notional+fee
                p.entry = p.cost/p.quantity
                p.liquidating = True
                p.dust_topups += 1
            else:
                self.positions[s] = Position(quantity, notional+fee, price,
                                             price*(1-intent.stop_fraction), ts)
        elif intent.side == "SELL":
            p = self.positions[s]
            if quantity > p.quantity + 1e-10:
                raise ValueError("cannot sell more than inventory")
            quantity = min(quantity, p.quantity)
            allocated = p.cost * quantity / p.quantity
            pnl = notional-fee-allocated
            self.cash += notional-fee
            self.realized += pnl
            self.round_trips.append(pnl)
            p.cost -= allocated
            p.quantity -= quantity
            p.liquidating = True
            if p.quantity < max(1e-12, quantity*1e-9):
                del self.positions[s]
                self.last_exit[s] = ts
        else:
            raise ValueError("unknown side")
        self.fees += fee
        self.turnover += notional
        record = {"ts": ts, "symbol": s, "side": intent.side, "quantity": quantity,
                  "price": price, "fee": fee, "pnl": pnl, "reason": intent.reason,
                  "decision_ts": intent.decision_ts}
        self.fills.append(record)
        return record


class Engine:
    def __init__(self, config):
        self.config = config
        self.learners = {s: Learner(config.horizon, config.uncertainty_penalty) for s in config.symbols}
        self.peak = config.initial_cash
        self.day = None
        self.day_equity = config.initial_cash
        self.halted = False
        self.daily_halted = False
        self.forecasts = {}
        self.trained = None
        if config.model_artifact:
            from .trained import TrainedPredictor
            from pathlib import Path
            import json
            raw=json.loads(Path(config.model_artifact).read_text(encoding="utf-8"))
            if raw.get("artifact_type")=="cost_meta":
                from .costaware import CostPredictor
                if config.strategy!="trend_meta":
                    raise ValueError("cost-meta artifact requires trend_meta policy")
                self.trained=CostPredictor(config.model_artifact,config.news_path)
            elif raw.get("artifact_type")=="trend_mixture":
                from .mixture import MixturePredictor
                if config.strategy!="forecast":
                    raise ValueError("trend mixture requires directional forecast policy")
                self.trained=MixturePredictor(config.model_artifact,config.news_path)
            elif raw.get("artifact_type")=="longterm_research":
                from .longterm import LongtermPredictor
                if config.strategy!="forecast":
                    raise ValueError("long-term research artifact requires forecast policy")
                self.trained=LongtermPredictor(config.model_artifact,config.news_path)
            else:
                self.trained = TrainedPredictor(config.model_artifact)
            if self.trained.artifact["horizon"]!=config.horizon:
                raise ValueError("trained model horizon mismatch")

    def observe(self, frame):
        self.forecasts = {s: self.learners[s].observe(frame[s]) for s in self.config.symbols}
        if self.trained:
            if self.trained.artifact.get('artifact_type') in ('trend_mixture','longterm_research'):
                self.trained.histories={s:self.learners[s].history for s in self.config.symbols}
            self.forecasts = {s:self.trained.forecast(s,self.learners[s].history,self.config.interval)
                              for s in self.config.symbols}
        return self.forecasts

    def loss_gate(self, equity, ts):
        c = self.config
        self.peak = max(self.peak, equity)
        # Challenge daily limits use Tehran local date (UTC+03:30; no current DST).
        day = (ts + 12600) // 86400
        if day != self.day:
            self.day, self.day_equity, self.daily_halted = day, equity, False
        if equity <= self.peak*(1-c.max_drawdown):
            self.halted = True
        if equity <= self.day_equity*(1-c.daily_loss):
            self.daily_halted = True
        return self.halted or self.daily_halted

    def plan(self, portfolio, frame, now=None, spread=None):
        c = self.config
        now = now if now is not None else next(iter(frame.values())).ts+c.interval
        if self.trained and (c.strategy=="trend_meta" or self.trained.artifact.get("artifact_type") in ("trend_mixture","longterm_research")):
            if self.trained.artifact.get('artifact_type') in ('trend_mixture','longterm_research'):
                self.trained.histories={s:self.learners[s].history for s in c.symbols}
            self.forecasts={s:self.trained.forecast(s,self.learners[s].history,c.interval,as_of=now) for s in c.symbols}
        prices = {s: b.close for s, b in frame.items()}
        equity = portfolio.equity(prices)
        blocked = self.loss_gate(equity, now)
        settle = bool(c.challenge_end and now >= c.challenge_end-c.settlement_buffer)
        entry_closed=settle or bool(c.challenge_end and c.entry_buffer_bars and now>=c.challenge_end-c.entry_buffer_bars*c.interval)
        decisions, intents = [], []
        cleanup_reserved = 0.0
        cleanup_risk = 0.0
        trends={}
        if c.strategy!="forecast":
            from .costaware import trend_state
            trends={s:trend_state(self.learners[s].history,c) for s in c.symbols}
        existing_risk = sum(max(0,prices[s]-p.stop)*p.quantity for s,p in portfolio.positions.items())
        for s, p in sorted(portfolio.positions.items()):
            f = self.forecasts.get(s)
            state=trends.get(s)
            if c.trailing_atr and f:
                # Update at this close for subsequent bars; no same-bar trailing-stop hindsight.
                p.stop=max(p.stop,prices[s]-c.trailing_atr*f.atr)
            why = None
            if p.liquidating:
                why = "continue_liquidation"
            elif settle:
                why = "challenge_settlement"
            elif blocked:
                why = "drawdown_or_daily_loss"
            elif prices[s] <= p.stop:
                why = "protective_stop"
            elif now-p.opened >= (c.max_holding_bars or c.horizon)*c.interval:
                why = "horizon_exit"
            elif c.strategy!="forecast" and state and state["exit"]:
                why = "trend_reversal"
            elif c.strategy=="forecast" and f and f.samples >= c.min_samples and f.mu < -c.edge_bps/10000:
                why = "model_negative_edge"
            if why:
                if p.quantity*prices[s]<c.min_notional:
                    # One bounded, algorithmic top-up makes a new exit order possible.
                    # Never hide dust or fabricate a fill below the exchange minimum.
                    budget = c.min_notional*1.05
                    gross_now = equity-portfolio.cash
                    time_ok = not c.challenge_end or now<c.challenge_end-c.interval
                    stop = max(c.min_stop_fraction,1-p.stop/prices[s])
                    if (p.dust_topups==0 and not blocked and time_ok and
                        cleanup_reserved+budget<=equity*c.dust_cleanup_fraction and
                        budget<=equity*c.position_cap-p.quantity*prices[s] and
                        cleanup_reserved+budget<=equity*c.gross_cap-gross_now and
                        cleanup_reserved+budget*(1+c.fee_bps/10000)<=portfolio.cash and
                        budget*stop<=equity*c.risk_fraction and
                        existing_risk+cleanup_risk+budget*stop<=equity*c.portfolio_stop_risk):
                        intents.append(Intent(s,"BUY",budget/prices[s],"dust_consolidation_for_exit",stop,now))
                        cleanup_reserved += budget*(1+c.fee_bps/10000)
                        cleanup_risk += budget*stop
                    decisions.append({"symbol":s,"reason":"dust_cleanup_planned" if any(
                        x.symbol==s and x.reason=="dust_consolidation_for_exit" for x in intents) else "untradeable_residual",
                        "residual_notional":p.quantity*prices[s],"topups":p.dust_topups})
                else:
                    intents.append(Intent(s, "SELL", p.quantity, why, decision_ts=now))
        gross = sum(p.quantity*prices[s] for s, p in portfolio.positions.items())
        risk = sum(max(0, prices[s]-p.stop)*p.quantity for s, p in portfolio.positions.items())
        risk += cleanup_risk
        reserved = cleanup_reserved
        gross += cleanup_reserved
        planned = len(portfolio.positions)
        candidates = sorted(self.forecasts.items(), key=lambda kv: -(kv[1].lower if kv[1] else -1e10))
        for s, f in candidates:
            reason = "no_features"
            q, stop = 0.0, 0.0
            sbps = spread.get(s, c.spread_bps) if spread else c.spread_bps
            costs = (2*c.fee_bps+sbps+2*c.slippage_bps)/10000
            if f:
                stop = max(c.min_stop_fraction, c.stop_atr*f.atr/prices[s])
                if f.samples < c.min_samples:
                    reason = "model_warmup"
                elif blocked or entry_closed:
                    reason = "risk_halt_or_settlement"
                elif s in portfolio.positions:
                    reason = "already_in_position"
                elif planned >= c.max_positions:
                    reason = "position_limit"
                elif now-portfolio.last_exit.get(s, 0) < c.cooldown*c.interval:
                    reason = "cooldown"
                elif sbps > c.max_spread_bps:
                    reason = "spread_too_wide"
                elif c.strategy!="forecast" and not (trends.get(s) or {}).get("entry",False):
                    reason = "trend_filter"
                elif c.strategy=="trend_meta" and f.contributions.get("net_profit_probability",0)<c.meta_min_probability:
                    reason = "meta_probability_too_low"
                elif c.news_required and f.contributions.get("news_coverage",0)<1:
                    reason = "news_unavailable"
                elif f.lower <= costs + c.edge_bps/10000:
                    reason = "edge_below_cost_and_uncertainty"
                else:
                    # Exposure/risk/cash/volume limits are applied jointly, including other intents.
                    budget = min(equity*c.position_cap, equity*c.gross_cap-gross,
                                 max(0, portfolio.cash-reserved)/(1+c.fee_bps/10000),
                                 equity*c.risk_fraction/stop,
                                 max(0, equity*c.portfolio_stop_risk-risk)/stop,
                                 frame[s].volume*prices[s]*c.participation)
                    if budget >= c.min_notional:
                        q = budget/prices[s]
                        reason = "learned_positive_net_edge"
                        intents.append(Intent(s, "BUY", q, reason, stop, now))
                        gross += budget
                        reserved += budget*(1+c.fee_bps/10000)
                        risk += budget*stop
                        planned += 1
                    else:
                        reason = "insufficient_risk_cash_or_liquidity_budget"
            decisions.append({"symbol": s, "reason": reason, "quantity": q,
                              "round_trip_cost": costs, "stop_fraction": stop,
                              "trend":trends.get(s),
                              "forecast": asdict(f) if f else None})
        return intents, decisions


def simulate_fill(portfolio, intent, bar, config, reference=None):
    # Reference=next open for normal orders; gap-through stops use the worse open.
    ref = bar.open if reference is None else reference
    drag = (config.spread_bps/2+config.slippage_bps)/10000
    price = ref*(1+drag if intent.side == "BUY" else 1-drag)
    used = sum(x["quantity"] for x in portfolio.fills if x["ts"]==bar.ts and x["symbol"]==intent.symbol)
    cap = max(0,bar.volume*config.participation-used)
    quantity = min(intent.quantity, cap)
    if intent.side == "BUY":
        if intent.symbol in portfolio.positions and intent.reason!="dust_consolidation_for_exit":
            return None
        quantity = min(quantity, portfolio.cash/(price*(1+config.fee_bps/10000)))
        if quantity*price < config.min_notional:
            return None
    else:
        if intent.symbol not in portfolio.positions:
            return None
        if intent.quantity*price < config.min_notional:
            return None  # exchange minimum still applies to a fresh exit order
        quantity = min(quantity, portfolio.positions[intent.symbol].quantity)
    if quantity <= 0:
        return None
    return portfolio.apply(intent, quantity, price, quantity*price*config.fee_bps/10000, bar.ts)
