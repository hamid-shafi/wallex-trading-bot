from dataclasses import asdict
from pathlib import Path
import json
import math
import os
import time
from .audit import Audit, canonical, manifest, source_hash
from .adaptive_risk import AdaptiveRisk, POLICY, reduce_entry, reduced_config
from .data import Bar, frames, load_bars
from .engine import Engine, Intent, Portfolio, Position
from .execution import Journal, LiveExecutor, UncertainOrder
from .fees import fetch_fee_rates
from .model import Forecast, Learner, Ridge
from .wallex import APIError,validate_market,round_precision


class RunLock:
    """OS file lock, released by OS after crash; prevents simultaneous engine processes."""
    def __init__(self,path):
        self.file = Path(path).open("a+b")
        self.file.seek(0)
        self.file.write(b"0")
        self.file.flush()
        self.file.seek(0)
        try:
            if os.name=="nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.file,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            raise RuntimeError("another engine owns this run directory")

    def close(self):
        self.file.close()


def dump_engine(engine):
    learners = {}
    for s,l in engine.learners.items():
        learners[s] = {"history":[asdict(b) for b in l.history],"pending":list(l.pending),
                       "last_ts":l.last_ts,"models":[m.__dict__ for m in l.models]}
    return {"learners":learners,"forecasts":{s:asdict(f) if f else None for s,f in engine.forecasts.items()},
            "peak":engine.peak,"day":engine.day,
            "day_equity":engine.day_equity,"halted":engine.halted,"daily_halted":engine.daily_halted}


def restore_engine(config,value):
    from collections import deque
    e = Engine(config)
    for s,raw in value["learners"].items():
        l = e.learners[s]
        l.history = [Bar(**b) for b in raw["history"]]
        l.pending = deque(raw["pending"])
        l.last_ts = raw["last_ts"]
        for m,state in zip(l.models,raw["models"]):
            m.__dict__.update(state)
    for k in ("peak","day","day_equity","halted","daily_halted"):
        setattr(e,k,value[k])
    e.forecasts = {s:Forecast(**f) if f else None for s,f in value.get("forecasts",{}).items()}
    return e


def dump_portfolio(p):
    return {**p.__dict__,"positions":{s:asdict(v) for s,v in p.positions.items()}}


def restore_portfolio(value):
    p = Portfolio(value["initial"])
    p.__dict__.update(value)
    p.positions = {s:Position(**v) for s,v in value["positions"].items()}
    return p


def open_order_list(obj):
    if isinstance(obj,list):
        return [x for child in obj for x in open_order_list(child)]
    if isinstance(obj,dict):
        if "clientOrderId" in obj:
            return [obj]
        return [x for child in obj.values() for x in open_order_list(child)]
    if obj is None:
        return []
    raise APIError("unrecognized open-order schema")


def check_balances(api,c,p,markets):
    raw = api.account(c,"balances")
    raw = raw.get("balances",raw)
    if not isinstance(raw,dict):
        raise APIError("credit balances need a verified schema mapping")
    for asset,row in raw.items():
        total = float(row["value"])
        locked = float(row.get("locked",0))
        if not math.isfinite(locked) or locked<0 or not math.isfinite(total) or total<0:
            raise APIError('invalid account balance values')
        if locked>1e-9:
            raise APIError("unexpected locked balance; reconcile outstanding orders")
        expected = p.cash if asset=="TMN" else sum(pos.quantity for s,pos in p.positions.items()
                                                  if markets[s]["baseAsset"]==asset)
        tolerance = max(1e-8,abs(expected)*1e-6) if asset!="TMN" else max(1.,abs(expected)*1e-6)
        if not math.isfinite(total) or abs(total-expected)>tolerance:
            raise APIError(f"balance mismatch in {asset}; deposits/manual trades/fees must be reconciled")
    required = {"TMN"}|{markets[s]["baseAsset"] for s in p.positions}
    if required-set(raw):
        raise APIError("required account balance absent")


def book_fill(p,intent,book,c,now):
    if now-book["received_at"]>c.stale_seconds:
        raise ValueError("stale paper book")
    levels = book["asks"] if intent.side=="BUY" else book["bids"]
    top = levels[0][0]
    if intent.quantity*top<c.min_notional:
        return None
    limit = top*(1+c.max_book_slippage_bps/10000 if intent.side=="BUY" else 1-c.max_book_slippage_bps/10000)
    left = intent.quantity
    if intent.side=="BUY":
        left = min(left,p.cash/(limit*(1+c.fee_bps/10000)))
    qty,value = 0.,0.
    for price,available in levels:
        if price>limit if intent.side=="BUY" else price<limit:
            break
        q = min(left,available)
        qty,value,left = qty+q,value+q*price,left-q
        if left<=1e-12:
            break
    if qty<=0 or (intent.side=="BUY" and value<c.min_notional):
        return None
    return p.apply(intent,qty,value/qty,value*c.fee_bps/10000,int(now))


def constrain_entry(intent,p,books,market,c):
    if intent.side!="BUY":
        return intent
    prices = {s:(b["asks"][0][0]+b["bids"][0][0])/2 for s,b in books.items()}
    equity = p.equity(prices)
    gross = equity-p.cash
    risk = sum(max(0,prices[s]-pos.stop)*pos.quantity for s,pos in p.positions.items())
    price = float(round_precision(books[intent.symbol]["asks"][0][0]*(1+c.max_book_slippage_bps/10000),market.get('tickSize',8),up=True))
    held = p.positions.get(intent.symbol)
    same_symbol_value = held.quantity*prices[intent.symbol] if held else 0
    budget = min(max(0,equity*c.position_cap-same_symbol_value),max(0,equity*c.gross_cap-gross),
                 p.cash/(1+c.fee_bps/10000),equity*c.risk_fraction/intent.stop_fraction,
                 max(0,equity*c.portfolio_stop_risk-risk)/intent.stop_fraction)
    intent.quantity = min(intent.quantity,budget/price)
    if intent.quantity*price<max(c.min_notional,float(market["minNotional"])):
        raise ValueError("live risk budget below minimum notional")
    return intent


def run_session(api,c,dataset,output,mode="shadow",iterations=1,poll_seconds=15,on_heartbeat=None,session_limits=None,
                adaptive_risk=False):
    # An experimental policy must never implicitly enable or alter real execution.
    if not isinstance(adaptive_risk,bool) or (adaptive_risk and mode!="shadow"):
        raise ValueError("adaptive risk is available only for explicit shadow/PAPER sessions")
    adaptive = AdaptiveRisk(c.news_path) if adaptive_risk else None
    out = Path(output)
    out.mkdir(parents=True,exist_ok=True)
    lock = RunLock(out/"engine.lock")
    journal = Journal(out/"session.sqlite")
    audit = Audit(out/"events.jsonl")
    try:
        if mode=="live":
            c.live_preflight(time.time())
        identity = {"source":source_hash(),"config":asdict(c),"mode":mode}
        if adaptive:
            identity['adaptive_risk']=POLICY
        if session_limits:
            session_limits.validate(c.initial_cash)
            identity['trial_limits']=asdict(session_limits)
        if c.model_artifact:
            from .data import digest_file
            identity["model_sha256"] = digest_file(c.model_artifact)
        if journal.get("identity") not in (None,identity):
            raise ValueError("source/config/mode changed; use a new run directory and reconcile account")
        journal.put("identity",identity)
        markets = api.markets()
        if any(s not in markets for s in c.symbols):
            raise ValueError("configured market unavailable")
        if mode=='live':
            markets={**markets,**{s:validate_market(markets[s]) for s in c.symbols}}
        if mode=="live" and any(c.min_notional<float(markets[s]["minNotional"]) for s in c.symbols):
            raise ValueError("set min_notional to at least the maximum allowed-market minimum")
        saved = journal.get("engine")
        e = restore_engine(c,saved) if saved else Engine(c)
        if not saved:
            for frame in frames(load_bars(dataset),c.symbols,c.interval):
                if next(iter(frame.values())).ts+c.interval>time.time():
                    raise ValueError("warmup includes an unclosed/future candle")
                e.observe(frame)
        p = (journal.replay(c.initial_cash) if mode=="live" else
             restore_portfolio(journal.get("portfolio")) if journal.get("portfolio") else Portfolio(c.initial_cash))
        executor = LiveExecutor(api,c,journal,audit) if mode=="live" else None
        if executor:
            executor.recover(markets)
            p = journal.replay(c.initial_cash)
            if open_order_list(api.account(c,"open_orders")):
                raise ValueError("account has standing orders; prevent self-trading before starting")
            check_balances(api,c,p,markets)
        # Failed account preflight must not consume the timed trial. Start the
        # durable clock only after balances and standing orders reconcile.
        if session_limits:
            session_limits.start(journal,int(time.time()))
        audit.write("session_start",int(time.time()),identity=identity)
        last_ts = min(l.last_ts for l in e.learners.values())
        journal.put("engine",dump_engine(e))
        count,errors = 0,0
        while iterations==0 or count<iterations:
            count += 1
            now = int(time.time())
            try:
                frame = {s:l.history[-1] for s,l in e.learners.items()}
                history_problem = False
                end = now//c.interval*c.interval
                if end>last_ts+c.interval:
                    try:
                        bars = [b for s in c.symbols for b in api.history(s,last_ts+c.interval,end,c.interval)]
                        if not bars:
                            raise APIError("no new closed candle")
                        incoming = frames(bars,c.symbols,c.interval)
                        if next(iter(incoming[0].values())).ts!=last_ts+c.interval:
                            raise APIError("runtime candle gap")
                        for frame in incoming:
                            e.observe(frame)
                            last_ts = next(iter(frame.values())).ts
                        journal.put("engine",dump_engine(e))
                    except (APIError,ValueError) as exc:
                        history_problem = True
                        audit.write("history_unavailable_risk_only",now,error=str(exc))
                # Fetch public books every poll for local protective stops and settlement.
                books = {s:api.depth(s) for s in c.symbols}
                now = int(time.time())
                if any(now-b["received_at"]>c.stale_seconds for b in books.values()):
                    raise APIError("book acquisition exceeded freshness budget")
                mids = {s:(b["asks"][0][0]+b["bids"][0][0])/2 for s,b in books.items()}
                adaptive_report = (adaptive.evaluate(c,{s:l.history for s,l in e.learners.items()},books,time.time())
                                   if adaptive else None)
                blocked = e.loss_gate(p.equity(mids),now)
                settle = bool(c.challenge_end and now>=c.challenge_end-c.settlement_buffer)
                kill = (out/"KILL").exists()
                trial = None
                if session_limits:
                    liquidation_equity=p.cash+sum(pos.quantity*books[s]['bids'][0][0]*(1-c.fee_bps/10000)
                                                  for s,pos in p.positions.items())
                    trial=session_limits.check(journal,liquidation_equity,now,halted=e.halted,manual=kill)
                    kill=kill or bool(trial['reason'])
                emergency = [Intent(s,"SELL",pos.quantity,"kill_or_settle_or_stop",decision_ts=now)
                             for s,pos in sorted(p.positions.items())
                             if (kill or settle or blocked or books[s]["bids"][0][0]<=pos.stop) and
                             pos.quantity*books[s]["bids"][0][0]>=c.min_notional]
                stale_candle = history_problem or now-(last_ts+c.interval)>c.interval+c.stale_seconds
                decision_key = journal.get("last_decision")
                normal,decisions = [],[]
                if decision_key!=last_ts and not stale_candle and not kill:
                    spreads = {s:(b["asks"][0][0]-b["bids"][0][0])/mids[s]*10000 for s,b in books.items()}
                    normal,decisions = e.plan(p,frame,now,spreads)
                    # Fixed candle identity, not poll wall-clock time, prevents restart duplicates.
                    for intent in normal:
                        intent.decision_ts = last_ts+c.interval
                    audit.write("decision",now,candle=last_ts,decisions=decisions,intents=[asdict(x) for x in normal])
                    journal.put("last_decision",last_ts)
                sells = {i.symbol:i for i in normal if i.side=="SELL"}
                sells.update({i.symbol:i for i in emergency})
                # Consolidation is an exit operation with one bounded top-up, not a new strategy entry.
                intents = list(sells.values())+[i for i in normal if i.side=="BUY" and not (blocked or kill) and
                          (not settle or i.reason=="dust_consolidation_for_exit")]
                if executor and intents:
                    check_balances(api,c,p,markets)
                    if open_order_list(api.account(c,"open_orders")):
                        raise APIError("unexpected open order")
                for intent in intents:
                    if executor and getattr(c,'operator_spot',False) and intent.side=='BUY':
                        # Refresh both liquidity-role rates before a new entry. A
                        # fee lookup must not prevent SELL exits from being sent.
                        rates=fetch_fee_rates(api,c)
                        audit.write('account_fee_check',int(time.time()),rates=rates,budget_bps=c.fee_bps)
                    fresh = api.depth(intent.symbol)
                    books[intent.symbol] = fresh
                    if session_limits and intent.side=='BUY':
                        latest=p.cash+sum(pos.quantity*books[s]['bids'][0][0]*(1-c.fee_bps/10000)
                                          for s,pos in p.positions.items())
                        trial=session_limits.check(journal,latest,int(time.time()),halted=e.halted,
                                                   manual=(out/'KILL').exists())
                        if trial['reason']:
                            kill=True
                            continue
                    entry_config = c
                    if adaptive and intent.side=='BUY':
                        # Re-evaluate at entry time using the newly fetched book;
                        # missing data or a zero budget never disables SELL exits.
                        adaptive_report=adaptive.evaluate(c,{s:l.history for s,l in e.learners.items()},
                                                          books,time.time())
                        intent=reduce_entry(intent,adaptive_report['multiplier'])
                        if intent is None:
                            audit.write('adaptive_entry_skipped',int(time.time()),
                                        reason='entries_blocked',policy=adaptive_report)
                            continue
                        entry_config=reduced_config(c,adaptive_report['multiplier'])
                    try:
                        intent = constrain_entry(intent,p,books,markets[intent.symbol],entry_config)
                    except ValueError as exc:
                        if adaptive and intent.side=='BUY' and str(exc)=='live risk budget below minimum notional':
                            audit.write('adaptive_entry_skipped',int(time.time()),symbol=intent.symbol,
                                        reason='below_minimum_after_risk_reduction',policy=adaptive_report)
                            continue
                        raise
                    if executor:
                        # Size from held inventory: quote-denominated SELL fees
                        # must not leave a spurious base-asset reserve behind.
                        # The executor separately bounds gross execution by the
                        # submitted payload and net debit by held inventory.
                        held=p.positions.get(intent.symbol)
                        inventory_limit=held.quantity if held and intent.side=='SELL' else None
                        result = executor.execute(intent,markets[intent.symbol],fresh,
                                                  inventory_limit=inventory_limit)
                        p = journal.replay(c.initial_cash)
                        audit.write("live_result",now,result=result)
                    else:
                        fill = book_fill(p,intent,fresh,c,time.time())
                        # Persist paper fill before reporting; no exchange writes in this branch.
                        journal.put("portfolio",dump_portfolio(p))
                        audit.write("paper_fill",now,fill=fill,book=fresh)
                journal.put("engine",dump_engine(e))
                journal.put("portfolio",dump_portfolio(p))
                summary = {"mode":mode,"polls":count,"cash_tmn":p.cash,"equity_tmn":p.equity(mids),
                           "realized_pnl_tmn":p.realized,"fees_tmn":p.fees,"fills":len(p.fills),
                           "settled_to_tmn":not p.positions,"stale_candle":stale_candle,
                           "last_candle":last_ts,"halted":e.halted,"daily_halted":e.daily_halted,
                           "positions":{s:asdict(pos) for s,pos in p.positions.items()}}
                if adaptive:
                    summary['adaptive_risk']=adaptive_report
                if trial:
                    liquidation_equity=p.cash+sum(pos.quantity*books[s]['bids'][0][0]*(1-c.fee_bps/10000)
                                                  for s,pos in p.positions.items())
                    trial=session_limits.check(journal,liquidation_equity,int(time.time()),halted=e.halted,
                                               manual=(out/'KILL').exists())
                    kill=kill or bool(trial['reason'])
                    summary['trial']=trial
                    summary['liquidation_equity_tmn']=liquidation_equity
                    summary['stopped']=bool(trial['reason']) and not p.positions
                    summary['manual_reconciliation_required']=bool(trial['reason']) and bool(p.positions) and not intents
                journal.put("summary",summary)
                audit.write("heartbeat",now,**summary)
                print(json.dumps(summary),flush=True)
                if on_heartbeat:
                    on_heartbeat(summary)
                errors = 0
                if (kill or settle) and not p.positions:
                    break
                if trial and summary['manual_reconciliation_required']:
                    break  # Preserve residual inventory; never fabricate a below-minimum exit.
            except (APIError,ValueError) as exc:
                errors += 1
                audit.write("data_or_risk_error",int(time.time()),error=str(exc),consecutive=errors)
                if errors>=3:
                    raise RuntimeError("three consecutive errors; no further submissions") from exc
                if iterations and count>=iterations:
                    raise
            if iterations==0 or count<iterations:
                time.sleep(min(60,poll_seconds*max(1,errors)))
        return journal.get("summary")
    finally:
        journal.close()
        lock.close()
