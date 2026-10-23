from dataclasses import asdict, replace
from pathlib import Path
import json
import math
import random
import statistics
import time
from .audit import Audit, atomic_json, manifest
from .data import digest_file, frames, load_bars
from .engine import Engine, Intent, Portfolio, simulate_fill


def bootstrap_days(points, repetitions=1000, seed=73):
    # Daily blocks preserve intraday dependence. Small samples remain weak evidence.
    byday = {}
    for p in points:
        byday.setdefault((p["ts"]+12600)//86400, []).append(p["equity"])
    prev = points[0]["equity"] if points else 1
    daily = []
    for vals in byday.values():
        daily.append(vals[-1]/prev-1)
        prev = vals[-1]
    if len(daily) < 5:
        return {"days": len(daily), "return_interval_95": None, "warning": "too few daily blocks"}
    rng = random.Random(seed)
    trials = sorted(math.prod(1+rng.choice(daily) for _ in daily)-1 for _ in range(repetitions))
    return {"days": len(daily), "return_interval_95": [trials[25], trials[974]],
            "warning": "descriptive block bootstrap; not proof of future alpha"}


def metrics(portfolio, curve, config, latency):
    peak, mdd = config.initial_cash, 0
    for p in curve:
        peak = max(peak, p["equity"])
        mdd = max(mdd, 1-p["equity"]/peak)
    wins = sum(x for x in portfolio.round_trips if x > 0)
    losses = -sum(x for x in portfolio.round_trips if x < 0)
    sorted_lat = sorted(latency)
    settled = not portfolio.positions
    return {"initial_cash_tmn": config.initial_cash, "final_cash_tmn": portfolio.cash,
            "final_equity_tmn": curve[-1]["equity"] if curve else portfolio.cash,
            "net_return_pct": (portfolio.cash/config.initial_cash-1)*100 if settled else None,
            "realized_pnl_tmn": portfolio.realized, "settled_to_tmn": settled,
            "max_drawdown_pct": mdd*100, "fills": len(portfolio.fills),
            "sell_fills": len(portfolio.round_trips), "turnover_tmn": portfolio.turnover,
            "average_fill_tmn": portfolio.turnover/max(1, len(portfolio.fills)),
            "fees_tmn": portfolio.fees, "win_rate": sum(x > 0 for x in portfolio.round_trips)/max(1,len(portfolio.round_trips)),
            "profit_factor": wins/losses if losses else None,
            "decision_latency_p95_ms": sorted_lat[min(len(sorted_lat)-1,int(len(sorted_lat)*.95))] if sorted_lat else 0,
            "residual_positions": {s: asdict(p) for s,p in portfolio.positions.items()},
            "uncertainty": bootstrap_days(curve)}


def run(bars, config, output=None, warmup=600, audit=True, ablation=False, baseline=None):
    config.validate()
    fs = frames(bars, config.symbols, config.interval)
    if len(fs) <= warmup+config.horizon+2:
        raise ValueError("not enough data for training and out-of-sample evaluation")
    e, p = Engine(config), Portfolio(config.initial_cash)
    pending, curve, latencies, decisions_all = [], [], [], []
    log = None
    if output:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        if (output/"decisions.jsonl").exists():
            raise ValueError("backtest output already exists; use a new directory")
        log = Audit(output/"decisions.jsonl") if audit else None
    for i, frame in enumerate(fs):
        ts = next(iter(frame.values())).ts
        # Fill orders decided on the previous closed candle; never same-candle close fills.
        for order in pending:
            fill = simulate_fill(p, order, frame[order.symbol], config)
            if log:
                log.write("fill" if fill else "unfilled", ts, fill=fill, intent=asdict(order))
        pending = []
        if i >= warmup:
            # Stops have priority. No high/low path assumed for profit targets (none used).
            for s, position in list(p.positions.items()):
                if frame[s].low <= position.stop:
                    intent = Intent(s,"SELL",position.quantity,"intrabar_stop",decision_ts=ts)
                    fill = simulate_fill(p,intent,frame[s],config,min(frame[s].open,position.stop))
                    if log:
                        log.write("fill",ts,fill=fill)
        started = time.perf_counter()
        forecasts = e.observe(frame)
        if ablation:
            # Full ablation: learned forecaster cannot approve any entry.
            for f in forecasts.values():
                if f:
                    f.mu, f.lower = 0.0, 0.0
        if baseline:
            for s,f in forecasts.items():
                if f:
                    history = e.learners[s].history
                    momentum = math.log(history[-1].close/history[-13].close)
                    f.mu = momentum if baseline=="momentum" else -momentum
                    f.lower = f.mu
        if i >= warmup:
            pending, decisions = e.plan(p, frame)
            # Predeclared end: stop entries in the final horizon and flatten over the buffer.
            remaining = len(fs)-1-i
            if remaining <= config.horizon:
                pending = [x for x in pending if x.side != "BUY" or x.reason=="dust_consolidation_for_exit"]
            if remaining <= max(1, config.settlement_buffer//config.interval):
                cleanups = [x for x in pending if x.reason=="dust_consolidation_for_exit"] if remaining>=2 else []
                pending = cleanups+[Intent(s,"SELL",pos.quantity,"evaluation_settlement",decision_ts=ts+config.interval)
                           for s,pos in sorted(p.positions.items()) if pos.quantity*frame[s].close>=config.min_notional]
            elapsed = (time.perf_counter()-started)*1000
            latencies.append(elapsed)
            prices = {s:b.close for s,b in frame.items()}
            equity = p.equity(prices)
            e.loss_gate(equity,ts+config.interval)
            point = {"ts":ts+config.interval,"equity":equity,"cash":p.cash}
            curve.append(point)
            record = {"ts":ts+config.interval,"decisions":decisions,"intents":[asdict(x) for x in pending]}
            decisions_all.append(record)
            if log:
                log.write("decision",ts+config.interval,**{k:v for k,v in record.items() if k!="ts"},
                          equity=equity, halted=e.halted, daily_halted=e.daily_halted)
    # No fictitious unlimited closing liquidity. Failed exits leave a disqualifying residual.
    report = metrics(p,curve,config,latencies)
    report.update({"mode":"historical_simulation", "warmup_bars":warmup,
                   "evaluation_bars":len(curve),"start":curve[0]["ts"],"end":curve[-1]["ts"],
                   "caveat":"OHLCV simulated fills; historical spread/depth are assumptions"})
    # Buy-and-hold uses the same first OOS open, fees and fill drag, with equal weights.
    drag = (config.spread_bps/2+config.slippage_bps)/10000
    fee = config.fee_bps/10000
    bh = sum(fs[-1][s].close*(1-drag)*(1-fee)/(fs[warmup][s].open*(1+drag)*(1+fee))
             for s in config.symbols)/len(config.symbols)-1
    report["buy_hold_return_pct"] = bh*100
    report["cash_return_pct"] = 0.0
    if output:
        atomic_json(output/"report.json",report)
        atomic_json(output/"equity.json",curve)
        atomic_json(output/"trades.json",p.fills)
    return report, curve, decisions_all


def evaluate_file(path, config, output, warmup=600):
    bars = load_bars(path)
    report,_,_ = run(bars,config,output,warmup)
    atomic_json(Path(output)/"manifest.json",manifest(config,digest_file(path),"backtest"))
    return report


def suite(path, config, output, warmup=600, fold_bars=864):
    bars = load_bars(path)
    fs = frames(bars,config.symbols,config.interval)
    root = Path(output)
    results = []
    # Disjoint OOS folds. Each training prefix uses only history preceding that fold.
    for n,start in enumerate(range(warmup,len(fs)-config.horizon-2,fold_bars)):
        end = min(len(fs),start+fold_bars)
        if end-start <= config.horizon+2:
            continue
        sample = [b for f in fs[:end] for b in f.values()]
        report,_,_ = run(sample,config,root/f"fold-{n:02d}",start)
        results.append(report)
    stress = {}
    for name,multiple in (("base",1),("cost_x2",2),("cost_x3",3)):
        c = replace(config,fee_bps=config.fee_bps*multiple,slippage_bps=config.slippage_bps*multiple,
                    spread_bps=config.spread_bps*multiple)
        stress[name] = run(bars,c,None,warmup,audit=False)[0]
    stress["no_ml"] = run(bars,config,None,warmup,audit=False,ablation=True)[0]
    stress["rule_momentum"] = run(bars,config,None,warmup,audit=False,baseline="momentum")[0]
    stress["rule_reversion"] = run(bars,config,None,warmup,audit=False,baseline="reversion")[0]
    result = {"walk_forward":results,"stress_and_ablation":stress,
              "selection_policy":"fixed configuration; no OOS parameter search",
              "note":"fold cash resets; do not sum fold returns or claim a live track record"}
    root.mkdir(parents=True,exist_ok=True)
    atomic_json(root/"suite.json",result)
    atomic_json(root/"manifest.json",manifest(config,digest_file(path),"walk_forward_suite"))
    return result
