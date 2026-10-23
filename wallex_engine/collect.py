"""Resumable public OHLCV collection; never sends API keys or exchange writes."""
import argparse
from collections import Counter
from pathlib import Path
import json
import time

from .audit import atomic_json
from .data import Bar, aggregate, digest_file, save_bars
from .wallex import PublicWallex


def collect(root, symbols, days=180, end=None, max_repair_hours=48):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root/"collection_plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan["symbols"] != symbols or plan["days"] != days:
            raise ValueError("existing collection has different arguments")
    else:
        end = end or int(time.time())//3600*3600
        plan = {"source":"https://api.wallex.ir/v1/udf/history", "symbols":symbols,
                "start":end-days*86400, "end":end, "days":days,
                "interval":3600, "resolution":"60", "created_at":int(time.time()),
                "selection":"fixed liquid direct TMN examples; current-survivor universe bias",
                "private_requests":0, "exchange_writes":0, "credentials_used":False}
        atomic_json(plan_path,plan)
    api = PublicWallex(timeout=25, retries=3)
    if not (root/"markets.json").exists():
        atomic_json(root/"markets.json",api.markets())
    rows, quality = [], {}
    for symbol in symbols:
        bars = {}
        invalid = []
        for left in range(plan["start"],plan["end"],20*86400):
            right = min(plan["end"],left+20*86400)
            path = root/"raw"/f"{symbol}-{left}-{right}.json"
            if path.exists():
                raw = json.loads(path.read_text(encoding="utf-8"))
            else:
                payload = api.request("GET","/v1/udf/history",{
                    "symbol":symbol,"resolution":"60","from":left,"to":right})
                raw = {"fetched_at":int(time.time()),"from":left,"to":right,
                       "symbol":symbol,"payload":payload}
                atomic_json(path,raw)
            data = raw["payload"]
            if data.get("s")=="no_data":
                continue
            if data.get("s")!="ok":
                raise ValueError(f"unexpected history status for {symbol}")
            arrays = [data[k] for k in ("t","o","h","l","c","v")]
            if len({len(a) for a in arrays})!=1:
                raise ValueError("inconsistent candle arrays")
            for ts,o,h,l,c,v in zip(*arrays):
                ts = int(ts)
                if ts<plan["start"] or ts+3600>plan["end"]:
                    continue
                try:
                    b = Bar(ts,symbol,*map(float,(o,h,l,c,v))).validate()
                except ValueError as exc:
                    invalid.append({"ts":ts,"reason":str(exc)})
                    continue
                if ts%3600:
                    raise ValueError("timestamp alignment mismatch")
                if ts in bars and bars[ts]!=b:
                    raise ValueError("conflicting closed candle duplicate")
                bars[ts]=b
            print(json.dumps({"symbol":symbol,"through":right,"bars":len(bars)}),flush=True)
        expected = set(range(plan["start"],plan["end"],3600))
        # An absent/bad hourly candle may only be repaired from all 60 real 1m bars.
        repair_candidates=sorted(expected-set(bars))
        for ts in repair_candidates[:max_repair_hours]:
            repair_path = root/"raw"/f"repair-{symbol}-{ts}.json"
            if repair_path.exists():
                repair = json.loads(repair_path.read_text(encoding="utf-8"))
            else:
                repair = api.request("GET","/v1/udf/history",{
                    "symbol":symbol,"resolution":"1","from":ts,"to":ts+3600})
                atomic_json(repair_path,repair)
            if repair.get("s")=="ok":
                minutes = []
                try:
                    for t,o,h,l,c,v in zip(*(repair[k] for k in ("t","o","h","l","c","v"))):
                        if ts<=int(t)<ts+3600:
                            minutes.append(Bar(int(t),symbol,*map(float,(o,h,l,c,v))).validate())
                    rebuilt = aggregate(minutes,3600,60)
                    if len(rebuilt)==1:
                        bars[ts]=rebuilt[0]
                except ValueError:
                    pass
        missing = sorted(expected-set(bars))
        quality[symbol] = {"bars":len(bars),"missing_count":len(missing),
                           "missing_timestamps":missing,"invalid_hourly_rejected":invalid,
                           "repair_candidates":len(repair_candidates),"repair_budget_hours":max_repair_hours,
                           "zero_volume":sum(b.volume==0 for b in bars.values())}
        rows.extend(bars.values())
    save_bars(root/"hourly.csv",rows)
    report = {**plan,"rows":len(rows),"data_sha256":digest_file(root/"hourly.csv"),
              "quality":quality,"completed_at":int(time.time()),
              "missing_data_policy":"never fabricate or forward-fill candles"}
    atomic_json(root/"data_quality.json",report)
    print(json.dumps({"rows":len(rows),"quality":{s:{k:v for k,v in q.items() if k!="missing_timestamps"}
                                                   for s,q in quality.items()}}),flush=True)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True)
    p.add_argument("--days",type=int,default=180)
    p.add_argument("--end",type=int)
    p.add_argument("--symbols",nargs="+",default=["BTCTMN","ETHTMN","SOLTMN","XRPTMN","BNBTMN","USDTTMN"])
    args=p.parse_args()
    collect(args.out,args.symbols,args.days,args.end)
