"""Bounded prospective shadow evaluation with auditable public-only network calls."""
import argparse
from dataclasses import asdict
from pathlib import Path
import json
import math
import time

from .audit import Audit, atomic_json, canonical, source_hash
from .config import Config
from .data import digest_file
from .runtime import run_session
from .wallex import PublicWallex


class ObservedPublic(PublicWallex):
    def __init__(self, output):
        super().__init__()
        self.audit=Audit(Path(output)/"market-events.jsonl")
        self.requests=0

    def request(self,method,path,params=None,body=None,private=False):
        # Parent rejects writes/private routes before any request can be sent.
        started=time.time()
        data=super().request(method,path,params,body,private)
        self.requests+=1
        self.audit.write("public_response",int(time.time()),method=method,path=path,params=params,
                         duration_ms=(time.time()-started)*1000,response=data)
        return data


def shadow(config,data,output,minutes=10,poll_seconds=30):
    if not 1<=minutes<=1440 or not 1<=poll_seconds<=60:
        raise ValueError("invalid bounded shadow duration")
    c=Config.load(config)
    if c.live_enabled or c.routes:
        raise ValueError("shadow evaluator requires disabled live mode and no account routes")
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    if (out/"shadow-result.json").exists():
        raise ValueError("completed shadow evidence must not be overwritten")
    api=ObservedPublic(out)
    start=int(time.time())
    atomic_json(out/"shadow-protocol.json",{"started_at":start,"planned_minutes":minutes,
                "config":asdict(c),"source_sha256":source_hash(),"warmup_sha256":digest_file(data),
                "model_sha256":digest_file(c.model_artifact) if c.model_artifact else None,
                "network_policy":"only GET /v1/markets, /v1/depth, /v1/udf/history",
                "credentials_used":False,"account_reads":0,"exchange_writes":0})
    summary=run_session(api,c,data,out,"shadow",math.ceil(minutes*60/poll_seconds)+1,poll_seconds)
    end=int(time.time())
    points=[]; counts={}; first_candle=None; last_candle=None
    for line in (out/"events.jsonl").read_text(encoding="utf-8").splitlines():
        obj=json.loads(line); counts[obj["kind"]]=counts.get(obj["kind"],0)+1
        if obj["kind"]=="heartbeat":
            p=obj["payload"]; points.append(p["equity_tmn"])
            first_candle=p["last_candle"] if first_candle is None else first_candle
            last_candle=p["last_candle"]
    peak=c.initial_cash; drawdown=0.
    for value in points:
        peak=max(peak,value); drawdown=max(drawdown,1-value/peak)
    result={"started_at":start,"ended_at":end,"elapsed_seconds":end-start,
            "summary":summary,"event_counts":counts,"public_requests":api.requests,
            "first_candle":first_candle,"last_candle":last_candle,
            "max_observed_drawdown_pct":drawdown*100,
            "net_cash_return_pct":(summary["cash_tmn"]/c.initial_cash-1)*100 if summary["settled_to_tmn"] else None,
            "mark_to_market_return_pct":(summary["equity_tmn"]/c.initial_cash-1)*100,
            "credentials_used":False,"exchange_writes":0,"account_reads":0,
            "execution_audit_valid":bool(Audit.verify(out/"events.jsonl")),
            "market_audit_valid":bool(Audit.verify(out/"market-events.jsonl")),
            "limitation":"short prospective shadow; displayed depth excludes latency/adverse selection and is not real execution",
            "profitability_proven":False}
    atomic_json(out/"shadow-result.json",result)
    print(json.dumps(result),flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True); p.add_argument("--data",required=True)
    p.add_argument("--out",required=True); p.add_argument("--minutes",type=int,default=10)
    p.add_argument("--poll-seconds",type=int,default=30)
    args=p.parse_args()
    shadow(args.config,args.data,args.out,args.minutes,args.poll_seconds)
