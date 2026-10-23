import argparse
from dataclasses import asdict
from pathlib import Path
import json
import sys
import time
from .audit import Audit, atomic_json
from .config import Config
from .data import Bar, metadata, save_bars, synthetic
from .mcp_bridge import MCPClient, serve
from .research import evaluate_file, suite
from .runtime import run_session
from .wallex import Wallex, PublicWallex


class MCPData:
    """Read-only market-data routing through Wallex MCP tool calls."""
    def __init__(self,client,mapping=None):
        self.client = client
        self.mapping = mapping or {}

    def invoke(self,logical,args=None):
        spec = self.mapping.get(logical,{"name":"wallex_"+logical})
        mapped = {spec.get("arguments",{}).get(k,k):v for k,v in (args or {}).items()}
        result = self.client.tool(spec["name"],mapped)
        for key in spec.get("result_path",[]):
            result = result[key]
        return result

    def markets(self):
        return self.invoke("markets")

    def depth(self,symbol):
        return self.invoke("depth",{"symbol":symbol})

    def history(self,symbol,start,end,interval):
        return [Bar(**b).validate() for b in self.invoke("candles",
                {"symbol":symbol,"start":start,"end":end,"interval":interval})["bars"]]


def main():
    parser = argparse.ArgumentParser(description="Wallex causal AI engine (TMN; default research/paper)")
    parser.add_argument("--config",default=None)
    sub = parser.add_subparsers(dest="command",required=True)
    p = sub.add_parser("demo",help="synthetic fixture; no profitability claim")
    p.add_argument("--out",default="runs/demo")
    p.add_argument("--bars",type=int,default=2400)
    p = sub.add_parser("fetch")
    p.add_argument("--out",required=True)
    p.add_argument("--days",type=int,default=14)
    p.add_argument("--end",type=int,default=None,help="exclusive UTC Unix second; defaults to last closed bar")
    for command in ("backtest","suite"):
        p = sub.add_parser(command)
        p.add_argument("--data",required=True)
        p.add_argument("--out",required=True)
        p.add_argument("--warmup",type=int,default=600)
        if command=="suite":
            p.add_argument("--fold-bars",type=int,default=864)
    for command in ("shadow","live"):
        p = sub.add_parser(command)
        p.add_argument("--data",required=True,help="warmup candles, not simulated trades")
        p.add_argument("--out",required=True)
        p.add_argument("--iterations",type=int,default=1,help="0 runs continuously")
        p.add_argument("--poll-seconds",type=int,default=15)
        if command=="shadow":
            p.add_argument("--mcp",action="store_true",help="route public data through local MCP bridge")
            p.add_argument("--mcp-settings",help="external stdio Wallex MCP command and tool-name mappings")
    p = sub.add_parser("mcp-server")
    p.add_argument("--data",default=None)
    p = sub.add_parser("mcp-check")
    p.add_argument("--data",required=True)
    p = sub.add_parser("verify-audit")
    p.add_argument("path")
    args = parser.parse_args()
    c = Config.load(args.config)
    if args.command=="demo":
        out = Path(args.out)
        out.mkdir(parents=True,exist_ok=True)
        path = out/"synthetic.csv"
        save_bars(path,synthetic(c.symbols,args.bars,c.interval))
        metadata(path,"SYNTHETIC_ENGINEERING_FIXTURE",{"seed":73,"not_real_market":True})
        r = evaluate_file(path,c,out/"backtest",min(600,args.bars//2))
        r["source_warning"] = "SYNTHETIC DATA; this is not evidence of investment performance"
        atomic_json(out/"backtest"/"report.json",r)
        print(json.dumps(r,indent=2))
    elif args.command=="fetch":
        if not 1<=args.days<=90:
            raise ValueError("days must be 1..90")
        end = args.end if args.end is not None else int(time.time())//c.interval*c.interval
        if end>time.time() or end%c.interval:
            raise ValueError("end must be a past aligned candle boundary")
        api = PublicWallex()
        bars = [b for s in c.symbols for b in api.history(s,end-args.days*86400,end,c.interval)]
        if not bars:
            raise ValueError("no data received; no empty dataset saved")
        save_bars(args.out,bars)
        metadata(args.out,"Wallex public OHLCV",{"fetched_at":int(time.time()),"interval":c.interval,
                 "symbols":c.symbols,"start":end-args.days*86400,"end":end})
        print(json.dumps({"rows":len(bars),"path":args.out}))
    elif args.command=="backtest":
        print(json.dumps(evaluate_file(args.data,c,args.out,args.warmup),indent=2))
    elif args.command=="suite":
        result = suite(args.data,c,args.out,args.warmup,args.fold_bars)
        print(json.dumps({"folds":len(result["walk_forward"]),"report":str(Path(args.out)/"suite.json")}))
    elif args.command in ("shadow","live"):
        if args.iterations<0 or not 1<=args.poll_seconds<=60:
            raise ValueError("invalid runtime bounds")
        client = None
        try:
            if getattr(args,"mcp_settings",None):
                settings = json.loads(Path(args.mcp_settings).read_text(encoding="utf-8"))
                client = MCPClient(settings["command"])
                api = MCPData(client,settings.get("tools"))
            elif getattr(args,"mcp",False):
                cmd = [sys.executable,"-X","utf8","-m","wallex_engine"]
                if args.config:
                    cmd += ["--config",args.config]
                client = MCPClient(cmd+["mcp-server","--data",args.data])
                api = MCPData(client)
            else:
                api = PublicWallex() if args.command=="shadow" else Wallex()
            run_session(api,c,args.data,args.out,args.command,args.iterations,args.poll_seconds)
        finally:
            if client:
                client.close()
    elif args.command=="mcp-server":
        serve(c,args.data)
    elif args.command=="mcp-check":
        cmd = [sys.executable,"-X","utf8","-m","wallex_engine"]
        if args.config:
            cmd += ["--config",args.config]
        client = MCPClient(cmd+["mcp-server","--data",args.data])
        try:
            print(json.dumps(client.call("tools/list"),indent=2))
            print(json.dumps(client.tool("engine_decision"),indent=2))
        finally:
            client.close()
    elif args.command=="verify-audit":
        n,h = Audit.verify(args.path)
        print(json.dumps({"events":n,"last_hash":h,"valid":True}))


if __name__=="__main__":
    try:
        main()
    except (ValueError,RuntimeError,OSError) as exc:
        print(f"ERROR: {exc}",file=sys.stderr)
        sys.exit(1)
