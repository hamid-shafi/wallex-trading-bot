"""Small stdio MCP adapter, protocol 2024-11-05 (line-delimited JSON-RPC).

Project-owned bridge to Wallex REST, NOT an official Wallex-hosted MCP service.
AI hosts can observe Wallex and run the learned strategy. No arbitrary order tool.
"""
from dataclasses import asdict
from pathlib import Path
import json
import os
import queue
import subprocess
import sys
import threading
import time
from .config import Config
from .data import frames, load_bars
from .engine import Engine, Portfolio
from .wallex import PublicWallex


def schema(properties=None,required=None):
    return {"type":"object","properties":properties or {},"required":required or [],"additionalProperties":False}


TOOLS = [
    {"name":"news_features","description":"Read timestamp-safe political, conflict, sanctions and monetary news features; no trading or account access.","inputSchema":schema()},
    {"name":"wallex_markets","description":"Read public Wallex market metadata.","inputSchema":schema()},
    {"name":"wallex_depth","description":"Read a current Wallex order book.","inputSchema":schema({"symbol":{"type":"string"}},["symbol"])},
    {"name":"wallex_candles","description":"Read closed OHLCV; interval is seconds.","inputSchema":schema(
        {"symbol":{"type":"string"},"start":{"type":"integer"},"end":{"type":"integer"},
         "interval":{"type":"integer","enum":[60,300,900,3600]}},["symbol","start","end","interval"])},
    {"name":"engine_decision","description":"Train the real causal AI on the configured local dataset and return a paper decision and explanation.",
     "inputSchema":schema()},
]


def dispatch(name,args,api,config,dataset=None):
    if name=="news_features":
        from .news import NewsIndex
        if not config.news_path:
            raise ValueError("news_path not configured")
        index=NewsIndex(config.news_path)
        values,context=index.at(int(time.time()))
        return {"features":values,"context":context,"items_stored":len(index.rows),
                "warning":"RSS topics are lexical features, not verified event facts; historical coverage may be missing"}
    if name=="wallex_markets":
        return api.markets()
    if name=="wallex_depth":
        if args["symbol"] not in config.symbols:
            raise ValueError("symbol outside configured universe")
        return api.depth(args["symbol"])
    if name=="wallex_candles":
        if args["symbol"] not in config.symbols or args["interval"] not in (60,300,900,3600):
            raise ValueError("invalid symbol/interval")
        if not 0<args["end"]-args["start"]<=90*86400:
            raise ValueError("history range must be <=90 days")
        return {"bars":[asdict(b) for b in api.history(args["symbol"],args["start"],args["end"],args["interval"])]}
    if name=="engine_decision":
        if not dataset:
            raise ValueError("server dataset not configured")
        fs = frames(load_bars(dataset),config.symbols,config.interval)
        e,p = Engine(config),Portfolio(config.initial_cash)
        for frame in fs:
            e.observe(frame)
        intents,decisions = e.plan(p,fs[-1])
        return {"mode":"paper_proposal","cash_assumed":config.initial_cash,
                "intents":[asdict(x) for x in intents],"decisions":decisions}
    raise ValueError("unknown tool")


def serve(config,dataset=None):
    api = PublicWallex()
    initialized = False
    for line in sys.stdin:
        msg = {}
        try:
            msg = json.loads(line)
            method,params = msg["method"],msg.get("params",{})
            if "id" not in msg:
                continue
            if method=="initialize":
                initialized = True
                result = {"protocolVersion":"2024-11-05","capabilities":{"tools":{}},
                          "serverInfo":{"name":"wallex-engine-bridge","version":"0.1.0"}}
            elif method=="ping":
                result = {}
            elif not initialized:
                raise ValueError("initialize first")
            elif method=="tools/list":
                result = {"tools":TOOLS}
            elif method=="tools/call":
                try:
                    args = params.get("arguments",{})
                    tool = next(t for t in TOOLS if t["name"]==params["name"])
                    fields = tool["inputSchema"]["properties"]
                    if set(args)-set(fields) or set(tool["inputSchema"]["required"])-set(args):
                        raise ValueError("invalid tool arguments")
                    data = dispatch(params["name"],args,api,config,dataset)
                    result = {"content":[{"type":"text","text":json.dumps(data,ensure_ascii=True)}]}
                except Exception as exc:
                    result = {"isError":True,"content":[{"type":"text","text":str(exc)}]}
            else:
                raise ValueError("unsupported method")
            response = {"jsonrpc":"2.0","id":msg["id"],"result":result}
        except Exception as exc:
            response = {"jsonrpc":"2.0","id":msg.get("id"),"error":{"code":-32602,"message":str(exc)}}
        print(json.dumps(response,ensure_ascii=True),flush=True)


class MCPClient:
    """stdio client for the bridge or organizer server using the same tool contract."""
    def __init__(self,command,timeout=30):
        flags = subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0
        self.process = subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                        stderr=sys.stderr,text=True,encoding="utf-8",creationflags=flags)
        self.responses = queue.Queue()
        self.timeout,self.counter = timeout,0
        threading.Thread(target=self._read,daemon=True).start()
        response = self.call("initialize",{"protocolVersion":"2024-11-05","capabilities":{},
                            "clientInfo":{"name":"wallex-engine","version":"0.1.0"}})
        if response.get("protocolVersion") != "2024-11-05":
            self.close()
            raise ValueError("MCP protocol version not supported by this minimal client")
        self.send({"jsonrpc":"2.0","method":"notifications/initialized"})

    def _read(self):
        for line in self.process.stdout:
            self.responses.put(line)
        self.responses.put(None)

    def send(self,message):
        self.process.stdin.write(json.dumps(message)+"\n")
        self.process.stdin.flush()

    def call(self,method,params=None):
        self.counter += 1
        self.send({"jsonrpc":"2.0","id":self.counter,"method":method,"params":params or {}})
        while True:
            line = self.responses.get(timeout=self.timeout)
            if line is None:
                raise RuntimeError("MCP server exited")
            response = json.loads(line)
            if response.get("id") != self.counter:
                continue
            if "error" in response:
                raise RuntimeError(str(response["error"]))
            return response["result"]

    def tool(self,name,args=None):
        result = self.call("tools/call",{"name":name,"arguments":args or {}})
        if result.get("isError"):
            raise RuntimeError(str(result["content"]))
        return json.loads(next(c["text"] for c in result["content"] if c["type"]=="text"))

    def close(self):
        if self.process.stdin:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=2)
        if self.process.stdout:
            self.process.stdout.close()
