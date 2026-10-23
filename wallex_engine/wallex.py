"""Wallex REST adapter. GET retries only; writes never retried blindly."""
from dataclasses import asdict
from decimal import Decimal, ROUND_DOWN, ROUND_UP
import json
import math
import os
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, quote
from urllib.request import Request, build_opener, HTTPRedirectHandler
from .data import Bar, aggregate


class APIError(RuntimeError):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise APIError("redirect refused; credentials stay on api.wallex.ir")


class Wallex:
    def __init__(self, key=None, timeout=15, retries=3):
        self.key = key if key is not None else os.getenv("WALLEX_API_KEY", "")
        self.timeout, self.retries, self.last = timeout, retries, 0.0
        self.opener = build_opener(NoRedirect())

    def request(self, method, path, params=None, body=None, private=False):
        if not path.startswith("/") or "://" in path or ".." in path or path.startswith("//"):
            raise ValueError("invalid API path")
        if private and not self.key:
            raise APIError("WALLEX_API_KEY is missing")
        url = "https://api.wallex.ir"+path
        if params:
            url += "?"+urlencode(params)
        headers = {"Accept":"application/json","Content-Type":"application/json",
                   "User-Agent":"wallex-hackathon-engine/0.1"}
        if private:
            headers["x-api-key"] = self.key
        raw = json.dumps(body).encode() if body is not None else None
        attempts = self.retries if method == "GET" else 1
        for attempt in range(attempts):
            delay = .25-(time.monotonic()-self.last)
            if delay > 0:
                time.sleep(delay)
            self.last = time.monotonic()
            try:
                with self.opener.open(Request(url,raw,headers,method=method),timeout=self.timeout) as r:
                    payload = json.load(r)
                if isinstance(payload, dict) and payload.get("success") is False:
                    raise APIError("Wallex returned success=false")
                return payload
            except HTTPError as exc:
                retryable = exc.code in (429,500,502,503,504)
                if method != "GET" or not retryable or attempt == attempts-1:
                    raise APIError(f"Wallex {method} HTTP {exc.code}") from exc
                pause = min(10,max(1,float(exc.headers.get("Retry-After",2**attempt))))
                time.sleep(pause)
            except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                if attempt == attempts-1:
                    raise APIError(f"Wallex {method} transport/JSON failure ({type(exc).__name__})") from exc
                time.sleep(2**attempt)

    def markets(self):
        d = self.request("GET","/v1/markets")
        result = d.get("result",{})
        markets = result.get("symbols",result)
        if not isinstance(markets,dict) or not markets:
            raise APIError("unexpected market schema")
        return markets

    def depth(self,symbol):
        started = time.time()
        d = self.request("GET","/v1/depth",{"symbol":symbol})["result"]
        asks = sorted([(float(x["price"]),float(x["quantity"])) for x in d["ask"]])
        bids = sorted([(float(x["price"]),float(x["quantity"])) for x in d["bid"]],reverse=True)
        if not asks or not bids or bids[0][0] >= asks[0][0]:
            raise APIError("empty or crossed order book")
        if any(not math.isfinite(p) or not math.isfinite(q) or p<=0 or q<=0 for p,q in asks+bids):
            raise APIError("invalid order book")
        # Request start, rather than completion, bounds local snapshot age conservatively.
        return {"symbol":symbol,"asks":asks,"bids":bids,"received_at":started,
                "exchange_timestamp_verified":False}

    def history(self,symbol,start,end,interval=300):
        resolution,source_interval = ("60",3600) if interval==3600 else ("1",60)
        # Chunk 1m at 3 days and 1h at 20 days, de-duplicate inclusive boundaries.
        window = (3 if source_interval==60 else 20)*86400
        bars = {}
        for left in range(start,end,window):
            right = min(end,left+window)
            d = self.request("GET","/v1/udf/history",{"symbol":symbol,"resolution":resolution,
                             "from":left,"to":right})
            if d.get("s") == "no_data":
                continue
            if d.get("s") != "ok":
                raise APIError("unexpected history status")
            arrays = [d[k] for k in ("t","o","h","l","c","v")]
            if len({len(a) for a in arrays}) != 1:
                raise APIError("history arrays have inconsistent lengths")
            for t,o,h,l,c,v in zip(*arrays):
                t = int(t)
                if t < start or t+source_interval > end:
                    continue  # excludes incomplete candle
                bar = Bar(t,symbol,*map(float,(o,h,l,c,v))).validate()
                if t in bars and bars[t] != bar:
                    raise APIError("conflicting historical duplicate")
                bars[t] = bar
        values = sorted(bars.values(),key=lambda b:b.ts)
        if any(b.ts % source_interval for b in values):
            raise APIError("history timestamps misaligned")
        if any(b.ts-a.ts != source_interval for a,b in zip(values,values[1:])):
            raise APIError("historical gaps or server resolution mismatch")
        return aggregate(values,interval,source_interval) if interval!=source_interval else values

    def account(self,config,name,method="GET",client_id=None,body=None):
        path = config.routes.get(name)
        if not path:
            raise APIError(f"account route not configured: {name}")
        if client_id is not None:
            path = path.replace("{client_id}",quote(client_id,safe=""))
        return self.request(method,path,body=body,private=True)["result"]


class PublicWallex(Wallex):
    """Public market data only. Reject writes/account access before network I/O."""
    PATHS = frozenset({"/v1/markets", "/v1/depth", "/v1/udf/history"})

    def __init__(self, **kwargs):
        super().__init__(key="", **kwargs)

    def request(self, method, path, params=None, body=None, private=False):
        if method != "GET" or path not in self.PATHS or private or body is not None:
            raise APIError("research is public GET-only; account access and exchange writes are blocked")
        return super().request(method, path, params=params)


def round_precision(value, digits, up=False):
    if isinstance(digits,bool) or int(digits)!=float(digits) or not 0<=int(digits)<=18:
        raise ValueError("Wallex stepSize/tickSize must be decimal-place counts; verify schema")
    d = Decimal(str(value))
    if not d.is_finite() or d<=0:
        raise ValueError("invalid decimal")
    return format(d.quantize(Decimal(1).scaleb(-int(digits)),rounding=ROUND_UP if up else ROUND_DOWN),"f")


def validate_market(market):
    """Validate required execution metadata; optional upper bounds may be absent.

    Missing exchange ceilings never remove engine cash, exposure or book limits.
    Precision fields are counts of decimal places, not tick amounts.
    """
    m=dict(market)
    try:
        for name in ('baseAsset','quoteAsset'):
            if not isinstance(m[name],str) or not m[name].isalnum():raise ValueError(name)
        for key,alias in (('stepSize','amount_precision'),('tickSize','price_precision')):
            v=m[key] if key in m else m[alias]
            if isinstance(v,bool) or int(v)!=float(v) or not 0<=int(v)<=18:raise ValueError(key)
            m[key]=int(v)
        for key in ('minQty','minNotional','maxQty','maxNotional'):
            if key.startswith('max') and m.get(key) is None:
                m[key]=None;continue
            value=float(m[key])
            if isinstance(m[key],bool) or not math.isfinite(value) or value<=0:raise ValueError(key)
            m[key]=value
        if m['maxQty'] is not None and m['maxQty']<m['minQty']:raise ValueError('quantity bounds')
        if m['maxNotional'] is not None and m['maxNotional']<m['minNotional']:raise ValueError('notional bounds')
        # This adapter submits LIMIT only. Do not infer spot closure from an
        # undocumented flag about the separate MARKET order type.
        if 'orderTypes' in m:
            if not isinstance(m['orderTypes'],list) or 'LIMIT' not in m['orderTypes']:
                raise ValueError('LIMIT orders unavailable')
    except (KeyError,ValueError,TypeError,OverflowError) as exc:
        raise APIError('required market metadata invalid or unavailable') from exc
    return m


def prepare_order(intent,market,book,config,now,client_id):
    market=validate_market(market)
    if intent.symbol not in config.symbols or market.get("quoteAsset") != "TMN":
        raise ValueError("market not permitted")
    if now-book["received_at"] > config.stale_seconds or now<book["received_at"]:
        raise ValueError("stale book")
    ask,bid = book["asks"][0][0],book["bids"][0][0]
    mid = (ask+bid)/2
    spread = (ask-bid)/mid*10000
    if spread > config.max_spread_bps and intent.side=="BUY":
        raise ValueError("spread exceeds limit")
    levels = book["asks"] if intent.side=="BUY" else book["bids"]
    # Marketable LIMIT has an explicit maximum slippage; it is not assumed IOC.
    worst = levels[0][0]*(1+config.max_book_slippage_bps/10000 if intent.side=="BUY"
                           else 1-config.max_book_slippage_bps/10000)
    available = sum(q for price,q in levels if (price<=worst if intent.side=="BUY" else price>=worst))
    price = round_precision(worst,market["tickSize"],up=intent.side=="BUY")
    upper=min(intent.quantity,available)
    if market['maxQty'] is not None:upper=min(upper,market['maxQty'])
    if market['maxNotional'] is not None:upper=min(upper,market['maxNotional']/float(price))
    qty = round_precision(upper,market["stepSize"])
    q,p = float(qty),float(price)
    if q<=0 or q<market["minQty"] or (market['maxQty'] is not None and q>market['maxQty']):
        raise ValueError("quantity outside exchange limits")
    if p*q < max(config.min_notional,float(market["minNotional"])):
        raise ValueError("below minimum notional; residual inventory requires settlement attention")
    return {"symbol":intent.symbol,"side":intent.side,"type":"LIMIT",
            "price":price,"quantity":qty,"client_id":client_id}
