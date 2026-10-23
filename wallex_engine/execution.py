"""Durable at-most-once submissions; ambiguous writes stop new trading."""
from dataclasses import asdict, replace
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
import hashlib
import json
import math
import sqlite3
import time
from .audit import canonical
from .engine import Intent, Portfolio
from .wallex import prepare_order


class UncertainOrder(RuntimeError):
    pass


def _decimal(value, label, positive=False):
    """Compare exchange decimals before converting the accounting result to float."""
    try:
        if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
            raise ValueError(label)
        number = Decimal(str(value))
        if not number.is_finite() or number < 0 or (positive and not number):
            raise ValueError(label)
        floating = float(number)
        if not math.isfinite(floating) or (number and not floating):
            raise ValueError(label)
        return number if number else Decimal(0)
    except (InvalidOperation, ValueError, TypeError, OverflowError) as exc:
        raise UncertainOrder(f"invalid {label}") from exc


def _bounds(intent, submitted_quantity=None, inventory_limit=None):
    if intent.side not in ("BUY", "SELL"):
        raise UncertainOrder("invalid intent side")
    intended = _decimal(intent.quantity, "intent quantity", positive=True)
    submitted = _decimal(intended if submitted_quantity is None else submitted_quantity,
                         "submitted quantity", positive=True)
    if submitted > intended:
        raise UncertainOrder("submitted quantity exceeds intent")
    inventory = None
    if intent.side == "SELL":
        inventory = _decimal(intended if inventory_limit is None else inventory_limit,
                             "pre-order inventory limit", positive=True)
        if submitted > inventory:
            raise UncertainOrder("submitted quantity exceeds pre-order inventory")
    elif inventory_limit is not None:
        # BUY has no base-inventory debit. Reject an accidental SELL-only argument.
        raise UncertainOrder("inventory limit is only valid for SELL")
    return submitted, inventory


class Journal:
    def __init__(self,path):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS orders (id TEXT PRIMARY KEY,intent TEXT,payload TEXT,status TEXT,result TEXT,seq INTEGER)")
        # Separate metadata keeps existing orders/replays and exchange payloads intact.
        self.db.execute("CREATE TABLE IF NOT EXISTS order_bounds (id TEXT PRIMARY KEY,submitted_quantity TEXT NOT NULL,inventory_limit TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY,value TEXT)")
        self.db.commit()

    def get(self,key):
        row = self.db.execute("SELECT value FROM state WHERE key=?",(key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self,key,value):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO state VALUES (?,?)",(key,canonical(value)))

    def reserve(self,oid,intent,payload,bounds=None):
        with self.db:
            self.db.execute("INSERT INTO orders VALUES (?,?,?,'SENDING',NULL,NULL)",
                            (oid,canonical(asdict(intent)),canonical(payload)))
            if bounds is not None:
                self.db.execute("INSERT INTO order_bounds VALUES (?,?,?)",
                                (oid,bounds["submitted_quantity"],bounds["inventory_limit"]))

    def execution_bounds(self,oid,intent):
        row = self.db.execute("SELECT payload FROM orders WHERE id=?",(oid,)).fetchone()
        if row is None:
            raise UncertainOrder("order has no durable reservation")
        try:
            payload_quantity = json.loads(row[0])["quantity"]
        except (KeyError,TypeError,ValueError) as exc:
            raise UncertainOrder("journal has no exact submitted quantity") from exc
        row = self.db.execute("SELECT submitted_quantity,inventory_limit FROM order_bounds WHERE id=?",
                              (oid,)).fetchone()
        # Legacy pending orders have no separate inventory budget: recover with the
        # exact saved payload and the conservative original intent quantity.
        submitted,inventory = _bounds(intent,payload_quantity,row[1] if row else None)
        if row and _decimal(row[0],"saved submitted quantity",positive=True) != submitted:
            raise UncertainOrder("saved execution bounds disagree with exchange payload")
        return submitted,inventory

    def finish(self,oid,result):
        with self.db:
            seq = self.db.execute("SELECT COALESCE(MAX(seq),0)+1 FROM orders").fetchone()[0]
            self.db.execute("UPDATE orders SET status='DONE',result=?,seq=? WHERE id=?",(canonical(result),seq,oid))

    def pending(self):
        return self.db.execute("SELECT id,intent FROM orders WHERE status IS NOT 'DONE'").fetchall()

    def replay(self,cash):
        p = Portfolio(cash)
        for intent,result in self.db.execute("SELECT intent,result FROM orders WHERE status='DONE' ORDER BY seq"):
            i,r = Intent(**json.loads(intent)),json.loads(result)
            if r["quantity"]:
                p.apply(i,r["quantity"],r["price"],r["fee"],r["ts"])
                p.fees += r.get("base_fee_tmn",0)
        return p

    def close(self):
        self.db.close()


def normalize_terminal(order,intent,market,now,submitted_quantity=None,inventory_limit=None):
    """Return inventory movement and quote cash flow without charging fees twice.

    ``quantity`` is net base received for BUY, total base debited for SELL.
    ``price`` converts that inventory movement back to the gross trade notional;
    only the quote fee is passed to Portfolio.apply. Base fees are already in the
    inventory movement and their per-fill quote valuation is reporting-only.
    """
    submitted,inventory = _bounds(intent,submitted_quantity,inventory_limit)
    if not isinstance(order,dict):
        raise UncertainOrder("invalid terminal order")
    if order.get("active") is not False:
        raise UncertainOrder("order is not confirmed inactive")
    gross = _decimal(order.get("executedQty"),"executed quantity")
    if gross > submitted:
        raise UncertainOrder("executed quantity exceeds exact submitted quantity")
    base,quote = market.get("baseAsset"),market.get("quoteAsset")
    if not isinstance(base,str) or not base or quote != "TMN" or base == quote:
        raise UncertainOrder("invalid fee-asset mapping")
    fills = order.get("fills",[])
    if not isinstance(fills,list):
        raise UncertainOrder("invalid exchange fills")
    if gross and not fills:
        raise UncertainOrder("filled order has no fee-asset breakdown")
    parsed = []
    for fill in fills:
        if not isinstance(fill,dict):
            raise UncertainOrder("invalid exchange fill")
        q = _decimal(fill.get("quantity"),"fill quantity",positive=True)
        price = _decimal(fill.get("price"),"fill price",positive=True)
        fee = _decimal(fill.get("fee"),"fill fee")
        asset = fill.get("feeAsset")
        if asset not in (base,quote) and fee:
            raise UncertainOrder("unmapped fee asset")
        parsed.append((q,price,fee,asset))
    # Enough precision for exact addition/products of every input decimal. The
    # inventory/overfill comparisons must not round away even a small base fee.
    values = [submitted,gross]+([inventory] if inventory is not None else [])
    values += [value for q,p,f,_ in parsed for value in (q,p,f)]
    nonzero = [value for value in values if value]
    span = max(v.adjusted() for v in nonzero)-min(v.as_tuple().exponent for v in nonzero)+1
    with localcontext() as context:
        context.prec = max(80,2*span+len(str(max(1,len(parsed)))))
        quantity=notional=base_fee=quote_fee=base_fee_tmn=Decimal(0)
        for q,price,fee,asset in parsed:
            quantity += q
            notional += q*price
            if asset == base:
                base_fee += fee
                base_fee_tmn += fee*price
            elif asset == quote:
                quote_fee += fee
        if quantity != gross:
            raise UncertainOrder("fill list disagrees with executed quantity; do not assume unreported fees")
        if quantity > submitted:
            raise UncertainOrder("fills exceed exact submitted quantity")
        effective = quantity-base_fee if intent.side=="BUY" else quantity+base_fee
        if gross and effective <= 0:
            raise UncertainOrder("invalid net fill")
        if inventory is not None and effective > inventory:
            raise UncertainOrder("base-asset debit exceeds pre-order inventory limit")
        amounts = {"quantity":effective,"price":notional/effective if effective else Decimal(0),
                   "fee":quote_fee,"base_fee_tmn":base_fee_tmn,"gross_quantity":quantity,
                   "gross_notional":notional,"base_fee_quantity":base_fee,"quote_fee_tmn":quote_fee}
        result = {}
        for name,value in amounts.items():
            # Portfolio uses floats; reject unrepresentable values instead of
            # committing a terminal result that could corrupt its arithmetic.
            result[name] = float(_decimal(value,"normalized "+name))
    return {**result,"ts":now,"order":order}


class LiveExecutor:
    def __init__(self,api,config,journal,audit):
        self.api,self.config,self.journal,self.audit = api,config,journal,audit

    def complete(self,oid,intent,market):
        # This GET is also the only recovery path after a timed-out POST.
        try:
            submitted,inventory = self.journal.execution_bounds(oid,intent)
            order = self.api.account(self.config,"order",client_id=oid)
            if order.get("active") is True:
                self.api.account(self.config,"cancel",method="DELETE",client_id=oid)
                order = self.api.account(self.config,"order",client_id=oid)
            if order.get("clientOrderId") != oid or order.get("symbol") != intent.symbol or order.get("side") != intent.side:
                raise UncertainOrder("order identity mismatch")
            result = normalize_terminal(order,intent,market,int(time.time()),
                                        submitted_quantity=submitted,inventory_limit=inventory)
        except Exception as exc:
            raise UncertainOrder(f"reconcile order {oid} before any new submission") from exc
        self.journal.finish(oid,result)
        self.audit.write("exchange_terminal",int(time.time()),client_id=oid,result=result)
        return result

    def recover(self,markets):
        for oid,raw in self.journal.pending():
            intent = Intent(**json.loads(raw))
            self.complete(oid,intent,markets[intent.symbol])

    def execute(self,intent,market,book,inventory_limit=None,submission_quantity=None):
        if self.journal.pending():
            raise UncertainOrder("unresolved order blocks new submissions")
        identity = canonical(asdict(intent))
        oid = "wx-"+hashlib.sha256(identity.encode()).hexdigest()[:28]
        if self.journal.db.execute("SELECT 1 FROM orders WHERE id=?",(oid,)).fetchone():
            return None  # immutable decision identity: never submit twice
        requested,inventory = _bounds(intent,submission_quantity,inventory_limit)
        submitted_intent = intent if submission_quantity is None else replace(intent,quantity=float(requested))
        payload = prepare_order(submitted_intent,market,book,self.config,time.time(),oid)
        submitted,inventory = _bounds(intent,payload["quantity"],inventory)
        if submitted > requested:
            raise UncertainOrder("prepared quantity exceeds requested submission quantity")
        bounds = {"submitted_quantity":str(submitted),
                  "inventory_limit":str(inventory) if inventory is not None else None}
        self.journal.reserve(oid,intent,payload,bounds)  # BOTH bounds durable BEFORE network
        self.audit.write("order_intent",int(time.time()),client_id=oid,intent=asdict(intent),
                         payload=payload,execution_bounds=bounds)
        try:
            response = self.api.account(self.config,"orders",method="POST",body=payload)
        except Exception as exc:
            # No retry, including after 4xx: a human/exchange must resolve permanent ambiguity.
            raise UncertainOrder(f"submission {oid} unresolved; automatic resubmission disabled") from exc
        if response.get("clientOrderId") != oid:
            raise UncertainOrder("exchange did not echo client_id; verify its semantics before live use")
        return self.complete(oid,intent,market)
