"""Synthetic fee/accounting fixtures; no network, credentials or trading runner."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from wallex_engine.config import Config
from wallex_engine.engine import Intent
from wallex_engine.execution import Journal, LiveExecutor, UncertainOrder, normalize_terminal


MARKET = {"baseAsset":"BTC","quoteAsset":"TMN","stepSize":8,"tickSize":0,
          "minQty":.00000001,"minNotional":1,"maxQty":None,"maxNotional":None}
BOOK = {"symbol":"BTCTMN","received_at":1000.,"asks":[(100.01,100.)],"bids":[(100.,100.)]}


def fill(quantity, price="100", fee="0", asset="TMN"):
    return {"quantity":str(quantity),"price":str(price),"fee":str(fee),"feeAsset":asset}


def terminal(quantity, fills, **extra):
    return {"active":False,"executedQty":str(quantity),"fills":fills,**extra}


class FakeAudit:
    def __init__(self):
        self.events=[]

    def write(self, kind, ts, **values):
        self.events.append((kind,ts,values))


class FakeAPI:
    """Only predetermined in-memory responses, with observable request history."""
    def __init__(self, order, after_post=None, timeout=False):
        self.order=deepcopy(order)
        self.after_post=after_post
        self.timeout=timeout
        self.calls=[]
        self.posted=None

    def account(self, config, name, method="GET", client_id=None, body=None):
        self.calls.append((name,method,client_id,deepcopy(body)))
        if name=="orders" and method=="POST":
            self.posted=deepcopy(body)
            if self.after_post:
                self.after_post(body)
            if self.timeout:
                raise TimeoutError("synthetic timeout after durable reservation")
            return {"clientOrderId":body["client_id"]}
        if name=="order" and method=="GET":
            result=deepcopy(self.order)
            result.setdefault("clientOrderId",client_id)
            result.setdefault("symbol","BTCTMN")
            result.setdefault("side","SELL")
            return result
        if name=="cancel" and method=="DELETE":
            self.order["active"]=False
            return {}
        raise AssertionError("unexpected fake API operation")


class TerminalFeeTests(unittest.TestCase):
    def normalized(self, side, quantity, fills, executed=None, **bounds):
        intent=Intent("BTCTMN",side,quantity,"synthetic",decision_ts=1000)
        order=terminal(quantity if executed is None else executed,fills)
        return normalize_terminal(order,intent,MARKET,1001,**bounds)

    def test_buy_base_fee_reduces_received_inventory_without_second_cash_charge(self):
        result=self.normalized("BUY",2,[fill(2,100,.006,"BTC")])
        self.assertEqual(result["quantity"],1.994)
        self.assertEqual(result["fee"],0)
        self.assertEqual(result["gross_quantity"],2)
        self.assertEqual(result["gross_notional"],200)
        self.assertEqual(result["base_fee_quantity"],.006)
        self.assertEqual(result["base_fee_tmn"],.6)
        self.assertAlmostEqual(result["quantity"]*result["price"],200)

    def test_sell_quote_fee_does_not_reduce_submitted_inventory(self):
        result=self.normalized("SELL",2,[fill(2,110,.66,"TMN")])
        self.assertEqual(result["quantity"],2)
        self.assertEqual(result["price"],110)
        self.assertEqual(result["fee"],.66)
        self.assertEqual(result["quote_fee_tmn"],.66)
        self.assertEqual(result["base_fee_tmn"],0)

    def test_sell_base_fee_can_debit_more_than_submission_with_separate_inventory_budget(self):
        result=self.normalized("SELL",1,[fill(1,110,.003,"BTC")],
                               submitted_quantity="1",inventory_limit="1.003")
        self.assertEqual(result["quantity"],1.003)
        self.assertEqual(result["gross_quantity"],1)
        self.assertAlmostEqual(result["quantity"]*result["price"],110)
        self.assertEqual(result["base_fee_tmn"],.33)
        with self.assertRaisesRegex(UncertainOrder,"pre-order inventory"):
            self.normalized("SELL",1,[fill(1,110,.003,"BTC")])

    def test_partial_sell_base_fee_uses_actual_fill_and_preserves_unfilled_inventory(self):
        result=self.normalized("SELL",1,[fill(.4,100,.0012,"BTC")],executed=".4",
                               submitted_quantity="1",inventory_limit="1.003")
        self.assertEqual(result["quantity"],.4012)
        self.assertEqual(result["gross_notional"],40)
        self.assertEqual(result["base_fee_tmn"],.12)

    def test_base_fee_valuation_uses_each_fill_price_not_aggregate_vwap(self):
        result=self.normalized("BUY",2,[fill(1,100,.01,"BTC"),fill(1,200,.03,"BTC")])
        self.assertEqual(result["gross_notional"],300)
        self.assertEqual(result["base_fee_quantity"],.04)
        self.assertEqual(result["base_fee_tmn"],7)  # Aggregate VWAP would incorrectly give 6.

    def test_mixed_base_and_quote_fees_remain_separate(self):
        result=self.normalized("BUY",2,[fill(1,100,.01,"BTC"),fill(1,200,3,"TMN")])
        self.assertEqual(result["quantity"],1.99)
        self.assertEqual(result["base_fee_tmn"],1)
        self.assertEqual(result["fee"],3)
        self.assertEqual(result["quote_fee_tmn"],3)

    def test_decimal_fill_sum_accepts_exact_point_one_plus_point_two(self):
        result=self.normalized("BUY",.3,[fill(".1",100),fill(".2",200)])
        self.assertEqual(result["gross_quantity"],.3)
        self.assertEqual(result["gross_notional"],50)

    def test_overfill_above_exact_payload_is_not_hidden_by_intent_or_tolerance(self):
        with self.assertRaisesRegex(UncertainOrder,"exact submitted"):
            self.normalized("SELL",2,[fill("1.00000001")],executed="1.00000001",
                            submitted_quantity="1.00000000",inventory_limit=2)

    def test_incomplete_or_excess_fill_list_is_not_silently_tolerated(self):
        for fills in ([fill(".99999999999")],[fill("1.00000000001")],[]):
            with self.subTest(fills=fills),self.assertRaises(UncertainOrder):
                self.normalized("BUY",1,fills)

    def test_inventory_overspend_including_tiny_base_fee_fails_closed(self):
        for fee in (".003000001","1e-100"):
            inventory="1.003" if fee.startswith(".") else "1"
            with self.subTest(fee=fee),self.assertRaisesRegex(UncertainOrder,"pre-order inventory"):
                self.normalized("SELL",1,[fill(1,100,fee,"BTC")],inventory_limit=inventory)

    def test_unknown_nonzero_fee_asset_and_missing_fee_details_fail_closed(self):
        cases=[fill(1,100,.01,"USDT"),{k:v for k,v in fill(1).items() if k!="fee"},
               {k:v for k,v in fill(1,100,.01).items() if k!="feeAsset"}]
        for value in cases:
            with self.subTest(fill=value),self.assertRaises(UncertainOrder):
                self.normalized("BUY",1,[value])

    def test_zero_fee_needs_no_invented_fee_asset(self):
        result=self.normalized("BUY",1,[fill(1,100,0,"")])
        self.assertEqual(result["fee"],0)
        self.assertEqual(result["base_fee_tmn"],0)

    def test_nonfinite_boolean_negative_and_missing_exchange_numbers_fail_closed(self):
        for field in ("quantity","price","fee"):
            for invalid in (True,False,"NaN","Infinity","-Infinity",-1,None):
                value=fill(1)
                value[field]=invalid
                with self.subTest(field=field,value=invalid),self.assertRaises(UncertainOrder):
                    self.normalized("BUY",1,[value])
        for invalid in (True,"NaN","Infinity",-1,None):
            with self.subTest(executed=invalid),self.assertRaises(UncertainOrder):
                normalize_terminal({"active":False,"executedQty":invalid,"fills":[]},
                                   Intent("BTCTMN","BUY",1,"fixture"),MARKET,1001)

    def test_zero_execution_is_explicit_and_cannot_contradict_fills(self):
        result=self.normalized("BUY",1,[],executed="0")
        for key in ("quantity","price","fee","gross_quantity","gross_notional","base_fee_tmn"):
            self.assertEqual(result[key],0)
        with self.assertRaises(UncertainOrder):
            self.normalized("BUY",1,[fill(.1)],executed="0")
        with self.assertRaises(UncertainOrder):
            normalize_terminal({"active":False},Intent("BTCTMN","BUY",1,"fixture"),MARKET,1001)

    def test_buy_fee_cannot_consume_all_received_base(self):
        for fee in (1,2):
            with self.subTest(fee=fee),self.assertRaisesRegex(UncertainOrder,"net fill"):
                self.normalized("BUY",1,[fill(1,100,fee,"BTC")])

    def test_inactive_confirmation_and_mapping_are_required(self):
        intent=Intent("BTCTMN","BUY",1,"fixture")
        for active in (None,True,0,"false"):
            with self.subTest(active=active),self.assertRaises(UncertainOrder):
                normalize_terminal(terminal(1,[fill(1)],active=active),intent,MARKET,1001)
        with self.assertRaisesRegex(UncertainOrder,"fee-asset"):
            normalize_terminal(terminal(1,[fill(1)]),intent,{**MARKET,"quoteAsset":"USDT"},1001)


class DurableFeeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/"session.sqlite"
        self.journal=Journal(self.path)
        self.addCleanup(lambda:self.journal.close())
        self.config=Config(symbols=["BTCTMN"],min_notional=1)
        self.audit=FakeAudit()
        self.clock=patch("wallex_engine.execution.time.time",return_value=1000.)
        self.clock.start()
        self.addCleanup(self.clock.stop)

    def executor(self,api):
        return LiveExecutor(api,self.config,self.journal,self.audit)

    def test_bounds_are_committed_before_post_and_never_sent_to_exchange(self):
        def inspect(body):
            with sqlite3.connect(self.path) as db:
                order=db.execute("SELECT status,payload FROM orders").fetchone()
                bounds=db.execute("SELECT submitted_quantity,inventory_limit FROM order_bounds").fetchone()
            self.assertEqual(order[0],"SENDING")
            self.assertEqual(json.loads(order[1]),body)
            self.assertEqual(bounds,("1.00000000","1.003"))
            self.assertEqual(set(body),{"symbol","side","type","price","quantity","client_id"})
        api=FakeAPI(terminal(1,[fill(1,100,.003,"BTC")]),after_post=inspect)
        result=self.executor(api).execute(Intent("BTCTMN","SELL",1,"fixture"),MARKET,BOOK,
                                          inventory_limit=1.003)
        self.assertEqual(result["quantity"],1.003)
        self.assertEqual(self.journal.pending(),[])

    def test_timeout_restart_recovers_persisted_inventory_budget_without_resubmission(self):
        intent=Intent("BTCTMN","SELL",1,"fixture")
        api=FakeAPI(terminal(1,[fill(1,100,.003,"BTC")]),timeout=True)
        with self.assertRaisesRegex(UncertainOrder,"automatic resubmission disabled"):
            self.executor(api).execute(intent,MARKET,BOOK,inventory_limit=1.003)
        self.assertEqual(len(self.journal.pending()),1)
        self.journal.close()
        self.journal=Journal(self.path)
        recovery=FakeAPI(terminal(1,[fill(1,100,.003,"BTC")]))
        executor=self.executor(recovery)
        executor.recover({"BTCTMN":MARKET})
        self.assertEqual([call[0] for call in recovery.calls],["order"])
        stored=json.loads(self.journal.db.execute("SELECT result FROM orders").fetchone()[0])
        self.assertEqual(stored["quantity"],1.003)
        self.assertEqual(stored["base_fee_tmn"],.3)
        executor.recover({"BTCTMN":MARKET})
        self.assertIsNone(executor.execute(intent,MARKET,BOOK,inventory_limit=1.003))
        self.assertEqual(len(recovery.calls),1)

    def test_optional_submission_reserve_preserves_original_intent(self):
        intent=Intent("BTCTMN","SELL",1.003,"fixture")
        api=FakeAPI(terminal(1,[fill(1,100,.003,"BTC")]))
        result=self.executor(api).execute(intent,MARKET,BOOK,inventory_limit=1.003,submission_quantity=1)
        raw=self.journal.db.execute("SELECT intent,payload FROM orders").fetchone()
        self.assertEqual(json.loads(raw[0]),asdict(intent))
        self.assertEqual(json.loads(raw[1])["quantity"],"1.00000000")
        self.assertEqual(result["quantity"],1.003)

    def test_market_capped_submission_rejects_overfill_even_below_original_intent(self):
        api=FakeAPI(terminal("1.00000001",[fill("1.00000001")]))
        with self.assertRaises(UncertainOrder):
            self.executor(api).execute(Intent("BTCTMN","SELL",2,"fixture"),
                                      {**MARKET,"maxQty":1},BOOK,inventory_limit=2)
        self.assertEqual(api.posted["quantity"],"1.00000000")
        self.assertEqual(len(self.journal.pending()),1)
        self.assertIsNone(self.journal.db.execute("SELECT result FROM orders").fetchone()[0])

    def test_inventory_overspend_stays_unresolved_and_blocks_new_submission(self):
        api=FakeAPI(terminal(1,[fill(1,100,.003,"BTC")]))
        executor=self.executor(api)
        with self.assertRaises(UncertainOrder):
            executor.execute(Intent("BTCTMN","SELL",1,"fixture"),MARKET,BOOK,inventory_limit=1)
        calls=len(api.calls)
        with self.assertRaisesRegex(UncertainOrder,"unresolved order"):
            executor.execute(Intent("BTCTMN","SELL",1,"new-decision"),MARKET,BOOK,inventory_limit=1)
        self.assertEqual(len(api.calls),calls)
        self.assertEqual(len(self.journal.pending()),1)

    def test_null_or_unknown_status_remains_pending_and_cannot_hide_ambiguity(self):
        intent=Intent("BTCTMN","SELL",1,"fixture")
        self.journal.reserve("ambiguous",intent,{"quantity":"1"})
        for status in (None,"UNKNOWN"):
            with self.subTest(status=status):
                with self.journal.db:
                    self.journal.db.execute("UPDATE orders SET status=?",(status,))
                api=FakeAPI({"active":False})  # No reliable executed quantity/fills.
                executor=self.executor(api)
                self.assertEqual(len(self.journal.pending()),1)
                with self.assertRaisesRegex(UncertainOrder,"unresolved order"):
                    executor.execute(Intent("BTCTMN","SELL",1,"new-decision"),MARKET,BOOK)
                self.assertEqual(api.calls,[])
                with self.assertRaises(UncertainOrder):
                    executor.recover({"BTCTMN":MARKET})
                self.assertEqual(len(self.journal.pending()),1)
                self.assertIsNone(self.journal.db.execute("SELECT result FROM orders").fetchone()[0])

    def test_legacy_pending_order_uses_saved_payload_and_conservative_intent_budget(self):
        intent=Intent("BTCTMN","SELL",2,"legacy")
        self.journal.reserve("legacy",intent,{"quantity":"1.00000000"})
        api=FakeAPI(terminal("1.00000001",[fill("1.00000001")]))
        with self.assertRaises(UncertainOrder):
            self.executor(api).recover({"BTCTMN":MARKET})
        self.assertEqual(len(self.journal.pending()),1)
        api.order=terminal(1,[fill(1,100,.003,"BTC")])
        self.executor(api).recover({"BTCTMN":MARKET})
        self.assertEqual(self.journal.pending(),[])

    def test_legacy_pending_order_cannot_invent_a_base_fee_allowance(self):
        self.journal.reserve("legacy",Intent("BTCTMN","SELL",1,"legacy"),{"quantity":"1"})
        api=FakeAPI(terminal(1,[fill(1,100,.003,"BTC")]))
        with self.assertRaises(UncertainOrder):
            self.executor(api).recover({"BTCTMN":MARKET})
        self.assertEqual(len(self.journal.pending()),1)

    def test_saved_payload_mismatch_or_missing_quantity_stops_before_lookup(self):
        intent=Intent("BTCTMN","SELL",2,"fixture")
        self.journal.reserve("mismatch",intent,{"quantity":"1"},
                             {"submitted_quantity":"2","inventory_limit":"2"})
        api=FakeAPI(terminal(1,[fill(1)]))
        with self.assertRaises(UncertainOrder):
            self.executor(api).recover({"BTCTMN":MARKET})
        self.assertEqual(api.calls,[])
        with self.journal.db:
            self.journal.db.execute("UPDATE orders SET payload='{}'")
        with self.assertRaises(UncertainOrder):
            self.executor(api).recover({"BTCTMN":MARKET})
        self.assertEqual(api.calls,[])

    def test_invalid_bounds_never_reserve_or_send(self):
        for kwargs in ({"inventory_limit":True},{"inventory_limit":float("nan")},
                       {"inventory_limit":float("inf")},{"inventory_limit":.5},
                       {"submission_quantity":1.01},{"submission_quantity":True},
                       {"submission_quantity":float("nan")},{"submission_quantity":0}):
            api=FakeAPI(terminal(1,[fill(1)]))
            with self.subTest(kwargs=kwargs),self.assertRaises(UncertainOrder):
                self.executor(api).execute(Intent("BTCTMN","SELL",1,"fixture"),MARKET,BOOK,**kwargs)
            self.assertEqual(api.calls,[])
            self.assertEqual(self.journal.pending(),[])

    def test_active_partial_order_cancels_then_normalizes_actual_fill(self):
        api=FakeAPI(terminal(".4",[fill(".4",100,".0012","BTC")],active=True))
        result=self.executor(api).execute(Intent("BTCTMN","SELL",1,"fixture"),MARKET,BOOK,
                                          inventory_limit=1.003)
        self.assertEqual([call[0] for call in api.calls],["orders","order","cancel","order"])
        self.assertEqual(result["quantity"],.4012)

    def test_mixed_fees_replay_cash_realized_and_reported_fees_once(self):
        buy=Intent("BTCTMN","BUY",2,"buy",decision_ts=1)
        sell=Intent("BTCTMN","SELL",1.99,"sell",decision_ts=2)
        buy_result=normalize_terminal(terminal(2,[fill(1,100,.01,"BTC"),fill(1,200,3,"TMN")]),
                                      buy,MARKET,1)
        sell_result=normalize_terminal(terminal(1.99,[fill(1.99,210,4.179,"TMN")]),sell,MARKET,2)
        for oid,intent,result in (("buy",buy,buy_result),("sell",sell,sell_result)):
            self.journal.reserve(oid,intent,{"quantity":str(intent.quantity)})
            self.journal.finish(oid,result)
        for _ in range(2):
            p=self.journal.replay(1000)
            self.assertEqual(p.positions,{})
            self.assertAlmostEqual(p.cash,1110.721)
            self.assertAlmostEqual(p.realized,110.721)
            self.assertAlmostEqual(p.fees,8.179)
            self.assertEqual(len(p.fills),2)

    def test_sell_base_fee_replay_debits_reserved_inventory_but_not_cash_twice(self):
        buy=Intent("BTCTMN","BUY",1.003,"buy",decision_ts=1)
        sell=Intent("BTCTMN","SELL",1,"sell",decision_ts=2)
        buy_result=normalize_terminal(terminal("1.003",[fill("1.003",100)]),buy,MARKET,1)
        self.journal.reserve("buy",buy,{"quantity":"1.003"})
        self.journal.finish("buy",buy_result)
        api=FakeAPI(terminal(1,[fill(1,110,.003,"BTC")]))
        self.executor(api).execute(sell,MARKET,BOOK,inventory_limit=1.003)
        self.journal.close()
        self.journal=Journal(self.path)
        p=self.journal.replay(1000)
        self.assertEqual(p.positions,{})
        self.assertAlmostEqual(p.cash,1009.7)
        self.assertAlmostEqual(p.realized,9.7)
        self.assertAlmostEqual(p.fees,.33)
        self.assertEqual(len(p.fills),2)

    def test_existing_result_rows_replay_without_new_component_fields(self):
        old=Intent("BTCTMN","BUY",1,"legacy")
        self.journal.reserve("legacy-done",old,{"offline":True})
        self.journal.finish("legacy-done",{"quantity":1,"price":100,"fee":.2,"ts":1})
        rows=self.journal.db.execute("SELECT * FROM orders").fetchall()
        self.journal.close()
        self.journal=Journal(self.path)
        self.assertEqual(self.journal.db.execute("SELECT * FROM orders").fetchall(),rows)
        portfolio=self.journal.replay(1000)
        self.assertEqual(portfolio.positions["BTCTMN"].quantity,1)
        self.assertAlmostEqual(portfolio.cash,899.8)
        self.assertEqual(portfolio.fees,.2)


if __name__=="__main__":
    unittest.main()
