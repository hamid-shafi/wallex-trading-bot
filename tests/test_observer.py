import io
import json
from dataclasses import asdict
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch

from observer import server
from wallex_engine.config import Config


class ObserverTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def make_run(self,name,started=100,mode='live',portfolio=None):
        directory=self.root/name/'session';directory.mkdir(parents=True)
        config=Config(initial_cash=2_000_000,gross_cap=.5,max_drawdown=.05)
        states={'identity':{'mode':mode,'config':asdict(config)},
                'trial':{'started_at':started,'deadline':started+1800,'reason':'time_limit',
                         'limits':{'seconds':1800,'floor_tmn':None,'ceiling_tmn':None}},
                'summary':{'cash_tmn':1_900_000,'realized_pnl_tmn':-100,'fees_tmn':30,'fills':1}}
        if portfolio is not None:states['portfolio']=portfolio
        with sqlite3.connect(directory/'session.sqlite') as db:
            db.execute('create table state (key text primary key, value text)')
            db.execute('create table orders (seq integer, id text, intent text, status text, result text)')
            db.executemany('insert into state values (?,?)',[(k,json.dumps(v)) for k,v in states.items()])
        return directory

    def test_recorded_risk_and_budget_override_seed_assumptions(self):
        self.make_run('operator-v4')
        data=server.journal(self.root)
        risk=server.session_risk(data)
        self.assertEqual(data['ledger']['cash_tmn'],2_000_000)
        self.assertEqual(risk['budget_tmn'],2_000_000)
        self.assertEqual(risk['risk']['gross_cap'],.5)
        self.assertEqual(risk['risk']['daily_loss'],.02)
        self.assertTrue(risk['risk']['limits_known'])
        self.assertIsNone(risk['risk']['floor'])
        self.assertEqual(risk['risk']['ends_at'],1900)

    def test_latest_run_and_live_paper_separation(self):
        self.make_run('operator-v3',100)
        self.make_run('operator-v4',200)
        self.make_run('paper-monday',300,mode='shadow')
        self.assertEqual(server.journal(self.root)['run_directory'],'paper-monday')
        self.assertEqual(server.journal(self.root,mode='live')['run_directory'],'operator-v4')
        self.assertEqual(server.journal(self.root,selected_run='operator-v3')['run_directory'],'operator-v3')

    def test_shadow_ledger_comes_from_portfolio_without_live_order_rows(self):
        fill={'ts':100,'symbol':'BTCTMN','side':'BUY','quantity':.001,'price':100_000,'fee':2}
        self.make_run('paper-monday',mode='shadow',portfolio={'cash':1_800_000,'realized':-27,'fees':12,'fills':[fill]})
        data=server.journal(self.root,selected_run='paper-monday')
        self.assertEqual(data['ledger'],{'cash_tmn':1_800_000,'realized_pnl_tmn':-27,'fees_tmn':12,'fills':1})
        self.assertEqual(data['paper_trades'][0]['quote_asset'],'TMN')
        self.assertEqual(data['effective_config']['initial_cash'],2_000_000)

    def test_shadow_summary_fallback_is_not_zero(self):
        self.make_run('paper-monday',mode='shadow')
        data=server.journal(self.root,selected_run='paper-monday')
        self.assertEqual(data['ledger']['fills'],1)
        self.assertEqual(data['ledger']['realized_pnl_tmn'],-100)

    def test_rejects_path_escape_and_symlink_runs(self):
        self.make_run('operator-v3')
        (self.root/'linked').symlink_to(self.root/'operator-v3',target_is_directory=True)
        for name in ('../operator-v3','/tmp','linked'):
            with self.subTest(name=name),self.assertRaises(ValueError):server.journal(self.root,selected_run=name)
        self.assertEqual(server.journal(self.root)['run_directory'],'operator-v3')

    def test_current_time_limit_and_residual_override_historical_maxqty(self):
        result=server.engine_presentation({'engine':{'phase':'observer_only'},'market_compatibility':'pass',
            'prior_engine':{'reason':'market_schema_missing_maxQty'},'journal':{
                'trial':{'reason':'time_limit'},'summary':{'manual_reconciliation_required':True,
                'settled_to_tmn':False,'positions':{'BTCTMN':{'quantity':1e-7}}}}})
        self.assertIn('پایان یافته',result['title'])
        self.assertIn('پایان مهلت زمانی',' '.join(result['warnings']))
        self.assertIn('تطبیق',' '.join(result['warnings']))
        self.assertIn('BTCTMN',' '.join(result['warnings']))
        self.assertNotIn('سقف اختیاری',' '.join(result['warnings']))
        self.assertIn('نسخهٔ قدیمی',' '.join(result['history']))

    def test_worker_unreachable_is_distinct_from_not_started(self):
        api=Mock();api.markets.side_effect=RuntimeError;api.request.side_effect=RuntimeError
        with patch.object(server,'urlopen',side_effect=OSError('offline')) as probe:
            unknown=server.snapshot(api,self.root)
            self.assertEqual(unknown['engine']['phase'],'unavailable')
            self.assertIn('worker_health_unavailable',unknown['errors'])
            probe.reset_mock()
            disabled=server.snapshot(api,self.root,worker_expected=False)
            self.assertEqual(disabled['engine']['phase'],'observer_only')
            probe.assert_not_called()

    def test_quote_asset_does_not_assume_toman(self):
        self.assertEqual(server.quote_asset('BTCUSDT'),'USDT')
        self.assertEqual(server.quote_asset('BTCTMN'),'TMN')
        self.assertIsNone(server.quote_asset('UNKNOWN'))

    def test_readonly_routes_and_signed_cookie_still_required(self):
        handler=server.handler(server.Store(),'a'*32,'test')
        request=handler.__new__(handler);request.headers={};request.send=Mock()
        request.path='/api/snapshot';request.do_GET();self.assertEqual(request.send.call_args.args[0],401)
        request.path='/orders';request.do_POST();self.assertEqual(request.send.call_args.args[0],405)
        request.path='/login';request.headers={'Content-Length':'38','Host':'local'}
        request.client_address=('127.0.0.1',1);request.rfile=io.BytesIO(('token='+'a'*32).encode())
        request.do_POST();cookie=request.send.call_args.kwargs['extra']['Set-Cookie']
        self.assertIn('HttpOnly',cookie);self.assertIn('SameSite=Strict',cookie);self.assertIn('Secure',cookie)
        request.headers={'Cookie':cookie.split(';',1)[0]};request.path='/api/snapshot';request.do_GET()
        self.assertEqual(request.send.call_args.args[0],200)
        request.headers={'Cookie':cookie.split(';',1)[0]+'x'};request.do_GET()
        self.assertEqual(request.send.call_args.args[0],401)

    def test_activation_gate_unchanged(self):
        self.assertFalse(server.engine_allowed({},'release'))
        self.assertFalse(server.engine_allowed({'OPERATOR_ARM':'I_ACCEPT_REAL_SPOT_TRADING','OPERATOR_START_RELEASE':'old'},'release'))
        self.assertTrue(server.engine_allowed({'OPERATOR_ARM':'I_ACCEPT_REAL_SPOT_TRADING','OPERATOR_START_RELEASE':'release'},'release'))

if __name__=='__main__':unittest.main()
