import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from wallex_engine.config import Config
from wallex_engine.data import Bar,save_bars
from wallex_engine.engine import Engine,Intent
from wallex_engine.runtime import run_session
from wallex_engine.session_limits import SessionLimits
from wallex_engine.wallex import PublicWallex,APIError


class PublicFixture:
    def __init__(self,now):self.now=now
    def markets(self):return {'BTCTMN':{'minNotional':100000,'tickSize':2}}
    def depth(self,symbol):
        return dict(asks=[(100000.,100.)],bids=[(99900.,100.)],received_at=self.now)
    def history(self,*args):raise AssertionError('fixture should use existing closed candle')
    def account(self,*args):raise AssertionError('paper mode accessed account')


class PaperDeadlineTests(unittest.TestCase):
    def test_resume_keeps_deadline_and_closes_only_virtual_position(self):
        now=1792780200
        limits=SessionLimits(None,None,604800,now+60)
        api=PublicFixture(now)
        config=Config(symbols=['BTCTMN'],interval=3600)
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()):
            root=Path(folder);csv=root/'bars.csv'
            save_bars(csv,[Bar(now//3600*3600-3600,'BTCTMN',100000,101000,99000,100000,100)])
            with patch('wallex_engine.runtime.time.time',return_value=now),patch.object(Engine,'plan',return_value=([Intent('BTCTMN','BUY',2,'fixture',.02)],[])) as plan:
                first=run_session(api,config,csv,root/'session',session_limits=limits)
                self.assertEqual(first['fills'],1)
                self.assertEqual(plan.call_count,1)
            api.now=now+61
            with patch('wallex_engine.runtime.time.time',return_value=now+61),patch.object(Engine,'plan',side_effect=AssertionError('entry planning after deadline')):
                final=run_session(api,config,csv,root/'session',session_limits=limits)
            self.assertEqual(final['trial']['deadline'],now+60)
            self.assertEqual(final['trial']['reason'],'time_limit')
            self.assertTrue(final['settled_to_tmn'])
            self.assertTrue(final['stopped'])
            self.assertEqual(final['fills'],2)
            self.assertGreater(final['fees_tmn'],0)
            self.assertLess(final['cash_tmn'],config.initial_cash)

    def test_public_client_rejects_private_reads_and_all_writes(self):
        api=PublicWallex()
        self.assertEqual(api.key,'')
        for method,path,private in [('POST','/v1/account/orders',False),('DELETE','/v1/account/orders',False),('GET','/v1/account/balances',True)]:
            with self.subTest(method=method,path=path),self.assertRaises((APIError,ValueError)):
                api.request(method,path,private=private)


if __name__=='__main__':unittest.main()
