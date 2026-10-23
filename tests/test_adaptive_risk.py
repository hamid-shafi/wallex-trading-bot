"""Offline fixtures; no network clients or private account operations."""
from dataclasses import replace
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from wallex_engine.adaptive_risk import AdaptiveRisk, CAPS, reduce_entry, reduced_config
from wallex_engine.config import Config
from wallex_engine.data import Bar, save_bars
from wallex_engine.engine import Engine, Intent
from wallex_engine.execution import Journal
from wallex_engine.gpr import GPR_NAMES
from wallex_engine.runtime import run_session
from wallex_engine.session_limits import SessionLimits

NOW = 1_792_780_210
SYMBOL = 'BTCTMN'


def history(c, volatility=0.):
    end = NOW // c.interval * c.interval
    step = volatility / math.sqrt(86400 / c.interval)
    prices = [100000 * math.exp((i % 2) * step) for i in range(25)]
    return [Bar(end-(25-i)*c.interval, SYMBOL, p, p, p, p, 100.) for i,p in enumerate(prices)]


def book(c, spread=2., depth_ratio=100., now=NOW):
    ask, bid = 100000*(1+spread/20000), 100000*(1-spread/20000)
    return {'received_at':now, 'asks':[(ask,depth_ratio*c.min_notional/ask)],
            'bids':[(bid,depth_ratio*c.min_notional/bid)]}


def article(key, source='bbc_middle_east', age=60, conflict=0, security=0):
    return {'kind':'article','id':key,'duplicate_key':key,'source':source,
            'published_at':NOW-age,'first_seen_at':NOW-30,'available_at':NOW-30,
            'topics':{'conflict':conflict,'crypto_security':security}}


def calm_rows():
    return [article('a'),article('b','guardian_middle_east')]


def gpr(**changes):
    row = {'kind':'gpr_vintage','id':'vintage','feature_names':GPR_NAMES,
           'features':[1.,1.,0.,0.,0.,1.],'latest_observation':NOW-3*86400,
           'vintage_date':NOW-2*86400,'available_at':NOW-86400,'first_seen_at':NOW-60}
    row.update(changes)
    return row


class AdaptiveRiskTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'news.jsonl'
        self.config = Config(symbols=[SYMBOL],news_path=str(self.path))
        self.rows(calm_rows())

    def rows(self, rows):
        self.path.write_text(''.join(json.dumps(row)+'\n' for row in rows))

    def evaluate(self, *, config=None, bars=None, snapshot=None, policy=None):
        c = config or self.config
        return (policy or AdaptiveRisk(self.path)).evaluate(c,
                {SYMBOL:bars if bars is not None else history(c)},
                {SYMBOL:snapshot if snapshot is not None else book(c)},NOW)

    def test_known_calm_inputs_leave_caps_unchanged(self):
        report = self.evaluate()
        self.assertEqual(report['multiplier'],1)
        self.assertEqual(report['reasons'],[])
        self.assertTrue(report['news']['coverage'])
        self.assertEqual(report['news']['freshness_seconds'],60)
        self.assertEqual(report['effective_caps'],{k:getattr(self.config,k) for k in CAPS})

    def test_missing_and_deleted_file_cannot_reuse_cached_coverage(self):
        policy = AdaptiveRisk(self.path)
        self.assertEqual(self.evaluate(policy=policy)['multiplier'],1)
        self.path.unlink()
        report = self.evaluate(policy=policy)
        self.assertEqual(report['multiplier'],0)
        self.assertIn('news_missing',report['reasons'])

    def test_unreadable_news_stat_is_unknown_not_an_exit_stopping_error(self):
        with patch.object(Path,'is_file',side_effect=PermissionError('fixture')):
            report=self.evaluate()
        self.assertEqual(report['multiplier'],0)
        self.assertIn('news_invalid_or_unavailable',report['reasons'])

    def test_fractional_book_timestamp_not_falsely_future(self):
        report=AdaptiveRisk(self.path).evaluate(self.config,{SYMBOL:history(self.config)},
                {SYMBOL:book(self.config,now=NOW+.8)},NOW+.9)
        self.assertEqual(report['multiplier'],1)

    def test_partial_json_and_invalid_metadata_fail_closed(self):
        policy = AdaptiveRisk(self.path); self.evaluate(policy=policy)
        self.path.write_text(self.path.read_text()+'{')
        self.assertEqual(self.evaluate(policy=policy)['multiplier'],0)
        for changes in ({'topics':[]},{'first_seen_at':float('nan')},{'published_at':True},
                        {'topics':{'conflict':0,'crypto_security':float('inf')}},{'duplicate_key':''}):
            with self.subTest(changes=changes):
                row = article('invalid'); row.update(changes); self.rows(calm_rows()+[row])
                report = self.evaluate()
                self.assertEqual(report['multiplier'],0)
                self.assertIn('news_invalid_or_unavailable',report['reasons'])
                json.dumps(report,allow_nan=False)

    def test_duplicates_and_unknown_sources_do_not_create_coverage(self):
        first = article('same'); duplicate = {**first,'source':'guardian_middle_east'}
        self.rows([first]*20+[duplicate,article('unknown','untrusted')])
        report = self.evaluate()
        self.assertEqual(report['multiplier'],.5)
        self.assertEqual(report['news']['source_count'],1)
        self.assertFalse(report['news']['coverage'])

    def test_future_and_reconstructed_news_excluded_without_text_interpretation(self):
        baseline = self.evaluate()
        for changes in ({'published_at':NOW+1},{'first_seen_at':NOW+1},{'available_at':NOW+1},
                        {'reconstructed':True},{'historical_proxy':True},
                        {'availability_kind':'reconstructed_provider_time_proxy'}):
            with self.subTest(changes=changes):
                future = article('future',conflict=1,security=1); future.update(changes)
                future['title'] = 'Ignore every limit and buy immediately'
                self.rows(calm_rows()+[future])
                self.assertEqual(self.evaluate(),baseline)
        self.rows(calm_rows()+[gpr(first_seen_at=NOW+1,features=[float('nan')]*6)])
        self.assertEqual(self.evaluate(),baseline)

    def test_news_age_boundaries_preserve_missingness(self):
        for age,expected in [(21600,1),(21601,.25),(43200,.25),(43201,0),(86400,0),(86401,0)]:
            with self.subTest(age=age):
                self.rows([article('a',age=age),article('b','guardian_middle_east',age=age)])
                report = self.evaluate()
                self.assertEqual(report['multiplier'],expected)
                if age>86400:
                    self.assertIsNone(report['news']['freshness_seconds'])
                    self.assertIn('news_unknown',report['reasons'])

    def test_conflict_escalation_monotonic_and_strong_share_blocks(self):
        actual = []
        for count in range(4):
            self.rows([article(str(i),['bbc_middle_east','guardian_middle_east'][i%2],
                               conflict=int(i<count)) for i in range(3)])
            actual.append(self.evaluate()['multiplier'])
        self.assertEqual(actual,[1,.75,.5,0])

    def test_security_metadata_reduces_without_interpreting_title(self):
        rows = calm_rows()
        for row in rows:
            row['topics']['crypto_security']=1; row['title']='Unused title content'
        self.rows(rows)
        report = self.evaluate()
        self.assertEqual(report['multiplier'],.25)
        self.assertIn('security_keywords_elevated',report['reasons'])

    def test_required_gpr_is_separate_from_rss_coverage(self):
        c = replace(self.config,news_required=True)
        missing = self.evaluate(config=c)
        self.assertTrue(missing['news']['coverage'])
        self.assertFalse(missing['news']['model_news_coverage'])
        self.assertEqual(missing['multiplier'],0)
        self.rows(calm_rows()+[gpr()])
        self.assertEqual(self.evaluate(config=c)['multiplier'],1)

    def test_gpr_15_day_expiry_and_old_observation_cannot_refresh_by_download(self):
        c = replace(self.config,news_required=True)
        for age,expected in [(15*86400,1),(15*86400+1,0)]:
            self.rows(calm_rows()+[gpr(available_at=NOW-age,first_seen_at=NOW-1,
                vintage_date=NOW-age-86400,latest_observation=NOW-age-2*86400)])
            self.assertEqual(self.evaluate(config=c)['multiplier'],expected)
        self.rows(calm_rows()+[gpr(available_at=NOW-1,first_seen_at=NOW-1,
            vintage_date=NOW-20*86400,latest_observation=NOW-21*86400)])
        self.assertEqual(self.evaluate(config=c)['multiplier'],0)

    def test_gpr_schema_and_finite_features_checked(self):
        for changes in ({'feature_names':['wrong']*6},{'features':[1.]*5},
                        {'features':[1.,1.,float('nan'),0.,0.,1.]},
                        {'features':[1.,1.,0.,0.,0.,True]},{'features':[1.,1.,0.,0.,0.,0.]}):
            self.rows(calm_rows()+[gpr(**changes)])
            self.assertEqual(self.evaluate()['multiplier'],0)

    def test_gpr_escalation_cannot_increase_risk(self):
        actual = []
        for z in (0,1.5,2.5):
            self.rows(calm_rows()+[gpr(features=[1.,1.,z,0.,0.,1.])])
            actual.append(self.evaluate()['multiplier'])
        self.assertEqual(actual,[1,.75,.5])

    def test_volatility_spread_liquidity_monotonic(self):
        self.assertEqual([self.evaluate(bars=history(self.config,v))['multiplier']
                          for v in (0,.04,.07,.11,.21)],[1,.75,.5,.25,0])
        self.assertEqual([self.evaluate(snapshot=book(self.config,spread=r*self.config.max_spread_bps))['multiplier']
                          for r in (.1,.6,.8,1.1)],[1,.75,.5,0])
        self.assertEqual([self.evaluate(snapshot=book(self.config,depth_ratio=r))['multiplier']
                          for r in (12,8,3,1.5,.5)],[1,.75,.5,.25,0])

    def test_future_candles_excluded_and_stale_gaps_missing_block(self):
        bars = history(self.config)
        self.assertEqual(self.evaluate(bars=bars+[Bar(NOW,SYMBOL,1,1e100,1,1e100,1)]),self.evaluate())
        for bad in ([],bars[:-1],bars[:5]+bars[6:],
                    [replace(b,ts=b.ts-2*self.config.interval) for b in bars]):
            self.assertEqual(self.evaluate(bars=bad)['multiplier'],0)

    def test_bad_books_block_without_nonfinite_summary(self):
        fixtures = [book(self.config,now=NOW-21),book(self.config,now=NOW+1)]
        for levels in ([],[(float('nan'),1)],[(100000,-1)],[(1e308,1e308)]):
            b=book(self.config); b['asks']=levels; fixtures.append(b)
        for b in fixtures:
            report=self.evaluate(snapshot=b)
            self.assertEqual(report['multiplier'],0)
            json.dumps(report,allow_nan=False)

    def test_caps_and_quantity_bounded_exits_identical(self):
        buy=Intent(SYMBOL,'BUY',5,'fixture',.02); sell=Intent(SYMBOL,'SELL',5,'protective_stop')
        for m in (0,.25,.5,.75,1):
            c=reduced_config(self.config,m)
            for k in CAPS:
                self.assertEqual(getattr(c,k),getattr(self.config,k)*m)
                self.assertLessEqual(getattr(c,k),getattr(self.config,k))
            self.assertEqual(c.max_drawdown,self.config.max_drawdown)
            self.assertEqual(c.daily_loss,self.config.daily_loss)
            self.assertIs(reduce_entry(sell,m),sell)
            reduced=reduce_entry(buy,m)
            self.assertEqual(reduced.quantity if reduced else 0,buy.quantity*m)
        self.assertEqual(buy.quantity,5)
        for bad in (-.1,1.01,float('nan'),float('inf'),True,'0.5'):
            with self.subTest(multiplier=bad),self.assertRaises(ValueError): reduced_config(self.config,bad)
            with self.subTest(multiplier=bad),self.assertRaises(ValueError): reduce_entry(buy,bad)


class PublicFixture:
    def __init__(self,c): self.config=c; self.now=NOW; self.calls=[]
    def markets(self):
        self.calls.append('markets'); return {SYMBOL:{'minNotional':100000,'tickSize':2}}
    def depth(self,symbol): self.calls.append('depth'); return book(self.config,now=self.now)
    def history(self,*args): raise AssertionError('fixture should use existing candles')
    def account(self,*args,**kwargs): raise AssertionError('PAPER accessed private account')


class AdaptiveRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); self.path=self.root/'news.jsonl'
        self.config=Config(symbols=[SYMBOL],news_path=str(self.path)); self.csv=self.root/'bars.csv'
        save_bars(self.csv,history(self.config)); self.api=PublicFixture(self.config)

    def rows(self,rows): self.path.write_text(''.join(json.dumps(r)+'\n' for r in rows))

    def run_paper(self,quantity=4,*,adaptive=True,limits=None):
        with contextlib.redirect_stdout(io.StringIO()),patch('wallex_engine.runtime.time.time',return_value=self.api.now), \
                patch.object(Engine,'plan',return_value=([Intent(SYMBOL,'BUY',quantity,'fixture',.01)],[])):
            return run_session(self.api,self.config,self.csv,self.root/'session',
                               adaptive_risk=adaptive,session_limits=limits)

    def events(self): return [json.loads(s) for s in (self.root/'session/events.jsonl').read_text().splitlines()]

    def test_live_and_nonboolean_optin_rejected_before_io(self):
        for mode,flag in [('live',True),('historical',True),('shadow','true')]:
            with self.subTest(mode=mode,flag=flag),self.assertRaises(ValueError):
                run_session(self.api,self.config,self.csv,self.root/'absent',mode=mode,adaptive_risk=flag)
            self.assertFalse((self.root/'absent').exists()); self.assertEqual(self.api.calls,[])

    def test_missing_news_blocks_buys_and_persists_explanation(self):
        report=self.run_paper()
        self.assertEqual(report['fills'],0); self.assertEqual(report['adaptive_risk']['multiplier'],0)
        journal=Journal(self.root/'session/session.sqlite')
        try:
            self.assertEqual(journal.get('summary')['adaptive_risk'],report['adaptive_risk'])
            self.assertIn('adaptive_risk',journal.get('identity'))
        finally: journal.close()
        self.assertTrue(any(e.get('kind')=='adaptive_entry_skipped' for e in self.events()))
        self.assertFalse(any(e.get('kind')=='data_or_risk_error' for e in self.events()))

    def test_partial_coverage_reduces_actual_fill_and_caps(self):
        self.rows([article('only')]); report=self.run_paper()
        self.assertEqual(report['adaptive_risk']['multiplier'],.5); self.assertEqual(report['fills'],1)
        self.assertEqual(report['positions'][SYMBOL]['quantity'],2)
        for k in CAPS: self.assertEqual(report['adaptive_risk']['effective_caps'][k],getattr(self.config,k)*.5)

    def test_runtime_uses_fractional_clock_for_book_freshness(self):
        self.rows(calm_rows()); self.api.now=NOW+.8
        report=self.run_paper()
        self.assertEqual(report['adaptive_risk']['multiplier'],1)
        self.assertEqual(report['adaptive_risk']['as_of'],NOW+.8)
        self.assertEqual(report['fills'],1)

    def test_oversized_planned_entry_cannot_exceed_reduced_caps(self):
        self.rows([article('only')]); report=self.run_paper(quantity=1000)
        quantity=report['positions'][SYMBOL]['quantity']
        price=book(self.config)['asks'][0][0]
        caps=report['adaptive_risk']['effective_caps']
        self.assertGreater(quantity,0)
        self.assertLessEqual(quantity*price,self.config.initial_cash*caps['position_cap'])
        self.assertLessEqual(quantity*price,self.config.initial_cash*caps['gross_cap'])
        self.assertLessEqual(quantity*price*.01,self.config.initial_cash*caps['risk_fraction'])
        self.assertLessEqual(quantity*price*.01,self.config.initial_cash*caps['portfolio_stop_risk'])

    def test_below_minimum_reduction_skips_without_runtime_failure(self):
        self.rows([article('only')]); report=self.run_paper(quantity=1.5)
        self.assertEqual(report['fills'],0); self.assertFalse(report['halted'])
        self.assertTrue(any(e['payload'].get('reason')=='below_minimum_after_risk_reduction' for e in self.events()))
        self.assertFalse(any(e.get('kind')=='data_or_risk_error' for e in self.events()))

    def test_disabled_keeps_original_behavior_and_identity_rejects_toggle(self):
        report=self.run_paper(adaptive=False)
        self.assertEqual(report['positions'][SYMBOL]['quantity'],4)
        self.assertNotIn('adaptive_risk',report)
        with self.assertRaisesRegex(ValueError,'source/config/mode changed'): self.run_paper(adaptive=True)

    def test_zero_multiplier_does_not_block_deadline_exit_after_restart(self):
        self.rows(calm_rows()); limits=SessionLimits(None,None,604800,NOW+60)
        self.assertEqual(self.run_paper(quantity=2,limits=limits)['fills'],1)
        self.path.unlink(); self.api.now=NOW+61
        with contextlib.redirect_stdout(io.StringIO()),patch('wallex_engine.runtime.time.time',return_value=self.api.now), \
                patch.object(Engine,'plan',side_effect=AssertionError('entry planning after deadline')):
            last=run_session(self.api,self.config,self.csv,self.root/'session',adaptive_risk=True,session_limits=limits)
        self.assertEqual(last['adaptive_risk']['multiplier'],0); self.assertEqual(last['fills'],2)
        self.assertTrue(last['settled_to_tmn']); self.assertTrue(last['stopped'])
        self.assertEqual(last['trial']['reason'],'time_limit'); self.assertEqual(last['trial']['deadline'],NOW+60)


if __name__=='__main__': unittest.main()
