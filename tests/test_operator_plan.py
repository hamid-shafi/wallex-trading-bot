"""Migration tests use synthetic local journals, never an account or bot runner."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from wallex_engine.audit import Audit, atomic_json, source_hash
from wallex_engine.data import digest_file
from wallex_engine.engine import Intent
from wallex_engine.execution import Journal
from wallex_engine import operator_plan
from wallex_engine.operator_trial import RUN_DIR, operator_configuration
from wallex_engine.session_limits import load_operator_limits


class OperatorPlanTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        self.root = self.base / RUN_DIR
        self.seed = self.base / 'seed'
        self.seed.mkdir()
        (self.root / 'session').mkdir(parents=True)
        self.now = 1_000
        self.end = 4_600
        atomic_json(self.seed / 'config.json', {})
        atomic_json(self.seed / 'model.json', {'offline_fixture': True})
        (self.root / 'model.json').write_bytes((self.seed / 'model.json').read_bytes())
        atomic_json(self.seed / 'session-plan.json', {
            'floor_tmn': None, 'ceiling_tmn': None,
            'ends_at': datetime.fromtimestamp(self.end, timezone.utc).isoformat(),
        })
        self.old_limits = {'floor_tmn': 4_000_000, 'ceiling_tmn': 6_000_000, 'seconds': 1800}
        self.old_trial = {'limits': self.old_limits, 'started_at': 100,
                          'deadline': 1900, 'reason': 'time_limit',
                          'triggered_at': 1900, 'trigger_equity_tmn': 4_990_000}
        # Migration time is after the old timeout and before the new absolute deadline.
        self.now = 2_000
        config = operator_configuration(self.root, self.seed)
        self.old_identity = {
            'source': operator_plan.PREVIOUS_RELEASE, 'config': asdict(config),
            'mode': 'live', 'model_sha256': digest_file(self.seed / 'model.json'),
            'trial_limits': self.old_limits,
        }
        self.old_deployment = {
            'source': operator_plan.PREVIOUS_RELEASE,
            'model': digest_file(self.seed / 'model.json'),
            'seed_config': digest_file(self.seed / 'config.json'),
            'seconds': 1800, 'account_scope': 'main_spot',
        }
        atomic_json(self.root / 'deployment.json', self.old_deployment)
        self.db_path = self.root / 'session/session.sqlite'
        with self.journal() as journal:
            journal.put('trial', self.old_trial)
            journal.put('identity', self.old_identity)
            journal.put('engine', {'halted': False, 'daily_halted': True, 'peak': 5_010_000})
            journal.put('summary', {'halted': False, 'manual_reconciliation_required': True})
            journal.put('portfolio', {'offline_preservation_marker': 'inventory-and-fees'})
            intent = Intent('BTCTMN', 'BUY', .1, 'fixture', .06, 101)
            journal.reserve('fixture-order', intent, {'offline': True})
            journal.finish('fixture-order', {'quantity': .1, 'price': 1_000_000,
                                              'fee': 350, 'ts': 101})
        self.audit = Audit(self.root / 'session/events.jsonl')
        self.audit.write('offline_fixture', 100)
        self.env_patch = patch.object(operator_plan.os, 'environ', {})
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

    def journal(self):
        from contextlib import closing
        return closing(Journal(self.db_path))

    def state(self):
        with sqlite3.connect(self.db_path) as db:
            return {k: json.loads(v) for k, v in db.execute('select key,value from state')}

    def orders(self, path=None):
        with sqlite3.connect(path or self.db_path) as db:
            return db.execute('select * from orders order by id').fetchall()

    def apply(self):
        return operator_plan.apply_plan(self.root, self.seed, source_hash(), now=self.now)

    def test_offline_renewal_preserves_fills_inventory_and_risk_state(self):
        old = self.state()
        old_orders = self.orders()
        result = self.apply()
        new = self.state()
        self.assertEqual(new['engine'], old['engine'])
        self.assertTrue(new['engine']['daily_halted'])
        self.assertEqual(new['portfolio'], old['portfolio'])
        self.assertEqual(self.orders(), old_orders)
        self.assertEqual(self.orders(result['backup']), old_orders)
        self.assertEqual(new['trial']['started_at'], old['trial']['started_at'])
        self.assertEqual(new['trial']['deadline'], self.end)
        self.assertIsNone(new['trial']['reason'])
        self.assertNotIn('triggered_at', new['trial'])
        self.assertEqual(new['plan_changes'][0]['previous_trial'], self.old_trial)
        self.assertEqual(new['plan_changes'][0]['previous_identity'], self.old_identity)
        self.assertEqual(result['exchange_requests'], 0)
        self.assertFalse(result['engine_started'])
        self.assertEqual(Audit.verify(self.root / 'session/events.jsonl')[0], 2)
        limits = load_operator_limits(self.seed, {})
        with self.journal() as journal:
            self.assertIsNone(limits.check(journal, 4_990_000, self.end - 1)['reason'])
            self.assertEqual(limits.check(journal, 4_990_000, self.end)['reason'], 'time_limit')

    def test_non_time_stop_is_never_cleared(self):
        for reason in ('operator_stop', 'capital_floor', 'capital_ceiling', 'drawdown_halt', None):
            with self.subTest(reason=reason):
                with self.journal() as journal:
                    journal.put('trial', {**self.old_trial, 'reason': reason})
                before = self.state()
                with self.assertRaises(ValueError):
                    self.apply()
                self.assertEqual(self.state(), before)

    def test_drawdown_latch_blocks_renewal_even_if_time_was_first_reason(self):
        with self.journal() as journal:
            journal.put('engine', {'halted': True})
        before = self.state()
        with self.assertRaisesRegex(ValueError, 'drawdown'):
            self.apply()
        self.assertEqual(self.state(), before)

    def test_ambiguous_order_blocks_renewal(self):
        with self.journal() as journal:
            journal.reserve('pending-fixture', Intent('BTCTMN', 'BUY', .1, 'fixture'), {})
        before = self.state()
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            self.apply()
        self.assertEqual(self.state(), before)

    def test_kill_file_blocks_renewal(self):
        (self.root / 'session/KILL').touch()
        with self.assertRaisesRegex(ValueError, 'operator stop'):
            self.apply()
        self.assertEqual(self.state()['trial'], self.old_trial)

    def test_wrong_release_is_rejected_before_ledger_change(self):
        with self.assertRaisesRegex(ValueError, 'exact installed release'):
            operator_plan.apply_plan(self.root, self.seed, 'unaccepted-release', now=self.now)
        self.assertEqual(self.state()['trial'], self.old_trial)

    def test_changed_model_is_rejected(self):
        atomic_json(self.root / 'model.json', {'different': True})
        with self.assertRaisesRegex(ValueError, 'persisted model'):
            self.apply()
        self.assertEqual(self.state()['trial'], self.old_trial)


if __name__ == '__main__':
    unittest.main()
