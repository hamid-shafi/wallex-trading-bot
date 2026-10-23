"""Offline upgrade fixtures; no network, activation or real account operations."""
from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from wallex_engine.audit import Audit, atomic_json, source_hash
from wallex_engine.engine import Engine, Intent
from wallex_engine.execution import Journal
from wallex_engine import operator_maintenance as upgrade
from wallex_engine.operator_trial import RUN_DIR, deployment_identity, operator_configuration
from wallex_engine.runtime import RunLock, dump_engine
from wallex_engine.session_limits import load_operator_limits
from wallex_engine.trained import FEATURE_NAMES


class FeeMaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / RUN_DIR
        self.seed = Path(self.temp.name) / 'seed'
        (self.root / 'session').mkdir(parents=True)
        self.seed.mkdir()
        self.env = patch.object(upgrade.os, 'environ', {})
        self.env.start()
        self.addCleanup(self.env.stop)
        atomic_json(self.seed / 'config.json', {})
        atomic_json(self.seed / 'model.json', {'offline_fixture': True, 'feature_names': FEATURE_NAMES,
                                              'horizon': 12})
        (self.root / 'model.json').write_bytes((self.seed / 'model.json').read_bytes())
        atomic_json(self.seed / 'session-plan.json', {
            'floor_tmn': None, 'ceiling_tmn': None,
            'ends_at': datetime.fromtimestamp(4600, timezone.utc).isoformat(),
        })
        limits = load_operator_limits(self.seed, {})
        self.deployment = {**deployment_identity(self.seed, limits), 'source': upgrade.PREVIOUS_RELEASE}
        atomic_json(self.root / 'deployment.json', self.deployment)
        identity = {'source': upgrade.PREVIOUS_RELEASE,
                    'config': asdict(operator_configuration(self.root, self.seed)),
                    'model_sha256': self.deployment['model'], 'mode': 'live',
                    'trial_limits': asdict(limits)}
        self.path = self.root / 'session/session.sqlite'
        with closing(Journal(self.path)) as journal:
            journal.put('identity', identity)
            journal.put('trial', {'limits': asdict(limits), 'started_at': 100,
                                 'deadline': limits.ends_at, 'reason': 'drawdown_halt'})
            engine = dump_engine(Engine(operator_configuration(self.root, self.seed)))
            journal.put('engine', {**engine, 'halted': True, 'daily_halted': True, 'peak': 5100000})
            journal.put('portfolio', {'cash': 4899650, 'fees': 350, 'positions': {'fixture': True}})
            journal.put('last_decision', 1800)
            journal.reserve('completed-fixture', Intent('BTCTMN', 'BUY', .1, 'fixture'), {})
            journal.finish('completed-fixture', {'quantity': .1, 'price': 1000000, 'fee': 350, 'ts': 101})
        Audit(self.root / 'session/events.jsonl').write('fixture', 101)

    def state(self):
        with closing(sqlite3.connect(self.path)) as db:
            return {k: json.loads(v) for k, v in db.execute('select key,value from state')}

    def orders(self, path=None):
        with closing(sqlite3.connect(path or self.path)) as db:
            return db.execute('select * from orders order by id').fetchall()

    def apply(self):
        return upgrade.apply_maintenance(self.root, self.seed, source_hash(), now=2000)

    def test_preserves_fills_cash_fees_deadline_and_all_stop_latches(self):
        kill = self.root / 'session/KILL'
        kill.touch()
        before, orders = self.state(), self.orders()
        result = self.apply()
        after = self.state()
        self.assertEqual(after['identity'], {**before['identity'], 'source': source_hash()})
        for key in before.keys() - {'identity'}:
            self.assertEqual(after[key], before[key])
        self.assertEqual(self.orders(), orders)
        self.assertEqual(self.orders(result['backup']), orders)
        self.assertTrue(kill.exists())
        self.assertEqual(result['deadline'], 4600)
        self.assertEqual(result['stop_reason'], 'drawdown_halt')
        self.assertEqual(result['exchange_requests'], 0)
        self.assertFalse(result['engine_started'])
        self.assertEqual(Audit.verify(self.root / 'session/events.jsonl')[0], 2)
        deployment = json.loads((self.root / 'deployment.json').read_text())
        self.assertEqual(deployment, {**self.deployment, 'source': source_hash()})

    def test_repeated_upgrade_does_not_modify_ledger(self):
        self.apply()
        before = self.state()
        with self.assertRaises(ValueError):
            self.apply()
        self.assertEqual(self.state(), before)

    def test_pending_orders_block_upgrade_without_changing_state(self):
        with closing(Journal(self.path)) as journal:
            journal.reserve('pending', Intent('BTCTMN', 'BUY', .1, 'fixture'), {})
        before = self.state()
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            self.apply()
        self.assertEqual(self.state(), before)
        self.assertFalse(list(self.root.glob('fee-backup-*.sqlite')))

    def test_changed_limits_model_or_unknown_release_block_upgrade(self):
        for field, value in (('source', 'unknown'), ('limits', {}), ('model', 'changed')):
            with self.subTest(field=field):
                atomic_json(self.root / 'deployment.json', {**self.deployment, field: value})
                before = self.state()
                with self.assertRaises(ValueError):
                    self.apply()
                self.assertEqual(self.state(), before)
        atomic_json(self.root / 'deployment.json', self.deployment)
        (self.root / 'model.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'persisted model'):
            self.apply()

    def test_wrong_target_release_blocks_upgrade(self):
        before = self.state()
        with self.assertRaisesRegex(ValueError, 'exact installed release'):
            upgrade.apply_maintenance(self.root, self.seed, 'incorrect', now=2000)
        self.assertEqual(self.state(), before)

    def test_running_worker_or_engine_blocks_upgrade(self):
        for name in ('worker.lock', 'session/engine.lock'):
            with self.subTest(lock=name):
                lock = RunLock(self.root / name)
                try:
                    before = self.state()
                    with self.assertRaisesRegex(RuntimeError, 'another engine'):
                        self.apply()
                    self.assertEqual(self.state(), before)
                finally:
                    lock.close()

    def test_corrupt_audit_blocks_upgrade(self):
        with (self.root / 'session/events.jsonl').open('a') as stream:
            stream.write('{"broken": true}\n')
        before = self.state()
        with self.assertRaises((ValueError, KeyError)):
            self.apply()
        self.assertEqual(self.state(), before)
        self.assertFalse(list(self.root.glob('fee-backup-*.sqlite')))

    def test_missing_or_incomplete_engine_cannot_lose_risk_state(self):
        for saved in (None, {}, {'halted': False}):
            with self.subTest(saved=saved):
                with closing(Journal(self.path)) as journal:
                    journal.put('engine', saved)
                before = self.state()
                with self.assertRaisesRegex(ValueError, 'engine/risk state'):
                    self.apply()
                self.assertEqual(self.state(), before)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("delete from state where key='engine'")
        with self.assertRaisesRegex(ValueError, 'engine/risk state'):
            self.apply()

    def test_unknown_and_null_order_statuses_require_reconciliation(self):
        for status in (None, 'UNKNOWN'):
            with self.subTest(status=status):
                with closing(sqlite3.connect(self.path)) as db, db:
                    db.execute('update orders set status=?', (status,))
                before = self.state()
                with self.assertRaisesRegex(ValueError, 'ambiguous'):
                    self.apply()
                self.assertEqual(self.state(), before)

    def test_missing_audit_cannot_start_a_new_chain(self):
        (self.root / 'session/events.jsonl').unlink()
        before = self.state()
        with self.assertRaisesRegex(ValueError, 'existing audit'):
            self.apply()
        self.assertEqual(self.state(), before)


if __name__ == '__main__':
    unittest.main()
