"""Offline fee-release upgrade. Preserve every fill, balance, stop and deadline.

This command cannot start an engine or contact the exchange. It only accepts
the reviewed ad64061 release and requires both engine locks to be free.
"""
import argparse
from contextlib import closing
from dataclasses import asdict
import json
import os
from pathlib import Path
import sqlite3
import time

from .audit import Audit, atomic_json, canonical, source_hash
from .data import digest_file
from .operator_trial import RUN_DIR, deployment_identity, operator_configuration
from .runtime import RunLock, restore_engine
from .session_limits import load_operator_limits


PREVIOUS_RELEASE = 'ad64061ff4fa4138c9b9e7a9ecfab9240055ae3f93b72d865de7a7b7fb03a0a0'


def apply_maintenance(root, seed, accepted_release, now=None):
    target_release = source_hash()
    if accepted_release != target_release:
        raise ValueError('accept the exact installed release')
    root, seed = Path(root), Path(seed)
    now = int(time.time() if now is None else now)
    path = root / 'session/session.sqlite'
    if root.name != RUN_DIR or not path.is_file():
        raise ValueError('existing operator ledger required; no reset allowed')
    limits = load_operator_limits(seed, os.environ).validate(5e6)
    worker = RunLock(root / 'worker.lock')
    try:
        engine = RunLock(root / 'session/engine.lock')
        try:
            old_deployment = json.loads((root / 'deployment.json').read_text())
            expected = deployment_identity(seed, limits)
            if old_deployment != {**expected, 'source': PREVIOUS_RELEASE}:
                raise ValueError('only the reviewed release with unchanged model/config/limits can upgrade')
            if digest_file(root / 'model.json') != expected['model']:
                raise ValueError('persisted model changed')
            with closing(sqlite3.connect(path)) as db:
                state = {k: json.loads(v) for k, v in db.execute('select key,value from state')}
                identity = state.get('identity', {})
                config = operator_configuration(root, seed)
                if (identity.get('source') != PREVIOUS_RELEASE or
                        identity.get('config') != asdict(config) or identity.get('mode') != 'live' or
                        identity.get('model_sha256') != expected['model'] or
                        identity.get('trial_limits') != asdict(limits)):
                    raise ValueError('ledger identity does not match reviewed release')
                if (state.get('trial', {}).get('limits') != asdict(limits) or
                        state['trial'].get('deadline') != limits.ends_at):
                    raise ValueError('existing deadline/limits must remain unchanged')
                saved = state.get('engine')
                required = {'learners', 'forecasts', 'peak', 'day', 'day_equity', 'halted', 'daily_halted'}
                if (not isinstance(saved, dict) or not required <= saved.keys() or
                        not isinstance(saved['learners'], dict) or set(saved['learners']) != set(config.symbols) or
                        not isinstance(saved['halted'], bool) or not isinstance(saved['daily_halted'], bool)):
                    raise ValueError('complete persisted engine/risk state required')
                try:
                    restore_engine(config, saved)
                except (KeyError, TypeError, ValueError, AttributeError) as exc:
                    raise ValueError('persisted engine/risk state cannot be restored') from exc
                if db.execute("select count(*) from orders where status IS NOT 'DONE'").fetchone()[0]:
                    raise ValueError('ambiguous orders require reconciliation first')
                if not (root / 'session/events.jsonl').is_file():
                    raise ValueError('existing audit chain required')
                audit = Audit(root / 'session/events.jsonl')
                backup = root / f'fee-backup-{now}.sqlite'
                deployment_backup = root / f'fee-deployment-backup-{now}.json'
                if backup.exists() or deployment_backup.exists():
                    raise ValueError('backup already exists; inspect previous attempt')
                with closing(sqlite3.connect(backup)) as destination:
                    db.backup(destination)
                backup.chmod(0o600)
                atomic_json(deployment_backup, old_deployment)
                deployment_backup.chmod(0o600)
                change = {'at': now, 'previous_identity': identity,
                          'new_source': target_release, 'backup': str(backup)}
                with db:
                    db.execute('update state set value=? where key=?',
                               (canonical({**identity, 'source': target_release}), 'identity'))
                    db.execute('insert or replace into state values (?,?)',
                               ('fee_maintenance_changes', canonical(state.get('fee_maintenance_changes', []) + [change])))
                atomic_json(root / 'deployment.json', expected)
                audit.write('operator_fee_maintenance', now, **change)
                return {'maintenance_prepared': True, 'source': target_release,
                        'deadline': state['trial']['deadline'], 'stop_reason': state['trial'].get('reason'),
                        'exchange_requests': 0, 'engine_started': False, 'backup': str(backup)}
        finally:
            engine.close()
    finally:
        worker.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', required=True)
    parser.add_argument('--accept-release', required=True)
    args = parser.parse_args()
    if os.getenv('OPERATOR_START_RELEASE') or os.getenv('OPERATOR_ARM'):
        raise ValueError('disable engine activation before offline maintenance')
    print(json.dumps(apply_maintenance(Path(os.getenv('DATA_DIR', '/data')) / RUN_DIR,
                                      Path(os.getenv('SEED_DIR', '/app/deploy/seed')),
                                      args.accept_release)))


if __name__ == '__main__':
    main()
