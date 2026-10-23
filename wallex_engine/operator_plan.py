"""Human-run, offline plan migration. Never starts an engine or calls the exchange."""
import argparse
from contextlib import closing
from dataclasses import asdict
import json
import os
from pathlib import Path
import sqlite3
import time
from .audit import Audit,atomic_json,canonical,source_hash
from .data import digest_file
from .operator_trial import RUN_DIR,operator_configuration,deployment_identity
from .runtime import RunLock
from .session_limits import load_operator_limits

PREVIOUS_RELEASE='5ecb196e6cea26ef6ccd1b6c78797f588e188fa46fd6f47f69ca4c65f2504ef4'


def apply_plan(root,seed,accepted_release,now=None):
    if accepted_release!=source_hash():raise ValueError('accept the exact installed release')
    root,seed=Path(root),Path(seed);now=int(time.time() if now is None else now)
    if root.name!=RUN_DIR:raise ValueError('unrecognized run directory')
    path=root/'session/session.sqlite'
    if not path.is_file():raise ValueError('existing ledger required; no reset allowed')
    if (root/'session/KILL').exists():raise ValueError('operator stop must be reconciled separately')
    limits=load_operator_limits(seed,os.environ).validate(5e6)
    if limits.ends_at is None or not now<limits.ends_at<=now+604800:raise ValueError('new absolute deadline required')
    # Both locks must be free: even a stopped worker can still own its process lock.
    worker=RunLock(root/'worker.lock')
    try:
        engine=RunLock(root/'session/engine.lock')
        try:
            old_deployment=json.loads((root/'deployment.json').read_text())
            if old_deployment.get('source')!=PREVIOUS_RELEASE:raise ValueError('only the reviewed previous release can migrate')
            expected=deployment_identity(seed,limits)
            if any(old_deployment.get(k)!=expected[k] for k in ('model','seed_config','account_scope')):
                raise ValueError('model, strategy or account changed; separate reconciliation required')
            if digest_file(root/'model.json')!=expected['model']:raise ValueError('persisted model changed')
            db=sqlite3.connect(path)
            try:
                state={k:json.loads(v) for k,v in db.execute('select key,value from state')}
                trial,identity,saved=state['trial'],state['identity'],state['engine']
                c=operator_configuration(root,seed)
                if (identity.get('source')!=PREVIOUS_RELEASE or identity.get('config')!=asdict(c) or
                        identity.get('mode')!='live' or identity.get('model_sha256')!=expected['model']):
                    raise ValueError('ledger identity does not match the reviewed strategy')
                if trial.get('reason')!='time_limit':raise ValueError('only a completed timed session may be renewed')
                if saved.get('halted') or state.get('summary',{}).get('halted'):
                    raise ValueError('drawdown stop cannot be cleared by a plan change')
                if db.execute("select count(*) from orders where status IS NOT 'DONE'").fetchone()[0]:
                    raise ValueError('ambiguous orders require reconciliation first')
                audit=Audit(root/'session/events.jsonl')
                backup=root/f'plan-backup-{now}.sqlite'
                if backup.exists():raise ValueError('backup already exists; inspect prior attempt')
                with closing(sqlite3.connect(backup)) as target:db.backup(target)
                atomic_json(root/f'deployment-backup-{now}.json',old_deployment)
                change={'at':now,'previous_trial':trial,'previous_identity':identity,'new_limits':asdict(limits)}
                history=state.get('plan_changes',[])+[change]
                new_trial={**trial,'limits':asdict(limits),'deadline':limits.ends_at,'reason':None,
                           'plan_changed_at':now}
                for key in ('triggered_at','trigger_equity_tmn'):new_trial.pop(key,None)
                new_identity={**identity,'source':source_hash(),'trial_limits':asdict(limits)}
                with db:
                    for key,value in (('trial',new_trial),('identity',new_identity),('plan_changes',history)):
                        db.execute('update state set value=? where key=?',(canonical(value),key)) if key in state else db.execute('insert into state values (?,?)',(key,canonical(value)))
                atomic_json(root/'deployment.json',expected)
                audit.write('operator_plan_change',now,previous_trial=trial,new_limits=asdict(limits))
                return {'plan_prepared':True,'deadline':limits.ends_at,'floor_tmn':limits.floor_tmn,
                        'ceiling_tmn':limits.ceiling_tmn,'exchange_requests':0,'engine_started':False,
                        'backup':str(backup)}
            finally:db.close()
        finally:engine.close()
    finally:worker.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true',required=True)
    parser.add_argument('--accept-release',required=True)
    args=parser.parse_args()
    if os.getenv('OPERATOR_START_RELEASE'):raise ValueError('disable engine start and restart observer before migration')
    print(json.dumps(apply_plan(Path(os.getenv('DATA_DIR','/data'))/RUN_DIR,
                               Path(os.getenv('SEED_DIR','/app/deploy/seed')),args.accept_release)))


if __name__=='__main__':main()
