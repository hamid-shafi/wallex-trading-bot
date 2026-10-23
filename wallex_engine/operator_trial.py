"""Manually armed spot trial. No key or arm switch is shipped in the package."""
from dataclasses import replace,asdict
from pathlib import Path
from http.server import ThreadingHTTPServer
import json
import os
import signal
import threading
import time

from .audit import atomic_json,source_hash
from .cloud import bootstrap,news_worker
from .config import Config
from .data import digest_file
from .runtime import run_session,RunLock
from .session_limits import SessionLimits,load_operator_limits
from .wallex import Wallex,PublicWallex,APIError

ARM='I_ACCEPT_REAL_SPOT_TRADING'
RUN_DIR='operator-v3'
ROUTES={'balances':'/v1/account/balances','open_orders':'/v1/account/openOrders',
        'orders':'/v1/account/orders','order':'/v1/account/orders/{client_id}',
        'cancel':'/v1/account/orders'}


class OperatorConfig(Config):
    operator_spot=True

    def live_preflight(self,now):
        self.validate()
        if os.getenv('OPERATOR_ARM')!=ARM:
            raise ValueError('manual operator activation is required')
        if os.getenv('OPERATOR_ACCEPT_MODEL_SHA256')!=digest_file(self.model_artifact):
            raise ValueError('operator must accept the exact research model hash')
        if os.getenv('OPERATOR_ACCOUNT_READY')!='5000000_TMN_NO_OTHER_ASSETS_OR_ORDERS':
            raise ValueError('operator must confirm the five-million TMN account and no manual trades')
        if os.getenv('OPERATOR_FEE_CONFIRMED')!='MAX_35_BPS_PER_SIDE':
            raise ValueError('operator must verify account fee <=0.35 percent per side')
        if (self.initial_cash!=5_000_000 or
                self.fee_bps!=35 or self.routes!=ROUTES or not self.live_enabled):
            raise ValueError('operator trial account/budget/routes mismatch')
        artifact=json.loads(Path(self.model_artifact).read_text(encoding='utf8'))
        if artifact.get('spec')!={'family':'committee','id':'committee_matched','news':True} or artifact.get('fit') is not None:
            raise ValueError('only the previous frozen rule policy is supported')
        if not artifact.get('paper_eligible') or now<artifact['training_cutoff']:
            raise ValueError('model unavailable')
        # This path records operator acceptance; it does NOT relabel research as validated.
        if not artifact.get('research_only'):
            raise ValueError('research provenance must remain intact')
        if not all((self.fee_verified,self.allowed_markets_verified,self.account_binding_verified)):
            raise ValueError('operator account settings incomplete')


class SpotWallex(Wallex):
    """Fixed spot-account routes only; no withdrawals, transfers, credit or margin."""
    def request(self,method,path,params=None,body=None,private=False):
        public=method=='GET' and path in PublicWallex.PATHS and not private and body is None
        account=(private and method=='GET' and
                 (path in (ROUTES['balances'],ROUTES['open_orders'],'/v1/account/fee') or path.startswith('/v1/account/orders/')))
        order=private and method=='POST' and path==ROUTES['orders'] and isinstance(body,dict) and body.get('type')=='LIMIT'
        cancel=private and method=='DELETE' and path==ROUTES['cancel'] and body is None and bool((params or {}).get('clientOrderId'))
        if not (public or account or order or cancel):
            raise APIError('operator trial rejects non-spot or unsupported requests')
        return super().request(method,path,params,body,private)

    def account(self,config,name,method='GET',client_id=None,body=None):
        if config.routes!=ROUTES: raise APIError('unrecognized spot-account routing')
        if name=='cancel':
            if method!='DELETE' or not client_id or body is not None:
                raise APIError('invalid cancellation request')
            return self.request('DELETE',ROUTES['cancel'],params={'clientOrderId':client_id},private=True).get('result',{})
        return super().account(config,name,method,client_id,body)


def operator_configuration(root,seed):
    root,seed=Path(root),Path(seed)
    base=OperatorConfig.load(seed/'config.json')
    return replace(base,live_enabled=True,routes=ROUTES.copy(),allowed_markets_verified=True,
                   account_binding_verified=True,fee_verified=True,model_artifact=str(root/'model.json'),
                   news_path=str(root/'news.jsonl'),paper_run=str(root/'session'))


def deployment_identity(seed,limits):
    seed=Path(seed)
    return {'source':source_hash(),'model':digest_file(seed/'model.json'),
            'seed_config':digest_file(seed/'config.json'),'limits':asdict(limits),'account_scope':'main_spot'}


def prepare(root,seed,seconds):
    root,seed=Path(root),Path(seed)
    if root.name!=RUN_DIR: raise ValueError(f'dedicated /data/{RUN_DIR} directory required')
    root.mkdir(parents=True,exist_ok=True)
    c=operator_configuration(root,seed)
    limits=load_operator_limits(seed,dict(os.environ,OPERATOR_TRIAL_SECONDS=str(seconds))).validate(c.initial_cash)
    identity=deployment_identity(seed,limits)
    path=root/'deployment.json'
    if path.exists():
        if json.loads(path.read_text())!=identity:
            raise ValueError('deployment identity changed; do not reset ledger')
    else:
        if (root/'session').exists(): raise ValueError('unrecognized previous session')
        (root/'model.json').write_bytes((seed/'model.json').read_bytes())
        (root/'news.jsonl').write_bytes((seed/'news.jsonl').read_bytes())
        atomic_json(path,identity)
    if digest_file(root/'model.json')!=identity['model']: raise ValueError('persisted model changed')
    return c,limits


def health_handler(state):
    from http.server import BaseHTTPRequestHandler
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in ('/','/healthz','/readyz'):
                self.send_error(404);return
            ready=state.get('phase')=='running' and time.time()-state.get('heartbeat',0)<180
            body=json.dumps({'service':'wallex-operator-trial','phase':state.get('phase'),
                             'armed':state.get('armed',False),'ready':ready}).encode()
            self.send_response(200 if self.path!='/readyz' or ready else 503)
            self.send_header('Content-Type','application/json');self.send_header('Cache-Control','no-store')
            self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def log_message(self,*args): pass
    return Handler


def main():
    state={'phase':'awaiting_operator','armed':False};stop=threading.Event();lock=None
    server=ThreadingHTTPServer(('0.0.0.0',int(os.getenv('PORT','8000'))),health_handler(state))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    def shutdown(*args): stop.set();raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,shutdown)
    try:
        if os.getenv('OPERATOR_ARM')!=ARM:
            print(json.dumps({'event':'awaiting_operator','exchange_requests':0}),flush=True)
            stop.wait();return
        # No automatic restart loop: ambiguous transactions and risk failures stay stopped.
        state.update(phase='preflight',armed=True)
        # Keep the failed first ledger intact. This release uses a fresh,
        # versioned ledger and never deletes or reinterprets the old records.
        root=Path(os.getenv('DATA_DIR','/data'))/RUN_DIR
        seed=Path(os.getenv('SEED_DIR','/app/deploy/seed'))
        seconds=int(os.getenv('OPERATOR_TRIAL_SECONDS','1800'))
        root.mkdir(parents=True,exist_ok=True)
        lock=RunLock(root/'worker.lock')
        c,limits=prepare(root,seed,seconds)
        c.live_preflight(time.time())
        if not os.getenv('WALLEX_API_KEY','').strip(): raise ValueError('operator key missing')
        if not (root/'session/session.sqlite').exists():
            if limits.ends_at is not None and not time.time()<limits.ends_at<=time.time()+604800:
                raise ValueError('session deadline expired or too distant; no new session allowed')
            artifact=json.loads(Path(c.model_artifact).read_text())
            if time.time()>artifact['valid_until']: raise ValueError('model expired; no new trial allowed')
            bootstrap(PublicWallex(timeout=15,retries=2),c,seed/'warmup.csv',root/'warmup.csv')
        threading.Thread(target=news_worker,args=(root,stop,state),daemon=True).start()
        def heartbeat(summary):
            state.update(phase='running',heartbeat=time.time())
        result=run_session(SpotWallex(timeout=15,retries=2),c,root/'warmup.csv',root/'session',
                           'live',iterations=0,poll_seconds=15,on_heartbeat=heartbeat,session_limits=limits)
        state['phase']='manual_reconciliation_required' if result.get('positions') else 'stopped'
        atomic_json(root/'result.json',result)
        stop.wait()  # Stays stopped across health probes; process exit is never an auto-rearm.
    except KeyboardInterrupt: state['phase']='stopped_by_operator'
    except Exception as exc:
        state['phase']='manual_reconciliation_required'
        # Keep diagnosis private, redact configured credentials, never reset a ledger.
        try:
            message=str(exc)[:1000]
            for name,value in os.environ.items():
                if len(value)>=8 and any(k in name.upper() for k in ('KEY','TOKEN','PASSWORD','SECRET')):
                    message=message.replace(value,'[redacted]')
            path=Path(os.getenv('DATA_DIR','/data'))/RUN_DIR/'failure.json'
            if path.parent.exists():atomic_json(path,{'error_type':type(exc).__name__,'message':message,'ts':int(time.time())})
        except Exception:pass
        # Exception text may contain private exchange data; report type only in public logs.
        print(json.dumps({'event':'trial_failed','error_type':type(exc).__name__}),flush=True)
        stop.wait()
    finally:
        stop.set();server.shutdown()
        if lock: lock.close()


if __name__=='__main__': main()
