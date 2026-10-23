"""Single-instance, persistent, public-data-only shadow worker for Liara."""
from dataclasses import asdict,replace
from datetime import datetime,timezone
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import json
import os
import signal
import threading
import time

from .audit import atomic_json,source_hash
from .config import Config
from .data import Bar,aggregate,digest_file,frames,load_bars,save_bars
from .runtime import run_session,RunLock
from .wallex import PublicWallex
from .session_limits import load_operator_limits


def shadow_config(seed,root):
    c=Config.load(seed)
    if c.live_enabled or c.routes:
        raise ValueError('cloud worker only accepts shadow mode with no account routes')
    return replace(c,model_artifact=str(root/'model.json'),news_path=str(root/'news.jsonl'),
        paper_run=str(root/'session'))


def prepare(root,seed):
    root=Path(root);seed=Path(seed);root.mkdir(parents=True,exist_ok=True)
    c=shadow_config(seed/'config.json',root)
    limits=load_operator_limits(seed,{}).validate(c.initial_cash) if (seed/'session-plan.json').exists() else None
    # A deployment never silently changes model, risk budget or starting capital.
    identity={'source_sha256':source_hash(),'seed_model_sha256':digest_file(seed/'model.json'),
              'config':asdict(c)}
    if limits:identity['session_limits']=asdict(limits)
    path=root/'deployment.json'
    if path.exists():
        if json.loads(path.read_text())!=identity:
            raise ValueError('deployment changed; explicit state migration is required; cash has not been reset')
    else:
        # Validate before writing an identity that a bad deployment cannot use.
        artifact=json.loads((seed/'model.json').read_text())
        if not artifact.get('research_only') or artifact.get('fit') is not None:
            raise ValueError('this deployment is restricted to the frozen research rule policy')
        (root/'model.json').write_bytes((seed/'model.json').read_bytes())
        (root/'news.jsonl').write_bytes((seed/'news.jsonl').read_bytes())
        atomic_json(root/'deployment.json',identity)
    if digest_file(root/'model.json')!=identity['seed_model_sha256']:
        raise ValueError('persisted model hash changed; refusing to reset or trade')
    atomic_json(root/'config.json',asdict(c))
    return c


def bootstrap(api,c,seed,output,end=None,hours=721):
    end=int(time.time())//3600*3600 if end is None else end
    if end%3600 or end>int(time.time())//3600*3600:
        raise ValueError('bootstrap end must be a completed UTC hour')
    start=end-hours*3600
    by={s:{} for s in c.symbols}
    for b in load_bars(seed):
        if b.symbol in by and start<=b.ts and b.ts+3600<=end:
            by[b.symbol][b.ts]=b
    def parse(payload,symbol,left,right,interval):
        rows={}
        if payload.get('s')=='no_data':return rows
        if payload.get('s')!='ok':raise ValueError('invalid public history status')
        arrays=[payload[k] for k in ('t','o','h','l','c','v')]
        if len({len(v) for v in arrays})!=1:raise ValueError('history length mismatch')
        for ts,o,h,l,close,volume in zip(*arrays):
            ts=int(ts)
            if not left<=ts or ts+interval>right:continue
            if ts%interval:raise ValueError('unaligned historical candle')
            try:b=Bar(ts,symbol,*map(float,(o,h,l,close,volume))).validate()
            except ValueError:continue
            if ts in rows and rows[ts]!=b:raise ValueError('conflicting duplicate candle')
            rows[ts]=b
        return rows
    for symbol in c.symbols:
        missing=sorted(set(range(start,end,3600))-set(by[symbol]))
        if missing:
            for left in range(missing[0],end,20*86400):
                right=min(end,left+20*86400)
                raw=api.request('GET','/v1/udf/history',{'symbol':symbol,'resolution':'60','from':left,'to':right})
                by[symbol].update(parse(raw,symbol,left,right,3600))
        missing=sorted(set(range(start,end,3600))-set(by[symbol]))
        for ts in missing[:48]:
            raw=api.request('GET','/v1/udf/history',{'symbol':symbol,'resolution':'1','from':ts,'to':ts+3600})
            minutes=list(parse(raw,symbol,ts,ts+3600,60).values())
            rebuilt=aggregate(minutes,3600,60)
            if len(rebuilt)==1:by[symbol][ts]=rebuilt[0]
        if set(range(start,end,3600))-set(by[symbol]):
            raise ValueError(f'contiguous warmup incomplete for {symbol}; no fabricated candles')
    bars=[b for values in by.values() for b in values.values()]
    assert len(frames(bars,c.symbols,3600))==hours
    save_bars(output,bars)
    return {'hours':hours,'closed_through':end,'sha256':digest_file(output)}


def merge_news(path,inputs):
    path=Path(path)
    existing=[json.loads(s) for s in path.read_text(encoding='utf8').splitlines() if s.strip()] if path.exists() else []
    known={r['id'] for r in existing};added=0
    for source in inputs:
        if not Path(source).exists():continue
        for line in Path(source).read_text(encoding='utf8').splitlines():
            if not line.strip():continue
            row=json.loads(line)
            if row['id'] not in known:
                existing.append(row);known.add(row['id']);added+=1
    temp=path.with_suffix('.jsonl.tmp')
    temp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in existing),encoding='utf8')
    os.replace(temp,path)
    return added


def news_worker(root,stop,state):
    from .news import collect as rss_collect
    from .gpr import collect as gpr_collect
    while not stop.is_set():
        now=int(time.time());day=datetime.fromtimestamp(now,timezone.utc).strftime('%Y%m%d')
        rss=root/'news-fetch'/'rss';gpr=root/'news-fetch'/day
        errors={};sources={}
        # RSS must still refresh when the slower historical GPR source is down.
        try:
            result=rss_collect(rss)
            sources=result.get('sources',{})
            for name,item in sources.items():
                if item.get('status')!='ok':errors[name]=item.get('error','unavailable')
        except Exception as exc:
            errors['rss']=type(exc).__name__
        try:
            result=gpr_collect(gpr,now-10*86400,now)
            if result.get('errors'):errors['gpr']='partial_source_failure'
        except Exception as exc:
            errors['gpr']=type(exc).__name__
        try:
            added=merge_news(root/'news.jsonl',[gpr/'news.jsonl',rss/'news.jsonl'])
            state.update(news_updated_at=int(time.time()),news_error=errors or None,
                         news_sources=sources)
            print(json.dumps({'event':'news_refresh','added':added,'errors':errors,
                              'credentials_used':False}),flush=True)
        except Exception as exc:
            state['news_error']={'merge':type(exc).__name__,**errors}
            print(json.dumps({'event':'news_unavailable','error':type(exc).__name__,
                              'entry_policy':'causal coverage gate remains enforced'}),flush=True)
        stop.wait(3600)



def health_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in ('/','/healthz','/readyz'):
                self.send_error(404);return
            ready=state.get('phase')=='running' and time.time()-state.get('last_progress_at',0)<180
            status=200 if self.path!='/readyz' or ready else 503
            # Public endpoints never disclose portfolio, news details, files or credentials.
            body=json.dumps({'service':'wallex-shadow','mode':'shadow','live_enabled':False,
                             'phase':state.get('phase','starting'),'ready':ready}).encode()
            self.send_response(status);self.send_header('Content-Type','application/json')
            self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)))
            self.end_headers();self.wfile.write(body)
        def log_message(self,*args):pass
    return Handler


def main():
    root=Path(os.getenv('DATA_DIR','/data')).resolve()
    seed=Path(os.getenv('SEED_DIR','/app/deploy/seed')).resolve()
    port=int(os.getenv('PORT','8000'))
    if os.getenv('APP_MODE','shadow')!='shadow':
        raise ValueError('this worker has no live trading mode')
    state={'phase':'starting'};stop=threading.Event()
    server=ThreadingHTTPServer((os.getenv('WORKER_BIND_HOST','0.0.0.0'),port),health_handler(state))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    lock=None
    def shutdown(*args):
        stop.set();raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,shutdown)
    try:
        root.mkdir(parents=True,exist_ok=True)
        lock=RunLock(root/'worker.lock')
        c=prepare(root,seed)
        limits=load_operator_limits(seed,{}).validate(c.initial_cash) if (seed/'session-plan.json').exists() else None
        api=PublicWallex(timeout=15,retries=2)
        threading.Thread(target=news_worker,args=(root,stop,state),daemon=True).start()
        warmup=root/'warmup.csv'
        def heartbeat(summary):
            state.update(phase='running',last_progress_at=time.time(),summary=summary)
            atomic_json(root/'status.json',dict(state,mode='shadow'))
        while not stop.is_set():
            try:
                if not (root/'session'/'session.sqlite').exists():
                    if limits and limits.ends_at is not None and time.time()>=limits.ends_at:
                        state['phase']='expired_before_start'
                        break
                    state['phase']='bootstrapping'
                    meta=bootstrap(api,c,seed/'warmup.csv',warmup)
                    atomic_json(root/'warmup-manifest.json',meta)
                state.update(phase='connecting',last_progress_at=0)
                # Always public-only API, fixed shadow mode and the same persistent journal.
                run_session(api,c,warmup,root/'session','shadow',iterations=0,poll_seconds=30,on_heartbeat=heartbeat,
                            session_limits=limits,adaptive_risk=True)
                state['phase']='stopped';break
            except (RuntimeError,OSError) as exc:
                state['phase']='retrying'
                print(json.dumps({'event':'worker_retry','error':type(exc).__name__,'cash_reset':False}),flush=True)
                stop.wait(30)
            except ValueError as exc:
                if not (root/'session'/'session.sqlite').exists():
                    state['phase']='waiting_for_complete_warmup'
                    print(json.dumps({'event':'warmup_unavailable','error':str(exc),'cash_reset':False}),flush=True)
                    stop.wait(60)
                else:raise
    except KeyboardInterrupt:
        state['phase']='stopping'
    except Exception as exc:
        state['phase']='failed'
        print(json.dumps({'event':'worker_failed','error':str(exc),'cash_reset':False}),flush=True)
        # Configuration/model/state failures must fail the deployment rather than
        # leave a green but nonfunctional web process. Persistent files stay intact.
        raise
    finally:
        stop.set();server.shutdown()
        if lock:atomic_json(root/'status.json',dict(state,mode='shadow'))
        if lock:lock.close()


if __name__=='__main__':main()
