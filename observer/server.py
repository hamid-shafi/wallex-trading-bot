import hashlib
from contextlib import closing
import hmac
import json
import math
import os
import re
from pathlib import Path
import secrets
import sqlite3
import subprocess
import signal
import sys
import threading
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import parse_qs,urlsplit
from urllib.request import urlopen
from wallex_engine.engine import Portfolio,Intent
from wallex_engine.wallex import Wallex,APIError,validate_market
from wallex_engine.audit import source_hash
from wallex_engine.config import Config
from wallex_engine.session_limits import load_operator_limits


class ReadAPI(Wallex):
    PATHS=frozenset(('/v1/markets','/v1/account/balances','/v1/account/openOrders','/v1/account/trades'))
    def request(self,method,path,params=None,body=None,private=False):
        if method!='GET' or path not in self.PATHS or body is not None:
            raise APIError('observer is strictly read-only')
        return super().request(method,path,params,body,private)


def records(obj,key):
    if isinstance(obj,list):return obj
    if isinstance(obj,dict):
        for name in (key,'trades' if key=='AccountLatestTrades' else key):
            if name in obj:return records(obj[name],key)
        if 'symbol' in obj:return [obj]
        if not obj:return []
        rows=[]
        for value in obj.values():rows.extend(records(value,key))
        return rows
    raise APIError('unrecognized exchange list schema')


def safe_run(root,name):
    if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,79}',name):
        raise ValueError('invalid run directory')
    run=root/name
    for path in (run,run/'session',run/'session/session.sqlite',run/'session/events.jsonl',run/'failure.json'):
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('linked or escaped run path')
    return run


def select_run(root,selected_run=None,mode=None):
    if selected_run is not None:
        safe_run(root,selected_run)
        return selected_run
    candidates=[]
    for directory in root.iterdir():
        try:path=safe_run(root,directory.name)/'session/session.sqlite'
        except ValueError:continue
        if not path.is_file():continue
        with closing(sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True,timeout=2)) as db:
            row=db.execute("select value from state where key='trial'").fetchone()
            identity=db.execute("select value from state where key='identity'").fetchone()
        if mode is not None and (not identity or json.loads(identity[0]).get('mode')!=mode):continue
        started=json.loads(row[0]).get('started_at') if row else None
        candidates.append((started if started is not None else path.stat().st_mtime,path.parent.parent.name))
    return max(candidates)[1] if candidates else None


def order_fee_details(symbol,status,result):
    """Report settled fees without changing the legacy quote-fee/cash contract.

    Base fees are already reflected in normalized inventory. Their TMN value is
    added for reporting only; it must not be deducted from cash a second time.
    Older terminal rows lack base_fee_quantity but can still report their saved
    base_fee_tmn value. Pending rows have unknown fees, not confirmed zero fees.
    """
    def amount(value):
        if value is None or isinstance(value,bool):return None
        try:number=float(value)
        except (TypeError,ValueError,OverflowError):return None
        return number if math.isfinite(number) and number>=0 else None

    quote=quote_asset(symbol)
    settled=status=='DONE' and quote=='TMN'
    quote_fee=amount(result.get('fee')) if settled else None
    base_fee=amount(result.get('base_fee_tmn',0)) if settled else None
    total=quote_fee+base_fee if quote_fee is not None and base_fee is not None else None
    return {'fee_tmn':amount(total),'quote_fee_tmn':quote_fee,'base_fee_tmn':base_fee,
            'base_fee_quantity':amount(result.get('base_fee_quantity')) if settled else None,
            'fee_quote_asset':quote,'fee_base_asset':symbol[:-len(quote)] if quote else None}


def journal(root,selected_run=None,mode=None):
    root=Path(root)
    run=select_run(root,selected_run,mode)
    out={'orders':[],'summary':None,'trial':None,'decisions':[],'events':[],'curve':[],
         'exists':False,'run_directory':run,'selection':'selected' if selected_run else 'latest_recorded'}
    if run is None:return out
    path=root/run/'session/session.sqlite'
    out['exists']=path.exists()
    if path.exists():
        with closing(sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True,timeout=2)) as db:
            states={k:json.loads(v) for k,v in db.execute("select key,value from state where key in ('summary','trial','identity','portfolio')")}
            identity=states.pop('identity',{});portfolio=states.pop('portfolio',{})
            config=identity.get('config');out['mode']=identity.get('mode')
            if config:
                checked=Config(**config).validate()
                out['effective_config']={key:getattr(checked,key) for key in
                                        ('initial_cash','gross_cap','max_drawdown','daily_loss','position_cap',
                                         'risk_fraction','portfolio_stop_risk','symbols')}
            out.update(states)
            for oid,intent,status,result in db.execute('select id,intent,status,result from orders order by rowid desc limit 100'):
                i=json.loads(intent);r=json.loads(result) if result else {}
                out['orders'].append(dict(id=oid,symbol=i['symbol'],side=i['side'],requested=i['quantity'],
                    reason=i['reason'],status=status,executed=r.get('quantity'),price=r.get('price'),fee=r.get('fee'),ts=r.get('ts'),
                    **order_fee_details(i['symbol'],status,r)))
            if out['mode']=='shadow':
                summary=out.get('summary') or {}
                out['ledger']={'cash_tmn':portfolio.get('cash',summary.get('cash_tmn')),
                               'realized_pnl_tmn':portfolio.get('realized',summary.get('realized_pnl_tmn')),
                               'fees_tmn':portfolio.get('fees',summary.get('fees_tmn')),
                               'fills':len(portfolio['fills']) if 'fills' in portfolio else summary.get('fills')}
                out['paper_trades']=[{**fill,'quote_asset':quote_asset(fill.get('symbol'))} for fill in portfolio.get('fills',[])[-100:]]
            elif config:
                p=Portfolio(checked.initial_cash)
                for intent,result in db.execute("select intent,result from orders where status='DONE' order by seq"):
                    i,r=Intent(**json.loads(intent)),json.loads(result)
                    if r['quantity']:
                        p.apply(i,r['quantity'],r['price'],r['fee'],r['ts']);p.fees+=r.get('base_fee_tmn',0)
                out['ledger']={'cash_tmn':p.cash,'realized_pnl_tmn':p.realized,'fees_tmn':p.fees,'fills':len(p.fills)}
            else:out['ledger_error']='recorded_configuration_missing'
    failure=root/run/'failure.json'
    if failure.exists():out['failure']=json.loads(failure.read_text())
    out['run_directory']=run
    eventfile=root/run/'session/events.jsonl'
    if eventfile.exists():
        with eventfile.open('rb') as f:
            size=eventfile.stat().st_size;f.seek(max(0,size-1_000_000))
            if f.tell():f.readline()
            lines=f.readlines()
        for line in lines:
            try:r=json.loads(line)
            except (ValueError,UnicodeError):continue
            kind,payload=r.get('kind'),r.get('payload',{})
            if kind=='heartbeat':out['curve'].append({'ts':r['ts'],'equity':payload.get('equity_tmn')})
            if kind=='decision':out['decisions']=[dict(d,ts=r['ts']) for d in payload.get('decisions',[])]
            if kind in ('session_start','decision','order_intent','exchange_terminal','data_or_risk_error'):
                out['events'].append({'kind':kind,'ts':r['ts']})
        out['events']=out['events'][-40:];out['curve']=out['curve'][-240:]
    return out


def snapshot(api,root,worker_expected=None):
    result={'captured_at':int(time.time()),'mode':'read_only','engine_running':False,'errors':[]}
    seed=Path(os.getenv('SEED_DIR','/app/deploy/seed'))
    try:
        planned=Config.load(seed/'config.json')
        limits=load_operator_limits(seed,os.environ).validate(planned.initial_cash)
        result['release_plan']={**asdict(limits),'initial_cash':planned.initial_cash,
                                'gross_cap':planned.gross_cap,'max_drawdown':planned.max_drawdown,
                                'daily_loss':planned.daily_loss,'position_cap':planned.position_cap,
                                'risk_fraction':planned.risk_fraction,'portfolio_stop_risk':planned.portfolio_stop_risk}
    except Exception:result['errors'].append('session_plan_invalid')
    if worker_expected is False:
        result['engine']={'phase':'observer_only','ready':False,'health_status':'not_started'}
    else:
        try:
            with urlopen('http://127.0.0.1:8001/healthz',timeout=1) as r:health=json.load(r)
            if not isinstance(health,dict) or not isinstance(health.get('phase'),str) or not isinstance(health.get('ready'),bool):
                raise ValueError('invalid worker health response')
            result['engine']={**health,'health_status':'ok'}
            result['engine_running']=health['phase']=='running' and health['ready'] is True
        except Exception:
            result['engine']={'phase':'unavailable','ready':False,'health_status':'unreachable'}
            result['errors'].append('worker_health_unavailable')
    prior=root/'observer/prior-status.json'
    if prior.exists():result['prior_engine']=json.loads(prior.read_text())
    try:result['journal']=journal(root,result['engine'].get('run_directory'),mode='live')
    except Exception:result['errors'].append('journal_unavailable');result['journal']={}
    j=result['journal']
    result.update(session_risk(j))
    markets={}
    try:
        markets=api.markets()
        symbols=['BTCTMN','ETHTMN','SOLTMN','XRPTMN','BNBTMN']
        for s in symbols:validate_market(markets[s])
        result['market_compatibility']='pass'
    except Exception:result['errors'].append('prices_unavailable')
    try:
        raw=api.request('GET','/v1/account/balances',private=True)['result'];balances=raw.get('balances',raw)
        assets=[];total=0.;missing=[]
        for name,row in balances.items():
            q,locked=float(row['value']),float(row.get('locked',0))
            if not math.isfinite(q) or not math.isfinite(locked) or q<0 or locked<0:raise APIError('invalid balance')
            if not (q or locked or name=='TMN'):continue
            bid=1. if name=='TMN' else float(markets.get(name+'TMN',{}).get('stats',{}).get('bidPrice',0))
            value=q*bid if bid>0 else None
            if value is None:missing.append(name)
            else:total+=value
            assets.append(dict(asset=name,quantity=q,locked=locked,free=max(0,q-locked),price_tmn=bid or None,value_tmn=value))
        result.update(assets=assets,unpriced_assets=missing,equity_tmn=total if not missing else None,
                      balance_at=int(time.time()))
    except Exception:result['errors'].append('balances_unavailable')
    for path,key,result_key in (('/v1/account/openOrders','orders','open_orders'),('/v1/account/trades','AccountLatestTrades','trades')):
        try:
            raw=api.request('GET',path,private=True)['result'];rows=records(raw,key)
            if not isinstance(rows,list):raise APIError('unknown exchange schema')
            allowed=('id','clientOrderId','symbol','side','isBuyer','quantity','executedQty','price','sum','fee','feeAsset','timestamp','status','active')
            result[result_key]=[{**{k:row[k] for k in allowed if k in row},
                                 'quantity':row.get('quantity',row.get('origQty')),
                                 'quote_asset':quote_asset(row.get('symbol'),markets)} for row in rows[:100]]
            result[result_key+'_at']=int(time.time())
        except Exception:result['errors'].append(result_key+'_unavailable')
    result['completed_at']=int(time.time())
    result['presentation']=engine_presentation(result)
    return result


def quote_asset(symbol,markets=None):
    market=(markets or {}).get(symbol,{})
    if isinstance(market.get('quoteAsset'),str):return market['quoteAsset']
    if isinstance(symbol,str):
        for quote in ('USDT','TMN','BTC','ETH'):
            if symbol.endswith(quote) and len(symbol)>len(quote):return quote
    return None


def session_risk(j):
    trial=j.get('trial') or {};limits=trial.get('limits',{});config=j.get('effective_config',{})
    return {'budget_tmn':config.get('initial_cash'),
            'risk':{'gross_cap':config.get('gross_cap'),'max_drawdown':config.get('max_drawdown'),
                    'daily_loss':config.get('daily_loss'),'position_cap':config.get('position_cap'),
                    'risk_fraction':config.get('risk_fraction'),'portfolio_stop_risk':config.get('portfolio_stop_risk'),
                    'floor':limits.get('floor_tmn'),'ceiling':limits.get('ceiling_tmn'),
                    'seconds':limits.get('seconds'),'ends_at':trial.get('deadline'),
                    'source':'journal' if j.get('exists') else 'unavailable','limits_known':bool(limits)}}


def engine_presentation(data):
    phase=data.get('engine',{}).get('phase')
    titles={'running':'موتور در حال اجراست','preflight':'موتور در حال بررسی پیش از شروع است',
            'stopped':'جلسه پایان یافته است','manual_reconciliation_required':'موتور متوقف است؛ تطبیق دفتر و صرافی لازم است',
            'stopped_by_operator':'موتور به درخواست کاربر متوقف شده است',
            'unavailable':'ارتباط با موتور برقرار نیست؛ وضعیت اجرای فعلی نامشخص است',
            'observer_only':'موتور توسط این سرویس شروع نشده است؛ داشبورد در حالت نظارت است'}
    titles.update(failed='اجرای موتور با خطا پایان یافته است',expired_before_start='مهلت جلسه پیش از شروع گذشته است',
                  stopping='موتور در حال توقف است',awaiting_operator='موتور منتظر شروع دستی است',
                  starting='موتور در حال آماده‌سازی داده‌ها است',bootstrapping='موتور در حال آماده‌سازی داده‌ها است')
    title=titles.get(phase,'وضعیت اجرای موتور تأیید نشده است')
    if phase=='running' and not data.get('engine_running'):title='ضربان موتور تازه یا آماده نیست؛ اجرای فعلی تأیید نمی‌شود'
    warnings=[];history=[];j=data.get('journal',{})
    summary=j.get('summary') or {};trial=j.get('trial') or {};stop_reason=trial.get('reason')
    reasons={'time_limit':'پایان مهلت زمانی','capital_floor':'رسیدن به مرز پایین','capital_ceiling':'رسیدن به مرز بالا',
             'drawdown_halt':'توقف افت سرمایه','operator_stop':'توقف کاربر'}
    if stop_reason:
        warnings.append('علت توقف در آخرین دفتر: '+reasons.get(stop_reason,stop_reason)+'.')
        if phase=='observer_only':title='جلسهٔ ثبت‌شده پایان یافته است؛ موتور توسط این سرویس شروع نشده است'
    if summary.get('manual_reconciliation_required') or phase=='manual_reconciliation_required':
        warnings.append('دفتر جلسه نیاز به تطبیق دارد؛ تسویهٔ کامل تأیید نشده است.')
    if summary.get('positions') and summary.get('settled_to_tmn') is False:
        warnings.append('دارایی باقی‌مانده در دفتر: '+', '.join(summary['positions'])+'. ماندهٔ دفتر را با کیف پول مقایسه کنید.')
    if data.get('prior_engine',{}).get('reason')=='market_schema_missing_maxQty':
        history.append('سابقهٔ نسخهٔ قدیمی: نبود سقف اختیاری حجم مانع ارسال سفارش شده بود. '+
                       ('این مشکل در بررسی فعلی بازارها رفع شده است.' if data.get('market_compatibility')=='pass' else 'بررسی فعلی سازگاری هنوز تأیید نشده است.'))
    failure=j.get('failure')
    last_heartbeat=max((row.get('ts',0) for row in j.get('curve',[])),default=0)
    current_failure=failure if (failure and failure.get('ts',0)>=max(last_heartbeat,trial.get('started_at',0))) else None
    if failure and not current_failure:history.append('در دفتر یک خطای قبلی ثبت شده است؛ این رکورد به‌تنهایی وضعیت فعلی موتور نیست.')
    return dict(title=title,warnings=warnings,history=history,current_failure=current_failure)


class Store:
    def __init__(self):self.data={'captured_at':None,'errors':['loading']};self.lock=threading.Lock()
    def get(self):
        with self.lock:return dict(self.data)
    def set(self,data):
        with self.lock:self.data=data


def handler(store,token,html,secure=True,paper_store=None,paper_html=None):
    # Session signing key is separate from the login token and never sent to the browser.
    sign_key=secrets.token_bytes(32);attempts={};auth_lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,status,body,mime='application/json',extra=None):
            body=body.encode() if isinstance(body,str) else body
            self.send_response(status);self.send_header('Content-Type',mime+'; charset=utf-8')
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Referrer-Policy','no-referrer');self.send_header('X-Frame-Options','DENY')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'")
            for k,v in (extra or {}).items():self.send_header(k,v)
            self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def authenticated(self):
            cookies={x.strip().split('=',1)[0]:x.strip().split('=',1)[1] for x in self.headers.get('Cookie','').split(';') if '=' in x}
            try:
                value,signature=cookies.get('wx_session','').rsplit('.',1)
                return (int(value.split(':')[0])>time.time() and hmac.compare_digest(signature,hmac.new(sign_key,value.encode(),hashlib.sha256).hexdigest()))
            except (ValueError,TypeError):return False
        def do_GET(self):
            path=urlsplit(self.path).path
            if paper_store is not None and path in ('/paper','/paper/','/paper/dashboard'):
                return self.send(200,paper_html,'text/html')
            if paper_store is not None and path=='/paper/api/snapshot':
                if not self.authenticated():return self.send(401,'{"error":"authentication_required"}')
                return self.send(200,json.dumps(paper_store.get(),ensure_ascii=False,allow_nan=False))
            if paper_store is not None and path=='/paper/healthz':
                return self.send(200,json.dumps({'service':'wallex-paper','mode':'shadow','live_enabled':False}))
            if path=='/healthz':return self.send(200,json.dumps({'service':'wallex-observer','mode':'read_only'}))
            if path in ('/','/dashboard'):return self.send(200,html,'text/html')
            if path=='/api/snapshot':
                if not self.authenticated():return self.send(401,'{"error":"authentication_required"}')
                return self.send(200,json.dumps(store.get(),ensure_ascii=False,allow_nan=False))
            return self.send(404,'{}')
        def do_POST(self):
            if self.path not in (('/login','/paper/login') if paper_store is not None else ('/login',)):
                return self.send(405,'{}')
            try:size=int(self.headers.get('Content-Length','0'))
            except ValueError:return self.send(400,'{}')
            if not 0<size<=512:return self.send(400,'{}')
            host=self.headers.get('Host','');origin=self.headers.get('Origin')
            if origin and urlsplit(origin).netloc!=host:return self.send(403,'{}')
            with auth_lock:
                failures=[t for t in attempts.get(self.client_address[0],[]) if time.time()-t<60]
                attempts[self.client_address[0]]=failures
                if len(failures)>=6:return self.send(429,'{}')
                supplied=parse_qs(self.rfile.read(size).decode(errors='replace')).get('token',[''])[0]
                if not hmac.compare_digest(supplied.encode(),token.encode()):
                    failures.append(time.time());return self.send(401,'{}')
            value=str(int(time.time())+43200)+':'+secrets.token_hex(16)
            sig=hmac.new(sign_key,value.encode(),hashlib.sha256).hexdigest()
            cookie='wx_session='+value+'.'+sig+'; HttpOnly; SameSite=Strict; Path=/; Max-Age=43200'+('; Secure' if secure else '')
            self.send(200,'{"ok":true}',extra={'Set-Cookie':cookie})
    return Handler


def main():
    if os.getenv('PAPER_CLOUD_MODE')=='1':
        from .paper_host import main as paper_main
        return paper_main()
    root=Path(os.getenv('DATA_DIR','/data'));private=root/'observer';private.mkdir(parents=True,exist_ok=True)
    token_file=private/'access-token'
    if not token_file.exists():
        fd=os.open(token_file,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as f:f.write(secrets.token_urlsafe(24))
    token=token_file.read_text().strip()
    if len(token)<24:raise ValueError('observer token invalid')
    # Previous arm switches cannot resume a changed release. Only a human's
    # exact-release opt-in enables the separate engine process.
    child=None
    if engine_allowed(os.environ,source_hash()):
        env=dict(os.environ,PORT='8001')
        child=subprocess.Popen([sys.executable,'-m','wallex_engine.operator_trial'],env=env)
    def shutdown(*args):
        if child is not None:child.terminate()
        raise SystemExit(0)
    signal.signal(signal.SIGTERM,shutdown)
    store=Store();api=ReadAPI(timeout=8,retries=1)
    def loop():
        while True:
            try:store.set(snapshot(api,root,worker_expected=child is not None))
            except Exception:store.set({'captured_at':int(time.time()),'errors':['snapshot_failed']})
            time.sleep(20)
    threading.Thread(target=loop,daemon=True).start()
    html=Path(__file__).with_name('index.html').read_bytes()
    ThreadingHTTPServer(('0.0.0.0',int(os.getenv('PORT','8000'))),handler(store,token,html)).serve_forever()


def engine_allowed(env,release):
    return env.get('OPERATOR_ARM')=='I_ACCEPT_REAL_SPOT_TRADING' and env.get('OPERATOR_START_RELEASE')==release


if __name__=='__main__':main()
