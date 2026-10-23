"""Loopback-only dashboard for a simulated portfolio. No exchange client or trading routes."""
from dataclasses import asdict
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import threading
import time
from urllib.request import urlopen

from .server import Store, engine_presentation, handler, journal, session_risk
from wallex_engine.session_limits import load_operator_limits
from wallex_engine.config import Config


def paper_snapshot(root, seed, worker_port=8732):
    now=int(time.time())
    j=journal(root, selected_run='paper-monday')
    summary=j.get('summary') or {}
    status_file=root/'paper-monday/status.json'
    status=json.loads(status_file.read_text()) if status_file.exists() else {}
    engine={'phase':'unavailable','ready':False,'health_status':'unreachable','run_directory':'paper-monday'}
    try:
        with urlopen(f'http://127.0.0.1:{worker_port}/healthz',timeout=1) as response:
            engine.update(json.load(response),health_status='available')
    except (OSError,ValueError):
        if status.get('phase') in ('stopped','expired_before_start','failed','stopping'):
            engine['phase']=status['phase']
    cash=summary.get('cash_tmn')
    assets=[] if cash is None else [dict(asset='TMN',quantity=cash,free=cash,locked=0,value_tmn=cash)]
    for symbol,position in summary.get('positions',{}).items():
        assets.append(dict(asset=symbol.removesuffix('TMN'),quantity=position['quantity'],
                           free=position['quantity'],locked=0,value_tmn=None))
    result=dict(captured_at=now,completed_at=now,mode='shadow',errors=[],journal=j,engine=engine,
                engine_running=engine.get('phase')=='running' and engine.get('ready') is True,
                equity_tmn=summary.get('equity_tmn'),assets=assets,open_orders=[],trades=[],
                market_compatibility='not_checked',**session_risk(j))
    config=Config.load(seed/'config.json')
    result['release_plan']={**asdict(load_operator_limits(seed,{})),
                            **{key:getattr(config,key) for key in ('initial_cash','gross_cap','max_drawdown')}}
    result['presentation']=engine_presentation(result)
    result['news']={'updated_at':status.get('news_updated_at'),'error':status.get('news_error'),'sources':status.get('news_sources',{})}
    result['presentation']['warnings'].insert(0,'این صفحه فقط شبیه‌سازی را نشان می‌دهد؛ موجودی و سفارش واقعی نیستند.')
    if status.get('news_error'):
        result['presentation']['warnings'].append('بعضی منابع خبر در دسترس نیستند؛ ورود با پوشش خبری موجود ارزیابی می‌شود.')
    if not engine.get('ready'):
        result['presentation']['warnings'].append('ارزش سبد از آخرین ضربان ثبت‌شده است؛ قیمت‌گذاری زنده تأیید نمی‌شود.')
    return result


def main():
    root=Path(os.environ['PAPER_STATE_ROOT']).resolve()
    seed=Path(os.environ['SEED_DIR']).resolve()
    root.mkdir(parents=True,exist_ok=True)
    token_file=root/'dashboard-token'
    if not token_file.exists():
        fd=os.open(token_file,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as output:output.write(secrets.token_urlsafe(32))
    store=Store()
    def refresh():
        while True:
            try:store.set(paper_snapshot(root,seed,int(os.getenv('PAPER_WORKER_PORT','8732'))))
            except Exception:store.set(dict(captured_at=int(time.time()),mode='shadow',errors=['paper_snapshot_failed']))
            time.sleep(10)
    threading.Thread(target=refresh,daemon=True).start()
    html=Path(__file__).with_name('index.html').read_text()
    html=html.replace('cat /data/observer/access-token','رمز اختصاصی داشبورد آزمایشی محلی')
    html=html.replace('رمز داشبورد را از خط فرمان همان برنامه در لیارا بخوانید. این رمز با کلید صرافی متفاوت است.',
                      'نمایش جلسهٔ شبیه‌سازی‌شدهٔ محلی؛ این صفحه به حساب معاملاتی متصل نیست.')
    ThreadingHTTPServer(('127.0.0.1',int(os.getenv('PORT','8733'))),
                        handler(store,token_file.read_text().strip(),html,secure=False)).serve_forever()


if __name__=='__main__':main()
