"""Liara PID-1 observer and supervised, credential-free paper worker."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer

from .server import ReadAPI, Store, handler, snapshot
from .paper_server import paper_snapshot


def worker_environment(env, root, seed):
    clean = {k: v for k, v in env.items()
             if not k.startswith(('WALLEX_', 'OPERATOR_'))}
    clean.update(APP_MODE='shadow', DATA_DIR=str(root / 'paper-monday'),
                 SEED_DIR=str(seed), PORT='8002', WORKER_BIND_HOST='127.0.0.1',
                 PYTHONUNBUFFERED='1')
    return clean


def terminal_status(root):
    try:
        data = json.loads((root / 'paper-monday/status.json').read_text())
        return data.get('phase') in ('stopped', 'expired_before_start')
    except (OSError, ValueError):
        return False


def paper_page(html):
    if '<script>' not in html:
        raise ValueError('dashboard script marker missing')
    return html.replace('<script>', "<script>window.WALLEX_API_PREFIX='/paper';\n", 1)


def supervise_worker(root, seed, stop, children):
    delay = 15
    while not stop.is_set() and not terminal_status(root):
        started = time.monotonic()
        try:
            child = subprocess.Popen(
                [sys.executable, '-m', 'wallex_engine.cloud'],
                cwd=Path(__file__).resolve().parents[1],
                env=worker_environment(os.environ, root, seed))
            children[:] = [child]
            print(json.dumps({'event': 'paper_worker_started', 'pid': child.pid,
                              'mode': 'shadow', 'credentials_passed': False}), flush=True)
            while child.poll() is None and not stop.wait(1):
                pass
            if stop.is_set() or terminal_status(root):
                break
            code = child.poll()
        except OSError as exc:
            code = type(exc).__name__
        print(json.dumps({'event': 'paper_worker_restart_pending',
                          'exit_code': code, 'delay': delay,
                          'cash_reset': False}), flush=True)
        stop.wait(delay)
        delay = 15 if time.monotonic() - started > 600 else min(delay * 2, 300)


def main():
    data = Path(os.getenv('DATA_DIR', '/data')).resolve()
    root = data / 'paper-sessions'
    seed = Path(__file__).resolve().parents[1] / 'deploy/paper-seed'
    root.mkdir(parents=True, exist_ok=True)
    token = (data / 'observer/access-token').read_text().strip()
    if len(token) < 24:
        raise ValueError('observer access token missing or invalid')
    stop = threading.Event()
    children = []

    def shutdown(*args):
        stop.set()
        for child in children:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    threading.Thread(target=supervise_worker, args=(root,seed,stop,children), daemon=True).start()
    live_store, paper_store = Store(), Store()
    api = ReadAPI(timeout=8, retries=1)

    def refresh_live():
        while not stop.is_set():
            try:
                live_store.set(snapshot(api, data, worker_expected=False))
            except Exception:
                live_store.set({'captured_at': int(time.time()), 'mode': 'read_only',
                                'errors': ['snapshot_failed']})
            stop.wait(20)

    def refresh_paper():
        while not stop.is_set():
            try:
                paper_store.set(paper_snapshot(root, seed, worker_port=8002))
            except Exception:
                paper_store.set({'captured_at': int(time.time()), 'mode': 'shadow',
                                 'errors': ['paper_snapshot_failed']})
            stop.wait(10)

    threading.Thread(target=refresh_live, daemon=True).start()
    threading.Thread(target=refresh_paper, daemon=True).start()
    html = Path(__file__).with_name('index.html').read_text()
    paper_html = paper_page(html)
    ThreadingHTTPServer(('0.0.0.0', int(os.getenv('PORT', '8000'))),
                        handler(live_store, token, html,
                                paper_store=paper_store, paper_html=paper_html)).serve_forever()


if __name__ == '__main__':
    main()
