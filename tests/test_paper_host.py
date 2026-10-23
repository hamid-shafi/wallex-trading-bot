import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from observer.paper_host import paper_page, supervise_worker, terminal_status, worker_environment
from observer.server import Store, handler


class PaperHostTests(unittest.TestCase):
    def test_supervisor_retries_launch_error_then_stops_completed_session(self):
        stop=Mock();stop.is_set.return_value=False
        child=Mock(pid=123);child.poll.return_value=0
        children=[]
        with patch('observer.paper_host.subprocess.Popen',side_effect=[OSError('launch failure'),child]) as launch, \
             patch('observer.paper_host.terminal_status',side_effect=[False,False,True]), \
             patch('builtins.print'):
            supervise_worker(Path('/data/paper-sessions'),Path('/release/seed'),stop,children)
        self.assertEqual(launch.call_count,2)
        stop.wait.assert_called_once_with(15)
        self.assertEqual(children,[child])
        self.assertEqual(launch.call_args_list[0],launch.call_args_list[1])

    def test_cloud_mode_bypasses_all_live_activation(self):
        from observer import server
        with patch.dict('os.environ',{'PAPER_CLOUD_MODE':'1','OPERATOR_ARM':'I_ACCEPT_REAL_SPOT_TRADING'}), \
             patch('observer.paper_host.main') as paper, \
             patch('observer.server.engine_allowed',side_effect=AssertionError('live gate reached')):
            server.main()
        paper.assert_called_once_with()

    def test_real_dashboard_prefix_is_injected_before_initialization(self):
        html=(Path(__file__).resolve().parents[1]/'observer/index.html').read_text()
        result=paper_page(html)
        self.assertIn("window.WALLEX_API_PREFIX='/paper';",result)
        self.assertLess(result.index("window.WALLEX_API_PREFIX='/paper';"),
                        result.index('const API_PREFIX'))

    def test_child_has_no_account_credentials_or_arm_switches(self):
        env=worker_environment({'WALLEX_API_KEY':'secret','OPERATOR_ARM':'armed',
                                'OPERATOR_START_RELEASE':'old','PORT':'8000','PATH':'/bin'},
                               Path('/data/paper-sessions'),Path('/release/seed'))
        self.assertFalse(any(k.startswith(('WALLEX_','OPERATOR_')) for k in env))
        self.assertEqual(env['APP_MODE'],'shadow')
        self.assertEqual(env['WORKER_BIND_HOST'],'127.0.0.1')
        self.assertEqual(env['DATA_DIR'],'/data/paper-sessions/paper-monday')

    def test_completed_worker_is_not_restarted(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'paper-monday').mkdir()
            self.assertFalse(terminal_status(root))
            for phase,terminal in [('running',False),('failed',False),('stopped',True),('expired_before_start',True)]:
                (root/'paper-monday/status.json').write_text(json.dumps({'phase':phase}))
                self.assertEqual(terminal_status(root),terminal)

    def test_paper_and_account_routes_share_auth_but_never_data(self):
        live,paper=Store(),Store()
        live.set({'mode':'read_only','sentinel':'account'})
        paper.set({'mode':'shadow','sentinel':'paper'})
        server=ThreadingHTTPServer(('127.0.0.1',0),handler(live,'a'*32,'account',False,paper,'paper'))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base='http://127.0.0.1:'+str(server.server_port)
        try:
            with self.assertRaises(HTTPError) as error:urlopen(base+'/paper/api/snapshot')
            self.assertEqual(error.exception.code,401)
            response=urlopen(Request(base+'/paper/login',data=('token='+'a'*32).encode()))
            cookie=response.headers['Set-Cookie'].split(';')[0]
            for path,expected in [('/paper/api/snapshot','paper'),('/api/snapshot','account')]:
                data=json.load(urlopen(Request(base+path,headers={'Cookie':cookie})))
                self.assertEqual(data['sentinel'],expected)
            self.assertEqual(urlopen(base+'/paper/').read(),b'paper')
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(base+'/paper/trade',data=b'anything'))
            self.assertEqual(error.exception.code,405)
        finally:
            server.shutdown();server.server_close();thread.join()


if __name__=='__main__':unittest.main()
