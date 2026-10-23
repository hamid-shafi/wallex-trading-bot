from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from wallex_engine.cloud import news_worker


class OneCycle:
    done=False
    def is_set(self):return self.done
    def wait(self,seconds):self.done=True


class CloudNewsTests(unittest.TestCase):
    def test_gpr_failure_does_not_skip_rss_and_partial_failure_is_visible(self):
        state={}
        result={'sources':{'ok_feed':{'status':'ok','items':5},
                           'down_feed':{'status':'unavailable','error':'TimeoutError'}}}
        with tempfile.TemporaryDirectory() as folder, \
             patch('wallex_engine.news.collect',return_value=result) as rss, \
             patch('wallex_engine.gpr.collect',side_effect=OSError), \
             patch('wallex_engine.cloud.merge_news',return_value=5), \
             patch('builtins.print'):
            news_worker(Path(folder),OneCycle(),state)
        self.assertEqual(rss.call_count,1)
        self.assertEqual(state['news_error'],{'down_feed':'TimeoutError','gpr':'OSError'})
        self.assertEqual(state['news_sources'],result['sources'])


if __name__=='__main__':unittest.main()
