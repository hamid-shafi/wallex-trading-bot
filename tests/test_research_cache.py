"""Offline regression checks for stale/reordered research cache rejection."""
from dataclasses import replace
from pathlib import Path
import importlib.util
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from wallex_engine.data import save_bars,synthetic
from wallex_engine.longterm import SYMBOLS
from wallex_engine import trend_study as study


class ResearchCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shared=tempfile.TemporaryDirectory(prefix='wallex-research-cache-fixture-')
        cls.shared_path=Path(cls.shared.name)
        cls.dataset=cls.shared_path/'hourly.csv'
        cls.cache=cls.shared_path/'features.npz'
        cls.bars=synthetic(SYMBOLS,count=600,interval=3600)
        save_bars(cls.dataset,cls.bars)
        cls.data=study.make_data(cls.dataset,cls.cache)

    @classmethod
    def tearDownClass(cls):
        cls.shared.cleanup()

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='wallex-research-cache-test-')
        self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)

    def cached_targets(self):
        path=self.path/'targets-336.npy'
        data=study.retarget(self.data,336,path)
        return path,data

    def test_feature_and_target_cache_round_trip(self):
        loaded=study.make_data(self.dataset,self.cache)
        for field in study.FEATURE_CACHE_FIELDS:
            np.testing.assert_array_equal(loaded[field],self.data[field])
        path,first=self.cached_targets()
        second=study.retarget(loaded,336,path)
        np.testing.assert_array_equal(first['y'],second['y'])
        self.assertGreater(np.isfinite(first['y']).sum(),0)
        self.assertEqual(first['target_horizon'],336)

    def test_same_row_count_different_market_data_cannot_reuse_feature_cache(self):
        changed=self.path/'changed.csv'
        bars=list(self.bars)
        b=bars[1000]
        bars[1000]=replace(b,open=b.open*1.01,high=b.high*1.01,low=b.low*1.01,close=b.close*1.01)
        save_bars(changed,bars)
        with self.assertRaisesRegex(ValueError,'rebuild'):
            study.make_data(changed,self.cache)

    def test_changed_feature_schema_rejects_cache(self):
        with patch.object(study,'FEATURE_NAMES',study.FEATURE_NAMES+['new_feature']):
            with self.assertRaisesRegex(ValueError,'schema'):
                study.make_data(self.dataset,self.cache)

    def test_changed_feature_source_rejects_cache(self):
        original=study.digest_file
        def source_change(path):
            return 'changed-source-hash' if Path(path).name=='trained.py' else original(path)
        with patch.object(study,'digest_file',side_effect=source_change):
            with self.assertRaisesRegex(ValueError,'source'):
                study.make_data(self.dataset,self.cache)

    def test_legacy_feature_and_target_caches_rejected(self):
        legacy=self.path/'legacy.npz'
        np.savez_compressed(legacy,**{k:self.data[k] for k in study.FEATURE_CACHE_FIELDS},
                            dataset_hash=study.digest_file(self.dataset))
        with self.assertRaisesRegex(ValueError,'legacy'):
            study.make_data(self.dataset,legacy)
        target=self.path/'legacy.npy'
        np.save(target,np.zeros_like(self.data['y']),allow_pickle=False)
        with self.assertRaisesRegex(ValueError,'legacy'):
            study.retarget(self.data,336,target)

    def test_target_horizon_must_match(self):
        path,_=self.cached_targets()
        with self.assertRaisesRegex(ValueError,'label parameters'):
            study.retarget(self.data,168,path)

    def test_reordered_rows_cannot_reuse_target_cache(self):
        path,_=self.cached_targets()
        reordered={**self.data,'locations':self.data['locations'][::-1],
                   'row_ts':self.data['row_ts'][::-1]}
        with self.assertRaisesRegex(ValueError,'rebuild'):
            study.retarget(reordered,336,path)

    def test_mutated_market_inputs_cannot_reuse_target_cache(self):
        path,_=self.cached_targets()
        frames=list(self.data['frames'])
        frame=dict(frames[200]); b=frame[SYMBOLS[0]]
        frame[SYMBOLS[0]]=replace(b,volume=b.volume+1)
        frames[200]=frame
        with self.assertRaisesRegex(ValueError,'rebuild'):
            study.retarget({**self.data,'frames':frames},336,path)

    def test_corrupted_array_rejected_even_with_original_metadata(self):
        path,_=self.cached_targets()
        with np.load(path,allow_pickle=False) as saved:
            content={k:saved[k] for k in saved.files}
        content['y']=np.zeros_like(content['y'])
        with path.open('wb') as handle:
            np.savez_compressed(handle,**content)
        with self.assertRaisesRegex(ValueError,'array contents'):
            study.retarget(self.data,336,path)

    def test_truncated_cache_rejected(self):
        path=self.path/'truncated.npz'
        path.write_bytes(self.cache.read_bytes()[:50])
        with self.assertRaisesRegex(ValueError,'rebuild'):
            study.make_data(self.dataset,path)

    def test_longterm_reader_uses_the_same_legacy_rejection(self):
        # Pandas is not installed in this review environment. This path must fail
        # before any vectorized work, so a stub proves it reaches the cache gate.
        source=ROOT/'wallex_engine/longterm_study.py'
        name='wallex_engine._cache_gate_review'
        spec=importlib.util.spec_from_file_location(name,source)
        module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'pandas':types.ModuleType('pandas')}):
            spec.loader.exec_module(module)
        legacy=self.path/'legacy.npz'
        np.savez_compressed(legacy,dataset_hash=study.digest_file(self.dataset))
        with self.assertRaisesRegex(ValueError,'legacy'):
            module.build_data([(self.dataset,legacy)],'not-read.jsonl')


if __name__=='__main__':
    unittest.main()
