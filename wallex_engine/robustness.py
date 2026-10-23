"""Evaluate a frozen policy on newly acquired older history; never reselect."""
import argparse
from dataclasses import replace
from pathlib import Path
import json
import math
import statistics
import time

import numpy as np

from .audit import atomic_json,source_hash
from .config import Config
from .data import digest_file
from .experiment import simulate,comparable
from .feature_study import feature_contexts,augment
from .mixture_study import prepare_experts,stream
from .trend_study import make_data


def passive(data,start,end,fraction=.1):
    indices=[i for i,t in enumerate(data['times']) if start<=t<end]
    first,last=indices[0],indices[-1]; initial=5000000.; drag=.00225; fee=.0035
    quantities={s:initial*fraction/len(data['symbols'])/(data['frames'][first][s].open*(1+drag)*(1+fee)) for s in data['symbols']}
    cash=initial*(1-fraction); peak=initial; dd=0.
    for i in indices:
        equity=cash+sum(quantities[s]*data['frames'][i][s].close for s in quantities)
        peak=max(peak,equity); dd=max(dd,1-equity/peak)
    final=cash+sum(quantities[s]*data['frames'][last][s].close*(1-drag)*(1-fee) for s in quantities)
    return {'net_return_pct':(final/initial-1)*100,'max_drawdown_pct':dd*100,
            'initial_exposure_fraction':fraction,'note':'fixed low-exposure passive comparator; no stops; not a trading recommendation'}


def moving_block(curve,initial=5000000.,seed=73):
    days={}
    for p in curve:
        days[(p['ts']+12600)//86400]=p['equity']
    values=np.asarray(list(days.values())); r=values/np.r_[initial,values[:-1]]-1
    rng=np.random.default_rng(seed); n=len(r)
    starts=rng.integers(0,n,size=(5000,(n+6)//7))
    ids=((starts[:,:,None]+np.arange(7))%n).reshape(5000,-1)[:,:n]
    trials=np.prod(1+r[ids],axis=1)-1
    return {'interval_95_pct':(np.quantile(trials,[.025,.975])*100).tolist(),
            'days':n,'block_days':7,'repetitions':5000,'warning':'descriptive only; policy selected after observing later history'}


def evaluate(path,news,selection,out,cache):
    root=Path(out); root.mkdir(parents=True,exist_ok=True)
    if (root/'results.json').exists():
        raise ValueError('completed assessment must not be overwritten')
    chosen=json.loads(Path(selection).read_text()); c=Config(**chosen['config'])
    atomic_json(root/'protocol.json',{'started_at':int(time.time()),'source_sha256':source_hash(),
        'selection_sha256':digest_file(selection),'data_sha256':digest_file(path),
        'frozen_at':chosen['frozen_at'],'features':chosen['features'],
        'role':'backward historical robustness; NOT a chronological prospective holdout',
        'restriction':'report all variants; no policy selection or threshold adjustment from this dataset',
        'exchange_writes':0,'account_reads':0})
    data=make_data(path,cache,news); start=data['times'][0]+90*86400; end=data['times'][-1]+3600
    grids,curves=prepare_experts(data,c,start,end)
    forecasts=stream(data,grids,curves,start,end,chosen['learned'],chosen['news'])
    contexts=feature_contexts(data)
    selected=augment(forecasts,chosen['features'],contexts,c.min_notional)
    report,curve=simulate(data,c,selected,start,end,root/'selected')
    ablation=[]
    for features in ([],['volatility'],['breadth'],['liquidity'],['volatility','breadth'],['volatility','breadth','liquidity']):
        r=simulate(data,c,augment(forecasts,features,contexts,c.min_notional),start,end)[0]
        ablation.append({'features':features,'report':r})
    stress=simulate(data,replace(c,fee_bps=70,spread_bps=50,slippage_bps=20,max_spread_bps=70),selected,start,end,root/'cost_x2_tradable')[0]
    # Keep decisions fixed as a second stress diagnostic; changed costs may change trade selection.
    trade_fees=report['fees_tmn']; extra_drag=report['turnover_tmn']*.00225
    repriced=report['net_return_pct']-(trade_fees+extra_drag)/c.initial_cash*100
    blocks=[simulate(data,c,selected,t,min(end,t+75*86400))[0] for t in range(start,end,75*86400)]
    windows=[simulate(data,c,selected,t,t+72*3600)[0] for t in range(start,end-72*3600+1,72*3600)]
    result={'selected':report,'feature_ablation':ablation,'cost_x2_tradable':stress,
        'fixed_fills_approx_double_cost_return_pct':repriced,'blocks':blocks,
        'positive_blocks':sum(comparable(r)>0 for r in blocks),'block_count':len(blocks),
        'three_day_windows':{'count':len(windows),'positive':sum(comparable(r)>0 for r in windows),
                             'median_return_pct':statistics.median(comparable(r) for r in windows),'reports':windows},
        'seven_day_bootstrap':moving_block(curve),'passive_10pct':passive(data,start,end),
        'cash_return_pct':0.,'selection_changed':False,'prospective_validated':False,
        'source_sha256':source_hash(),'exchange_writes':0}
    atomic_json(root/'results.json',result)
    print(json.dumps({'return':report['net_return_pct'],'drawdown':report['max_drawdown_pct'],
        'sells':report['sell_fills'],'cost_x2':stress['net_return_pct'],'blocks_positive':result['positive_blocks'],
        'blocks':len(blocks),'bootstrap':result['seven_day_bootstrap']['interval_95_pct']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--news',required=True)
    p.add_argument('--selection',required=True); p.add_argument('--out',required=True); p.add_argument('--cache',required=True)
    a=p.parse_args(); evaluate(a.data,a.news,a.selection,a.out,a.cache)
