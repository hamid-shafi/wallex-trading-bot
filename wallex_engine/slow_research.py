"""Declared low-turnover directional baselines. Scores are not return forecasts."""
import argparse
from dataclasses import asdict,replace
from pathlib import Path
import hashlib
import json
import math
import statistics
import time

import numpy as np

from .audit import atomic_json,canonical,source_hash
from .config import Config
from .data import digest_file
from .experiment import simulate,iso,comparable
from .trend_study import make_data


def signals(data,fast,slow,band,news=False):
    close=np.asarray([[f[s].close for s in data['symbols']] for f in data['frames']])
    cumulative=np.vstack((np.zeros((1,len(data['symbols']))),np.cumsum(close,axis=0)))
    output={}
    news_values=data['news_x'][:,-6:]
    for row,((i,j),atr) in enumerate(zip(data['locations'],data['atr'])):
        i,j=int(i),int(j)
        if i+1<slow or data['times'][i]-data['times'][i-slow+1]!=(slow-1)*3600:
            continue
        fm=(cumulative[i+1,j]-cumulative[i+1-fast,j])/fast
        sm=(cumulative[i+1,j]-cumulative[i+1-slow,j])/slow
        ratio=fm/sm-1
        enter=ratio>band and close[i,j]>fm
        exit=ratio < -band
        # Encoded utility for the shared execution simulator, NOT predicted profit.
        utility=.03+min(.1,max(0,ratio)) if enter else (-.03 if exit else 0.)
        risk=1.
        if news:
            if news_values[row,-1]<1:
                utility=min(0,utility)
            elif news_values[row,2]>1.5 and news_values[row,3]>.2:
                risk=2.
        output[(i,data['symbols'][j])]=(utility,0.,float(atr*risk),1000,'directional-rule',
            {'directional_utility':utility,'fast_slow_ratio':ratio,'news_coverage':float(news_values[row,-1]),
             'news_risk_multiplier':risk,'not_a_return_forecast':True})
    return output


def study(path,news,out,cache):
    root=Path(out); root.mkdir(parents=True,exist_ok=True)
    if (root/'selection.json').exists():
        raise ValueError('frozen study cannot be overwritten')
    specs=[{'fast':f,'slow':s,'band':b,'news':n}
           for f,s in ((24,168),(72,336),(168,720),(336,720))
           for b in (0.,.01) for n in (False,True)]
    protocol={'created_at':int(time.time()),'data_sha256':digest_file(path),'source_sha256':source_hash(),
       'specifications':specs,'previous_data_status':'all 540 days are development; no future validation claim',
       'new_history_status':'older-market sealed until policy selection is frozen; backward historical robustness only',
       'selection':'median six 75-day blocks minus 0.5 median drawdown; >=4 positive and >=12 sell fills; rank all failures diagnostically',
       'costs':'35bps per side fee +25bps full spread +10bps per side slippage',
       'orders':0,'cash_baseline':0,'score_is_expected_return':False}
    atomic_json(root/'protocol.json',protocol)
    d=make_data(path,cache,news)
    begin=d['times'][0]+90*86400; end=d['times'][-1]+3600
    blocks=[(t,min(end,t+75*86400)) for t in range(begin,end,75*86400)]
    c=Config(symbols=d['symbols'],interval=3600,horizon=72,strategy='forecast',
             max_holding_bars=2160,fee_bps=35,spread_bps=25,slippage_bps=10,
             stop_atr=6.,min_stop_fraction=.05,cooldown=72,entry_buffer_bars=24,
             min_samples=300,uncertainty_penalty=0.,edge_bps=0,settlement_buffer=7200)
    candidates=[]
    for spec in specs:
        stream=signals(d,**spec)
        reports=[simulate(d,c,stream,left,right)[0] for left,right in blocks]
        returns=[comparable(r) for r in reports]
        total=simulate(d,c,stream,begin,end)[0]
        eligible=sum(v>0 for v in returns)>=4 and sum(r['sell_fills'] for r in reports)>=12
        score=statistics.median(returns)-.5*statistics.median(r['max_drawdown_pct'] for r in reports)
        candidates.append({'spec':spec,'returns_pct':returns,'reports':reports,'score':score,'eligible':eligible,
                           'development_total':total})
        print(json.dumps({'spec':spec,'returns':returns,'score':score,'total':total['net_return_pct']}),flush=True)
    atomic_json(root/'candidates.json',candidates)
    eligible=[v for v in candidates if v['eligible']]
    best=max(eligible or candidates,key=lambda v:v['score'])
    chosen={**best,'validation_eligible':bool(eligible),'frozen_at':int(time.time()),'config':asdict(c),
            'candidate_count':len(candidates),'protocol_sha256':digest_file(root/'protocol.json')}
    atomic_json(root/'selection.json',chosen)
    simulate(d,c,signals(d,**best['spec']),begin,end,root/'development_selected')
    print(json.dumps({'selection':best['spec'],'eligible':bool(eligible)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--news',required=True)
    p.add_argument('--out',required=True); p.add_argument('--cache',required=True)
    a=p.parse_args(); study(a.data,a.news,a.out,a.cache)
