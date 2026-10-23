"""Fixed monthly learning from counterfactual expert portfolios, net of costs."""
import argparse
from dataclasses import asdict,replace
from pathlib import Path
import bisect
import json
import math
import statistics
import time

from .audit import atomic_json,source_hash
from .config import Config
from .data import digest_file
from .experiment import simulate,comparable
from .mixture import EXPERTS,combine,news_risk
from .slow_research import signals
from .trend_study import make_data
from .gpr import GPR_NAMES


def fit_weights(curves,cutoff,learned):
    if not learned:
        return [1/9]*9
    scores=[]
    for curve in curves:
        past=[p['equity'] for p in curve if cutoff-90*86400<=p['ts']<=cutoff]
        if len(past)<24:
            scores.append(0.); continue
        peak=past[0]; dd=0.
        for e in past:
            peak=max(peak,e); dd=max(dd,1-e/peak)
        scores.append(math.log(past[-1]/past[0])-.5*dd)
    logits=[max(-10,min(10,30*s)) for s in scores]+[0.]
    highest=max(logits); exp=[math.exp(s-highest) for s in logits]; total=sum(exp)
    return [.05/9+.95*v/total for v in exp]


def prepare_experts(data,config,start,end):
    grids=[]; curves=[]
    for spec in EXPERTS:
        grid=signals(data,**spec)
        _,curve=simulate(data,config,grid,start,end)
        grids.append(grid); curves.append(curve)
    return grids,curves


def stream(data,grids,curves,start,end,learned,news,models=None):
    models={} if models is None else models
    output={}; current=None
    news_values=data['news_x'][:,-6:]
    for n,((i,j),atr,t) in enumerate(zip(data['locations'],data['atr'],data['row_ts'])):
        i,j,t=int(i),int(j),int(t)
        if not start<=t<end:
            continue
        left=start+(t-start)//(30*86400)*(30*86400)
        if left!=current:
            current=left; weights=fit_weights(curves,left,learned); models[left]=weights
        utilities=[g.get((i,data['symbols'][j]),(0.,))[0] for g in grids]
        # All trend windows must be fully observed; no short-window fallback across gaps.
        if any((i,data['symbols'][j]) not in g for g in grids):
            continue
        utility,strength=combine(utilities,weights)
        multiplier=news_risk(news_values[n]) if news else 1.
        if news and not news_values[n,-1]:
            utility=min(0,utility)
        output[(i,data['symbols'][j])]=(utility,0.,float(atr*multiplier),1000,'learned-trend-committee',
            {'directional_utility':utility,'committee_strength':strength,'not_a_return_forecast':True,
             'news_coverage':float(news_values[n,-1]),'news_risk_multiplier':multiplier})
    return output


def model(data,curves,cutoff,learned,news,eligible):
    return {'artifact_type':'trend_mixture','experts':EXPERTS,'weights':fit_weights(curves,cutoff,learned),
        'symbols':data['symbols'],'interval':3600,'horizon':72,'training_cutoff':int(cutoff),
        'valid_until':int(cutoff+30*86400),'use_news':news,'news_feature_names':GPR_NAMES,
        'learning':'monthly softmax of prior 90d net expert log returns minus half drawdown; cash expert; 5pct prior floor',
        'research_only':True,'validation_eligible':eligible,'created_at':int(time.time()),
        'signal_type':'directional utility; not expected return','historical_validation_is_prospective':False}


def study(path,news,out,cache):
    root=Path(out); root.mkdir(parents=True,exist_ok=True)
    if (root/'selection.json').exists():
        raise ValueError('selection frozen')
    atomic_json(root/'protocol.json',{'created_at':int(time.time()),'source_sha256':source_hash(),
        'dataset_sha256':digest_file(path),'experts':EXPERTS,'learning_rate':30,'lookback_days':90,
        'rebias_period_days':30,'deadband':.25,'news_risk':'1+min(1,max(0,z)/3+max(0,change7)/3)',
        'candidates':4,'selection':'median six 75d returns minus half median drawdown; >=4 positive and >=12 sell fills',
        'development':'all previously viewed 540d; no held-out future data',
        'older_history':'sealed until selection; backward robustness with later policy choice bias',
        'exchange_writes':0})
    d=make_data(path,cache,news); start=d['times'][0]+90*86400; end=d['times'][-1]+3600
    c=Config(symbols=d['symbols'],interval=3600,horizon=72,strategy='forecast',max_holding_bars=2160,
       fee_bps=35,spread_bps=25,slippage_bps=10,stop_atr=6.,min_stop_fraction=.05,cooldown=72,
       entry_buffer_bars=24,min_samples=300,uncertainty_penalty=0.,edge_bps=0,settlement_buffer=7200)
    grids,curves=prepare_experts(d,c,start,end)
    candidates=[]
    for learned in (False,True):
        for use_news in (False,True):
            models={}; forecasts=stream(d,grids,curves,start,end,learned,use_news,models)
            blocks=[simulate(d,c,forecasts,t,min(end,t+75*86400))[0] for t in range(start,end,75*86400)]
            total=simulate(d,c,forecasts,start,end)[0]; returns=[comparable(r) for r in blocks]
            score=statistics.median(returns)-.5*statistics.median(r['max_drawdown_pct'] for r in blocks)
            eligible=sum(v>0 for v in returns)>=4 and sum(r['sell_fills'] for r in blocks)>=12
            row={'learned':learned,'news':use_news,'returns_pct':returns,'blocks':blocks,'total':total,
                 'score':score,'eligible':eligible,'monthly_weights':models}
            candidates.append(row)
            print(json.dumps({'learned':learned,'news':use_news,'total':total['net_return_pct'],
                              'returns':returns,'eligible':eligible}),flush=True)
    atomic_json(root/'candidates.json',candidates)
    eligible=[x for x in candidates if x['eligible']]
    best=max(eligible or candidates,key=lambda x:x['score'])
    selected={**best,'validation_eligible':bool(eligible),'frozen_at':int(time.time()),'config':asdict(c)}
    atomic_json(root/'selection.json',selected)
    forecasts=stream(d,grids,curves,start,end,best['learned'],best['news'])
    simulate(d,c,forecasts,start,end,root/'development')
    stress=simulate(d,replace(c,fee_bps=70,spread_bps=50,slippage_bps=20),forecasts,start,end,root/'cost_x2')[0]
    atomic_json(root/'stress.json',stress)
    atomic_json(root/'model.json',model(d,curves,end,best['learned'],best['news'],bool(eligible)))
    print(json.dumps({'selected_learned':best['learned'],'selected_news':best['news'],
                     'selected_return':best['total']['net_return_pct'],'cost_x2_return':stress['net_return_pct']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--news',required=True)
    p.add_argument('--out',required=True); p.add_argument('--cache',required=True)
    a=p.parse_args(); study(a.data,a.news,a.out,a.cache)
