"""Paired feature ablation before opening newly acquired older market prices."""
import argparse
from dataclasses import asdict,replace
from pathlib import Path
import json
import shutil
import statistics
import time

from .audit import atomic_json,source_hash
from .config import Config
from .data import digest_file
from .experiment import simulate,comparable
from .mixture_study import stream,prepare_experts,model
from .trend_study import make_data
from .regime import context,adjust


def feature_contexts(data):
    histories={s:[] for s in data['symbols']}; result={}
    for i,frame in enumerate(data['frames']):
        for s,b in frame.items():
            histories[s].append(b)
        for s in data['symbols']:
            result[(i,s)]=context(histories,s)
    return result


def augment(forecasts,features,contexts,min_notional=100000.):
    output={}
    for key,value in forecasts.items():
        u,unc,atr,samples,digest,parts=value
        u,atr=adjust(u,atr,features,contexts.get(key),min_notional)
        output[key]=(u,unc,atr,samples,digest,{**parts,'regime_features':contexts.get(key),'risk_features':features})
    return output


def study(path,news,parent,out,cache):
    root=Path(out); root.mkdir(parents=True,exist_ok=True)
    if (root/'selection.json').exists():
        raise ValueError('feature selection frozen')
    selected=json.loads((Path(parent)/'selection.json').read_text()); c=Config(**selected['config'])
    options=[[],['volatility'],['breadth'],['liquidity'],['volatility','breadth'],['volatility','breadth','liquidity']]
    atomic_json(root/'protocol.json',{'created_at':int(time.time()),'source_sha256':source_hash(),
        'data_sha256':digest_file(path),'parent_selection_sha256':digest_file(Path(parent)/'selection.json'),
        'options':options,'selection':'same median return minus half drawdown; >=4/6 positive, >=12 sell fills',
        'thresholds':{'breadth':.6,'vol_multiplier_cap':2,'liquidity_min_multiple':2},
        'older_market_status':'not opened for policy tuning; only counts/quality were inspected',
        'future_profitability_proven':False,'exchange_writes':0})
    shutil.copytree(Path(__file__).parent,root/'source_snapshot',ignore=shutil.ignore_patterns('__pycache__'))
    d=make_data(path,cache,news); start=d['times'][0]+90*86400; end=d['times'][-1]+3600
    grids,curves=prepare_experts(d,c,start,end)
    forecasts=stream(d,grids,curves,start,end,selected['learned'],selected['news'])
    contexts=feature_contexts(d); candidates=[]
    for features in options:
        amended=augment(forecasts,features,contexts,c.min_notional)
        blocks=[simulate(d,c,amended,t,min(end,t+75*86400))[0] for t in range(start,end,75*86400)]
        total=simulate(d,c,amended,start,end)[0]; returns=[comparable(r) for r in blocks]
        score=statistics.median(returns)-.5*statistics.median(r['max_drawdown_pct'] for r in blocks)
        eligible=sum(v>0 for v in returns)>=4 and sum(r['sell_fills'] for r in blocks)>=12
        row={'features':features,'blocks':blocks,'returns_pct':returns,'total':total,'score':score,'eligible':eligible}
        candidates.append(row)
        print(json.dumps({'features':features,'total':total['net_return_pct'],'drawdown':total['max_drawdown_pct'],
                           'returns':returns,'eligible':eligible}),flush=True)
    atomic_json(root/'candidates.json',candidates)
    eligible=[r for r in candidates if r['eligible']]; best=max(eligible or candidates,key=lambda x:x['score'])
    frozen={**best,'config':asdict(c),'learned':selected['learned'],'news':selected['news'],
            'validation_eligible':bool(eligible),'frozen_at':int(time.time()),'selection_rule':'declared block score'}
    atomic_json(root/'selection.json',frozen)
    final=model(d,curves,end,selected['learned'],selected['news'],bool(eligible))
    final['risk_features']=best['features']; final['min_notional']=c.min_notional
    atomic_json(root/'model.json',final)
    amended=augment(forecasts,best['features'],contexts,c.min_notional)
    simulate(d,c,amended,start,end,root/'development')
    # Retain the original spread guard result and separately permit stressed spreads.
    simulate(d,replace(c,fee_bps=70,spread_bps=50,slippage_bps=20),amended,start,end,root/'cost_x2_guard_kept')
    simulate(d,replace(c,fee_bps=70,spread_bps=50,slippage_bps=20,max_spread_bps=70),amended,start,end,root/'cost_x2_tradable')
    print(json.dumps({'selected_features':best['features'],'total':best['total']['net_return_pct']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--news',required=True)
    p.add_argument('--parent',required=True); p.add_argument('--out',required=True); p.add_argument('--cache',required=True)
    a=p.parse_args(); study(a.data,a.news,a.parent,a.out,a.cache)
