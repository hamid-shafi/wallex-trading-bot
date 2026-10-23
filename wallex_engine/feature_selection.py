"""Declared small feature family with costs, concentration and selection diagnostics."""
import argparse
from dataclasses import asdict,replace
from datetime import datetime
from itertools import product
from pathlib import Path
import json
import math
import shutil
import statistics
import time

from .audit import atomic_json,source_hash
from .config import Config
from .data import digest_file
from .experiment import simulate
from .feature_study import feature_contexts,augment
from .mixture_study import stream,model
from .selection_stats import daily_returns,episodes,cscv,family_bootstrap
from .slow_research import signals
from .mixture import EXPERTS
from .trend_study import make_data

import numpy as np


def annual_growth(report):
    if not report['settled_to_tmn'] or report['net_return_pct'] is None:
        return -100.
    parse=lambda s:datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()
    days=(parse(report['end'])-parse(report['start']))/86400
    return 36500*math.log1p(report['net_return_pct']/100)/days


def assess(rows):
    normal=[r['normal'] for r in rows]; stress=[r['cost_x2'] for r in rows]
    all_reports=normal+stress
    failures=sum(not r['settled_to_tmn'] for r in all_reports)
    profitable=all(r['settled_to_tmn'] and (r['net_return_pct'] or 0)>0 for r in all_reports)
    maximum_dd=max(r['max_drawdown_pct'] for r in all_reports)
    closed=[p for r in rows for p in r['episodes']]
    pnl=sum(r['final_cash_tmn']-r['initial_cash_tmn'] for r in normal)
    largest=max((p['pnl'] for p in closed),default=0.)
    share=largest/pnl if pnl>0 else 1e6
    best_removed=[]
    for r in rows:
        best=max((p['pnl'] for p in r['episodes']),default=0.)
        best_removed.append(r['normal']['final_cash_tmn']-r['normal']['initial_cash_tmn']-best)
    blocks=[b for r in rows for b in r['blocks']]
    active=[b for b in blocks if b['sell_fills']>0]
    positive=sum(b['settled_to_tmn'] and (b['net_return_pct'] or 0)>0 for b in active)
    positive_rate=positive/max(1,len(active))
    worst_block=min((b['net_return_pct'] if b['settled_to_tmn'] else -100 for b in blocks),default=-100.)
    worst_growth=min(annual_growth(r) for r in all_reports)
    score=worst_growth-.5*maximum_dd-.25*max(0,-worst_block)-5*max(0,share-.5)
    passes=(profitable and maximum_dd<=5.5 and len(closed)>=40 and
            all(len(r['episodes'])>=5 for r in rows) and share<=.5 and
            sum(p>0 for p in best_removed)>=max(1,len(rows)-1) and positive_rate>=.5)
    return {'score':score,'strict_criteria_passed':passes,'settlement_failures':failures,
        'all_normal_and_double_cost_positive':profitable,'worst_drawdown_pct':maximum_dd,
        'closed_episodes':len(closed),'largest_episode_share_of_combined_net_pnl':share,
        'positive_active_block_fraction':positive_rate,'active_blocks':len(active),
        'positive_active_blocks':positive,'net_pnl_without_best_episode_by_era':best_removed,
        'minimum_annualized_log_growth_pct':worst_growth,'worst_75d_return_pct':worst_block}


def study(base,out):
    root=Path(out); root.mkdir(parents=True,exist_ok=True); base=Path(base)
    if (root/'selection.json').exists():
        raise ValueError('selection already frozen; do not retune')
    prior=base/'runs/adaptive-research-20261022'
    c=Config(**json.loads((prior/'feature-study/selection.json').read_text())['config'])
    specs=[]
    for news,vol,breadth,liq,eff in product((False,True),repeat=5):
        features=[n for n,flag in zip(('volatility','breadth','liquidity','efficiency'),(vol,breadth,liq,eff)) if flag]
        specs.append({'id':f'n{int(news)}v{int(vol)}b{int(breadth)}l{int(liq)}e{int(eff)}',
                      'news':news,'features':features})
    protocol={'declared_at':int(time.time()),'source_sha256':source_hash(),'candidate_count':len(specs),
        'candidates':specs,'fixed_config':asdict(c),'prior_trials_lower_bound':226,
        'data_role':'ALL eras are previously inspected development; no new out-of-sample claim',
        'new_hypothesis':'168-hour absolute log-return displacement / path length >=0.20; entry filter only',
        'fixed_thresholds':{'breadth':.6,'efficiency':.2,'volatility_cap':2,'liquidity_multiple':2},
        'strict_criteria':{'all_era_net_returns_positive_at_base_and_double_cost':True,
            'all_era_settled_to_TMN':True,'max_observed_drawdown_pct':5.5,
            'total_closed_episodes':40,'minimum_episodes_per_era':5,'max_largest_episode_share':.5,
            'positive_excluding_best_trade_in_at_least_two_eras':True,'positive_active_75d_block_fraction':.5},
        'ranking':'minimum annualized log growth across eras/costs - 0.5 worst drawdown - 0.25 worst block loss - 5 excess concentration over 50pct',
        'evaluation_blocks':'75 days, remainder included only if at least 45 days; fresh account per block',
        'diagnostics':'12-block retrospective CSCV; 2000 within-era 7-day paired centered maximum-t bootstrap; seed73',
        'publication_rule':'strict failures remain visible; cash baseline; paper-only even if historical criteria pass',
        'reference':'Bailey and Lopez de Prado 2014 deflated-sharpe.pdf; methodology inspiration, not a DSR implementation',
        'exchange_writes':0,'account_reads':0}
    atomic_json(root/'protocol.json',protocol)
    shutil.copytree(base/'wallex_engine',root/'source_snapshot',ignore=shutil.ignore_patterns('__pycache__'))
    datasets=[('2022_2024',prior/'third-market/hourly.csv',prior/'third-features.npz'),
              ('2024_2025',prior/'older-market/hourly.csv',prior/'older-features.npz'),
              ('2025_2026',base/'runs/trend-research-20261022/market/hourly.csv',base/'runs/trend-research-20261022/features-72h.npz')]
    all_rows={s['id']:[] for s in specs}; matrices=[]; last_data=None
    for era,path,cache in datasets:
        d=make_data(path,cache,str(prior/'news/all-periods.jsonl'))
        start=d['times'][0]+90*86400; end=d['times'][-1]+3600
        contexts=feature_contexts(d); grids=[signals(d,**spec) for spec in EXPERTS]
        streams={flag:stream(d,grids,[[]]*8,start,end,False,flag) for flag in (False,True)}
        returns=[]
        atomic_json(root/era/'data-provenance.json',{'path':str(path),'sha256':digest_file(path),
            'news_sha256':digest_file(prior/'news/all-periods.jsonl'),'news_coverage_fraction':d['news_coverage_fraction'],
            'start':start,'end':end})
        for spec in specs:
            amended=augment(streams[spec['news']],spec['features'],contexts,c.min_notional)
            capture={}; normal,curve=simulate(d,c,amended,start,end,capture=capture)
            closed=episodes(capture['fills'])
            stress=simulate(d,replace(c,fee_bps=70,spread_bps=50,slippage_bps=20,max_spread_bps=70),amended,start,end)[0]
            blocks=[simulate(d,c,amended,t,min(end,t+75*86400))[0] for t in range(start,end,75*86400)
                    if end-t>=45*86400]
            row={'era':era,'spec':spec,'normal':normal,'cost_x2':stress,'episodes':closed,'blocks':blocks}
            all_rows[spec['id']].append(row); returns.append(daily_returns(curve))
            atomic_json(root/era/(spec['id']+'.json'),row)
            print(json.dumps({'era':era,'id':spec['id'],'return':normal['net_return_pct'],
                 'stress':stress['net_return_pct'],'settled_stress':stress['settled_to_tmn']}),flush=True)
        matrices.append(np.column_stack(returns)); last_data=d
    rows=[{'spec':s,'assessment':assess(all_rows[s['id']]),'eras':all_rows[s['id']]} for s in specs]
    eligible=[r for r in rows if r['assessment']['strict_criteria_passed']]
    best=max(eligible or rows,key=lambda r:r['assessment']['score'])
    maxprofit=max(rows,key=lambda r:statistics.median(annual_growth(e['normal']) for e in r['eras']))
    atomic_json(root/'candidates.json',rows)
    frozen={'spec':best['spec'],'assessment':best['assessment'],'strict_criteria_passed':bool(eligible),
        'frozen_at':int(time.time()),'config':asdict(c),'research_only':True,'prospective_validated':False,
        'best_normal_profit_spec':maxprofit['spec'],'eligible_count':len(eligible),
        'protocol_sha256':digest_file(root/'protocol.json')}
    atomic_json(root/'selection.json',frozen)
    groups=[]
    for held in range(3):
        winner=max(rows,key=lambda r:assess([e for i,e in enumerate(r['eras']) if i!=held])['score'])
        groups.append({'held_era':datasets[held][0],'selected_from_other_eras':winner['spec'],
                       'held_normal_return_pct':winner['eras'][held]['normal']['net_return_pct'],
                       'held_double_cost_return_pct':winner['eras'][held]['cost_x2']['net_return_pct']})
    diagnostics={'cscv':cscv(np.concatenate(matrices)), 'family_bootstrap':family_bootstrap(matrices),
        'retrospective_leave_one_era_out':groups,'not_a_causal_forward_test':True,
        'dsr_not_claimed':'prior trials have incompatible histories and dependent choices; independent-trial count is unknown'}
    atomic_json(root/'diagnostics.json',diagnostics)
    cutoff=last_data['times'][-1]+3600
    artifact=model(last_data,[[]]*8,cutoff,False,best['spec']['news'],True)
    artifact.update({'risk_features':best['spec']['features'],'min_notional':c.min_notional,
         'paper_eligible':True,'historical_strict_criteria_passed':bool(eligible),'prospective_validated':False,
         'selection_sha256':digest_file(root/'selection.json'),'selection_scope':'retrospective research candidate only'})
    atomic_json(root/'model.json',artifact)
    paper=replace(c,model_artifact=str(root/'model.json'),news_path=str(prior/'news/all-periods.jsonl'),news_required=best['spec']['news'])
    atomic_json(root/'paper-config.json',asdict(paper))
    # Reproduce selected paths with full audit evidence after freezing the choice.
    for era,path,cache in datasets:
        d=make_data(path,cache,str(prior/'news/all-periods.jsonl')); start=d['times'][0]+90*86400; end=d['times'][-1]+3600
        contexts=feature_contexts(d); grids=[signals(d,**spec) for spec in EXPERTS]
        forecasts=augment(stream(d,grids,[[]]*8,start,end,False,best['spec']['news']),best['spec']['features'],contexts,c.min_notional)
        check=simulate(d,c,forecasts,start,end,root/'selected'/era)[0]
        original=next(r for r in best['eras'] if r['era']==era)['normal']
        assert check['net_return_pct']==original['net_return_pct']
    print(json.dumps({'selected':best['spec'],'strict_passed':bool(eligible),'eligible_count':len(eligible),
        'max_normal_profit':maxprofit['spec'],'pbo':diagnostics['cscv'].get('pbo')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--base',default='.'); p.add_argument('--out',required=True)
    a=p.parse_args(); study(Path(a.base).resolve(),a.out)
