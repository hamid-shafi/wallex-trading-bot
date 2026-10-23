"""Bounded technical-filter ablation; recent failed day excluded from selection."""
from pathlib import Path
from dataclasses import asdict,replace
import argparse
import json
import math
import shutil
import statistics
import time
import numpy as np

from .audit import atomic_json,source_hash
from .data import aggregate,digest_file
from .experiment import simulate
from .feature_selection import annual_growth
from .longterm_study import build_data,configuration
from .longterm import SYMBOLS,decision
from .technical import indicators,context
from .selection_stats import episodes,daily_returns,family_bootstrap

RECENT_DAY_START=1792652400
SPECS=[
    {'id':'previous_moderate','family':'committee','news':True,'risk_fraction':.0075},
    {'id':'trend_risk_1pct','family':'committee','news':True,'risk_fraction':.01},
    *[{'id':name,'family':'technical_committee','filters':filters,'news':True,'risk_fraction':.01}
      for name,filters in [
        ('trend_quality',['quality']),
        ('macd_rsi',['macd_rsi']),
        ('candle_veto',['candles']),
        ('cost_regime',['cost_regime']),
        ('cost_regime_candles',['cost_regime','candles']),
        ('combined_filters',['quality','macd_rsi','candles','cost_regime'])]]]


def config(spec):
    return replace(configuration(spec,'moderate'),risk_fraction=spec['risk_fraction'],
        portfolio_stop_risk=.0225 if spec['risk_fraction']==.0075 else .03).validate()


def technical_arrays(d):
    arrays={}; times=np.asarray(d['times']); checked=0
    for j,s in enumerate(SYMBOLS):
        hourly=[f[s] for f in d['frames']]
        six=aggregate(hourly,21600,3600)
        close_times=np.asarray([b.ts+21600 for b in six])
        rows=[indicators(six[max(0,i-99):i+1]) for i in range(len(six))]
        keys=set().union(*(r.keys() for r in rows))
        out={k:np.zeros(len(times),dtype=bool if k.endswith('_6h') and any(t in k for t in ('engulfing','star','hammer','doji')) or k=='technical_available' else float) for k in keys}
        selected=np.searchsorted(close_times,times+3600,side='right')-1
        for i,k in enumerate(selected):
            if k<0: continue
            for key,value in rows[int(k)].items(): out[key][i]=value
        arrays[s]=out
        # Check exported portable path against batch features at predeclared prefixes.
        valid=np.flatnonzero(d['valid'][:,j])
        for i in valid[::max(1,len(valid)//12)]:
            live=context(hourly[max(0,i-720):i+1],int(times[i])+3600)
            for key,value in live.items():
                if not math.isclose(float(value),float(out[key][i]),abs_tol=1e-10,rel_tol=1e-10):
                    raise ValueError(f'technical batch/portable mismatch: {s} {i} {key}')
            checked+=1
    return arrays,checked


def grid(d,technical,spec,start,end):
    result={}
    for n in np.flatnonzero((d['row_ts']>=start)&(d['row_ts']<end)):
        i,j=int(d['ii'][n]),int(d['jj'][n]); s=SYMBOLS[j]
        aux={k:v[i,j] for k,v in d['aux'].items()}
        aux.update({k:v[i] for k,v in technical[s].items()})
        aux['raw_momentum_168']=aux['momentum_168']
        aux['breadth_720']=float(np.mean(d['aux']['above_long'][i]))
        u,atr,parts=decision(spec,d['x'][n],aux)
        result[(i,s)]=(u,0.,float(atr),1000,spec['id'],parts)
    return result


def summarize(rows):
    reports=[r[k] for r in rows for k in ('normal','cost_x2')]
    valid=all(r['settled_to_tmn'] and r['net_return_pct'] is not None for r in reports)
    annual=[annual_growth(r['cost_x2']) for r in rows]
    dd=max(r['max_drawdown_pct'] for r in reports)
    closed=[p for r in rows for p in r['episodes']]
    blocks=[b for r in rows for b in r['blocks'] if b['fills']>0]
    positive=sum(b['settled_to_tmn'] and (b['net_return_pct'] or 0)>0 for b in blocks)/max(1,len(blocks))
    passed=valid and all((r['net_return_pct'] or 0)>0 for r in reports) and dd<=15 and all(len(r['episodes'])>=8 for r in rows) and len(closed)>=16 and positive>=.55
    return {'development_passed':passed,'minimum_double_cost_annualized_log_growth_pct':min(annual),
        'maximum_drawdown_pct':dd,'closed_episodes':len(closed),'positive_active_90d_block_fraction':positive,
        'score':min(annual)-.3*dd+positive,
        'all_normal_and_stress_positive_and_settled':valid and all((r['net_return_pct'] or 0)>0 for r in reports)}


def evaluate(d,technical,spec,era,start,end,out):
    c=config(spec); forecasts=grid(d,technical,spec,start,end); capture={}
    normal,curve=simulate(d,c,forecasts,start,end,capture=capture)
    stress=simulate(d,replace(c,fee_bps=70,spread_bps=50,slippage_bps=20,max_spread_bps=70),forecasts,start,end)[0]
    blocks=[simulate(d,c,forecasts,t,min(end,t+90*86400))[0] for t in range(start,end,90*86400) if end-t>=60*86400]
    row={'era':era,'normal':normal,'cost_x2':stress,'blocks':blocks,'episodes':episodes(capture['fills'])}
    atomic_json(out/'candidates'/spec['id']/(era+'.json'),row)
    atomic_json(out/'curves'/spec['id']/(era+'.json'),curve)
    atomic_json(out/'trades'/spec['id']/(era+'.json'),capture['fills'])
    print(json.dumps({'candidate':spec['id'],'era':era,'return':normal['net_return_pct'],
        'cost_x2':stress['net_return_pct'],'dd':normal['max_drawdown_pct'],'episodes':len(row['episodes'])}),flush=True)
    return row,curve


def run(base,out):
    base=Path(base).resolve(); out=Path(out).resolve(); out.mkdir(parents=True,exist_ok=False)
    previous=json.loads((base/'evidence/longterm/protocol.json').read_text())
    eras=[(label,start,min(end,RECENT_DAY_START)) for label,start,end in previous['eras']]
    prior=base/'runs/adaptive-research-20261022'
    sources=[(prior/'third-market/hourly.csv',prior/'third-features.npz'),
        (prior/'older-market/hourly.csv',prior/'older-features.npz'),
        (base/'runs/trend-research-20261022/market/hourly.csv',base/'runs/trend-research-20261022/features-72h.npz')]
    news=prior/'news/all-periods.jsonl'
    atomic_json(out/'protocol.json',{'declared_at':int(time.time()),'source_sha256':source_hash(),
        'candidates':SPECS,'configurations':{s['id']:asdict(config(s)) for s in SPECS},
        'development_eras':eras[:2],'retrospective_audit_era':eras[2],
        'all_data_previously_viewed':True,'independent_final_holdout':False,
        'prior_trials_lower_bound':269,'current_family_size':len(SPECS),
        'recent_day_excluded_at_or_after':RECENT_DAY_START,'no_model_training':True,
        'selection':'choose only from first two eras; freeze before third; no replacement using audit results',
        'promotion':'development passed, audit normal/stress positive and settled, each drawdown<=15, audit>=8 closed episodes; >10% historical log-growth improvement vs baseline; no more than one era underperforms baseline',
        'technical':'100 complete 6h candles; RSI14 Wilder, MACD12/26 signal9; deterministic engulfing/shooting star veto; no intrabar pattern guessing',
        'candle_filter':'veto new entry only; bearish trend exit preserved; positive candle is not standalone buy advice',
        'cost_regime':'past 7d log momentum>=0.023 and >=60% markets above 30d mean; heuristic proxy, not expected net return',
        'news':'archived GPR vintages plus assumed 48h delay for old historical replay; original local receipt unknown',
        'research_data_hashes':{str(p):digest_file(p) for p,_ in sources},'news_sha256':digest_file(news),
        'exchange_writes':0,'account_reads':0,'credentials_used':False,
        'references':['https://fairmodel.econ.yale.edu/ec439/mosk.pdf','https://www.nber.org/system/files/working_papers/w22208/w22208.pdf','https://www.carmamaths.org/resources/jon/backtest2.pdf']})
    shutil.copytree(base/'wallex_engine',out/'source_snapshot',ignore=shutil.ignore_patterns('__pycache__'))
    d=build_data(sources,str(news)); technical,parity=technical_arrays(d)
    atomic_json(out/'data-quality.json',{'common_hours':len(d['times']),'feature_rows':len(d['x']),
        'portable_batch_prefix_checks':parity,'signals_require_721_contiguous_hours':True,
        'no_forward_fill':True,'evaluated_end_exclusive':RECENT_DAY_START})
    rows=[]; matrices=[[],[]]
    for spec in SPECS:
        results=[]
        for ei,(era,start,end) in enumerate(eras[:2]):
            result,curve=evaluate(d,technical,spec,era,start,end,out)
            results.append(result); matrices[ei].append(daily_returns(curve))
        rows.append({'spec':spec,'development':results,'assessment':summarize(results)})
    atomic_json(out/'development-results.json',rows)
    eligible=[r for r in rows if r['assessment']['development_passed']]
    winner=max(eligible or rows,key=lambda r:r['assessment']['score'])
    selection={'spec':winner['spec'],'assessment':winner['assessment'],'frozen_at':int(time.time()),
        'development_eligible':len(eligible),'selected_before_audit':True,
        'independent_holdout':False,'protocol_sha256':digest_file(out/'protocol.json')}
    atomic_json(out/'selection-before-audit.json',selection)
    # Audit exactly the preselected winner and two declared controls. No reselection.
    ids={winner['spec']['id'],'previous_moderate','trend_risk_1pct'}; audit={}
    for spec in SPECS:
        if spec['id'] in ids:
            audit[spec['id']]=evaluate(d,technical,spec,*eras[2],out)[0]
    atomic_json(out/'audit-results.json',audit)
    base_row=next(r for r in rows if r['spec']['id']=='previous_moderate')
    all_winner=winner['development']+[audit[winner['spec']['id']]]
    all_base=base_row['development']+[audit['previous_moderate']]
    rate=lambda values:sum(math.log1p(r['normal']['net_return_pct']/100) for r in values) if all(r['normal']['settled_to_tmn'] for r in values) else -math.inf
    gain=rate(all_winner); old=rate(all_base)
    hold=audit[winner['spec']['id']]
    audit_pass=all(hold[k]['settled_to_tmn'] and (hold[k]['net_return_pct'] or 0)>0 and hold[k]['max_drawdown_pct']<=15 for k in ('normal','cost_x2')) and len(hold['episodes'])>=8
    beats=sum((w['normal']['net_return_pct'] or -1e6)>=(b['normal']['net_return_pct'] or -1e6) for w,b in zip(all_winner,all_base))
    improved=gain>1.1*old and beats>=2
    promote=winner['assessment']['development_passed'] and audit_pass and improved and winner['spec']['id']!='previous_moderate'
    days=sum((end-start)/86400 for _,start,end in eras)
    diagnostics=family_bootstrap([np.column_stack(m) for m in matrices])
    atomic_json(out/'diagnostics.json',{'development_family_bootstrap':diagnostics,'prior_trials_not_fully_corrected':True,'future_profit_probability_claimed':False})
    summary={'preselected':winner['spec'],'development_passed':winner['assessment']['development_passed'],
        'retrospective_audit_passed':audit_pass,'historical_improvement_gate_passed':improved,
        'research_promotion_allowed':promote,'live_promotion_allowed':False,
        'independent_final_holdout':False,'prospective_validated':False,
        'winner_normal_returns_pct':[r['normal']['net_return_pct'] for r in all_winner],
        'winner_double_cost_returns_pct':[r['cost_x2']['net_return_pct'] for r in all_winner],
        'baseline_normal_returns_pct':[r['normal']['net_return_pct'] for r in all_base],
        'winner_weekly_equivalent_pct':100*math.expm1(gain*7/days) if math.isfinite(gain) else None,
        'baseline_weekly_equivalent_pct':100*math.expm1(old*7/days) if math.isfinite(old) else None,
        'selected_before_audit_sha256':digest_file(out/'selection-before-audit.json'),
        'recent_day_not_used_for_selection':True,'source_sha256':source_hash()}
    cutoff=RECENT_DAY_START
    artifact={'artifact_type':'longterm_research','feature_names':__import__('wallex_engine.longterm',fromlist=['NAMES']).NAMES,
        'symbols':SYMBOLS,'horizon':72,'interval':3600,'spec':winner['spec'],'fit':None,
        'training_cutoff':cutoff,'valid_until':int(time.time())+30*86400,
        'research_only':True,'paper_eligible':promote,'historical_strict_criteria_passed':False,
        'prospective_validated':False,'created_at':int(time.time()),
        'selection_sha256':digest_file(out/'selection-before-audit.json')}
    atomic_json(out/'model.json',artifact)
    c=replace(config(winner['spec']),model_artifact=str(out/'model.json'),news_path=str(news),news_required=True)
    atomic_json(out/'paper-config.json',asdict(c)); atomic_json(out/'summary.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--base',default='.');p.add_argument('--out',required=True)
    a=p.parse_args();run(a.base,a.out)
