"""Bounded, causal long-horizon comparisons on explicitly reused history."""
import argparse
from dataclasses import asdict,replace
from pathlib import Path
import json
import math
import shutil
import statistics
import time

import numpy as np
import pandas as pd

from .audit import atomic_json,source_hash
from .config import Config
from .data import load_bars,digest_file,save_bars
from .experiment import simulate,grow_tree,tree_predict,predict,iso
from .feature_selection import annual_growth
from .longterm import SYMBOLS,NAMES,EXTRA_NAMES,decision
from .mixture import EXPERTS
from .news import NewsIndex
from .selection_stats import daily_returns,episodes,cscv,family_bootstrap
from .trained import FEATURE_NAMES,trained_features
from .trend_study import load_feature_cache


SPECS=[
    {'id':'committee_original_risk','family':'committee','news':True,'liquidity':False,'original_risk':True},
    {'id':'committee_matched','family':'committee','news':True},
    {'id':'slow_momentum_news','family':'momentum','news':True},
    {'id':'slow_momentum_no_news','family':'momentum','news':False},
    {'id':'channel_breakout','family':'breakout','news':True},
    {'id':'relative_strength','family':'relative','news':True},
    {'id':'trend_pullback','family':'pullback','news':True},
    *[{'id':f'{family}_{h}h','family':family,'target_hours':h,'news':True}
      for family in ('ridge','boost') for h in (168,336)]]


def build_data(sources,news_path):
    by={s:{} for s in SYMBOLS}; cached=[]; conflicts=[]
    for path,cache in sources:
        bars=load_bars(path)
        for b in bars:
            if b.symbol not in by: continue
            old=by[b.symbol].get(b.ts)
            if old is not None and old!=b: conflicts.append({'symbol':b.symbol,'ts':b.ts})
            by[b.symbol][b.ts]=b
        if cache and Path(cache).exists():
            saved=load_feature_cache(cache,path,SYMBOLS)
            cached.append({k:saved[k] for k in ('x','atr','row_ts','locations')})
    times=np.asarray(sorted(set.intersection(*(set(v) for v in by.values()))),dtype=np.int64)
    lookup={int(t):i for i,t in enumerate(times)}
    fs=[{s:by[s][int(t)] for s in SYMBOLS} for t in times]
    arrays={name:np.asarray([[getattr(f[s],name) for s in SYMBOLS] for f in fs])
            for name in ('open','high','low','close','volume')}
    n=len(times); m=len(SYMBOLS); close=arrays['close']; log=np.log(close)
    r=np.vstack((np.zeros((1,m)),np.diff(log,axis=0)))
    rv=lambda length:pd.DataFrame(r).rolling(length,min_periods=length).std(ddof=0).to_numpy()
    mean=lambda length:pd.DataFrame(close).rolling(length,min_periods=length).mean().to_numpy()
    vol=np.maximum(.0003,rv(168)); v24=rv(24); v720=rv(720)
    means={v:mean(v) for v in (24,72,168,336,720)}
    lag=lambda length:np.vstack((np.full((length,m),np.nan),log[length:]-log[:-length]))
    mom7=lag(168); mom30=lag(720); median=np.median(mom7,axis=1)[:,None]
    high720=pd.DataFrame(arrays['high']).shift(1).rolling(720,min_periods=720).max().to_numpy()
    low168=pd.DataFrame(arrays['low']).shift(1).rolling(168,min_periods=168).min().to_numpy()
    maxclose=pd.DataFrame(close).rolling(720,min_periods=720).max().to_numpy()
    breadth=np.mean(close>means[720],axis=1)[:,None]*np.ones((1,m))
    path=pd.DataFrame(np.abs(r)).rolling(168,min_periods=168).sum().to_numpy()
    displacement=pd.DataFrame(r).rolling(168,min_periods=168).sum().to_numpy()
    efficiency=np.abs(displacement)/np.maximum(1e-12,path)
    liquidity=pd.DataFrame(close*arrays['volume']).rolling(24,min_periods=24).median().to_numpy()*.01
    contiguous=np.zeros(n,dtype=bool); contiguous[720:]=(times[720:]-times[:-720])==720*3600
    base=np.full((n,m,len(FEATURE_NAMES)+m),np.nan); atr=np.full((n,m),np.nan)
    for saved in cached:
        ii=np.asarray([lookup[int(t)-3600] for t in saved['row_ts']]); jj=saved['locations'][:,1].astype(int)
        base[ii,jj]=saved['x']; atr[ii,jj]=saved['atr']
    missing=np.argwhere(contiguous[:,None]&~np.isfinite(base[:,:,0]))
    for i,j in missing:
        result=trained_features([f[SYMBOLS[j]] for f in fs[i-168:i+1]],72)
        if result:
            x,_,tr=result; base[i,j]=x+[float(j==k) for k in range(m)]; atr[i,j]=tr
    extras=np.stack((np.clip(lag(336)/(vol*math.sqrt(336)),-5,5),
        np.clip(mom30/(vol*math.sqrt(720)),-5,5),v720*100,
        np.clip(np.log(close/maxclose)/(vol*math.sqrt(720)),-5,5),
        np.clip(np.log(means[72]/means[720])/(vol*math.sqrt(720)),-5,5),
        np.clip((mom7-median)/(vol*math.sqrt(168)),-5,5),
        np.clip(np.broadcast_to(median,mom7.shape)/(vol*math.sqrt(168)),-5,5),breadth,efficiency),axis=2)
    index=NewsIndex(news_path); news=np.asarray([index.at(int(t)+3600,reconstructed=True)[0] for t in times])
    x=np.concatenate((base,extras,np.broadcast_to(news[:,None,:],(n,m,6))),axis=2)
    valid=contiguous[:,None]&np.isfinite(x).all(axis=2)&np.isfinite(atr)
    ii,jj=np.nonzero(valid)
    strength=np.zeros((n,m))
    for spec in EXPERTS:
        fast=means[spec['fast']]; slow=means[spec['slow']]; ratio=fast/slow-1
        strength+=(np.where((ratio>spec['band'])&(close>fast),1,np.where(ratio<-spec['band'],-1,0)))/9
    committee=np.where(strength>.25,.03+.02*strength,np.where(strength<-.25,-.03,0.))
    aux={'atr':atr,'volatility_ratio':v24/vol,'committee':committee,'strength':strength,
         'trend_entry':(close>means[72])&(means[72]>means[720]),'trend_exit':means[72]<means[720],
         'above_long':close>means[720],'momentum_168':mom7,'momentum_720':mom30,
         'relative_168':extras[:,:,5],'z72':base[:,:,19],
         'breakout_entry':close>high720,'breakout_exit':close<low168,'liquidity':liquidity}
    return {'times':times.tolist(),'frames':fs,'symbols':SYMBOLS,'x':x[ii,jj],
        'ii':ii,'jj':jj,'row_ts':times[ii]+3600,'valid':valid,'aux':aux,'arrays':arrays,
        'news':news,'conflicts':conflicts,'atr':atr}


def labels(d,horizon):
    i=d['ii']; j=d['jj']; a=d['arrays']; times=np.asarray(d['times']); n=len(times)
    eligible=(i+horizon<n)&(i+1<n)
    end=np.minimum(n-1,i+horizon); nxt=np.minimum(n-1,i+1)
    eligible &= (times[end]-times[i]==horizon*3600)&(a['volume'][nxt,j]>0)
    entry=a['open'][nxt,j]*1.00225
    values=d['x'][:,-6:]; ratio=d['aux']['volatility_ratio'][i,j]
    multiplier=np.clip(ratio,1,2)*(1+np.minimum(1,np.maximum(0,values[:,2])/3+np.maximum(0,values[:,3])/3))
    stop=entry*(1-np.maximum(.06,6*d['atr'][i,j]*multiplier/a['close'][i,j]))
    exit_price=a['close'][end,j].copy(); stopped=np.zeros(len(i),dtype=bool)
    for k in range(1,horizon+1):
        step=np.minimum(n-1,i+k)
        hit=eligible&~stopped&(a['low'][step,j]<=stop)&(a['volume'][step,j]>0)
        exit_price[hit]=np.minimum(a['open'][step[hit],j[hit]],stop[hit]); stopped|=hit
    eligible &= stopped|(a['volume'][end,j]>0)
    y=exit_price*.99775*.9965/(entry*1.0035)-1
    y=np.clip(y,-.5,1.); y[~eligible]=np.nan
    return y,d['row_ts']+horizon*3600


def numeric_fit(x,y,weights,kind):
    if kind=='ridge':
        p=np.eye(x.shape[1])*100.; p[0,0]=1.
        w=np.linalg.solve(x.T@(x*weights[:,None])+p,x.T@(y*weights))
        return {'kind':'ridge','weights':w.tolist()}
    thresholds=[np.unique(np.quantile(x[:,j],np.linspace(.1,.9,9))) for j in range(x.shape[1])]
    bins=np.column_stack([np.searchsorted(t,x[:,j],side='left') for j,t in enumerate(thresholds)])
    bias=float(np.average(y,weights=weights)); fitted=np.full(len(y),bias); trees=[]
    for _ in range(24):
        tree=grow_tree(x,np.clip(y-fitted,-.5,.5),weights,bins,thresholds,2,min_leaf=100,l2=50.)
        fitted+=.05*tree_predict(tree,x); trees.append(tree)
    return {'kind':'boost','bias':bias,'learning_rate':.05,'trees':trees}


def fit(d,y,maturity,cutoff,kind):
    t=d['row_ts']; x=d['x']
    mature=np.isfinite(y)&(maturity<=cutoff)&(t%43200==0)
    # Choose the calibration window using sample availability only, before any
    # outcome-dependent model fitting or performance comparison.
    calibration_days=42
    calibration=mature&(t>=cutoff-calibration_days*86400)
    if calibration.sum()<80:
        calibration_days=84
        calibration=mature&(t>=cutoff-calibration_days*86400)
    fitting=mature&(maturity<=cutoff-calibration_days*86400)&(t>=cutoff-365*86400)
    if fitting.sum()<700 or calibration.sum()<80:
        return {'fit_valid':False,'training_cutoff':int(cutoff),'training_samples':int(fitting.sum()),
                'calibration_samples':int(calibration.sum()),'calibration_days':calibration_days,
                'reason':'insufficient matured history'}
    tx=x[fitting]; ty=y[fitting]; w=np.exp(-(cutoff-t[fitting])/(180*86400)*math.log(2))
    model=numeric_fit(tx,ty,w,kind)
    # Separate regularized classifier; the label is a counterfactual fixed-horizon trade.
    p=np.eye(tx.shape[1])*100.; p[0,0]=1.; beta=np.zeros(tx.shape[1]); target=(ty>0).astype(float)
    for _ in range(15):
        probability=1/(1+np.exp(-np.clip(tx@beta,-30,30)))
        curvature=np.maximum(.001,probability*(1-probability))*w
        step=np.linalg.solve(tx.T@(tx*curvature[:,None])+p,tx.T@((target-probability)*w)-p@beta)
        beta+=step
        if np.max(np.abs(step))<1e-7: break
    cx=x[calibration]; truth=(y[calibration]>0).astype(float)
    raw=predict(model,cx); bias=float(np.clip(np.mean(y[calibration]-raw),-.03,.03))
    logits=cx@beta; pbias=0.
    for _ in range(10):
        pp=1/(1+np.exp(-np.clip(logits+pbias,-30,30)))
        pbias=float(np.clip(pbias+np.sum(truth-pp)/max(.1,np.sum(pp*(1-pp))),-2,2))
    pp=1/(1+np.exp(-np.clip(logits+pbias,-30,30)))
    return {'fit_valid':True,'training_cutoff':int(cutoff),'training_samples':int(fitting.sum()),
        'calibration_samples':int(calibration.sum()),'return_model':model,'return_bias':bias,
        'return_rmse':float(np.sqrt(np.mean((y[calibration]-raw-bias)**2))),
        'probability_weights':beta.tolist(),'probability_bias':pbias,
        'calibration_brier':float(np.mean((pp-truth)**2)),
        'calibration_constant_brier':float(np.mean((np.mean(truth)-truth)**2)),
        'latest_fit_maturity':int(maturity[fitting].max()),
        'latest_calibration_maturity':int(maturity[calibration].max()),
        'calibration_days':calibration_days,
        'purged_calibration_boundary':int(cutoff-calibration_days*86400)}


def learned_predictions(d,start,end,out):
    result={}; all_models={}
    for h in (168,336):
        y,maturity=labels(d,h)
        for kind in ('ridge','boost'):
            key=f'{kind}_{h}h'; pred=np.full(len(y),np.nan); probability=np.zeros(len(y)); model_ids=np.full(len(y),-1)
            models={}
            for cutoff in range(start,end,30*86400):
                model=fit(d,y,maturity,cutoff,kind); models[cutoff]=model
                atomic_json(Path(out)/'fitted-models'/key/(str(cutoff)+'.json'),model)
                rows=(d['row_ts']>=cutoff)&(d['row_ts']<cutoff+30*86400)
                model_ids[rows]=cutoff
                if model['fit_valid']:
                    pred[rows]=predict(model['return_model'],d['x'][rows])+model['return_bias']
                    probability[rows]=1/(1+np.exp(-np.clip(d['x'][rows]@np.asarray(model['probability_weights'])+model['probability_bias'],-35,35)))
                print(json.dumps({'fit':key,'cutoff':cutoff,'valid':model['fit_valid'],'samples':model['training_samples']}),flush=True)
            result[key]=(pred,probability,model_ids); all_models[key]=models
    return result,all_models


def grid(d,spec,start,end,predictions,models):
    output={}; learned=spec['family'] in ('ridge','boost')
    if learned: pred,probability,model_ids=predictions[spec['id']]
    for n in np.flatnonzero((d['row_ts']>=start)&(d['row_ts']<end)):
        i,j=int(d['ii'][n]),int(d['jj'][n]); aux={k:v[i,j] for k,v in d['aux'].items()}
        x=d['x'][n]
        if not spec.get('news',True): x=x.copy(); x[-6:]=0
        model=models[spec['id']].get(int(model_ids[n])) if learned else None
        u,atr,parts=decision(spec,x,aux,model,float(pred[n]) if learned else None,float(probability[n]) if learned else None)
        output[(i,SYMBOLS[j])]=(u,0.,float(atr),1000,spec['id'],parts)
    return output


def configuration(spec,profile='conservative'):
    c=Config(symbols=SYMBOLS,interval=3600,horizon=72,min_samples=300,strategy='forecast',
        fee_bps=35,spread_bps=25,slippage_bps=10,edge_bps=0,uncertainty_penalty=0,
        stop_atr=6,min_stop_fraction=.06,max_holding_bars=2160,cooldown=168,entry_buffer_bars=24,
        settlement_buffer=7200,position_cap=.2,gross_cap=.6,risk_fraction=.005,
        portfolio_stop_risk=.015,max_positions=3,max_drawdown=.10,daily_loss=.02)
    if profile=='moderate': c=replace(c,gross_cap=.8,position_cap=.25,risk_fraction=.0075,portfolio_stop_risk=.0225,max_drawdown=.15)
    if spec.get('original_risk'):
        c=replace(c,min_stop_fraction=.05,risk_fraction=.003,portfolio_stop_risk=.008,
            gross_cap=.5,position_cap=.25,max_positions=2,max_drawdown=.05,cooldown=72)
    if spec['family']=='breakout': c=replace(c,trailing_atr=8.)
    if spec['family']=='pullback': c=replace(c,max_holding_bars=168)
    if spec['family'] in ('ridge','boost'): c=replace(c,max_holding_bars=spec['target_hours']*2)
    return c.validate()


def assess(rows,profile):
    reports=[r[k] for r in rows for k in ('normal','cost_x2')]
    settled=all(r['settled_to_tmn'] for r in reports)
    positive=settled and all((r['net_return_pct'] or 0)>0 for r in reports)
    annual=[annual_growth(r['normal']) for r in rows]
    dd=max(r['max_drawdown_pct'] for r in reports)
    normal=[r['normal'] for r in rows]; closed=[p for r in rows for p in r['episodes']]
    total=sum(r['final_cash_tmn']-r['initial_cash_tmn'] for r in normal)
    largest=max((p['pnl'] for p in closed),default=0.)
    concentration=largest/total if total>0 else 1e6
    without=[r['normal']['final_cash_tmn']-r['normal']['initial_cash_tmn']-max((p['pnl'] for p in r['episodes']),default=0.) for r in rows]
    blocks=[b for r in rows for b in r['blocks']]; active=[b for b in blocks if b['fills']>0]
    fraction=sum(b['settled_to_tmn'] and (b['net_return_pct'] or 0)>0 for b in active)/max(1,len(active))
    worst=min((b['net_return_pct'] if b['settled_to_tmn'] else -100. for b in blocks),default=-100.)
    passes=positive and dd<=(12 if profile=='conservative' else 17) and min(annual)>=5 and statistics.median(annual)>=15 and len(closed)>=50 and all(len(r['episodes'])>=8 for r in rows) and concentration<=.4 and sum(v>0 for v in without)>=2 and fraction>=.55
    score=min(annual)-.5*dd-.25*max(0,-worst)+2*fraction
    return {'strict_criteria_passed':passes,'score':score,'all_scenarios_settled':settled,
        'all_scenarios_positive':positive,'annualized_log_growth_pct':annual,'worst_drawdown_pct':dd,
        'closed_episodes':len(closed),'largest_episode_share_combined_net_pnl':concentration,
        'net_pnl_without_best_episode_by_era':without,'positive_active_90d_block_fraction':fraction,
        'active_blocks':len(active),'worst_90d_block_return_pct':worst}


def run(base,out,profile='conservative'):
    base=Path(base); out=Path(out); out.mkdir(parents=True,exist_ok=False)
    prior=base/'runs/adaptive-research-20261022'
    sources=[(prior/'third-market/hourly.csv',prior/'third-features.npz'),
        (prior/'older-market/hourly.csv',prior/'older-features.npz'),
        (base/'runs/trend-research-20261022/market/hourly.csv',base/'runs/trend-research-20261022/features-72h.npz')]
    news=prior/'news/all-periods.jsonl'
    prior_rows=json.loads((base/'evidence/feature-selection/candidates.json').read_text())
    prior_winner=next(r for r in prior_rows if r['spec']['id']=='n1v1b0l0e0')
    parse=lambda s:int(pd.Timestamp(s).timestamp())
    eras=[(r['era'],parse(r['normal']['start']),parse(r['normal']['end'])) for r in prior_winner['eras']]
    protocol={'declared_at':int(time.time()),'source_sha256':source_hash(),'specifications':SPECS,
        'profile':profile,'configurations':{s['id']:asdict(configuration(s,profile)) for s in SPECS},
        'data_role':'all previously viewed development; causal model refits are NOT new independent future validation',
        'prior_trials_lower_bound':258,'input_hashes':{str(p):digest_file(p) for p,_ in sources},
        'news_sha256':digest_file(news),'eras':eras,'duplicate_policy':'later dataset replaces overlapping public candle; conflicts recorded',
        'training':'365d rolling; 12h row grid; labels must mature; 42d purged calibration, 84d fallback solely if fewer than 80 matured samples; 180d recency half-life',
        'revision_reason':'42d window can lose most eligible observations after one real data gap and 30d feature warmup; fallback chooses by sample counts, never PnL',
        'labels':'7d/14d next-open counterfactual stop-aware net return; exact portfolio PnL validated separately',
        'fixed_model_settings':{'ridge':100,'boost_depth':2,'boost_trees':24,'learning_rate':.05,'min_leaf':100,'probability_gate':.55,'rmse_fraction':.1},
        'strict_criteria':{'all_era_positive_and_settled_at_normal_and_double_cost':True,
            'minimum_annualized_log_growth_pct':5,'median_annualized_log_growth_pct':15,
            'maximum_observed_drawdown_pct':12 if profile=='conservative' else 17,
            'minimum_closed_episodes':50,'minimum_per_era':8,'maximum_best_episode_share':.4,
            'positive_without_best_episode_in_at_least_two_eras':True,'positive_active_90d_blocks':.55},
        'ranking':'minimum annualized log growth - half maximum drawdown - quarter worst block loss + twice positive active block fraction',
        'post_run_rule':'never retune this run; all failures retained; select research candidate even if strict criteria fail',
        'news_availability':'same vintage archive and assumed +48h publication lag; no instantaneous political direction classifier',
        'exchange_writes':0,'account_reads':0,'credentials_used':False}
    atomic_json(out/'protocol.json',protocol)
    shutil.copytree(base/'wallex_engine',out/'source_snapshot',ignore=shutil.ignore_patterns('__pycache__'))
    d=build_data(sources,str(news))
    atomic_json(out/'data-quality.json',{'common_hours':len(d['times']),'feature_rows':len(d['x']),
        'overlapping_candle_conflicts':d['conflicts'],'no_forward_fill':True,'warmup_hours':721,
        'base_cache_targets_used':False,'feature_names':NAMES})
    predictions,models=learned_predictions(d,eras[0][1],eras[-1][2],out)
    rows=[]; matrices=[[] for _ in eras]; all_curves={}
    for spec in SPECS:
        c=configuration(spec,profile); results=[]; curves=[]
        for era_index,(era,start,end) in enumerate(eras):
            forecasts=grid(d,spec,start,end,predictions,models); capture={}
            normal,curve=simulate(d,c,forecasts,start,end,capture=capture)
            stress=simulate(d,replace(c,fee_bps=70,spread_bps=50,slippage_bps=20,max_spread_bps=70),forecasts,start,end)[0]
            blocks=[simulate(d,c,forecasts,t,min(end,t+90*86400))[0] for t in range(start,end,90*86400) if end-t>=60*86400]
            result={'era':era,'normal':normal,'cost_x2':stress,'blocks':blocks,'episodes':episodes(capture['fills'])}
            atomic_json(out/'candidates'/spec['id']/(era+'.json'),result)
            results.append(result); curves.append(curve); matrices[era_index].append(daily_returns(curve))
            print(json.dumps({'policy':spec['id'],'era':era,'return':normal['net_return_pct'],'stress':stress['net_return_pct'],'drawdown':normal['max_drawdown_pct']}),flush=True)
        row={'spec':spec,'assessment':assess(results,profile),'eras':results}; rows.append(row); all_curves[spec['id']]=curves
    atomic_json(out/'candidates.json',rows)
    eligible=[r for r in rows if r['assessment']['strict_criteria_passed']]
    best=max(eligible or rows,key=lambda r:r['assessment']['score'])
    selection={'spec':best['spec'],'assessment':best['assessment'],'frozen_at':int(time.time()),
        'strict_criteria_passed':bool(eligible),'eligible_count':len(eligible),'candidate_count':len(SPECS),
        'protocol_sha256':digest_file(out/'protocol.json'),'profile':profile,
        'prospective_validated':False,'research_only':True,'config':asdict(configuration(best['spec'],profile))}
    atomic_json(out/'selection.json',selection)
    diagnostics={'cscv':cscv(np.concatenate([np.column_stack(m) for m in matrices])),
        'family_bootstrap':family_bootstrap([np.column_stack(m) for m in matrices]),
        'future_profit_probability_claimed':False,'prior_trials_not_fully_corrected':True}
    atomic_json(out/'diagnostics.json',diagnostics)
    cutoff=d['times'][-1]+3600; selected_spec=best['spec']; final_fit=None
    if selected_spec['family'] in ('ridge','boost'):
        y,maturity=labels(d,selected_spec['target_hours']); final_fit=fit(d,y,maturity,cutoff,selected_spec['family'])
    artifact={'artifact_type':'longterm_research','feature_names':NAMES,'symbols':SYMBOLS,
        'horizon':72,'interval':3600,'spec':selected_spec,'fit':final_fit,'training_cutoff':cutoff,
        'valid_until':cutoff+30*86400,'research_only':True,'paper_eligible':True,
        'historical_strict_criteria_passed':bool(eligible),'prospective_validated':False,
        'selection_sha256':digest_file(out/'selection.json'),'created_at':int(time.time())}
    atomic_json(out/'model.json',artifact)
    c=replace(configuration(selected_spec,profile),model_artifact=str(out/'model.json'),
        news_path=str(news),news_required=selected_spec.get('news',True))
    atomic_json(out/'paper-config.json',asdict(c))
    for era,start,end in eras:
        forecasts=grid(d,selected_spec,start,end,predictions,models)
        report=simulate(d,configuration(selected_spec,profile),forecasts,start,end,out/'selected'/era)[0]
        expected=next(r for r in best['eras'] if r['era']==era)['normal']
        assert report['net_return_pct']==expected['net_return_pct']
    save_bars(out/'warmup.csv',[b for f in d['frames'][-1440:] for b in f.values()])
    print(json.dumps({'selected':selected_spec,'strict_criteria_passed':bool(eligible),'eligible':len(eligible)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--base',default='.'); p.add_argument('--out',required=True)
    p.add_argument('--profile',choices=('conservative','moderate'),default='conservative')
    a=p.parse_args(); run(Path(a.base).resolve(),Path(a.out).resolve(),a.profile)
