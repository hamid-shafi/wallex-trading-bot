"""Retrospective diagnostics; no statistic here certifies future profits."""
import itertools
import math

import numpy as np


def daily_returns(curve,initial=5000000.):
    days={}
    for point in curve:
        days[(point['ts']+12600)//86400]=point['equity']
    values=np.asarray(list(days.values()))
    return values/np.r_[initial,values[:-1]]-1


def episodes(fills):
    opened={}; closed=[]
    for fill in fills:
        s=fill['symbol']
        if fill['side']=='BUY':
            p=opened.setdefault(s,{'symbol':s,'opened':fill['ts'],'quantity':0.,'pnl':0.})
            p['quantity']+=fill['quantity']
        else:
            p=opened[s]; p['quantity']-=fill['quantity']; p['pnl']+=fill['pnl']
            if p['quantity']<max(1e-12,fill['quantity']*1e-9):
                closed.append({**p,'closed':fill['ts']}); del opened[s]
    return closed


def cscv(matrix,parts=12):
    """Select by in-sample Sharpe, rank on complementary blocks, average ties."""
    a=np.asarray(matrix,dtype=float)
    if a.ndim!=2 or not np.isfinite(a).all() or parts%2 or len(a)<parts*2:
        raise ValueError('invalid CSCV return matrix')
    # Identical economic paths must not masquerade as independent choices.
    keep=[]
    for i in range(a.shape[1]):
        if not any(np.max(np.abs(a[:,i]-a[:,j]))<1e-12 for j in keep):
            keep.append(i)
    a=a[:,keep]
    if len(keep)<2:
        return {'pbo':None,'unique_candidates':len(keep),'reason':'no distinguishable choices'}
    blocks=np.array_split(np.arange(len(a)),parts); lambdas=[]; below=[]
    for chosen in itertools.combinations(range(parts),parts//2):
        training=np.concatenate([blocks[i] for i in chosen])
        test=np.concatenate([blocks[i] for i in range(parts) if i not in chosen])
        score=a[training].mean(axis=0)/np.maximum(1e-12,a[training].std(axis=0))
        # All tied in-sample winners are evaluated rather than picking by column order.
        winners=np.flatnonzero(np.isclose(score,score.max(),rtol=1e-10,atol=1e-12))
        oos=a[test].mean(axis=0)/np.maximum(1e-12,a[test].std(axis=0))
        ranks=[]
        for winner in winners:
            same=np.isclose(oos,oos[winner],rtol=1e-10,atol=1e-12)
            rank=(np.sum((oos<oos[winner])&~same)+.5*np.sum(same))/len(oos)
            rank=float(np.clip(rank,1e-9,1-1e-9)); ranks.append(rank)
            lambdas.append(math.log(rank/(1-rank)))
        below.append(sum(r<=.5 for r in ranks)/len(ranks))
    return {'pbo':float(np.mean(below)),'splits':len(below),'parts':parts,
            'unique_candidates':len(keep),'unique_column_indices':keep,
            'median_rank_logit':float(np.median(lambdas)),
            'warning':'retrospective CSCV with unpurged long positions; diagnostic only, not a causal trading test'}


def family_bootstrap(era_matrices,repetitions=2000,seed=73,block=7):
    """Paired within-era moving blocks, centered null, maximum t across the family."""
    arrays=[np.asarray(v,dtype=float) for v in era_matrices]
    combined=np.concatenate(arrays)
    n=len(combined)
    observed=combined.mean(axis=0)/np.maximum(1e-12,combined.std(axis=0))*math.sqrt(n)
    rng=np.random.default_rng(seed); maximum=[]
    for offset in range(0,repetitions,100):
        count=min(100,repetitions-offset); summed=np.zeros((count,combined.shape[1])); squares=summed.copy()
        for array in arrays:
            centered=array-array.mean(axis=0); length=len(array)
            starts=rng.integers(0,length,size=(count,(length+block-1)//block))
            indices=((starts[:,:,None]+np.arange(block))%length).reshape(count,-1)[:,:length]
            draws=centered[indices]; summed+=draws.sum(axis=1); squares+=(draws**2).sum(axis=1)
        mean=summed/n; sd=np.sqrt(np.maximum(1e-24,squares/n-mean**2))
        maximum.extend((mean/sd*math.sqrt(n)).max(axis=1).tolist())
    max_values=np.asarray(maximum)
    corrected=[float((1+np.sum(max_values>=t))/(repetitions+1)) for t in observed]
    return {'family_adjusted_p_values':corrected,'observed_daily_t':observed.tolist(),
            'repetitions':repetitions,'block_days':block,'seed':seed,
            'scope':'this declared family only; earlier adaptive research and data reuse are not corrected',
            'warning':'approximate conditional resampling under a centered within-era null; not future-profit probabilities'}
