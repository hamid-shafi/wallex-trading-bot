"""Portable long-horizon research policies. All inputs are closed, past candles."""
from pathlib import Path
import hashlib
import json
import math
import statistics

from .gpr import GPR_NAMES
from .model import Forecast
from .trained import FEATURE_NAMES,trained_features,predict_numeric
from .mixture import EXPERTS,combine,news_risk

SYMBOLS=['BTCTMN','ETHTMN','SOLTMN','XRPTMN','BNBTMN']
EXTRA_NAMES=['momentum_336','momentum_720','volatility_720','drawdown_720',
             'trend_72_720','relative_momentum_168','market_momentum_168',
             'breadth_720','efficiency_168']
NAMES=FEATURE_NAMES+['asset_'+s for s in SYMBOLS]+EXTRA_NAMES+GPR_NAMES


def clip(v):
    return max(-5.,min(5.,v))


def sigmoid(v):
    return 1/(1+math.exp(-max(-35.,min(35.,v))))


def row(histories,symbol,news_values):
    if any(len(histories.get(s,[]))<721 for s in SYMBOLS):
        return None
    b=histories[symbol][-721:]; latest=b[-1].ts
    if any(h[-1].ts!=latest or any(y.ts-x.ts!=3600 for x,y in zip(h[-721:-1],h[-720:]))
           for h in (histories[s] for s in SYMBOLS)):
        return None
    base=trained_features(b,72)
    if base is None: return None
    x,_,atr=base
    returns=[math.log(y.close/x.close) for x,y in zip(b,b[1:])]
    vol=max(.0003,statistics.pstdev(returns[-168:]))
    v24=statistics.pstdev(returns[-24:]); v720=statistics.pstdev(returns)
    mom7=math.log(b[-1].close/b[-169].close); mom30=math.log(b[-1].close/b[0].close)
    market=[math.log(h[-1].close/h[-169].close) for h in (histories[s] for s in SYMBOLS)]
    median=statistics.median(market)
    breadth=sum(h[-1].close>statistics.fmean(v.close for v in h[-720:])
                for h in (histories[s] for s in SYMBOLS))/len(SYMBOLS)
    mean72=statistics.fmean(v.close for v in b[-72:]); mean720=statistics.fmean(v.close for v in b[-720:])
    extra=[clip(math.log(b[-1].close/b[-337].close)/(vol*math.sqrt(336))),
           clip(mom30/(vol*math.sqrt(720))),v720*100,
           clip(math.log(b[-1].close/max(v.close for v in b[-720:]))/(vol*math.sqrt(720))),
           clip(math.log(mean72/mean720)/(vol*math.sqrt(720))),
           clip((mom7-median)/(vol*math.sqrt(168))),clip(median/(vol*math.sqrt(168))),
           breadth,abs(sum(returns[-168:]))/max(1e-12,sum(abs(r) for r in returns[-168:]))]
    utilities=[]
    for spec in EXPERTS:
        fast=statistics.fmean(v.close for v in b[-spec['fast']:])
        slow=statistics.fmean(v.close for v in b[-spec['slow']:])
        ratio=fast/slow-1
        utilities.append(1 if ratio>spec['band'] and b[-1].close>fast else -1 if ratio<-spec['band'] else 0)
    committee,strength=combine(utilities,[1/9]*9)
    aux={'atr':atr,'volatility_ratio':v24/vol,'committee':committee,'strength':strength,
        'trend_entry':b[-1].close>mean72 and mean72>mean720,
        'trend_exit':mean72<mean720,'above_long':b[-1].close>mean720,
        'momentum_168':mom7,'momentum_720':mom30,'relative_168':extra[5],
        'z72':x[19],'breakout_entry':b[-1].close>max(v.high for v in b[:-1]),
        'breakout_exit':b[-1].close<min(v.low for v in b[-169:-1]),
        'liquidity':statistics.median(v.close*v.volume for v in b[-24:])*.01}
    return x+[float(symbol==s) for s in SYMBOLS]+extra+list(news_values),aux


def decision(spec,x,aux,fit=None,raw_prediction=None,raw_probability=None):
    """Return directional utility and adjusted ATR; utility is not a price target."""
    family=spec['family']; use_news=spec.get('news',True); values=x[-6:]
    risk=max(1.,min(2.,aux['volatility_ratio']))
    if use_news: risk*=news_risk(values)
    atr=aux['atr']*risk
    utility=0.; parts={'policy':family,'news_coverage':values[-1],
                       'risk_multiplier':risk,'not_a_return_forecast':True}
    if family=='committee':
        utility=aux['committee']
    elif family=='momentum':
        utility=.04 if aux['trend_entry'] else -.03 if aux['trend_exit'] else 0.
    elif family=='breakout':
        utility=.04 if aux['breakout_entry'] else -.03 if aux['breakout_exit'] else 0.
    elif family=='relative':
        utility=(.04+.005*max(0,min(4,aux['relative_168']))) if aux['above_long'] and aux['momentum_720']>0 and aux['relative_168']>0 else 0.
        if aux['trend_exit'] or aux['relative_168']<-.5: utility=-.03
    elif family=='pullback':
        utility=.04 if aux['above_long'] and not aux['trend_exit'] and aux['z72']<-.75 else 0.
        if aux['trend_exit'] or aux['z72']>1: utility=-.03
    elif family=='technical_committee':
        from .technical import apply_filters
        utility,details=apply_filters(spec,aux)
        parts.update(details)
    elif family in ('ridge','boost'):
        if fit and fit.get('fit_valid'):
            width=len(NAMES)
            if raw_prediction is None:
                raw_prediction=predict_numeric(fit['return_model'],x)+fit['return_bias']
            if raw_probability is None:
                raw_probability=sigmoid(sum(a*b for a,b in zip(x,fit['probability_weights']))+fit['probability_bias'])
            assert len(x)==width
            lower=raw_prediction-.1*fit['return_rmse']
            approved=aux['committee']>0 and lower>0 and raw_probability>=.55
            utility=(.04+min(.04,max(0,lower)) if approved else
                     -.03 if aux['committee']<0 or raw_prediction<-.02 else 0.)
            parts.update({'counterfactual_net_prediction':raw_prediction,
                'counterfactual_profit_probability':raw_probability,'calibration_brier':fit['calibration_brier'],
                'target_hours':spec['target_hours'],'actual_trade_success_probability':False})
    else:
        raise ValueError('unknown long-term research policy')
    if (use_news and values[-1]<1) or (spec.get('liquidity',True) and aux['liquidity']<200000):
        utility=min(0.,utility)
    parts['directional_utility']=utility
    return utility,atr,parts


class LongtermPredictor:
    def __init__(self,path,news_path=''):
        raw=Path(path).read_bytes(); self.hash=hashlib.sha256(raw).hexdigest(); self.artifact=json.loads(raw)
        a=self.artifact
        if a['symbols']!=SYMBOLS or a['feature_names']!=NAMES:
            raise ValueError('long-term feature schema mismatch')
        from .news import NewsIndex
        self.news=NewsIndex(news_path) if news_path else None
        self.histories={}

    def forecast(self,symbol,history,interval,as_of=None):
        a=self.artifact
        if interval!=3600 or symbol not in SYMBOLS: raise ValueError('long-term symbol/interval mismatch')
        if not history: return None
        now=history[-1].ts+3600 if as_of is None else as_of
        if now<history[-1].ts+3600: raise ValueError('decision precedes closed candle')
        if history[-1].ts+3600<a['training_cutoff'] or now>a['valid_until']: return None
        values,context=self.news.at(now) if self.news else ([0.]*6,{})
        if context.get('feature_names')!=GPR_NAMES: values=[0.]*6
        result=row(self.histories,symbol,values)
        if result is None: return None
        x,aux=result
        if a['spec']['family']=='technical_committee':
            from .technical import context
            aux.update(context(self.histories[symbol][-721:],now))
            aux['raw_momentum_168']=aux['momentum_168']
            aux['breadth_720']=x[len(FEATURE_NAMES)+len(SYMBOLS)+7]
        if not a['spec'].get('news',True): x[-6:]=[0.]*6
        u,atr,parts=decision(a['spec'],x,aux,a.get('fit'))
        if not a.get('paper_eligible',False): u=min(0.,u)
        parts['directional_utility']=u
        return Forecast(u,0.,u,atr,1000,0.,parts,self.hash[:16])
