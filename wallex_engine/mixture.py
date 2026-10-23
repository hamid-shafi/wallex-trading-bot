"""Portable learned trend committee. Utility is directional, not expected profit."""
from pathlib import Path
import hashlib
import json
import math
import statistics

from .model import Forecast
from .news import NewsIndex
from .regime import context as regime_context,adjust


EXPERTS=[{'fast':f,'slow':s,'band':b,'news':False}
         for f,s in ((24,168),(72,336),(168,720),(336,720)) for b in (0.,.01)]


def combine(utilities,weights):
    # Ninth expert is cash with zero utility. Deadband avoids weak-consensus churn.
    strength=sum(w*(1 if u>0 else -1 if u<0 else 0) for w,u in zip(weights,utilities))
    utility=.03+.02*strength if strength>.25 else (-.03 if strength<-.25 else 0.)
    return utility,strength


def news_risk(values):
    return 1+min(1.,max(0.,values[2])/3+max(0.,values[3])/3)


class MixturePredictor:
    def __init__(self,path,news_path=''):
        raw=Path(path).read_bytes(); self.artifact=json.loads(raw)
        self.hash=hashlib.sha256(raw).hexdigest()
        a=self.artifact
        if a['experts']!=EXPERTS or len(a['weights'])!=9 or abs(sum(a['weights'])-1)>1e-8 or any(w<0 for w in a['weights']):
            raise ValueError('invalid trend committee schema/weights')
        self.news=NewsIndex(news_path) if news_path else None
        self.histories={}

    def forecast(self,symbol,history,interval,as_of=None):
        a=self.artifact
        if symbol not in a['symbols'] or interval!=3600:
            raise ValueError('mixture symbol/interval mismatch')
        if len(history)<720:
            return None
        b=history[-720:]; now=b[-1].ts+interval if as_of is None else as_of
        if now<b[-1].ts+interval:
            raise ValueError('decision precedes candle availability')
        if b[-1].ts+interval<a['training_cutoff'] or now>a['valid_until']:
            return None
        if any(y.ts-x.ts!=interval for x,y in zip(b,b[1:])):
            return None
        utilities=[]
        for spec in EXPERTS:
            fast=statistics.fmean(v.close for v in b[-spec['fast']:])
            slow=statistics.fmean(v.close for v in b[-spec['slow']:])
            ratio=fast/slow-1
            utilities.append(1. if ratio>spec['band'] and b[-1].close>fast else -1. if ratio<-spec['band'] else 0.)
        utility,strength=combine(utilities,a['weights'])
        values,context=self.news.at(now) if self.news else ([0.]*6,{})
        if context.get('feature_names')!=a.get('news_feature_names'):
            values=[0.]*6
        multiplier=news_risk(values) if a['use_news'] else 1.
        if a['use_news'] and not values[-1]:
            utility=min(0,utility)
        if not a.get('validation_eligible',False):
            utility=min(0,utility)
        atr=statistics.fmean(max(y.high-y.low,abs(y.high-x.close),abs(y.low-x.close)) for x,y in zip(b[-15:-1],b[-14:]))
        values_context=regime_context(self.histories,symbol) if symbol in self.histories else None
        utility,adjusted_atr=adjust(utility,atr*multiplier,a.get('risk_features',[]),values_context,
                                    a.get('min_notional',100000.)) if a.get('risk_features') else (utility,atr*multiplier)
        return Forecast(utility,0.,utility,adjusted_atr,1000,0.,
           {'directional_utility':utility,'committee_strength':strength,'not_a_return_forecast':True,
            'news_risk_multiplier':multiplier,'news_coverage':values[-1],
            'cash_expert_weight':a['weights'][-1],'learned_weights':a['weights'],
            'validation_eligible':a.get('validation_eligible',False),'regime_features':values_context},self.hash[:16])
