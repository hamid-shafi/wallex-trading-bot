"""Cost-aware supervised trade filtering and low-turnover trend policy primitives."""
from pathlib import Path
import hashlib
import json
import math
import statistics

from .model import Forecast
from .trained import FEATURE_NAMES,trained_features


def sigmoid(x):
    return 1/(1+math.exp(-max(-35.,min(35.,x))))


def trend_state(history,c):
    if len(history)<c.trend_slow:
        return None
    b=history[-c.trend_slow:]
    if any(y.ts-x.ts!=c.interval for x,y in zip(b,b[1:])):
        return None
    fast=statistics.fmean(x.close for x in b[-c.trend_fast:])
    slow=statistics.fmean(x.close for x in b)
    ratio=fast/slow-1
    return {"entry":ratio>c.trend_entry_bps/10000 and b[-1].close>fast,
            "exit":ratio < -c.trend_exit_bps/10000,
            "fast_mean":fast,"slow_mean":slow,"fast_slow_ratio":ratio,
            "close_above_fast":b[-1].close>fast}


class CostPredictor:
    def __init__(self,path,news_path=""):
        raw=Path(path).read_bytes()
        self.hash=hashlib.sha256(raw).hexdigest()
        self.artifact=json.loads(raw)
        if self.artifact["feature_names"]!=FEATURE_NAMES:
            raise ValueError("cost-aware feature schema mismatch")
        from .news import NewsIndex
        self.news=NewsIndex(news_path) if news_path else None

    def forecast(self,symbol,history,interval,as_of=None):
        a=self.artifact
        if symbol not in a["symbols"] or interval!=a["interval"]:
            raise ValueError("cost-aware symbol/interval mismatch")
        if not history or history[-1].ts+interval<a["training_cutoff"]:
            return None
        row=trained_features(history,a["horizon"],interval)
        if row is None:
            return None
        x,scale,atr=row
        x += [float(symbol==s) for s in a["symbols"]]
        decision_time=history[-1].ts+interval if as_of is None else as_of
        if decision_time<history[-1].ts+interval:
            raise ValueError("news decision time precedes closed price candle")
        news_values,context=self.news.at(decision_time) if self.news else ([0.]*6,{})
        if a.get("use_news") and context.get("feature_names")!=a.get("news_feature_names"):
            news_values=[0.]*6
        if a.get("use_news"):
            x+=news_values
        net=(sum(v*w for v,w in zip(x,a["return_weights"]))+a["return_bias"])*scale
        probability=sigmoid(sum(v*w for v,w in zip(x,a["probability_weights"]))+a["probability_bias"])
        if x[6]<=0 or x[15]<=0:
            probability=0.
            net=min(net,-.0001)
        if not a.get("fit_valid",True) or not a.get("validation_eligible",True):
            probability=0.
            net=min(net,-.0001)
        uncertainty=a["return_rmse"]*scale
        # Engine subtracts the actual configured/book cost; restore the label's cost once.
        gross=net+a["label_round_trip_cost"]
        names=FEATURE_NAMES+["asset_"+s for s in a["symbols"]]+a.get("news_feature_names",[])
        parts={name:x[i]*a["return_weights"][i]*scale for i,name in enumerate(names)}
        if a.get("use_news"):
            parts["news_logit_contribution"]=sum(v*w for v,w in zip(news_values,a["probability_weights"][-len(news_values):]))
        parts.update({"net_profit_probability":probability,"expected_net_return":net,
                      "calibration_bias":a["return_bias"]*scale,"news_coverage":news_values[-1],
                      "rss_conflict_count":context.get("rss_topics",{}).get("conflict",0),
                      "rss_sanctions_count":context.get("rss_topics",{}).get("sanctions",0)})
        return Forecast(gross,uncertainty,gross-a["penalty"]*uncertainty,atr,
                        a["training_samples"],0.,parts,self.hash[:16])
