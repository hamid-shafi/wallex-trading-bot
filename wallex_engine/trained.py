"""Portable trained-model inference. No network or optional training dependencies."""
from pathlib import Path
import hashlib
import json
import math
import statistics

from .model import Forecast, NAMES, features


FEATURE_NAMES = NAMES + [f"momentum_{n}" for n in (6,24,72,168)] + [
    "volatility_24", "volatility_168", "z72", "z168", "volume_24_168",
    "hour_sin", "hour_cos", "day_sin", "day_cos"]


def trained_features(history, horizon, interval=3600):
    if len(history)<169:
        return None
    b=history[-169:]
    if any(y.ts-x.ts!=interval for x,y in zip(b,b[1:])):
        return None
    base=features(b,horizon)
    x,scale,atr=base
    returns=[math.log(b[i].close/b[i-1].close) for i in range(1,len(b))]
    v24=max(statistics.pstdev(returns[-24:]),.0003)
    v168=max(statistics.pstdev(returns),.0003)
    x += [max(-5,min(5,sum(returns[-n:])/(v168*math.sqrt(n)))) for n in (6,24,72,168)]
    x += [v24*100, v168*100]
    for n in (72,168):
        values=[a.close for a in b[-n:]]
        x.append(max(-5,min(5,(values[-1]-statistics.fmean(values))/max(statistics.pstdev(values),values[-1]*.0003))))
    recent=statistics.fmean(a.volume for a in b[-24:])
    long=statistics.fmean(a.volume for a in b[-168:])
    x.append(max(-5,min(5,math.log(max(recent,1e-12)/max(long,1e-12)))))
    hour=(b[-1].ts+interval)%86400/86400*math.tau
    day=((b[-1].ts+interval)//86400)%7/7*math.tau
    x += [math.sin(hour),math.cos(hour),math.sin(day),math.cos(day)]
    return x,scale,atr


def tree_value(tree, x):
    while "value" not in tree:
        tree=tree["left"] if x[tree["feature"]]<=tree["threshold"] else tree["right"]
    return tree["value"]


def predict_numeric(model,x):
    if model["kind"]=="ridge":
        return sum(a*b for a,b in zip(model["weights"],x))
    if model["kind"]=="boost":
        return model["bias"]+sum(model["learning_rate"]*tree_value(t,x) for t in model["trees"])
    raise ValueError("unsupported trained model")


class TrainedPredictor:
    def __init__(self,path):
        self.path=Path(path)
        raw=self.path.read_bytes()
        self.hash=hashlib.sha256(raw).hexdigest()
        self.artifact=json.loads(raw)
        if self.artifact["feature_names"]!=FEATURE_NAMES:
            raise ValueError("trained feature schema mismatch")

    def forecast(self,symbol,history,interval):
        a=self.artifact
        if symbol not in a["symbols"] or interval!=a["interval"]:
            raise ValueError("trained symbol/interval mismatch")
        if not history or history[-1].ts+interval<a["training_cutoff"]:
            return None
        row=trained_features(history,a["horizon"],interval)
        if row is None:
            return None
        x,scale,atr=row
        x += [float(symbol==s) for s in a["symbols"]]
        prediction=predict_numeric(a["model"],x)+a["calibration_bias"]
        mu=prediction*scale
        uncertainty=a["calibration_rmse"]*scale
        return Forecast(mu,uncertainty,mu-a["penalty"]*uncertainty,atr,
                        a["training_samples"],0.,{"trained_prediction":mu},self.hash[:16])
