"""Chronological research with frozen model selection and prequential final test.

Training needs NumPy; exported inference uses only the standard library.
No exchange writes or account credentials exist in this module.
"""
import argparse
from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import json
import math
import statistics
import time

import numpy as np

from .audit import Audit, atomic_json, canonical, source_hash
from .config import Config
from .data import digest_file, load_bars, save_bars
from .engine import Engine, Intent, Portfolio, simulate_fill
from .model import Forecast
from .research import metrics
from .trained import FEATURE_NAMES, predict_numeric, trained_features


def iso(ts):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime(ts))


def prepare(bars,symbols,horizon):
    # Histories retain gaps: a feature/label window crossing a gap is ineligible.
    histories={s:[] for s in symbols}
    data={s:{b.ts:b for b in bars if b.symbol==s} for s in symbols}
    times=sorted(set.intersection(*(set(data[s]) for s in symbols)))
    frames=[{s:data[s][t] for s in symbols} for t in times]
    rows, locations, scale, atr, targets, maturities=[],[],[],[],[],[]
    for i,frame in enumerate(frames):
        t=times[i]
        for j,s in enumerate(symbols):
            histories[s].append(frame[s])
            f=trained_features(histories[s],horizon)
            if f is None:
                continue
            x,sc,tr=f
            x += [float(s==v) for v in symbols]
            rows.append(x); locations.append((i,j)); scale.append(sc); atr.append(tr)
            valid=i+horizon<len(times) and times[i+horizon]-t==horizon*3600
            targets.append(max(-5,min(5,math.log(frames[i+horizon][s].close/frames[i+1][s].open)/sc)) if valid else float("nan"))
            maturities.append(t+(horizon+1)*3600)
    return {"x":np.asarray(rows,dtype=float),"y":np.asarray(targets),
            "locations":np.asarray(locations),"scale":np.asarray(scale),"atr":np.asarray(atr),
            "maturity":np.asarray(maturities),"times":times,"frames":frames,
            "row_ts":np.asarray([times[i]+3600 for i,j in locations]),"symbols":symbols}


def grow_tree(x, gradient, weight, bins, thresholds, depth, min_leaf=100, l2=50.):
    # Histogram squared-error trees; regularized leaves and fixed minimum sample size.
    n=len(gradient)
    total_w=float(weight.sum()); total_g=float((gradient*weight).sum())
    leaf={"value":total_g/(total_w+l2)}
    if depth==0 or n<2*min_leaf:
        return leaf
    best=None
    for feature,edges in enumerate(thresholds):
        if not len(edges):
            continue
        ids=bins[:,feature]
        counts=np.bincount(ids,minlength=len(edges)+1)
        weights=np.bincount(ids,weights=weight,minlength=len(edges)+1)
        grads=np.bincount(ids,weights=gradient*weight,minlength=len(edges)+1)
        nc=np.cumsum(counts)[:-1]; nw=np.cumsum(weights)[:-1]; ng=np.cumsum(grads)[:-1]
        gain=ng**2/(nw+l2)+(total_g-ng)**2/(total_w-nw+l2)-total_g**2/(total_w+l2)
        gain[(nc<min_leaf)|(n-nc<min_leaf)]=-np.inf
        k=int(np.argmax(gain))
        if np.isfinite(gain[k]) and gain[k]>1e-6 and (best is None or gain[k]>best[0]):
            best=(float(gain[k]),feature,k)
    if best is None:
        return leaf
    _,f,k=best
    mask=bins[:,f]<=k
    return {"feature":f,"threshold":float(thresholds[f][k]),
            "left":grow_tree(x[mask],gradient[mask],weight[mask],bins[mask],thresholds,depth-1,min_leaf,l2),
            "right":grow_tree(x[~mask],gradient[~mask],weight[~mask],bins[~mask],thresholds,depth-1,min_leaf,l2)}


def tree_predict(tree,x):
    if "value" in tree:
        return np.full(len(x),tree["value"])
    mask=x[:,tree["feature"]]<=tree["threshold"]
    out=np.empty(len(x))
    out[mask]=tree_predict(tree["left"],x[mask])
    out[~mask]=tree_predict(tree["right"],x[~mask])
    return out


def predict(model,x):
    if model["kind"]=="ridge":
        return x@np.asarray(model["weights"])
    return model["bias"]+sum((model["learning_rate"]*tree_predict(t,x) for t in model["trees"]),start=np.zeros(len(x)))


def train_numeric(x,y,weight,spec):
    if spec["kind"]=="ridge":
        penalty=np.eye(x.shape[1])*spec["l2"]
        penalty[0,0]=1.
        w=np.linalg.solve(x.T@(x*weight[:,None])+penalty,x.T@(y*weight))
        return {"kind":"ridge","weights":w.tolist()}
    thresholds=[np.unique(np.quantile(x[:,j],np.linspace(.05,.95,19))) for j in range(x.shape[1])]
    bins=np.column_stack([np.searchsorted(edges,x[:,j],side="left") for j,edges in enumerate(thresholds)])
    bias=float(np.average(y,weights=weight))
    fitted=np.full(len(y),bias); trees=[]
    for n in range(100):
        gradient=np.clip(y-fitted,-2.,2.)
        tree=grow_tree(x,gradient,weight,bins,thresholds,spec["depth"])
        fitted+=.05*tree_predict(tree,x)
        trees.append(tree)
    return {"kind":"boost","bias":bias,"learning_rate":.05,"trees":trees}


def fit(data,cutoff,spec,shuffle=False):
    # Fit and calibration targets must have matured strictly before the decision.
    mature=np.isfinite(data["y"]) & (data["maturity"]<=cutoff)
    calibration=mature & (data["row_ts"]>=cutoff-7*86400)
    fitting=mature & (data["maturity"]<=cutoff-7*86400)
    if fitting.sum()<1000 or calibration.sum()<100:
        raise ValueError("insufficient fit/calibration data")
    x,y=data["x"][fitting],data["y"][fitting].copy()
    if shuffle:
        y=np.random.default_rng(73).permutation(y)
    weight=np.exp(-(cutoff-data["row_ts"][fitting])/(60*86400)*math.log(2))
    model=train_numeric(x,y,weight,spec)
    residual=data["y"][calibration]-predict(model,data["x"][calibration])
    bias=float(np.clip(np.mean(residual),-.5,.5))
    rmse=float(np.sqrt(np.mean((residual-bias)**2)))
    return {"feature_names":FEATURE_NAMES,"symbols":data["symbols"],"interval":3600,
            "horizon":spec["horizon"],"model":model,"calibration_bias":bias,
            "calibration_rmse":rmse,"training_samples":int(fitting.sum()),
            "calibration_samples":int(calibration.sum()),"training_cutoff":cutoff,
            "fit_latest_label_maturity":int(data["maturity"][fitting].max()),
            "calibration_latest_label_maturity":int(data["maturity"][calibration].max()),
            "spec":spec,"penalty":0.,"created_at":int(time.time()),"shuffled_targets":shuffle,
            "uncertainty_note":"past calibration RMSE, not a probability of profit or a confidence interval"}


def forecast_grid(data,artifact,start,end):
    mask=(data["row_ts"]>=start)&(data["row_ts"]<end)
    pred=predict(artifact["model"],data["x"][mask])+artifact["calibration_bias"]
    out={}
    model_hash=hashlib.sha256(canonical(artifact).encode()).hexdigest()[:16]
    for row,prediction,scale,atr in zip(data["locations"][mask],pred,data["scale"][mask],data["atr"][mask]):
        i,j=map(int,row)
        out[(i,data["symbols"][j])]=(float(prediction*scale),float(artifact["calibration_rmse"]*scale),
                                    float(atr),artifact["training_samples"],model_hash)
    return out


def simulate(data,c,forecasts,start,end,output=None,baseline=None,capture=None):
    e,p=Engine(replace(c,model_artifact="")),Portfolio(c.initial_cash)
    curve=[]; pending=[]; decisions=[]
    fs=data["frames"]; times=data["times"]
    if c.strategy!="forecast":
        for s in c.symbols:
            e.learners[s].history=[f[s] for t,f in zip(times,fs) if t<start]
    indices=[i for i,t in enumerate(times) if start<=t<end]
    if not indices:
        raise ValueError("empty simulation")
    for i in indices:
        frame=fs[i]; t=times[i]
        if c.strategy!="forecast":
            # Histories are causal prefixes, including gaps. No learning is done here.
            for s in c.symbols:
                e.learners[s].history.append(frame[s])
        # A missing candle cancels pending entries; protective exits wait for next available open.
        gap=i>0 and times[i]-times[i-1]!=3600
        for intent in pending:
            if gap and intent.side=="BUY":
                continue
            simulate_fill(p,intent,frame[intent.symbol],c)
        for s,pos in list(p.positions.items()):
            if frame[s].low<=pos.stop:
                simulate_fill(p,Intent(s,"SELL",pos.quantity,"intrabar_stop",decision_ts=t),frame[s],c,min(frame[s].open,pos.stop))
        e.forecasts={}
        for s in c.symbols:
            v=forecasts.get((i,s))
            if v is not None:
                mu,unc,atr,samples,model_hash=v[:5]
                contributions=v[5] if len(v)>5 else {}
                if baseline=="momentum":
                    mu=math.log(frame[s].close/fs[max(0,i-c.horizon)][s].close)
                    unc=0.
                elif baseline=="cash":
                    mu=0.
                e.forecasts[s]=Forecast(mu,unc,mu-c.uncertainty_penalty*unc,atr,samples,0.,contributions,model_hash)
            else:
                e.forecasts[s]=None
        pending,why=e.plan(p,frame)
        remaining=end-(t+3600)
        if remaining<=(c.entry_buffer_bars or c.horizon)*3600:
            pending=[x for x in pending if x.side!="BUY" or x.reason=="dust_consolidation_for_exit"]
        if remaining<=c.settlement_buffer:
            pending=[x for x in pending if x.reason=="dust_consolidation_for_exit" and remaining>=7200]
            pending += [Intent(s,"SELL",pos.quantity,"evaluation_settlement",decision_ts=t+3600)
                        for s,pos in sorted(p.positions.items()) if pos.quantity*frame[s].close>=c.min_notional]
        prices={s:b.close for s,b in frame.items()}
        equity=p.equity(prices)
        e.loss_gate(equity,t+3600)
        curve.append({"ts":t+3600,"equity":equity,"cash":p.cash})
        if output:
            decisions.append({"ts":t+3600,"decisions":why,"intents":[asdict(x) for x in pending],"gap":gap})
    report=metrics(p,curve,c,[])
    report["decision_latency_p95_ms"]=None  # not measured by this research simulator
    report.update({"start":iso(times[indices[0]]),"end":iso(times[indices[-1]]+3600),
                   "mark_to_market_return_pct":(curve[-1]["equity"]/c.initial_cash-1)*100,
                   "missing_hours":sum(max(0,(times[b]-times[a])//3600-1) for a,b in zip(indices,indices[1:])),
                   "fill_model":"next available open, fees/spread/slippage, participation, stops and real minimum notional",
                   "cash_return_pct":0.})
    drag=(c.spread_bps/2+c.slippage_bps)/10000
    fee=c.fee_bps/10000
    # Risk-matched buy-and-hold: same 50% gross cap, costs both sides, marked exits.
    bh=sum(fs[indices[-1]][s].close*(1-drag)*(1-fee)/(fs[indices[0]][s].open*(1+drag)*(1+fee))-1 for s in c.symbols)/len(c.symbols)*c.gross_cap
    report["buy_hold_50pct_return_pct"]=bh*100
    if capture is not None:
        capture['fills']=list(p.fills)
    if output:
        out=Path(output); out.mkdir(parents=True,exist_ok=True)
        atomic_json(out/"report.json",report); atomic_json(out/"equity.json",curve); atomic_json(out/"trades.json",p.fills)
        log=Audit(out/"decisions.jsonl")
        if log.seq:
            raise ValueError("research simulation output already exists")
        for d in decisions:
            log.write("decision",d["ts"],**{k:v for k,v in d.items() if k!="ts"})
    return report,curve


def comparable(report):
    return report["net_return_pct"] if report["settled_to_tmn"] else -100.


def research(dataset,out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    if (out/"selection.json").exists():
        raise ValueError("selection already frozen; use a new declared research run, never overwrite holdout")
    bars=load_bars(dataset)
    # Exclude USDT from trading a priori due to its additional missing data; no return-based universe selection.
    symbols=["BTCTMN","ETHTMN","SOLTMN","XRPTMN","BNBTMN"]
    start=min(b.ts for b in bars); end=max(b.ts for b in bars)+3600
    vstart=start+90*86400; test_start=start+135*86400
    blocks=[(vstart+k*15*86400,vstart+(k+1)*15*86400) for k in range(3)]
    policy={"dataset_sha256":digest_file(dataset),"source_sha256":source_hash(),
            "train_start":iso(start),"validation_start":iso(vstart),"holdout_start":iso(test_start),"end":iso(end),
            "horizons":[6,12,24],"algorithms":["ridge_l2_100","ridge_l2_1000","boost_depth2","boost_depth3"],
            "edge_bps":[0,25,50],"uncertainty_penalty":[0.,.5],"seed":73,
            "fee_bps_per_side":35.,"spread_bps":25.,"slippage_bps_per_side":10.,
            "retraining":"each 15-day block, mature labels only, seven-day calibration",
            "selection_score":"median validation return - 0.5*median drawdown; >=12 sell fills and >=2/3 positive blocks",
            "holdout_rule":"no parameter selection from holdout; earlier matured holdout labels may train later fixed-schedule models",
            "research_only":True,"exchange_writes":0,"credentials_used":False,
            "universe_bias":"current surviving liquid examples; USDT excluded for extra missingness",
            "training_dependency":"numpy; portable stdlib inference"}
    atomic_json(out/"protocol.json",policy)
    data_by_h={}
    forecasts={}; candidates=[]; artifacts={}
    base=Config(symbols=symbols,interval=3600,fee_bps=35,spread_bps=25,slippage_bps=10,
                min_notional=100000,settlement_buffer=6*3600,max_spread_bps=60,
                min_stop_fraction=.015,stop_atr=3.,cooldown=3,live_enabled=False)
    for h in policy["horizons"]:
        data_by_h[h]=prepare(bars,symbols,h)
        data=data_by_h[h]
        specs=[{"kind":"ridge","l2":a,"horizon":h} for a in (100,1000)]
        specs += [{"kind":"boost","depth":d,"horizon":h} for d in (2,3)]
        for spec in specs:
            name=f'{spec["kind"]}-{spec.get("l2",spec.get("depth"))}-h{h}'
            streams=[]
            for k,(left,right) in enumerate(blocks):
                artifact=fit(data,left,spec)
                atomic_json(out/"validation_models"/f"{name}-{k}.json",artifact)
                streams.append(forecast_grid(data,artifact,left,right))
            forecasts[name]=streams; artifacts[name]=spec
            for edge in policy["edge_bps"]:
                for penalty in policy["uncertainty_penalty"]:
                    c=replace(base,horizon=h,edge_bps=edge,uncertainty_penalty=penalty)
                    reports=[simulate(data,c,streams[k],left,right)[0] for k,(left,right) in enumerate(blocks)]
                    returns=[comparable(r) for r in reports]
                    sells=sum(r["sell_fills"] for r in reports)
                    positive=sum(v>0 for v in returns)
                    score=statistics.median(returns)-.5*statistics.median(r["max_drawdown_pct"] for r in reports)
                    eligible=sells>=12 and positive>=2
                    candidates.append({"name":name,"spec":spec,"edge_bps":edge,"penalty":penalty,
                                       "returns_pct":returns,"sell_fills":sells,"positive_blocks":positive,
                                       "score":score,"eligible":eligible,"reports":reports})
            print(json.dumps({"trained":name,"candidates":len(candidates)}),flush=True)
    atomic_json(out/"all_validation_candidates.json",candidates)
    eligible=[x for x in candidates if x["eligible"]]
    selected=max(eligible or candidates,key=lambda x:x["score"])
    selection={**selected,"eligible_selection":bool(eligible),"frozen_at":int(time.time()),
               "candidate_count":len(candidates),"test_start":test_start,"test_end":end,
               "protocol_sha256":digest_file(out/"protocol.json"),"dataset_sha256":digest_file(dataset)}
    # This file is written and hashed before any final-test simulations are evaluated.
    atomic_json(out/"selection.json",selection)
    selection_hash=digest_file(out/"selection.json")
    print(json.dumps({"selection":{k:selection[k] for k in ("name","edge_bps","penalty","returns_pct","eligible_selection")},"hash":selection_hash}),flush=True)
    h=selected["spec"]["horizon"]; data=data_by_h[h]
    c=replace(base,horizon=h,edge_bps=selected["edge_bps"],uncertainty_penalty=selected["penalty"])
    full_stream={}; placebo_stream={}; test_models=[]
    for k,left in enumerate(range(test_start,end,15*86400)):
        right=min(end,left+15*86400)
        a=fit(data,left,selected["spec"]); a["penalty"]=selected["penalty"]
        atomic_json(out/"test_models"/f"block-{k}.json",a)
        test_models.append({"cutoff":iso(left),"through":iso(right),"artifact_sha256":digest_file(out/"test_models"/f"block-{k}.json")})
        full_stream.update(forecast_grid(data,a,left,right))
        shuffled=fit(data,left,selected["spec"],shuffle=True)
        placebo_stream.update(forecast_grid(data,shuffled,left,right))
    holdout,curve=simulate(data,c,full_stream,test_start,end,out/"holdout")
    stress={}
    for name,mult in (("cost_x2",2),("cost_x3",3)):
        sc=replace(c,fee_bps=c.fee_bps*mult,spread_bps=c.spread_bps*mult,slippage_bps=c.slippage_bps*mult)
        stress[name]=simulate(data,sc,full_stream,test_start,end,out/name)[0]
    baselines={"momentum":simulate(data,c,full_stream,test_start,end,out/"momentum",baseline="momentum")[0],
               "shuffled_target":simulate(data,c,placebo_stream,test_start,end,out/"shuffled_target")[0],
               "cash":simulate(data,c,full_stream,test_start,end,out/"cash",baseline="cash")[0]}
    windows=[]
    for k,left in enumerate(range(test_start,end-72*3600+1,72*3600)):
        report,_=simulate(data,c,full_stream,left,left+72*3600,out/"challenge_windows"/f"window-{k:02d}")
        windows.append(report)
    # Fit the exact selected method on all labels that have now matured for prospective shadow use.
    final=fit(data,end,selected["spec"]); final["penalty"]=selected["penalty"]
    final.update({"dataset_sha256":digest_file(dataset),"selection_sha256":selection_hash,
                  "source_sha256":source_hash(),"research_only":True})
    atomic_json(out/"model.json",final)
    deploy=replace(c,model_artifact=str(out/"model.json").replace("\\","/"))
    atomic_json(out/"shadow-config.json",asdict(deploy))
    # Runtime strict warmup uses only the final contiguous all-market suffix.
    last=len(data["times"])-1; first=last
    while first>0 and data["times"][first]-data["times"][first-1]==3600:
        first-=1
    save_bars(out/"shadow-warmup.csv",[b for f in data["frames"][first:] for b in f.values()])
    returns=[comparable(r) for r in windows]
    interval=holdout["uncertainty"]["return_interval_95"]
    gate=(selection["eligible_selection"] and comparable(holdout)>0 and len(windows)>=10 and
          sum(v>0 for v in returns)>=len(windows)*.6 and holdout["sell_fills"]>=20 and
          comparable(stress["cost_x2"])>0 and interval is not None and interval[0]>0)
    result={"selection_sha256":selection_hash,"selected_name":selected["name"],"candidate_count":len(candidates),
            "validation_returns_pct":selected["returns_pct"],"validation_selection_eligible":selection["eligible_selection"],
            "holdout":holdout,"stress":stress,"baselines":baselines,"test_models":test_models,
            "challenge_windows":{"count":len(windows),"positive":sum(v>0 for v in returns),
                                 "median_return_pct":statistics.median(returns),"worst_return_pct":min(returns),
                                 "all":windows},
            "profitability_gate_passed":gate,"status":"HISTORICAL_CANDIDATE" if gate else "NOT_PROVEN_PROFITABLE",
            "reality_checks":["only public Wallex data; no API key sent","all exchange writes blocked",
                              "OHLCV historical depth and spread unavailable; assumed costs",
                              "short sample and selection bias remain; no guarantee of future returns",
                              "earlier holdout observations may train later blocks under a frozen schedule",
                              "no retuning on final test; retain losing experiments"],
            "completed_at":int(time.time())}
    atomic_json(out/"results.json",result)
    print(json.dumps({"status":result["status"],"holdout_return_pct":holdout["net_return_pct"],
                      "holdout_drawdown_pct":holdout["max_drawdown_pct"],"sells":holdout["sell_fills"],
                      "windows_positive":result["challenge_windows"]["positive"],"windows":len(windows)}),flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data",required=True); p.add_argument("--out",required=True)
    args=p.parse_args()
    research(args.data,args.out)
