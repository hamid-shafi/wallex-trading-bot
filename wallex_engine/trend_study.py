"""Frozen selection of cost-aware trade filters; separate previously unseen history."""
import argparse
from dataclasses import asdict,replace
from pathlib import Path
import hashlib
import json
import math
import shutil
import statistics
import tempfile
import time
import zipfile

import numpy as np

from .audit import atomic_json,canonical,source_hash
from .config import Config
from .data import digest_file,load_bars,save_bars
from .experiment import comparable,iso,prepare,simulate
from .news import NewsIndex,NEWS_NAMES
from .trained import FEATURE_NAMES


FEATURE_CACHE_FIELDS=("x","y","locations","scale","atr","maturity","row_ts")


def feature_cache_identity(dataset,symbols):
    """Bind reusable numerical arrays to their inputs and generating definitions."""
    source=Path(__file__).parent
    return {"format":"wallex-research-cache","version":1,"kind":"features-and-72h-labels",
        "dataset_sha256":digest_file(dataset),"symbols":list(symbols),"interval":3600,
        "feature_names":FEATURE_NAMES+["asset_"+s for s in symbols],"target_horizon":72,
        "label_policy":{"entry":"next_open","fee_bps_per_side":35,"spread_bps":25,
            "slippage_bps_per_side":10,"stop_atr":4.,"minimum_stop_fraction":.03,
            "scaled_target_clip":5.},
        "numpy_version":np.__version__,
        "source_sha256":{name:digest_file(source/name) for name in
            ("data.py","model.py","trained.py","experiment.py","trend_study.py")}}


def _array_identity(array):
    value=np.ascontiguousarray(array)
    return {"shape":list(value.shape),"dtype":value.dtype.str,
            "sha256":hashlib.sha256(value.tobytes()).hexdigest()}


def _load_research_cache(path,identity,fields):
    """Never silently reuse a legacy, mismatched, or partially written cache."""
    saved=None
    try:
        with Path(path).open("rb") as handle:
            saved=np.load(handle,allow_pickle=False)
            if not isinstance(saved,np.lib.npyio.NpzFile) or "cache_identity" not in saved.files:
                raise ValueError("legacy cache has no complete identity")
            metadata=json.loads(str(saved["cache_identity"].item()))
            if metadata["definition"]!=identity:
                raise ValueError("dataset, schema, source, or label parameters changed")
            arrays={key:saved[key] for key in fields}
            if metadata["arrays"]!={key:_array_identity(value) for key,value in arrays.items()}:
                raise ValueError("array contents, shape, dtype, or row identity changed")
            return arrays
    except (ValueError,KeyError,TypeError,OSError,EOFError,zipfile.BadZipFile) as exc:
        raise ValueError(f"research cache invalid: {path}; rebuild in a new cache path ({exc})") from exc
    finally:
        if isinstance(saved,np.lib.npyio.NpzFile):
            saved.close()


def _save_research_cache(path,identity,arrays):
    """Metadata and data share one atomic archive, including for legacy .npy paths."""
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    metadata={"definition":identity,"arrays":{k:_array_identity(v) for k,v in arrays.items()}}
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix=path.name+".",suffix=".tmp",delete=False) as handle:
            temporary=Path(handle.name)
            np.savez_compressed(handle,**arrays,cache_identity=canonical(metadata))
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_feature_cache(path,dataset,symbols):
    return _load_research_cache(path,feature_cache_identity(dataset,symbols),FEATURE_CACHE_FIELDS)


def make_data(path,cache,news_path=""):
    bars=load_bars(path)
    symbols=["BTCTMN","ETHTMN","SOLTMN","XRPTMN","BNBTMN"]
    cache=Path(cache)
    identity=feature_cache_identity(path,symbols)
    if cache.exists():
        saved=_load_research_cache(cache,identity,FEATURE_CACHE_FIELDS)
        by={s:{b.ts:b for b in bars if b.symbol==s} for s in symbols}
        times=sorted(set.intersection(*(set(v) for v in by.values())))
        data={k:saved[k] for k in FEATURE_CACHE_FIELDS}
        data.update({"times":times,"frames":[{s:by[s][t] for s in symbols} for t in times],"symbols":symbols})
    else:
        data=prepare(bars,symbols,72)
        fees=.0035; drag=.00225
        targets=np.full(len(data["y"]),np.nan)
        for n,((i,j),scale,atr) in enumerate(zip(data["locations"],data["scale"],data["atr"])):
            i,j=int(i),int(j); s=symbols[j]
            if not np.isfinite(data["y"][n]) or data["frames"][i+1][s].volume<=0:
                continue
            entry=data["frames"][i+1][s].open*(1+drag)
            stop=entry*(1-max(.03,4*atr/data["frames"][i][s].close))
            exit_price=data["frames"][i+72][s].close
            for k in range(i+1,i+73):
                b=data["frames"][k][s]
                if b.low<=stop and b.volume>0:
                    exit_price=min(b.open,stop)
                    break
            net=(exit_price*(1-drag)*(1-fees)/(entry*(1+fees)))-1
            targets[n]=np.clip(net/scale,-5,5)
        data["y"]=targets
        _save_research_cache(cache,identity,{k:data[k] for k in FEATURE_CACHE_FIELDS})
    data["cache_identity"]=identity
    index=NewsIndex(news_path) if news_path else None
    snapshots={}
    schema=NEWS_NAMES
    if index:
        for t in set(map(int,data["row_ts"])):
            values,context=index.at(t,reconstructed=True)
            snapshots[t]=values
            schema=context.get("feature_names",schema)
    news=np.asarray([snapshots.get(int(t),[0.]*6) for t in data["row_ts"]])
    data["news_coverage_fraction"]=float(np.mean(news[:,-1])) if len(news) else 0.
    data["news_x"]=np.column_stack((data["x"],news))
    data["news_feature_names"]=schema
    return data


def fit_meta(data,cutoff,l2,use_news=False,shuffle=False):
    xall=data["news_x"] if use_news else data["x"]
    # Non-overlapping sampling grid reduces, but does not remove, dependent labels.
    domain=(data["x"][:,6]>0)&(data["x"][:,15]>0)
    mature=np.isfinite(data["y"])&(data["maturity"]<=cutoff)&domain&(data["row_ts"]%21600==0)
    if use_news:
        mature &= data["news_x"][:,-1]>0
    train=mature&(data["maturity"]<=cutoff-28*86400)&(data["row_ts"]>=cutoff-180*86400)
    calibration=mature&(data["row_ts"]>=cutoff-28*86400)
    if train.sum()<300 or calibration.sum()<30:
        width=xall.shape[1]
        return {"artifact_type":"cost_meta","feature_names":FEATURE_NAMES,"symbols":data["symbols"],
                "interval":3600,"horizon":72,"return_weights":[0.]*width,"probability_weights":[0.]*width,
                "return_bias":-1.,"probability_bias":-35.,"return_rmse":0.,"training_samples":0,
                "calibration_samples":int(calibration.sum()),"eligible_fit_samples":int(train.sum()),
                "training_cutoff":int(cutoff),"latest_fit_maturity":None,"latest_calibration_maturity":None,
                "calibration_brier":None,"calibration_base_brier":None,"label_round_trip_cost":.0115,
                "l2":l2,"penalty":0.,"use_news":use_news,
                "news_feature_names":data["news_feature_names"] if use_news else [],
                "target_horizon":data.get("target_horizon",72),"research_only":True,
                "fit_valid":False,"abstain_reason":"insufficient fitting/calibration samples",
                "created_at":int(time.time()),"shuffle":shuffle}
    x=xall[train]; y=data["y"][train].copy()
    if shuffle:
        y=np.random.default_rng(73).permutation(y)
    weights=np.exp(-(cutoff-data["row_ts"][train])/(90*86400)*math.log(2))
    penalty=np.eye(x.shape[1])*l2; penalty[0,0]=1.
    rw=np.linalg.solve(x.T@(x*weights[:,None])+penalty,x.T@(y*weights))
    target=(y>0).astype(float)
    beta=np.zeros(x.shape[1])
    for _ in range(15):
        p=1/(1+np.exp(-np.clip(x@beta,-30,30)))
        curvature=np.maximum(.001,p*(1-p))*weights
        step=np.linalg.solve(x.T@(x*curvature[:,None])+penalty,x.T@((target-p)*weights)-penalty@beta)
        beta+=step
        if np.max(np.abs(step))<1e-7:
            break
    residual=data["y"][calibration]-xall[calibration]@rw
    bias=float(np.clip(np.mean(residual),-.25,.25))
    logits=xall[calibration]@beta
    truth=(data["y"][calibration]>0).astype(float); pbias=0.
    for _ in range(10):
        p=1/(1+np.exp(-np.clip(logits+pbias,-30,30)))
        pbias=float(np.clip(pbias+np.sum(truth-p)/max(.1,np.sum(p*(1-p))),-2,2))
    p=1/(1+np.exp(-np.clip(logits+pbias,-30,30)))
    return {"artifact_type":"cost_meta","feature_names":FEATURE_NAMES,"symbols":data["symbols"],
            "interval":3600,"horizon":72,"return_weights":rw.tolist(),"probability_weights":beta.tolist(),
            "return_bias":bias,"probability_bias":pbias,"return_rmse":float(np.sqrt(np.mean((residual-bias)**2))),
            "training_samples":int(train.sum()),"calibration_samples":int(calibration.sum()),
            "training_cutoff":int(cutoff),"latest_fit_maturity":int(data["maturity"][train].max()),
            "latest_calibration_maturity":int(data["maturity"][calibration].max()),
            "calibration_brier":float(np.mean((p-truth)**2)),"calibration_base_brier":float(np.mean((np.mean(truth)-truth)**2)),
            "label_round_trip_cost":.0115,"label":f"{data.get('target_horizon',72)}h stop-aware counterfactual net return; OHLCV assumptions",
            "l2":l2,"penalty":0.,"use_news":use_news,"news_feature_names":data["news_feature_names"] if use_news else [],
            "news_availability":"dated news-risk vintages with assumed 48h publication lag" if use_news else "RSS context only; no historical news fitted",
            "training_regime":"trend feature >0 and 72h momentum >0",
            "fit_valid":True,
            "target_horizon":data.get("target_horizon",72),
            "research_only":True,"created_at":int(time.time()),"shuffle":shuffle}


def retarget(data,horizon,path):
    """Keep causal features fixed, align net-profit labels with the holding horizon."""
    if not isinstance(horizon,int) or isinstance(horizon,bool) or horizon<=0:
        raise ValueError("target horizon must be a positive integer")
    if horizon==72:
        return {**data,"target_horizon":72}
    if "cache_identity" not in data:
        raise ValueError("retarget requires feature data with a validated cache identity")
    identity={"format":"wallex-research-cache","version":1,"kind":"retargeted-labels",
        "feature_identity":data["cache_identity"],"target_horizon":horizon,
        "source_sha256":digest_file(__file__),
        "input_arrays":{key:_array_identity(data[key]) for key in
            ("times","locations","row_ts","scale","atr")},
        "market_rows":_array_identity(np.asarray([
            [[getattr(frame[s],field) for field in ("ts","open","high","low","close","volume")]
             for s in data["symbols"]] for frame in data["frames"]],dtype=np.float64))}
    path=Path(path)
    if path.exists():
        y=_load_research_cache(path,identity,("y",))["y"]
        if y.shape!=data["y"].shape:
            raise ValueError(f"research cache invalid: {path}; target row count changed; rebuild in a new cache path")
    else:
        y=np.full(len(data["y"]),np.nan)
        for n,((i,j),scale,atr) in enumerate(zip(data["locations"],data["scale"],data["atr"])):
            i,j=int(i),int(j); s=data["symbols"][j]
            if i+horizon>=len(data["times"]) or data["times"][i+horizon]-data["times"][i]!=horizon*3600:
                continue
            if data["frames"][i+1][s].volume<=0:
                continue
            entry=data["frames"][i+1][s].open*1.00225
            stop=entry*(1-max(.03,4*atr/data["frames"][i][s].close))
            price=data["frames"][i+horizon][s].close
            for k in range(i+1,i+horizon+1):
                b=data["frames"][k][s]
                if b.low<=stop and b.volume>0:
                    price=min(b.open,stop); break
            y[n]=np.clip((price*.99775*.9965/(entry*1.0035)-1)/scale,-5,5)
        _save_research_cache(path,identity,{"y":y})
    return {**data,"y":y,"maturity":data["row_ts"]+horizon*3600,"target_horizon":horizon}


def stream(data,a,start,end,no_meta=False):
    mask=(data["row_ts"]>=start)&(data["row_ts"]<end)
    x=data["news_x"][mask] if a["use_news"] else data["x"][mask]
    net=(x@np.asarray(a["return_weights"])+a["return_bias"])*data["scale"][mask]
    p=1/(1+np.exp(-np.clip(x@np.asarray(a["probability_weights"])+a["probability_bias"],-30,30)))
    result={}; model_hash=hashlib.sha256(canonical(a).encode()).hexdigest()[:16]
    coverage=data["news_x"][mask,-1]
    for loc,v,prob,scale,atr,raw,covered in zip(data["locations"][mask],net,p,data["scale"][mask],data["atr"][mask],data["x"][mask],coverage):
        i,j=map(int,loc)
        if raw[6]<=0 or raw[15]<=0:
            prob=0.; v=min(v,-.0001)
        if a["use_news"] and not covered:
            prob=0.; v=min(v,-.0001)
        if no_meta:
            prob=1.; v=.10
        result[(i,data["symbols"][j])]=(float(v+a["label_round_trip_cost"]),float(a["return_rmse"]*scale),
            float(atr),a["training_samples"],model_hash,{"net_profit_probability":float(prob),"expected_net_return":float(v),"news_coverage":float(covered)})
    return result


def study(path,out,news_path="",resume_selection=None):
    root=Path(out); root.mkdir(parents=True,exist_ok=True)
    if (root/"selection.json").exists():
        raise ValueError("selection is frozen; cannot overwrite or retune this historical test")
    data=make_data(path,root.parent/"features-72h.npz",news_path)
    datasets={h:retarget(data,h,root.parent/f"targets-{h}.npy") for h in (72,336)}
    begin=data["times"][0]; end=data["times"][-1]+3600
    validation_start=begin+120*86400; test_start=begin+240*86400; test_end=begin+360*86400
    protocol={"source_sha256":source_hash(),"dataset_sha256":digest_file(path),
              "first_train_start":iso(begin),"validation_start":iso(validation_start),"test_start":iso(test_start),
              "test_end":iso(test_end),"previously_seen_comparison_start":iso(test_end),"end":iso(end),
              "method":"cost-aware ridge return + logistic net-profit filter, causal trend exits, no fixed 24h churn",
              "selection":"median four 30-day validation returns minus half median drawdown; >=8 sell fills and >=2 positive blocks",
              "hindsight_caveat":"method was designed after seeing recent 2026 failures; earlier historical test is not prospective proof",
              "news_coverage_fraction":data["news_coverage_fraction"],"live_orders":0,"account_access":False}
    protocol["target_alignment"]="72h/336h cost-aware labels matched to maximum holding duration; fixed before final tests"
    atomic_json(root/"protocol.json",protocol)
    shutil.copytree(Path(__file__).parent,root/"source_snapshot",ignore=shutil.ignore_patterns("__pycache__"),dirs_exist_ok=True)
    base=Config(symbols=data["symbols"],interval=3600,horizon=72,strategy="trend_meta",fee_bps=35,spread_bps=25,
                slippage_bps=10,min_notional=100000,settlement_buffer=2*3600,entry_buffer_bars=24,
                min_stop_fraction=.03,stop_atr=4.,cooldown=12,max_spread_bps=60,
                trend_entry_bps=25,trend_exit_bps=25,uncertainty_penalty=0.)
    blocks=[(validation_start+i*30*86400,validation_start+(i+1)*30*86400) for i in range(4)]
    use_news_options=[False,True] if data["news_coverage_fraction"]>.75 else [False]
    candidates=[]; original_selection=None
    if resume_selection:
        prior=Path(resume_selection)
        original_selection=json.loads(prior.read_text())
        candidates=json.loads((prior.parent/"candidates.json").read_text())
        use_news_options=[]
        protocol["resumed_selection_sha256"]=digest_file(prior)
        protocol["resumption_fix"]="insufficient calibration now abstains; grid/selection unchanged; no final-test metrics were available before failure"
        atomic_json(root/"protocol.json",protocol)
    for use_news in use_news_options:
        for l2 in (10,100):
            streams_by_hold={}
            for holding in (72,336):
                d=datasets[holding]; streams=[]
                for k,(left,right) in enumerate(blocks):
                    a=fit_meta(d,left,l2,use_news)
                    atomic_json(root/"validation_models"/f"l2-{l2}-h{holding}-news-{int(use_news)}-{k}.json",a)
                    streams.append(stream(d,a,left,right))
                streams_by_hold[holding]=streams
            for fast,slow in ((24,72),(48,168)):
                for holding in (72,336):
                    for probability in (.45,.55):
                        for edge in (0,25):
                            for trail in (0.,6.):
                                c=replace(base,trend_fast=fast,trend_slow=slow,max_holding_bars=holding,
                                          meta_min_probability=probability,edge_bps=edge,trailing_atr=trail,news_required=use_news)
                                reports=[simulate(data,c,streams_by_hold[holding][k],left,right)[0] for k,(left,right) in enumerate(blocks)]
                                returns=[comparable(r) for r in reports]
                                score=statistics.median(returns)-.5*statistics.median(r["max_drawdown_pct"] for r in reports)
                                sells=sum(r["sell_fills"] for r in reports)
                                eligible=sells>=8 and sum(v>0 for v in returns)>=2
                                candidates.append({"l2":l2,"use_news":use_news,"holding":holding,"config":asdict(c),"score":score,
                                                   "eligible":eligible,"returns_pct":returns,"sell_fills":sells,"reports":reports})
            print(json.dumps({"fitted_l2":l2,"news":use_news,"candidates":len(candidates)}),flush=True)
    atomic_json(root/"candidates.json",candidates)
    eligible=[r for r in candidates if r["eligible"]]
    best=max(eligible or candidates,key=lambda r:r["score"])
    selection={**best,"eligible_selection":bool(eligible),"frozen_at":int(time.time()),"candidate_count":len(candidates),
               "protocol_sha256":digest_file(root/"protocol.json")}
    if original_selection:
        assert best["config"]==original_selection["config"] and best["l2"]==original_selection["l2"]
        selection["frozen_at"]=original_selection["frozen_at"]
        selection["original_selection_sha256"]=digest_file(resume_selection)
    atomic_json(root/"selection.json",selection)
    print(json.dumps({"selected":{k:best[k] for k in ("l2","use_news","returns_pct","sell_fills","score")}}),flush=True)
    data=datasets[best["holding"]]
    c=Config(**best["config"]); full={}; no_meta={}; shuffled={}
    for k,left in enumerate(range(test_start,end,30*86400)):
        right=min(end,left+30*86400)
        a=fit_meta(data,left,best["l2"],best["use_news"])
        atomic_json(root/"test_models"/f"block-{k}.json",a)
        full.update(stream(data,a,left,right)); no_meta.update(stream(data,a,left,right,no_meta=True))
        placebo=fit_meta(data,left,best["l2"],best["use_news"],shuffle=True)
        shuffled.update(stream(data,placebo,left,right))
    report,curve=simulate(data,c,full,test_start,test_end,root/"unseen_history")
    stress=simulate(data,replace(c,fee_bps=70,spread_bps=50,slippage_bps=20),full,test_start,test_end,root/"cost_x2")[0]
    controls={"no_ml":simulate(data,replace(c,strategy="trend"),no_meta,test_start,test_end,root/"no_ml")[0],
              "shuffled":simulate(data,c,shuffled,test_start,test_end,root/"shuffled")[0]}
    paired={}
    if data["news_coverage_fraction"]>.75:
        for with_news in (False,True):
            paired_stream={}
            for left in range(test_start,test_end,30*86400):
                a=fit_meta(data,left,best["l2"],with_news)
                paired_stream.update(stream(data,a,left,min(test_end,left+30*86400)))
            name="with_news" if with_news else "without_news"
            paired[name]=simulate(data,replace(c,news_required=with_news),paired_stream,test_start,test_end,root/"news_ablation"/name)[0]
    comparison=simulate(data,c,full,end-45*86400,end,root/"previous_45d_comparison")[0]
    windows=[simulate(data,c,full,t,t+72*3600,root/"challenge_windows"/f"window-{k:02d}")[0]
             for k,t in enumerate(range(test_start,test_end-72*3600+1,72*3600))]
    final=fit_meta(data,end,best["l2"],best["use_news"])
    final["validation_eligible"]=bool(eligible)
    final["selection_sha256"]=digest_file(root/"selection.json")
    atomic_json(root/"model.json",final)
    config=replace(c,model_artifact=str(root/"model.json").replace("\\","/"),news_path=news_path,news_required=best["use_news"])
    atomic_json(root/"shadow-config.json",asdict(config))
    first=len(data["times"])-1
    while first>0 and data["times"][first]-data["times"][first-1]==3600:
        first-=1
    save_bars(root/"shadow-warmup.csv",[b for f in data["frames"][first:] for b in f.values()])
    result={"protocol":protocol,"selection":{k:selection[k] for k in ("l2","use_news","returns_pct","candidate_count","eligible_selection")},
            "unseen_history":report,"cost_x2":stress,"controls":controls,"previous_45d_comparison":comparison,
            "challenge_windows":{"count":len(windows),"positive":sum(comparable(r)>0 for r in windows),
                                 "median_return_pct":statistics.median(comparable(r) for r in windows),"reports":windows},
            "historical_positive":comparable(report)>0,"prospective_profitability_proven":False,
            "news_ablation":paired,"news_improvement_proven":False,"real_orders":0}
    result["release_status"]="RESEARCH_ONLY" if eligible else "ABSTAIN_NO_VALIDATED_CANDIDATE"
    atomic_json(root/"results.json",result)
    print(json.dumps({"unseen_return_pct":report["net_return_pct"],"drawdown_pct":report["max_drawdown_pct"],
                      "previous_45d_return_pct":comparison["net_return_pct"],"cost_x2_return_pct":stress["net_return_pct"],
                      "positive_3d_windows":result["challenge_windows"]["positive"],"windows":len(windows)}),flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--data",required=True); p.add_argument("--out",required=True)
    p.add_argument("--news",default=""); p.add_argument("--resume-selection")
    a=p.parse_args(); study(a.data,a.out,a.news,a.resume_selection)
