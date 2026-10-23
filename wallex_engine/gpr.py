"""Download dated news-risk vintages from the research authors' public archive."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import io
import json
import math
import statistics
import time
from urllib.request import Request,urlopen

from .audit import atomic_json

GPR_NAMES=["gpr_log_level","gpr_log_mean7","gpr_z30","gpr_change7","gpr_log_vol7","gpr_coverage"]


def collect(root,start,end):
    import pandas as pd
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    directory=root/"gpr-directory.json"
    if not directory.exists():
        request=Request("https://api.github.com/repos/iacoviel/iacoviel.github.io/contents/gpr_archive_files",headers={"User-Agent":"wallex-public-research"})
        atomic_json(directory,json.load(urlopen(request,timeout=25)))
    items=json.loads(directory.read_text())
    versions=[]
    for item in items:
        name=item["name"]
        if not (name.startswith("data_gpr_daily_recent_") and name.endswith(".dta")):
            continue
        day=int(datetime.strptime(name[-12:-4],"%Y%m%d").replace(tzinfo=timezone.utc).timestamp())
        if start-15*86400<=day<=end:
            versions.append((day,item))
    rawdir=root/"gpr-raw"; rawdir.mkdir(exist_ok=True)
    def one(entry):
        day,item=entry; path=rawdir/item["name"]
        if not path.exists():
            raw=urlopen(Request(item["download_url"],headers={"User-Agent":"wallex-public-research"}),timeout=35).read(8_000_000)
            path.write_bytes(raw)
        raw=path.read_bytes()
        blob=hashlib.sha1(f"blob {len(raw)}\0".encode()+raw).hexdigest()
        if blob!=item["sha"]:
            raise ValueError("GPR archive blob hash mismatch")
        frame=pd.read_stata(io.BytesIO(raw))
        days=frame["date"]
        eligible=frame[days<pd.Timestamp(day,unit="s")].tail(30)
        values=[float(x) for x in eligible["GPRD"]]
        if len(values)<30 or any(not math.isfinite(v) or v<0 for v in values):
            raise ValueError("invalid vintage observations")
        logs=[math.log1p(x) for x in values]
        vec=[logs[-1]/5,statistics.fmean(logs[-7:])/5,
             max(-5,min(5,(logs[-1]-statistics.fmean(logs))/max(.05,statistics.pstdev(logs)))),
             logs[-1]-logs[-8],statistics.pstdev(logs[-7:]),1.]
        return {"kind":"gpr_vintage","id":item["name"],"vintage_date":day,
                "available_at":day+48*3600,"first_seen_at":int(time.time()),"features":vec,
                "latest_observation":int(eligible["date"].iloc[-1].timestamp()),
                "source_url":item["download_url"],"git_blob_sha1":blob,
                "feature_names":GPR_NAMES,"publication_assumption":"48h after dated archive version; local historical receipt unknown"}
    records=[]; errors=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[(entry,pool.submit(one,entry)) for entry in versions]
        for entry,future in futures:
            try:
                records.append(future.result())
            except Exception as exc:
                errors.append({"file":entry[1]["name"],"error":type(exc).__name__,"detail":str(exc)[:150]})
            if (len(records)+len(errors))%10==0:
                print(json.dumps({"gpr_vintages":len(records),"errors":len(errors)}),flush=True)
    path=root/"news.jsonl"
    existing=[json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    known={r["id"] for r in existing}
    with path.open("a",encoding="utf-8") as f:
        for r in sorted(records,key=lambda r:r["available_at"]):
            if r["id"] not in known:
                f.write(json.dumps(r)+"\n")
    result={"downloaded_at":int(time.time()),"vintages":len(records),"errors":errors,
            "source":"Caldara and Iacoviello (2022), Measuring Geopolitical Risk; author-hosted vintage archive",
            "license":"CC BY; source and authors credited","source_page":"https://www.matteoiacoviello.com/gpr.htm",
            "availability":"dated vintage + conservative 48h; historical publication-time assumption, not local receipt",
            "not_a_real_time_war_alert":True}
    atomic_json(root/"gpr-status.json",result)
    print(json.dumps(result),flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--out",required=True)
    p.add_argument("--start",type=int,required=True); p.add_argument("--end",type=int,required=True)
    a=p.parse_args(); collect(a.out,a.start,a.end)
