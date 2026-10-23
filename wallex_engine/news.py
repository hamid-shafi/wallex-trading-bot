"""Timestamped public news ingestion. News is data, never executable instructions."""
import argparse
from datetime import datetime,timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
import hashlib
import html
import json
import math
import re
import time
import unicodedata
from urllib.parse import urlencode,urlsplit,urlunsplit
from urllib.request import Request,urlopen
import xml.etree.ElementTree as ET

from .audit import atomic_json


FEEDS={
    "irna_politics":"https://www.irna.ir/rss/tp/5",
    "guardian_middle_east":"https://www.theguardian.com/world/middleeast/rss",
    "federal_reserve":"https://www.federalreserve.gov/feeds/press_all.xml",
    "bbc_middle_east":"https://feeds.bbci.co.uk/news/world/middle_east/rss.xml",
    "dw_world":"https://rss.dw.com/rdf/rss-en-world",
}
QUERIES={
    "conflict":"(Iran OR Israel OR Ukraine OR Russia) (war OR missile OR airstrike OR military)",
    "sanctions":"(Iran OR Russia) (sanctions OR embargo OR nuclear)",
    "diplomacy":"(Iran OR Israel OR Ukraine OR Russia) (ceasefire OR peace OR negotiations)",
    "macro":"(inflation OR \"interest rates\" OR \"Federal Reserve\")",
}
NEWS_NAMES=["conflict_share","sanctions_share","diplomacy_share","macro_share","conflict_change_7d","news_coverage"]
KEYWORDS={
    "conflict":r"\b(war|missile|airstrike|military|invasion|bombing)\b|جنگ|موشک|حمله نظامی|بمباران",
    "sanctions":r"\b(sanction\w*|embargo|nuclear)\b|تحریم|هسته[‌ ]ای",
    "diplomacy":r"\b(ceasefire|peace|negotiation\w*|diplomacy)\b|آتش[‌ -]?بس|مذاکر|صلح",
    "macro":r"\b(inflation|interest rate|federal reserve|monetary|central bank)\b|تورم|نرخ بهره|بانک مرکزی",
    "crypto_security":r"\b(crypto|bitcoin|exchange)\b.{0,50}\b(hack|breach|exploit|outage)\b|هک.*صرافی",
}


def normalized(text):
    text=unicodedata.normalize("NFKC",html.unescape(text)).lower().replace("ي","ی").replace("ك","ک")
    return re.sub(r"\s+"," ",re.sub(r"<[^>]+>"," ",text)).strip()


def timestamp(text):
    try:
        d=parsedate_to_datetime(text)
    except (ValueError,TypeError):
        d=datetime.fromisoformat(text.replace("Z","+00:00"))
    if d.tzinfo is None:
        raise ValueError("news timestamp needs a timezone")
    return int(d.timestamp())


def parse_feed(raw,source,seen_at):
    if len(raw)>4_000_000:
        raise ValueError("oversized feed")
    doc=ET.fromstring(raw)
    items=doc.findall(".//item")
    items+=doc.findall(".//{http://www.w3.org/2005/Atom}entry")
    out=[]
    for item in items:
        def get(name):
            return item.findtext(name) or item.findtext("{http://www.w3.org/2005/Atom}"+name) or ""
        title=normalized(get("title"))[:500]
        link=get("link")
        if not link:
            atom=item.find("{http://www.w3.org/2005/Atom}link")
            link=atom.get("href","") if atom is not None else ""
        when=get("pubDate") or get("published") or get("updated")
        try:
            published=timestamp(when)
        except (ValueError,TypeError,OverflowError):
            continue
        url=urlsplit(link)
        if not title or url.scheme not in ("http","https") or published>seen_at+300:
            continue
        link=urlunsplit((url.scheme,url.netloc,url.path,url.query,""))
        version=hashlib.sha256((source+"|"+link+"|"+title).encode()).hexdigest()
        duplicate=hashlib.sha256(re.sub(r"\W+","",title).encode()).hexdigest()
        out.append({"kind":"article","id":version,"duplicate_key":duplicate,"source":source,
                    "domain":url.hostname,"url":link,"title":title,"published_at":published,
                    "first_seen_at":seen_at,"available_at":max(published,seen_at),
                    "topics":{k:int(bool(re.search(p,title,re.I))) for k,p in KEYWORDS.items()},
                    "classification":"deterministic keyword features; not verified event facts"})
    return out


def fetch(url,timeout=15):
    req=Request(url,headers={"User-Agent":"WallexResearchBot/0.2 public-news","Accept":"application/xml,application/json,text/xml"})
    with urlopen(req,timeout=timeout) as r:
        raw=r.read(4_000_001)
        if len(raw)>4_000_000:
            raise ValueError("oversized news response")
        return raw


def collect(root,start=None,end=None):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    path=root/"news.jsonl"
    old=[json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    ids={x["id"] for x in old}; added=[]; status={}
    for source,url in FEEDS.items():
        seen=int(time.time())
        try:
            raw=fetch(url,12)
            raw_path=root/"raw"/f"{source}-{seen}.xml"
            raw_path.parent.mkdir(parents=True,exist_ok=True); raw_path.write_bytes(raw)
            rows=parse_feed(raw,source,seen)
            new=[r for r in rows if r["id"] not in ids]
            added+=new; ids.update(r["id"] for r in new)
            status[source]={"status":"ok","items":len(rows),"new":len(new),"url":url}
        except Exception as exc:
            status[source]={"status":"unavailable","error":type(exc).__name__,"detail":str(exc)[:160],"url":url}
    if start is not None and end is not None:
        for topic,query in QUERIES.items():
            params={"query":query,"mode":"timelinevolraw","format":"json","timelinesmooth":1,
                    "startdatetime":time.strftime("%Y%m%d%H%M%S",time.gmtime(start)),
                    "enddatetime":time.strftime("%Y%m%d%H%M%S",time.gmtime(end))}
            url="https://api.gdeltproject.org/api/v2/doc/doc?"+urlencode(params)
            try:
                raw=fetch(url,25); data=json.loads(raw)
                atomic_json(root/"raw"/f"gdelt-{topic}-{start}-{end}.json",data)
                series=data.get("timeline",[])
                points=series[0]["data"] if series else []
                if not points:
                    raise ValueError("empty historical news series")
                stamps=[int(datetime.strptime(p["date"],"%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).timestamp()) for p in points]
                period=min((b-a for a,b in zip(stamps,stamps[1:]) if b>a),default=86400)
                for ts,p in zip(stamps,points):
                    key=f"gdelt-{topic}-{ts}-{period}"
                    if key not in ids:
                        added.append({"kind":"bucket","id":key,"topic":topic,"bucket_start":ts,
                                      "bucket_end":ts+period,"available_at":ts+period+3600,
                                      "first_seen_at":int(time.time()),"count":float(p["value"]),
                                      "normalizer":float(p.get("norm",0)),"availability_kind":"reconstructed_provider_time_proxy"})
                        ids.add(key)
                status["gdelt_"+topic]={"status":"ok","points":len(points),"period_seconds":period}
            except Exception as exc:
                status["gdelt_"+topic]={"status":"unavailable","error":type(exc).__name__,"detail":str(exc)[:160]}
            time.sleep(2)
    with path.open("a",encoding="utf-8") as f:
        for row in added:
            f.write(json.dumps(row,ensure_ascii=False)+"\n")
    result={"collected_at":int(time.time()),"added":len(added),"sources":status,
            "key_used":False,"rss_availability":"max(publication, local first observation)",
            "historical_warning":"GDELT timeline is reconstructed; not proof of historical local receipt"}
    atomic_json(root/"collection-status.json",result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


class NewsIndex:
    def __init__(self,path):
        self.path=Path(path)
        self.stamp=None
        self.rows=[]
        self.refresh()

    def refresh(self):
        if not self.path.is_file():
            return
        stat=self.path.stat(); stamp=(stat.st_mtime_ns,stat.st_size)
        if stamp!=self.stamp:
            rows=[json.loads(s) for s in self.path.read_text(encoding="utf-8").splitlines() if s.strip()]
            self.rows=rows
            self.stamp=stamp

    def at(self,when,reconstructed=False):
        self.refresh()
        visible=[]
        for row in self.rows:
            available=row["available_at"]
            if row["kind"] in ("bucket","gpr_vintage") and not reconstructed:
                available=max(available,row["first_seen_at"])
            if available<=when:
                visible.append(row)
        values={}; ages=[]
        for topic in QUERIES:
            buckets=[r for r in visible if r["kind"]=="bucket" and r["topic"]==topic and 0<=when-r["available_at"]<=72*3600]
            r=max(buckets,key=lambda b:b["available_at"]) if buckets else None
            values[topic]=100*r["count"]/r["normalizer"] if r and r["normalizer"]>0 else 0.
            if r and r["normalizer"]>0:
                ages.append(when-r["available_at"])
        old=[r for r in visible if r["kind"]=="bucket" and r["topic"]=="conflict" and 6*86400<=when-r["available_at"]<=8*86400 and r["normalizer"]>0]
        prior=sum(100*r["count"]/r["normalizer"] for r in old)/len(old) if old else values["conflict"]
        # Preserve missingness. Missing coverage must never masquerade as calm news.
        vector=[values[t] for t in QUERIES]+[values["conflict"]-prior,float(len(ages)==len(QUERIES))]
        articles=[r for r in visible if r["kind"]=="article" and when-r["published_at"]<=86400]
        unique={r["duplicate_key"]:r for r in articles}
        context={"rss_topics":{k:sum(r["topics"][k] for r in unique.values()) for k in KEYWORDS},
                 "rss_sources":len({r["source"] for r in articles}),"articles":len(unique),
                 "historical_proxy":reconstructed,"news_coverage":vector[-1],
                 "latest_publication":max((r["published_at"] for r in articles),default=None)}
        if any(r["kind"]=="gpr_vintage" for r in self.rows):
            from .gpr import GPR_NAMES
            eligible=[r for r in visible if r["kind"]=="gpr_vintage" and 0<=when-r["available_at"]<=15*86400]
            latest=max(eligible,key=lambda r:r["available_at"]) if eligible else None
            vector=latest["features"] if latest else [0.]*len(GPR_NAMES)
            context.update({"feature_names":GPR_NAMES,"news_coverage":vector[-1],
                            "vintage":latest["id"] if latest else None,
                            "publication_time_assumed":bool(latest)})
        return vector,context


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--out",required=True)
    p.add_argument("--start",type=int); p.add_argument("--end",type=int)
    a=p.parse_args(); collect(a.out,a.start,a.end)
