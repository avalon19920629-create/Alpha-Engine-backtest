"""Phase 2, Gate 1: test a *small* alternative-quote provider sample.

This is evidence collection only. Stooq and Yahoo adjusted closes are NOT
interchangeable; require overlap-return agreement before any score use.
It does not change the production engine or make trading recommendations.
"""
from __future__ import annotations
import io, json, pathlib, time, urllib.error, urllib.request
import numpy as np
import pandas as pd

OUT=pathlib.Path("phase2_targeted_quotes")
OUT.mkdir(exist_ok=True)
# 8 high-attention missing historical index constituents plus 2 controls.
NAMES=["AVB","EA","EQR","WBD","CTRA","BK","SIVB","FRC","NVDA","AVGO"]
CONTROLS={"NVDA","AVGO"}
MIN_POINTS=252
END="20261112"; START="20200101"

def fetch_stooq(symbol):
    url=f"https://stooq.com/q/d/l/?s={symbol.lower()}.us&i=d&d1={START}&d2={END}"
    request=urllib.request.Request(
        url,headers={"User-Agent":"Mozilla/5.0 (academic-research/1.0)",
                     "Accept":"text/csv,text/plain,*/*"})
    try:
        with urllib.request.urlopen(request,timeout=14) as res:
            raw=res.read(4*1024*1024)
        if not raw or len(raw)<30:
            return None, "empty_data_or_too_short"
        body=raw.decode("utf-8-sig",errors="replace")
        if not body.lstrip().lower().startswith("date,"):
            return None,"non_csv_response:"+body[:100].replace("\n"," ")
        df=pd.read_csv(io.StringIO(body))
        required={"Date","Close"}
        if not required.issubset(df.columns):
            return None,"missing_required_columns"
        df["Date"]=pd.to_datetime(df.Date,errors="coerce")
        df["Close"]=pd.to_numeric(df.Close,errors="coerce")
        df=df.dropna(subset=["Date","Close"]).sort_values("Date").set_index("Date")
        df=df[(df.Close>0)&(~df.index.duplicated(keep="last"))]
        if df.empty:
            return None,"no_valid_price_rows"
        return df[["Close"]],None
    except Exception as ex:
        return None,f"{type(ex).__name__}:{str(ex)[:180]}"

def main():
    from pathlib import Path
    frozen=pd.read_parquet("frozen_prices/prices_original.parquet")
    rows=[]
    allquotes=[]
    for name in NAMES:
        d,err=fetch_stooq(name)
        r={"ticker":name,"control":name in CONTROLS,
           "source":"stooq.com historical daily US symbol",
           "request":"2020-01-01 through 2026-11-12",
           "point_count":0 if d is None else len(d),
           "first_date":None if d is None else str(d.index.min().date()),
           "last_date":None if d is None else str(d.index.max().date()),
           "pre_2022_03_31_points":0 if d is None else int((d.index<=pd.Timestamp("2022-03-31")).sum()),
           "pre_2022_09_30_points":0 if d is None else int((d.index<=pd.Timestamp("2022-09-30")).sum()),
           "error":err}
        if d is not None:
            if name in CONTROLS and name in frozen:
                a=d.Close.pct_change(fill_method=None)
                b=frozen[name].pct_change(fill_method=None)
                both=pd.concat([a,b],axis=1,keys=["stooq","yahoo"]).dropna()
                r["control_shared_daily_returns"]=len(both)
                r["return_correlation"]=None if len(both)<40 else float(both.stooq.corr(both.yahoo))
                r["median_abs_daily_return_diff"]=None if len(both)<40 else float((both.stooq-both.yahoo).abs().median())
                r["large_return_differences_1pct"]=None if len(both)<40 else int(((both.stooq-both.yahoo).abs()>0.01).sum())
            series=d.Close.rename(name)
            allquotes.append(series)
        rows.append(r)
        print(f"{name}: {r['point_count']} rows; {err or 'success'}",flush=True)
        time.sleep(0.25)
    df=pd.DataFrame(rows)
    df.to_csv(OUT/"alternative_history_fetch_feasibility.csv",index=False)
    if allquotes:
        quoted=pd.concat(allquotes,axis=1).sort_index()
        quoted.to_parquet(OUT/"alternate_raw_stooq_closes_unadjusted_verification_needed.parquet")
    controls=df[df.control&df.point_count.gt(MIN_POINTS)]
    quality_ok=(len(controls)==2 and
                controls.return_correlation.gt(0.99).all() and
                controls.median_abs_daily_return_diff.lt(0.002).all())
    status={"status":"completed_source_feasibility_only",
            "source":"stooq.com public daily close",
            "tested_missing":len(NAMES)-len(CONTROLS),
            "available_missing":int(((~df.control)&df.point_count.ge(MIN_POINTS)).sum()),
            "trading_price_equivalence_established":bool(quality_ok),
            "unsafe_to_recalculate_cagr":True,
            "disclaimer":"Even matching daily returns for controls does NOT prove correct delisting histories or adjusted dividends for failed symbols. These quotes can be used only for a further as-of score-ranking sensitivity after identity and corporate-action checks."}
    (OUT/"pilot_metadata.json").write_text(json.dumps(status,ensure_ascii=False,indent=2))
    print(df.to_string(index=False),flush=True)
    print(json.dumps(status,ensure_ascii=False),flush=True)

if __name__=="__main__": main()
