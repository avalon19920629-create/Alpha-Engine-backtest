"""Phase 2 forensic continuation: recovery audit for former S&P500 members.

Important: this is a *partial*, provenance-qualified sensitivity experiment.
Not survivorship-free: permanently delisted stocks, renamed symbols and Yahoo
historical datasets can remain inaccessible or ambiguous. The primary result
is the extent of missing historical constituents. Performance with additional
retrievable shares is conditional and exploratory.
"""
from __future__ import annotations
import hashlib, json, math, platform
from pathlib import Path
import numpy as np
import pandas as pd
import alpha_engine_backtest as engine
import phase2_point_in_time_membership as pit

OUT=Path("phase2_recovery")
START="2022-01-01"
END="2026-11-11"
RATIOS=(0,60,100)
SIZES=(6,12)
MAX_STALE_CALENDAR_DAYS=8

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as fp:
        for b in iter(lambda:fp.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()

def is_available_at(prices, tick, t):
    if tick not in prices: return False
    s=prices.loc[:t,tick].dropna()
    if len(s)<252: return False
    if (pd.Timestamp(t)-pd.Timestamp(s.index[-1])).days>MAX_STALE_CALENDAR_DAYS: return False
    if not np.isfinite(s.iloc[-1]) or s.iloc[-1]<=0: return False
    return True

def build_calendar_coverage(prices, original, recovered, date_list, member):
    rows,missingrows=[],[]
    for date in date_list:
        members=member[date]
        in_original={x for x in original if pit.canonical(x) in members and is_available_at(prices,x,date)}
        in_recovered={x for x in recovered if pit.canonical(x) in members and is_available_at(prices,x,date)}
        missing_raw=members-set(pit.canonical(x) for x in original)
        # Alias exception: pre-2022 FB history may map to META existing prices
        if "FB" in missing_raw and "META" in original:
            missing_raw.discard("FB")
        rows.append({
            "screen_date":str(date.date()),
            "official_historical_member_count_proxy":len(members),
            "historical_eligible_in_original":len(in_original),
            "former_members_with_recovered_prices":len(in_recovered),
            "historical_eligible_total_partial":len(in_original|in_recovered),
            "historical_names_uncovered_total":len(missing_raw-set(pit.canonical(x) for x in in_recovered)),
            "membership_missing_from_original":len(missing_raw),
            "restored_asof_date":len(in_recovered),
            "fraction_members_with_historical_quotes_partial":len(in_original|in_recovered)/len(members),
            "uncovered_symbols":";".join(sorted(missing_raw-set(pit.canonical(x) for x in in_recovered))),
        })
        for sym in sorted(missing_raw):
            missingrows.append({"screen_date":str(date.date()),"ticker":sym,"recovered":sym in set(pit.canonical(x) for x in in_recovered)})
    return pd.DataFrame(rows),pd.DataFrame(missingrows)

def one_portfolio(prices, prices_ret, us, jp, t, n, r, bmode, memo):
    parts=[]
    for region,tickers in (("US",us),("JP",jp)):
        key=(str(t.date()),region)
        if key not in memo:
            elig=[x for x in tickers if is_available_at(prices,x,t)]
            base=engine.score_universe(prices,elig,t)
            residual=engine.compute_residual_momentum_score(prices,base.index,t,region,benchmark_mode=bmode,method="simple")
            memo[key]=(base,residual)
        base,residual=memo[key]
        v={"base_weight":1-r/100,"residual_weight":r/100}
        p=engine.combine_residual_score(base,residual,v).head(n//2)
        p["Region"]=region
        parts.append(p)
    p=pd.concat(parts)
    if len(p)!=n:
        raise RuntimeError(f"Short historical holdings {t} R{r}N{n}: {len(p)}")
    inv=1/p.Volatility
    p["Weight"]=inv/inv.sum()
    return p

def simulate(prices, us, jp, date_list, membership):
    ret=prices.pct_change(fill_method=None)
    out=pd.DataFrame(index=prices.loc[START:END].index)
    selections=[]
    bmode=engine.build_benchmark_modes()["broad_default"]
    orig_names=set(us)
    for r in RATIOS:
        for n in SIZES:
            k=f"R{r}_N{n}"
            out[k]=0.
            memo={}
            for j,t in enumerate(date_list):
                trade=prices.index[prices.index>t][0]
                next_t=date_list[j+1] if j+1<len(date_list) else pd.Timestamp(END)
                interval=out.index[(out.index>=trade)&(out.index<=next_t)]
                names=[x for x in us if pit.canonical(x) in membership[t]]
                p=one_portfolio(prices,ret,names,jp,t,n,r,bmode,memo)
                w=p.Weight
                out.loc[interval,k]=ret.reindex(interval)[list(w.index)].mul(w).sum(axis=1)
                for ticker in p.index:
                    selections.append({
                        "variant":k,"screen_date":str(t.date()),"ticker":ticker,
                        "region":p.loc[ticker,"Region"],"weight":float(p.loc[ticker,"Weight"]),
                        "is_recovered_former_member":ticker not in orig_names
                    })
    return out,pd.DataFrame(selections)

def main():
    OUT.mkdir(exist_ok=True)
    info={
      "status":"started",
      "reference_run":"43107722015",
      "historic_membership_source":"shardul0701/SP500-Survivorship-bias-data-2004-2026",
      "historic_membership_source_sha":"6ff2a610543d12e56b0290c83ed972c8954b8b41",
      "important_limits":[
        "This recovers accessible historical members only, NOT all historical firms.",
        "Yahoo may retain quoted survivors but drop bankrupt/delisted or change company identity for reused symbols.",
        "No complete corporate action / ticker-identity audit for the additional names.",
        "Contemporaneous historical membership feed is retrospective community reconstruction.",
        "Historical Japanese universe is unchanged and not PIT membership controlled.",
        "The old daily-weight rebalancing accounting and cross-market missing returns are unchanged.",
        "FX, trading costs, taxes, and bid-ask spread omitted.",
        "A strong or weak model CAGR remains sample-dependent and not investable expectation.",
      ]
    }
    try:
        md=json.loads(Path("frozen_prices/metadata.json").read_text())
        basefile=Path("frozen_prices/prices_original.parquet")
        assert sha256(basefile)==md["price_file_sha256"]
        frozen=pd.read_parquet(basefile)
        origuniverse=json.loads(Path("frozen_prices/universe.json").read_text())
        original=[x for x in origuniverse["US"] if x in frozen and frozen[x].notna().sum()>=252]
        jp=[x for x in origuniverse["JP"] if x in frozen and frozen[x].notna().sum()>=252]
        date_list=[pd.Timestamp(t) for t in engine.rebalance_dates(frozen.index,START,END)
                   if (frozen.index>t).any()]
        assert len(original)==499 and len(jp)==108 and len(date_list)==19
        _cache={}
        membership={t:pit.membership_asof(t,_cache) for t in date_list}
        canonical_current={pit.canonical(x) for x in original}
        historical_union=set().union(*[membership[t] for t in date_list])
        extra=sorted(historical_union-canonical_current)
        # META ticker's historic FB alias already matched in pit.membership_asof.
        if "FB" in extra and "META" in original: extra.remove("FB")
        # Exclude non-equity ticker symbols and complicated identity aliases only where proven;
        # all unconfirmed identities are retained in audit but NOT treated as proof.
        (OUT/"historical_missing_symbols.json").write_text(
            json.dumps({"screen_dates":[str(x.date()) for x in date_list],
                        "missing_from_original_stock_pool":extra},indent=2))
        print(f"Former member symbols to attempt: {len(extra)}",flush=True)

        # yfinance convenience download automatically returns adjusted closes.
        # Avoid a source-price revision for the already-frozen 2026 dataset.
        downloaded=engine.download_live_prices(extra,START,END,batch_size=45)
        if not downloaded.empty:
            downloaded.index=pd.to_datetime(downloaded.index).tz_localize(None).astype("datetime64[ns]")
            downloaded=downloaded.reindex(frozen.index)
            downloaded=downloaded.drop(columns=[x for x in downloaded.columns if x in frozen],errors="ignore")
        downloaded.to_parquet(OUT/"downloaded_historical_former_quotes.parquet")
        data=pd.concat([frozen,downloaded],axis=1)
        assert data.columns.is_unique and data.index.equals(frozen.index)
        for col in frozen:
            pd.testing.assert_series_equal(data[col],frozen[col],check_exact=True)
        summary=[]
        for t in extra:
            s=data[t].dropna() if t in data else pd.Series(dtype=float)
            dates=[d for d in date_list if pit.canonical(t) in membership[d]]
            eligible_dates=[d for d in dates if is_available_at(data,t,d)]
            summary.append({
                "ticker":t,
                "downloaded_column":t in downloaded,
                "price_points":len(s),
                "earliest_quote_date":str(s.index.min().date()) if len(s) else None,
                "last_quote_date":str(s.index.max().date()) if len(s) else None,
                "membership_screen_dates":len(dates),
                "eligible_screen_dates":len(eligible_dates),
                "status":"historical_price_usable_some_dates" if eligible_dates else "not_usable_at_historical_screen_dates",
                "caution":"no_ticker_identity_or_reuse_verification"
            })
        x=pd.DataFrame(summary)
        x.to_csv(OUT/"historical_symbol_quote_recovery.csv",index=False)
        recovered=x.loc[x.eligible_screen_dates>0,"ticker"].tolist()
        coverage, miss=build_calendar_coverage(data,original,recovered,date_list,membership)
        coverage.to_csv(OUT/"historical_membership_coverage_extended.csv",index=False)
        miss.to_csv(OUT/"uncovered_historical_symbols_by_screen.csv",index=False)
        print("coverage",coverage[[
          "screen_date","historical_eligible_in_original",
          "former_members_with_recovered_prices","historical_eligible_total_partial",
          "historical_names_uncovered_total"]].to_string(index=False),flush=True)
        print("Ticker recovery",len(extra),len(recovered),"available historically",flush=True)
        assert len(recovered) <= len(extra)
        assert coverage["former_members_with_recovered_prices"].sum()>=0
        us=original+recovered
        ret, selected=simulate(data,us,jp,date_list,membership)
        old=pd.read_csv("phase2_reference/historical_membership_sensitivity.csv")
        compare=[]
        for r in RATIOS:
            for n in SIZES:
                key=f"R{r}_N{n}"
                m=engine.metrics(ret[key])
                prior=old.loc[old.variant==key].iloc[0]
                compare.append({
                    "variant":key,
                    "original_current_pool_cagr":prior.original_cagr,
                    "restrict_current_pool_cagr":prior.filtered_cagr,
                    "add_retrievable_former_members_cagr":m["CAGR"],
                    "delta_from_restricted_pp":100*(m["CAGR"]-prior.filtered_cagr),
                    "delta_from_unrestricted_pp":100*(m["CAGR"]-prior.original_cagr),
                    "restricted_mdd":prior.filtered_mdd,
                    "recovery_mdd":m["Max_Drawdown"],
                    "restored_holdings_count":len(selected[
                        (selected.variant==key)&(selected.is_recovered_former_member)]),
                    "restored_screen_count":selected[
                        (selected.variant==key)&(selected.is_recovered_former_member)].screen_date.nunique()
                })
        pd.DataFrame(compare).to_csv(OUT/"partial_recovery_sensitivity.csv",index=False)
        ret.to_csv(OUT/"daily_returns_partial_recovery.csv",index_label="date")
        selected.to_csv(OUT/"holdings_partial_recovery.csv",index=False)
        info.update({
            "status":"completed_partial_quotes_exploration",
            "historical_missing_distinct_ticker_symbols":len(extra),
            "quote_eligible_distinct_symbols":len(recovered),
            "quoted_symbols_unverifiable_identity":len(recovered),
            "restored_price_digest":sha256(OUT/"downloaded_historical_former_quotes.parquet"),
            "original_price_sha256":md["price_file_sha256"],
            "original_price_columns_unchanged":True,
            "screen_dates":len(date_list),
            "date_start":str(date_list[0].date()),
            "date_end":str(date_list[-1].date()),
            "pandas":pd.__version__,"python":platform.python_version()
        })
        print(pd.DataFrame(compare).to_string(index=False),flush=True)
    except Exception as exc:
        info["status"]="failed_no_conclusion"
        info["error"]=repr(exc)
        raise
    finally:
        (OUT/"metadata.json").write_text(json.dumps(info,indent=2,ensure_ascii=False,default=str))

if __name__=="__main__":
    main()
