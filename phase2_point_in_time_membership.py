"""Phase 2: partial point-in-time member-set sensitivity.

The source prices and 2026 survivor-stock pool are frozen. Historical member lists
restrict that contemporary pool at each 2022-2026 screen date. Thus this corrects
future index additions within the observed pool, but DOES NOT restore former
members or their prices. It is NOT survivorship-bias-free backtesting.
"""
from __future__ import annotations
import hashlib
import json
import math
import platform
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

import alpha_engine_backtest as engine

ROOT=Path("phase2_pit")
SRC=Path("historical_sp500/src/sp500_ticker_history")
SOURCE_SHA="6ff2a610543d12e56b0290c83ed972c8954b8b41"
OLD_RUN_ID=43107722015
START="2022-01-01"
END="2026-11-11"
RATIOS=(0,60,100)
SIZES=(6,12)

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def canonical(t):
    return str(t).upper().replace(".","-").strip()

def membership_asof(d, cache):
    d=pd.Timestamp(d)
    year=int(d.year)
    if year not in cache:
        file=SRC/f"sp500-ticker-changes-{year}.yaml"
        if not file.exists():
            raise RuntimeError(f"Missing historical membership year {year}")
        doc=yaml.safe_load(file.read_text())
        assert int(doc["year"]) == year
        base=set(canonical(s) for s in doc["tickers_on_Jan_1"])
        assert 450 <= len(base) <= 560, f"implausible membership at year start {year}: {len(base)}"
        cache[year]=(base,doc.get("changes") or {},sha256(file))
    members, changes, digest = cache[year]
    members=members.copy()
    for date, change in sorted(changes.items(),key=lambda x:str(x[0])):
        if pd.Timestamp(date)>d:
            break
        for t in change.get("difference",[]) or []:
            members.discard(canonical(t))
        for t in change.get("union",[]) or []:
            members.add(canonical(t))
    # Only clear, documented identity aliases; report all other unknown renames.
    # META was FB prior to June 2022, but Yahoo history for META uses the new name.
    if "FB" in members:
        members.add("META")
    # Three verified same-security symbol changes, confirmed by SEC/company filings:
    # BLL -> BALL effective 2022-05-10
    # ANTM -> ELV effective 2022-06-28
    # ABC -> COR effective 2023-08-30
    # Yahoo adjusted-close histories retain the newer symbol and include pre-rename prices.
    for historical, modern in (("BLL","BALL"),("ANTM","ELV"),("ABC","COR")):
        if historical in members:
            members.add(modern)
    if "SATS" in members and d>=pd.Timestamp("2026-06-24"):
        members.add("ECHO")
    assert 450 <= len(members) <= 560, f"implausible membership {d.date()}: {len(members)}"
    return members

def select_cached(prices, asof, region, us, jp, cfg, cache, benchmark):
    mode,r,n=cfg
    candidates=(us if region=="US" else jp)
    key=(str(asof),region,mode)
    if key not in cache:
        base=engine.score_universe(prices,[t for t in candidates if t in prices],asof)
        residual=engine.compute_residual_momentum_score(
            prices,base.index,asof,region,benchmark_mode=benchmark,method="simple")
        cache[key]=(base,residual)
    base,resid=cache[key]
    variant={"base_weight":1-r/100,"residual_weight":r/100}
    ranked=engine.combine_residual_score(base,resid,variant).copy()
    # Deterministic tie resolution: 2024-06-28 R60 has an exact APP/NVDA score tie.
    ranked["tie_symbol"]=ranked.index.astype(str)
    ranked=ranked.sort_values(
        ["Final_Score","residual_score","base_score","tie_symbol"],
        ascending=[False,False,False,True], kind="mergesort")
    return ranked.head(n//2)

def run_models(prices,us, jp, membership, membership_no_alias):
    returns=pd.DataFrame(index=prices.loc[START:END].index)
    dates=engine.rebalance_dates(prices.index,START,END)
    sigdate=[t for t in dates if (prices.index>t).any()]
    ret=prices.pct_change(fill_method=None)
    benchmark=engine.build_benchmark_modes()["broad_default"]
    cache={}
    picks=[]
    risks=[]
    for mode in ("current_2026_pool","historical_members_within_current_pool","historical_with_verified_aliases"):
        for r in RATIOS:
            for n in SIZES:
                col=f"{mode}_R{r}_N{n}"
                returns[col]=0.
                for i, t in enumerate(sigdate):
                    future=prices.index[prices.index>t]
                    trade=future[0]
                    next_t=sigdate[i+1] if i+1<len(sigdate) else pd.Timestamp(END)
                    hold=returns.index[(returns.index>=trade)&(returns.index<=next_t)]
                    members_for_t=(membership[pd.Timestamp(t)] if mode=="historical_with_verified_aliases" else
                                   membership_no_alias[pd.Timestamp(t)])
                    us_t=us if mode=="current_2026_pool" else [
                        x for x in us if canonical(x) in members_for_t
                    ]
                    pieces=[select_cached(prices,t,"US",us_t,jp,(mode,r,n),cache,benchmark),
                            select_cached(prices,t,"JP",us_t,jp,(mode,r,n),cache,benchmark)]
                    if any(not x.empty for x in pieces):
                        selected=pd.concat(pieces)
                    else: selected=pd.DataFrame()
                    if len(selected)!=n:
                        raise RuntimeError(f"short picks {col} {t.date()} got {len(selected)}")
                    inv=1/selected.Volatility
                    w=inv/inv.sum()
                    returns.loc[hold,col]=ret.reindex(hold)[list(w.index)].mul(w).sum(axis=1)
                    for ticker in w.index:
                        picks.append({
                            "mode":mode,"variant":f"R{r}_N{n}","screen_date":str(t.date()),
                            "trade_date":str(trade.date()),"ticker":ticker,
                            "region":"JP" if ticker.endswith(".T") else "US",
                            "Weight":float(w[ticker]),
                            "is_member_historical":bool(canonical(ticker) in membership[pd.Timestamp(t)]) if not ticker.endswith(".T") else None,
                        })
    return returns,pd.DataFrame(picks),sigdate

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    meta={"status":"started","source_repo":"shardul0701/SP500-Survivorship-bias-data-2004-2026",
          "source_commit_pinned":SOURCE_SHA,
          "price_run_id":OLD_RUN_ID,
          "membership_filter":"historical_member_intersection_with_Nov2026_current_US_universe",
          "what_is_corrected":"contemporary S&P500 members that had not joined at screen date",
          "what_is_not_corrected":"historical constituents that left index before Nov2026, missing/delisted price histories, Japanese universe historical membership, point-in-time corporate actions, fees, FX and executable trading",
          "baseline_score":"original code unchanged; R in 0/60/100; N in 6/12, US/JP equal counts, inverse-vol weights, quarterly rebalance, daily fixed weights",
          "known_ticker_aliases":["FB -> META","SATS -> ECHO after 2026-06-24"],
          "lookahead_warning":"Historical membership obtained using hindsight for audit only, not a trading feature at 2026-06-24"}
    try:
        md=json.loads(Path("frozen_prices/metadata.json").read_text())
        src=Path("frozen_prices/prices_original.parquet")
        assert sha256(src)==md["price_file_sha256"],"bad frozen input checksum"
        prices=pd.read_parquet(src)
        universe=json.loads(Path("frozen_prices/universe.json").read_text())
        prior=pd.read_csv("frozen_prices/real_six_cell_comparison.csv")
        oldselected=pd.read_csv("frozen_prices/real_selected_tickers.csv")
        us=[x for x in universe["US"] if x in prices and prices[x].notna().sum()>=252]
        jp=[x for x in universe["JP"] if x in prices and prices[x].notna().sum()>=252]
        assert len(us)==499 and len(jp)==108,(len(us),len(jp))
        sig=engine.rebalance_dates(prices.index,START,END)
        sig=[pd.Timestamp(t) for t in sig if (prices.index>t).any()]
        cache={}
        member={t:membership_asof(t,cache) for t in sig}
        assert pd.Timestamp("2026-06-30") in member
        verified_aliases=(("BLL","BALL"),("ANTM","ELV"),("ABC","COR"))
        member_noalias={}
        for t, matched in member.items():
            raw=matched.copy()
            for historical, modern in verified_aliases:
                if historical in raw:
                    raw.discard(modern)
            member_noalias[t]=raw
        # As of June 24, BE excluded; LITE and SNDK included.
        jun=membership_asof("2026-06-24",cache)
        assert "BE" not in jun and "LITE" in jun and "SNDK" in jun
        memrows=[]
        for t in sig:
            allowed=member_noalias[t]
            active=[s for s in us if canonical(s) in allowed]
            memrows.append({
                "screen_date":str(t.date()),"historical_sp500_member_count":len(allowed),
                "2026_pool_price_available":len(us),
                "intersection_price_available":len(active),
                "contemporary_pool_not_historical_count":len(us)-len(active),
            })
        pd.DataFrame(memrows).to_csv(ROOT/"historical_membership_coverage.csv",index=False)
        returns, selected, dates=run_models(prices,us,jp,member,member_noalias)
        assert len(dates)==19
        old=oldselected.copy()
        old["screen_date"]=pd.to_datetime(old.screen_date).dt.strftime("%Y-%m-%d")
        for r in RATIOS:
            for n in SIZES:
                oldcfg=old[(old.config_R==r)&(old.config_N==n)]
                ours=selected[(selected["mode"]=="current_2026_pool")&(selected.variant==f"R{r}_N{n}")]
                assert len(ours)==len(oldcfg),(r,n,len(ours),len(oldcfg))
                for date,g in oldcfg.groupby("screen_date"):
                    a=set(g.ticker)
                    b=set(ours[ours.screen_date==date].ticker)
                    if a!=b:
                        raise AssertionError(f"legacy selection mismatched R{r} N{n} {date}: {a^b}")
                m=engine.metrics(returns[f"current_2026_pool_R{r}_N{n}"])
                historic=prior[(prior.R==r)&(prior.N==n)].iloc[0]
                for field in ("CAGR","Annualized_Volatility","Max_Drawdown"):
                    assert abs(m[field]-float(historic[field]))<1e-9,(r,n,field,m[field],historic[field])
        # The original experiment is reproduced exactly before the experimental adjustment.
        audit=[]
        for r in RATIOS:
            for n in SIZES:
                var=f"R{r}_N{n}"
                a=engine.metrics(returns[f"current_2026_pool_{var}"])
                b=engine.metrics(returns[f"historical_members_within_current_pool_{var}"])
                c=engine.metrics(returns[f"historical_with_verified_aliases_{var}"])
                before=selected[(selected["mode"]=="historical_members_within_current_pool")&(selected.variant==var)]
                after=selected[(selected["mode"]=="historical_with_verified_aliases")&(selected.variant==var)]
                alias_changed=sum(
                    set(g.ticker)!=set(after[after.screen_date==date].ticker)
                    for date,g in before.groupby("screen_date"))
                base=selected[(selected["mode"]=="current_2026_pool")&(selected.variant==var)&(selected.region=="US")]
                filtered=selected[(selected["mode"]=="historical_members_within_current_pool")&(selected.variant==var)&(selected.region=="US")]
                join=base.merge(filtered,on="screen_date",suffixes=("_a","_b"))
                overlaps=base.groupby("screen_date").apply(lambda g: len(set(g.ticker)&set(filtered[filtered.screen_date==g.name].ticker)), include_groups=False)
                elig=base.apply(lambda z: canonical(z.ticker) in member_noalias[pd.Timestamp(z.screen_date)],axis=1)
                audit.append({
                    "variant":var,
                    "original_cagr":a["CAGR"],"filtered_cagr":b["CAGR"],
                    "alias_corrected_cagr":c["CAGR"],
                    "alias_effect_on_cagr_pp":100*(c["CAGR"]-b["CAGR"]),
                    "alias_screen_dates_with_selection_change":int(alias_changed),
                    "alias_corrected_mdd":c["Max_Drawdown"],
                    "delta_cagr_pp":100*(b["CAGR"]-a["CAGR"]),
                    "original_mdd":a["Max_Drawdown"],"filtered_mdd":b["Max_Drawdown"],
                    "delta_mdd_pp":100*(b["Max_Drawdown"]-a["Max_Drawdown"]),
                    "original_vol":a["Annualized_Volatility"],"filtered_vol":b["Annualized_Volatility"],
                    "original_annualized_sharpe_rf0":returns[f"current_2026_pool_{var}"].mean()/returns[f"current_2026_pool_{var}"].std()*np.sqrt(252),
                    "filtered_annualized_sharpe_rf0":returns[f"historical_members_within_current_pool_{var}"].mean()/returns[f"historical_members_within_current_pool_{var}"].std()*np.sqrt(252),
                    "us_original_total_picks":len(base),
                    "us_nonmember_picks":int((~elig).sum()),
                    "us_nonmember_share":float((~elig).mean()),
                    "mean_us_topn_overlap":float(overlaps.mean()),
                    "us_full_match_screen_dates":int((overlaps==n//2).sum()),
                    "us_screen_dates":len(overlaps),
                })
        pd.DataFrame(audit).to_csv(ROOT/"historical_membership_sensitivity.csv",index=False)
        selected.to_csv(ROOT/"selected_historical_membership_comparison.csv",index=False)
        returns.to_csv(ROOT/"daily_returns_comparison.csv",index_label="date")
        bydate=[]
        for d in dates:
            for r in RATIOS:
                for n in SIZES:
                    v=f"R{r}_N{n}"
                    a=selected[(selected["mode"]=="current_2026_pool")&(selected.variant==v)&(selected.screen_date==str(d.date()))&(selected.region=="US")]
                    b=selected[(selected["mode"]=="historical_members_within_current_pool")&(selected.variant==v)&(selected.screen_date==str(d.date()))&(selected.region=="US")]
                    bydate.append({
                        "screen_date":str(d.date()),"variant":v,
                        "us_current_picks": ";".join(a.ticker.tolist()),
                        "us_alias_corrected_picks":";".join(selected[
                            (selected["mode"]=="historical_with_verified_aliases")&
                            (selected.variant==v)&
                            (selected.screen_date==str(d.date()))&
                            (selected.region=="US")].ticker.tolist()),
                        "us_filtered_picks":";".join(b.ticker.tolist()),
                        "us_not_yet_member_picks":";".join(t for t in a.ticker if canonical(t) not in member_noalias[d]),
                        "overlap":len(set(a.ticker)&set(b.ticker)),
                    })
        pd.DataFrame(bydate).to_csv(ROOT/"change_by_screen_date.csv",index=False)
        meta["status"]="completed_alias_effort_pilot"
        meta["verified_aliases"]=["BLL->BALL","ANTM->ELV","ABC->COR"]
        meta["tie_break_rule"]="Final_Score desc, residual_score desc, base_score desc, ticker asc; mergesort"
        meta["comparison"]="Both historical membership modes are computed with the same frozen stock prices and deterministic tie rules."
        meta["source_year_file_sha256"]={str(y):z[2] for y,z in cache.items()}
        meta["input_price_sha256"]=md["price_file_sha256"]
        meta["pandas"]=pd.__version__
        meta["python"]=platform.python_version()
        meta["distinct_screen_dates"]=len(dates)
        meta["validation"]="all 6 current-pool runs numerically and in positions reproduce frozen original"
        print(pd.DataFrame(audit).to_string(index=False),flush=True)
        print("PIT members example June24/30",len(jun),"BE", "BE" in jun, "LITE","LITE" in jun,flush=True)
    except Exception as e:
        meta["status"]="failed_no_investment_conclusions"
        meta["error"]=str(e)
        raise
    finally:
        (ROOT/"metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2,default=str),encoding="utf-8")

if __name__=="__main__":
    main()
