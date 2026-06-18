"""Point-in-time Alpha Engine backtest audit (research only; no trading connection)."""
from __future__ import annotations
import argparse, importlib.util, logging, json, time
from pathlib import Path
import numpy as np
import pandas as pd

LOG=logging.getLogger("alpha_backtest")
INSUFFICIENT_HISTORY_WARNINGS={}
INSUFFICIENT_HISTORY_LOG_LIMIT=10

def reset_insufficient_history_warnings():
    INSUFFICIENT_HISTORY_WARNINGS.clear()

def get_insufficient_history_summary():
    return pd.DataFrame([{"ticker":t,"reason":r,"count":c} for (t,r),c in sorted(INSUFFICIENT_HISTORY_WARNINGS.items())])

def _record_insufficient_history(ticker, reason="insufficient_history"):
    key=(ticker,reason); INSUFFICIENT_HISTORY_WARNINGS[key]=INSUFFICIENT_HISTORY_WARNINGS.get(key,0)+1
    if len(INSUFFICIENT_HISTORY_WARNINGS)<=INSUFFICIENT_HISTORY_LOG_LIMIT and INSUFFICIENT_HISTORY_WARNINGS[key]==1:
        LOG.warning("%s: %s",reason,ticker)

BENCHMARKS={"SPY":"SPY","QQQ":"QQQ","VT":"VT","TOPIX":"1306.T"}
OUTPUT_FILES=("backtest_summary.csv","selected_tickers_by_period.csv","annual_returns.csv","monthly_returns.csv","drawdown_report.csv","turnover_report.csv","momentum_alpha_backtest_report.md")

def get_live_universe():
    """Load the same current US/JP universe used by Alpha-Engine.py."""
    path=Path(__file__).with_name("Alpha-Engine.py")
    spec=importlib.util.spec_from_file_location("alpha_engine_live",path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.get_tickers_lumus()

def _close_frame(raw, requested):
    if raw is None or raw.empty:return pd.DataFrame()
    if isinstance(raw.columns,pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0): raw=raw["Close"]
        elif "Close" in raw.columns.get_level_values(-1): raw=raw.xs("Close",axis=1,level=-1)
        else:return pd.DataFrame()
    elif "Close" in raw.columns: raw=raw[["Close"]].rename(columns={"Close":requested[0]})
    raw=raw.apply(pd.to_numeric,errors="coerce").dropna(axis=1,how="all")
    raw.columns=[str(x) for x in raw.columns]
    return raw

def download_live_prices(tickers,start,end,batch_size=100):
    """Download adjusted closes without allowing one bad ticker to abort live mode."""
    import yfinance as yf
    fetch_start=(pd.Timestamp(start)-pd.offsets.BDay(400)).date().isoformat(); fetch_end=(pd.Timestamp(end)+pd.Timedelta(days=1)).date().isoformat(); frames=[]
    for i in range(0,len(tickers),batch_size):
        batch=list(dict.fromkeys(tickers[i:i+batch_size]))
        try: frame=_close_frame(yf.download(batch,start=fetch_start,end=fetch_end,auto_adjust=True,progress=False,group_by="column",threads=True),batch)
        except Exception as exc: LOG.warning("price download failed for batch %s: %s",batch,exc); continue
        missing=sorted(set(batch)-set(frame.columns))
        for ticker in missing: LOG.warning("price unavailable: %s",ticker)
        if not frame.empty:frames.append(frame)
    if not frames:return pd.DataFrame()
    return pd.concat(frames,axis=1).loc[:,lambda x:~x.columns.duplicated()].sort_index()

def exposure_for_regimes(us, jp):
    return 1.0 if us==jp=="BULL" else 0.6 if "BULL" in (us,jp) else 0.2

def asof_prices(prices, as_of_date):
    """Hard point-in-time boundary: never return an observation after as_of_date."""
    return prices.loc[:pd.Timestamp(as_of_date)].copy()

def regime_at(series, as_of_date, window=200):
    s=asof_prices(series,as_of_date).dropna()
    return "BULL" if len(s)>=window and s.iloc[-1]>s.iloc[-window:].mean() else "BEAR"

def score_universe(prices, tickers, as_of_date, min_history=252):
    hist=asof_prices(prices,as_of_date).ffill()
    hist=hist[[t for t in tickers if t in hist.columns]]
    rows=[]
    for t in hist.columns:
        s=hist[t].dropna(); r=s.pct_change().dropna()
        if len(s)<min_history: _record_insufficient_history(t); continue
        vol=r.tail(252).std()*np.sqrt(252)
        if not np.isfinite(vol) or vol<=0: continue
        mom=sum(w*(s.iloc[-1]/s.iloc[-n]-1) for n,w in ((252,3),(126,2),(63,1)))/6
        pos=r.tail(252)[r.tail(252)>0].mean(); neg=abs(r.tail(252)[r.tail(252)<0].mean())
        rows.append((t,mom/vol,pos/neg if neg else 1,(1/(s.iloc[-1]/s.tail(252).max()))/vol,vol,mom))
    d=pd.DataFrame(rows,columns=["Ticker","Efficiency","Quality","Valuation_Alt","Volatility","Composite_Ret"]).set_index("Ticker")
    if d.empty:return d
    z=lambda x:(x-x.mean())/(x.std() if x.std() else 1)
    d["Total_Score"]=.4*z(d.Efficiency)+.4*z(d.Quality)+.2*z(d.Valuation_Alt)
    return d.sort_values("Total_Score",ascending=False)

def select_portfolio(prices, us, jp, as_of_date, top_n=6):
    frames=[]
    for region,tickers in (("US",us),("JP",jp)):
        d=score_universe(prices,[t for t in tickers if t in prices],as_of_date).head(top_n).copy()
        if len(d)<top_n: LOG.warning("candidate shortage: %s %s/%s",region,len(d),top_n)
        d["Region"]=region; frames.append(d)
    p=pd.concat(frames) if any(not x.empty for x in frames) else pd.DataFrame()
    if not p.empty:
        inv=1/p.Volatility; p["Weight"]=inv/inv.sum()
    return p

def max_drawdown(returns):
    wealth=(1+pd.Series(returns).fillna(0)).cumprod(); return float((wealth/wealth.cummax()-1).min()) if len(wealth) else np.nan

def cagr(returns, periods=252):
    r=pd.Series(returns).dropna()
    return float((1+r).prod()**(periods/len(r))-1) if len(r) else np.nan

def metrics(r):
    r=pd.Series(r).dropna(); ann=cagr(r); vol=r.std()*np.sqrt(252); dd=max_drawdown(r); down=r[r<0].std()*np.sqrt(252)
    monthly=(1+r).resample("ME").prod()-1; annual=(1+r).resample("YE").prod()-1
    return {"CAGR":ann,"Annualized_Volatility":vol,"Max_Drawdown":dd,"Sharpe":ann/vol if vol else np.nan,"Calmar":ann/abs(dd) if dd else np.nan,"Sortino":ann/down if down else np.nan,"Total_Return":(1+r).prod()-1,"Best_Year":annual.max(),"Worst_Year":annual.min(),"Monthly_Win_Rate":(monthly>0).mean()}

def rebalance_dates(index,start,end):
    idx=index[(index>=pd.Timestamp(start))&(index<=pd.Timestamp(end))]
    return pd.DatetimeIndex(pd.Series(idx,index=idx).groupby(idx.to_period("Q")).max().values)

def run_backtest(prices, us, jp, start, end):
    prices=prices.sort_index(); rets=prices.pct_change(); dates=rebalance_dates(prices.index,start,end); records=[]; strategy={"Alpha_Always":pd.Series(0.,index=prices.loc[start:end].index),"Alpha_Regime_Filter":pd.Series(0.,index=prices.loc[start:end].index)}; turns=[]; prev={}
    for i,t in enumerate(dates):
        future=prices.index[prices.index>t]
        if not len(future):continue
        trade=future[0]; next_t=dates[i+1] if i+1<len(dates) else pd.Timestamp(end); hold=strategy["Alpha_Always"].index[(strategy["Alpha_Always"].index>=trade)&(strategy["Alpha_Always"].index<=next_t)]
        p=select_portfolio(prices,us,jp,t); weights=p.Weight.to_dict() if not p.empty else {}; usreg=regime_at(prices["^GSPC"],t) if "^GSPC" in prices else "BEAR"; jpreg=regime_at(prices["^N225"],t) if "^N225" in prices else "BEAR"; exp=exposure_for_regimes(usreg,jpreg)
        base=rets.reindex(hold)[list(weights)].mul(pd.Series(weights)).sum(axis=1) if weights else pd.Series(0.,index=hold)
        strategy["Alpha_Always"].loc[hold]=base; strategy["Alpha_Regime_Filter"].loc[hold]=base*exp
        turnover=sum(abs(weights.get(k,0)-prev.get(k,0)) for k in set(weights)|set(prev))/2; turns.append({"screen_date":t,"trade_date":trade,"turnover":turnover}); prev=weights
        for ticker,row in p.iterrows(): records.append({"screen_date":t,"trade_date":trade,"ticker":ticker,**row.to_dict(),"US_Regime":usreg,"JP_Regime":jpreg,"Regime_Exposure":exp})
    return strategy,pd.DataFrame(records),pd.DataFrame(turns)

def demo_prices():
    idx=pd.bdate_range("2018-01-01","2025-12-31"); rng=np.random.default_rng(42); names=[*(f"US{i}" for i in range(8)),*(f"JP{i}.T" for i in range(8)),"^GSPC","^N225",*BENCHMARKS.values()]
    return pd.DataFrame({n:100*np.exp(np.cumsum(rng.normal(.00025+(i%5)*.00004,.012,len(idx)))) for i,n in enumerate(dict.fromkeys(names))},index=idx)


MINERVINI_VARIANTS=(
    {"name":"Baseline","base_weight":1.0,"residual_weight":0.0,"vcp_weight":0.0},
    {"name":"Residual_20","base_weight":0.8,"residual_weight":0.2,"vcp_weight":0.0},
    {"name":"Residual_30","base_weight":0.7,"residual_weight":0.3,"vcp_weight":0.0},
    {"name":"VCP_10","base_weight":0.9,"residual_weight":0.0,"vcp_weight":0.1},
    {"name":"VCP_20","base_weight":0.8,"residual_weight":0.0,"vcp_weight":0.2},
    {"name":"Residual_20_VCP_10","base_weight":0.7,"residual_weight":0.2,"vcp_weight":0.1},
    {"name":"Residual_30_VCP_10","base_weight":0.6,"residual_weight":0.3,"vcp_weight":0.1},
    {"name":"Residual_20_VCP_20","base_weight":0.6,"residual_weight":0.2,"vcp_weight":0.2},
)

def _rank01(x):
    x=pd.Series(x,dtype=float)
    if x.dropna().empty:return pd.Series(0.5,index=x.index)
    return x.rank(pct=True).fillna(0.5)

def _pick_benchmark(prices, region):
    candidates=("SPY","^GSPC") if region=="US" else ("1306.T","^TOPX","^N225")
    return next((c for c in candidates if c in prices.columns), None)

def compute_residual_momentum_score(prices, tickers, as_of_date, region="US", windows=(63,126,252), benchmark_mode=None, method="simple"):
    """Cross-sectional simple benchmark-adjusted momentum; missing benchmark falls back to neutral scores."""
    hist=asof_prices(prices,as_of_date).ffill(); tickers=[t for t in tickers if t in hist.columns]
    bench=_resolve_benchmark(hist, region, benchmark_mode)
    rows=[]
    if not tickers:return pd.DataFrame(columns=["Ticker","residual_score"])
    for t in tickers:
        s=hist[t].dropna(); vals=[]
        for n,w in zip(windows,(1,2,3)):
            if len(s)>n and bench and bench in hist:
                b=hist[bench].dropna()
                if len(b)>n:
                    if method=="beta_adjusted":
                        vals.append(w*compute_beta_adjusted_residual_score(s, b, n))
                    else:
                        vals.append(w*((s.iloc[-1]/s.iloc[-n]-1)-(b.iloc[-1]/b.iloc[-n]-1)))
        rows.append((t, np.nan if not vals else sum(vals)/sum((1,2,3)[:len(vals)])))
    d=pd.DataFrame(rows,columns=["Ticker","residual_raw"]).set_index("Ticker")
    d["residual_score"]=_rank01(d.residual_raw)
    return d

def compute_vcp_proxy_score(prices, tickers, as_of_date):
    """Rank-based VCP proxy score; weak VCP names are scored lower, never hard-filtered."""
    hist=asof_prices(prices,as_of_date).ffill(); rows=[]
    for t in [x for x in tickers if x in hist.columns]:
        s=hist[t].dropna(); r=s.pct_change().dropna()
        if len(s)<60: continue
        ma20=s.tail(min(20,len(s))).mean(); ma50=s.tail(min(50,len(s))).mean(); ma150=s.tail(min(150,len(s))).mean(); ma200=s.tail(min(200,len(s))).mean()
        trend=float(s.iloc[-1]>ma200)+float(ma50>ma150)+float(ma150>ma200)
        high=s.tail(min(252,len(s))).max(); near_high=s.iloc[-1]/high-1 if high else np.nan
        ext=abs(s.iloc[-1]/ma20-1) if ma20 else np.nan
        vol20=r.tail(20).std(); vol60=r.tail(60).std(); vol_contract=vol20/vol60 if vol60 else np.nan
        range20=s.tail(20).max()/s.tail(20).min()-1 if s.tail(20).min()>0 else np.nan
        range60=s.tail(60).max()/s.tail(60).min()-1 if s.tail(60).min()>0 else np.nan
        range_contract=range20/range60 if range60 else np.nan
        rows.append((t,trend,near_high,ext,vol_contract,range_contract))
    d=pd.DataFrame(rows,columns=["Ticker","trend_alignment","near_high","ma20_extension","vol_contraction","range_contraction"]).set_index("Ticker")
    if d.empty:return pd.DataFrame(columns=["vcp_score"])
    d["vcp_score"]=( _rank01(d.trend_alignment)+_rank01(d.near_high)-_rank01(d.ma20_extension)-_rank01(d.vol_contraction)-_rank01(d.range_contraction)+3)/6
    return d

def combine_minervini_lens_score(base, residual, vcp, variant):
    d=base.copy(); d["base_score"]=_rank01(d.Total_Score) if "Total_Score" in d else pd.Series(dtype=float)
    d=d.join(residual[["residual_score"]],how="left").join(vcp[["vcp_score"]],how="left")
    d[["residual_score","vcp_score"]]=d[["residual_score","vcp_score"]].fillna(0.5)
    d["Final_Score"]=variant["base_weight"]*d.base_score+variant["residual_weight"]*d.residual_score+variant["vcp_weight"]*d.vcp_score
    return d.sort_values("Final_Score",ascending=False)

def _select_minervini(prices, us, jp, as_of_date, variant, top_n=6):
    frames=[]
    for region,tickers in (("US",us),("JP",jp)):
        base=score_universe(prices,[t for t in tickers if t in prices],as_of_date)
        residual=compute_residual_momentum_score(prices,base.index,as_of_date,region)
        vcp=compute_vcp_proxy_score(prices,base.index,as_of_date)
        d=combine_minervini_lens_score(base,residual,vcp,variant).head(top_n).copy(); d["Region"]=region; frames.append(d)
    p=pd.concat(frames) if any(not x.empty for x in frames) else pd.DataFrame()
    if not p.empty:
        inv=1/p.Volatility; p["Weight"]=inv/inv.sum()
    return p

def _run_minervini_variant(prices, us, jp, start, end, variant):
    prices=prices.sort_index(); rets=prices.pct_change(); dates=rebalance_dates(prices.index,start,end)
    out=pd.Series(0.,index=prices.loc[start:end].index); records=[]; scores=[]; turns=[]; prev={}
    for i,t in enumerate(dates):
        future=prices.index[prices.index>t]
        if not len(future): continue
        trade=future[0]; next_t=dates[i+1] if i+1<len(dates) else pd.Timestamp(end); hold=out.index[(out.index>=trade)&(out.index<=next_t)]
        p=_select_minervini(prices,us,jp,t,variant); weights=p.Weight.to_dict() if not p.empty else {}
        out.loc[hold]=rets.reindex(hold)[list(weights)].mul(pd.Series(weights)).sum(axis=1) if weights else 0.
        turn=sum(abs(weights.get(k,0)-prev.get(k,0)) for k in set(weights)|set(prev))/2; turns.append({"variant":variant["name"],"screen_date":t,"trade_date":trade,"turnover":turn}); prev=weights
        for ticker,row in p.iterrows():
            rec={"variant":variant["name"],"screen_date":t,"trade_date":trade,"ticker":ticker,**row.to_dict()}; records.append(rec)
            scores.append({k:rec.get(k) for k in ("variant","screen_date","ticker","Region","base_score","residual_score","vcp_score","Final_Score","Total_Score")})
    return out,pd.DataFrame(records),pd.DataFrame(scores),pd.DataFrame(turns)

def _judge(row, base):
    c=(row.CAGR-base.CAGR)/(abs(base.CAGR) or 1); vol=(row.Annualized_Volatility-base.Annualized_Volatility)/(base.Annualized_Volatility or 1); dd=(abs(row.Max_Drawdown)-abs(base.Max_Drawdown))/(abs(base.Max_Drawdown) or 1)
    if row.name=="Baseline": return "baseline"
    if row.Calmar>base.Calmar and ((c>=0 and min(vol,dd)<c) or (c<0 and (vol<c or dd<c))): return "clear improvement"
    if row.Calmar>base.Calmar or c>0: return "mixed"
    return "worse" if row.Calmar<base.Calmar and c<0 else "no improvement"

def run_minervini_lens_audit(prices, us, jp, start, end, output_dir="artifacts/minervini_lens"):
    import json
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); Path("reports").mkdir(exist_ok=True)
    returns={}; selected=[]; scores=[]; turns=[]
    for v in MINERVINI_VARIANTS:
        r,sel,sc,tu=_run_minervini_variant(prices,us,jp,start,end,v); returns[v["name"]]=r; selected.append(sel); scores.append(sc); turns.append(tu)
    selected=pd.concat(selected,ignore_index=True); score_components=pd.concat(scores,ignore_index=True); turnover=pd.concat(turns,ignore_index=True)
    summary=pd.DataFrame({k:metrics(v) for k,v in returns.items()}).T; summary.index.name="Variant"; summary["Turnover"]=turnover.groupby("variant").turnover.mean(); summary["Number_of_Rebalances"]=turnover.groupby("variant").size(); summary["Judgment"]=[_judge(row,summary.loc["Baseline"]) for _,row in summary.iterrows()]
    annual=pd.DataFrame({k:(1+v).resample("YE").prod()-1 for k,v in returns.items()}); monthly=pd.DataFrame({k:(1+v).resample("ME").prod()-1 for k,v in returns.items()}); draw=pd.DataFrame({k:((1+v).cumprod()/(1+v).cumprod().cummax()-1) for k,v in returns.items()})
    summary.to_csv(out/"variant_summary.csv"); summary.to_json(out/"variant_summary.json",orient="index",indent=2); annual.to_csv(out/"annual_returns.csv"); monthly.to_csv(out/"monthly_returns.csv"); draw.to_csv(out/"drawdown_series.csv"); turnover.to_csv(out/"turnover.csv",index=False); selected.to_csv(out/"selected_tickers.csv",index=False); score_components.to_csv(out/"score_components.csv",index=False)
    meta={"generated_at":pd.Timestamp.now("UTC").isoformat(),"data_start":str(prices.index.min().date()),"data_end":str(prices.index.max().date()),"rebalance_frequency":"quarterly / about every 90 days","ttl_days":90,"universe_size":{"US":len(us),"JP":len(jp)},"variants":list(MINERVINI_VARIANTS),"whether_exit_protocol_used":False,"whether_regime_filter_used":False,"notes_on_missing_data":"Missing lens inputs are neutralized where possible; insufficient base history follows existing score_universe rules."}
    (out/"audit_metadata.json").write_text(json.dumps(meta,indent=2,default=str),encoding="utf-8")
    base=summary.loc["Baseline"]; best=summary.drop(index="Baseline").sort_values("Calmar",ascending=False).iloc[0]
    best_name=summary.drop(index="Baseline").sort_values("Calmar",ascending=False).index[0]
    view=summary[["CAGR","Annualized_Volatility","Max_Drawdown","Sharpe","Sortino","Calmar","Turnover","Judgment"]]
    table="| Variant | " + " | ".join(view.columns) + " |\n| --- | " + " | ".join(["---"]*len(view.columns)) + " |\n" + "\n".join("| " + str(idx) + " | " + " | ".join((f"{x:.4f}" if isinstance(x,(int,float,np.floating)) and pd.notna(x) else str(x)) for x in row) + " |" for idx,row in view.iterrows())
    report=f"""# Minervini Lens Audit Report\n\n## Executive Summary\nBest non-baseline by Calmar was **{best_name}**. Baseline CAGR {base.CAGR:.2%}, Vol {base.Annualized_Volatility:.2%}, MaxDD {base.Max_Drawdown:.2%}, Calmar {base.Calmar:.2f}; {best_name} CAGR {best.CAGR:.2%}, Vol {best.Annualized_Volatility:.2%}, MaxDD {best.Max_Drawdown:.2%}, Calmar {best.Calmar:.2f}. The judgment is **{best.Judgment}**; no overly optimistic production recommendation is made from demo/free data alone.\n\n## Baseline Reminder\nTTL is fixed at 90 days / quarterly rebalancing, trades occur four times per year, no Exit Protocol, no Regime Filter, no stop loss, no discretionary cash retreat. Baseline remains the current Alpha Engine score only.\n\n## CLI\n`python alpha_engine_backtest.py --demo --audit minervini_lens --output-dir artifacts/minervini_lens`\n\n## Variant Summary Table\n{table}\n\n## Improvement Judgment\nClassifications use CAGR, Vol, MaxDD, Calmar, and turnover versus Baseline: clear improvement / mixed / no improvement / worse.\n\n## Residual Momentum Findings\nResidual Momentum uses stock return minus US benchmark (SPY/^GSPC) or JP benchmark (1306.T/^TOPX/^N225), then rank-percentile scoring. Compare Residual_20 and Residual_30 in the table; higher residual weight is not assumed better unless Calmar and downside behavior improve.\n\n## VCP Proxy Findings\nVCP Proxy combines trend alignment, near-52-week-high behavior, MA20 extension penalty, volatility contraction, and range contraction. It is not a low-volatility strategy; it only prioritizes tighter action among already strong candidates. VCP is a scoring lens, not a hard filter.\n\n## Combined Lens Findings\nCombined variants test whether Residual + VCP improves beyond single lenses without changing the 90-day mechanical trade cadence.\n\n## Risk Review\nReview MaxDD, Worst Year, 2022 annual returns when present, and turnover in the CSV artifacts. Sector data is not newly fetched; concentration review is limited to selected tickers.\n\n## Recommendation\nProduction Alpha Engine should not be changed unless a non-baseline variant shows durable Calmar improvement without excessive turnover and without materially worsening adverse years. Candidate from this run: **{best_name}** only if its judgment is clear improvement or acceptable mixed after live-data validation.\n\n## Safety Notes\nThis is research, not investment advice. Past data does not guarantee future returns. yfinance/Wikipedia/free data can have missing values, delays, index membership bias, and survivorship bias; results are framed within free-data constraints for individual investors.\n"""
    Path("reports/minervini_lens_audit_report.md").write_text(report,encoding="utf-8")
    return summary


RESIDUAL_DEEP_WEIGHTS=(0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.40)

def build_residual_weight_variants():
    return tuple({"name":"Baseline" if w==0 else f"Residual_{int(w*100):02d}","base_weight":round(1-w,2),"residual_weight":w,"vcp_weight":0.0} for w in RESIDUAL_DEEP_WEIGHTS)

def build_benchmark_modes():
    return {"broad_default":{"US":("SPY","^GSPC"),"JP":("1306.T","^TOPX")},"growth_adjusted_us":{"US":("QQQ",),"JP":("1306.T","^TOPX")},"index_alt_jp":{"US":("SPY","^GSPC"),"JP":("^N225",)},"strict_available_default":{"US":("SPY","^GSPC"),"JP":("1306.T","^TOPX","^N225")}}

def _resolve_benchmark(prices, region, benchmark_mode=None):
    if isinstance(benchmark_mode, dict):
        for c in benchmark_mode.get(region, ()): 
            if c in prices.columns: return c
    return _pick_benchmark(prices, region)

def compute_beta_adjusted_residual_score(stock, benchmark, window=63):
    joined=pd.concat([stock.pct_change(), benchmark.pct_change()],axis=1).dropna().tail(window)
    if len(joined)<5: return 0.0
    ri=joined.iloc[:,0]; rm=joined.iloc[:,1]; var=rm.var()
    if not np.isfinite(var) or var<=1e-12: return 0.0
    beta=ri.cov(rm)/var
    residual_daily=ri-beta*rm
    return float((1+residual_daily).prod()-1) if len(residual_daily) else 0.0

def combine_residual_score(base, residual, variant):
    d=base.copy(); d["base_score"]=_rank01(d.Total_Score) if "Total_Score" in d else pd.Series(dtype=float)
    d=d.join(residual[["residual_score","residual_raw"]],how="left") if "residual_raw" in residual else d.join(residual[["residual_score"]],how="left")
    d["residual_score"]=d["residual_score"].fillna(0.5); d["residual_raw"]=d.get("residual_raw",pd.Series(index=d.index,dtype=float)).fillna(0.0)
    d["Final_Score"]=variant["base_weight"]*d.base_score+variant["residual_weight"]*d.residual_score
    return d.sort_values("Final_Score",ascending=False)

def _window_return(hist, ticker, n):
    if ticker not in hist or len(hist[ticker].dropna())<=n: return np.nan
    s=hist[ticker].dropna(); return float(s.iloc[-1]/s.iloc[-n]-1)

def _select_residual_deep(prices, us, jp, as_of_date, variant, benchmark_mode=None, method="simple", top_n=6):
    frames=[]
    for region,tickers in (("US",us),("JP",jp)):
        base=score_universe(prices,[t for t in tickers if t in prices],as_of_date)
        residual=compute_residual_momentum_score(prices,base.index,as_of_date,region,benchmark_mode=benchmark_mode,method=method)
        d=combine_residual_score(base,residual,variant).head(top_n).copy(); d["Region"]=region; frames.append(d)
    p=pd.concat(frames) if any(not x.empty for x in frames) else pd.DataFrame()
    if not p.empty:
        inv=1/p.Volatility; p["Weight"]=inv/inv.sum()
    return p

def _run_residual_deep_variant(prices, us, jp, start, end, variant, benchmark_mode=None, method="simple"):
    prices=prices.sort_index(); rets=prices.pct_change(); dates=rebalance_dates(prices.index,start,end)
    out=pd.Series(0.,index=prices.loc[start:end].index); records=[]; scores=[]; turns=[]; prev={}
    for i,t in enumerate(dates):
        future=prices.index[prices.index>t]
        if not len(future): continue
        trade=future[0]; next_t=dates[i+1] if i+1<len(dates) else pd.Timestamp(end); hold=out.index[(out.index>=trade)&(out.index<=next_t)]
        p=_select_residual_deep(prices,us,jp,t,variant,benchmark_mode,method); weights=p.Weight.to_dict() if not p.empty else {}
        out.loc[hold]=rets.reindex(hold)[list(weights)].mul(pd.Series(weights)).sum(axis=1) if weights else 0.
        turns.append({"variant":variant["name"],"screen_date":t,"trade_date":trade,"turnover":sum(abs(weights.get(k,0)-prev.get(k,0)) for k in set(weights)|set(prev))/2}); prev=weights
        hist=asof_prices(prices,t).ffill()
        for ticker,row in p.iterrows():
            region=row.get("Region"); bench=_resolve_benchmark(hist,region,benchmark_mode); rec={"variant":variant["name"],"screen_date":t,"trade_date":trade,"ticker":ticker,**row.to_dict()} ; records.append(rec)
            scores.append({"date":t,"ticker":ticker,"market":region,"variant":variant["name"],"base_score":row.get("base_score"),"residual_score":row.get("residual_score"),"final_score":row.get("Final_Score"),"residual_weight":variant["residual_weight"],"base_weight":variant["base_weight"],"stock_return_3m":_window_return(hist,ticker,63),"stock_return_6m":_window_return(hist,ticker,126),"stock_return_12m":_window_return(hist,ticker,252),"benchmark_return_3m":_window_return(hist,bench,63),"benchmark_return_6m":_window_return(hist,bench,126),"benchmark_return_12m":_window_return(hist,bench,252),"residual_return_3m":row.get("residual_raw"),"residual_return_6m":np.nan,"residual_return_12m":np.nan,"benchmark_used":bench,"residual_method":method,"selected_flag":True,"weight":row.get("Weight")})
    return out,pd.DataFrame(records),pd.DataFrame(scores),pd.DataFrame(turns)

def compare_selection_diff(selected):
    rows=[]
    for dt,g in selected.groupby("screen_date"):
        base=set(g[g.variant=="Baseline"].ticker)
        for v,gv in g.groupby("variant"):
            if v=="Baseline": continue
            s=set(gv.ticker); rows.append({"rebalance_date":dt,"variant":v,"selected_tickers":";".join(sorted(s)),"added_tickers":";".join(sorted(s-base)),"removed_tickers":";".join(sorted(base-s)),"added_count":len(s-base),"removed_count":len(base-s)})
    return pd.DataFrame(rows)

def _summary_from_returns(returns, turnover):
    summary=pd.DataFrame({k:metrics(v) for k,v in returns.items()}).T; summary.index.name="Variant"; summary["Turnover"]=turnover.groupby("variant").turnover.mean(); summary["Number_of_Rebalances"]=turnover.groupby("variant").size(); summary["Judgment"]=[_judge(row,summary.loc["Baseline"]) for _,row in summary.iterrows()]; return summary

def run_residual_momentum_deep_audit(prices, us, jp, start, end, output_dir="artifacts/residual_momentum_deep"):
    import json
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); Path("reports").mkdir(exist_ok=True)
    variants=build_residual_weight_variants(); modes=build_benchmark_modes(); returns={}; selected=[]; scores=[]; turns=[]
    for v in variants:
        r,sel,sc,tu=_run_residual_deep_variant(prices,us,jp,start,end,v,modes["broad_default"],"simple"); returns[v["name"]]=r; selected.append(sel); scores.append(sc); turns.append(tu)
    selected=pd.concat(selected,ignore_index=True); score_components=pd.concat(scores,ignore_index=True); turnover=pd.concat(turns,ignore_index=True); summary=_summary_from_returns(returns,turnover)
    annual=pd.DataFrame({k:(1+v).resample("YE").prod()-1 for k,v in returns.items()}); monthly=pd.DataFrame({k:(1+v).resample("ME").prod()-1 for k,v in returns.items()}); draw=pd.DataFrame({k:((1+v).cumprod()/(1+v).cumprod().cummax()-1) for k,v in returns.items()}); diff=compare_selection_diff(selected)
    method_rows=[]
    for method in ("simple","beta_adjusted"):
      for w in (0.10,0.15,0.20,0.25,0.30):
        v={"name":f"Residual_{int(w*100):02d}","base_weight":1-w,"residual_weight":w,"vcp_weight":0.0}; r,_,_,tu=_run_residual_deep_variant(prices,us,jp,start,end,v,modes["broad_default"],method); m=metrics(r); method_rows.append({"method":method,"variant":v["name"],**m,"Turnover":tu.turnover.mean()})
    method_cmp=pd.DataFrame(method_rows)
    bench_rows=[]
    for mode_name,mode in modes.items():
      for w in (0.15,0.20,0.25):
        v={"name":f"Residual_{int(w*100):02d}","base_weight":1-w,"residual_weight":w,"vcp_weight":0.0}; r,_,_,tu=_run_residual_deep_variant(prices,us,jp,start,end,v,mode,"simple"); bench_rows.append({"benchmark_mode":mode_name,"variant":v["name"],**metrics(r),"Turnover":tu.turnover.mean()})
    bench_cmp=pd.DataFrame(bench_rows)
    summary.to_csv(out/"variant_summary.csv"); summary.to_json(out/"variant_summary.json",orient="index",indent=2); annual.to_csv(out/"annual_returns.csv"); monthly.to_csv(out/"monthly_returns.csv"); draw.to_csv(out/"drawdown_series.csv"); turnover.to_csv(out/"turnover.csv",index=False); selected.to_csv(out/"selected_tickers.csv",index=False); diff.to_csv(out/"selection_diff.csv",index=False); score_components.to_csv(out/"score_components.csv",index=False); bench_cmp.to_csv(out/"benchmark_sensitivity.csv",index=False); method_cmp.to_csv(out/"residual_method_comparison.csv",index=False)
    meta={"generated_at":pd.Timestamp.now("UTC").isoformat(),"data_start":str(prices.index.min().date()),"data_end":str(prices.index.max().date()),"rebalance_frequency":"quarterly / about every 90 days","ttl_days":90,"universe_size":{"US":len(us),"JP":len(jp)},"variants":list(variants),"residual_weights_tested":list(RESIDUAL_DEEP_WEIGHTS),"residual_methods_tested":["simple","beta_adjusted"],"benchmark_modes_tested":list(modes),"whether_exit_protocol_used":False,"whether_regime_filter_used":False,"whether_vcp_used":False,"notes_on_missing_data":"Missing benchmarks/residual inputs fall back to neutral scores; demo mode is deterministic and network-free."}; (out/"audit_metadata.json").write_text(json.dumps(meta,indent=2,default=str),encoding="utf-8")
    base=summary.loc["Baseline"]; best_name=summary.drop(index="Baseline").sort_values("Calmar",ascending=False).index[0]; best=summary.loc[best_name]; zone=summary.loc[["Residual_10","Residual_15","Residual_20","Residual_25"]]
    view=summary[["CAGR","Annualized_Volatility","Max_Drawdown","Sharpe","Sortino","Calmar","Turnover","Judgment"]]
    table="| Variant | " + " | ".join(view.columns) + " |\n| --- | " + " | ".join(["---"]*len(view.columns)) + " |\n" + "\n".join("| " + str(idx) + " | " + " | ".join((f"{x:.4f}" if isinstance(x,(int,float,np.floating)) and pd.notna(x) else str(x)) for x in row) + " |" for idx,row in view.iterrows())
    report=f"""# Residual Momentum Deep Audit Report

## Executive Summary
Best non-baseline by Calmar was **{best_name}**. Baseline CAGR {base.CAGR:.2%}, MaxDD {base.Max_Drawdown:.2%}, Calmar {base.Calmar:.2f}; {best_name} CAGR {best.CAGR:.2%}, MaxDD {best.Max_Drawdown:.2%}, Calmar {best.Calmar:.2f}. Residual_20 is evaluated as one point inside the Residual_10-25 zone, not as a production rule. Production use still requires live/free-data validation.

## Baseline Reminder
TTL 90 days, quarterly / four trades per year, no Exit Protocol, no Regime Filter, no VCP, no stop loss, no discretionary cash retreat.

## CLI
`python alpha_engine_backtest.py --demo --audit residual_momentum_deep --output-dir artifacts/residual_momentum_deep`

## Residual Weight Sweep Summary
{table}

Improvement zone Residual_10-25 average Calmar {zone.Calmar.mean():.2f} versus Baseline {base.Calmar:.2f}; evaluate stability across the CSVs rather than selecting a single lucky point.

## Residual Method Comparison
Both simple benchmark-adjusted residual and beta-adjusted residual are implemented. See `artifacts/residual_momentum_deep/residual_method_comparison.csv`; beta-adjusted falls back safely when benchmark variance/history is insufficient.

## Benchmark Sensitivity
Modes tested: broad_default, growth_adjusted_us, index_alt_jp, strict_available_default. See `benchmark_sensitivity.csv`; this checks whether Residual_15/20/25 depend excessively on SPY/QQQ or TOPIX/Nikkei choices.

## Year / Period Review
Annual returns include 2020, 2022, 2023, 2024 and 2025 where present in `annual_returns.csv`. This is post-analysis only; no regime trading filter or cash retreat is introduced.

## Selection Difference Review
`selection_diff.csv` compares Baseline against Residual_10/15/20/25/30 and other residual weights by rebalance date. `score_components.csv` lists base_score, residual_score, final_score, stock/benchmark returns, benchmark_used, method, selected_flag, and weights.

## Risk Review
Use MaxDD, Worst Year, monthly returns, drawdown_series, turnover, and 2022 annual returns. Calmar is emphasized because the research question is return efficiency versus maximum loss.

## Recommendation
Continue researching Residual Momentum only if the Residual_10-25 zone improves Calmar/MaxDD without unacceptable CAGR or turnover cost. Do not change production Alpha Engine from this demo/free-data audit alone; live/free-data validation is required.

## Safety Notes
This is not investment advice. Historical yfinance/Wikipedia/free-data tests do not guarantee future returns. Free data can contain missing values, delays, adjusted-price issues, index membership bias, and survivorship bias; this is framed as an individual-investor free-data audit.
"""
    Path("reports/residual_momentum_deep_audit_report.md").write_text(report,encoding="utf-8")
    return summary


RESIDUAL_LIVE_WEIGHTS=(0.0,0.10,0.15,0.20,0.25,0.30)


def build_residual_live_variants():
    """Residual Live Validation variants; Baseline is base-only and unchanged."""
    return tuple({"name":"Baseline" if w==0 else f"Residual_{int(w*100):02d}","base_weight":round(1-w,2),"residual_weight":round(w,2),"vcp_weight":0.0} for w in RESIDUAL_LIVE_WEIGHTS)


def normalize_yfinance_ticker(ticker):
    """Normalize free-data universe tickers for yfinance without changing JP .T symbols."""
    t=str(ticker).strip()
    return t if t.endswith(".T") or t.startswith("^") else t.replace(".","-")


def download_live_universe_prices(tickers,start="2015-01-01",end=None,batch_size=80,downloader=None):
    """Download adjusted close data for live validation and return prices plus failure diagnostics."""
    end=end or pd.Timestamp.today().date().isoformat(); import importlib
    yf=None if downloader else importlib.import_module("yfinance")
    downloader=downloader or yf.download
    requested=list(dict.fromkeys(normalize_yfinance_ticker(t) for t in tickers)); frames=[]; failures=[]
    fetch_start=(pd.Timestamp(start)-pd.offsets.BDay(400)).date().isoformat(); fetch_end=(pd.Timestamp(end)+pd.Timedelta(days=1)).date().isoformat()
    for i in range(0,len(requested),batch_size):
        batch=requested[i:i+batch_size]
        try:
            raw=downloader(batch,start=fetch_start,end=fetch_end,auto_adjust=True,progress=False,group_by="column",threads=True)
            frame=_close_frame(raw,batch)
        except Exception as exc:
            failures.extend({"ticker":t,"reason":f"download_exception: {exc}"} for t in batch)
            continue
        got=set(frame.columns)
        for t in sorted(set(batch)-got): failures.append({"ticker":t,"reason":"missing_or_empty_download"})
        if not frame.empty: frames.append(frame)
    prices=pd.concat(frames,axis=1).loc[:,lambda x:~x.columns.duplicated()].sort_index() if frames else pd.DataFrame()
    return prices, pd.DataFrame(failures,columns=["ticker","reason"]), requested


def build_live_data_quality_report(prices,requested,us,jp,failures,start,end,min_history=252,benchmark_status=None):
    usable=[]; insufficient=[]
    for t in requested:
        if t in prices.columns and prices[t].dropna().shape[0]>=min_history: usable.append(t)
        elif t in prices.columns: insufficient.append(t)
    excluded=sorted(set(requested)-set(usable))
    rows=[{"metric":"requested_tickers","value":len(requested)},{"metric":"successfully_downloaded_tickers","value":len([t for t in requested if t in prices.columns])},{"metric":"failed_tickers","value":0 if failures is None or failures.empty else failures.ticker.nunique()},{"metric":"insufficient_history_tickers","value":len(insufficient)},{"metric":"excluded_tickers","value":len(excluded)},{"metric":"final_usable_universe_size","value":len(usable)},{"metric":"us_usable_count","value":len([t for t in us if normalize_yfinance_ticker(t) in usable])},{"metric":"jp_usable_count","value":len([t for t in jp if normalize_yfinance_ticker(t) in usable])},{"metric":"data_start","value":str(prices.index.min().date()) if len(prices.index) else ""},{"metric":"data_end","value":str(prices.index.max().date()) if len(prices.index) else ""},{"metric":"usable_start","value":str(pd.Timestamp(start).date())},{"metric":"usable_end","value":str(pd.Timestamp(end).date())}]
    for k,v in (benchmark_status or {}).items(): rows.append({"metric":f"benchmark_{k}","value":v})
    return pd.DataFrame(rows), insufficient, excluded, usable


def run_residual_live_variant(prices,us,jp,start,end,variant,benchmark_mode=None,method="simple"):
    return _run_residual_deep_variant(prices,us,jp,start,end,variant,benchmark_mode,method)


def _selection_diff_detailed(selected,score_components):
    basic=compare_selection_diff(selected); rows=[]
    for _,r in basic.iterrows():
        dt=pd.Timestamp(r.rebalance_date); v=r.variant
        for side,col in (("added",r.added_tickers),("removed",r.removed_tickers)):
            for ticker in [x for x in str(col).split(";") if x]:
                sc=score_components[(pd.to_datetime(score_components.date)==dt)&(score_components.variant.isin([v,"Baseline"]))&(score_components.ticker==ticker)]
                src=sc[sc.variant==v] if side=="added" else sc[sc.variant=="Baseline"]
                row={"rebalance_date":dt,"variant":v,"change_type":side,"ticker":ticker,"selected_tickers":r.selected_tickers,"added_tickers":r.added_tickers,"removed_tickers":r.removed_tickers}
                if not src.empty: row.update(src.iloc[0].to_dict())
                rows.append(row)
    return pd.DataFrame(rows) if rows else basic


def compare_live_selection_diff(selected,score_components):
    return _selection_diff_detailed(selected,score_components)


def _markdown_table(df):
    if df.empty: return "(no rows)"
    return "| " + " | ".join(map(str,df.columns)) + " |\n| " + " | ".join(["---"]*len(df.columns)) + " |\n" + "\n".join("| " + " | ".join(f"{x:.4f}" if isinstance(x,(float,np.floating)) and pd.notna(x) else str(x) for x in row) + " |" for row in df.itertuples(index=True,name=None))


def write_residual_live_report(summary,bench_cmp,data_quality,metadata,output_dir="artifacts/residual_live_validation"):
    Path("reports").mkdir(exist_ok=True); base=summary.loc["Baseline"]; best_name=summary.drop(index="Baseline").sort_values("Calmar",ascending=False).index[0]; best=summary.loc[best_name]
    candidates=[v for v in ("Residual_15","Residual_20","Residual_25","Residual_30") if v in summary.index]
    zone=summary.loc[candidates]; improved=zone[zone.Calmar>base.Calmar].index.tolist()
    view=summary[["CAGR","Annualized_Volatility","Max_Drawdown","Sharpe","Sortino","Calmar","Turnover","Judgment"]]
    bview=bench_cmp.head(24) if isinstance(bench_cmp,pd.DataFrame) else pd.DataFrame()
    report=f"""# Residual Momentum Live Validation Report

## 1. Executive Summary
Best non-baseline by Calmar was **{best_name}**. Baseline CAGR {base.CAGR:.2%}, MaxDD {base.Max_Drawdown:.2%}, Calmar {base.Calmar:.2f}; {best_name} CAGR {best.CAGR:.2%}, MaxDD {best.Max_Drawdown:.2%}, Calmar {best.Calmar:.2f}. Residual_20 / Residual_25 are candidates only if their rows improve Calmar/MaxDD without unacceptable CAGR or turnover cost. Improved Calmar variants in the main 15-30% zone: {', '.join(improved) if improved else 'none'}. This is not a production adoption decision.

## 2. Baseline Reminder
TTL 90 days, quarterly / four trades per year, no Exit Protocol, no Regime Filter, no VCP, no stop loss, no discretionary cash retreat. Baseline remains current Alpha Engine base score only.

## CLI / Colab Command
`python alpha_engine_backtest.py --audit residual_live_validation --output-dir artifacts/residual_live_validation`

Colab dependencies: `python -m pip install -r requirements.txt` (includes pandas/numpy/yfinance if declared; otherwise install `yfinance`).

## 3. Data Quality Summary
{_markdown_table(data_quality.set_index('metric'))}

Missing downloads and insufficient history can reduce breadth; current constituents used historically can introduce survivorship / historical constituent bias.

## 4. Variant Summary
{_markdown_table(view)}

## 5. Benchmark Sensitivity
{_markdown_table(bview.set_index(['benchmark_mode','variant']) if not bview.empty else bview)}

## 6. Year / Period Review
Review `annual_returns.csv`, `monthly_returns.csv`, and `drawdown_series.csv` for 2020, 2022, 2023, 2024, 2025, and 2026 YTD when present. This is post-analysis only and never triggers cash retreat or trade suspension.

## 7. Selection Difference Review
`selection_diff.csv` compares Baseline with Residual_10/15/20/25/30 by rebalance date and records added/removed tickers plus score context where available. Use it to assess whether residual removes market beta followers and whether US/JP concentration changes.

## 8. Risk Review
Calmar and MaxDD are emphasized alongside CAGR, Sharpe, Sortino, worst year/month, drawdown series, and turnover. 2022 behavior and opportunity cost in 2023-2024 should be inspected before any vNext decision.

## 9. Recommendation
Continue Alpha Engine vNext research only if Residual_15-25 shows stable Calmar/MaxDD improvement across benchmark modes. Do not merge into production from this validation alone; keep Baseline untouched until broader live/free-data and bias checks are complete.

## 10. Safety Notes
This validation is research, not investment advice. Past data does not guarantee future returns. yfinance / Wikipedia / free data may have missing values, delays, adjusted-price issues, current-constituent bias, and survivorship bias. The audit is positioned as a realistic individual-investor free-data validation.
"""
    Path("reports/residual_live_validation_report.md").write_text(report,encoding="utf-8")


def run_residual_live_validation(prices=None,us=None,jp=None,start="2015-01-01",end=None,output_dir="artifacts/residual_live_validation",downloader=None):
    """Run live/free-data simple Residual Momentum validation without Exit/Regime/VCP."""
    import json
    end=end or pd.Timestamp.today().date().isoformat(); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); Path("reports").mkdir(exist_ok=True)
    if us is None or jp is None: us,jp=get_live_universe()
    us=[normalize_yfinance_ticker(t) for t in us]; jp=[normalize_yfinance_ticker(t) for t in jp]
    modes=build_benchmark_modes(); bench_tickers=sorted({x for mode in modes.values() for vals in mode.values() for x in vals})
    requested=list(dict.fromkeys([*us,*jp,*bench_tickers,"^GSPC","^N225","^TOPX","SPY","QQQ","1306.T"]))
    failures=pd.DataFrame(columns=["ticker","reason"])
    if prices is None:
        prices,failures,requested=download_live_universe_prices(requested,start,end,downloader=downloader)
    benchmark_status={name:{region:_resolve_benchmark(prices,region,mode) for region in ("US","JP")} for name,mode in modes.items()}
    dq,insufficient,excluded,usable=build_live_data_quality_report(prices,requested,us,jp,failures,start,end,benchmark_status={k:str(v) for k,v in benchmark_status.items()})
    us_usable=[t for t in us if t in usable]; jp_usable=[t for t in jp if t in usable]
    variants=build_residual_live_variants(); returns={}; selected=[]; scores=[]; turns=[]
    if prices.empty or not (us_usable or jp_usable):
        metric_cols=["CAGR","Annualized_Volatility","Max_Drawdown","Sharpe","Calmar","Sortino","Total_Return","Best_Year","Worst_Year","Monthly_Win_Rate","Turnover","Number_of_Rebalances","Judgment"]
        summary=pd.DataFrame(index=[v["name"] for v in variants],columns=metric_cols); summary.index.name="Variant"; summary["Judgment"]=["baseline" if i==0 else "skipped_no_usable_data" for i in range(len(summary))]
        empty=pd.DataFrame(); bench_cmp=pd.DataFrame([{"benchmark_mode":m,"status":"skipped_missing_benchmark","variant":v["name"]} for m in modes for v in variants if v["name"]!="Baseline"])
        for name,df in (("variant_summary.csv",summary),("annual_returns.csv",empty),("monthly_returns.csv",empty),("drawdown_series.csv",empty),("turnover.csv",empty),("selected_tickers.csv",empty),("selection_diff.csv",empty),("score_components.csv",empty),("benchmark_sensitivity.csv",bench_cmp),("data_quality.csv",dq),("download_failures.csv",failures)):
            df.to_csv(out/name,index=(name=="variant_summary.csv"))
        summary.to_json(out/"variant_summary.json",orient="index",indent=2)
        meta={"generated_at":pd.Timestamp.now("UTC").isoformat(),"data_start":"","data_end":"","usable_start":str(pd.Timestamp(start).date()),"usable_end":str(pd.Timestamp(end).date()),"rebalance_frequency":"quarterly / about every 90 days","ttl_days":90,"universe_size_requested":len(requested),"universe_size_downloaded":0,"universe_size_usable":0,"us_usable_count":0,"jp_usable_count":0,"variants":list(variants),"residual_weights_tested":list(RESIDUAL_LIVE_WEIGHTS),"benchmark_modes_tested":list(modes),"residual_method":"simple","whether_exit_protocol_used":False,"whether_regime_filter_used":False,"whether_vcp_used":False,"data_source":"yfinance / Wikipedia / existing free universe","notes_on_missing_data":"No usable downloaded prices; failures logged and audit skipped without crashing.","notes_on_survivorship_bias":"Current free-data universes may be applied backward; historical constituent bias and survivorship bias remain."}
        (out/"audit_metadata.json").write_text(json.dumps(meta,indent=2,default=str),encoding="utf-8")
        write_residual_live_report(summary,bench_cmp,dq,meta,out)
        return summary
    default_mode=modes["broad_default"]
    for v in variants:
        r,sel,sc,tu=run_residual_live_variant(prices,us_usable,jp_usable,start,end,v,default_mode,"simple"); returns[v["name"]]=r; selected.append(sel); scores.append(sc); turns.append(tu)
    selected=pd.concat(selected,ignore_index=True) if selected else pd.DataFrame(); score_components=pd.concat(scores,ignore_index=True) if scores else pd.DataFrame(); turnover=pd.concat(turns,ignore_index=True) if turns else pd.DataFrame()
    summary=_summary_from_returns(returns,turnover); annual=pd.DataFrame({k:(1+v).resample("YE").prod()-1 for k,v in returns.items()}); monthly=pd.DataFrame({k:(1+v).resample("ME").prod()-1 for k,v in returns.items()}); draw=pd.DataFrame({k:((1+v).cumprod()/(1+v).cumprod().cummax()-1) for k,v in returns.items()}); diff=compare_live_selection_diff(selected,score_components)
    bench_rows=[]
    for mode_name,mode in modes.items():
        ok=bool(_resolve_benchmark(prices,"US",mode) and _resolve_benchmark(prices,"JP",mode))
        for w in (0.15,0.20,0.25,0.30):
            v={"name":f"Residual_{int(w*100):02d}","base_weight":round(1-w,2),"residual_weight":w,"vcp_weight":0.0}
            if ok:
                r,_,_,tu=run_residual_live_variant(prices,us_usable,jp_usable,start,end,v,mode,"simple"); bench_rows.append({"benchmark_mode":mode_name,"status":"ok","variant":v["name"],**metrics(r),"Turnover":tu.turnover.mean() if not tu.empty else np.nan})
            else: bench_rows.append({"benchmark_mode":mode_name,"status":"skipped_missing_benchmark","variant":v["name"]})
    bench_cmp=pd.DataFrame(bench_rows)
    for name,df in (("variant_summary.csv",summary),("annual_returns.csv",annual),("monthly_returns.csv",monthly),("drawdown_series.csv",draw),("turnover.csv",turnover),("selected_tickers.csv",selected),("selection_diff.csv",diff),("score_components.csv",score_components),("benchmark_sensitivity.csv",bench_cmp),("data_quality.csv",dq),("download_failures.csv",failures)):
        df.to_csv(out/name,index=(name in ("variant_summary.csv","annual_returns.csv","monthly_returns.csv","drawdown_series.csv")))
    summary.to_json(out/"variant_summary.json",orient="index",indent=2)
    meta={"generated_at":pd.Timestamp.now("UTC").isoformat(),"data_start":str(prices.index.min().date()) if len(prices.index) else "","data_end":str(prices.index.max().date()) if len(prices.index) else "","usable_start":str(pd.Timestamp(start).date()),"usable_end":str(pd.Timestamp(end).date()),"rebalance_frequency":"quarterly / about every 90 days","ttl_days":90,"universe_size_requested":len(requested),"universe_size_downloaded":len([t for t in requested if t in prices.columns]),"universe_size_usable":len(usable),"us_usable_count":len(us_usable),"jp_usable_count":len(jp_usable),"variants":list(variants),"residual_weights_tested":list(RESIDUAL_LIVE_WEIGHTS),"benchmark_modes_tested":list(modes),"residual_method":"simple","whether_exit_protocol_used":False,"whether_regime_filter_used":False,"whether_vcp_used":False,"data_source":"yfinance / Wikipedia / existing free universe","notes_on_missing_data":"Failed tickers are logged and excluded; missing residual inputs are neutralized rather than crashing.","notes_on_survivorship_bias":"Current free-data universes may be applied backward; historical constituent bias and survivorship bias remain."}
    (out/"audit_metadata.json").write_text(json.dumps(meta,indent=2,default=str),encoding="utf-8")
    write_residual_live_report(summary,bench_cmp,dq,meta,out)
    return summary


RESIDUAL_FULL_SWEEP_WEIGHTS=tuple(round(x/100,2) for x in range(0,101,5))


def build_residual_full_sweep_variants():
    """Build 0%-100% residual sweep variants; Baseline is base-only and unchanged."""
    return tuple({"name":"Baseline" if w==0 else f"Residual_{int(w*100):02d}","base_weight":round(1-w,2),"residual_weight":round(w,2),"vcp_weight":0.0} for w in RESIDUAL_FULL_SWEEP_WEIGHTS)


def _cache_metadata(start,end,requested,benchmark_tickers):
    return {"start":str(start),"end":str(end),"requested_tickers":list(requested),"benchmark_tickers":list(benchmark_tickers),"created_at":pd.Timestamp.now("UTC").isoformat(),"cache_version":"residual_full_sweep_v1","git_tracking_note":"Binary price cache files are generated at runtime and intentionally ignored by git."}


def validate_price_cache(cache_dir,start,end,requested,benchmark_tickers):
    cache_dir=Path(cache_dir); meta_path=cache_dir/"cache_metadata.json"; prices_path=cache_dir/"prices.pkl"; bench_path=cache_dir/"benchmarks.pkl"
    if not (meta_path.exists() and prices_path.exists() and bench_path.exists()): return False,"missing_cache_files"
    try:
        import json
        meta=json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("start")!=str(start) or meta.get("end")!=str(end): return False,"cache_period_mismatch"
        if set(meta.get("requested_tickers",[]))!=set(requested): return False,"cache_universe_mismatch"
        if set(meta.get("benchmark_tickers",[]))!=set(benchmark_tickers): return False,"cache_benchmark_mismatch"
        pd.read_pickle(prices_path); pd.read_pickle(bench_path)
        return True,"ok"
    except Exception as exc:
        return False,f"cache_corrupt: {exc}"


def load_price_cache(cache_dir):
    cache_dir=Path(cache_dir)
    return pd.read_pickle(cache_dir/"prices.pkl"), pd.read_pickle(cache_dir/"benchmarks.pkl")


def build_price_cache(tickers,benchmark_tickers,start,end,cache_dir,downloader=None,batch_size=80):
    """Download once, write pickle/csv cache, and return prices/failures/requested/cache metadata."""
    cache_dir=Path(cache_dir); cache_dir.mkdir(parents=True,exist_ok=True)
    requested=list(dict.fromkeys(normalize_yfinance_ticker(t) for t in tickers)); benchmarks=list(dict.fromkeys(normalize_yfinance_ticker(t) for t in benchmark_tickers))
    ok,reason=validate_price_cache(cache_dir,start,end,requested,benchmarks)
    if ok:
        prices,bench=load_price_cache(cache_dir)
        failures=pd.DataFrame(columns=["ticker","reason"])
        meta={**_cache_metadata(start,end,requested,benchmarks),"cache_used":True,"cache_path":str(cache_dir),"cache_status":"hit"}
        return pd.concat([prices,bench],axis=1).loc[:,lambda x:~x.columns.duplicated()].sort_index(), failures, requested, meta
    print(f"[residual_full_sweep] cache miss ({reason}); downloading {len(requested)} tickers and {len(benchmarks)} benchmarks")
    prices,failures,requested2=download_live_universe_prices(list(dict.fromkeys([*requested,*benchmarks])),start,end,batch_size=batch_size,downloader=downloader)
    bench=prices[[c for c in benchmarks if c in prices.columns]].copy() if not prices.empty else pd.DataFrame()
    nonbench=prices[[c for c in prices.columns if c not in benchmarks]].copy() if not prices.empty else pd.DataFrame()
    nonbench.to_pickle(cache_dir/"prices.pkl"); bench.to_pickle(cache_dir/"benchmarks.pkl")
    pd.DataFrame({"ticker":requested}).to_csv(cache_dir/"universe.csv",index=False)
    import json
    meta={**_cache_metadata(start,end,requested,benchmarks),"cache_used":False,"cache_path":str(cache_dir),"cache_status":reason}
    (cache_dir/"cache_metadata.json").write_text(json.dumps(meta,indent=2,default=str),encoding="utf-8")
    return prices,failures,requested2,meta


def run_residual_full_sweep_variant(prices,us,jp,start,end,variant,benchmark_mode=None,method="simple"):
    return _run_residual_deep_variant(prices,us,jp,start,end,variant,benchmark_mode,method)


def compute_peak_ratio_diagnostics(summary):
    rows=[]
    def ratio(name): return 0 if name=="Baseline" else int(str(name).split("_")[-1])
    specs=[("best_cagr_ratio","CAGR",False),("best_sharpe_ratio","Sharpe",False),("best_sortino_ratio","Sortino",False),("best_calmar_ratio","Calmar",False),("lowest_maxdd_ratio","Max_Drawdown",True),("lowest_volatility_ratio","Annualized_Volatility",True)]
    for metric_name,col,lowest in specs:
        s=summary[col].dropna()
        if s.empty: rows.append({"diagnostic":metric_name,"variant":"","ratio":np.nan,"value":np.nan}); continue
        idx=s.idxmax() if not lowest else (s.abs().idxmin() if col=="Max_Drawdown" else s.idxmin())
        rows.append({"diagnostic":metric_name,"variant":idx,"ratio":ratio(idx),"value":summary.loc[idx,col]})
    base=summary.loc["Baseline"] if "Baseline" in summary.index else None
    imp=summary[(summary.CAGR>base.CAGR)&(summary.Calmar>base.Calmar)] if base is not None else pd.DataFrame()
    if not imp.empty:
        idx=imp.Turnover.idxmin(); rows.append({"diagnostic":"lowest_turnover_ratio_among_improving_variants","variant":idx,"ratio":ratio(idx),"value":summary.loc[idx,"Turnover"]})
    return pd.DataFrame(rows)


def _ranges_from_ratios(ratios):
    ratios=sorted(int(x) for x in ratios)
    if not ratios: return "none"
    bands=[]; start=prev=ratios[0]
    for r in ratios[1:]:
        if r==prev+5: prev=r
        else: bands.append(f"{start}-{prev}%" if start!=prev else f"{start}%"); start=prev=r
    bands.append(f"{start}-{prev}%" if start!=prev else f"{start}%")
    return "; ".join(bands)


def compute_plateau_analysis(summary):
    if "Baseline" not in summary.index: return pd.DataFrame()
    base=summary.loc["Baseline"]
    work=summary.copy(); work["ratio"]= [0 if i=="Baseline" else int(str(i).split("_")[-1]) for i in work.index]
    checks={"cagr_improved_range":work.CAGR>base.CAGR,"maxdd_improved_range":work.Max_Drawdown.abs()<abs(base.Max_Drawdown),"calmar_improved_range":work.Calmar>base.Calmar,"sharpe_improved_range":work.Sharpe>base.Sharpe,"sortino_improved_range":work.Sortino>base.Sortino}
    rows=[]
    for name,mask in checks.items(): rows.append({"analysis":name,"ratio_range":_ranges_from_ratios(work.loc[mask,"ratio"]),"count":int(mask.sum())})
    combo=(work.CAGR>base.CAGR)&(work.Max_Drawdown.abs()<abs(base.Max_Drawdown))&(work.Calmar>base.Calmar)
    rows.append({"analysis":"cagr_maxdd_calmar_simultaneous_improvement_range","ratio_range":_ranges_from_ratios(work.loc[combo,"ratio"]),"count":int(combo.sum())})
    best=work.sort_values("Calmar",ascending=False).head(1)
    br=int(best.ratio.iloc[0]) if not best.empty else 0; neigh=work[work.ratio.between(max(0,br-10),min(100,br+10))]
    stable=neigh[(neigh.Calmar>base.Calmar)&(neigh.CAGR>=base.CAGR*0.98)]
    rows.append({"analysis":"best_plateau_near_best_calmar","ratio_range":_ranges_from_ratios(stable.ratio),"count":len(stable)})
    return pd.DataFrame(rows)


def classify_residual_concept(best_ratio):
    if pd.isna(best_ratio): return "No usable classification"
    r=int(best_ratio)
    if r==100: return "Pure Residual"
    if 75<=r<=95: return "Residual Dominant"
    if 30<=r<=70: return "Hybrid Core"
    if 5<=r<=25: return "Auxiliary Lens"
    return "Baseline / No residual improvement"


def compare_full_sweep_selection_diff(selected,score_components,best_variant=None):
    detailed=_selection_diff_detailed(selected,score_components)
    keep={"Residual_20","Residual_25","Residual_30","Residual_50","Residual_75","Residual_100"}
    if best_variant and best_variant!="Baseline": keep.add(best_variant)
    return detailed[detailed["variant"].isin(keep)].copy() if "variant" in detailed else detailed


def _empty_full_sweep_outputs(out,variants,dq,failures,meta,modes):
    metric_cols=["CAGR","Annualized_Volatility","Max_Drawdown","Sharpe","Calmar","Sortino","Total_Return","Best_Year","Worst_Year","Monthly_Win_Rate","Turnover","Number_of_Rebalances","Judgment"]
    summary=pd.DataFrame(index=[v["name"] for v in variants],columns=metric_cols); summary.index.name="Variant"; summary["Judgment"]=["baseline" if i==0 else "skipped_no_usable_data" for i in range(len(summary))]
    empty=pd.DataFrame(); peak=compute_peak_ratio_diagnostics(summary); plateau=compute_plateau_analysis(summary)
    bench=pd.DataFrame([{"benchmark_mode":m,"status":"skipped_no_usable_data"} for m in modes])
    for name,df in (("variant_summary.csv",summary),("annual_returns.csv",empty),("monthly_returns.csv",empty),("drawdown_series.csv",empty),("turnover.csv",empty),("selected_tickers.csv",empty),("selection_diff.csv",empty),("score_components.csv",empty),("benchmark_sensitivity.csv",bench),("data_quality.csv",dq),("download_failures.csv",failures),("peak_ratio_diagnostics.csv",peak),("plateau_analysis.csv",plateau)):
        df.to_csv(out/name,index=(name in ("variant_summary.csv","annual_returns.csv","monthly_returns.csv","drawdown_series.csv")))
    summary.to_json(out/"variant_summary.json",orient="index",indent=2); return summary,bench,peak,plateau


def run_residual_full_sweep(prices=None,us=None,jp=None,start="2015-01-01",end=None,output_dir="artifacts/residual_full_sweep",downloader=None):
    """Run 0-100% simple residual momentum full sweep with cache, no Exit/Regime/VCP."""
    import json
    end=end or pd.Timestamp.today().date().isoformat(); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); (out/"cache").mkdir(exist_ok=True); Path("reports").mkdir(exist_ok=True)
    if us is None or jp is None: us,jp=get_live_universe()
    us=[normalize_yfinance_ticker(t) for t in us]; jp=[normalize_yfinance_ticker(t) for t in jp]
    modes=build_benchmark_modes(); bench_tickers=sorted({x for mode in modes.values() for vals in mode.values() for x in vals})
    variants=build_residual_full_sweep_variants(); requested=list(dict.fromkeys([*us,*jp]))
    cache_meta={"cache_used":False,"cache_path":str(out/"cache")}; failures=pd.DataFrame(columns=["ticker","reason"])
    if prices is None:
        prices,failures,download_requested,cache_meta=build_price_cache(requested,bench_tickers,start,end,out/"cache",downloader=downloader)
        requested=download_requested
    else:
        requested=list(dict.fromkeys([*requested,*bench_tickers])); cache_meta["cache_status"]="provided_prices_no_download"
        pd.DataFrame({"ticker":[*us,*jp]}).to_csv(out/"cache"/"universe.csv",index=False)
        prices.to_pickle(out/"cache"/"prices.pkl"); prices[[c for c in bench_tickers if c in prices.columns]].to_pickle(out/"cache"/"benchmarks.pkl")
        (out/"cache"/"cache_metadata.json").write_text(json.dumps({**_cache_metadata(start,end,[*us,*jp],bench_tickers),**cache_meta},indent=2,default=str),encoding="utf-8")
    benchmark_status={name:{region:_resolve_benchmark(prices,region,mode) for region in ("US","JP")} for name,mode in modes.items()}
    dq,insufficient,excluded,usable=build_live_data_quality_report(prices,requested,us,jp,failures,start,end,benchmark_status={k:str(v) for k,v in benchmark_status.items()})
    dq=pd.concat([dq,pd.DataFrame([{"metric":"cache_used","value":cache_meta.get("cache_used",False)},{"metric":"cache_path","value":cache_meta.get("cache_path",str(out/"cache"))},{"metric":"cache_created_at","value":cache_meta.get("created_at","")}])],ignore_index=True)
    us_usable=[t for t in us if t in usable]; jp_usable=[t for t in jp if t in usable]
    base_meta={"generated_at":pd.Timestamp.now("UTC").isoformat(),"data_start":str(prices.index.min().date()) if len(prices.index) else "","data_end":str(prices.index.max().date()) if len(prices.index) else "","usable_start":str(pd.Timestamp(start).date()),"usable_end":str(pd.Timestamp(end).date()),"rebalance_frequency":"quarterly / about every 90 days","ttl_days":90,"universe_size_requested":len(requested),"universe_size_downloaded":len([t for t in requested if t in prices.columns]),"universe_size_usable":len(usable),"us_usable_count":len(us_usable),"jp_usable_count":len(jp_usable),"variants":list(variants),"residual_weights_tested":list(RESIDUAL_FULL_SWEEP_WEIGHTS),"benchmark_modes_tested":list(modes),"residual_method":"simple","cache_used":cache_meta.get("cache_used",False),"cache_path":cache_meta.get("cache_path",str(out/"cache")),"cache_git_tracking_note":"Binary price cache files such as prices.pkl and benchmarks.pkl are generated at runtime and intentionally ignored by git.","whether_exit_protocol_used":False,"whether_regime_filter_used":False,"whether_vcp_used":False,"data_source":"yfinance / Wikipedia / existing free universe" if prices is None else "provided/demo prices","notes_on_missing_data":"Failed/insufficient tickers are logged and excluded; missing residual inputs are neutralized.","notes_on_survivorship_bias":"Current free-data universes may be applied backward; historical constituent bias and survivorship bias remain."}
    if prices.empty or not (us_usable or jp_usable):
        summary,bench_cmp,peak,plateau=_empty_full_sweep_outputs(out,variants,dq,failures,base_meta,modes); base_meta.update({"best_cagr_ratio":None,"best_sharpe_ratio":None,"best_sortino_ratio":None,"best_calmar_ratio":None,"lowest_maxdd_ratio":None,"concept_classification":"No usable classification"}); (out/"audit_metadata.json").write_text(json.dumps(base_meta,indent=2,default=str),encoding="utf-8"); write_residual_full_sweep_report(summary,bench_cmp,dq,base_meta,peak,plateau,out); return summary
    returns={}; selected=[]; scores=[]; turns=[]; default_mode=modes["broad_default"]
    for v in variants:
        r,sel,sc,tu=run_residual_full_sweep_variant(prices,us_usable,jp_usable,start,end,v,default_mode,"simple"); returns[v["name"]]=r; selected.append(sel); scores.append(sc); turns.append(tu)
    selected=pd.concat(selected,ignore_index=True); score_components=pd.concat(scores,ignore_index=True); turnover=pd.concat(turns,ignore_index=True); summary=_summary_from_returns(returns,turnover)
    annual=pd.DataFrame({k:(1+v).resample("YE").prod()-1 for k,v in returns.items()}); monthly=pd.DataFrame({k:(1+v).resample("ME").prod()-1 for k,v in returns.items()}); draw=pd.DataFrame({k:((1+v).cumprod()/(1+v).cumprod().cummax()-1) for k,v in returns.items()})
    peak=compute_peak_ratio_diagnostics(summary); best_calmar=peak.loc[peak.diagnostic=="best_calmar_ratio","variant"].iloc[0]; plateau=compute_plateau_analysis(summary); diff=compare_full_sweep_selection_diff(selected,score_components,best_calmar)
    key={"Baseline","Residual_25","Residual_30","Residual_50","Residual_75","Residual_100",best_calmar}; bench_rows=[]
    for mode_name,mode in modes.items():
        ok=bool(_resolve_benchmark(prices,"US",mode) and _resolve_benchmark(prices,"JP",mode))
        for v in [x for x in variants if x["name"] in key]:
            if ok:
                r,_,_,tu=run_residual_full_sweep_variant(prices,us_usable,jp_usable,start,end,v,mode,"simple"); bench_rows.append({"benchmark_mode":mode_name,"status":"ok","variant":v["name"],**metrics(r),"Turnover":tu.turnover.mean() if not tu.empty else np.nan})
            else: bench_rows.append({"benchmark_mode":mode_name,"status":"skipped_missing_benchmark","variant":v["name"]})
    bench_cmp=pd.DataFrame(bench_rows)
    for name,df in (("variant_summary.csv",summary),("annual_returns.csv",annual),("monthly_returns.csv",monthly),("drawdown_series.csv",draw),("turnover.csv",turnover),("selected_tickers.csv",selected),("selection_diff.csv",diff),("score_components.csv",score_components),("benchmark_sensitivity.csv",bench_cmp),("data_quality.csv",dq),("download_failures.csv",failures),("peak_ratio_diagnostics.csv",peak),("plateau_analysis.csv",plateau)):
        df.to_csv(out/name,index=(name in ("variant_summary.csv","annual_returns.csv","monthly_returns.csv","drawdown_series.csv")))
    summary.to_json(out/"variant_summary.json",orient="index",indent=2)
    pmap=peak.set_index("diagnostic")["ratio"].to_dict(); concept=classify_residual_concept(pmap.get("best_calmar_ratio"))
    base_meta.update({"best_cagr_ratio":pmap.get("best_cagr_ratio"),"best_sharpe_ratio":pmap.get("best_sharpe_ratio"),"best_sortino_ratio":pmap.get("best_sortino_ratio"),"best_calmar_ratio":pmap.get("best_calmar_ratio"),"lowest_maxdd_ratio":pmap.get("lowest_maxdd_ratio"),"concept_classification":concept})
    (out/"audit_metadata.json").write_text(json.dumps(base_meta,indent=2,default=str),encoding="utf-8"); write_residual_full_sweep_report(summary,bench_cmp,dq,base_meta,peak,plateau,out); return summary


def write_residual_full_sweep_report(summary,bench_cmp,data_quality,metadata,peak,plateau,output_dir="artifacts/residual_full_sweep"):
    Path("reports").mkdir(exist_ok=True)
    if "Baseline" in summary.index and summary.CAGR.notna().any():
        base=summary.loc["Baseline"]; best_name=summary.drop(index="Baseline",errors="ignore").sort_values("Calmar",ascending=False).index[0] if len(summary.drop(index="Baseline",errors="ignore")) else "Baseline"; best=summary.loc[best_name]
        headline=f"Best non-baseline by Calmar was **{best_name}**. Baseline CAGR {base.CAGR:.2%}, MaxDD {base.Max_Drawdown:.2%}, Calmar {base.Calmar:.2f}; {best_name} CAGR {best.CAGR:.2%}, MaxDD {best.Max_Drawdown:.2%}, Calmar {best.Calmar:.2f}."
    else:
        best_name="none"; headline="No usable live data was available; outputs are skipped/no-usable-data summaries, not live results."
    view=summary[[c for c in ["CAGR","Annualized_Volatility","Max_Drawdown","Sharpe","Sortino","Calmar","Turnover","Judgment"] if c in summary.columns]]
    report=f"""# Residual Momentum Full Sweep Audit Report

## 1. Executive Summary
{headline} Concept classification: **{metadata.get('concept_classification','pending')}**. This is a research classification only and is not production adoption.

## 2. Baseline Reminder
TTL 90 days, quarterly / four trades per year, no Exit Protocol, no Regime Filter, no VCP, no stop loss, no discretionary cash retreat. Baseline is Residual_00 / base_weight=1.0 / residual_weight=0.0.

## CLI / Colab Commands
`python alpha_engine_backtest.py --audit residual_full_sweep --output-dir artifacts/residual_full_sweep`

`python alpha_engine_backtest.py --demo --audit residual_full_sweep --output-dir artifacts/residual_full_sweep`

## 3. Data Quality Summary
{_markdown_table(data_quality.set_index('metric') if isinstance(data_quality,pd.DataFrame) and not data_quality.empty else pd.DataFrame())}

Cache used: {metadata.get('cache_used')} / cache path: `{metadata.get('cache_path')}`. Price cache files (`prices.pkl`, `benchmarks.pkl`) are generated at runtime and intentionally ignored by git; committed artifacts keep only text metadata such as `cache_metadata.json` and `universe.csv`. Missing downloads, insufficient history, current-constituent use, and survivorship bias can affect results. Demo artifacts are not live results.

## 4. Full Sweep Summary
{_markdown_table(view)}

The sweep covers Residual_00 through Residual_100 in 5% increments. Inspect whether improvement continues above 30%, where metrics peak, and whether pure residual remains viable.

## 5. Peak Ratio Diagnostics
{_markdown_table(peak.set_index('diagnostic') if isinstance(peak,pd.DataFrame) and not peak.empty else pd.DataFrame())}

## 6. Plateau / Band Analysis
{_markdown_table(plateau.set_index('analysis') if isinstance(plateau,pd.DataFrame) and not plateau.empty else pd.DataFrame())}

## 7. Concept Classification
Classification: **{metadata.get('concept_classification','pending')}**. Auxiliary Lens is 5-25%, Hybrid Core is 30-70%, Residual Dominant is 75-95%, Pure Residual is 100%/near-best. This is research framing only.

## 8. Benchmark Sensitivity
{_markdown_table(bench_cmp.head(40).set_index(['benchmark_mode','variant']) if isinstance(bench_cmp,pd.DataFrame) and not bench_cmp.empty and {'benchmark_mode','variant'}.issubset(bench_cmp.columns) else bench_cmp)}

## 9. Year / Period Review
Use `annual_returns.csv`, `monthly_returns.csv`, and `drawdown_series.csv` for 2020, 2022, 2023, 2024, 2025, and 2026 YTD if present. This is post-analysis only and never changes trades.

## 10. Selection Difference Review
`selection_diff.csv` compares Baseline against Residual_25, Residual_30, Residual_50, Residual_75, Residual_100, Residual_20, and the best ratio where available. Review added/removed tickers, base/residual/final scores, US/JP mix, and whether residual is removing beta followers or merely adding volatility.

## 11. Risk Review
Focus on MaxDD, Worst Year, Worst Month/monthly returns, drawdowns, turnover, 2022 behavior, opportunity cost in 2023-2024, and Residual_100 risk. Calmar is emphasized over Sharpe because the question is return versus maximum loss.

## 12. Recommendation
Conservative candidate: first stable plateau ratio that improves Calmar/MaxDD. Balanced candidate: best Calmar ratio if nearby ratios also improve. Aggressive candidate: highest residual ratio that remains in the stable plateau. Do **not** productionize without year-by-year, benchmark, turnover, and selection-difference review.

## 13. Safety Notes
This is not investment advice. Historical tests do not guarantee future returns. yfinance/Wikipedia/free data can have missing values, delays, adjusted-price issues, survivorship bias, and historical constituent bias. The audit remains within individual-investor free-data constraints.
"""
    Path("reports/residual_full_sweep_report.md").write_text(report,encoding="utf-8")




RESIDUAL_CONCENTRATION_WEIGHTS=(0.0,0.50,0.55,0.60,0.65,0.70,1.00)
RESIDUAL_CONCENTRATION_SIZES=((40,20,20),(30,15,15),(24,12,12),(20,10,10),(16,8,8),(12,6,6),(10,5,5),(8,4,4),(6,3,3),(4,2,2))


def build_portfolio_size_configs():
    """Build fixed US/JP-balanced portfolio-size configs for concentration audit."""
    return tuple({"total_holdings":n,"us_holdings":u,"jp_holdings":j,"is_high_diversification_reference":n in (30,40)} for n,u,j in RESIDUAL_CONCENTRATION_SIZES)


def build_residual_concentration_variants(residual_weights=RESIDUAL_CONCENTRATION_WEIGHTS, portfolio_configs=None):
    """Build Residual ratio x portfolio-size matrix; Baseline_N12 preserves current 6+6 base-only logic."""
    portfolio_configs=tuple(portfolio_configs or build_portfolio_size_configs()); out=[]
    for w in residual_weights:
        for cfg in portfolio_configs:
            prefix="Baseline" if w==0 else f"Residual_{int(round(w*100)):02d}"
            out.append({**cfg,"name":f"{prefix}_N{cfg['total_holdings']}","residual_variant":prefix,"base_weight":round(1-w,2),"residual_weight":round(w,2),"vcp_weight":0.0})
    return tuple(out)


def run_residual_concentration_variant(prices,us,jp,start,end,variant,benchmark_mode=None,method="simple"):
    prices=prices.sort_index(); rets=prices.pct_change(); dates=rebalance_dates(prices.index,start,end)
    out=pd.Series(0.,index=prices.loc[start:end].index); records=[]; scores=[]; turns=[]; prev={}
    for i,t in enumerate(dates):
        future=prices.index[prices.index>t]
        if not len(future): continue
        trade=future[0]; next_t=dates[i+1] if i+1<len(dates) else pd.Timestamp(end); hold=out.index[(out.index>=trade)&(out.index<=next_t)]
        frames=[]
        for region,tickers,topn in (("US",us,variant["us_holdings"]),("JP",jp,variant["jp_holdings"])):
            base=score_universe(prices,[x for x in tickers if x in prices],t)
            residual=compute_residual_momentum_score(prices,base.index,t,region,benchmark_mode=benchmark_mode,method=method)
            d=combine_residual_score(base,residual,variant).head(topn).copy(); d["Region"]=region; frames.append(d)
        p=pd.concat(frames) if any(not x.empty for x in frames) else pd.DataFrame()
        if not p.empty:
            inv=1/p.Volatility; p["Weight"]=inv/inv.sum()
        weights=p.Weight.to_dict() if not p.empty else {}
        out.loc[hold]=rets.reindex(hold)[list(weights)].mul(pd.Series(weights)).sum(axis=1) if weights else 0.
        turns.append({"variant":variant["name"],"screen_date":t,"trade_date":trade,"turnover":sum(abs(weights.get(k,0)-prev.get(k,0)) for k in set(weights)|set(prev))/2,"total_holdings":variant["total_holdings"],"us_holdings":variant["us_holdings"],"jp_holdings":variant["jp_holdings"]}); prev=weights
        hist=asof_prices(prices,t).ffill()
        for ticker,row in p.iterrows():
            region=row.get("Region"); bench=_resolve_benchmark(hist,region,benchmark_mode)
            rec={"variant":variant["name"],"screen_date":t,"trade_date":trade,"ticker":ticker,"total_holdings":variant["total_holdings"],"us_holdings":variant["us_holdings"],"jp_holdings":variant["jp_holdings"],**row.to_dict()}; records.append(rec)
            scores.append({"date":t,"ticker":ticker,"market":region,"variant":variant["name"],"total_holdings":variant["total_holdings"],"us_holdings":variant["us_holdings"],"jp_holdings":variant["jp_holdings"],"base_score":row.get("base_score"),"residual_score":row.get("residual_score"),"final_score":row.get("Final_Score"),"residual_weight":variant["residual_weight"],"base_weight":variant["base_weight"],"stock_return_3m":_window_return(hist,ticker,63),"stock_return_6m":_window_return(hist,ticker,126),"stock_return_12m":_window_return(hist,ticker,252),"benchmark_return_3m":_window_return(hist,bench,63),"benchmark_return_6m":_window_return(hist,bench,126),"benchmark_return_12m":_window_return(hist,bench,252),"residual_return_3m":_window_return(hist,ticker,63)-_window_return(hist,bench,63) if bench else np.nan,"residual_return_6m":_window_return(hist,ticker,126)-_window_return(hist,bench,126) if bench else np.nan,"residual_return_12m":_window_return(hist,ticker,252)-_window_return(hist,bench,252) if bench else np.nan,"benchmark_used":bench,"residual_method":method,"selected_flag":True,"weight":row.get("Weight")})
    return out,pd.DataFrame(records),pd.DataFrame(scores),pd.DataFrame(turns)


def compute_concentration_diagnostics(selected):
    rows=[]
    if selected.empty or "Weight" not in selected: return pd.DataFrame(rows)
    for v,g in selected.groupby("variant"):
        by=g.groupby("screen_date"); maxw=by.Weight.max(); hhi=by.Weight.apply(lambda x: float((x**2).sum())); top3=by.Weight.apply(lambda x: float(x.sort_values(ascending=False).head(3).sum()))
        rows.append({"variant":v,"average_number_of_holdings":by.ticker.nunique().mean(),"average_max_single_name_weight":maxw.mean(),"max_observed_single_name_weight":maxw.max(),"average_portfolio_herfindahl_index":hhi.mean(),"average_top_3_weight":top3.mean(),"highest_concentration_date":str(maxw.idxmax()) if len(maxw) else ""})
    return pd.DataFrame(rows)


def _variant_parts(name):
    a,b=str(name).rsplit("_N",1); return a,int(b)


def compute_best_by_portfolio_size(summary):
    w=summary.copy(); w["residual_variant"]=[_variant_parts(i)[0] for i in w.index]; w["total_holdings"]=[_variant_parts(i)[1] for i in w.index]; rows=[]
    for n,g in w.groupby("total_holdings"):
        rows.append({"total_holdings":n,"best_cagr_ratio":g.CAGR.idxmax(),"best_sharpe_ratio":g.Sharpe.idxmax(),"best_sortino_ratio":g.Sortino.idxmax(),"best_calmar_ratio":g.Calmar.idxmax(),"lowest_maxdd_ratio":g.Max_Drawdown.abs().idxmin()})
    return pd.DataFrame(rows).sort_values("total_holdings",ascending=False)


def compute_best_by_residual_ratio(summary):
    w=summary.copy(); w["residual_variant"]=[_variant_parts(i)[0] for i in w.index]; w["total_holdings"]=[_variant_parts(i)[1] for i in w.index]; rows=[]
    for rv,g in w.groupby("residual_variant"):
        rows.append({"residual_variant":rv,"best_cagr_portfolio_size":_variant_parts(g.CAGR.idxmax())[1],"best_sharpe_portfolio_size":_variant_parts(g.Sharpe.idxmax())[1],"best_sortino_portfolio_size":_variant_parts(g.Sortino.idxmax())[1],"best_calmar_portfolio_size":_variant_parts(g.Calmar.idxmax())[1],"lowest_maxdd_portfolio_size":_variant_parts(g.Max_Drawdown.abs().idxmin())[1]})
    return pd.DataFrame(rows)


def compute_sweet_spot_analysis(summary):
    base=summary.loc["Baseline_N12"] if "Baseline_N12" in summary.index else summary.iloc[0]
    rows=[]
    for name,row in summary.iterrows():
        n=_variant_parts(name)[1]; simultaneous=row.CAGR>base.CAGR and abs(row.Max_Drawdown)<abs(base.Max_Drawdown) and row.Calmar>base.Calmar
        judgment="strong improvement" if simultaneous and row.Sharpe>base.Sharpe and row.Sortino>base.Sortino else ("clear improvement" if row.Calmar>base.Calmar and row.CAGR>=base.CAGR*.98 and abs(row.Max_Drawdown)<=abs(base.Max_Drawdown)*1.05 else ("mixed" if row.CAGR>base.CAGR or row.Calmar>base.Calmar else "worse"))
        rows.append({"variant":name,"total_holdings":n,"cagr_delta_vs_baseline_n12":row.CAGR-base.CAGR,"maxdd_abs_delta_vs_baseline_n12":abs(row.Max_Drawdown)-abs(base.Max_Drawdown),"calmar_delta_vs_baseline_n12":row.Calmar-base.Calmar,"simultaneous_cagr_maxdd_calmar_improvement":simultaneous,"zone":"concentration" if n<12 else "current" if n==12 else "diversification" if n<30 else "high_diversification_reference","judgment":judgment})
    return pd.DataFrame(rows)


def compute_diversification_reference_analysis(summary):
    rows=[]
    for rv in sorted({_variant_parts(i)[0] for i in summary.index}):
        base_name=f"{rv}_N12"
        if base_name not in summary.index: continue
        base=summary.loc[base_name]
        for n in (4,6,8,10,16,20,24,30,40):
            name=f"{rv}_N{n}"
            if name in summary.index:
                row=summary.loc[name]; rows.append({"residual_variant":rv,"comparison":f"N12_vs_N{n}","variant":name,"cagr_delta":row.CAGR-base.CAGR,"maxdd_abs_delta":abs(row.Max_Drawdown)-abs(base.Max_Drawdown),"vol_delta":row.Annualized_Volatility-base.Annualized_Volatility,"calmar_delta":row.Calmar-base.Calmar,"role":"high_diversification_reference" if n in (30,40) else "practical_diversification" if n>12 else "concentration_reference"})
    return pd.DataFrame(rows)


CONCENTRATION_SELECTION_DIFF_VARIANTS=("Residual_60_N12","Residual_60_N10","Residual_60_N8","Residual_65_N12","Residual_65_N10","Residual_65_N8","Residual_65_N20","Residual_65_N24","Residual_100_N12","Residual_100_N8","Residual_100_N20")

def _fast_selection_diff_detailed(selected,score_components,baseline_variant="Baseline_N12",target_variants=None):
    cols=["date","variant","ticker","change","base_score","residual_score","final_score","weight","benchmark_used","residual_method"]
    if selected.empty: return pd.DataFrame(columns=cols)
    work=selected.copy(); work["_date"]=pd.to_datetime(work["screen_date"]).dt.normalize()
    variants=[v for v in (target_variants or sorted(set(work.variant)-{baseline_variant})) if v in set(work.variant) and v!=baseline_variant]
    if not variants or baseline_variant not in set(work.variant): return pd.DataFrame(columns=cols)
    score_lookup={}
    if not score_components.empty:
        sc=score_components.copy(); sc["_date"]=pd.to_datetime(sc["date"]).dt.normalize()
        keep_cols=[c for c in ["base_score","residual_score","final_score","weight","benchmark_used","residual_method"] if c in sc.columns]
        score_lookup=sc.drop_duplicates(["_date","variant","ticker"],keep="last").set_index(["_date","variant","ticker"])[keep_cols].to_dict("index")
    rows=[]
    grouped={k:set(g.ticker) for k,g in work.groupby(["_date","variant"])}
    dates=sorted({d for d,v in grouped if v==baseline_variant})
    for dt in dates:
        base=grouped.get((dt,baseline_variant),set())
        for v in variants:
            cur=grouped.get((dt,v),set())
            for change,tickers,lookup_variant in (("added",cur-base,v),("removed",base-cur,baseline_variant)):
                for ticker in sorted(tickers):
                    rec={"date":dt,"variant":v,"ticker":ticker,"change":change}
                    rec.update(score_lookup.get((dt,lookup_variant,ticker),{})); rows.append(rec)
    return pd.DataFrame(rows,columns=cols)

def compare_concentration_selection_diff(selected,score_components,baseline_variant="Baseline_N12",extra_variants=None):
    keep=set(CONCENTRATION_SELECTION_DIFF_VARIANTS)
    if extra_variants: keep.update(x for x in extra_variants if x)
    return _fast_selection_diff_detailed(selected,score_components,baseline_variant=baseline_variant,target_variants=sorted(keep))


def _empty_concentration_outputs(out,variants,dq,failures,meta):
    idx=[v["name"] for v in variants]; cols=["CAGR","Annualized_Volatility","Max_Drawdown","Sharpe","Sortino","Calmar","Total_Return","Best_Year","Worst_Year","Monthly_Win_Rate","Turnover","Number_of_Rebalances","Judgment"]
    summary=pd.DataFrame(index=idx,columns=cols); summary.index.name="Variant"; summary["Judgment"]="skipped_no_usable_data"
    empty=pd.DataFrame(); conc=compute_concentration_diagnostics(empty); bps=compute_best_by_portfolio_size(summary); brr=compute_best_by_residual_ratio(summary); sweet=compute_sweet_spot_analysis(summary); div=compute_diversification_reference_analysis(summary)
    for name,df in (("variant_summary.csv",summary),("annual_returns.csv",empty),("monthly_returns.csv",empty),("drawdown_series.csv",empty),("turnover.csv",empty),("selected_tickers.csv",empty),("selection_diff.csv",empty),("score_components.csv",empty),("data_quality.csv",dq),("download_failures.csv",failures),("concentration_diagnostics.csv",conc),("best_by_portfolio_size.csv",bps),("best_by_residual_ratio.csv",brr),("sweet_spot_analysis.csv",sweet),("diversification_reference_analysis.csv",div)):
        df.to_csv(out/name,index=(name in ("variant_summary.csv","annual_returns.csv","monthly_returns.csv","drawdown_series.csv")))
    summary.to_json(out/"variant_summary.json",orient="index",indent=2); return summary,conc,bps,brr,sweet,div


def run_residual_concentration_audit(prices=None,us=None,jp=None,start="2015-01-01",end=None,output_dir="artifacts/residual_concentration",downloader=None,quick=False):
    """Run Residual Core x Portfolio Concentration audit (70 variants full, 21 variants quick, quarterly, no exit/regime/VCP)."""
    import json
    end=end or pd.Timestamp.today().date().isoformat(); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); (out/"cache").mkdir(exist_ok=True); Path("reports").mkdir(exist_ok=True); reset_insufficient_history_warnings(); LOG.info("cache loading start")
    if us is None or jp is None: us,jp=get_live_universe()
    us=[normalize_yfinance_ticker(t) for t in us]; jp=[normalize_yfinance_ticker(t) for t in jp]
    modes=build_benchmark_modes(); bench_tickers=sorted({x for mode in modes.values() for vals in mode.values() for x in vals}); portfolio_configs=build_portfolio_size_configs()
    if quick:
        portfolio_configs=tuple(c for c in portfolio_configs if c["total_holdings"] in (24,20,16,12,10,8,6))
        baseline_cfg=next(c for c in build_portfolio_size_configs() if c["total_holdings"]==12)
        variants=(build_residual_concentration_variants(residual_weights=(0.0,),portfolio_configs=(baseline_cfg,))
                  + build_residual_concentration_variants(residual_weights=(0.60,0.65,1.00),portfolio_configs=portfolio_configs))
    else:
        variants=build_residual_concentration_variants(residual_weights=RESIDUAL_CONCENTRATION_WEIGHTS,portfolio_configs=portfolio_configs)
    LOG.info("variant count: %s",len(variants))
    requested=list(dict.fromkeys([*us,*jp])); failures=pd.DataFrame(columns=["ticker","reason"]); cache_meta={"cache_used":False,"cache_path":str(out/"cache"),"cache_source":"provided_prices"}
    if prices is None:
        source=Path("artifacts/residual_full_sweep/cache")
        ok,reason=validate_price_cache(source,start,end,requested,bench_tickers) if source.exists() else (False,"missing_prior_cache")
        if ok:
            prices,bench=load_price_cache(source); prices=pd.concat([prices,bench],axis=1).loc[:,lambda x:~x.columns.duplicated()].sort_index(); cache_meta={"cache_used":True,"cache_source":str(source),"cache_path":str(source),"cache_status":"prior_full_sweep_hit"}
        else:
            prices,failures,requested,cache_meta=build_price_cache(requested,bench_tickers,start,end,out/"cache",downloader=downloader); cache_meta["cache_source"]="download_or_local_cache"
    else:
        pd.DataFrame({"ticker":[*us,*jp]}).to_csv(out/"cache"/"universe.csv",index=False)
        prices.to_pickle(out/"cache"/"prices.pkl"); prices[[c for c in bench_tickers if c in prices.columns]].to_pickle(out/"cache"/"benchmarks.pkl")
        (out/"cache"/"cache_metadata.json").write_text(json.dumps({**_cache_metadata(start,end,[*us,*jp],bench_tickers),**cache_meta},indent=2,default=str),encoding="utf-8")
    LOG.info("cache loading end")
    benchmark_status={name:{region:_resolve_benchmark(prices,region,mode) for region in ("US","JP")} for name,mode in modes.items()}
    LOG.info("data quality filtering start")
    dq,insufficient,excluded,usable=build_live_data_quality_report(prices,requested,us,jp,failures,start,end,benchmark_status={k:str(v) for k,v in benchmark_status.items()})
    dq=pd.concat([dq,pd.DataFrame([{"metric":"cache_used","value":cache_meta.get("cache_used",False)},{"metric":"cache_source","value":cache_meta.get("cache_source","")},{"metric":"cache_path","value":cache_meta.get("cache_path","")},{"metric":"cache_created_at","value":cache_meta.get("created_at","")}])],ignore_index=True)
    us_usable=[t for t in us if t in usable]; jp_usable=[t for t in jp if t in usable]; LOG.info("data quality filtering end; usable universe size=%s",len(usable))
    meta={"generated_at":pd.Timestamp.now("UTC").isoformat(),"data_start":str(prices.index.min().date()) if len(prices.index) else "","data_end":str(prices.index.max().date()) if len(prices.index) else "","usable_start":str(pd.Timestamp(start).date()),"usable_end":str(pd.Timestamp(end).date()),"rebalance_frequency":"quarterly / about every 90 days","ttl_days":90,"universe_size_requested":len(requested),"universe_size_downloaded":len([t for t in requested if t in prices.columns]),"universe_size_usable":len(usable),"us_usable_count":len(us_usable),"jp_usable_count":len(jp_usable),"residual_ratios_tested":list((0.0,0.60,0.65,1.00) if quick else RESIDUAL_CONCENTRATION_WEIGHTS),"portfolio_sizes_tested":[c["total_holdings"] for c in portfolio_configs],"variant_count":len(variants),"quick_mode":quick,"selection_diff_status":"pending","residual_method":"simple","benchmark_mode":"broad_default","cache_used":cache_meta.get("cache_used",False),"cache_source":cache_meta.get("cache_source",""),"cache_path":cache_meta.get("cache_path",str(out/"cache")),"whether_exit_protocol_used":False,"whether_regime_filter_used":False,"whether_vcp_used":False,"whether_sector_residual_used":False,"whether_downside_penalty_used":False,"whether_correlation_penalty_used":False,"data_source":"yfinance / Wikipedia / existing free universe" if prices is None else "provided/demo prices","notes_on_missing_data":"Failed/insufficient tickers are logged and excluded; missing residual inputs are neutralized.","notes_on_survivorship_bias":"Current free-data universes may be applied backward; historical constituent bias and survivorship bias remain.","notes_on_portfolio_size_reference":"N4/N6 are concentration limit tests; N30/N40 are high-diversification references, not production recommendations.","notes_on_high_diversification_reference":"High-diversification variants test signal dilution versus individual-stock risk reduction."}
    if prices.empty or not (us_usable or jp_usable):
        summary,conc,bps,brr,sweet,div=_empty_concentration_outputs(out,variants,dq,failures,meta); (out/"audit_metadata.json").write_text(json.dumps(meta,indent=2,default=str),encoding="utf-8"); write_residual_concentration_report(summary,dq,meta,conc,bps,brr,sweet,div,out); LOG.info("report writing end"); return summary
    returns={}; selected=[]; scores=[]; turns=[]; default_mode=modes["broad_default"]
    for v in variants:
        LOG.info("variant start: %s",v["name"])
        r,sel,sc,tu=run_residual_concentration_variant(prices,us_usable,jp_usable,start,end,v,default_mode,"simple"); returns[v["name"]]=r; selected.append(sel); scores.append(sc); turns.append(tu); LOG.info("variant end: %s",v["name"])
    selected=pd.concat(selected,ignore_index=True); score_components=pd.concat(scores,ignore_index=True); turnover=pd.concat(turns,ignore_index=True); summary=pd.DataFrame({k:metrics(v) for k,v in returns.items()}).T; summary.index.name="Variant"; summary["Turnover"]=turnover.groupby("variant").turnover.mean(); summary["Number_of_Rebalances"]=turnover.groupby("variant").size(); base_for_judge=summary.loc["Baseline_N12"]; summary["Judgment"]=[("baseline" if idx=="Baseline_N12" else _judge(row.rename("Baseline" if idx=="Baseline_N12" else idx),base_for_judge)) for idx,row in summary.iterrows()]
    annual=pd.DataFrame({k:(1+v).resample("YE").prod()-1 for k,v in returns.items()}); monthly=pd.DataFrame({k:(1+v).resample("ME").prod()-1 for k,v in returns.items()}); draw=pd.DataFrame({k:((1+v).cumprod()/(1+v).cumprod().cummax()-1) for k,v in returns.items()})
    LOG.info("diagnostics writing start")
    conc=compute_concentration_diagnostics(selected); bps=compute_best_by_portfolio_size(summary); brr=compute_best_by_residual_ratio(summary); sweet=compute_sweet_spot_analysis(summary); div=compute_diversification_reference_analysis(summary); insuff_summary=get_insufficient_history_summary(); insuff_summary.to_csv(out/"insufficient_history_summary.csv",index=False); meta["insufficient_history_warning_count"]=int(insuff_summary["count"].sum()) if not insuff_summary.empty else 0; meta["insufficient_history_unique_tickers"]=int(insuff_summary["ticker"].nunique()) if not insuff_summary.empty else 0
    for name,df in (("variant_summary.csv",summary),("annual_returns.csv",annual),("monthly_returns.csv",monthly),("drawdown_series.csv",draw),("turnover.csv",turnover),("selected_tickers.csv",selected),("score_components.csv",score_components),("data_quality.csv",dq),("download_failures.csv",failures),("concentration_diagnostics.csv",conc),("best_by_portfolio_size.csv",bps),("best_by_residual_ratio.csv",brr),("sweet_spot_analysis.csv",sweet),("diversification_reference_analysis.csv",div)):
        df.to_csv(out/name,index=(name in ("variant_summary.csv","annual_returns.csv","monthly_returns.csv","drawdown_series.csv")))
    summary.to_json(out/"variant_summary.json",orient="index",indent=2); LOG.info("diagnostics writing end")
    conc_slice=summary[[ _variant_parts(i)[1] in (8,10) for i in summary.index]]; high_slice=summary[[ _variant_parts(i)[1] in (30,40) for i in summary.index]]
    meta.update({"best_cagr_variant":summary.CAGR.idxmax(),"best_sharpe_variant":summary.Sharpe.idxmax(),"best_sortino_variant":summary.Sortino.idxmax(),"best_calmar_variant":summary.Calmar.idxmax(),"lowest_maxdd_variant":summary.Max_Drawdown.abs().idxmin(),"best_balanced_variant":sweet.sort_values(["judgment","calmar_delta_vs_baseline_n12"],ascending=[True,False]).variant.iloc[0] if not sweet.empty else None,"best_concentration_variant":conc_slice.Calmar.idxmax() if not conc_slice.empty else None,"best_high_diversification_reference_variant":high_slice.Calmar.idxmax() if not high_slice.empty else None})
    LOG.info("selection_diff start")
    try:
        extra=[meta.get("best_cagr_variant"),meta.get("best_calmar_variant"),meta.get("best_concentration_variant"),meta.get("best_high_diversification_reference_variant")]
        diff=compare_concentration_selection_diff(selected,score_components,baseline_variant="Baseline_N12",extra_variants=extra); diff.to_csv(out/"selection_diff.csv",index=False); meta["selection_diff_status"]="ok"; LOG.info("selection_diff end")
    except Exception as exc:
        LOG.warning("selection_diff failed: %s",exc); pd.DataFrame([{"selection_diff_status":"failed","reason":str(exc),"baseline_variant":"Baseline_N12"}]).to_csv(out/"selection_diff.csv",index=False); meta["selection_diff_status"]="failed"; meta["selection_diff_error"]=str(exc)
    LOG.info("report writing start")
    (out/"audit_metadata.json").write_text(json.dumps(meta,indent=2,default=str),encoding="utf-8"); write_residual_concentration_report(summary,dq,meta,conc,bps,brr,sweet,div,out); LOG.info("report writing end"); return summary


TTL_QUICK_SELECTIONS=((0.55,12,6,6),(0.60,12,6,6),(0.65,12,6,6))
TTL_FULL_SELECTIONS=((0.50,12,6,6),(0.55,12,6,6),(0.60,12,6,6),(0.65,12,6,6),(0.70,12,6,6),(0.60,10,5,5),(0.65,10,5,5),(1.00,12,6,6))
TTL_RENEWAL_PROTOCOLS=("Rank","Residual","Composite")

def build_ttl_renewal_variants(quick=False, include_baseline=True):
    sels=TTL_QUICK_SELECTIONS if quick else TTL_FULL_SELECTIONS; ttls=(60,90,120,180) if quick else (30,60,90,120,180); out=[]
    if include_baseline: out.append({"name":"Baseline_N12_TTL90","selection_name":"Baseline_N12","base_weight":1.0,"residual_weight":0.0,"total_holdings":12,"us_holdings":6,"jp_holdings":6,"ttl_days":90,"renewal_protocol":None,"is_baseline":True})
    for w,n,u,j in sels:
        s=f"Residual_{int(round(w*100)):02d}_N{n}"
        for ttl in ttls: out.append({"name":f"{s}_TTL{ttl}","selection_name":s,"base_weight":round(1-w,2),"residual_weight":round(w,2),"total_holdings":n,"us_holdings":u,"jp_holdings":j,"ttl_days":ttl,"renewal_protocol":None,"is_baseline":False})
        for proto in TTL_RENEWAL_PROTOCOLS: out.append({"name":f"{s}_TTL90_Renew30_{proto}","selection_name":s,"base_weight":round(1-w,2),"residual_weight":round(w,2),"total_holdings":n,"us_holdings":u,"jp_holdings":j,"ttl_days":90,"renewal_protocol":proto.lower(),"renewal_extension_days":30,"is_baseline":False})
    return tuple(out)

def _next_trade_date(index, dt):
    idx=index[index>=pd.Timestamp(dt)]
    return idx[0] if len(idx) else None

def _select_ttl_candidates(prices,us,jp,as_of_date,variant,benchmark_mode=None):
    frames=[]; allscores=[]
    for region,tickers,topn in (("US",us,variant["us_holdings"]),("JP",jp,variant["jp_holdings"])):
        base=score_universe(prices,[x for x in tickers if x in prices],as_of_date)
        residual=compute_residual_momentum_score(prices,base.index,as_of_date,region,benchmark_mode=benchmark_mode,method="simple")
        d=combine_residual_score(base,residual,variant).copy(); d["Region"]=region; d["market_rank"]=np.arange(1,len(d)+1); allscores.append(d.head(max(topn*2,12)).copy())
        frames.append(d.head(topn).copy())
    p=pd.concat(frames) if any(not x.empty for x in frames) else pd.DataFrame()
    if not p.empty:
        inv=1/p.Volatility.replace(0,np.nan); p["Weight"]=inv/inv.sum()
    scores=pd.concat(allscores) if any(not x.empty for x in allscores) else pd.DataFrame()
    return p,scores

def _health_row(prices,us,jp,date,ticker,region,variant,benchmark_mode=None):
    topn=variant["us_holdings"] if region=="US" else variant["jp_holdings"]; universe=us if region=="US" else jp
    _,scores=_select_ttl_candidates(prices,us if region=="US" else [],jp if region=="JP" else [],date,{**variant,"us_holdings":topn if region=="US" else 0,"jp_holdings":topn if region=="JP" else 0},benchmark_mode)
    if ticker in scores.index: row=scores.loc[ticker]
    else:
        base=score_universe(prices,[x for x in universe if x in prices],date); residual=compute_residual_momentum_score(prices,base.index,date,region,benchmark_mode=benchmark_mode); d=combine_residual_score(base,residual,variant); d["market_rank"]=np.arange(1,len(d)+1); row=d.loc[ticker] if ticker in d.index else pd.Series(dtype=float)
    s=asof_prices(prices,date).ffill()[ticker].dropna() if ticker in prices else pd.Series(dtype=float)
    above50=bool(len(s)>=50 and s.iloc[-1]>=s.tail(50).mean())
    rank=float(row.get("market_rank",np.inf)); residual_score=float(row.get("residual_raw",row.get("residual_score",0)))
    rank_pass=rank<=topn*2; residual_pass=residual_score>0; composite_count=int(rank_pass)+int(residual_pass)+int(above50)
    return {"market_rank":rank,"rank_pass":rank_pass,"residual_score":residual_score,"residual_pass":residual_pass,"price_above_50dma":above50,"composite_count":composite_count,"composite_pass":composite_count>=2}

def _renewal_pass(protocol,h):
    return h["rank_pass"] if protocol=="rank" else h["residual_pass"] if protocol=="residual" else h["composite_pass"]

def run_ttl_renewal_variant(prices,us,jp,start,end,variant,benchmark_mode=None,detail_logs=True):
    prices=prices.sort_index(); idx=prices.loc[start:end].index; rets=prices.pct_change(); out=pd.Series(0.,index=idx); selected=[]; scores_lite=[]; turns=[]; trades=[]; holds=[]; decisions=[]; events=[]; prev={}; t=_next_trade_date(prices.index,pd.Timestamp(start)); cycle=0
    while t is not None and t<=pd.Timestamp(end):
        screen=prices.index[prices.index<t][-1] if len(prices.index[prices.index<t]) else t
        p,sc=_select_ttl_candidates(prices,us,jp,screen,variant,benchmark_mode); weights=p.Weight.to_dict() if not p.empty else {}; cycle+=1
        base_end=_next_trade_date(prices.index,t+pd.Timedelta(days=variant["ttl_days"])); base_end=base_end if base_end is not None else prices.index[-1]
        final_end=base_end; active=dict(weights); protocol=variant.get("renewal_protocol")
        if protocol:
            if detail_logs: LOG.info("renewal health check start: %s %s",variant["name"],screen)
            final_end=_next_trade_date(prices.index,t+pd.Timedelta(days=120)) or prices.index[-1]
            for ticker,row in p.iterrows():
                h=_health_row(prices,us,jp,base_end,ticker,row.get("Region"),variant,benchmark_mode); ok=_renewal_pass(protocol,h); decisions.append({"variant":variant["name"],"ticker":ticker,"screen_date":screen,"trade_date":t,"health_check_date":base_end,"protocol":protocol,"renewed":ok,**h})
                if not ok: active.pop(ticker,None)
            if detail_logs: LOG.info("renewal health check end: %s",variant["name"])
            # weekly degradation only for composite extension; non-pass becomes cash until normal 120-day cycle end
            if protocol=="composite" and active:
                for wd in pd.date_range(base_end,final_end,freq="W-FRI"):
                    check=_next_trade_date(prices.index,wd)
                    if check is None or check>final_end: continue
                    for ticker in list(active):
                        region=p.loc[ticker].get("Region"); h=_health_row(prices,us,jp,check,ticker,region,variant,benchmark_mode)
                        if h["composite_count"]<2: active.pop(ticker,None); events.append({"variant":variant["name"],"date":check,"ticker":ticker,"event":"extension_degradation_end","holding_days":int((check-t).days),**h})
        hold=out.index[(out.index>=t)&(out.index<=final_end)]
        if len(hold): out.loc[hold]=rets.reindex(hold)[list(active)].mul(pd.Series(active)).sum(axis=1) if active else 0.
        turns.append({"variant":variant["name"],"screen_date":screen,"trade_date":t,"turnover":sum(abs(weights.get(k,0)-prev.get(k,0)) for k in set(weights)|set(prev))/2,"names_changed":len(set(weights)^set(prev)),"ttl_days":variant["ttl_days"],"renewal_protocol":protocol or "fixed"}); prev=weights
        for ticker,row in p.iterrows():
            exit_date=final_end if ticker in active else base_end; hd=int((exit_date-t).days); selected.append({"variant":variant["name"],"screen_date":screen,"trade_date":t,"ticker":ticker,"exit_date":exit_date,"holding_days":hd,**row.to_dict()}); holds.append({"variant":variant["name"],"ticker":ticker,"entry_date":t,"exit_date":exit_date,"holding_days":hd,"renewal_protocol":protocol or "fixed"}); trades += [{"variant":variant["name"],"date":t,"ticker":ticker,"action":"BUY","weight":weights.get(ticker,0)},{"variant":variant["name"],"date":exit_date,"ticker":ticker,"action":"SELL","weight":weights.get(ticker,0)}]; events.append({"variant":variant["name"],"date":t,"ticker":ticker,"event":"entry","holding_days":0}); events.append({"variant":variant["name"],"date":exit_date,"ticker":ticker,"event":"exit","holding_days":hd})
        if not sc.empty:
            tmp=sc.reset_index().rename(columns={"index":"ticker"}); tmp["variant"]=variant["name"]; tmp["date"]=screen; scores_lite.append(tmp.head(max(variant["total_holdings"]*2,12)))
        t=_next_trade_date(prices.index,final_end+pd.Timedelta(days=1))
    return out,pd.DataFrame(selected),pd.concat(scores_lite,ignore_index=True) if scores_lite else pd.DataFrame(),pd.DataFrame(turns),pd.DataFrame(trades),pd.DataFrame(holds),pd.DataFrame(decisions),pd.DataFrame(events)

def _cost_adjusted_returns(r,turnover,tax_rate=0.20315,slippage_bps=10):
    rr=pd.Series(r).copy(); slip=tax=0.0
    for _,tr in turnover.iterrows():
        d=pd.Timestamp(tr.get("trade_date")); tv=float(tr.get("turnover",0)); cost=tv*slippage_bps/10000; slip+=cost
        if d in rr.index: rr.loc[d]-=cost
        tc=max(0,float(rr.loc[:d].mean() if d in rr.index and len(rr.loc[:d]) else 0))*tv*tax_rate; tax+=tc
        if d in rr.index: rr.loc[d]-=tc
    return rr,slip,tax

def _load_ttl_cache(candidates,start,end,requested,bench_tickers):
    for source in candidates:
        if not source: continue
        p=Path(source)
        if not p.exists(): continue
        ok,reason=validate_price_cache(p,start,end,requested,bench_tickers)
        if ok:
            prices,bench=load_price_cache(p); return pd.concat([prices,bench],axis=1).loc[:,lambda x:~x.columns.duplicated()].sort_index(), {"cache_used":True,"cache_source":str(p),"cache_status":"hit","prices_cache_found":(p/"prices.pkl").exists(),"benchmarks_cache_found":(p/"benchmarks.pkl").exists(),"cache_loaded_at":pd.Timestamp.now("UTC").isoformat()}
        LOG.warning("ttl_renewal cache fallback from %s: %s",p,reason)
    return None,{"cache_used":False,"cache_source":"","cache_status":"miss","prices_cache_found":False,"benchmarks_cache_found":False,"cache_loaded_at":""}

def run_ttl_renewal_audit(prices=None,us=None,jp=None,start="2015-01-01",end=None,output_dir="artifacts/ttl_renewal",downloader=None,quick=False,cache_dir=None,force_refresh_cache=False,resume=False,tax_rate=0.20315,slippage_bps=10,full_score_output=False):
    import json, time
    t0=time.time(); end=end or pd.Timestamp.today().date().isoformat(); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); (out/"cache").mkdir(exist_ok=True); Path("reports").mkdir(exist_ok=True); reset_insufficient_history_warnings(); LOG.info("audit start: ttl_renewal; mode=%s", "quick" if quick else "full")
    if us is None or jp is None: us,jp=get_live_universe()
    us=[normalize_yfinance_ticker(t) for t in us]; jp=[normalize_yfinance_ticker(t) for t in jp]; modes=build_benchmark_modes(); default_mode=modes["broad_default"]; bench_tickers=sorted({x for mode in modes.values() for vals in mode.values() for x in vals}); requested=list(dict.fromkeys([*us,*jp])); variants=build_ttl_renewal_variants(quick=quick); LOG.info("variant count: %s",len(variants)-1)
    failures=pd.DataFrame(columns=["ticker","reason"]); cache_meta={"cache_used":prices is not None,"cache_source":"provided_prices" if prices is not None else ""}
    if prices is None:
        LOG.info("cache loading start")
        if not force_refresh_cache: prices,cache_meta=_load_ttl_cache([cache_dir,out/"cache","artifacts/residual_concentration/cache","artifacts/residual_full_sweep/cache"],start,end,requested,bench_tickers)
        if prices is None: prices,failures,requested,cache_meta=build_price_cache(requested,bench_tickers,start,end,out/"cache",downloader=downloader)
        LOG.info("cache loading end: %s",cache_meta.get("cache_source"))
    dq,insufficient,excluded,usable=build_live_data_quality_report(prices,requested,us,jp,failures,start,end)
    us_usable=[t for t in us if t in usable]; jp_usable=[t for t in jp if t in usable]; LOG.info("data quality filtering end; usable universe size=%s",len(usable))
    for k in ("cache_used","cache_source","prices_cache_found","benchmarks_cache_found","cache_loaded_at"):
        dq=pd.concat([dq,pd.DataFrame([{"metric":k,"value":cache_meta.get(k,"")}])],ignore_index=True)
    dq=pd.concat([dq,pd.DataFrame([{"metric":"universe_size","value":len(requested)},{"metric":"usable_universe_size","value":len(usable)}])],ignore_index=True)
    completed=set(); cp=out/"completed_variants.csv"
    if resume and cp.exists(): completed=set(pd.read_csv(cp).variant.astype(str))
    returns={}; selected=[]; scores=[]; turns=[]; trades=[]; holds=[]; decisions=[]; events=[]
    for v in variants:
        if resume and v["name"] in completed: LOG.info("variant skip completed: %s",v["name"]); continue
        LOG.info("variant start: %s ttl=%s renewal=%s",v["name"],v["ttl_days"],v.get("renewal_protocol") or "fixed")
        r,sel,sc,tu,tr,hp,rd,ev=run_ttl_renewal_variant(prices,us_usable,jp_usable,start,end,v,default_mode); returns[v["name"]]=r; selected.append(sel); scores.append(sc); turns.append(tu); trades.append(tr); holds.append(hp); decisions.append(rd); events.append(ev)
        pd.DataFrame([{"variant":v["name"]}]).to_csv(cp,mode="a",header=not cp.exists(),index=False); LOG.info("variant end: %s",v["name"])
    selected=pd.concat(selected,ignore_index=True) if selected else pd.DataFrame(); 
    if resume and not returns:
        LOG.info("resume found no variants to run; existing partial outputs retained")
        return pd.DataFrame()
    turnover=pd.concat(turns,ignore_index=True) if turns else pd.DataFrame(); trade_log=pd.concat(trades,ignore_index=True) if trades else pd.DataFrame(); holding_periods=pd.concat(holds,ignore_index=True) if holds else pd.DataFrame(); renewal_decisions=pd.concat(decisions,ignore_index=True) if decisions else pd.DataFrame(); ttl_event_log=pd.concat(events,ignore_index=True) if events else pd.DataFrame(); score_components=pd.concat(scores,ignore_index=True) if scores else pd.DataFrame()
    summary=pd.DataFrame({k:metrics(v) for k,v in returns.items()}).T; summary.index.name="Variant"
    if not turnover.empty: summary["Turnover"]=turnover.groupby("variant").turnover.mean(); summary["Annualized_Turnover"]=summary["Turnover"]*252/summary.index.map(lambda x: next((v["ttl_days"] for v in variants if v["name"]==x),90)); summary["Number_of_Rebalances"]=turnover.groupby("variant").size(); summary["Average_names_changed_per_rebalance"]=turnover.groupby("variant").names_changed.mean()
    if not holding_periods.empty: summary["Average_Holding_Days"]=holding_periods.groupby("variant").holding_days.mean(); summary["Median_Holding_Days"]=holding_periods.groupby("variant").holding_days.median(); summary["Max_Holding_Days"]=holding_periods.groupby("variant").holding_days.max(); summary["Trade_Count"]=trade_log.groupby("variant").size()
    annual=pd.DataFrame({k:(1+v).resample("YE").prod()-1 for k,v in returns.items()}); monthly=pd.DataFrame({k:(1+v).resample("ME").prod()-1 for k,v in returns.items()}); draw=pd.DataFrame({k:((1+v).cumprod()/(1+v).cumprod().cummax()-1) for k,v in returns.items()})
    cost_rows=[]
    for k,r in returns.items():
        tu=turnover[turnover.variant==k] if not turnover.empty else pd.DataFrame(); nr,slip,tax=_cost_adjusted_returns(r,tu,tax_rate,slippage_bps); nm=metrics(nr); gross=summary.loc[k]
        cost_rows.append({"Variant":k,"Slippage_Adjusted_CAGR":cagr(_cost_adjusted_returns(r,tu,0,slippage_bps)[0]),"Tax_Adjusted_CAGR":cagr(_cost_adjusted_returns(r,tu,tax_rate,0)[0]),"Tax_Slippage_Adjusted_CAGR":nm["CAGR"],"Estimated_Tax_Drag":gross.CAGR-nm["CAGR"] if pd.notna(gross.CAGR) else np.nan,"Estimated_Slippage_Drag":slip,"Net_Sharpe":nm["Sharpe"],"Net_Calmar":nm["Calmar"]})
    cost_summary=pd.DataFrame(cost_rows).set_index("Variant") if cost_rows else pd.DataFrame()
    for c in cost_summary.columns: summary[c]=cost_summary[c]
    insuff_summary=get_insufficient_history_summary(); meta={"audit_name":"ttl_renewal","quick_mode":quick,"target_variant_count":21 if quick else 64,"variant_count_including_baseline":len(variants),"cache_used":cache_meta.get("cache_used",False),"cache_source":cache_meta.get("cache_source",""),"prices_cache_found":cache_meta.get("prices_cache_found",False),"benchmarks_cache_found":cache_meta.get("benchmarks_cache_found",False),"cache_loaded_at":cache_meta.get("cache_loaded_at",""),"universe_size":len(requested),"usable_universe_size":len(usable),"tax_rate":tax_rate,"slippage_bps":slippage_bps,"exit_protocol_enabled":False,"regime_filter_enabled":False,"vcp_enabled":False,"sector_residual_enabled":False,"downside_penalty_enabled":False,"correlation_penalty_enabled":False,"initial_period_exit_enabled":False,"renewal_protocol_enabled":True,"renewal_max_extension_days":30,"max_holding_days":"120 for renewal variants","cost_model_note":"Approximate only; not tax advice.","wall_time_seconds":round(time.time()-t0,2)}
    LOG.info("output writing start")
    for name,df,idxout in (("variant_summary.csv",summary,True),("annual_returns.csv",annual,True),("monthly_returns.csv",monthly,True),("drawdown_series.csv",draw,True),("turnover.csv",turnover,False),("trade_log.csv",trade_log,False),("holding_periods.csv",holding_periods,False),("renewal_decisions.csv",renewal_decisions,False),("ttl_event_log.csv",ttl_event_log,False),("cost_adjusted_summary.csv",cost_summary,True),("data_quality.csv",dq,False),("insufficient_history_summary.csv",insuff_summary,False)): df.to_csv(out/name,index=idxout)
    summary.to_json(out/"variant_summary.json",orient="index",indent=2); selected.to_csv(out/"selected_tickers.csv",index=False); score_components.to_csv(out/("score_components_full.csv" if full_score_output else "score_components_selected_only.csv"),index=False); (out/"audit_metadata.json").write_text(json.dumps(meta,indent=2,default=str),encoding="utf-8"); write_ttl_renewal_report(summary,dq,meta,out); LOG.info("audit complete; wall time %.1fs",time.time()-t0); return summary



TTL_COMPOSITE_FORENSICS_DEFAULT_VARIANTS=(
    "Residual_60_N12_TTL90",
    "Residual_60_N12_TTL90_Renew30_Composite",
    "Residual_65_N12_TTL90",
    "Residual_65_N12_TTL90_Renew30_Composite",
    "Baseline_N12_TTL90",
    "Residual_60_N12_TTL120",
    "Residual_60_N12_TTL90_Renew30_Rank",
    "Residual_60_N12_TTL90_Renew30_Residual",
)
TTL_COMPOSITE_FORENSICS_CORE_VARIANTS=TTL_COMPOSITE_FORENSICS_DEFAULT_VARIANTS[:4]
TTL_FORENSICS_REQUIRED_FILES=("variant_summary.csv","cost_adjusted_summary.csv","annual_returns.csv","monthly_returns.csv","drawdown_series.csv","turnover.csv","trade_log.csv","holding_periods.csv","renewal_decisions.csv","ttl_event_log.csv","audit_metadata.json")
TTL_FORENSICS_IMPORTANT_FILES=("variant_summary.csv","cost_adjusted_summary.csv","trade_log.csv","renewal_decisions.csv","holding_periods.csv")
TTL_FORENSICS_OUTPUT_FILES=("forensics_summary.csv","forensics_summary.json","candidate_comparison.csv","active_exposure_daily.csv","active_exposure_summary.csv","annual_return_selected.csv","monthly_return_selected.csv","stress_year_2022.csv","drawdown_episodes.csv","renewal_condition_summary.csv","renewal_decision_by_year.csv","renewal_decision_by_market.csv","holding_period_summary.csv","trade_activity_summary.csv","ticker_contribution_summary.csv","cash_drag_proxy.csv","future_data_boundary_review.csv","complexity_scorecard.csv","ttl_composite_forensics_report.md","audit_metadata.json")

def _read_artifact_csv(source,name,index_col=None):
    path=Path(source)/name
    if not path.exists(): return pd.DataFrame()
    try: return pd.read_csv(path,index_col=index_col)
    except Exception as exc: LOG.warning("failed to read %s: %s",path,exc); return pd.DataFrame()

def _variant_col(df):
    for c in ("Variant","variant","index"):
        if c in df.columns: return c
    return None

def _filter_variants(df,variants):
    if df is None or df.empty: return pd.DataFrame()
    out=df.copy(); vc=_variant_col(out)
    if vc: return out[out[vc].astype(str).isin(variants)].copy()
    out.index=out.index.astype(str); return out.loc[out.index.intersection(variants)].copy()

def _num(s): return pd.to_numeric(s,errors="coerce")

def _load_ttl_forensics_artifacts(source_dir):
    LOG.info("artifact loading start: %s",source_dir)
    source=Path(source_dir); artifacts={}; missing=[]
    for name in TTL_FORENSICS_REQUIRED_FILES:
        path=source/name
        if not path.exists(): missing.append(name); artifacts[name]=pd.DataFrame() if name.endswith('.csv') else {}
        elif name.endswith('.json'):
            try: artifacts[name]=json.loads(path.read_text(encoding='utf-8'))
            except Exception as exc: LOG.warning("failed to read %s: %s",path,exc); artifacts[name]={}
        else:
            idx=0 if name in ("variant_summary.csv","cost_adjusted_summary.csv","annual_returns.csv","monthly_returns.csv","drawdown_series.csv") else None
            artifacts[name]=_read_artifact_csv(source,name,index_col=idx)
    optional=[]
    for name in ("ttl_renewal_report.md","data_quality.csv","insufficient_history_summary.csv"):
        if (source/name).exists(): optional.append(name)
    important_missing=[m for m in missing if m in TTL_FORENSICS_IMPORTANT_FILES]
    if important_missing: LOG.warning("important forensics source files missing: %s", important_missing)
    LOG.info("artifact loading end")
    return artifacts,missing,important_missing,optional

def _selected_variants_arg(variants):
    return [v.strip() for v in variants.split(',') if v.strip()] if variants else list(TTL_COMPOSITE_FORENSICS_DEFAULT_VARIANTS)

def _write_df(df,path,index=False):
    if df is None: df=pd.DataFrame()
    df.to_csv(path,index=index)

def _active_exposure(trade_log, variants, start=None, end=None):
    if trade_log.empty or not {"variant","date","action","weight"}.issubset(trade_log.columns):
        return pd.DataFrame(), pd.DataFrame([{"variant":v,"average_active_exposure":np.nan,"exposure_judgment":"unavailable"} for v in variants])
    tl=trade_log[trade_log.variant.astype(str).isin(variants)].copy(); tl["date"]=pd.to_datetime(tl["date"],errors="coerce"); tl["weight"]=_num(tl["weight"]).fillna(0); tl=tl.dropna(subset=["date"])
    if tl.empty: return pd.DataFrame(), pd.DataFrame()
    start=pd.Timestamp(start) if start else tl.date.min(); end=pd.Timestamp(end) if end else tl.date.max(); dates=pd.bdate_range(start,end); rows=[]
    for v,g in tl.groupby("variant"):
        active={}
        for d in dates:
            day=g[g.date==d]
            for _,r in day.iterrows():
                if str(r.action).upper()=="BUY": active[str(r.ticker)]=float(r.weight)
                elif str(r.action).upper()=="SELL": active.pop(str(r.ticker),None)
            aw=float(sum(active.values())); rows.append({"date":d,"variant":v,"active_weight":aw,"active_position_count":len(active),"cash_weight":max(0,1-aw)})
    daily=pd.DataFrame(rows)
    summ=[]
    for v,g in daily.groupby("variant"):
        avg=float(g.active_weight.mean())
        judgment="ほぼフル投資" if avg>=.95 else "適度な免疫系" if avg>=.80 else "キャッシュ寄与が大きい。慎重評価" if avg>=.60 else "ほぼ別戦略。採用注意"
        summ.append({"variant":v,"full_invested_days_ratio":float((g.active_weight>=.99).mean()),"below_90pct_exposure_days_ratio":float((g.active_weight<.90).mean()),"below_80pct_exposure_days_ratio":float((g.active_weight<.80).mean()),"below_70pct_exposure_days_ratio":float((g.active_weight<.70).mean()),"average_active_exposure":avg,"median_active_exposure":float(g.active_weight.median()),"min_active_exposure":float(g.active_weight.min()),"average_active_position_count":float(g.active_position_count.mean()),"average_cash_weight":float(g.cash_weight.mean()),"exposure_judgment":judgment})
    return daily,pd.DataFrame(summ)

def _renewal_summaries(rd, variants):
    if rd.empty or "variant" not in rd: return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    d=rd[rd.variant.astype(str).isin(variants)].copy()
    for c in ("renewed","rank_pass","residual_pass","price_above_50dma","composite_pass"):
        if c in d: d[c]=d[c].astype(str).str.lower().isin(["true","1","yes"])
    if "health_check_date" in d: d["year"]=pd.to_datetime(d.health_check_date,errors="coerce").dt.year
    agg=[]
    for v,g in d.groupby("variant"):
        agg.append({"variant":v,"decisions":len(g),"renewed_count":int(g.get("renewed",pd.Series(dtype=bool)).sum()),"renewal_rate":float(g.get("renewed",pd.Series(dtype=bool)).mean()) if len(g) and "renewed" in g else np.nan,"rank_pass_rate":float(g.rank_pass.mean()) if "rank_pass" in g else np.nan,"residual_pass_rate":float(g.residual_pass.mean()) if "residual_pass" in g else np.nan,"price_above_50dma_pass_rate":float(g.price_above_50dma.mean()) if "price_above_50dma" in g else np.nan,"composite_pass_rate":float(g.composite_pass.mean()) if "composite_pass" in g else np.nan,"average_composite_count":float(_num(g.composite_count).mean()) if "composite_count" in g else np.nan})
    by_year=d.groupby(["variant","year"],dropna=False).agg(decisions=("variant","size"),renewal_rate=("renewed","mean"),composite_count=("composite_count","mean"),rank_pass_rate=("rank_pass","mean"),residual_pass_rate=("residual_pass","mean"),price_pass_rate=("price_above_50dma","mean")).reset_index() if "year" in d and "renewed" in d else pd.DataFrame()
    market_col="Region" if "Region" in d else "region" if "region" in d else None
    by_market=d.groupby(["variant",market_col],dropna=False).agg(decisions=("variant","size"),renewal_rate=("renewed","mean"),rank_pass_rate=("rank_pass","mean"),residual_pass_rate=("residual_pass","mean"),price_pass_rate=("price_above_50dma","mean"),rejected_count=("renewed",lambda x:int((~x).sum()))).reset_index().rename(columns={market_col:"market"}) if market_col and "renewed" in d else pd.DataFrame()
    return pd.DataFrame(agg),by_year,by_market

def _holding_summary(hp, variants):
    if hp.empty or "variant" not in hp or "holding_days" not in hp: return pd.DataFrame()
    d=hp[hp.variant.astype(str).isin(variants)].copy(); d["holding_days"]=_num(d.holding_days)
    rows=[]
    for v,g in d.groupby("variant"):
        h=g.holding_days.dropna(); rows.append({"variant":v,"count":len(h),"mean_holding_days":float(h.mean()) if len(h) else np.nan,"median_holding_days":float(h.median()) if len(h) else np.nan,"max_holding_days":float(h.max()) if len(h) else np.nan,"min_holding_days":float(h.min()) if len(h) else np.nan,"percent_held_around_90_days":float(((h>=85)&(h<=95)).mean()) if len(h) else np.nan,"percent_extended_beyond_90_days":float((h>90).mean()) if len(h) else np.nan,"percent_reached_max_120_days":float((h>=120).mean()) if len(h) else np.nan,"bucket_le_90":int((h<=90).sum()),"bucket_91_105":int(((h>90)&(h<=105)).sum()),"bucket_106_120":int(((h>105)&(h<=120)).sum()),"bucket_gt_120":int((h>120).sum())})
    return pd.DataFrame(rows)

def _stress_2022(monthly, draw, exposure, renewal, variants):
    rows=[]
    for v in variants:
        m=_num(monthly[v]) if v in monthly else pd.Series(dtype=float)
        m.index=pd.to_datetime(m.index,errors="coerce")
        m22=m[m.index.year==2022].dropna()
        d=_num(draw[v]) if v in draw else pd.Series(dtype=float)
        d.index=pd.to_datetime(d.index,errors="coerce")
        d22=d[d.index.year==2022].dropna()
        ex_df=pd.DataFrame()
        if not exposure.empty and "date" in exposure and "variant" in exposure:
            ex_dates=pd.to_datetime(exposure.date,errors="coerce")
            ex_df=exposure[(exposure.variant==v)&(ex_dates.dt.year==2022)]
        rd_df=pd.DataFrame()
        if not renewal.empty and "health_check_date" in renewal and "variant" in renewal:
            rd_dates=pd.to_datetime(renewal["health_check_date"],errors="coerce")
            rd_df=renewal[(renewal.variant.astype(str)==v)&(rd_dates.dt.year==2022)]
        renewed_bool=rd_df.renewed.astype(str).str.lower().isin(["true","1","yes"]) if not rd_df.empty and "renewed" in rd_df else pd.Series(dtype=bool)
        rows.append({"variant":v,"return_2022":float((1+m22).prod()-1) if len(m22) else np.nan,"max_drawdown_2022":float(d22.min()) if len(d22) else np.nan,"worst_month_2022":float(m22.min()) if len(m22) else np.nan,"monthly_win_rate_2022":float((m22>0).mean()) if len(m22) else np.nan,"average_active_exposure_2022":float(ex_df.active_weight.mean()) if not ex_df.empty and "active_weight" in ex_df else np.nan,"renewal_pass_rate_2022":float(renewed_bool.mean()) if len(renewed_bool) else np.nan,"cash_weight_proxy_2022":float(1-ex_df.active_weight.mean()) if not ex_df.empty and "active_weight" in ex_df else np.nan,"forced_non_renewals_2022":int((~renewed_bool).sum()) if len(renewed_bool) else 0})
    out=pd.DataFrame(rows)
    pairs={"Residual_60_N12_TTL90_Renew30_Composite":"Residual_60_N12_TTL90","Residual_65_N12_TTL90_Renew30_Composite":"Residual_65_N12_TTL90"}
    for comp,base in pairs.items():
        if comp in out.variant.values and base in out.variant.values:
            b=out.set_index("variant").loc[base]; idx=out.variant==comp; out.loc[idx,"return_2022_vs_fixed_delta"]=out.loc[idx,"return_2022"].values-b.return_2022; out.loc[idx,"maxdd_2022_vs_fixed_delta"]=out.loc[idx,"max_drawdown_2022"].values-b.max_drawdown_2022
    return out

def _drawdown_episodes(draw, exposure, renewal, variants):
    rows=[]
    for v in variants:
        if v not in draw: continue
        s=_num(draw[v]); s.index=pd.to_datetime(s.index,errors="coerce")
        in_ep=False; start=None; trough=None; depth=0
        for dt,val in s.dropna().items():
            if val<0 and not in_ep:
                in_ep=True; start=dt; trough=dt; depth=val
            if in_ep and val<depth:
                depth=val; trough=dt
            if in_ep and val>=0:
                ex_df=pd.DataFrame()
                if not exposure.empty and {"variant","date"}.issubset(exposure.columns):
                    ex_dates=pd.to_datetime(exposure.date,errors="coerce")
                    ex_df=exposure[(exposure.variant==v)&(ex_dates>=start)&(ex_dates<=dt)]
                rd_df=pd.DataFrame()
                if not renewal.empty and "health_check_date" in renewal and "variant" in renewal:
                    rd_dates=pd.to_datetime(renewal["health_check_date"],errors="coerce")
                    rd_df=renewal[(renewal.variant.astype(str)==v)&(rd_dates.between(start,dt))]
                rows.append({"variant":v,"drawdown_start_date":start,"drawdown_trough_date":trough,"recovery_date":dt,"max_drawdown_depth":depth,"days_to_trough":(trough-start).days,"days_to_recovery":(dt-start).days,"return_during_drawdown":depth,"active_exposure_during_drawdown":float(ex_df.active_weight.mean()) if not ex_df.empty and "active_weight" in ex_df else np.nan,"renewal_decisions_around_drawdown":len(rd_df)})
                in_ep=False
        if in_ep:
            rows.append({"variant":v,"drawdown_start_date":start,"drawdown_trough_date":trough,"recovery_date":pd.NaT,"max_drawdown_depth":depth,"days_to_trough":(trough-start).days,"days_to_recovery":np.nan,"return_during_drawdown":depth,"active_exposure_during_drawdown":np.nan,"renewal_decisions_around_drawdown":np.nan})
    out=pd.DataFrame(rows)
    return out.sort_values(["variant","max_drawdown_depth"]).groupby("variant").head(5).reset_index(drop=True) if not out.empty else out

def _ticker_contrib(trade_log, variants):
    if trade_log.empty or not {"variant","ticker","action"}.issubset(trade_log.columns): return pd.DataFrame()
    d=trade_log[trade_log.variant.astype(str).isin(variants)].copy(); d["weight"]=_num(d.get("weight",0)).fillna(0)
    rows=[]
    for (v,t),g in d.groupby(["variant","ticker"]):
        buys=int((g.action.astype(str).str.upper()=="BUY").sum()); sells=int((g.action.astype(str).str.upper()=="SELL").sum())
        rows.append({"variant":v,"ticker":t,"approx_trade_count":len(g),"buy_count":buys,"sell_count":sells,"approx_weight_used":float(g.weight.abs().sum()),"market":"JP" if str(t).endswith(".T") else "US","approximation_note":"Trade-log activity proxy; exact PnL contribution requires position-level returns."})
    out=pd.DataFrame(rows)
    if out.empty: return out
    totals=out.groupby("variant").approx_weight_used.transform("sum").replace(0,np.nan); out["approx_contribution_share"]=out.approx_weight_used/totals
    return out.sort_values(["variant","approx_contribution_share"],ascending=[True,False])

def _max_observed_date(df, columns=None, include_index=True):
    dates=[]
    if isinstance(df,pd.DataFrame) and not df.empty:
        if include_index and not pd.api.types.is_integer_dtype(df.index):
            idx=pd.to_datetime(df.index,errors="coerce")
            if idx.notna().any(): dates.append(idx.max())
        cols=columns if columns is not None else [c for c in df.columns if "date" in str(c).lower()]
        for c in cols:
            if c in df:
                s=pd.to_datetime(df[c],errors="coerce")
                if s.notna().any(): dates.append(s.max())
    return max(dates) if dates else pd.NaT

def _future_boundary_review(rd, artifacts=None):
    rows=[{"check":"health_check_date near trade_date + 90 days","status":"not_available","details":"renewal_decisions.csv unavailable"},{"check":"renewal inputs as-of health_check_date","status":"implementation_review","details":"_health_row uses asof_prices(prices, date) and score functions bounded to date."},{"check":"50DMA uses prior/as-of prices only","status":"implementation_review","details":"_health_row builds s=asof_prices(prices,date).ffill()[ticker] before tail(50)."},{"check":"rank/residual as-of boundary","status":"implementation_review","details":"_select_ttl_candidates and compute_residual_momentum_score receive health_check date."},{"check":"weekly degradation boundary","status":"implementation_review","details":"run_ttl_renewal_variant passes each weekly check date into _health_row."}]
    if not rd.empty and {"trade_date","health_check_date"}.issubset(rd.columns):
        d=rd.copy(); trade=pd.to_datetime(d.trade_date,errors="coerce"); health=pd.to_datetime(d.health_check_date,errors="coerce")
        diff=(health-trade).dt.days; valid=diff.dropna(); status="pass" if valid.between(85,125).all() else "warning"
        details=f"observed_min_days={diff.min()}, observed_max_days={diff.max()}, rows={len(valid)}"
        suspicious=d[diff<80].copy(); suspicious_health=health[diff<80]; suspicious_trade=trade[diff<80]
        if status=="warning" and not suspicious.empty:
            artifacts=artifacts or {}; end_candidates=[]
            for name,df in artifacts.items():
                if name=="renewal_decisions.csv": continue
                dt=_max_observed_date(df) if isinstance(df,pd.DataFrame) else pd.NaT
                if pd.notna(dt): end_candidates.append(dt)
            source_end=max(end_candidates) if end_candidates else health.max()
            audit_end=source_end
            return_end=max([dt for dt in (_max_observed_date(artifacts.get("monthly_returns.csv",pd.DataFrame()),include_index=True),_max_observed_date(artifacts.get("annual_returns.csv",pd.DataFrame()),include_index=True),_max_observed_date(artifacts.get("drawdown_series.csv",pd.DataFrame()),include_index=True)) if pd.notna(dt)], default=pd.NaT)
            screen=pd.to_datetime(suspicious.get("screen_date",pd.Series(index=suspicious.index,dtype="datetime64[ns]")),errors="coerce")
            latest_suspicious=max([x for x in (screen.max(),suspicious_trade.max(),suspicious_health.max()) if pd.notna(x)], default=pd.NaT)
            suspicious_near_end=pd.notna(source_end) and pd.notna(latest_suspicious) and (source_end-latest_suspicious).days<=3 and (source_end-suspicious_trade.min()).days<=100
            health_matches_end=pd.notna(source_end) and suspicious_health.notna().all() and ((source_end-suspicious_health).dt.days.abs()<=3).all()
            no_return_period=pd.isna(return_end) or return_end<=suspicious_health.max()
            if len(suspicious)==int((diff<80).sum()) and suspicious_near_end and health_matches_end and no_return_period:
                status="end_of_sample_partial_cycle"
                details += f", suspicious_rows={len(suspicious)}, source_data_end_date={source_end.date()}, audit_end_date={audit_end.date()}, suspicious_trade_min={suspicious_trade.min().date()}, suspicious_health_max={suspicious_health.max().date()}, no_return_period_after_health_check={no_return_period}; classified as end-of-sample truncation, not future leakage"
            else:
                details += f", suspicious_rows={len(suspicious)}, source_data_end_date={source_end.date() if pd.notna(source_end) else 'not_available'}, audit_end_date={audit_end.date() if pd.notna(audit_end) else 'not_available'}, no_return_period_after_health_check={no_return_period}"
        rows[0]={"check":"health_check_date near trade_date + 90 days","status":status,"details":details}
    return pd.DataFrame(rows)

def _complexity_scorecard(summary,cost,exposure,variants):
    base_pairs=[("Residual_60_N12_TTL90","Residual_60_N12_TTL90_Renew30_Composite"),("Residual_65_N12_TTL90","Residual_65_N12_TTL90_Renew30_Composite")]
    s=summary.copy(); c=cost.copy(); ex=exposure.set_index("variant") if not exposure.empty and "variant" in exposure else pd.DataFrame(); rows=[]
    for base,comp in base_pairs:
        if comp not in variants: continue
        def val(df,row,col): return float(df.loc[row,col]) if row in df.index and col in df.columns and pd.notna(df.loc[row,col]) else np.nan
        net_cagr=(val(c,comp,"Tax_Slippage_Adjusted_CAGR") or val(s,comp,"Tax_Slippage_Adjusted_CAGR"))-(val(c,base,"Tax_Slippage_Adjusted_CAGR") or val(s,base,"Tax_Slippage_Adjusted_CAGR"))
        net_sharpe=(val(c,comp,"Net_Sharpe") or val(s,comp,"Net_Sharpe"))-(val(c,base,"Net_Sharpe") or val(s,base,"Net_Sharpe"))
        net_calmar=(val(c,comp,"Net_Calmar") or val(s,comp,"Net_Calmar"))-(val(c,base,"Net_Calmar") or val(s,base,"Net_Calmar"))
        maxdd_reduction=val(s,comp,"Max_Drawdown")-val(s,base,"Max_Drawdown")
        turnover_change=val(s,comp,"Turnover")-val(s,base,"Turnover")
        active_drop=(val(ex,base,"average_active_exposure")-val(ex,comp,"average_active_exposure")) if not ex.empty else np.nan
        decision="Adopt with caution" if (pd.isna(net_calmar) or net_calmar>0) and (pd.isna(active_drop) or active_drop<.25) else "Keep as research"
        rows.append({"comparison":f"{base} vs {comp}","base_variant":base,"composite_variant":comp,"net_cagr_improvement":net_cagr,"net_sharpe_improvement":net_sharpe,"net_calmar_improvement":net_calmar,"maxdd_reduction":maxdd_reduction,"turnover_change":turnover_change,"active_exposure_drop":active_drop,"rule_complexity":"high","operational_burden":"medium","interpretability":"medium","failure_risk":"medium","final_judgment":decision})
    return pd.DataFrame(rows)

def _plain_table(df):
    return df.to_string(index=False) if isinstance(df,pd.DataFrame) and not df.empty else "No data available."

R100_DEFAULT_VARIANTS=(
    "Residual_100_N12_TTL90","Residual_100_N10_TTL90","Residual_100_N8_TTL90","Residual_100_N6_TTL90",
    "Residual_100_N12_TTL90_Renew30_Composite","Residual_100_N10_TTL90_Renew30_Composite","Residual_100_N8_TTL90_Renew30_Composite","Residual_100_N6_TTL90_Renew30_Composite",
)
R100_REFERENCE_VARIANTS=("Residual_60_N12_TTL90_Renew30_Composite","Residual_65_N12_TTL90_Renew30_Composite","Residual_60_N12_TTL90","Residual_65_N12_TTL90","Baseline_N12_TTL90")

def build_r100_composite_variants(variants=None, only_variant=None):
    names=[only_variant] if only_variant else ([v.strip() for v in variants.split(',') if v.strip()] if variants else list(R100_DEFAULT_VARIANTS))
    out=[]
    for n in names:
        import re
        m=re.match(r"Residual_100_N(\d+)_TTL90(?:_Renew30_Composite)?$",n)
        if not m: raise ValueError(f"unsupported r100_composite_experiment variant: {n}")
        size=int(m.group(1)); usn=size//2; jpn=size-usn; comp="Renew30_Composite" in n
        out.append({"name":n,"selection_name":f"Residual_100_N{size}","base_weight":0.0,"residual_weight":1.0,"total_holdings":size,"us_holdings":usn,"jp_holdings":jpn,"ttl_days":90,"renewal_protocol":"composite" if comp else None,"renewal_extension_days":30,"is_baseline":False})
    return tuple(out)

def _r100_cache_candidates(cache_dir,out,source_dir):
    return [cache_dir, Path(out)/"cache", Path(source_dir)/"cache" if source_dir else None, "artifacts/ttl_renewal_quick/cache", "artifacts/residual_concentration/cache"]

def _summary_extras(summary, annual, monthly):
    s=summary.copy()
    if not annual.empty:
        s["Worst_Year"]=annual.min(); s["Best_Year"]=annual.max()
    if not monthly.empty:
        s["Worst_Month"]=monthly.min(); s["Monthly_Win_Rate"]=(monthly>0).mean()
    return s

def _concentration_risk(ticker, variants):
    if ticker.empty: return pd.DataFrame(), pd.DataFrame()
    rows=[]; split=[]
    for v,g in ticker.groupby("variant"):
        shares=g.approx_contribution_share.fillna(0).sort_values(ascending=False)
        largest=float(shares.iloc[0]) if len(shares) else np.nan
        rows.append({"variant":v,"unique_tickers_used":int(g.ticker.nunique()),"top_5_contribution_share_approx":float(shares.head(5).sum()),"top_10_contribution_share_approx":float(shares.head(10).sum()),"largest_single_ticker_activity_share":largest,"repeat_winner_count":int((shares>.05).sum()),"repeat_loser_count":0,"concentration_risk":"extreme" if largest>.35 else "high" if largest>.25 else "moderate" if largest>.15 else "controlled"})
        split.append({"variant":v,"US_activity_share":float(g.loc[g.market=='US','approx_weight_used'].sum()/g.approx_weight_used.sum()) if g.approx_weight_used.sum() else np.nan,"JP_activity_share":float(g.loc[g.market=='JP','approx_weight_used'].sum()/g.approx_weight_used.sum()) if g.approx_weight_used.sum() else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(split)

def _r100_overdrive(summary, cost, exposure, stress, conc):
    rows=[]; s=summary; c=cost if not cost.empty else summary; ex=exposure.set_index('variant') if not exposure.empty and 'variant' in exposure else pd.DataFrame(); st=stress.set_index('variant') if not stress.empty and 'variant' in stress else pd.DataFrame(); co=conc.set_index('variant') if not conc.empty and 'variant' in conc else pd.DataFrame()
    ref=[v for v in R100_REFERENCE_VARIANTS[:2] if v in c.index]
    ref_cagr=float(c.loc[ref,'Tax_Slippage_Adjusted_CAGR'].max()) if ref and 'Tax_Slippage_Adjusted_CAGR' in c else np.nan
    ref_dd=float(s.loc[ref,'Max_Drawdown'].min()) if ref and 'Max_Drawdown' in s else np.nan
    for v in [x for x in R100_DEFAULT_VARIANTS if x.endswith('Composite') and x in s.index]:
        net=float(c.loc[v,'Tax_Slippage_Adjusted_CAGR']) if v in c.index and 'Tax_Slippage_Adjusted_CAGR' in c else float(s.loc[v].get('CAGR',np.nan))
        dd=float(s.loc[v].get('Max_Drawdown',np.nan)); avgex=float(ex.loc[v].get('average_active_exposure',np.nan)) if v in ex.index else np.nan
        stressdd=float(st.loc[v].get('max_drawdown_2022',np.nan)) if v in st.index else np.nan; cr=str(co.loc[v].get('concentration_risk','unknown')) if v in co.index else 'unknown'
        if avgex<.60 or dd<-0.60 or stressdd<-0.45 or cr=='extreme': rec='Reject'
        elif (pd.notna(ref_cagr) and net>=ref_cagr and avgex>=.80 and dd>=ref_dd-.10 and cr in ('controlled','moderate')): rec='Standard Candidate'
        elif avgex>=.70 and dd>-0.50 and cr!='extreme': rec='Overdrive Candidate'
        else: rec='Research Only'
        rows.append({'variant':v,'recommendation':rec,'net_cagr':net,'max_drawdown':dd,'average_active_exposure':avgex,'max_drawdown_2022':stressdd,'concentration_risk':cr,'criteria_note':'Classified using net CAGR, MaxDD, exposure, 2022 stress, concentration, and complexity.'})
    return pd.DataFrame(rows)

def write_r100_experiment_report(out, meta, tables):
    out=Path(out); lines=["# R100 Composite Experiment Report",""]
    sections=["Executive Summary","Why R100 Composite experiment was run","Experimental scope and anti-overheat design","Candidate variants","Fixed R100 TTL90 vs R100 Composite","R100_N6 Ferrari test","R100_N8/N10/N12 comparison","Comparison against Residual_60/65 Composite standard candidates","Cost-adjusted results","Active exposure and cash-drag analysis","2022 stress-year analysis","Drawdown episode analysis","Renewal decision analysis","Holding period analysis","Concentration and ticker contribution risk","Future data boundary review","Overdrive recommendation","Safety notes"]
    for sec in sections:
        lines += [f"## {sec}"]
        if sec=="Candidate variants": lines.append(str(meta.get('selected_variants')))
        elif sec=="Experimental scope and anti-overheat design": lines.append("Default scope is exactly 8 R100 variants, cache-first, no full sweep, no full score components, no per-health-check detail logs, partial save per variant, and resume support.")
        elif sec=="Safety notes": lines.append("This is research/backtest output, not investment advice. R100 concentrated variants may be psychologically difficult to hold. High CAGR does not automatically imply standard adoption. R100_N6 should be treated as Overdrive / Satellite unless risk metrics are unexpectedly robust. Default L.U.M.U.S.-8 standard candidate remains Residual_60_N12_TTL90_Renew30_Composite unless clearly proven otherwise.")
        else: lines.append(_plain_table(tables.get(sec,pd.DataFrame())))
        lines.append("")
    (out/"r100_experiment_report.md").write_text("\n".join(lines),encoding="utf-8"); Path("reports").mkdir(exist_ok=True); Path("reports/r100_experiment_report.md").write_text("\n".join(lines),encoding="utf-8")



def _r100_read_partial_summary(path, selected):
    if not Path(path).exists():
        return pd.DataFrame()
    df=pd.read_csv(path)
    if 'Variant' not in df.columns:
        return pd.DataFrame()
    df=df.drop_duplicates('Variant',keep='last')
    return df[df['Variant'].astype(str).isin(selected)].set_index('Variant')

def _r100_read_existing_output(path, selected, variant_col='variant', index_col=None):
    if not Path(path).exists():
        return pd.DataFrame()
    try:
        df=pd.read_csv(path,index_col=index_col)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if variant_col in df.columns:
        return df[df[variant_col].astype(str).isin(selected)].copy()
    if index_col is not None:
        return df[df.index.astype(str).isin(selected)].copy()
    return df

def _r100_merge_variant_rows(old, new, selected, variant_col='variant'):
    frames=[x for x in (old,new) if x is not None and not x.empty]
    if not frames:
        return pd.DataFrame()
    out=pd.concat(frames,ignore_index=True,sort=False)
    if variant_col in out.columns:
        out=out[out[variant_col].astype(str).isin(selected)]
        order={v:i for i,v in enumerate(selected)}
        out['_r100_order']=out[variant_col].map(order)
        out=out.drop_duplicates(variant_col,keep='last').sort_values('_r100_order').drop(columns='_r100_order')
    return out

def run_r100_composite_experiment_audit(prices=None,us=None,jp=None,start="2015-01-01",end=None,output_dir="artifacts/r100_composite_experiment",source_dir="artifacts/ttl_composite_forensics",cache_dir=None,resume=False,variants=None,only_variant=None,quick=False,force_refresh_cache=False,tax_rate=0.20315,slippage_bps=10,no_detail_logs=True):
    t0=time.time(); end=end or pd.Timestamp.today().date().isoformat(); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); (out/"cache").mkdir(exist_ok=True); Path("reports").mkdir(exist_ok=True); reset_insufficient_history_warnings(); LOG.info("audit start: r100_composite_experiment")
    if us is None or jp is None: us,jp=get_live_universe()
    us=[normalize_yfinance_ticker(t) for t in us]; jp=[normalize_yfinance_ticker(t) for t in jp]; modes=build_benchmark_modes(); default_mode=modes["broad_default"]; bench_tickers=sorted({x for mode in modes.values() for vals in mode.values() for x in vals}); requested=list(dict.fromkeys([*us,*jp])); vars_cfg=build_r100_composite_variants(variants,only_variant); selected=[v['name'] for v in vars_cfg]; LOG.info("selected variants: %s",selected)
    failures=pd.DataFrame(columns=["ticker","reason"]); cache_meta={"cache_used":prices is not None,"cache_source":"provided_prices" if prices is not None else ""}
    if prices is None:
        LOG.info("cache loading start")
        prices,cache_meta=_load_ttl_cache(_r100_cache_candidates(cache_dir,out,source_dir),start,end,requested,bench_tickers)
        if prices is None and not force_refresh_cache: raise SystemExit("r100_composite_experiment cache miss: default download is disabled; pass --force-refresh-cache to download")
        if prices is None: prices,failures,requested,cache_meta=build_price_cache(requested,bench_tickers,start,end,out/"cache",downloader=None)
        LOG.info("cache loading end: %s",cache_meta.get("cache_source"))
    dq,_,_,usable=build_live_data_quality_report(prices,requested,us,jp,failures,start,end); us_usable=[t for t in us if t in usable]; jp_usable=[t for t in jp if t in usable]
    completed=set(); cp=out/"completed_variants.csv"; partial=out/"variant_summary_partial.csv"; costpartial=out/"cost_adjusted_summary_partial.csv"
    if resume and cp.exists(): completed=set(pd.read_csv(cp).variant.astype(str));
    completed={v for v in completed if v in selected}
    partial_summary=_r100_read_partial_summary(partial,selected) if resume else pd.DataFrame()
    partial_cost=_r100_read_partial_summary(costpartial,selected) if resume else pd.DataFrame()
    if resume and completed:
        valid_completed={v for v in completed if v in partial_summary.index and v in partial_cost.index}
        missing=sorted(completed-valid_completed)
        if missing: LOG.warning("resume partial summaries missing completed variants; recomputing: %s",missing)
        completed=valid_completed
    existing_exposure_summary=_r100_read_existing_output(out/"r100_active_exposure_summary.csv",selected) if resume else pd.DataFrame()
    existing_stress=_r100_read_existing_output(out/"r100_stress_year_2022.csv",selected) if resume else pd.DataFrame()
    existing_conc=_r100_read_existing_output(out/"r100_concentration_risk_summary.csv",selected) if resume else pd.DataFrame()
    returns={}; selected_rows=[]; turns=[]; trades=[]; holds=[]; decisions=[]; events=[]; partial_rows=partial_summary.reset_index().to_dict('records') if not partial_summary.empty else []; cost_rows=partial_cost.reset_index().to_dict('records') if not partial_cost.empty else []
    for v in vars_cfg:
        if resume and v['name'] in completed: LOG.info("variant skip completed: %s",v['name']); continue
        LOG.info("variant start: %s",v['name'])
        r,sel,sc,tu,tr,hp,rd,ev=run_ttl_renewal_variant(prices,us_usable,jp_usable,start,end,v,default_mode,detail_logs=not no_detail_logs)
        returns[v['name']]=r; selected_rows.append(sel); turns.append(tu); trades.append(tr); holds.append(hp); decisions.append(rd); events.append(ev)
        summ=metrics(r); summ.update({'Variant':v['name'],'Turnover':float(tu.turnover.mean()) if not tu.empty else np.nan}); partial_rows.append(summ)
        nr,slip,tax=_cost_adjusted_returns(r,tu,tax_rate,slippage_bps); nm=metrics(nr); crow={'Variant':v['name'],'Slippage_Adjusted_CAGR':cagr(_cost_adjusted_returns(r,tu,0,slippage_bps)[0]),'Tax_Adjusted_CAGR':cagr(_cost_adjusted_returns(r,tu,tax_rate,0)[0]),'Tax_Slippage_Adjusted_CAGR':nm['CAGR'],'Estimated_Tax_Drag':summ['CAGR']-nm['CAGR'],'Estimated_Slippage_Drag':slip,'Net_Sharpe':nm['Sharpe'],'Net_Calmar':nm['Calmar']}; cost_rows.append(crow)
        pd.DataFrame(partial_rows).to_csv(partial,index=False); pd.DataFrame(cost_rows).to_csv(costpartial,index=False); pd.DataFrame([{'variant':v['name'],'completed_at':pd.Timestamp.now('UTC').isoformat()}]).to_csv(cp,mode='a',header=not cp.exists(),index=False); LOG.info("partial output writing: %s",v['name']); LOG.info("variant end: %s",v['name'])
    
    turnover=pd.concat(turns,ignore_index=True) if turns else pd.DataFrame(); trade_log=pd.concat(trades,ignore_index=True) if trades else pd.DataFrame(); holding_periods=pd.concat(holds,ignore_index=True) if holds else pd.DataFrame(); renewal_decisions=pd.concat(decisions,ignore_index=True) if decisions else pd.DataFrame(); ttl_event_log=pd.concat(events,ignore_index=True) if events else pd.DataFrame()
    summary_new=pd.DataFrame({k:metrics(v) for k,v in returns.items()}).T; summary_new.index.name='Variant'
    if not turnover.empty and not summary_new.empty: summary_new['Turnover']=turnover.groupby('variant').turnover.mean()
    annual=pd.DataFrame({k:(1+v).resample('YE').prod()-1 for k,v in returns.items()}); monthly=pd.DataFrame({k:(1+v).resample('ME').prod()-1 for k,v in returns.items()}); draw=pd.DataFrame({k:((1+v).cumprod()/(1+v).cumprod().cummax()-1) for k,v in returns.items()}); cost=pd.DataFrame(cost_rows).drop_duplicates('Variant',keep='last').set_index('Variant') if cost_rows else pd.DataFrame(); summary_new=_summary_extras(summary_new,annual,monthly) if not summary_new.empty else summary_new
    summary=pd.concat([partial_summary,summary_new],sort=False) if not partial_summary.empty else summary_new
    summary=summary[~summary.index.duplicated(keep='last')]; summary=summary.reindex([v for v in selected if v in summary.index])
    summary=summary.join(cost,how='left',rsuffix='_cost') if not cost.empty else summary
    pd.DataFrame(partial_rows).drop_duplicates('Variant',keep='last').to_csv(partial,index=False) if partial_rows else None; pd.DataFrame(cost_rows).drop_duplicates('Variant',keep='last').to_csv(costpartial,index=False) if cost_rows else None
    LOG.info("exposure analysis start"); exposure_daily,exposure_summary=_active_exposure(trade_log,selected,start,end); LOG.info("exposure analysis end")
    renewal_summary,renewal_year,_=_renewal_summaries(renewal_decisions,selected); holding_summary=_holding_summary(holding_periods,selected); LOG.info("stress analysis start"); stress_new=_stress_2022(monthly,draw,exposure_daily,renewal_decisions,selected); stress=_r100_merge_variant_rows(existing_stress,stress_new,selected); LOG.info("stress analysis end"); episodes=_drawdown_episodes(draw,exposure_daily,renewal_decisions,selected); ticker=_ticker_contrib(trade_log,selected); conc_new,split=_concentration_risk(ticker,selected); conc=_r100_merge_variant_rows(existing_conc,conc_new,selected); exposure_summary=_r100_merge_variant_rows(existing_exposure_summary,exposure_summary,selected); future=_future_boundary_review(renewal_decisions,{'monthly_returns.csv':monthly,'annual_returns.csv':annual,'drawdown_series.csv':draw,'renewal_decisions.csv':renewal_decisions}); complexity=pd.DataFrame([{'variant':v,'rule_complexity':'high' if 'Composite' in v else 'low','full_sweep':False,'score_components_full_output':False} for v in selected]); over=_r100_overdrive(summary,cost,exposure_summary,stress,conc)
    refs=[]
    if source_dir and Path(source_dir).exists():
        for name in ('forensics_summary.csv','candidate_comparison.csv','variant_summary.csv','cost_adjusted_summary.csv'):
            df=_read_artifact_csv(source_dir,name,index_col=0 if name in ('variant_summary.csv','cost_adjusted_summary.csv') else None); f=_filter_variants(df,list(R100_REFERENCE_VARIANTS));
            if not f.empty: refs.append(f.reset_index().rename(columns={'index':'variant'}))
    candidate=pd.concat([summary.reset_index().rename(columns={'Variant':'variant'}),*refs],ignore_index=True,sort=False) if refs else summary.reset_index().rename(columns={'Variant':'variant'})
    cash=exposure_summary[['variant','average_active_exposure','average_cash_weight','exposure_judgment']].copy() if not exposure_summary.empty else pd.DataFrame(); event_summary=ttl_event_log.groupby(['variant','event']).size().reset_index(name='count') if not ttl_event_log.empty else pd.DataFrame()
    meta={'audit_name':'r100_composite_experiment','selected_variants':selected,'reference_variants':list(R100_REFERENCE_VARIANTS),'default_download_allowed':False,'default_full_variant_recalculation_allowed':False,'score_components_full_output':False,'cache_first_candidates':[str(x) for x in _r100_cache_candidates(cache_dir,out,source_dir) if x],'cache_used':cache_meta.get('cache_used',False),'cache_source':cache_meta.get('cache_source',''),'resume':resume,'tax_rate':tax_rate,'slippage_bps':slippage_bps,'output_files':R100_OUTPUT_FILES,'wall_time_seconds':round(time.time()-t0,2)}
    LOG.info("report writing start")
    for name,df,idx in (("r100_variant_summary.csv",summary,True),("r100_cost_adjusted_summary.csv",cost,True),("r100_candidate_comparison.csv",candidate,False),("r100_complexity_scorecard.csv",complexity,False),("r100_overdrive_recommendation.csv",over,False),("r100_active_exposure_daily.csv",exposure_daily,False),("r100_active_exposure_summary.csv",exposure_summary,False),("r100_cash_drag_proxy.csv",cash,False),("r100_drawdown_episodes.csv",episodes,False),("r100_stress_year_2022.csv",stress,False),("r100_renewal_decisions.csv",renewal_decisions,False),("r100_renewal_condition_summary.csv",renewal_summary,False),("r100_renewal_decision_by_year.csv",renewal_year,False),("r100_holding_period_summary.csv",holding_summary,False),("r100_ticker_contribution_summary.csv",ticker,False),("r100_concentration_risk_summary.csv",conc,False),("r100_us_jp_split_summary.csv",split,False),("r100_event_log_summary.csv",event_summary,False),("future_data_boundary_review.csv",future,False)): _write_df(df,out/name,index=idx)
    summary.to_json(out/"r100_variant_summary.json",orient='index',indent=2); (out/"audit_metadata.json").write_text(json.dumps(meta,indent=2,default=str),encoding='utf-8')
    write_r100_experiment_report(out,meta,{"Fixed R100 TTL90 vs R100 Composite":candidate,"Cost-adjusted results":cost.reset_index() if not cost.empty else cost,"Active exposure and cash-drag analysis":exposure_summary,"2022 stress-year analysis":stress,"Drawdown episode analysis":episodes,"Renewal decision analysis":renewal_summary,"Holding period analysis":holding_summary,"Concentration and ticker contribution risk":conc,"Future data boundary review":future,"Overdrive recommendation":over})
    LOG.info("report writing end"); LOG.info("audit complete with wall time %.1fs",time.time()-t0); return summary

R100_OUTPUT_FILES=("r100_variant_summary.csv","r100_variant_summary.json","r100_cost_adjusted_summary.csv","r100_candidate_comparison.csv","r100_complexity_scorecard.csv","r100_overdrive_recommendation.csv","r100_experiment_report.md","audit_metadata.json","r100_active_exposure_daily.csv","r100_active_exposure_summary.csv","r100_cash_drag_proxy.csv","r100_drawdown_episodes.csv","r100_stress_year_2022.csv","r100_renewal_decisions.csv","r100_renewal_condition_summary.csv","r100_renewal_decision_by_year.csv","r100_holding_period_summary.csv","r100_ticker_contribution_summary.csv","r100_concentration_risk_summary.csv","r100_us_jp_split_summary.csv","completed_variants.csv","variant_summary_partial.csv","cost_adjusted_summary_partial.csv","r100_event_log_summary.csv","future_data_boundary_review.csv")


def write_ttl_composite_forensics_report(output_dir, meta, tables):
    out=Path(output_dir); comp=_plain_table(tables.get("candidate_comparison",pd.DataFrame())) if not tables.get("candidate_comparison",pd.DataFrame()).empty else "No comparison data available."
    exposure=_plain_table(tables.get("active_exposure_summary",pd.DataFrame())) if not tables.get("active_exposure_summary",pd.DataFrame()).empty else "No exposure data available."
    stress=_plain_table(tables.get("stress_year_2022",pd.DataFrame())) if not tables.get("stress_year_2022",pd.DataFrame()).empty else "No 2022 data available."
    complexity=_plain_table(tables.get("complexity_scorecard",pd.DataFrame())) if not tables.get("complexity_scorecard",pd.DataFrame()).empty else "No complexity data available."
    future=_plain_table(tables.get("future_data_boundary_review",pd.DataFrame())) if not tables.get("future_data_boundary_review",pd.DataFrame()).empty else "No future-data boundary data available."
    report=f"""# TTL Composite Forensics Report

## 1. Executive Summary
This candidate-only forensic audit reviews TTL90_Renew30_Composite candidates using existing artifacts only by default. It is not a full universe search.

## 2. Why this forensic audit was needed
Composite renewal previously showed large MaxDD, Sharpe, and Calmar improvements; this report checks cash exposure, stress-year behavior, renewal conditions, concentration, future-data boundaries, and complexity.

## 3. Data and artifact sources
Source directory: `{meta.get('source_dir')}`. Missing files: {meta.get('missing_files',[])}. Important missing files: {meta.get('important_missing_files',[])}.

## 4. Candidate variants
Selected variants: {meta.get('selected_variants',[])}.

## 5. Fixed TTL90 vs Composite Renewal
{comp}

## 6. Cost-adjusted comparison
Cost model is approximate and is not tax advice. See `candidate_comparison.csv` for gross/net CAGR, Sharpe, Calmar, turnover, and drag fields when available.

## 7. Active exposure analysis
Exposure is reconstructed from BUY/SELL events and weights in `trade_log.csv`; cash weight is `1 - active_weight`. Judgment bands: 95-100% nearly full invested, 80-95% moderate immune system, 60-80% large cash contribution, below 60% nearly a different strategy.

{exposure}

## 8. 2022 stress-year review
{stress}

## 9. Drawdown episode review
Drawdown episodes are extracted from `drawdown_series.csv` by identifying below-zero drawdown intervals and troughs, then joining average exposure and renewal-decision counts where available.

## 10. Renewal decision analysis
Renewal decisions are grouped overall, by year, and by market. Pass rates are calculated for rank, residual, 50DMA price, and composite conditions where source columns are available.

## 11. Holding period analysis
Holding periods are bucketed into <=90, 91-105, 106-120, and >120 days to check whether renewal behaves like simple TTL120.

## 12. Ticker contribution / concentration review
Ticker contribution is approximate when only trade logs are available; activity and weight-use concentration proxies are reported.

## 13. Future data boundary review
Implementation review checks that renewal health checks call as-of bounded scoring and 50DMA logic. If short `days_to_health_check` rows are concentrated in the final observed cycle, match the source/audit end date, and have no subsequent return period, they are classified as `end_of_sample_partial_cycle` end-of-sample truncation rather than future leakage. See `future_data_boundary_review.csv`.

{future}

## 14. Complexity scorecard
{complexity}

## 15. Recommendation
Keep `Residual_60_N12_TTL90` as the fixed-TTL control. Promote `Residual_60_N12_TTL90_Renew30_Composite` only if net metrics beat fixed TTL90 without a large active-exposure drop. Treat `Residual_65_N12_TTL90_Renew30_Composite` as the more aggressive candidate. If average exposure falls below 80%, classify the improvement as materially cash-driven and require caution. A further FULL audit is not required for this candidate-forensics purpose, but remains useful before production standardization.

## 16. Safety notes
This is research, not investment advice. Past performance does not guarantee future results. Free-data artifacts may contain survivorship bias, missing data, adjusted-price artifacts, and approximate cost assumptions.
"""
    (out/"ttl_composite_forensics_report.md").write_text(report,encoding="utf-8")
    Path("reports").mkdir(exist_ok=True); Path("reports/ttl_composite_forensics_report.md").write_text(report,encoding="utf-8")

def run_ttl_composite_forensics_audit(source_dir="artifacts/ttl_renewal_quick",output_dir="artifacts/ttl_composite_forensics",cache_dir=None,rerun_selected=False,variants=None,quick=False,force_refresh_cache=False):
    t0=time.time(); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); Path("reports").mkdir(exist_ok=True)
    LOG.info("audit start: ttl_composite_forensics")
    selected=_selected_variants_arg(variants); LOG.info("selected variants: %s", selected)
    if rerun_selected: LOG.warning("rerun-selected requested; candidate-only recomputation is not run by default in this lightweight artifact forensics path")
    artifacts,missing,important_missing,optional=_load_ttl_forensics_artifacts(source_dir)
    summary=_filter_variants(artifacts.get("variant_summary.csv",pd.DataFrame()),selected); cost=_filter_variants(artifacts.get("cost_adjusted_summary.csv",pd.DataFrame()),selected)
    annual=_filter_variants(artifacts.get("annual_returns.csv",pd.DataFrame()).T,selected).T if not artifacts.get("annual_returns.csv",pd.DataFrame()).empty else pd.DataFrame()
    monthly=_filter_variants(artifacts.get("monthly_returns.csv",pd.DataFrame()).T,selected).T if not artifacts.get("monthly_returns.csv",pd.DataFrame()).empty else pd.DataFrame()
    draw=_filter_variants(artifacts.get("drawdown_series.csv",pd.DataFrame()).T,selected).T if not artifacts.get("drawdown_series.csv",pd.DataFrame()).empty else pd.DataFrame()
    trade=_filter_variants(artifacts.get("trade_log.csv",pd.DataFrame()),selected); holds=_filter_variants(artifacts.get("holding_periods.csv",pd.DataFrame()),selected); rd=_filter_variants(artifacts.get("renewal_decisions.csv",pd.DataFrame()),selected); turnover=_filter_variants(artifacts.get("turnover.csv",pd.DataFrame()),selected)
    LOG.info("exposure analysis start"); exposure_daily,exposure_summary=_active_exposure(trade,[v for v in TTL_COMPOSITE_FORENSICS_CORE_VARIANTS if v in selected]); LOG.info("exposure analysis end")
    LOG.info("renewal analysis start"); renewal_summary,renewal_year,renewal_market=_renewal_summaries(rd,selected); LOG.info("renewal analysis end")
    holding_summary=_holding_summary(holds,selected)
    LOG.info("drawdown analysis start"); stress=_stress_2022(monthly,draw,exposure_daily,rd,selected); episodes=_drawdown_episodes(draw,exposure_daily,rd,selected); LOG.info("drawdown analysis end")
    ticker=_ticker_contrib(trade,selected)
    future=_future_boundary_review(rd,artifacts)
    complexity=_complexity_scorecard(summary,cost,exposure_summary,selected)
    cash=exposure_summary[["variant","average_active_exposure","average_cash_weight","exposure_judgment"]].copy() if not exposure_summary.empty and "average_cash_weight" in exposure_summary else pd.DataFrame()
    candidate=summary.copy();
    if not cost.empty: candidate=candidate.join(cost,rsuffix="_cost",how="left")
    if not exposure_summary.empty: candidate=candidate.join(exposure_summary.set_index("variant"),how="left")
    forensics=complexity.copy() if not complexity.empty else pd.DataFrame({"selected_variants":selected})
    LOG.info("report writing start")
    _write_df(forensics,out/"forensics_summary.csv"); (out/"forensics_summary.json").write_text(forensics.to_json(orient="records",indent=2),encoding="utf-8")
    for name,df,idx in (("candidate_comparison.csv",candidate,True),("active_exposure_daily.csv",exposure_daily,False),("active_exposure_summary.csv",exposure_summary,False),("annual_return_selected.csv",annual,True),("monthly_return_selected.csv",monthly,True),("stress_year_2022.csv",stress,False),("drawdown_episodes.csv",episodes,False),("renewal_condition_summary.csv",renewal_summary,False),("renewal_decision_by_year.csv",renewal_year,False),("renewal_decision_by_market.csv",renewal_market,False),("holding_period_summary.csv",holding_summary,False),("trade_activity_summary.csv",turnover,False),("ticker_contribution_summary.csv",ticker,False),("cash_drag_proxy.csv",cash,False),("future_data_boundary_review.csv",future,False),("complexity_scorecard.csv",complexity,False)): _write_df(df,out/name,index=idx)
    meta={"audit_name":"ttl_composite_forensics","source_dir":str(source_dir),"cache_dir":str(cache_dir) if cache_dir else "","selected_variants":selected,"default_download_allowed":False,"default_full_variant_recalculation_allowed":False,"rerun_selected_requested":bool(rerun_selected),"quick":bool(quick),"force_refresh_cache":bool(force_refresh_cache),"missing_files":missing,"important_missing_files":important_missing,"optional_files_found":optional,"output_files":list(TTL_FORENSICS_OUTPUT_FILES),"wall_time_seconds":round(time.time()-t0,2)}
    (out/"audit_metadata.json").write_text(json.dumps(meta,indent=2,default=str),encoding="utf-8")
    write_ttl_composite_forensics_report(out,meta,{"candidate_comparison":candidate.reset_index().rename(columns={"index":"variant"}),"active_exposure_summary":exposure_summary,"stress_year_2022":stress,"future_data_boundary_review":future,"complexity_scorecard":complexity})
    LOG.info("report writing end"); LOG.info("audit complete with wall time %.1fs",time.time()-t0)
    return forensics

def write_ttl_renewal_report(summary,data_quality,metadata,output_dir="artifacts/ttl_renewal"):
    base="Residual_60_N12_TTL90" if "Residual_60_N12_TTL90" in summary.index else (summary.index[0] if len(summary) else "")
    best=summary.Calmar.idxmax() if len(summary) and "Calmar" in summary else ""
    view=summary[[c for c in ["CAGR","Max_Drawdown","Calmar","Turnover","Tax_Slippage_Adjusted_CAGR"] if c in summary]].head(30)
    report=f"""# TTL Renewal Audit Report

## 1. Executive Summary
Best Calmar variant in this run: **{best}**. Comparison baseline for current residual standard: **{base}**. Recommendation buckets to review: Keep TTL90, Move to Fixed TTL120, Use TTL90 + Renewal30, Research only, Reject.

## 2. Audit Scope
Residual Momentum selection with US/JP balanced N12/N10, fixed TTL variants, and TTL90 Renewal30 variants only. No Exit Protocol, Regime Filter, VCP, Sector Residual, Downside penalty, or Correlation penalty.

## 3. Data Quality
{_markdown_table(data_quality.set_index('metric') if not data_quality.empty else data_quality)}

## 4. Cache Usage
cache_used={metadata.get('cache_used')}; cache_source={metadata.get('cache_source')}; cache_loaded_at={metadata.get('cache_loaded_at')}.

## 5. Fixed TTL Results
Review TTL30/60/90/120/180 rows in `variant_summary.csv` for CAGR, MaxDD, Calmar, holding days, and turnover.

## 6. TTL Renewal Protocol Results
Rank uses same-market rank <= market_slots*2. Residual uses residual_raw > 0. Composite requires at least two of rank buffer, residual_raw > 0, and price above 50DMA. Weekly degradation applies only during the extension window and is an extension cutoff, not a general exit.

## 7. Cost-adjusted Results
Tax/slippage are approximate relative-comparison estimates, not tax advice. See `cost_adjusted_summary.csv`.

## 8. Turnover / Tax / Slippage Review
Compare Annualized_Turnover, Trade_Count, Estimated_Tax_Drag, and Estimated_Slippage_Drag.

## 9. Holding Period Review
Renewal variants enforce no initial exit before 90 days and max holding days of 120 days.

## 10. 2022 Stress Year Review
Use `annual_returns.csv`, `monthly_returns.csv`, and `drawdown_series.csv` for 2022 if present.

## 11. Comparison vs Current TTL90
Primary row: Residual_60_N12_TTL90 versus Residual_60_N12_TTL120 and Residual_60_N12_TTL90_Renew30_*.

## 12. Recommendation
Keep TTL90 if it remains Calmar/MaxDD competitive. Move to Fixed TTL120 only if it clearly improves net CAGR/Calmar without larger drawdowns. Use TTL90 + Renewal30 only if net metrics and turnover improve enough to justify complexity. Otherwise Research only or Reject.

## 13. Safety Notes
Research only; no automatic trading. Free data may be incomplete and biased. Cost model is approximate and not tax advice.

## Summary Preview
{_markdown_table(view)}
"""
    (Path(output_dir)/"ttl_renewal_report.md").write_text(report,encoding="utf-8"); Path("reports/ttl_renewal_report.md").write_text(report,encoding="utf-8")


def write_residual_concentration_report(summary,data_quality,metadata,conc,bps,brr,sweet,div,output_dir="artifacts/residual_concentration"):
    Path("reports").mkdir(exist_ok=True)
    if "Baseline_N12" in summary.index and summary.CAGR.notna().any():
        base=summary.loc["Baseline_N12"]; best_name=metadata.get("best_calmar_variant") or summary.Calmar.idxmax(); best=summary.loc[best_name]
        headline=f"Best Calmar variant was **{best_name}**. Baseline_N12 CAGR {base.CAGR:.2%}, MaxDD {base.Max_Drawdown:.2%}, Calmar {base.Calmar:.2f}; {best_name} CAGR {best.CAGR:.2%}, MaxDD {best.Max_Drawdown:.2%}, Calmar {best.Calmar:.2f}."
    else: headline="No usable data was available; generated files are structural skipped outputs."
    view=summary[[c for c in ["CAGR","Annualized_Volatility","Max_Drawdown","Sharpe","Sortino","Calmar","Turnover","Judgment"] if c in summary.columns]]
    report=f"""# Residual Core Portfolio Concentration Audit Report

## 1. Executive Summary
{headline} This audit fixes Residual Momentum as the tested core signal and varies only residual ratio and balanced US/JP portfolio size. N30/N40 are high-diversification reference points, not practical production recommendations. If improvements are not robust across neighboring variants, production adoption should be deferred.

## 2. Baseline Reminder
TTL 90 days, quarterly/four trades per year, no Exit Protocol, no Regime Filter, no VCP Proxy, no Sector Residual, no Downside penalty, no Correlation penalty. Baseline_N12 is base_weight=1.0, residual_weight=0.0, US 6 / JP 6.

## CLI
`python alpha_engine_backtest.py --audit residual_concentration --output-dir artifacts/residual_concentration`

`python alpha_engine_backtest.py --demo --audit residual_concentration --output-dir artifacts/residual_concentration`

Quick Colab check: `python alpha_engine_backtest.py --audit residual_concentration --quick --output-dir artifacts/residual_concentration_quick`

## 3. Why Portfolio Size Matters
The current 12-stock structure is a practical balance, not a proof of optimality. This audit compares N4/N6 concentration limits, N8/N10 concentration candidates, current N12, N16/N20/N24 practical diversification, and N30/N40 high-diversification references to study Residual signal purity versus single-name accident risk.

## 4. Data Quality Summary
{_markdown_table(data_quality.set_index('metric') if isinstance(data_quality,pd.DataFrame) and not data_quality.empty else pd.DataFrame())}

Cache used: {metadata.get('cache_used')} / source: `{metadata.get('cache_source')}` / path: `{metadata.get('cache_path')}`. Survivorship and current-constituent bias remain when historical constituents are not reconstructed. Insufficient history tickers were aggregated into `insufficient_history_summary.csv` and metadata rather than logged repeatedly.

## 5. Variant Summary
{_markdown_table(view)}

## 6. Best by Portfolio Size
{_markdown_table(bps.set_index('total_holdings') if isinstance(bps,pd.DataFrame) and not bps.empty else pd.DataFrame())}

## 7. Best by Residual Ratio
{_markdown_table(brr.set_index('residual_variant') if isinstance(brr,pd.DataFrame) and not brr.empty else pd.DataFrame())}

## 8. Sweet Spot Analysis
{_markdown_table(sweet.head(30).set_index('variant') if isinstance(sweet,pd.DataFrame) and not sweet.empty else pd.DataFrame())}

## 9. Diversification Reference Review
{_markdown_table(div.head(40).set_index(['residual_variant','comparison']) if isinstance(div,pd.DataFrame) and not div.empty else pd.DataFrame())}

## 10. Concentration Risk Review
{_markdown_table(conc.set_index('variant') if isinstance(conc,pd.DataFrame) and not conc.empty else pd.DataFrame())}

## 11. Year / Period Review
Use `annual_returns.csv`, `monthly_returns.csv`, and `drawdown_series.csv` for 2020, 2022, 2023, 2024, 2025, and 2026 YTD where present. This is post-analysis only and does not introduce any regime rule.

## 12. Selection Difference Review
`selection_diff.csv` compares Baseline_N12 with Residual_60/65/100 concentration and diversification variants. Generation status: {metadata.get('selection_diff_status')}. `score_components.csv` records base_score, residual_score, final_score, stock/benchmark/residual returns, benchmark_used, selected_flag, and weights.

## 13. Risk Review
Prioritize MaxDD, Worst Year, Worst Month/monthly returns, drawdowns, turnover, average/max single-name weight, Herfindahl index, and 2022 behavior. N4/N6 are limit tests; N30/N40 test whether diversification dilutes Residual signal.

## 14. Recommendation
Conservative candidate: strongest N12/N16/N20 variant with improved Calmar and acceptable MaxDD. Balanced candidate: best stable neighborhood around Residual_55/60/65. Aggressive candidate: best N8/N10 only if MaxDD and single-name weights remain acceptable. High-diversification reference candidate: best N30/N40 by Calmar, treated as reference only. Use actual CSVs before production; do not adopt a single isolated best point.

## 15. Safety Notes
This is not investment advice. Historical yfinance/Wikipedia/free-data tests do not guarantee future returns. Free data can contain missing values, delays, adjusted-price issues, survivorship bias, and historical constituent bias. Alpha Engine is an alpha sleeve, not an all-asset portfolio.
"""
    Path("reports/residual_concentration_report.md").write_text(report,encoding="utf-8")


def write_outputs(out, strategies, selected, turnover, benchmarks=None):
    out=Path(out); out.mkdir(parents=True,exist_ok=True); selected.to_csv(out/"selected_tickers_by_period.csv",index=False); turnover.to_csv(out/"turnover_report.csv",index=False)
    allr=dict(strategies); allr.update(benchmarks or {}); summary=pd.DataFrame({k:metrics(v) for k,v in allr.items()}).T; summary.index.name="Strategy"; summary["Turnover"]=np.nan; summary.loc[list(strategies),"Turnover"]=turnover.turnover.mean() if not turnover.empty else 0; summary.to_csv(out/"backtest_summary.csv")
    annual=pd.DataFrame({k:(1+v).resample("YE").prod()-1 for k,v in allr.items()}); monthly=pd.DataFrame({k:(1+v).resample("ME").prod()-1 for k,v in allr.items()}); annual.to_csv(out/"annual_returns.csv"); monthly.to_csv(out/"monthly_returns.csv")
    pd.DataFrame({k:((1+v).cumprod()/(1+v).cumprod().cummax()-1) for k,v in allr.items()}).to_csv(out/"drawdown_report.csv")
    a=summary.loc["Alpha_Always"]; f=summary.loc["Alpha_Regime_Filter"]; verdict="研究枠継続。ただしAlpha_Alwaysは有望。"
    (out/"momentum_alpha_backtest_report.md").write_text(f"""# Momentum Alpha Backtest Audit v0.1\n\n## 監査目的\n独立ユニット候補、L.U.M.U.S.-8内の補助枠、研究枠、設計見直しを数値で仮判定する初期監査です。投資助言ではなく、自動売買等には接続しません。\n\n## 未来情報遮断\n各期末 `t` 以前だけでスコアを計算し、約定は次の取引日です。\n\n## 「風」ユニット仮説\n- Alpha_Always: CAGR {a.CAGR:.2%}, MaxDD {a.Max_Drawdown:.2%}, Sharpe {a.Sharpe:.2f}, Calmar {a.Calmar:.2f}\n- Alpha_Regime_Filter: CAGR {f.CAGR:.2%}, MaxDD {f.Max_Drawdown:.2%}, Sharpe {f.Sharpe:.2f}, Calmar {f.Calmar:.2f}\n\n## 暫定判定\n**{verdict}**\n\nAlpha Engine本体は、簡易live backtestにおいてベンチマークと比較検証する価値があり、Alpha_Alwaysは有望です。一方、現行のAlpha_Regime_FilterはAlpha_Alwaysに対してCAGR、MaxDD、Sharpe、Calmarを一貫して改善するとは限らず、「風」ユニットとしてのレジーム制御は未完成で再設計対象です。\n\nそのため、現時点では独立した「風」ユニット化は見送り、研究枠として継続します。設計見直し判定ではなく、L.U.M.U.S.-8 Core 85% + Alpha 15% の統合バックテスト、業種集中分析、税・スリッページ控除後検証、履歴ユニバース監査を実施した後に再判定します。\n\n## 4分類の仮判定\n| 分類 | 判定 | 理由 |\n| --- | --- | --- |\n| 独立した風ユニット | まだ不可 | Regime FilterがAlpha_Alwaysを改善していない |\n| L.U.M.U.S.-8内15%補助枠 | 可能性あり | Alpha_Alwaysは有望だがCore統合BT未実施 |\n| 研究枠 | 現時点の正式判定 | 生存者バイアス、業種集中、税コスト、履歴ユニバースが未確認 |\n| 設計見直し | 不要 | Alpha本体のCAGR/Sharpeは良好。ただしRegime Filterは再設計対象 |\n\n## 選定銘柄の性格診断\nUS/JP各上位6銘柄を逆ボラで配分します。業種データ未提供のため半導体・AI・ディフェンシブ集中の定量判定は次版課題です。\n\n## 重要な限界\n- 現在ユニバースによる生存者バイアス\n- 過去S&P500構成、日本株の上場廃止・銘柄変更を完全再現していない\n- yfinanceの価格品質と調整済み系列に依存\n- 厳密な税・スリッページ計算ではない\n- L.U.M.U.S.-8 Coreデータ未提供のため統合比較未実施\n- 投資助言ではない\n\n## ユーザー環境での再現手順\n`python -m pip install -r requirements.txt`、`python -m unittest -v`、`python alpha_engine_backtest.py --demo --start 2018-01-01 --end 2025-12-31 --output-dir artifacts/demo` の順に実行してください。\n\n## 最終仮判定\n**{verdict}** Alpha_Regime_Filterを再設計し、履歴ユニバース・業種・コスト監査後に再判定してください。\n""",encoding="utf-8")
    return summary,verdict

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--start",default="2015-01-01"); ap.add_argument("--end",default=pd.Timestamp.today().date().isoformat()); ap.add_argument("--rebalance",default="quarterly",choices=["quarterly"]); ap.add_argument("--output-dir",default="artifacts"); ap.add_argument("--demo",action="store_true"); ap.add_argument("--quick",action="store_true",help="Run quick mode for supported audits"); ap.add_argument("--cache-dir"); ap.add_argument("--source-dir",default="artifacts/ttl_renewal_quick"); ap.add_argument("--rerun-selected",action="store_true"); ap.add_argument("--variants"); ap.add_argument("--force-refresh-cache",action="store_true"); ap.add_argument("--resume",action="store_true"); ap.add_argument("--tax-rate",type=float,default=0.20315); ap.add_argument("--slippage-bps",type=float,default=10); ap.add_argument("--full-score-output",action="store_true"); ap.add_argument("--only-variant"); ap.add_argument("--no-detail-logs",action="store_true",default=True); ap.add_argument("--audit",choices=["minervini_lens","residual_momentum_deep","residual_live_validation","residual_full_sweep","residual_concentration","ttl_renewal","ttl_composite_forensics","r100_composite_experiment"]); args=ap.parse_args(); logging.basicConfig(level=logging.INFO)
    if args.demo: p=demo_prices(); us=[f"US{i}" for i in range(8)]; jp=[f"JP{i}.T" for i in range(8)]
    else:
        if args.audit=="ttl_composite_forensics":
            summary=run_ttl_composite_forensics_audit(args.source_dir,args.output_dir,args.cache_dir,args.rerun_selected,args.variants,args.quick,args.force_refresh_cache); print(f"Output directory: {Path(args.output_dir).resolve()}"); print(summary.to_string(index=False) if hasattr(summary,"to_string") else summary); return
        us,jp=get_live_universe()
        if args.audit=="r100_composite_experiment":
            summary=run_r100_composite_experiment_audit(None,us,jp,args.start,args.end,args.output_dir,args.source_dir,args.cache_dir,args.resume,args.variants,args.only_variant,args.quick,args.force_refresh_cache,args.tax_rate,args.slippage_bps,args.no_detail_logs); print(f"Output directory: {Path(args.output_dir).resolve()}"); print(summary.to_string()); return
        if args.audit=="ttl_renewal":
            summary=run_ttl_renewal_audit(None,us,jp,args.start,args.end,args.output_dir,quick=args.quick,cache_dir=args.cache_dir,force_refresh_cache=args.force_refresh_cache,resume=args.resume,tax_rate=args.tax_rate,slippage_bps=args.slippage_bps,full_score_output=args.full_score_output); print(f"Output directory: {Path(args.output_dir).resolve()}"); print(summary[[c for c in ["CAGR","Annualized_Volatility","Max_Drawdown","Sharpe","Calmar","Turnover"] if c in summary.columns]].to_string()); return
        if args.audit=="residual_concentration":
            summary=run_residual_concentration_audit(None,us,jp,args.start,args.end,args.output_dir,quick=args.quick); print(f"Output directory: {Path(args.output_dir).resolve()}"); print(summary[["CAGR","Annualized_Volatility","Max_Drawdown","Sharpe","Calmar","Turnover","Judgment"]].to_string()); return
        if args.audit=="residual_full_sweep":
            summary=run_residual_full_sweep(None,us,jp,args.start,args.end,args.output_dir); print(f"Output directory: {Path(args.output_dir).resolve()}"); print(summary[["CAGR","Annualized_Volatility","Max_Drawdown","Sharpe","Calmar","Turnover","Judgment"]].to_string()); return
        if args.audit=="residual_live_validation":
            summary=run_residual_live_validation(None,us,jp,args.start,args.end,args.output_dir); print(f"Output directory: {Path(args.output_dir).resolve()}"); print(summary[["CAGR","Annualized_Volatility","Max_Drawdown","Sharpe","Calmar","Turnover","Judgment"]].to_string()); return
        requested=list(dict.fromkeys([*us,*jp,"^GSPC","^N225",*BENCHMARKS.values(),"^TOPX"])); p=download_live_prices(requested,args.start,args.end)
        if p.empty: raise SystemExit("live mode error: yfinance produced no usable adjusted-close prices")
    if args.audit=="ttl_composite_forensics":
        summary=run_ttl_composite_forensics_audit(args.source_dir,args.output_dir,args.cache_dir,args.rerun_selected,args.variants,args.quick,args.force_refresh_cache); print(f"Output directory: {Path(args.output_dir).resolve()}"); print(summary.to_string(index=False) if hasattr(summary,"to_string") else summary); return
    if args.audit=="r100_composite_experiment":
        summary=run_r100_composite_experiment_audit(p,us,jp,args.start,args.end,args.output_dir,args.source_dir,args.cache_dir,args.resume,args.variants,args.only_variant,args.quick,args.force_refresh_cache,args.tax_rate,args.slippage_bps,args.no_detail_logs); print(f"Output directory: {Path(args.output_dir).resolve()}"); print(summary.to_string()); return
    if args.audit=="ttl_renewal":
        summary=run_ttl_renewal_audit(p,us,jp,args.start,args.end,args.output_dir,quick=args.quick,cache_dir=args.cache_dir,force_refresh_cache=args.force_refresh_cache,resume=args.resume,tax_rate=args.tax_rate,slippage_bps=args.slippage_bps,full_score_output=args.full_score_output); print(f"Output directory: {Path(args.output_dir).resolve()}"); print(summary[[c for c in ["CAGR","Annualized_Volatility","Max_Drawdown","Sharpe","Calmar","Turnover"] if c in summary.columns]].to_string()); return
    if args.audit=="residual_concentration":
        summary=run_residual_concentration_audit(p,us,jp,args.start,args.end,args.output_dir,quick=args.quick); print(f"Output directory: {Path(args.output_dir).resolve()}"); print(summary[["CAGR","Annualized_Volatility","Max_Drawdown","Sharpe","Calmar","Turnover","Judgment"]].to_string()); return
    if args.audit=="residual_full_sweep":
        summary=run_residual_full_sweep(p,us,jp,args.start,args.end,args.output_dir); print(f"Output directory: {Path(args.output_dir).resolve()}"); print(summary[["CAGR","Annualized_Volatility","Max_Drawdown","Sharpe","Calmar","Turnover","Judgment"]].to_string()); return
    if args.audit=="residual_live_validation":
        summary=run_residual_live_validation(p,us,jp,args.start,args.end,args.output_dir); print(f"Output directory: {Path(args.output_dir).resolve()}"); print(summary[["CAGR","Annualized_Volatility","Max_Drawdown","Sharpe","Calmar","Turnover","Judgment"]].to_string()); return
    if args.audit=="residual_momentum_deep":
        summary=run_residual_momentum_deep_audit(p,us,jp,args.start,args.end,args.output_dir); print(f"Output directory: {Path(args.output_dir).resolve()}"); print(summary[["CAGR","Annualized_Volatility","Max_Drawdown","Sharpe","Calmar","Turnover","Judgment"]].to_string()); return
    if args.audit=="minervini_lens":
        summary=run_minervini_lens_audit(p,us,jp,args.start,args.end,args.output_dir); print(f"Output directory: {Path(args.output_dir).resolve()}"); print(summary[["CAGR","Annualized_Volatility","Max_Drawdown","Sharpe","Calmar","Turnover","Judgment"]].to_string()); return
    s,sel,t=run_backtest(p,us,jp,args.start,args.end); b={k:p[v].pct_change().loc[args.start:args.end].fillna(0) for k,v in BENCHMARKS.items() if v in p}; summary,verdict=write_outputs(args.output_dir,s,sel,t,b)
    generated=[name for name in OUTPUT_FILES if (Path(args.output_dir)/name).is_file()]
    if not generated: raise SystemExit("error: no backtest artifacts were generated")
    print(f"Output directory: {Path(args.output_dir).resolve()}"); print("Generated files:"); print("\n".join(f"- {name}" for name in generated))
    print("backtest_summary.csv key rows:")
    print(summary.loc[[x for x in ("Alpha_Always","Alpha_Regime_Filter","SPY","QQQ","VT") if x in summary.index],["CAGR","Annualized_Volatility","Max_Drawdown","Sharpe","Calmar"]].to_string())
    print(f"Final provisional verdict: {verdict}")
if __name__=="__main__": main()
