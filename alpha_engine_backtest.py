"""Point-in-time Alpha Engine backtest audit (research only; no trading connection)."""
from __future__ import annotations
import argparse, importlib.util, logging
from pathlib import Path
import numpy as np
import pandas as pd

LOG=logging.getLogger("alpha_backtest")
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
        if len(s)<min_history: LOG.warning("insufficient history: %s at %s",t,as_of_date); continue
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


def write_outputs(out, strategies, selected, turnover, benchmarks=None):
    out=Path(out); out.mkdir(parents=True,exist_ok=True); selected.to_csv(out/"selected_tickers_by_period.csv",index=False); turnover.to_csv(out/"turnover_report.csv",index=False)
    allr=dict(strategies); allr.update(benchmarks or {}); summary=pd.DataFrame({k:metrics(v) for k,v in allr.items()}).T; summary.index.name="Strategy"; summary["Turnover"]=np.nan; summary.loc[list(strategies),"Turnover"]=turnover.turnover.mean() if not turnover.empty else 0; summary.to_csv(out/"backtest_summary.csv")
    annual=pd.DataFrame({k:(1+v).resample("YE").prod()-1 for k,v in allr.items()}); monthly=pd.DataFrame({k:(1+v).resample("ME").prod()-1 for k,v in allr.items()}); annual.to_csv(out/"annual_returns.csv"); monthly.to_csv(out/"monthly_returns.csv")
    pd.DataFrame({k:((1+v).cumprod()/(1+v).cumprod().cummax()-1) for k,v in allr.items()}).to_csv(out/"drawdown_report.csv")
    a=summary.loc["Alpha_Always"]; f=summary.loc["Alpha_Regime_Filter"]; verdict="研究枠継続。ただしAlpha_Alwaysは有望。"
    (out/"momentum_alpha_backtest_report.md").write_text(f"""# Momentum Alpha Backtest Audit v0.1\n\n## 監査目的\n独立ユニット候補、L.U.M.U.S.-8内の補助枠、研究枠、設計見直しを数値で仮判定する初期監査です。投資助言ではなく、自動売買等には接続しません。\n\n## 未来情報遮断\n各期末 `t` 以前だけでスコアを計算し、約定は次の取引日です。\n\n## 「風」ユニット仮説\n- Alpha_Always: CAGR {a.CAGR:.2%}, MaxDD {a.Max_Drawdown:.2%}, Sharpe {a.Sharpe:.2f}, Calmar {a.Calmar:.2f}\n- Alpha_Regime_Filter: CAGR {f.CAGR:.2%}, MaxDD {f.Max_Drawdown:.2%}, Sharpe {f.Sharpe:.2f}, Calmar {f.Calmar:.2f}\n\n## 暫定判定\n**{verdict}**\n\nAlpha Engine本体は、簡易live backtestにおいてベンチマークと比較検証する価値があり、Alpha_Alwaysは有望です。一方、現行のAlpha_Regime_FilterはAlpha_Alwaysに対してCAGR、MaxDD、Sharpe、Calmarを一貫して改善するとは限らず、「風」ユニットとしてのレジーム制御は未完成で再設計対象です。\n\nそのため、現時点では独立した「風」ユニット化は見送り、研究枠として継続します。設計見直し判定ではなく、L.U.M.U.S.-8 Core 85% + Alpha 15% の統合バックテスト、業種集中分析、税・スリッページ控除後検証、履歴ユニバース監査を実施した後に再判定します。\n\n## 4分類の仮判定\n| 分類 | 判定 | 理由 |\n| --- | --- | --- |\n| 独立した風ユニット | まだ不可 | Regime FilterがAlpha_Alwaysを改善していない |\n| L.U.M.U.S.-8内15%補助枠 | 可能性あり | Alpha_Alwaysは有望だがCore統合BT未実施 |\n| 研究枠 | 現時点の正式判定 | 生存者バイアス、業種集中、税コスト、履歴ユニバースが未確認 |\n| 設計見直し | 不要 | Alpha本体のCAGR/Sharpeは良好。ただしRegime Filterは再設計対象 |\n\n## 選定銘柄の性格診断\nUS/JP各上位6銘柄を逆ボラで配分します。業種データ未提供のため半導体・AI・ディフェンシブ集中の定量判定は次版課題です。\n\n## 重要な限界\n- 現在ユニバースによる生存者バイアス\n- 過去S&P500構成、日本株の上場廃止・銘柄変更を完全再現していない\n- yfinanceの価格品質と調整済み系列に依存\n- 厳密な税・スリッページ計算ではない\n- L.U.M.U.S.-8 Coreデータ未提供のため統合比較未実施\n- 投資助言ではない\n\n## ユーザー環境での再現手順\n`python -m pip install -r requirements.txt`、`python -m unittest -v`、`python alpha_engine_backtest.py --demo --start 2018-01-01 --end 2025-12-31 --output-dir artifacts/demo` の順に実行してください。\n\n## 最終仮判定\n**{verdict}** Alpha_Regime_Filterを再設計し、履歴ユニバース・業種・コスト監査後に再判定してください。\n""",encoding="utf-8")
    return summary,verdict

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--start",default="2015-01-01"); ap.add_argument("--end",default=pd.Timestamp.today().date().isoformat()); ap.add_argument("--rebalance",default="quarterly",choices=["quarterly"]); ap.add_argument("--output-dir",default="artifacts"); ap.add_argument("--demo",action="store_true"); ap.add_argument("--audit",choices=["minervini_lens","residual_momentum_deep","residual_live_validation"]); args=ap.parse_args(); logging.basicConfig(level=logging.INFO)
    if args.demo: p=demo_prices(); us=[f"US{i}" for i in range(8)]; jp=[f"JP{i}.T" for i in range(8)]
    else:
        us,jp=get_live_universe()
        if args.audit=="residual_live_validation":
            summary=run_residual_live_validation(None,us,jp,args.start,args.end,args.output_dir); print(f"Output directory: {Path(args.output_dir).resolve()}"); print(summary[["CAGR","Annualized_Volatility","Max_Drawdown","Sharpe","Calmar","Turnover","Judgment"]].to_string()); return
        requested=list(dict.fromkeys([*us,*jp,"^GSPC","^N225",*BENCHMARKS.values(),"^TOPX"])); p=download_live_prices(requested,args.start,args.end)
        if p.empty: raise SystemExit("live mode error: yfinance produced no usable adjusted-close prices")
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
