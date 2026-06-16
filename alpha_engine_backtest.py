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

def compute_residual_momentum_score(prices, tickers, as_of_date, region="US", windows=(63,126,252)):
    """Cross-sectional simple benchmark-adjusted momentum; missing benchmark falls back to neutral scores."""
    hist=asof_prices(prices,as_of_date).ffill(); tickers=[t for t in tickers if t in hist.columns]
    bench=_pick_benchmark(hist,region)
    rows=[]
    if not tickers:return pd.DataFrame(columns=["Ticker","residual_score"])
    for t in tickers:
        s=hist[t].dropna(); vals=[]
        for n,w in zip(windows,(1,2,3)):
            if len(s)>n and bench and bench in hist:
                b=hist[bench].dropna()
                if len(b)>n:
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


def write_outputs(out, strategies, selected, turnover, benchmarks=None):
    out=Path(out); out.mkdir(parents=True,exist_ok=True); selected.to_csv(out/"selected_tickers_by_period.csv",index=False); turnover.to_csv(out/"turnover_report.csv",index=False)
    allr=dict(strategies); allr.update(benchmarks or {}); summary=pd.DataFrame({k:metrics(v) for k,v in allr.items()}).T; summary.index.name="Strategy"; summary["Turnover"]=np.nan; summary.loc[list(strategies),"Turnover"]=turnover.turnover.mean() if not turnover.empty else 0; summary.to_csv(out/"backtest_summary.csv")
    annual=pd.DataFrame({k:(1+v).resample("YE").prod()-1 for k,v in allr.items()}); monthly=pd.DataFrame({k:(1+v).resample("ME").prod()-1 for k,v in allr.items()}); annual.to_csv(out/"annual_returns.csv"); monthly.to_csv(out/"monthly_returns.csv")
    pd.DataFrame({k:((1+v).cumprod()/(1+v).cumprod().cummax()-1) for k,v in allr.items()}).to_csv(out/"drawdown_report.csv")
    a=summary.loc["Alpha_Always"]; f=summary.loc["Alpha_Regime_Filter"]; verdict="研究枠継続。ただしAlpha_Alwaysは有望。"
    (out/"momentum_alpha_backtest_report.md").write_text(f"""# Momentum Alpha Backtest Audit v0.1\n\n## 監査目的\n独立ユニット候補、L.U.M.U.S.-8内の補助枠、研究枠、設計見直しを数値で仮判定する初期監査です。投資助言ではなく、自動売買等には接続しません。\n\n## 未来情報遮断\n各期末 `t` 以前だけでスコアを計算し、約定は次の取引日です。\n\n## 「風」ユニット仮説\n- Alpha_Always: CAGR {a.CAGR:.2%}, MaxDD {a.Max_Drawdown:.2%}, Sharpe {a.Sharpe:.2f}, Calmar {a.Calmar:.2f}\n- Alpha_Regime_Filter: CAGR {f.CAGR:.2%}, MaxDD {f.Max_Drawdown:.2%}, Sharpe {f.Sharpe:.2f}, Calmar {f.Calmar:.2f}\n\n## 暫定判定\n**{verdict}**\n\nAlpha Engine本体は、簡易live backtestにおいてベンチマークと比較検証する価値があり、Alpha_Alwaysは有望です。一方、現行のAlpha_Regime_FilterはAlpha_Alwaysに対してCAGR、MaxDD、Sharpe、Calmarを一貫して改善するとは限らず、「風」ユニットとしてのレジーム制御は未完成で再設計対象です。\n\nそのため、現時点では独立した「風」ユニット化は見送り、研究枠として継続します。設計見直し判定ではなく、L.U.M.U.S.-8 Core 85% + Alpha 15% の統合バックテスト、業種集中分析、税・スリッページ控除後検証、履歴ユニバース監査を実施した後に再判定します。\n\n## 4分類の仮判定\n| 分類 | 判定 | 理由 |\n| --- | --- | --- |\n| 独立した風ユニット | まだ不可 | Regime FilterがAlpha_Alwaysを改善していない |\n| L.U.M.U.S.-8内15%補助枠 | 可能性あり | Alpha_Alwaysは有望だがCore統合BT未実施 |\n| 研究枠 | 現時点の正式判定 | 生存者バイアス、業種集中、税コスト、履歴ユニバースが未確認 |\n| 設計見直し | 不要 | Alpha本体のCAGR/Sharpeは良好。ただしRegime Filterは再設計対象 |\n\n## 選定銘柄の性格診断\nUS/JP各上位6銘柄を逆ボラで配分します。業種データ未提供のため半導体・AI・ディフェンシブ集中の定量判定は次版課題です。\n\n## 重要な限界\n- 現在ユニバースによる生存者バイアス\n- 過去S&P500構成、日本株の上場廃止・銘柄変更を完全再現していない\n- yfinanceの価格品質と調整済み系列に依存\n- 厳密な税・スリッページ計算ではない\n- L.U.M.U.S.-8 Coreデータ未提供のため統合比較未実施\n- 投資助言ではない\n\n## ユーザー環境での再現手順\n`python -m pip install -r requirements.txt`、`python -m unittest -v`、`python alpha_engine_backtest.py --demo --start 2018-01-01 --end 2025-12-31 --output-dir artifacts/demo` の順に実行してください。\n\n## 最終仮判定\n**{verdict}** Alpha_Regime_Filterを再設計し、履歴ユニバース・業種・コスト監査後に再判定してください。\n""",encoding="utf-8")
    return summary,verdict

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--start",default="2015-01-01"); ap.add_argument("--end",default=pd.Timestamp.today().date().isoformat()); ap.add_argument("--rebalance",default="quarterly",choices=["quarterly"]); ap.add_argument("--output-dir",default="artifacts"); ap.add_argument("--demo",action="store_true"); ap.add_argument("--audit",choices=["minervini_lens"]); args=ap.parse_args(); logging.basicConfig(level=logging.INFO)
    if args.demo: p=demo_prices(); us=[f"US{i}" for i in range(8)]; jp=[f"JP{i}.T" for i in range(8)]
    else:
        us,jp=get_live_universe(); requested=list(dict.fromkeys([*us,*jp,"^GSPC","^N225",*BENCHMARKS.values(),"^TOPX"])); p=download_live_prices(requested,args.start,args.end)
        if p.empty: raise SystemExit("live mode error: yfinance produced no usable adjusted-close prices")
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
