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
    hist=asof_prices(prices,tickers if tickers else []).ffill()
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

def write_outputs(out, strategies, selected, turnover, benchmarks=None):
    out=Path(out); out.mkdir(parents=True,exist_ok=True); selected.to_csv(out/"selected_tickers_by_period.csv",index=False); turnover.to_csv(out/"turnover_report.csv",index=False)
    allr=dict(strategies); allr.update(benchmarks or {}); summary=pd.DataFrame({k:metrics(v) for k,v in allr.items()}).T; summary.index.name="Strategy"; summary["Turnover"]=np.nan; summary.loc[list(strategies),"Turnover"]=turnover.turnover.mean() if not turnover.empty else 0; summary.to_csv(out/"backtest_summary.csv")
    annual=pd.DataFrame({k:(1+v).resample("YE").prod()-1 for k,v in allr.items()}); monthly=pd.DataFrame({k:(1+v).resample("ME").prod()-1 for k,v in allr.items()}); annual.to_csv(out/"annual_returns.csv"); monthly.to_csv(out/"monthly_returns.csv")
    pd.DataFrame({k:((1+v).cumprod()/(1+v).cumprod().cummax()-1) for k,v in allr.items()}).to_csv(out/"drawdown_report.csv")
    a=summary.loc["Alpha_Always"]; f=summary.loc["Alpha_Regime_Filter"]; verdict="研究枠へ降格" if f.Sharpe<=0 or f.Max_Drawdown<-.35 else "独立ユニット化可能" if f.Sharpe>=a.Sharpe and f.Max_Drawdown>=a.Max_Drawdown else "L.U.M.U.S.-8内の15%補助枠として継続"
    (out/"momentum_alpha_backtest_report.md").write_text(f"""# Momentum Alpha Backtest Audit v0.1\n\n## 監査目的\n独立ユニット候補、L.U.M.U.S.-8内の補助枠、研究枠、設計見直しを数値で仮判定する初期監査です。投資助言ではなく、自動売買等には接続しません。\n\n## 未来情報遮断\n各期末 `t` 以前だけでスコアを計算し、約定は次の取引日です。\n\n## 「風」ユニット仮説\n- Alpha_Always: CAGR {a.CAGR:.2%}, MaxDD {a.Max_Drawdown:.2%}, Sharpe {a.Sharpe:.2f}, Calmar {a.Calmar:.2f}\n- Alpha_Regime_Filter: CAGR {f.CAGR:.2%}, MaxDD {f.Max_Drawdown:.2%}, Sharpe {f.Sharpe:.2f}, Calmar {f.Calmar:.2f}\n\n## 4分類の仮判定\n1. 独立ユニット化可能\n2. L.U.M.U.S.-8内の15%補助枠として継続\n3. 研究枠へ降格\n4. 設計見直し\n\n今回の数値による仮判定は **{verdict}** です。\n\n## 選定銘柄の性格診断\nUS/JP各上位6銘柄を逆ボラで配分します。業種データ未提供のため半導体・AI・ディフェンシブ集中の定量判定は次版課題です。\n\n## 重要な限界\n- 現在ユニバースによる生存者バイアス\n- 過去S&P500構成、日本株の上場廃止・銘柄変更を完全再現していない\n- yfinanceの価格品質と調整済み系列に依存\n- 厳密な税・スリッページ計算ではない\n- L.U.M.U.S.-8 Coreデータ未提供のため統合比較未実施\n- 依存関係をインストールできないネットワーク制限環境では、デモとテストを実行できない\n- 投資助言ではない\n\n## ユーザー環境での再現手順\n`python -m pip install -r requirements.txt`、`python -m unittest -v`、`python alpha_engine_backtest.py --demo --start 2018-01-01 --end 2025-12-31 --output-dir artifacts/demo` の順に実行してください。\n\n## 最終仮判定\n**{verdict}**。上記CAGR、MaxDD、Sharpe、Calmarを根拠とし、履歴ユニバース・業種・コスト監査後に再判定してください。\n""",encoding="utf-8")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--start",default="2015-01-01"); ap.add_argument("--end",default=pd.Timestamp.today().date().isoformat()); ap.add_argument("--rebalance",default="quarterly",choices=["quarterly"]); ap.add_argument("--output-dir",default="artifacts"); ap.add_argument("--demo",action="store_true"); args=ap.parse_args(); logging.basicConfig(level=logging.INFO)
    if args.demo: p=demo_prices(); us=[f"US{i}" for i in range(8)]; jp=[f"JP{i}.T" for i in range(8)]
    else:
        us,jp=get_live_universe(); requested=list(dict.fromkeys([*us,*jp,"^GSPC","^N225",*BENCHMARKS.values(),"^TOPX"])); p=download_live_prices(requested,args.start,args.end)
        if p.empty: raise SystemExit("live mode error: yfinance produced no usable adjusted-close prices")
    s,sel,t=run_backtest(p,us,jp,args.start,args.end); b={k:p[v].pct_change().loc[args.start:args.end].fillna(0) for k,v in BENCHMARKS.items() if v in p}; write_outputs(args.output_dir,s,sel,t,b)
    generated=[name for name in OUTPUT_FILES if (Path(args.output_dir)/name).is_file()]
    if not generated: raise SystemExit("error: no backtest artifacts were generated")
    print(f"Saved artifacts to {Path(args.output_dir).resolve()}"); print("Generated files:"); print("\n".join(f"- {name}" for name in generated))
if __name__=="__main__": main()
