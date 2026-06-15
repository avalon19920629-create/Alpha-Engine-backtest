"""L.U.M.U.S.-8 Core + Alpha integration audit; research only, never trades."""
from __future__ import annotations
import argparse, logging
from pathlib import Path
import numpy as np
import pandas as pd
import alpha_engine_backtest as alpha

LOG=logging.getLogger("core_alpha")
CORE_TICKERS=("VT","BNDX","TLT","TIP","GLDM","DBC","XLRE","CASH")
FILES=("core_alpha_backtest_summary.csv","core_alpha_annual_returns.csv","core_alpha_monthly_returns.csv","core_alpha_drawdown_report.csv","core_alpha_equity_curves.csv","core_alpha_turnover_report.csv","core_alpha_comparison_vs_core.csv","core_alpha_integration_report.md","selected_tickers_by_period.csv")

def load_core_weights(path):
    d=pd.read_csv(path)
    if list(d.columns)!=["ticker","weight"]: raise ValueError("core weights must have ticker,weight columns")
    d["ticker"]=d.ticker.astype(str).str.upper(); d["weight"]=pd.to_numeric(d.weight,errors="raise")
    if d.ticker.duplicated().any() or (d.weight<0).any(): raise ValueError("core weights require unique tickers and non-negative weights")
    if not np.isclose(d.weight.sum(),1.0): raise ValueError(f"core weight total must equal 1.0 (got {d.weight.sum():.6f})")
    return d.set_index("ticker").weight

def core_returns(prices, weights, start, end):
    unknown=set(weights.index)-set(CORE_TICKERS)
    if unknown: LOG.warning("non-standard Core tickers: %s",sorted(unknown))
    needed=[t for t,w in weights.items() if t!="CASH" and w>0]
    missing=[t for t in needed if t not in prices or prices[t].dropna().empty]
    if missing: raise ValueError(f"Core price data unavailable for positive-weight ETF(s): {', '.join(missing)}")
    idx=prices.loc[start:end].index
    asset=pd.DataFrame(index=idx)
    for t in weights.index: asset[t]=0.0 if t=="CASH" or t not in prices else prices[t].pct_change().reindex(idx)
    if asset[needed].isna().any().any():
        LOG.warning("Core ETF return gaps filled with 0%%")
        asset[needed]=asset[needed].fillna(0)
    out=pd.Series(index=idx,dtype=float,name="Core_Only")
    for _, dates in pd.Series(idx,index=idx).groupby(idx.to_period("Q")):
        block=asset.loc[dates.values]; wealth=(1+block).cumprod(); notionals=wealth.mul(weights,axis=1); out.loc[block.index]=notionals.sum(axis=1).pct_change().fillna(block.iloc[0].dot(weights)).values
    return out.fillna(0),asset

def combine(core, alpha_always, alpha_regime=None):
    x=pd.concat({"Core_Only":core,"Alpha_Always_Only":alpha_always},axis=1).fillna(0)
    for a in (10,15,20): x[f"Core{100-a}_Alpha{a}"]=x.Core_Only*(1-a/100)+x.Alpha_Always_Only*a/100
    if alpha_regime is not None:
        for a in (10,15): x[f"Core{100-a}_AlphaRegime{a}"]=x.Core_Only*(1-a/100)+alpha_regime.reindex(x.index).fillna(0)*a/100
    return x

def comparison(summary):
    c=summary.loc["Core_Only"]
    return pd.DataFrame({"CAGR_improvement_vs_Core_Only":summary.CAGR-c.CAGR,"MaxDD_change_vs_Core_Only":summary.Max_Drawdown-c.Max_Drawdown,"Sharpe_change_vs_Core_Only":summary.Sharpe-c.Sharpe,"Calmar_change_vs_Core_Only":summary.Calmar-c.Calmar})

def verdict(summary):
    c,a=summary.loc["Core_Only"],summary.loc["Core85_Alpha15"]
    cagr=a.CAGR>c.CAGR; dd=a.Max_Drawdown>=c.Max_Drawdown-.05; risk=(a.Sharpe>=c.Sharpe or a.Calmar>=c.Calmar)
    if cagr and dd and risk:return "1. L.U.M.U.S.-8内15%補助枠として採用候補"
    if cagr and risk:return "2. 10%以下の小型補助枠として継続検証"
    if a.Max_Drawdown<c.Max_Drawdown and not risk:return "3. 研究枠継続"
    return "4. 統合不採用"

def write_outputs(out, returns, weights, selected, alpha_turnover):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    summary=pd.DataFrame({k:alpha.metrics(returns[k]) for k in returns}).T; summary.index.name="Strategy"; summary["Turnover"]=0.0
    at=float(alpha_turnover.turnover.mean()) if not alpha_turnover.empty else 0.0
    summary.loc["Alpha_Always_Only","Turnover"]=at
    for a in (10,15,20): summary.loc[f"Core{100-a}_Alpha{a}","Turnover"]=at*a/100
    comp=comparison(summary); summary.join(comp).to_csv(out/FILES[0]); comp.to_csv(out/FILES[6])
    annual=returns.apply(lambda s:(1+s).resample("YE").prod()-1); monthly=returns.apply(lambda s:(1+s).resample("ME").prod()-1)
    annual.to_csv(out/FILES[1]); monthly.to_csv(out/FILES[2]); ((1+returns).cumprod()/(1+returns).cumprod().cummax()-1).to_csv(out/FILES[3]); (1+returns).cumprod().to_csv(out/FILES[4])
    pd.DataFrame({"Strategy":summary.index,"Turnover":summary.Turnover.values}).to_csv(out/FILES[5],index=False); selected.to_csv(out/FILES[8],index=False)
    v=verdict(summary); rows="\n".join(f"| {i} | {r.CAGR:.2%} | {r.Max_Drawdown:.2%} | {r.Sharpe:.2f} | {r.Calmar:.2f} |" for i,r in summary.iterrows())
    wr="\n".join(f"| {t} | {w:.2%} |" for t,w in weights.items())
    (out/FILES[7]).write_text(f"""# Core + Alpha Integration Backtest Report

## 1. 監査目的
Alpha EngineをL.U.M.U.S.-8の10〜15%補助Alpha枠として採用する価値があるかを検証する初期監査です。

## 2. Core設定
| ticker | weight |\n|---|---:|\n{wr}

## 3. 比較結果
| Strategy | CAGR | MaxDD | Sharpe | Calmar |\n|---|---:|---:|---:|---:|\n{rows}

## 4. 採用判定
**{v}**

Core85_Alpha15について、Core_Only比のCAGR、MaxDD、Sharpe/Calmar、およびTurnoverを確認した仮判定です。コスト控除後の有用性とCoreの防御思想維持は追加検証が必要です。主判定はAlpha_Alwaysであり、Alpha_Regime_Filterは参考です。

## 5. 重要な限界
- Alpha側は現在ユニバースによる生存者バイアスを含む
- 過去S&P500構成、日本株の上場廃止、銘柄変更を完全再現していない
- yfinance価格品質に依存する
- 税・スリッページは簡易または未考慮
- Core比率は `config/lumus8_core_weights.csv` に依存する
- 投資助言ではない
- 自動売買、自動売却、自動配分変更には接続しない
""",encoding="utf-8")
    return summary.join(comp),v

def run_integration(prices, us, jp, weights, start, end, out):
    strategies,selected,turn=alpha.run_backtest(prices,us,jp,start,end); core,_=core_returns(prices,weights,start,end)
    r=combine(core,strategies["Alpha_Always"],strategies.get("Alpha_Regime_Filter"))
    for name,ticker in alpha.BENCHMARKS.items():
        if ticker in prices:r[name]=prices[ticker].pct_change().reindex(r.index).fillna(0)
    return write_outputs(out,r,weights,selected,turn)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--start",default="2015-01-01"); p.add_argument("--end",default=pd.Timestamp.today().date().isoformat()); p.add_argument("--core-weights",default="config/lumus8_core_weights.csv"); p.add_argument("--output-dir",default="artifacts/core_alpha"); p.add_argument("--demo",action="store_true"); a=p.parse_args(); logging.basicConfig(level=logging.INFO); w=load_core_weights(a.core_weights)
    if a.demo: prices=alpha.demo_prices(); us=[f"US{i}" for i in range(8)]; jp=[f"JP{i}.T" for i in range(8)]
    else:
        us,jp=alpha.get_live_universe(); prices=alpha.download_live_prices(list(dict.fromkeys([*us,*jp,"^GSPC","^N225",*alpha.BENCHMARKS.values(),*[t for t in w.index if t!="CASH"]])),a.start,a.end)
    summary,v=run_integration(prices,us,jp,w,a.start,a.end,a.output_dir)
    print(f"Output directory: {Path(a.output_dir).resolve()}\nGenerated files:\n"+"\n".join(f"- {f}" for f in FILES if (Path(a.output_dir)/f).exists())); print(summary.loc[[x for x in ("Core_Only","Alpha_Always_Only","Core90_Alpha10","Core85_Alpha15","Core80_Alpha20") if x in summary.index], ["CAGR","Max_Drawdown","Sharpe","Calmar"]]); print(f"Final provisional verdict: {v}")
if __name__=="__main__":main()
