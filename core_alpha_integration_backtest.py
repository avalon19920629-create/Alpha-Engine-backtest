"""L.U.M.U.S.-8 Core profiles + Alpha integration audit; research only, never trades."""
from __future__ import annotations
import argparse, logging
from pathlib import Path
import numpy as np
import pandas as pd
import alpha_engine_backtest as alpha

LOG=logging.getLogger("core_alpha")
FILES=("core_alpha_profile_summary.csv","core_alpha_profile_comparison_vs_core.csv","core_alpha_profile_annual_returns.csv","core_alpha_profile_monthly_returns.csv","core_alpha_profile_drawdown_report.csv","core_alpha_profile_equity_curves.csv","core_alpha_profile_turnover_report.csv","selected_tickers_by_period.csv","core_alpha_profile_report.md")
LEGACY_FILES=("core_alpha_backtest_summary.csv","core_alpha_annual_returns.csv","core_alpha_monthly_returns.csv","core_alpha_drawdown_report.csv","core_alpha_equity_curves.csv","core_alpha_turnover_report.csv","core_alpha_comparison_vs_core.csv","core_alpha_integration_report.md","selected_tickers_by_period.csv")

def load_core_weights(path):
    d=pd.read_csv(path)
    if list(d.columns)!=["ticker","weight"]: raise ValueError("core weights must have ticker,weight columns")
    d["ticker"]=d.ticker.astype(str).str.upper(); d["weight"]=pd.to_numeric(d.weight,errors="raise")
    if d.ticker.duplicated().any() or (d.weight<0).any(): raise ValueError("core weights require unique tickers and non-negative weights")
    if not np.isclose(d.weight.sum(),1.0): raise ValueError(f"core weight total must equal 1.0 (got {d.weight.sum():.6f})")
    return d.set_index("ticker").weight

def load_core_profiles(path):
    d=pd.read_csv(path)
    if list(d.columns)!=["profile","ticker","weight"]: raise ValueError("core profiles must have profile,ticker,weight columns")
    d["profile"]=d.profile.astype(str).str.strip(); d["ticker"]=d.ticker.astype(str).str.upper(); d["weight"]=pd.to_numeric(d.weight,errors="raise")
    if (d.weight<0).any() or d.duplicated(["profile","ticker"]).any(): raise ValueError("profiles require unique tickers and non-negative weights")
    totals=d.groupby("profile").weight.sum(); bad=totals[~np.isclose(totals,1.0)]
    if len(bad): raise ValueError("core profile weight total must equal 1.0: "+", ".join(f"{k}={v:.6f}" for k,v in bad.items()))
    return {name:g.set_index("ticker").weight for name,g in d.groupby("profile",sort=False)}

def required_core_tickers(profiles):
    return sorted({t for w in profiles.values() for t,v in w.items() if v>0 and t!="CASH"})

def core_returns(prices, weights, start, end):
    needed=[t for t,w in weights.items() if t!="CASH" and w>0]
    missing=[t for t in needed if t not in prices or prices[t].dropna().empty]
    if missing: raise ValueError(f"Core price data unavailable for positive-weight asset(s): {', '.join(missing)}")
    idx=prices.loc[start:end].index; asset=pd.DataFrame(index=idx)
    for t in weights.index:
        asset[t]=0.0 if t=="CASH" or t not in prices else prices[t].ffill().pct_change().reindex(idx)
    if asset[needed].isna().any().any(): LOG.warning("Core return gaps filled with 0%%"); asset[needed]=asset[needed].fillna(0)
    out=pd.Series(index=idx,dtype=float)
    for _,dates in pd.Series(idx,index=idx).groupby(idx.to_period("Q")):
        block=asset.loc[dates.values]; notionals=(1+block).cumprod().mul(weights,axis=1)
        out.loc[block.index]=notionals.sum(axis=1).pct_change().fillna(block.iloc[0].dot(weights)).values
    return out.fillna(0).rename("Core_Only"),asset

def combine(core, alpha_always, alpha_regime=None):
    x=pd.concat({"Core_Only":core,"Alpha_Always_Only":alpha_always},axis=1).fillna(0)
    for a in (10,15,20): x[f"Core{100-a}_Alpha{a}"]=x.Core_Only*(1-a/100)+x.Alpha_Always_Only*a/100
    if alpha_regime is not None: x["Core85_AlphaRegime15"]=x.Core_Only*.85+alpha_regime.reindex(x.index).fillna(0)*.15
    return x

def verdict(summary, profile):
    c,a=summary.loc[f"{profile}_Core_Only"],summary.loc[f"{profile}_Core85_Alpha15"]
    if a.CAGR>c.CAGR and a.Max_Drawdown>=c.Max_Drawdown-.05 and (a.Sharpe>=c.Sharpe or a.Calmar>=c.Calmar): return "1. 15%補助Alpha枠として採用候補"
    if a.CAGR>c.CAGR and (a.Sharpe>=c.Sharpe or a.Calmar>=c.Calmar): return "2. 10%以下の小型補助枠として継続検証"
    if a.Max_Drawdown<c.Max_Drawdown: return "3. 研究枠継続"
    return "4. 統合不採用"

def evaluation_returns(returns, start, end):
    """Restrict every performance series to the requested evaluation window."""
    frame=pd.DataFrame(returns).sort_index()
    frame=frame.loc[(frame.index>=pd.Timestamp(start)) & (frame.index<=pd.Timestamp(end))]
    if frame.empty: raise ValueError(f"no evaluation returns between {start} and {end}")
    empty=[name for name in frame if frame[name].dropna().empty]
    if empty: raise ValueError("no evaluation returns for: "+", ".join(empty))
    return frame

def evaluation_metrics(r):
    r=pd.Series(r).dropna(); equity=(1+r).cumprod()
    years=(r.index[-1]-r.index[0]).days/365.25
    if years<=0: raise ValueError("evaluation returns must span more than one calendar day")
    final=float(equity.iloc[-1]); cagr=final**(1/years)-1; dd=float((equity/equity.cummax()-1).min())
    vol=r.std()*np.sqrt(252); down=r[r<0].std()*np.sqrt(252); monthly=(1+r).resample("ME").prod()-1; annual=(1+r).resample("YE").prod()-1
    return {"CAGR":cagr,"Annualized_Volatility":vol,"Max_Drawdown":dd,"Sharpe":cagr/vol if vol else np.nan,"Calmar":cagr/abs(dd) if dd else np.nan,"Sortino":cagr/down if down else np.nan,"Total_Return":final-1,"Best_Year":annual.max(),"Worst_Year":annual.min(),"Monthly_Win_Rate":(monthly>0).mean(),"Evaluation_Start":r.index[0].date().isoformat(),"Evaluation_End":r.index[-1].date().isoformat(),"Evaluation_Years":years}

def write_profile_outputs(out, returns, profiles, selected, alpha_turnover, start=None, end=None, warmup_start=None):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    start=start or returns.index.min(); end=end or returns.index.max(); returns=evaluation_returns(returns,start,end)
    summary=pd.DataFrame({k:evaluation_metrics(v) for k,v in returns.items()}).T; summary.index.name="Strategy"; summary["Turnover"]=0.0
    at=float(alpha_turnover.turnover.mean()) if not alpha_turnover.empty else 0.0
    if "Alpha_Always_Only" in summary.index: summary.loc["Alpha_Always_Only","Turnover"]=at
    for p in profiles:
        for a in (10,15,20): summary.loc[f"{p}_Core{100-a}_Alpha{a}","Turnover"]=at*a/100
    rows=[]
    for p in profiles:
        c=summary.loc[f"{p}_Core_Only"]
        for name in summary.index[summary.index.str.startswith(p+"_")]:
            r=summary.loc[name]; rows.append({"Profile":p,"Strategy":name,"CAGR_improvement_vs_same_Core_Only":r.CAGR-c.CAGR,"MaxDD_change_vs_same_Core_Only":r.Max_Drawdown-c.Max_Drawdown,"Sharpe_change_vs_same_Core_Only":r.Sharpe-c.Sharpe,"Calmar_change_vs_same_Core_Only":r.Calmar-c.Calmar})
    comp=pd.DataFrame(rows).set_index("Strategy"); summary.to_csv(out/FILES[0]); comp.to_csv(out/FILES[1])
    annual=returns.apply(lambda s:(1+s).resample("YE").prod()-1); monthly=returns.apply(lambda s:(1+s).resample("ME").prod()-1)
    annual.to_csv(out/FILES[2]); monthly.to_csv(out/FILES[3]); equity=(1+returns).cumprod(); (equity/equity.cummax()-1).to_csv(out/FILES[4]); equity.to_csv(out/FILES[5]); summary[["Turnover"]].to_csv(out/FILES[6]); selected.to_csv(out/FILES[7],index=False)
    verdicts={p:verdict(summary,p) for p in profiles}; weights="\n".join(f"| {p} | {t} | {w:.2%} |" for p,ws in profiles.items() for t,w in ws.items()); results="\n".join(f"| {i} | {r.CAGR:.2%} | {r.Max_Drawdown:.2%} | {r.Sharpe:.2f} | {r.Calmar:.2f} | {r.Turnover:.2%} |" for i,r in summary.iterrows()); decisions="\n".join(f"- **{p}: {v}**" for p,v in verdicts.items())
    (out/FILES[8]).write_text(f"""# L.U.M.U.S.-8 Core Profiles + Alpha Integration Backtest

## 1. 監査目的
Alpha EngineをMAX_CAGR_MDD25およびROBUST_MEDIANのL.U.M.U.S.-8 Coreに10% / 15% / 20%混ぜた場合の有効性を検証する初期監査です。

## 2. 評価期間とウォームアップ
- Evaluation start: {summary.Evaluation_Start.min()}
- Evaluation end: {summary.Evaluation_End.max()}
- Warmup data start: {warmup_start if warmup_start is not None else "not available"}
- Metrics are calculated only from evaluation-period returns.
- Warmup data is used only for Alpha scoring and price-history preparation, not for performance metrics.

価格取得やAlphaスコア計算のために評価開始日前のデータを取得しているが、CAGR、MaxDD、Sharpe、Calmar、Total_Returnなどの成績指標は --start 以降、--end 以前の評価期間リターンのみから計算している。

## 3. Coreプロファイル
| Profile | Ticker | Weight |\n|---|---|---:|\n{weights}

## 4. GLD / GLDM と SHY / CASH の扱い
- GLDはGLDMの長期履歴プロキシとして使用している
- SHY / CASH は短期債または現金相当枠であり、SHYは短期債または現金相当枠として使用している
- 実運用ではGLDMや現金に置換する可能性がある

## 5. 比較結果
| Strategy | CAGR | MaxDD | Sharpe | Calmar | Turnover |\n|---|---:|---:|---:|---:|---:|\n{results}

## 6. 採用判定
{decisions}

Core85_Alpha15のCAGR改善、限定的なMaxDD悪化、Sharpe/Calmar改善、Coreの防御思想、Turnoverとコスト控除後の可能性に基づく仮判定です。Alpha_Regimeは参考であり主判定には使用しません。

## 7. 最終コメント
Alpha Engine本体が有望か、15%混合が有効か、10%混合の方が美しいか、攻撃型Coreとロバスト型CoreのどちらにAlphaが合うかは上表と同一Core比較で判断します。現段階では独立した風ユニットではなく、補助Alpha枠としての採用可能性を監査するものです。

## 8. 重要な限界
- Alpha側は現在ユニバースによる生存者バイアスを含む
- 過去S&P500構成、日本株の上場廃止、銘柄変更を完全再現していない
- yfinance価格品質に依存する
- 税・スリッページは簡易または未考慮
- GLDはGLDMの履歴プロキシである
- SHYは現金相当枠の履歴プロキシである
- BTC-USDは暗号資産価格系列であり、ETFとは取引日・市場時間が異なる可能性がある
- 投資助言ではない
- 自動売買、自動売却、自動配分変更には接続しない
""",encoding="utf-8")
    return summary,verdicts

def run_profiles(prices, us, jp, profiles, start, end, out):
    strategies,selected,turn=alpha.run_backtest(prices,us,jp,start,end); allr={"Alpha_Always_Only":strategies["Alpha_Always"],"Alpha_Regime":strategies["Alpha_Regime_Filter"]}
    for p,w in profiles.items():
        core,_=core_returns(prices,w,start,end); mixed=combine(core,strategies["Alpha_Always"],strategies.get("Alpha_Regime_Filter"))
        for n,s in mixed.items():
            if n!="Alpha_Always_Only": allr[f"{p}_{n}"]=s
    for name,ticker in (("SPY","SPY"),("QQQ","QQQ"),("VT","VT")):
        if ticker in prices: allr[name]=prices[ticker].pct_change().reindex(allr["Alpha_Always_Only"].index).fillna(0)
    return write_profile_outputs(out,pd.DataFrame(allr),profiles,selected,turn,start,end,prices.index.min().date().isoformat())

def write_outputs(out, returns, weights, selected, alpha_turnover):
    """Compatibility wrapper for the original single-Core API and filenames."""
    out=Path(out); profiles={"LEGACY":weights}; renamed=returns.rename(columns={c:(c if c=="Alpha_Always_Only" else "LEGACY_"+c) for c in returns.columns})
    summary,verdicts=write_profile_outputs(out,renamed,profiles,selected,alpha_turnover)
    core=summary.loc["LEGACY_Core_Only"]; summary["CAGR_improvement_vs_Core_Only"]=summary.CAGR-core.CAGR
    mapping=dict(zip(FILES,LEGACY_FILES))
    for src,dst in mapping.items():
        if (out/src).exists(): (out/dst).write_bytes((out/src).read_bytes())
    return summary,verdicts["LEGACY"]

def main():
    p=argparse.ArgumentParser(); p.add_argument("--start",default="2015-01-01"); p.add_argument("--end",default=pd.Timestamp.today().date().isoformat()); p.add_argument("--core-profiles"); p.add_argument("--core-weights"); p.add_argument("--output-dir",default="artifacts/core_alpha_profiles"); p.add_argument("--demo",action="store_true"); a=p.parse_args(); logging.basicConfig(level=logging.INFO)
    if a.core_profiles: profiles=load_core_profiles(a.core_profiles)
    elif a.core_weights: profiles={"Core":load_core_weights(a.core_weights)}
    else: profiles=load_core_profiles("config/lumus8_core_profiles.csv")
    if a.demo:
        prices=alpha.demo_prices(); us=[f"US{i}" for i in range(8)]; jp=[f"JP{i}.T" for i in range(8)]; rng=np.random.default_rng(9)
        for t in required_core_tickers(profiles):
            if t not in prices: prices[t]=100*np.exp(np.cumsum(rng.normal(.0002,.007,len(prices))))
    else:
        us,jp=alpha.get_live_universe(); requested=list(dict.fromkeys([*us,*jp,"^GSPC","^N225",*alpha.BENCHMARKS.values(),*required_core_tickers(profiles)])); prices=alpha.download_live_prices(requested,a.start,a.end)
        if prices.empty: raise SystemExit("live mode error: yfinance produced no usable prices")
    summary,v=run_profiles(prices,us,jp,profiles,a.start,a.end,a.output_dir)
    generated=[name for name in FILES if (Path(a.output_dir)/name).is_file()]
    print(f"Output directory: {Path(a.output_dir).resolve()}"); print("Generated files:"); print("\n".join(f"- {name}" for name in generated))
    print("core_alpha_profile_summary.csv key rows:"); print(summary[["CAGR","Total_Return","Max_Drawdown","Sharpe","Calmar","Evaluation_Start","Evaluation_End","Evaluation_Years"]].to_string())
    print(f"Final provisional verdict: {v}")
if __name__=="__main__": main()
