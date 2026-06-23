"""Alpha Engine live holdings TTL manager.

Operational helper only: reads a user-confirmed live holdings ledger, evaluates the
existing audited TTL90/Renew30 Composite health checks on current data, and writes
manual review artifacts. It never connects to a broker, sends orders, or creates a
ledger from planned quantities alone.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

import alpha_engine_backtest as alpha
from alpha_engine_live_screener import build_live_variant

TTL_DAYS = 90
RENEWAL_DAYS = 30
WEEKLY_REVIEW_DAY = "FRI"
DATA_LOOKBACK_MONTHS = 18
DEFAULT_OUTPUT_ROOT = "artifacts/ttl_manager_runs"
TTL_DAY_CONVENTION = "calendar_days; audited backtest uses pd.Timedelta(days=ttl_days) and next available trading date"

LEDGER_COLUMNS = ["ticker","region","actual_shares","actual_entry_date","actual_entry_price_local","currency","source_screen_run","source_order_plan","status","cycle_start_date","ttl_review_due_date","renewal_expiry_date","last_review_date","last_decision","notes"]
DECISION_COLUMNS = ["ticker","region","actual_shares","actual_entry_date","cycle_start_date","ttl_review_due_date","renewal_expiry_date","status_before","status_after","review_type","rank_in_region","rank_buffer_pass","residual_raw","residual_condition_pass","price","sma_50","price_above_50dma_pass","composite_pass_count","composite_pass","decision","decision_as_of_date","decision_market","sell_not_before_date","benchmark","warning"]
SELL_COLUMNS = ["ticker","region","actual_shares","currency","status_before_decision","decision_type","decision_as_of_date","decision_market","sell_not_before_date","recommended_action","order_type","reason","rank_condition_pass","residual_condition_pass","price_above_50dma_pass","composite_pass_count","benchmark","reference_price_local","reference_price_date","warning"]


def _git_commit_hash() -> str:
    try: return subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()
    except Exception: return "unknown"

def _run_folder(root: str | Path, now: pd.Timestamp | None = None) -> Path:
    out = Path(root) / (now or pd.Timestamp.now("UTC")).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=False); return out

def _block(msg: str) -> None:
    raise RuntimeError(f"DATA_BLOCKED — no trading recommendation generated.\n{msg}\nResolve the data or ledger issue and rerun the TTL Manager.")

def _normalize_region(x: str) -> str:
    r = str(x).upper().strip()
    if r not in {"US","JP"}: raise ValueError(f"invalid region: {x}")
    return r

def _validate_ledger(df: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    missing = [c for c in LEDGER_COLUMNS if c not in df.columns]
    if missing: _block(f"Ledger missing required columns: {missing}")
    d = df.copy()
    if d.empty: _block("Ledger has no holdings rows.")
    if d["ticker"].astype(str).duplicated().any(): _block("Duplicate ticker rows in live holdings ledger.")
    d["region"] = d["region"].map(_normalize_region)
    d["actual_shares"] = pd.to_numeric(d["actual_shares"], errors="coerce")
    if d["actual_shares"].isna().any() or (d["actual_shares"] <= 0).any(): _block("actual_shares must be known and positive.")
    for c in ["actual_entry_date","cycle_start_date","ttl_review_due_date","renewal_expiry_date","last_review_date"]:
        d[c] = pd.to_datetime(d[c], errors="coerce")
    if d["actual_entry_date"].isna().any() or d["cycle_start_date"].isna().any(): _block("Entry/cycle dates are missing or invalid.")
    if (d["actual_entry_date"] > today).any() or (d["cycle_start_date"] > today).any(): _block("Ledger contains future entry/cycle dates.")
    d["status"] = d["status"].astype(str).str.upper().str.strip()
    return d

def create_fill_template(order_plan_dir: str | Path, output_root: str | Path) -> Path:
    src = Path(order_plan_dir) / "buy_order_plan.csv"
    if not src.exists(): src = Path(order_plan_dir) / "order_plan.csv"
    if not src.exists(): raise FileNotFoundError("buy_order_plan.csv/order_plan.csv not found")
    plan = pd.read_csv(src)
    out = _run_folder(output_root)
    rows = pd.DataFrame({
        "ticker": plan.get("ticker", ""), "region": plan.get("region", ""),
        "planned_shares_reference_only": plan.get("planned_shares", plan.get("shares", "")),
        "actual_shares": "", "actual_entry_date": "", "actual_entry_price_local": "",
        "currency": plan.get("currency", ""), "source_order_plan": str(src),
        "notes": "Fill actual_* from broker confirmations; do not copy planned shares blindly.",
    })
    rows.to_csv(out/"actual_fills_template.csv", index=False); return out

def initialize_ledger(actual_fills: str | Path, ledger_path: str | Path, ttl_days: int = TTL_DAYS, renewal_days: int = RENEWAL_DAYS) -> None:
    f = pd.read_csv(actual_fills)
    required = ["ticker","region","actual_shares","actual_entry_date","actual_entry_price_local","currency"]
    missing = [c for c in required if c not in f.columns]
    if missing: _block(f"actual fills missing required columns: {missing}")
    if f[["actual_shares","actual_entry_date","actual_entry_price_local"]].isna().any().any(): _block("Actual fill fields are incomplete; planned values are not accepted.")
    today = pd.Timestamp.now("UTC").normalize().tz_localize(None)
    if f["ticker"].astype(str).duplicated().any(): _block("Duplicate ticker in fills.")
    f["actual_shares"] = pd.to_numeric(f["actual_shares"], errors="coerce")
    if f["actual_shares"].isna().any() or (f["actual_shares"] <= 0).any(): _block("Invalid actual_shares.")
    entry = pd.to_datetime(f["actual_entry_date"], errors="coerce")
    if entry.isna().any() or (entry > today).any(): _block("Invalid or future actual_entry_date.")
    rows = []
    for i, r in f.iterrows():
        e = entry.iloc[i].normalize(); rows.append({
            "ticker": alpha.normalize_yfinance_ticker(r.ticker), "region": _normalize_region(r.region), "actual_shares": r.actual_shares,
            "actual_entry_date": e.date().isoformat(), "actual_entry_price_local": r.actual_entry_price_local, "currency": r.currency,
            "source_screen_run": r.get("source_screen_run", ""), "source_order_plan": r.get("source_order_plan", ""), "status": "ACTIVE",
            "cycle_start_date": e.date().isoformat(), "ttl_review_due_date": (e + pd.Timedelta(days=ttl_days)).date().isoformat(),
            "renewal_expiry_date": "", "last_review_date": "", "last_decision": "", "notes": r.get("notes", "")})
    Path(ledger_path).parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows, columns=LEDGER_COLUMNS).to_csv(ledger_path, index=False)

def _benchmark_status(prices: pd.DataFrame, mode: dict) -> dict:
    status = {}
    for region in ("US","JP"):
        cands = list(mode.get(region, ())); avail = [c for c in cands if c in prices.columns and prices[c].dropna().shape[0] >= 252]
        used = alpha._resolve_benchmark(prices, region, mode)
        status[region] = {"candidates": cands, "available": avail, "used": used, "status": "ok" if used else "missing"}
    return status

def _next_session(prices: pd.DataFrame, asof) -> str:
    idx = prices.index[prices.index > pd.Timestamp(asof)]
    return pd.Timestamp(idx[0]).date().isoformat() if len(idx) else (pd.Timestamp(asof)+pd.offsets.BDay(1)).date().isoformat()

def _health(prices, us, jp, dt, ticker, region, variant, mode):
    h = alpha._health_row(prices, us, jp, dt, ticker, region, variant, mode)
    s = alpha.asof_prices(prices, dt).ffill()[ticker].dropna() if ticker in prices.columns else pd.Series(dtype=float)
    if len(s) < 252: _block(f"Insufficient price history for {ticker}.")
    if not np.isfinite(h.get("residual_score", np.nan)): _block(f"Composite residual cannot be calculated for {ticker}.")
    bench = alpha._resolve_benchmark(alpha.asof_prices(prices, dt).ffill(), region, mode)
    if not bench: _block(f"{region} benchmark unavailable; no silent substitution allowed.")
    return {**h, "price": float(s.iloc[-1]), "sma_50": float(s.tail(50).mean()), "benchmark": bench}

def run_ttl_manager(ledger_path: str | Path, output_root: str | Path = DEFAULT_OUTPUT_ROOT, ttl_days: int = TTL_DAYS, renewal_days: int = RENEWAL_DAYS, weekly_review_day: str = WEEKLY_REVIEW_DAY, data_lookback_months: int = DATA_LOOKBACK_MONTHS, residual_ratio: int = 60, total_holdings: int = 12, prices: pd.DataFrame | None = None, us: list[str] | None = None, jp: list[str] | None = None, downloader=None) -> Path:
    t0=time.time(); started=pd.Timestamp.now("UTC"); today=started.normalize().tz_localize(None)
    if not Path(ledger_path).exists(): _block("Live holdings ledger not found.")
    ledger=_validate_ledger(pd.read_csv(ledger_path), today)
    variant=build_live_variant(residual_ratio,total_holdings); variant.update({"ttl_days":ttl_days,"renewal_protocol":"composite","renewal_extension_days":renewal_days})
    if us is None or jp is None: us,jp=alpha.get_live_universe()
    us=[alpha.normalize_yfinance_ticker(t) for t in us]; jp=[alpha.normalize_yfinance_ticker(t) for t in jp]
    mode=alpha.build_benchmark_modes()["broad_default"]; benches=sorted({x for vals in mode.values() for x in vals})
    requested=list(dict.fromkeys([*us,*jp,*ledger.ticker.astype(str),*benches])); failures=pd.DataFrame(columns=["ticker","reason"])
    start=(today-pd.DateOffset(months=data_lookback_months)).date().isoformat(); end=today.date().isoformat()
    if prices is None: prices, failures, requested = alpha.download_live_universe_prices(requested, start, end, batch_size=80, downloader=downloader)
    prices=prices.sort_index(); status=_benchmark_status(prices,mode)
    if status["US"]["status"]!="ok" or status["JP"]["status"]!="ok": _block(f"Benchmark unavailable: {status}")
    dq, insufficient, excluded, usable = alpha.build_live_data_quality_report(prices, requested, us, jp, failures, start, end, min_history=252, benchmark_status={k: json.dumps(v, default=str) for k,v in status.items()})
    us_usable=[t for t in us if t in usable or t in set(ledger.ticker)]; jp_usable=[t for t in jp if t in usable or t in set(ledger.ticker)]
    decisions=[]; sells=[]; recons=[]; proposed=ledger.copy();
    for _c in ["status","ttl_review_due_date","renewal_expiry_date","last_review_date","last_decision","notes"]:
        if _c in proposed.columns: proposed[_c] = proposed[_c].astype(object)
    data_end=prices.index.max()
    for idx,row in ledger.iterrows():
        status_before=row.status; ticker=row.ticker; region=row.region
        if status_before in {"SELL_PENDING","RENEWAL_FAILED_SELL_PENDING"}: _block(f"{ticker} is already sell pending; refusing duplicate sell handling.")
        cycle=row.cycle_start_date; age=(today-cycle).days; due=cycle+pd.Timedelta(days=ttl_days); expiry=cycle+pd.Timedelta(days=ttl_days+renewal_days)
        review_type="not_due"; status_after=status_before; decision="HOLD_UNTIL_TTL_REVIEW"; h={"market_rank":np.nan,"rank_pass":False,"residual_score":np.nan,"residual_pass":False,"price":np.nan,"sma_50":np.nan,"price_above_50dma":False,"composite_count":0,"composite_pass":False,"benchmark":""}; sell_nb=""; warning=""
        if age < ttl_days and status_before=="ACTIVE": pass
        elif age >= ttl_days + renewal_days or (status_before=="RENEWED" and today >= expiry):
            review_type="renewal_expiry"; status_after="RECONSTITUTION_REQUIRED"; decision="RUN_LIVE_SCREENER_FOR_NEW_CYCLE"; recons.append({"ticker":ticker,"region":region,"actual_shares":row.actual_shares,"cycle_start_date":cycle.date().isoformat(),"renewal_expiry_date":expiry.date().isoformat(),"decision":"RECONSTITUTION_REQUIRED","instruction":"Run alpha_engine_live_screener.py and alpha_engine_order_planner.py; do not treat as second Renew30."})
        else:
            h=_health(prices, us_usable, jp_usable, data_end, ticker, region, variant, mode)
            if status_before=="ACTIVE":
                review_type="ttl_review"; ok=bool(h["composite_pass"]); status_after="RENEWED" if ok else "SELL_PENDING"; decision="RENEW_30_DAYS" if ok else "SELL_MANUAL_AND_HOLD_CASH"
            elif status_before=="RENEWED":
                review_type="weekly_renewal_review"; ok=bool(h["composite_pass"]); status_after="RENEWED" if ok else "RENEWAL_FAILED_SELL_PENDING"; decision="CONTINUE_WEEKLY_MONITORING" if ok else "SELL_MANUAL_AND_HOLD_CASH"
            else: _block(f"Unsupported status for live TTL review: {status_before}")
            sell_nb = "" if h["composite_pass"] else _next_session(prices, data_end)
            if not h["composite_pass"]:
                sells.append({"ticker":ticker,"region":region,"actual_shares":row.actual_shares,"currency":row.currency,"status_before_decision":status_before,"decision_type":review_type,"decision_as_of_date":pd.Timestamp(data_end).date().isoformat(),"decision_market":region,"sell_not_before_date":sell_nb,"recommended_action":"SELL_MANUAL_MARKET_NEXT_TRADABLE_SESSION","order_type":"MANUAL_MARKET","reason":"Composite failed; 2 of 3 audited conditions not met. Sold names are not immediately replaced; cash is held until reconstitution.","rank_condition_pass":h["rank_pass"],"residual_condition_pass":h["residual_pass"],"price_above_50dma_pass":h["price_above_50dma"],"composite_pass_count":h["composite_count"],"benchmark":h["benchmark"],"reference_price_local":h["price"],"reference_price_date":pd.Timestamp(data_end).date().isoformat(),"warning":"Manual proposal only. Verify brokerage holdings and next tradable session; no automatic order is sent."})
        decisions.append({"ticker":ticker,"region":region,"actual_shares":row.actual_shares,"actual_entry_date":row.actual_entry_date.date().isoformat(),"cycle_start_date":cycle.date().isoformat(),"ttl_review_due_date":due.date().isoformat(),"renewal_expiry_date":expiry.date().isoformat(),"status_before":status_before,"status_after":status_after,"review_type":review_type,"rank_in_region":h["market_rank"],"rank_buffer_pass":h["rank_pass"],"residual_raw":h["residual_score"],"residual_condition_pass":h["residual_pass"],"price":h["price"],"sma_50":h["sma_50"],"price_above_50dma_pass":h["price_above_50dma"],"composite_pass_count":h["composite_count"],"composite_pass":h["composite_pass"],"decision":decision,"decision_as_of_date":pd.Timestamp(data_end).date().isoformat(),"decision_market":region,"sell_not_before_date":sell_nb,"benchmark":h["benchmark"],"warning":warning})
        proposed.loc[idx,"status"]=status_after; proposed.loc[idx,"ttl_review_due_date"]=due.date().isoformat(); proposed.loc[idx,"renewal_expiry_date"]=expiry.date().isoformat() if status_after in {"RENEWED","RECONSTITUTION_REQUIRED"} else row.renewal_expiry_date; proposed.loc[idx,"last_review_date"]=pd.Timestamp(data_end).date().isoformat(); proposed.loc[idx,"last_decision"]=decision
    out=_run_folder(output_root, started)
    dec=pd.DataFrame(decisions,columns=DECISION_COLUMNS); dec.to_csv(out/"ttl_review_decisions.csv",index=False); dec[dec.review_type=="weekly_renewal_review"].to_csv(out/"renewal_decisions.csv",index=False)
    pd.DataFrame(sells,columns=SELL_COLUMNS).to_csv(out/"sell_order_plan.csv",index=False); pd.DataFrame(recons).to_csv(out/"reconstitution_required.csv",index=False); proposed.to_csv(out/"live_holdings_ledger_proposed.csv",index=False); dq.to_csv(out/"data_quality.csv",index=False); failures.to_csv(out/"download_failures.csv",index=False)
    meta={"mode":"live_ttl_manager","git_commit_hash":_git_commit_hash(),"run_started_at":started.isoformat(),"runtime_seconds":round(time.time()-t0,2),"ttl_day_convention":TTL_DAY_CONVENTION,"ttl_days":ttl_days,"renewal_days":renewal_days,"weekly_review_day":weekly_review_day,"ledger_path":str(ledger_path),"ledger_row_count":len(ledger),"reviewed_ticker_count":int((dec.review_type!="not_due").sum()),"renewed_ticker_count":int((dec.status_after=="RENEWED").sum()),"sell_pending_ticker_count":int(dec.status_after.isin(["SELL_PENDING","RENEWAL_FAILED_SELL_PENDING"]).sum()),"reconstitution_required_ticker_count":int((dec.status_after=="RECONSTITUTION_REQUIRED").sum()),"data_start":str(prices.index.min().date()) if len(prices.index) else "","data_end":str(prices.index.max().date()) if len(prices.index) else "","benchmark_status":status,"decision_market_note":"US and JP close times differ; decisions use available as-of close data and sell_not_before_date is the next available session in downloaded market calendar."}
    (out/"metadata.json").write_text(json.dumps(meta,indent=2,default=str),encoding="utf-8"); _write_report(out,meta,dec,pd.DataFrame(sells),pd.DataFrame(recons),dq); return out

def _md(df): return alpha._markdown_table(df) if isinstance(df,pd.DataFrame) and not df.empty else "(no rows)"
def _write_report(out, meta, dec, sells, recons, dq):
    report=f"""# Alpha Engine Live TTL Manager Report\n\n**LIVE TTL MANAGER — not automatic trading**\n\n- Run started: {meta['run_started_at']}\n- Git commit hash: {meta['git_commit_hash']}\n- TTL days / Renewal days: {meta['ttl_days']} / {meta['renewal_days']}\n- Day convention: {meta['ttl_day_convention']}\n- Current live holdings rows: {meta['ledger_row_count']}\n- Reviewed ticker count: {meta['reviewed_ticker_count']}\n\n## Reviewed tickers\n{_md(dec[dec.review_type!='not_due'][['ticker','region','status_before','status_after','review_type','decision','decision_as_of_date','decision_market','sell_not_before_date']])}\n\n## Renewed tickers\n{_md(dec[dec.status_after=='RENEWED'][['ticker','region','composite_pass_count','decision']])}\n\n## Sell candidates\n{_md(sells)}\n\n## Renewal degradation sell candidates\n{_md(sells[sells.decision_type=='weekly_renewal_review'] if not sells.empty else sells)}\n\n## 120-day expiry / reconstitution required\n{_md(recons)}\n\n## Next manual work\n- Verify shares, prices, and execution status in the brokerage screen before any order.\n- Manual sell proposals use actual ledger shares and never assume same-close execution.\n- Sold tickers are not immediately replenished; hold cash until the 120-day reconstitution workflow.\n- For reconstitution, run `alpha_engine_live_screener.py` and then `alpha_engine_order_planner.py`; this TTL manager does not optimize or place replacement orders.\n- US and Japan market close / timezone differences require checking each `decision_as_of_date`, `decision_market`, and `sell_not_before_date`.\n\n## Data quality and benchmark status\n```json\n{json.dumps(meta['benchmark_status'], indent=2, default=str)}\n```\n\n{_md(dq)}\n"""
    (out/"ttl_review_report.md").write_text(report,encoding="utf-8")

def apply_execution_confirmation(ledger_path: str | Path | None = None, execution_confirmation: str | Path | None = None) -> None:
    """Apply explicit user-confirmed executions to the ledger.

    Required confirmation columns: ticker, action, execution_status, executed_shares,
    execution_date. SELL rows reduce shares only for FILLED/PARTIAL_FILLED rows with
    positive executed_shares; CANCELED/REJECTED/UNFILLED rows never change holdings.
    This function intentionally does not infer executions from order plans.
    """
    if not ledger_path or not execution_confirmation:
        _block("Both --ledger-path and --execution-confirmation are required; no automatic zeroing occurs without confirmed executions.")
    lp, cp = Path(ledger_path), Path(execution_confirmation)
    if not lp.exists() or not cp.exists(): _block("Ledger or execution confirmation file is missing.")
    today = pd.Timestamp.now("UTC").normalize().tz_localize(None)
    ledger = _validate_ledger(pd.read_csv(lp), today)
    conf = pd.read_csv(cp)
    required = ["ticker","action","execution_status","executed_shares","execution_date"]
    missing = [c for c in required if c not in conf.columns]
    if missing: _block(f"execution confirmation missing required columns: {missing}")
    for _, r in conf.iterrows():
        ticker = alpha.normalize_yfinance_ticker(r.ticker)
        if ticker not in set(ledger.ticker): _block(f"execution confirmation references unknown ticker: {ticker}")
        action = str(r.action).upper().strip(); status = str(r.execution_status).upper().strip()
        shares = pd.to_numeric(r.executed_shares, errors="coerce")
        dt = pd.to_datetime(r.execution_date, errors="coerce")
        if pd.isna(dt) or dt > today: _block(f"invalid execution_date for {ticker}")
        if status in {"CANCELED","CANCELLED","REJECTED","UNFILLED","NOT_FILLED"}:
            continue
        if status not in {"FILLED","PARTIAL_FILLED"}: _block(f"unsupported execution_status for {ticker}: {status}")
        if pd.isna(shares) or shares <= 0: _block(f"executed_shares must be positive for confirmed execution: {ticker}")
        idx = ledger.index[ledger.ticker == ticker][0]
        if action == "SELL":
            old_shares = float(ledger.loc[idx, "actual_shares"])
            new_shares = max(0.0, old_shares - float(shares))
            ledger.loc[idx, "actual_shares"] = new_shares
            ledger.loc[idx, "status"] = "SOLD" if new_shares == 0 else "ACTIVE"
            ledger.loc[idx, "last_review_date"] = dt.date().isoformat()
            ledger.loc[idx, "last_decision"] = f"CONFIRMED_{status}_SELL"
        elif action == "BUY":
            _block("BUY execution confirmations for new cycles must be initialized from actual fills; this updater only reduces existing sell confirmations safely.")
        else:
            _block(f"unsupported action for {ticker}: {action}")
    ledger.to_csv(lp, index=False)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--create-fill-template",action="store_true"); ap.add_argument("--initialize-ledger",action="store_true"); ap.add_argument("--apply-execution-confirmation",action="store_true"); ap.add_argument("--order-plan-dir"); ap.add_argument("--actual-fills"); ap.add_argument("--ledger-path"); ap.add_argument("--execution-confirmation"); ap.add_argument("--output-root",default=DEFAULT_OUTPUT_ROOT); ap.add_argument("--ttl-days",type=int,default=TTL_DAYS); ap.add_argument("--renewal-days",type=int,default=RENEWAL_DAYS); ap.add_argument("--weekly-review-day",default=WEEKLY_REVIEW_DAY); ap.add_argument("--data-lookback-months",type=int,default=DATA_LOOKBACK_MONTHS); args=ap.parse_args()
    if args.create_fill_template: print(f"Output directory: {create_fill_template(args.order_plan_dir,args.output_root)}"); return
    if args.initialize_ledger: initialize_ledger(args.actual_fills,args.ledger_path,args.ttl_days,args.renewal_days); print(f"Ledger initialized: {args.ledger_path}"); return
    if args.apply_execution_confirmation: apply_execution_confirmation(args.ledger_path,args.execution_confirmation); return
    if not args.ledger_path: ap.error("--ledger-path is required")
    print(f"Output directory: {run_ttl_manager(args.ledger_path,args.output_root,args.ttl_days,args.renewal_days,args.weekly_review_day,args.data_lookback_months)}")
if __name__ == "__main__": main()
