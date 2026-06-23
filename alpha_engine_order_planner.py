"""Alpha Engine buy order planner.

Creates a human-reviewable initial deployment buy plan from a Live Screener run.
It never places orders and has no brokerage/API trading integration.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

TOTAL_CAPITAL_JPY = 3_000_000
USDJPY_RATE = None
CASH_BUFFER_PCT = 0.01
PRICE_BUFFER_PCT = 0.02
US_ORDER_UNIT = 1
JP_ORDER_UNIT = 1

REQUIRED_PLAN_COLUMNS = [
    "ticker", "region", "rank_in_region", "target_weight", "target_amount_jpy", "currency",
    "reference_price_local", "buffered_price_local", "usd_jpy_rate", "order_unit",
    "planned_shares", "planned_cost_local", "planned_cost_jpy", "actual_weight_after_rounding",
    "allocation_gap_jpy", "action", "note",
]


@dataclass(frozen=True)
class UsdJpyRate:
    rate: float
    source: str
    fetched_at: str
    manual: bool


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def fetch_usd_jpy_rate() -> UsdJpyRate:
    """Fetch USD/JPY from Yahoo Finance chart endpoint, or raise explicitly."""
    url = "https://query1.finance.yahoo.com/v8/finance/chart/JPY=X?range=1d&interval=1m"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        result = payload["chart"]["result"][0]
        prices = result["indicators"]["quote"][0]["close"]
        timestamps = result.get("timestamp") or []
        valid = [(ts, px) for ts, px in zip(timestamps, prices) if px is not None and float(px) > 0]
        if not valid:
            raise ValueError("no positive USD/JPY close values returned")
        ts, rate = valid[-1]
        fetched_at = pd.to_datetime(int(ts), unit="s", utc=True).isoformat()
        return UsdJpyRate(float(rate), "Yahoo Finance JPY=X", fetched_at, False)
    except Exception as exc:
        raise RuntimeError("USD/JPY rate was not provided and automatic retrieval failed; rerun with --usd-jpy-rate.") from exc


def resolve_usd_jpy_rate(usd_jpy_rate: float | None) -> UsdJpyRate:
    if usd_jpy_rate is not None:
        if usd_jpy_rate <= 0:
            raise ValueError("USD/JPY rate must be positive.")
        return UsdJpyRate(float(usd_jpy_rate), "manual --usd-jpy-rate", pd.Timestamp.now("UTC").isoformat(), True)
    return fetch_usd_jpy_rate()


def _find_price_column(df: pd.DataFrame) -> str:
    candidates = [
        "reference_price_local", "latest_price", "last_price", "close", "Close", "price", "Price",
        "current_price", "Current_Price", "asof_price",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError("selected_tickers.csv does not contain a usable latest/reference price column.")


def _invalid_reference_price_message(ticker: str) -> str:
    return (
        f"Invalid reference_price_local for {ticker}.\n"
        "Expected one numeric latest price from Live Screener.\n"
        "Re-run the Live Screener with a valid price output."
    )


def _validate_reference_price_local(value: Any, ticker: str) -> float:
    if isinstance(value, str):
        text = value.strip()
        if not text or "\n" in text or "dtype:" in text or "Name:" in text:
            raise ValueError(_invalid_reference_price_message(ticker))
        value = text
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ValueError(_invalid_reference_price_message(ticker)) from None
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(_invalid_reference_price_message(ticker))
    return numeric


def _validate_inputs(selected: pd.DataFrame, total_capital_jpy: float, cash_buffer_pct: float, price_buffer_pct: float, us_order_unit: int, jp_order_unit: int) -> None:
    for col in ("ticker", "region", "Weight"):
        if col not in selected.columns:
            raise ValueError(f"selected_tickers.csv is missing required column: {col}")
    if total_capital_jpy <= 0:
        raise ValueError("TOTAL_CAPITAL_JPY must be positive.")
    if not 0 <= cash_buffer_pct < 1:
        raise ValueError("CASH_BUFFER_PCT must be >= 0 and < 1.")
    if price_buffer_pct < 0:
        raise ValueError("PRICE_BUFFER_PCT must be >= 0.")
    if us_order_unit <= 0 or jp_order_unit <= 0:
        raise ValueError("Order units must be positive integers.")


def build_buy_order_plan(
    screen_run_dir: str | Path,
    total_capital_jpy: float = TOTAL_CAPITAL_JPY,
    usd_jpy_rate: float | None = USDJPY_RATE,
    cash_buffer_pct: float = CASH_BUFFER_PCT,
    price_buffer_pct: float = PRICE_BUFFER_PCT,
    us_order_unit: int = US_ORDER_UNIT,
    jp_order_unit: int = JP_ORDER_UNIT,
    output_dir: str | Path | None = None,
) -> Path:
    screen_run_dir = Path(screen_run_dir)
    selected_path = screen_run_dir / "selected_tickers.csv"
    if not selected_path.exists():
        raise FileNotFoundError(f"selected_tickers.csv not found in {screen_run_dir}")
    selected = pd.read_csv(selected_path)
    _validate_inputs(selected, total_capital_jpy, cash_buffer_pct, price_buffer_pct, us_order_unit, jp_order_unit)
    price_col = _find_price_column(selected)
    rate = resolve_usd_jpy_rate(usd_jpy_rate)
    deployable = float(total_capital_jpy) * (1 - float(cash_buffer_pct))

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for _, row in selected.iterrows():
        ticker = str(row["ticker"])
        region = str(row["region"]).upper()
        weight = float(row["Weight"])
        target_jpy = deployable * weight
        reference_price = _validate_reference_price_local(row[price_col], ticker)
        note_parts: list[str] = []
        if region == "US":
            currency = "USD"; order_unit = int(us_order_unit)
            buffered = reference_price * (1 + price_buffer_pct)
            target_local = target_jpy / rate.rate
            shares = math.floor(target_local / buffered / order_unit) * order_unit
            cost_local = shares * reference_price
            cost_jpy = cost_local * rate.rate
        elif region == "JP":
            currency = "JPY"; order_unit = int(jp_order_unit)
            buffered = reference_price * (1 + price_buffer_pct)
            shares = math.floor(target_jpy / buffered / order_unit) * order_unit
            cost_local = shares * reference_price
            cost_jpy = cost_local
        else:
            raise ValueError(f"Unsupported region for {ticker}: {region}")
        if shares == 0:
            msg = f"WARNING: planned_shares=0 for {ticker}; review target amount, price, and order unit."
            warnings.append(msg); note_parts.append(msg)
        actual_weight = cost_jpy / deployable if deployable else 0.0
        rows.append({
            "ticker": ticker, "region": region, "rank_in_region": row.get("rank_in_region", ""),
            "target_weight": weight, "target_amount_jpy": target_jpy, "currency": currency,
            "reference_price_local": reference_price, "buffered_price_local": buffered,
            "usd_jpy_rate": rate.rate, "order_unit": order_unit, "planned_shares": int(shares),
            "planned_cost_local": cost_local, "planned_cost_jpy": cost_jpy,
            "actual_weight_after_rounding": actual_weight, "allocation_gap_jpy": target_jpy - cost_jpy,
            "action": "BUY", "note": "; ".join(note_parts),
        })
    plan = pd.DataFrame(rows, columns=REQUIRED_PLAN_COLUMNS)
    planned_us = float(plan.loc[plan.region == "US", "planned_cost_jpy"].sum())
    planned_jp = float(plan.loc[plan.region == "JP", "planned_cost_jpy"].sum())
    planned_total = planned_us + planned_jp
    residual_cash = float(total_capital_jpy) - planned_total
    if planned_total > deployable + 1e-6:
        raise RuntimeError("Planned total cost exceeds deployable capital.")
    if residual_cash < -1e-6:
        raise RuntimeError("Residual cash would be negative.")

    out = Path(output_dir) if output_dir else screen_run_dir
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([{
        "total_capital_jpy": float(total_capital_jpy),
        "cash_buffer_jpy": float(total_capital_jpy) * float(cash_buffer_pct),
        "deployable_capital_jpy": deployable,
        "planned_us_cost_jpy": planned_us,
        "planned_jp_cost_jpy": planned_jp,
        "planned_total_cost_jpy": planned_total,
        "residual_cash_jpy": residual_cash,
        "residual_cash_pct": residual_cash / float(total_capital_jpy),
        "USDJPY_RATE": rate.rate,
        "PRICE_BUFFER_PCT": float(price_buffer_pct),
        "US_ORDER_UNIT": int(us_order_unit),
        "JP_ORDER_UNIT": int(jp_order_unit),
    }])
    meta = {
        "mode": "buy_order_plan_only", "screen_run_dir": str(screen_run_dir),
        "git_commit_hash": _git_commit_hash(), "run_datetime": pd.Timestamp.now("UTC").isoformat(),
        "usd_jpy_rate": rate.__dict__, "warnings": warnings,
        "no_automatic_trading": True,
    }
    plan.to_csv(out / "buy_order_plan.csv", index=False)
    summary.to_csv(out / "buy_order_summary.csv", index=False)
    (out / "order_plan_metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    _write_report(out, meta, plan, summary.iloc[0].to_dict(), rate)
    return out


def _md_table(df: pd.DataFrame) -> str:
    try:
        import alpha_engine_backtest as alpha
        return alpha._markdown_table(df) if not df.empty else "(no rows)"
    except Exception:
        return df.to_markdown(index=False) if not df.empty else "(no rows)"


def _write_report(out: Path, meta: dict, plan: pd.DataFrame, summary: dict, rate: UsdJpyRate) -> None:
    us_cols = ["ticker", "planned_shares", "planned_cost_local", "planned_cost_jpy", "actual_weight_after_rounding"]
    report = f"""# Alpha Engine Buy Order Planner Report

**ORDER PLAN ONLY — not automatic trading**

- Source Live Screener run folder: {meta['screen_run_dir']}
- Git commit hash: {meta['git_commit_hash']}
- Run datetime: {meta['run_datetime']}
- Total buy budget JPY: {summary['total_capital_jpy']:.0f}
- Cash buffer JPY: {summary['cash_buffer_jpy']:.0f}
- Deployable capital JPY: {summary['deployable_capital_jpy']:.0f}
- USD/JPY rate: {rate.rate:.6f} ({'manual input' if rate.manual else rate.source}, timestamp: {rate.fetched_at})
- Price buffer pct: {summary['PRICE_BUFFER_PCT']:.4f}
- Expected residual cash JPY: {summary['residual_cash_jpy']:.0f}

## US planned buys
{_md_table(plan[plan.region == 'US'][us_cols])}

## Japan planned buys
{_md_table(plan[plan.region == 'JP'][us_cols])}

## Full plan
{_md_table(plan)}

## Warnings / required review
{chr(10).join('- ' + w for w in meta.get('warnings', [])) if meta.get('warnings') else '- None'}

Prices are reference values only. Reconfirm all prices, tickers, quantities, currencies, fees, taxes, and account constraints on the brokerage screen immediately before submitting any order manually.

This file is an order plan for human review only. It does not place trades, connect to a brokerage API, or automate execution.
"""
    (out / "buy_order_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Alpha Engine buy order planner (plan only; no automatic trading)")
    ap.add_argument("--screen-run-dir", required=True)
    ap.add_argument("--total-capital-jpy", type=float, default=TOTAL_CAPITAL_JPY)
    ap.add_argument("--usd-jpy-rate", type=float, default=USDJPY_RATE)
    ap.add_argument("--cash-buffer-pct", type=float, default=CASH_BUFFER_PCT)
    ap.add_argument("--price-buffer-pct", type=float, default=PRICE_BUFFER_PCT)
    ap.add_argument("--us-order-unit", type=int, default=US_ORDER_UNIT)
    ap.add_argument("--jp-order-unit", type=int, default=JP_ORDER_UNIT)
    ap.add_argument("--output-dir")
    args = ap.parse_args()
    out = build_buy_order_plan(**vars(args))
    print(f"Buy order plan output directory: {out}")


if __name__ == "__main__":
    main()
