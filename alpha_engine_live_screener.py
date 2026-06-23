"""Lightweight Alpha Engine live Residual Momentum screener.

This module ranks the current US and Japan equity universes only. It does not run a
backtest, TTL renewal, performance metrics, trade history reconstruction, or any
automatic trading workflow.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

import alpha_engine_backtest as alpha

LOG = logging.getLogger("alpha_live_screener")

RESIDUAL_RATIO = 60
TOTAL_HOLDINGS = 12
LOOKBACK_MONTHS = 18
DEFAULT_OUTPUT_ROOT = "artifacts/live_screening_runs"


def validate_total_holdings(total_holdings: int) -> tuple[int, int]:
    if int(total_holdings) != total_holdings or total_holdings <= 0:
        raise ValueError("TOTAL_HOLDINGS must be a positive even integer.")
    if total_holdings % 2:
        raise ValueError(f"TOTAL_HOLDINGS must be even for equal US/JP allocation; got {total_holdings}.")
    return total_holdings // 2, total_holdings // 2


def build_live_variant(residual_ratio: int, total_holdings: int) -> dict:
    if residual_ratio < 0 or residual_ratio > 100:
        raise ValueError("RESIDUAL_RATIO must be between 0 and 100.")
    us_n, jp_n = validate_total_holdings(int(total_holdings))
    w = round(int(residual_ratio) / 100, 2)
    return {
        "name": f"Residual_{int(residual_ratio):02d}_N{int(total_holdings)}",
        "base_weight": round(1 - w, 2),
        "residual_weight": w,
        "vcp_weight": 0.0,
        "total_holdings": int(total_holdings),
        "us_holdings": us_n,
        "jp_holdings": jp_n,
    }


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _run_folder(output_root: str | Path, residual_ratio: int, total_holdings: int, now: pd.Timestamp | None = None) -> Path:
    ts = (now or pd.Timestamp.now("UTC")).strftime("%Y%m%dT%H%M%SZ")
    return Path(output_root) / f"{ts}_residual{int(residual_ratio)}_n{int(total_holdings)}"


def _benchmark_status(prices: pd.DataFrame, benchmark_mode: dict) -> dict:
    rows = {}
    for region in ("US", "JP"):
        candidates = list(benchmark_mode.get(region, ()))
        available = [c for c in candidates if c in prices.columns and prices[c].dropna().shape[0] >= 252]
        used = alpha._resolve_benchmark(prices[available] if available else prices, region, benchmark_mode)
        rows[region] = {"candidates": candidates, "available": available, "used": used, "status": "ok" if used else "missing"}
    return rows


def _score_region(prices: pd.DataFrame, tickers: list[str], region: str, as_of_date, variant: dict, benchmark_mode: dict, selected_n: int) -> pd.DataFrame:
    base = alpha.score_universe(prices, [t for t in tickers if t in prices.columns], as_of_date)
    residual = alpha.compute_residual_momentum_score(prices, base.index, as_of_date, region, benchmark_mode=benchmark_mode, method="simple")
    scored = alpha.combine_residual_score(base, residual, variant).copy()
    hist = alpha.asof_prices(prices, as_of_date).ffill()
    bench = alpha._resolve_benchmark(hist, region, benchmark_mode)
    scored["ticker"] = scored.index
    scored["region"] = region
    scored["benchmark"] = bench
    scored["base_momentum_score"] = scored["base_score"]
    scored["composite_score"] = scored["Final_Score"]
    scored["momentum_3m"] = [alpha._window_return(hist, t, 63) for t in scored.index]
    scored["momentum_6m"] = [alpha._window_return(hist, t, 126) for t in scored.index]
    scored["momentum_12m"] = [alpha._window_return(hist, t, 252) for t in scored.index]
    scored["data_available"] = True
    scored["adopted"] = False
    scored["rank_in_region"] = np.arange(1, len(scored) + 1)
    scored["total_rank"] = scored["rank_in_region"]
    if not scored.empty:
        scored.loc[scored.head(selected_n).index, "adopted"] = True
    return scored.reset_index(drop=True)


def run_live_screening(prices: pd.DataFrame | None = None, us: list[str] | None = None, jp: list[str] | None = None, residual_ratio: int = RESIDUAL_RATIO, total_holdings: int = TOTAL_HOLDINGS, output_root: str | Path = DEFAULT_OUTPUT_ROOT, downloader=None) -> Path:
    """Run one current-date screening pass and write all live screening artifacts."""
    t0 = time.time(); started = pd.Timestamp.now("UTC")
    variant = build_live_variant(residual_ratio, total_holdings)
    us_n, jp_n = variant["us_holdings"], variant["jp_holdings"]
    if us is None or jp is None:
        us, jp = alpha.get_live_universe()
    us = [alpha.normalize_yfinance_ticker(t) for t in us]
    jp = [alpha.normalize_yfinance_ticker(t) for t in jp]
    mode = alpha.build_benchmark_modes()["broad_default"]
    bench_tickers = sorted({x for vals in mode.values() for x in vals})
    end = pd.Timestamp.today(tz="UTC").date().isoformat()
    start = (pd.Timestamp(end) - pd.DateOffset(months=LOOKBACK_MONTHS)).date().isoformat()
    requested = list(dict.fromkeys([*us, *jp, *bench_tickers]))
    failures = pd.DataFrame(columns=["ticker", "reason"])
    if prices is None:
        prices, failures, requested = alpha.download_live_universe_prices(requested, start, end, batch_size=80, downloader=downloader)
    prices = prices.sort_index()
    status = _benchmark_status(prices, mode)
    if status["JP"]["status"] != "ok":
        raise RuntimeError(f"JP benchmark unavailable; refusing to silently substitute. candidates={status['JP']['candidates']} available={status['JP']['available']}")
    if status["US"]["status"] != "ok":
        raise RuntimeError(f"US benchmark unavailable; candidates={status['US']['candidates']} available={status['US']['available']}")
    data_end = prices.index.max() if len(prices.index) else pd.Timestamp(end)
    dq, insufficient, excluded, usable = alpha.build_live_data_quality_report(prices, requested, us, jp, failures, start, end, min_history=252, benchmark_status={k: json.dumps(v, default=str) for k, v in status.items()})
    us_usable = [t for t in us if t in usable]
    jp_usable = [t for t in jp if t in usable]
    if len(us_usable) < us_n or len(jp_usable) < jp_n:
        raise RuntimeError(f"Insufficient usable universe for requested allocation: US {len(us_usable)}/{us_n}, JP {len(jp_usable)}/{jp_n}.")
    us_rank = _score_region(prices, us_usable, "US", data_end, variant, mode, us_n)
    jp_rank = _score_region(prices, jp_usable, "JP", data_end, variant, mode, jp_n)
    selected = pd.concat([us_rank[us_rank.adopted], jp_rank[jp_rank.adopted]], ignore_index=True)
    inv = 1 / selected["Volatility"].astype(float)
    selected["Weight"] = inv / inv.sum()
    weights = selected[["ticker", "region", "Weight", "Volatility", "composite_score", "base_momentum_score", "residual_score", "benchmark"]].copy()
    score_components = pd.concat([us_rank, jp_rank], ignore_index=True)
    quality_rows = []
    for t in requested:
        n = int(prices[t].dropna().shape[0]) if t in prices.columns else 0
        quality_rows.append({"ticker": t, "downloaded": t in prices.columns, "observations": n, "data_available": t in usable, "excluded": t in excluded, "insufficient_history": t in insufficient})
    out = _run_folder(output_root, residual_ratio, total_holdings, started); out.mkdir(parents=True, exist_ok=False)
    us_rank.to_csv(out / "ranked_candidates_us.csv", index=False); jp_rank.to_csv(out / "ranked_candidates_jp.csv", index=False)
    selected.to_csv(out / "selected_tickers.csv", index=False); weights.to_csv(out / "adopted_weights.csv", index=False)
    score_components.to_csv(out / "score_components.csv", index=False); pd.DataFrame(quality_rows).to_csv(out / "data_quality.csv", index=False); failures.to_csv(out / "download_failures.csv", index=False)
    runtime = round(time.time() - t0, 2)
    meta = {"mode": "live_screening", "run_datetime": started.isoformat(), "git_commit_hash": _git_commit_hash(), "residual_ratio": int(residual_ratio), "total_holdings": int(total_holdings), "us_holdings": us_n, "jp_holdings": jp_n, "data_fetch_start": start, "data_fetch_end": end, "data_end_date": str(pd.Timestamp(data_end).date()), "runtime_seconds": runtime, "benchmarks": {k: v["used"] for k, v in status.items()}, "benchmark_status": status, "requested_ticker_count": len(requested), "successful_ticker_count": len([t for t in requested if t in prices.columns]), "excluded_ticker_count": len(excluded), "output_folder": str(out)}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    _write_report(out, meta, us_rank, jp_rank, selected, weights, failures, dq)
    LOG.info("Live screening output: %s", out)
    return out


def _md_table(df: pd.DataFrame) -> str:
    return alpha._markdown_table(df) if not df.empty else "(no rows)"


def _write_report(out: Path, meta: dict, us_rank: pd.DataFrame, jp_rank: pd.DataFrame, selected: pd.DataFrame, weights: pd.DataFrame, failures: pd.DataFrame, dq: pd.DataFrame) -> None:
    report = f"""# Alpha Engine Live Screener Report\n\n**LIVE SCREENING — not automatic trading**\n\n- residual_ratio: {meta['residual_ratio']}\n- total_holdings: {meta['total_holdings']} (US {meta['us_holdings']} / JP {meta['jp_holdings']})\n- data_end_date: {meta['data_end_date']}\n- runtime_seconds: {meta['runtime_seconds']}\n- git_commit_hash: {meta['git_commit_hash']}\n\n## US ranking top\n{_md_table(us_rank.head(20)[['rank_in_region','ticker','composite_score','base_momentum_score','residual_score','momentum_12m','momentum_6m','momentum_3m','Volatility','benchmark','adopted']])}\n\n## Japan ranking top\n{_md_table(jp_rank.head(20)[['rank_in_region','ticker','composite_score','base_momentum_score','residual_score','momentum_12m','momentum_6m','momentum_3m','Volatility','benchmark','adopted']])}\n\n## Final selected tickers and weights\n{_md_table(weights)}\n\n## Benchmark status\n```json\n{json.dumps(meta['benchmark_status'], indent=2, default=str)}\n```\n\n## Missing / excluded / notes\n- Requested tickers: {meta['requested_ticker_count']}\n- Successfully downloaded tickers: {meta['successful_ticker_count']}\n- Excluded tickers: {meta['excluded_ticker_count']}\n- Download failures: {0 if failures.empty else failures.ticker.nunique()}\n\n{_md_table(dq)}\n\nThis screener only ranks current candidates. It does not run backtests, TTL90/Renew30 holding logic, historical trade reconstruction, CAGR, Sharpe, MDD, annual returns, or automatic orders.\n"""
    (out / "screen_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Alpha Engine lightweight live Residual Momentum screener")
    ap.add_argument("--residual-ratio", type=int, default=RESIDUAL_RATIO)
    ap.add_argument("--total-holdings", type=int, default=TOTAL_HOLDINGS)
    ap.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    args = ap.parse_args(); logging.basicConfig(level=logging.INFO)
    out = run_live_screening(residual_ratio=args.residual_ratio, total_holdings=args.total_holdings, output_root=args.output_root)
    print(f"Output directory: {out}")


if __name__ == "__main__":
    main()
