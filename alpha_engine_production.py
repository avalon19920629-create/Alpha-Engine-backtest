"""Colab production Gearbox for Alpha Engine profile-controlled runs.

This module intentionally does not add a new optimizer.  It runs the existing
TTL90/Renew30/Composite selection engine with one resolved production profile.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import alpha_engine_backtest as alpha

ENGINE_PROFILE = "ROBUST"
ALLOW_OVERDRIVE = False
ALLOW_CUSTOM_PROFILE = False
CUSTOM_RESIDUAL_RATIO = None
CUSTOM_PORTFOLIO_N = None
RUN_MODE = "production"
TTL_DAYS = 90
RENEWAL_DAYS = 30
HEALTH_CHECK = "Composite"


@dataclass(frozen=True)
class GearboxConfig:
    profile: str
    residual_ratio: int
    portfolio_n: int
    role: str
    ttl_days: int = TTL_DAYS
    renewal_days: int = RENEWAL_DAYS
    health_check: str = HEALTH_CHECK
    run_mode: str = RUN_MODE

    @property
    def variant_name(self) -> str:
        return f"Residual_{self.residual_ratio}_N{self.portfolio_n}_TTL{self.ttl_days}_Renew{self.renewal_days}_{self.health_check}"

    def to_variant(self) -> dict:
        us_n = self.portfolio_n // 2
        jp_n = self.portfolio_n - us_n
        w = self.residual_ratio / 100
        return {
            "name": self.variant_name,
            "selection_name": f"Residual_{self.residual_ratio}_N{self.portfolio_n}",
            "base_weight": round(1 - w, 2),
            "residual_weight": round(w, 2),
            "total_holdings": self.portfolio_n,
            "us_holdings": us_n,
            "jp_holdings": jp_n,
            "ttl_days": self.ttl_days,
            "renewal_protocol": self.health_check.lower(),
            "renewal_extension_days": self.renewal_days,
            "is_baseline": False,
        }


def resolve_engine_config(
    profile: str = ENGINE_PROFILE,
    *,
    allow_overdrive: bool = ALLOW_OVERDRIVE,
    allow_custom_profile: bool = ALLOW_CUSTOM_PROFILE,
    custom_residual_ratio: int | None = CUSTOM_RESIDUAL_RATIO,
    custom_portfolio_n: int | None = CUSTOM_PORTFOLIO_N,
    run_mode: str = RUN_MODE,
) -> GearboxConfig:
    profile = (profile or "ROBUST").upper()
    if profile == "ROBUST":
        return GearboxConfig(profile, 60, 12, "standard_robust_alpha", run_mode=run_mode)
    if profile == "FERRARI":
        if not allow_overdrive:
            raise PermissionError("FERRARI requires ALLOW_OVERDRIVE=True; it is overdrive/satellite alpha, not the standard machine.")
        return GearboxConfig(profile, 100, 6, "overdrive_satellite_alpha", run_mode=run_mode)
    if profile == "CUSTOM":
        if not allow_custom_profile:
            raise PermissionError("CUSTOM requires ALLOW_CUSTOM_PROFILE=True and is for research metadata only.")
        if custom_residual_ratio is None or custom_portfolio_n is None:
            raise ValueError("CUSTOM requires explicit custom_residual_ratio and custom_portfolio_n.")
        return GearboxConfig(profile, int(custom_residual_ratio), int(custom_portfolio_n), "custom_research_alpha", run_mode="research")
    raise ValueError(f"Unknown ENGINE_PROFILE: {profile}")


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def make_run_dir(output_root: str | Path, profile: str, now: pd.Timestamp | None = None) -> Path:
    ts = (now or pd.Timestamp.now("UTC")).strftime("%Y%m%dT%H%M%SZ")
    out = Path(output_root) / f"{ts}_{profile.lower()}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def _copy_cache_reference(cache_source: str | Path, run_dir: Path) -> dict:
    source = Path(cache_source) if cache_source else None
    if not source or not source.exists():
        return {"price_cache_saved": False, "price_cache_source": str(cache_source or "")}
    dest = run_dir / "price_cache"
    shutil.copytree(source, dest, dirs_exist_ok=True)
    return {"price_cache_saved": True, "price_cache_source": str(source), "price_cache_run_copy": str(dest)}


def _decision_reasons(selected: pd.DataFrame, decisions: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not selected.empty:
        for _, r in selected.iterrows():
            rows.append({"ticker": r.get("ticker"), "trade_date": r.get("trade_date"), "exit_date": r.get("exit_date"), "reason": "selected_and_weighted", "weight": r.get("Weight")})
    if not decisions.empty:
        for _, r in decisions.iterrows():
            rows.append({"ticker": r.get("ticker"), "trade_date": r.get("trade_date"), "exit_date": r.get("health_check_date"), "reason": "extended_by_composite_health_check" if bool(r.get("renewed")) else "cash_after_composite_health_check_failed", "weight": None})
    if not events.empty:
        for _, r in events[events.get("event", pd.Series(dtype=str)).astype(str).eq("extension_degradation_end")].iterrows():
            rows.append({"ticker": r.get("ticker"), "trade_date": None, "exit_date": r.get("date"), "reason": "sold_after_extension_degradation", "weight": None})
    return pd.DataFrame(rows)


def write_run_report(run_dir: Path, metadata: dict, selected: pd.DataFrame, decisions: pd.DataFrame, reasons: pd.DataFrame) -> None:
    warning = ""
    if metadata["profile"] == "FERRARI":
        warning = "\n> WARNING: FERRARI is overdrive/satellite alpha and must not be confused with the ROBUST standard machine.\n"
    tickers = ", ".join(selected.get("ticker", pd.Series(dtype=str)).astype(str).drop_duplicates().head(50)) if not selected.empty else "none"
    report = f"""# Alpha Engine Production Run Report
{warning}
## Gearbox
- Profile: {metadata['profile']}
- Role: {metadata['role']}
- Variant: {metadata['variant_name']}
- residual_ratio: {metadata['residual_ratio']}
- portfolio_n: {metadata['portfolio_n']}
- TTL / Renew / Health Check: {metadata['ttl_days']} / {metadata['renewal_days']} / {metadata['health_check']}
- Git commit hash: {metadata['git_commit']}
- Run mode: {metadata['run_mode']}

## Runtime and data
- Execution datetime UTC: {metadata['run_started_at']}
- Data fetched at UTC: {metadata['data_fetched_at']}
- Data start: {metadata['data_start']}
- Data end: {metadata['data_end']}
- Cache metadata: `{metadata.get('cache_metadata_path', '')}`

## Selections and weights
Selected tickers: {tickers}

Detailed selections are saved in `selected_tickers.csv`; weights are in its `Weight` column.

## Renewal / sell / extend / cash reasons
Renewal decisions are saved in `renewal_decisions.csv` and summarized in `decision_reasons.csv`.
Decision rows: {len(decisions)}; reason rows: {len(reasons)}.
"""
    (run_dir / "run_report.md").write_text(report, encoding="utf-8")


def run_production(config: GearboxConfig, start="2015-01-01", end=None, output_root="artifacts/production_runs", demo=False, cache_dir=None, downloader=None) -> Path:
    end = end or pd.Timestamp.today().date().isoformat()
    run_started = pd.Timestamp.now("UTC")
    run_dir = make_run_dir(output_root, config.profile, run_started)
    if config.profile == "FERRARI":
        print("WARNING: FERRARI is overdrive/satellite alpha; do not confuse it with the ROBUST standard machine.")
    us, jp = alpha.get_live_universe() if not demo else ([f"US{i}" for i in range(8)], [f"JP{i}.T" for i in range(8)])
    us = [alpha.normalize_yfinance_ticker(t) for t in us]
    jp = [alpha.normalize_yfinance_ticker(t) for t in jp]
    modes = alpha.build_benchmark_modes(); benchmark_mode = modes["broad_default"]
    bench_tickers = sorted({x for mode in modes.values() for vals in mode.values() for x in vals})
    requested = list(dict.fromkeys([*us, *jp]))
    if demo:
        prices = alpha.demo_prices(); cache_meta = {"cache_used": True, "cache_source": "demo_prices"}
        prices.to_pickle(run_dir / "prices_demo.pkl")
    else:
        prices, failures, requested, cache_meta = alpha.build_price_cache(requested, bench_tickers, start, end, cache_dir or (run_dir / "cache"), downloader=downloader)
        failures.to_csv(run_dir / "download_failures.csv", index=False)
    data_fetched = pd.Timestamp.now("UTC").isoformat()
    cache_info = _copy_cache_reference(cache_dir or (run_dir / "cache"), run_dir) if not demo else {"price_cache_saved": True, "price_cache_source": "demo_prices", "price_cache_run_copy": str(run_dir / "prices_demo.pkl")}
    dq, _, _, usable = alpha.build_live_data_quality_report(prices, requested, us, jp, pd.DataFrame(), start, end)
    us_usable = [t for t in us if t in usable]; jp_usable = [t for t in jp if t in usable]
    returns, selected, scores, turnover, trades, holds, decisions, events = alpha.run_ttl_renewal_variant(prices, us_usable, jp_usable, start, end, config.to_variant(), benchmark_mode)
    reasons = _decision_reasons(selected, decisions, events)
    summary = pd.DataFrame({config.variant_name: alpha.metrics(returns)}).T
    metadata = {**config.__dict__, "variant_name": config.variant_name, "git_commit": git_commit_hash(), "run_started_at": run_started.isoformat(), "data_fetched_at": data_fetched, "data_start": str(prices.index.min().date()) if len(prices.index) else "", "data_end": str(prices.index.max().date()) if len(prices.index) else "", "experiment_or_production": config.run_mode, **cache_meta, **cache_info, "cache_metadata_path": str((run_dir / "price_cache" / "cache_metadata.json") if (run_dir / "price_cache" / "cache_metadata.json").exists() else "")}
    for name, df in (("selected_tickers.csv", selected), ("adopted_weights.csv", selected[["trade_date", "ticker", "Weight", "Region"]] if not selected.empty else pd.DataFrame()), ("renewal_decisions.csv", decisions), ("decision_reasons.csv", reasons), ("turnover.csv", turnover), ("trade_log.csv", trades), ("holding_periods.csv", holds), ("ttl_event_log.csv", events), ("score_components.csv", scores), ("data_quality.csv", dq), ("run_summary.csv", summary)):
        df.to_csv(run_dir / name, index=name != "run_summary.csv")
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    write_run_report(run_dir, metadata, selected, decisions, reasons)
    return run_dir


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=ENGINE_PROFILE, choices=["ROBUST", "FERRARI", "CUSTOM"])
    ap.add_argument("--allow-overdrive", action="store_true", default=ALLOW_OVERDRIVE)
    ap.add_argument("--allow-custom-profile", action="store_true", default=ALLOW_CUSTOM_PROFILE)
    ap.add_argument("--custom-residual-ratio", type=int, default=CUSTOM_RESIDUAL_RATIO)
    ap.add_argument("--custom-portfolio-n", type=int, default=CUSTOM_PORTFOLIO_N)
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=pd.Timestamp.today().date().isoformat())
    ap.add_argument("--output-root", default="artifacts/production_runs")
    ap.add_argument("--cache-dir")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args(argv)
    cfg = resolve_engine_config(args.profile, allow_overdrive=args.allow_overdrive, allow_custom_profile=args.allow_custom_profile, custom_residual_ratio=args.custom_residual_ratio, custom_portfolio_n=args.custom_portfolio_n)
    out = run_production(cfg, args.start, args.end, args.output_root, args.demo, args.cache_dir)
    print(f"Output directory: {out.resolve()}")


if __name__ == "__main__":
    main()
