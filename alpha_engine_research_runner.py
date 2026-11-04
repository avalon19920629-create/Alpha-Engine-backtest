"""Research-only price snapshots around the unchanged Alpha Engine live screener.

No ranking, selection, weighting, backtest, or trading logic lives here. A run
downloads once, persists its exact input, invokes the existing screener, and
compares that output with an independent invocation on the saved input.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import alpha_engine_backtest as alpha
import alpha_engine_live_screener as live

VERSION = "1.0"
KNOWN_ENGINE_SHA = "075f0985c76c2212552732534d7f3c741270a80d"
DEFAULT_OUTPUT_ROOT = "/content/drive/MyDrive/alpha_engine/research_runs"
ENGINE_FILES = (
    "Alpha-Engine.py", "alpha_engine_backtest.py", "alpha_engine_live_screener.py",
    "alpha_engine_production.py", "alpha_engine_order_planner.py",
    "alpha_engine_ttl_manager.py", "core_alpha_integration_backtest.py",
)
ENGINE_OUTPUTS = (
    "ranked_candidates_us.csv", "ranked_candidates_jp.csv", "selected_tickers.csv",
    "adopted_weights.csv", "score_components.csv", "data_quality.csv",
    "download_failures.csv", "metadata.json", "screen_report.md",
)
REGRESSION_FILES = ENGINE_OUTPUTS[:5]


class ResearchRunError(RuntimeError):
    """A failed run whose partial snapshot and diagnostics remain available."""

    def __init__(self, message: str, run_dir: Path):
        super().__init__(message)
        self.run_dir = run_dir


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _source_identity(source_ref: str, expected_source_sha: str | None) -> dict:
    if not source_ref or source_ref.startswith("-"):
        raise ValueError("SOURCE_REF must explicitly identify a local Git commit/ref.")
    root = Path(__file__).resolve().parent

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)

    try:
        sha = git("rev-parse", "--verify", "--end-of-options", source_ref + "^{commit}").decode().strip()
        if expected_source_sha is not None and sha != expected_source_sha:
            raise ValueError("SOURCE_REF moved after it was pinned; expected_source_sha mismatch.")
        hashes = {}
        for name in ENGINE_FILES:
            expected = git("show", f"{sha}:{name}")
            actual = (root / name).read_bytes()
            if actual != expected:
                raise ValueError(f"Engine source differs from SOURCE_REF: {name}. Use a matching checkout; no checkout/reset is performed.")
            hashes[name] = hashlib.sha256(actual).hexdigest()
        changed = [name for name in ENGINE_FILES if git("show", f"{KNOWN_ENGINE_SHA}:{name}") != (root / name).read_bytes()]
        head = git("rev-parse", "HEAD").decode().strip()
    except (subprocess.CalledProcessError, OSError) as exc:
        raise ValueError("Could not verify SOURCE_REF and the known engine baseline in this checkout.") from exc
    return {"source_ref": source_ref, "git_commit_sha": sha, "runner_checkout_sha": head,
            "known_engine_baseline": KNOWN_ENGINE_SHA, "engine_files_changed_from_baseline": changed,
            "engine_source_sha256": hashes, "runner_sha256": _hash(Path(__file__))}


def _universe(us, jp) -> tuple[list[str], list[str]]:
    result = []
    for region, values in (("US", us), ("JP", jp)):
        if not values or any(not isinstance(t, str) or not t.strip() for t in values):
            raise ValueError(f"{region} universe must contain ticker names.")
        normalized = [alpha.normalize_yfinance_ticker(t) for t in values]
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"Duplicate normalized {region} ticker in universe.")
        result.append(normalized)
    if set(result[0]) & set(result[1]):
        raise ValueError("US and JP universes must not overlap.")
    return result[0], result[1]


def _quality(prices: pd.DataFrame, us: list[str], jp: list[str], benchmarks: list[str], failures: pd.DataFrame) -> pd.DataFrame:
    failed = set(failures["ticker"]) if "ticker" in failures else set()
    requested = [(t, "US") for t in us] + [(t, "JP") for t in jp] + [(t, "benchmark") for t in benchmarks]
    rows = []
    for ticker, region in requested:
        positions = [i for i, c in enumerate(prices.columns) if c == ticker]
        series = prices.iloc[:, positions[0]] if positions else pd.Series(dtype=float)
        numeric = pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(series.dtype) and not pd.api.types.is_complex_dtype(series.dtype)
        observed = int(series.notna().sum())
        values = series.dropna().to_numpy(dtype=float) if numeric else np.array([])
        invalid = int((~np.isfinite(values) | (values <= 0)).sum())
        reasons = []
        if not positions or observed == 0:
            reasons.append("missing_or_empty_download")
        elif observed < 252:
            reasons.append("insufficient_history")
        if len(positions) > 1:
            reasons.append("duplicate_column")
        if not numeric:
            reasons.append("non_numeric_data")
        if invalid:
            reasons.append("non_positive_or_infinite_price")
        rows.append({"ticker": ticker, "region": region, "downloaded": bool(positions and observed),
                     "observations": observed, "missing_values": int(series.isna().sum()),
                     "insufficient_history": bool(positions and observed < 252),
                     "excluded": bool(reasons), "exclusion_reason": ";".join(reasons),
                     "download_failure_reported": ticker in failed, "duplicate_columns": max(0, len(positions) - 1),
                     "invalid_prices": invalid, "dtype": str(series.dtype)})
    return pd.DataFrame(rows)


def _validate_prices(prices: pd.DataFrame, requested: list[str], cutoff: pd.Timestamp) -> None:
    if prices.empty or not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Price data must be nonempty with a DatetimeIndex.")
    if prices.index.hasnans or prices.index.has_duplicates or prices.columns.has_duplicates:
        raise ValueError("Duplicate days/columns or missing dates in price data.")
    if prices.index.tz is not None or not prices.index.equals(prices.index.normalize()):
        raise ValueError("Daily price index must be timezone-naive midnight dates; original data is not silently rewritten.")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("Price dates must be ordered; original data is not silently reordered.")
    if prices.index.max() > cutoff:
        raise ValueError("Future price observation after the screening cutoff.")
    if not all(isinstance(t, str) for t in prices.columns) or set(prices.columns) - set(requested):
        raise ValueError("Unexpected ticker columns in download; refusing to omit original columns.")
    for name in prices:
        series = prices[name]
        if not pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype) or pd.api.types.is_complex_dtype(series.dtype):
            raise ValueError(f"Non-numeric price data: {name}.")
        values = series.dropna().to_numpy(dtype=float)
        if (~np.isfinite(values) | (values <= 0)).any():
            raise ValueError(f"Non-positive or infinite price: {name}.")


def _benchmark_check(prices: pd.DataFrame, mode: dict) -> dict:
    status = live._benchmark_status(prices, mode)
    for region in ("US", "JP"):
        actual = alpha._resolve_benchmark(prices, region, mode)
        if (actual not in mode[region] or actual not in prices or
                prices[actual].dropna().shape[0] < 252 or status[region]["used"] != actual):
            raise ValueError(f"{region} benchmark missing, insufficient history, or inconsistent engine benchmark selection.")
    return status


def _save_prices(prices: pd.DataFrame, benchmarks: list[str], folder: Path, metadata: dict) -> pd.DataFrame:
    stock = prices.loc[:, [t for t in prices if t not in benchmarks]]
    bench = prices.loc[:, [t for t in prices if t in benchmarks]]
    stock.to_parquet(folder / "price_cache.parquet", engine="pyarrow", index=True)
    bench.to_parquet(folder / "benchmark_prices.parquet", engine="pyarrow", index=True)
    reloaded = pd.concat([pd.read_parquet(folder / "price_cache.parquet"),
                          pd.read_parquet(folder / "benchmark_prices.parquet")], axis=1).loc[:, prices.columns]
    # Frequency is an in-memory hint, not part of the observed daily price data.
    assert_frame_equal(prices, reloaded, check_exact=True, check_freq=False)
    metadata.update({"column_order": list(prices.columns), "index_name": prices.index.name,
                     "dtypes": {str(t): str(prices[t].dtype) for t in prices},
                     "row_count": len(prices), "normalization": "none; original values, dates, column order, dtypes and NaN positions retained",
                     "hash_algorithm": "sha256", "sha256": {name: _hash(folder / name) for name in ("price_cache.parquet", "benchmark_prices.parquet")},
                     "roundtrip_status": "passed"})
    _write_json(folder / "price_cache_metadata.json", metadata)
    return reloaded


def load_research_prices(run_dir: str | Path) -> pd.DataFrame:
    """Load an immutable snapshot with hash verification; never re-download."""
    folder = Path(run_dir) / "price_data"
    meta = json.loads((folder / "price_cache_metadata.json").read_text(encoding="utf-8"))
    for name in ("price_cache.parquet", "benchmark_prices.parquet"):
        if _hash(folder / name) != meta["sha256"][name]:
            raise ValueError(f"Price cache hash mismatch: {name}")
    frame = pd.concat([pd.read_parquet(folder / "price_cache.parquet"),
                       pd.read_parquet(folder / "benchmark_prices.parquet")], axis=1)
    if frame.columns.has_duplicates or set(frame.columns) != set(meta["column_order"]):
        raise ValueError("Price cache columns do not match the snapshot metadata.")
    frame = frame.loc[:, meta["column_order"]]
    if len(frame) != meta["row_count"] or {str(t): str(frame[t].dtype) for t in frame} != meta["dtypes"]:
        raise ValueError("Price cache shape/dtypes do not match the snapshot metadata.")
    return frame


def _screening_checks(folder: Path, total_holdings: int, status: dict) -> dict:
    selected = pd.read_csv(folder / "selected_tickers.csv")
    weights = pd.read_csv(folder / "adopted_weights.csv")
    components = pd.read_csv(folder / "score_components.csv")
    expected = {"US": total_holdings // 2, "JP": total_holdings // 2}
    if selected.groupby("region").size().to_dict() != expected or selected.ticker.duplicated().any():
        raise ValueError("Selected regional counts or unique tickers do not match settings.")
    if weights.ticker.tolist() != selected.ticker.tolist():
        raise ValueError("Adopted weights and selected ticker order differ.")
    values = weights.Weight.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any() or not np.isclose(values.sum(), 1.0, rtol=0, atol=1e-10):
        raise ValueError("Adopted weights must be positive, finite, and sum to one.")
    np.testing.assert_allclose(selected.Weight, weights.Weight, rtol=1e-12, atol=1e-12)
    assert_frame_equal(selected[list(weights.columns)].reset_index(drop=True), weights.reset_index(drop=True),
                       check_exact=False, rtol=1e-12, atol=1e-12)
    for region, name in (("US", "ranked_candidates_us.csv"), ("JP", "ranked_candidates_jp.csv")):
        ranking = pd.read_csv(folder / name)
        expected_tickers = ranking.head(expected[region]).ticker.tolist()
        actual = selected.loc[selected.region == region]
        if actual.ticker.tolist() != expected_tickers or ranking.loc[ranking.adopted, "ticker"].tolist() != expected_tickers:
            raise ValueError(f"{region} selected tickers do not match ranking order/adopted flags.")
        if ranking.rank_in_region.tolist() != list(range(1, len(ranking) + 1)):
            raise ValueError(f"{region} ranking sequence is inconsistent.")
        if set(ranking.benchmark) != {status[region]["used"]}:
            raise ValueError(f"{region} ranking benchmark differs from engine metadata.")
        expected_components = components.loc[components.region == region].reset_index(drop=True)
        assert_frame_equal(ranking, expected_components, check_exact=False, rtol=1e-12, atol=1e-12)
        cols = [c for c in ranking if c in actual]
        assert_frame_equal(ranking.head(expected[region])[cols].reset_index(drop=True), actual[cols].reset_index(drop=True), check_exact=False, rtol=1e-12, atol=1e-12)
    scores = selected[["composite_score", "base_momentum_score", "residual_score", "Volatility", "reference_price_local"]].to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Selected prices or score components are non-finite.")
    return {"selected_counts": expected, "weights_sum": float(values.sum()), "ranking_and_score_consistency": "passed"}


def _audit(run_dir: Path, metadata: dict, checks: dict, error: str | None = None) -> None:
    status = "failed" if error else "passed"
    notes = [
        "engine_outputs/download_failures.csv belongs to the unchanged screener's provided-prices path and does not describe the research download. Use price_data/research_download_failures.csv and price_data_quality.csv.",
        "Engine output files were relocated verbatim to engine_outputs. The engine's original output_folder metadata is preserved; research_run_metadata.json records the actual location.",
        "Current constituents and retrospectively adjusted prices are not a historical point-in-time universe or proof of information available at that historical date.",
        "Runtime regression excludes timestamps, durations and folder names; compares ordered rankings, tickers, weights and score components.",
    ]
    report = {"status": status, "checks": checks, "error": error, "notes": notes}
    cache_metadata_path = run_dir / "price_data" / "price_cache_metadata.json"
    if cache_metadata_path.exists():
        cache_metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
        cache_metadata.update(metadata)
        _write_json(cache_metadata_path, cache_metadata)
    _write_json(run_dir / "audit" / "validation_report.json", report)
    lines = ["# Alpha Engine Research Operations v1.0", "", f"Status: **{status}**", "", "Research only; no trading connection.", ""]
    lines += [f"- {key}: {json.dumps(value, ensure_ascii=False)}" for key, value in checks.items()]
    if error:
        lines += ["", f"Failure: {error}"]
    lines += ["", *[f"- {note}" for note in notes]]
    (run_dir / "audit" / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(run_dir / "research_run_metadata.json", metadata)
    manifest = {"algorithm": "sha256", "self_excluded": "audit/file_manifest.json", "files": {}}
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path != run_dir / "audit" / "file_manifest.json":
            manifest["files"][path.relative_to(run_dir).as_posix()] = {"sha256": _hash(path), "bytes": path.stat().st_size}
    _write_json(run_dir / "audit" / "file_manifest.json", manifest)


def run_research(*, source_ref: str, expected_source_sha: str | None = None,
                 residual_ratio: int = 100, total_holdings: int = 6,
                 lookback_months: int = 18, output_root: str | Path = DEFAULT_OUTPUT_ROOT,
                 prices: pd.DataFrame | None = None, us: list[str] | None = None,
                 jp: list[str] | None = None, downloader=None) -> Path:
    """Create a new, self-contained research snapshot, or retain a failed run.

    Injected prices are intended for offline tests/research and require explicit
    US/JP memberships. They are never silently filled, repaired or downloaded.
    """
    identity = _source_identity(source_ref, expected_source_sha)
    for name, value in (("residual_ratio", residual_ratio), ("total_holdings", total_holdings)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer.")
    variant = live.build_live_variant(residual_ratio, total_holdings)
    if lookback_months != live.LOOKBACK_MONTHS:
        raise ValueError("LOOKBACK_MONTHS must match the unchanged screener's 18 months.")
    if (us is None) != (jp is None) or (prices is not None and us is None):
        raise ValueError("Provide both us and jp when supplying prices or a custom universe.")
    started = _utc_now()
    cutoff = pd.Timestamp(started).tz_convert("UTC").tz_localize(None).normalize()
    end = cutoff.date().isoformat()
    start = (cutoff - pd.DateOffset(months=lookback_months)).date().isoformat()
    root = Path(output_root).expanduser().resolve()
    repo = Path(__file__).resolve().parent
    if root == repo or repo in root.parents:
        raise ValueError("Research output must be outside the source checkout and production history.")
    # UUID prevents same-second runs from colliding. mkdir(exist_ok=False) also
    # protects against an explicit collision, without overwriting any folder.
    run_dir = root / f"{pd.Timestamp(started).strftime('%Y%m%dT%H%M%SZ')}_residual{residual_ratio}_n{total_holdings}_{uuid.uuid4().hex[:12]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    data = run_dir / "price_data"
    data.mkdir()
    (run_dir / "audit").mkdir()
    versions = {"python": platform.python_version()}
    for package in ("pandas", "numpy", "yfinance", "pyarrow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    metadata = {"version": VERSION, **identity, "run_started_at": started, "run_finished_at": None,
                "run_mode": "research_only", "cache_policy": "new_snapshot", "status": "incomplete",
                "validation_status": "incomplete", "residual_ratio": residual_ratio, "total_holdings": total_holdings,
                "us_holdings": variant["us_holdings"], "jp_holdings": variant["jp_holdings"],
                "lookback_months": lookback_months, "requested_start": start, "requested_end": end,
                "screening_cutoff_date": end, "download_expanded_start": (pd.Timestamp(start) - pd.offsets.BDay(400)).date().isoformat(),
                "download_end_exclusive": (cutoff + pd.Timedelta(days=1)).date().isoformat(),
                "data_source": "provided_prices" if prices is not None else "yfinance",
                "auto_adjust": None if prices is not None else True, "price_type": "adjusted Close (not raw Close or volume)",
                "versions": versions, "output_folder": str(run_dir), "engine_output_folder": str(run_dir / "engine_outputs")}
    _write_json(run_dir / "research_run_metadata.json", metadata)
    checks = {"source_verified": "passed"}
    try:
        if us is None:
            us, jp = alpha.get_live_universe()
        us, jp = _universe(us, jp)
        mode = alpha.build_benchmark_modes()["broad_default"]
        benchmarks = sorted({t for values in mode.values() for t in values})
        if (set(us) | set(jp)) & set(benchmarks):
            raise ValueError("Equity universe must not contain benchmark tickers.")
        requested = list(dict.fromkeys([*us, *jp, *benchmarks]))
        pd.DataFrame([{"ticker": t, "region": region} for region, tickers in (("US", us), ("JP", jp)) for t in tickers]).to_csv(data / "requested_universe.csv", index=False)
        metadata.update({"us_universe": us, "jp_universe": jp, "benchmark_candidates": mode,
                         "requested_equity_count": len(us) + len(jp), "requested_benchmark_count": len(benchmarks)})
        failures = pd.DataFrame(columns=["ticker", "reason"])
        if prices is None:
            prices, failures, download_requested = alpha.download_live_universe_prices(requested, start, end, batch_size=80, downloader=downloader)
            metadata["download_requested_tickers"] = list(download_requested)
        if not isinstance(prices, pd.DataFrame):
            raise ValueError("Downloader must return a price DataFrame.")
        failures.to_csv(data / "research_download_failures.csv", index=False)
        quality = _quality(prices, us, jp, benchmarks, failures)
        quality.to_csv(data / "price_data_quality.csv", index=False)
        metadata.update({"successful_ticker_count": int(quality.downloaded.sum()),
                         "successful_equity_count": int(quality.loc[quality.region != "benchmark", "downloaded"].sum()),
                         "successful_benchmark_count": int(quality.loc[quality.region == "benchmark", "downloaded"].sum()),
                         "insufficient_history_count": int(quality.insufficient_history.sum()),
                         "excluded_ticker_count": int(quality.excluded.sum()), "download_failure_count": int(len(failures)),
                         "duplicate_day_count": int(prices.index.duplicated().sum()),
                         "duplicate_column_count": int(prices.columns.duplicated().sum())})
        _validate_prices(prices, requested, cutoff)
        metadata.update({"observed_start": prices.index.min().date().isoformat(), "observed_end": prices.index.max().date().isoformat()})
        cache_meta = {**metadata, "created_at": _utc_now()}
        restored = _save_prices(prices, benchmarks, data, cache_meta)
        metadata["price_cache_sha256"] = cache_meta["sha256"]
        checks["parquet_roundtrip"] = "passed"
        status = _benchmark_check(prices, mode)
        metadata["benchmarks"] = {region: status[region]["used"] for region in ("US", "JP")}
        metadata["benchmark_status"] = status
        checks["benchmarks"] = "passed"
        # Keep the original frame unchanged, including calendar-related NaNs.
        before = prices.copy(deep=True)
        staging = run_dir / ".engine_staging"
        engine_folder = live.run_live_screening(prices=prices, us=us, jp=jp, residual_ratio=residual_ratio,
                                                total_holdings=total_holdings, output_root=staging)
        original_hashes = {name: _hash(engine_folder / name) for name in ENGINE_OUTPUTS}
        metadata["engine_reported_output_folder"] = str(engine_folder)
        engine_folder.rename(run_dir / "engine_outputs")
        staging.rmdir()
        engine_folder = run_dir / "engine_outputs"
        assert_frame_equal(before, prices, check_exact=True)
        checks["input_unchanged_by_engine"] = "passed"
        if original_hashes != {name: _hash(engine_folder / name) for name in ENGINE_OUTPUTS}:
            raise ValueError("Engine output changed during archival.")
        checks["engine_artifacts_preserved"] = "passed"
        engine_metadata = json.loads((engine_folder / "metadata.json").read_text(encoding="utf-8"))
        if (engine_metadata["data_end_date"] != metadata["observed_end"] or
                engine_metadata["benchmarks"] != metadata["benchmarks"] or
                engine_metadata["data_fetch_start"] != start or engine_metadata["data_fetch_end"] != end):
            raise ValueError("Engine acquisition dates, data end or benchmarks do not agree with the saved input.")
        checks["screening"] = _screening_checks(engine_folder, total_holdings, status)
        # A sufficient raw history does not guarantee an engine score (constant
        # prices, for example, have zero volatility). Record actual omissions.
        ranked = set(pd.read_csv(engine_folder / "score_components.csv").ticker)
        quality["engine_ranked"] = quality.ticker.isin(ranked) & (quality.region != "benchmark")
        for row in quality.index[quality.region != "benchmark"]:
            if not quality.at[row, "engine_ranked"]:
                reason = quality.at[row, "exclusion_reason"]
                quality.at[row, "exclusion_reason"] = ";".join(filter(None, (reason, "not_ranked_by_engine")))
                quality.at[row, "excluded"] = True
        quality.to_csv(data / "price_data_quality.csv", index=False)
        metadata["excluded_ticker_count"] = int(quality.excluded.sum())
        metadata["excluded_equity_count"] = int(quality.loc[quality.region != "benchmark", "excluded"].sum())
        metadata["ranked_equity_count"] = int(len(ranked))
        # Separate original-input execution and disk-input execution demonstrate
        # that persistence did not change engine results; metadata is excluded.
        with tempfile.TemporaryDirectory(prefix="alpha-research-regression-") as temporary:
            comparison = live.run_live_screening(prices=restored, us=us, jp=jp, residual_ratio=residual_ratio,
                                                 total_holdings=total_holdings, output_root=temporary)
            for name in REGRESSION_FILES:
                assert_frame_equal(pd.read_csv(engine_folder / name), pd.read_csv(comparison / name),
                                   check_exact=False, rtol=1e-12, atol=1e-12)
        checks["regression"] = {"status": "passed", "files": list(REGRESSION_FILES), "rtol": 1e-12, "atol": 1e-12}
        # Recheck that neither the computation nor a concurrent edit changed the
        # engine source after the input was attributed to SOURCE_REF.
        _source_identity(source_ref, identity["git_commit_sha"])
        metadata.update({"run_finished_at": _utc_now(), "status": "completed", "validation_status": "passed"})
        _audit(run_dir, metadata, checks)
    except Exception as exc:
        metadata.update({"run_finished_at": _utc_now(), "status": "failed", "validation_status": "failed",
                         "error": f"{type(exc).__name__}: {exc}"})
        _audit(run_dir, metadata, checks, metadata["error"])
        raise ResearchRunError(metadata["error"], run_dir) from exc
    return run_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Alpha Engine Research Operations v1.0 (research_only; new_snapshot)")
    parser.add_argument("--source-ref", required=True, help="Explicit local engine Git ref/SHA; verified without checkout or fetch")
    parser.add_argument("--expected-source-sha", help="Reject a ref that moved since notebook pinning")
    parser.add_argument("--residual-ratio", type=int, default=100)
    parser.add_argument("--total-holdings", type=int, default=6)
    parser.add_argument("--lookback-months", type=int, default=18)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    try:
        result = run_research(**vars(args))
    except ResearchRunError as exc:
        print(f"Research run failed: {exc}")
        print(f"Failed run directory: {exc.run_dir}")
        return 1
    print(f"Output directory: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
