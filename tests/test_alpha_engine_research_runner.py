"""Offline acceptance tests for the research snapshot wrapper.

The unmodified screener is the numerical reference; synthetic adjusted-close
data exercises persistence and audit behavior without contacting a provider.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

import alpha_engine_backtest as alpha
import alpha_engine_live_screener as live
import alpha_engine_research_runner as research


SOURCE_REF = "075f0985c76c2212552732534d7f3c741270a80d"
ENGINE_FILES = (
    "ranked_candidates_us.csv", "ranked_candidates_jp.csv",
    "selected_tickers.csv", "adopted_weights.csv", "score_components.csv",
    "data_quality.csv", "download_failures.csv", "metadata.json", "screen_report.md",
)
COMPARISON_FILES = ENGINE_FILES[:5]
US = [f"US{i}" for i in range(8)]
JP = [f"JP{i}.T" for i in range(8)]


@pytest.fixture
def prices():
    end = pd.Timestamp.now("UTC").tz_localize(None).normalize() - pd.Timedelta(days=1)
    index = pd.bdate_range(end=end, periods=600, name="Date")
    mode = alpha.build_benchmark_modes()["broad_default"]
    benchmarks = list(dict.fromkeys(t for values in mode.values() for t in values))
    # Deliberately interleave stock and benchmark columns to exercise reconstruction.
    names = [US[0], benchmarks[0], *JP, *US[1:], *benchmarks[1:]]
    rng = np.random.default_rng(48071)
    frame = pd.DataFrame({
        ticker: 100 * np.exp(np.cumsum(rng.normal(.0003 + i * .00001, .008 + i * .0001, len(index))))
        for i, ticker in enumerate(names)
    }, index=index)
    frame[US[1]] = frame[US[1]].astype("float32")
    return frame


@pytest.fixture(autouse=True)
def no_implicit_network(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Offline research tests must not perform implicit data acquisition")
    monkeypatch.setattr(alpha, "get_live_universe", unexpected)
    monkeypatch.setattr(alpha, "download_live_universe_prices", unexpected)


def run(tmp_path, prices, **kwargs):
    return research.run_research(
        source_ref=SOURCE_REF, output_root=tmp_path / "research_runs",
        prices=prices, us=US, jp=JP, **kwargs,
    )


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_prices(run_dir, original):
    root = Path(run_dir) / "price_data"
    stocks = pd.read_parquet(root / "price_cache.parquet")
    benchmarks = pd.read_parquet(root / "benchmark_prices.parquet")
    assert not (set(stocks.columns) & set(benchmarks.columns))
    return pd.concat([stocks, benchmarks], axis=1).loc[:, original.columns]


def assert_failed(exc):
    run_dir = Path(exc.value.run_dir)
    metadata = read_json(run_dir / "research_run_metadata.json")
    assert metadata["status"] in {"failed", "incomplete"}
    report = read_json(run_dir / "audit" / "validation_report.json")
    assert report["status"] in {"failed", "incomplete"}
    return run_dir


def test_snapshot_parquet_roundtrip_preserves_values_order_dtypes_and_input(tmp_path, prices):
    original = prices.copy(deep=True)
    out = run(tmp_path, prices)
    pd.testing.assert_frame_equal(prices, original, check_exact=True)
    pd.testing.assert_frame_equal(read_prices(out, original), original, check_exact=True, check_freq=False)
    assert {p.name for p in (out / "engine_outputs").iterdir()} == set(ENGINE_FILES)
    for name in ["price_cache_metadata.json", "requested_universe.csv", "research_download_failures.csv", "price_data_quality.csv"]:
        assert (out / "price_data" / name).is_file()
    for name in ["validation_report.json", "validation_report.md", "file_manifest.json"]:
        assert (out / "audit" / name).is_file()
    metadata = read_json(out / "research_run_metadata.json")
    assert metadata["status"] == "completed"
    assert read_json(out / "audit" / "validation_report.json")["status"] == "passed"
    assert metadata["source_ref"] == SOURCE_REF
    assert metadata["git_commit_sha"] == SOURCE_REF
    assert metadata["run_mode"] == "research_only"
    assert metadata["cache_policy"] == "new_snapshot"
    assert metadata["validation_status"] == "passed"
    assert metadata["residual_ratio"] == 100
    assert metadata["total_holdings"] == 6
    assert metadata["us_holdings"] == metadata["jp_holdings"] == 3
    assert metadata["lookback_months"] == 18
    assert metadata["requested_equity_count"] == len(US) + len(JP)
    assert metadata["requested_benchmark_count"] == 4
    assert metadata["successful_equity_count"] == len(US) + len(JP)
    assert metadata["successful_benchmark_count"] == 4
    assert metadata["insufficient_history_count"] == metadata["excluded_ticker_count"] == 0
    assert metadata["benchmarks"] == {"US": "SPY", "JP": "1306.T"}
    assert metadata["data_source"] == "provided_prices"
    # Injected test data is marked explicitly, instead of falsely claiming that
    # yfinance auto-adjusted synthetic prices.
    assert metadata["auto_adjust"] is None
    assert "adjusted Close" in metadata["price_type"]
    assert metadata["observed_start"] == original.index.min().date().isoformat()
    assert metadata["observed_end"] == original.index.max().date().isoformat()
    assert metadata["requested_end"] == metadata["screening_cutoff_date"]
    assert pd.Timestamp(metadata["observed_end"]) <= pd.Timestamp(metadata["requested_end"])
    assert pd.Timestamp(metadata["requested_start"]) == pd.Timestamp(metadata["requested_end"]) - pd.DateOffset(months=18)
    assert pd.Timestamp(metadata["download_expanded_start"]) == pd.Timestamp(metadata["requested_start"]) - pd.offsets.BDay(400)
    started, finished = (pd.Timestamp(metadata[k]) for k in ("run_started_at", "run_finished_at"))
    assert started.utcoffset().total_seconds() == finished.utcoffset().total_seconds() == 0
    assert finished >= started
    assert Path(metadata["output_folder"]) == out
    assert set(metadata["versions"]) >= {"python", "pandas", "yfinance", "pyarrow"}
    assert all(metadata["versions"][package] != "not installed" for package in ["python", "pandas", "yfinance", "pyarrow"])
    assert metadata["engine_files_changed_from_baseline"] == []
    cache_meta = read_json(out / "price_data" / "price_cache_metadata.json")
    assert cache_meta["status"] == metadata["status"]
    assert cache_meta["validation_status"] == metadata["validation_status"]
    assert cache_meta["benchmarks"] == metadata["benchmarks"]
    for name in ["price_cache.parquet", "benchmark_prices.parquet"]:
        digest = hashlib.sha256((out / "price_data" / name).read_bytes()).hexdigest()
        assert cache_meta["sha256"][name] == metadata["price_cache_sha256"][name] == digest
    manifest = read_json(out / "audit" / "file_manifest.json")
    assert manifest["algorithm"] == "sha256"
    actual_files = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    assert set(manifest["files"]) == actual_files - {"audit/file_manifest.json"}
    for name, recorded in manifest["files"].items():
        content = (out / name).read_bytes()
        assert recorded == {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}


def test_download_once_and_preserve_real_download_failures(tmp_path, prices, monkeypatch):
    missing = US[-1]
    downloaded = prices.drop(columns=missing)
    monkeypatch.setattr(alpha, "get_live_universe", lambda: (US, JP))
    # Use the real existing downloader wrapper with a network-free provider.
    # The autouse network trap is replaced with the saved function below.
    monkeypatch.setattr(alpha, "download_live_universe_prices", REAL_DOWNLOAD)
    provider = Mock(return_value=downloaded)
    out = research.run_research(
        source_ref=SOURCE_REF, output_root=tmp_path / "research_runs", downloader=provider,
    )
    assert provider.call_count == 1
    call = provider.call_args
    assert call.kwargs["auto_adjust"] is True
    failures = pd.read_csv(out / "price_data" / "research_download_failures.csv")
    assert missing in set(failures["ticker"])
    assert failures.loc[failures.ticker == missing, "reason"].str.len().min() > 0
    quality = pd.read_csv(out / "price_data" / "price_data_quality.csv").set_index("ticker")
    assert quality.loc[missing, "observations"] == 0
    assert bool(quality.loc[missing, "excluded"])
    metadata = read_json(out / "research_run_metadata.json")
    assert metadata["data_source"] == "yfinance"
    assert metadata["auto_adjust"] is True
    assert metadata["download_failure_count"] == 1
    assert metadata["successful_equity_count"] == len(US) + len(JP) - 1
    # Supplied-price engine artifacts stay unchanged; its empty failures do not
    # overwrite the authoritative acquisition failures saved by the wrapper.
    assert pd.read_csv(out / "engine_outputs" / "download_failures.csv").empty
    report = (out / "audit" / "validation_report.md").read_text().lower()
    assert "research_download_failures.csv" in report


def test_short_history_stock_is_recorded_without_repairing_original(tmp_path, prices):
    prices.loc[prices.index[:-100], US[-1]] = np.nan
    out = run(tmp_path, prices)
    quality = pd.read_csv(out / "price_data" / "price_data_quality.csv").set_index("ticker")
    assert quality.loc[US[-1], "observations"] == 100
    assert bool(quality.loc[US[-1], "insufficient_history"])
    assert bool(quality.loc[US[-1], "excluded"])
    assert US[-1] not in set(pd.read_csv(out / "engine_outputs" / "ranked_candidates_us.csv").ticker)
    pd.testing.assert_frame_equal(read_prices(out, prices), prices, check_exact=True, check_freq=False)


def test_constant_stock_omitted_by_engine_has_explicit_quality_exclusion(tmp_path, prices):
    prices[US[-1]] = 100.0
    out = run(tmp_path, prices)
    ranked = pd.read_csv(out / "engine_outputs" / "ranked_candidates_us.csv")
    assert US[-1] not in set(ranked.ticker)
    quality = pd.read_csv(out / "price_data" / "price_data_quality.csv").set_index("ticker")
    constant = quality.loc[US[-1]]
    assert constant["observations"] == len(prices)
    assert bool(constant["downloaded"])
    assert not bool(constant["insufficient_history"])
    assert not bool(constant["engine_ranked"])
    assert bool(constant["excluded"])
    assert "not_ranked_by_engine" in constant["exclusion_reason"]
    metadata = read_json(out / "research_run_metadata.json")
    assert metadata["excluded_ticker_count"] == 1
    assert metadata["insufficient_history_count"] == 0
    pd.testing.assert_frame_equal(read_prices(out, prices), prices, check_exact=True, check_freq=False)


@pytest.mark.parametrize("region", ["US", "JP"])
def test_missing_region_benchmark_stops_and_marks_failed(tmp_path, prices, region):
    mode = alpha.build_benchmark_modes()["broad_default"]
    # Also remove fallback benchmarks that could silently neutralize residuals.
    removed = set(mode[region]) | ({"^N225"} if region == "JP" else {"QQQ"})
    prices = prices.drop(columns=[t for t in removed if t in prices])
    with pytest.raises(research.ResearchRunError) as exc:
        run(tmp_path, prices)
    assert_failed(exc)


def test_present_benchmark_with_insufficient_history_is_not_accepted(tmp_path, prices):
    for ticker in alpha.build_benchmark_modes()["broad_default"]["JP"]:
        prices.loc[prices.index[:-50], ticker] = np.nan
    with pytest.raises(research.ResearchRunError) as exc:
        run(tmp_path, prices)
    assert_failed(exc)


def test_first_present_benchmark_cannot_be_empty_even_when_alternative_exists(tmp_path, prices):
    # Existing status reporting may choose ^TOPX, while scoring resolves the
    # first present 1306.T column. The wrapper must flag the inconsistency;
    # repairing or removing that column would change the original engine input.
    prices["1306.T"] = np.nan
    with pytest.raises(research.ResearchRunError) as exc:
        run(tmp_path, prices)
    assert_failed(exc)


@pytest.mark.parametrize("problem", ["duplicate_day", "duplicate_ticker", "nonnumeric", "zero", "negative", "infinite", "future"])
def test_invalid_raw_prices_fail_without_success_marker(tmp_path, prices, problem):
    if problem == "duplicate_day":
        prices = pd.concat([prices.iloc[:1], prices])
    elif problem == "duplicate_ticker":
        prices = pd.concat([prices, prices[[US[0]]]], axis=1)
    elif problem == "nonnumeric":
        prices[US[0]] = prices[US[0]].astype(object)
        prices.loc[prices.index[-1], US[0]] = "not-a-price"
    elif problem == "future":
        extra = prices.iloc[[-1]].copy()
        extra.index = pd.DatetimeIndex([pd.Timestamp.now("UTC").tz_localize(None).normalize() + pd.Timedelta(days=2)])
        prices = pd.concat([prices, extra])
    else:
        prices.loc[prices.index[-1], US[0]] = {"zero": 0, "negative": -1, "infinite": np.inf}[problem]
    with pytest.raises(research.ResearchRunError) as exc:
        run(tmp_path, prices)
    assert_failed(exc)


def test_holiday_nans_survive_cache_and_same_screening(tmp_path, prices):
    prices.loc[prices.index[::11], JP[0]] = np.nan
    prices.loc[prices.index[-1], US[0]] = np.nan
    out = run(tmp_path, prices)
    cached = read_prices(out, prices)
    pd.testing.assert_frame_equal(cached, prices, check_exact=True, check_freq=False)
    assert cached[JP[0]].isna().sum() == prices[JP[0]].isna().sum()


@pytest.mark.parametrize("residual,holdings", [(100, 6), (60, 12)])
def test_rankings_selections_weights_scores_match_unmodified_screener(tmp_path, prices, residual, holdings):
    out = run(tmp_path, prices, residual_ratio=residual, total_holdings=holdings)
    direct = live.run_live_screening(
        prices=prices.copy(deep=True), us=US, jp=JP,
        residual_ratio=residual, total_holdings=holdings, output_root=tmp_path / "direct",
    )
    for name in COMPARISON_FILES:
        actual = pd.read_csv(out / "engine_outputs" / name)
        expected = pd.read_csv(direct / name)
        pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-10, atol=1e-12)
        if "ticker" in actual:
            assert actual.ticker.tolist() == expected.ticker.tolist()
    selected = pd.read_csv(out / "engine_outputs" / "selected_tickers.csv")
    assert selected.groupby("region").size().to_dict() == {"US": holdings // 2, "JP": holdings // 2}
    weights = pd.read_csv(out / "engine_outputs" / "adopted_weights.csv")
    assert weights.Weight.sum() == pytest.approx(1.0, abs=1e-10)


def test_engine_failure_keeps_price_snapshot_and_failed_audit(tmp_path, prices, monkeypatch):
    def broken(**kwargs):
        raise RuntimeError("synthetic screening failure")
    monkeypatch.setattr(live, "run_live_screening", broken)
    with pytest.raises(research.ResearchRunError) as exc:
        run(tmp_path, prices)
    out = assert_failed(exc)
    assert (out / "price_data" / "price_cache.parquet").is_file()
    assert "synthetic screening failure" in json.dumps(read_json(out / "research_run_metadata.json"))


def test_cache_readback_value_mismatch_cannot_pass_validation(tmp_path, prices, monkeypatch):
    real_read = pd.read_parquet

    def changed_cache(path, *args, **kwargs):
        frame = real_read(path, *args, **kwargs)
        if Path(path).name == "price_cache.parquet":
            frame.iloc[0, 0] += 1.0
        return frame

    monkeypatch.setattr(pd, "read_parquet", changed_cache)
    with pytest.raises(research.ResearchRunError) as exc:
        run(tmp_path, prices)
    assert_failed(exc)


def test_independent_snapshot_loader_preserves_data_and_rejects_changed_file(tmp_path, prices):
    out = run(tmp_path, prices)
    loaded = research.load_research_prices(out)
    pd.testing.assert_frame_equal(loaded, prices, check_exact=True, check_freq=False)
    path = out / "price_data" / "price_cache.parquet"
    path.write_bytes(path.read_bytes() + b"changed-after-snapshot")
    with pytest.raises(ValueError, match="hash mismatch"):
        research.load_research_prices(out)


def test_two_runs_never_overwrite_previous_history(tmp_path, prices):
    first = run(tmp_path, prices)
    before = {str(p.relative_to(first)): p.read_bytes() for p in first.rglob("*") if p.is_file()}
    second = run(tmp_path, prices)
    assert first != second
    assert second.is_dir()
    after = {str(p.relative_to(first)): p.read_bytes() for p in first.rglob("*") if p.is_file()}
    assert after == before


@pytest.mark.parametrize("kwargs", [{"total_holdings": 5}, {"lookback_months": 12}, {"residual_ratio": -1}])
def test_invalid_or_unsupported_configuration_rejected(tmp_path, prices, kwargs):
    with pytest.raises((ValueError, research.ResearchRunError)):
        run(tmp_path, prices, **kwargs)


def test_unknown_source_ref_is_rejected_instead_of_using_head(tmp_path, prices):
    with pytest.raises((ValueError, research.ResearchRunError)):
        research.run_research(
            source_ref="missing-research-source-reference-48071",
            output_root=tmp_path / "research_runs", prices=prices, us=US, jp=JP,
        )


def test_changed_source_reference_is_rejected_before_starting_run(tmp_path, prices):
    with pytest.raises(ValueError, match="expected_source_sha"):
        research.run_research(
            source_ref=SOURCE_REF, expected_source_sha="0" * 40,
            output_root=tmp_path / "research_runs", prices=prices, us=US, jp=JP,
        )
    assert not (tmp_path / "research_runs").exists()


def test_modified_engine_bytes_cannot_be_attributed_to_pinned_source(tmp_path, prices, monkeypatch):
    read_bytes = Path.read_bytes

    def changed_source(path):
        data = read_bytes(path)
        if path.name == "alpha_engine_live_screener.py":
            return data + b"\n# simulated uncommitted source edit\n"
        return data

    # Simulate an uncommitted edit at the filesystem read boundary; leave the
    # real engine source and its execution behavior untouched.
    monkeypatch.setattr(Path, "read_bytes", changed_source)
    with pytest.raises(ValueError, match="Engine source differs"):
        run(tmp_path, prices)
    assert not (tmp_path / "research_runs").exists()


# Retain the real dependency before fixtures install network traps. All provider
# calls in the one acquisition test remain mocked; this exercises the adapter.
REAL_DOWNLOAD = alpha.download_live_universe_prices
