"""Offline acceptance tests for the causal, read-only Dow event study."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import alpha_engine_backtest as alpha
import alpha_engine_live_screener as live
import dow_momentum_study as study


US = ["SNDK", "DELL", "US_CONTROL"]
JP = ["4062.T", "JP_CONTROL.T"]
BENCHMARKS = {"US": "SPY", "JP": "1306.T"}


@pytest.fixture(autouse=True)
def prohibit_engine_execution_or_downloads(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Dow study must not execute selection/trading logic or acquire new prices")
    for name in ("get_live_universe", "download_live_prices", "download_live_universe_prices", "run_backtest", "run_ttl_renewal_variant"):
        monkeypatch.setattr(alpha, name, forbidden)
    monkeypatch.setattr(live, "run_live_screening", forbidden)
    monkeypatch.setattr(live, "build_live_variant", forbidden)


@pytest.fixture
def prices():
    index = pd.bdate_range("2024-01-02", periods=400, name="Date")
    t = np.arange(len(index), dtype=float)
    data = {}
    for i, ticker in enumerate(US + JP):
        trend = .0008 * np.minimum(t, 260) - .0014 * np.maximum(t - 260, 0)
        data[ticker] = 100 * np.exp(trend + .10 * np.sin(t / (8 + i) + i * .7) + .018 * np.sin(t / 2.7))
    data["SPY"] = 100 * np.exp(.00025 * t + .015 * np.sin(t / 19))
    data["1306.T"] = 200 * np.exp(.0002 * t + .02 * np.sin(t / 23))
    frame = pd.DataFrame(data, index=index)
    frame.loc[index[::17], "4062.T"] = np.nan
    frame.loc[index[::29], "1306.T"] = np.nan
    return frame


def _hashes(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@pytest.fixture
def snapshot(tmp_path, prices):
    root = tmp_path / "fixed_snapshot"
    data = root / "price_data"
    data.mkdir(parents=True)
    engine = root / "engine_outputs"
    engine.mkdir()
    prices[US + JP].to_parquet(data / "price_cache.parquet")
    prices[list(BENCHMARKS.values())].to_parquet(data / "benchmark_prices.parquet")
    meta = {
        "column_order": list(prices.columns), "row_count": len(prices),
        "dtypes": {str(t): str(prices[t].dtype) for t in prices},
        "index_name": prices.index.name, "roundtrip_status": "passed",
        "sha256": {name: hashlib.sha256((data / name).read_bytes()).hexdigest()
                   for name in ("price_cache.parquet", "benchmark_prices.parquet")},
    }
    (data / "price_cache_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    metadata = {
        "status": "completed", "validation_status": "passed", "us_universe": US, "jp_universe": JP,
        "benchmarks": BENCHMARKS, "observed_start": prices.index[0].date().isoformat(),
        "observed_end": prices.index[-1].date().isoformat(), "residual_ratio": 100, "total_holdings": 6,
        "run_mode": "research_only", "git_commit_sha": "075f0985c76c2212552732534d7f3c741270a80d",
    }
    (root / "research_run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    pd.DataFrame([{"ticker": t, "region": "US" if t in US else "JP", "observations": int(prices[t].count()),
                   "downloaded": True, "excluded": False} for t in US + JP]).to_csv(data / "price_data_quality.csv", index=False)
    filled = prices.ffill()
    scores = []
    for ticker in US + JP:
        row = {"ticker": ticker, "region": "US" if ticker in US else "JP"}
        for label, n in (("3m", 63), ("6m", 126), ("12m", 252)):
            # The old engine's real endpoint convention is the oracle; no
            # screening/selection function is executed to build this fixture.
            row[f"momentum_{label}"] = alpha._window_return(filled, ticker, n)
        scores.append(row)
    pd.DataFrame(scores).to_csv(engine / "score_components.csv", index=False)
    return root


def _ordered(frame, keys):
    return frame.sort_values([c for c in keys if c in frame]).reset_index(drop=True)


def test_swings_are_strict_and_confirmed_only_after_span_observations():
    index = pd.bdate_range("2025-01-02", periods=9)
    series = pd.Series([10, 9, 8, 9, 10, 9, 7, 8, 9], index=index, dtype=float)
    pivots = study.detect_swings(series, span=2)
    low = pivots[pivots.kind == "low"]
    assert low.pivot_position.tolist() == [2, 6]
    assert low.confirmed_position.tolist() == [4, 8]
    assert low.pivot_date.tolist() == [index[2], index[6]]
    assert low.confirmed_date.tolist() == [index[4], index[8]]
    assert low.value.tolist() == [8, 7]
    tied = pd.Series([5, 4, 3, 3, 4, 5, 6], index=index[:7], dtype=float)
    assert study.detect_swings(tied, span=2).query("kind == 'low'").empty


def test_consecutive_strict_lows_are_retained_in_deterministic_order():
    index = pd.bdate_range("2025-01-02", periods=11)
    series = pd.Series([6, 5, 4, 5, 6, 6, 5, 3, 4, 5, 6], index=index, dtype=float)
    pivots = study.detect_swings(series, span=1)
    assert pivots.kind.tolist() == ["low", "low"]
    assert pivots.pivot_position.tolist() == [2, 7]
    pd.testing.assert_frame_equal(pivots, study.detect_swings(series, span=1))


@pytest.mark.parametrize("cut", [260, 289, 350])
def test_prefix_features_pivots_and_alerts_equal_full_history_without_lookahead(prices, cut):
    daily, pivots, alerts = study.compute_features(prices, US, JP, benchmarks=BENCHMARKS)
    cutoff = prices.index[cut]
    prefix_daily, prefix_pivots, prefix_alerts = study.compute_features(prices.loc[:cutoff], US, JP, benchmarks=BENCHMARKS)
    pd.testing.assert_frame_equal(
        _ordered(daily[daily.date <= cutoff], ["ticker", "series_kind", "date"]),
        _ordered(prefix_daily, ["ticker", "series_kind", "date"]), check_exact=False, rtol=1e-12, atol=1e-12,
    )
    pd.testing.assert_frame_equal(
        _ordered(pivots[pivots.confirmed_date <= cutoff], ["ticker", "series_kind", "confirmed_date", "kind"]),
        _ordered(prefix_pivots, ["ticker", "series_kind", "confirmed_date", "kind"]),
    )
    pd.testing.assert_frame_equal(
        _ordered(alerts[alerts.signal_date <= cutoff], ["ticker", "series_kind", "signal_date"]),
        _ordered(prefix_alerts, ["ticker", "series_kind", "signal_date"]), check_exact=False, rtol=1e-12, atol=1e-12,
    )


def test_alert_reference_is_known_at_signal_date_and_input_is_unchanged(prices):
    original = prices.copy(deep=True)
    daily, pivots, alerts = study.compute_features(prices, US, JP, benchmarks=BENCHMARKS)
    assert not alerts.empty
    assert (alerts.reference_confirmed_date <= alerts.signal_date).all()
    assert (alerts.reference_pivot_date < alerts.reference_confirmed_date).all()
    assert (pivots.confirmed_position - pivots.pivot_position == 5).all()
    pd.testing.assert_frame_equal(prices, original, check_exact=True)


def test_holidays_use_common_valid_observations_without_forward_filling(prices):
    daily, _, _ = study.compute_features(prices, US, JP, benchmarks=BENCHMARKS)
    expected = prices[["4062.T", "1306.T"]].dropna().index
    for kind in ("price", "relative"):
        rows = daily[(daily.ticker == "4062.T") & (daily.series_kind == kind)].sort_values("date")
        assert rows.date.tolist() == expected.tolist()
        assert rows.common_position.tolist() == list(range(len(expected)))
    assert len(expected) < len(prices["4062.T"].dropna())


def test_momentum_is_63_prices_not_63_intervals_and_engine_definition_is_separate(prices):
    daily, _, _ = study.compute_features(prices, US, JP, benchmarks=BENCHMARKS)
    common = prices[["4062.T", "1306.T"]].dropna()["4062.T"]
    row = daily[(daily.ticker == "4062.T") & (daily.series_kind == "price")].sort_values("date").iloc[-1]
    assert row.momentum_3m == pytest.approx(common.iloc[-1] / common.iloc[-63] - 1)
    assert row.momentum_6m == pytest.approx(common.iloc[-1] / common.iloc[-126] - 1)
    assert row.momentum_12m == pytest.approx(common.iloc[-1] / common.iloc[-252] - 1)
    expected_engine = alpha._window_return(prices.loc[:row.date].ffill(), "4062.T", 63)
    assert row.engine_momentum_3m == pytest.approx(expected_engine)
    assert not np.isclose(row.momentum_3m, row.engine_momentum_3m)


def test_independent_events_first_reference_and_cooldown_are_not_raw_count():
    dates = pd.bdate_range("2025-01-02", periods=60)
    rows = []
    for ticker, kind, position, reference in [
        ("A", "price", 20, 10), ("A", "price", 22, 10),
        ("A", "price", 30, 21), ("A", "price", 41, 35),
        ("A", "relative", 21, 10), ("B", "price", 21, 10),
    ]:
        rows.append({"ticker": ticker, "market": "US", "series_kind": kind,
                     "signal_date": dates[position], "date": dates[position], "common_position": position,
                     "reference_pivot_date": dates[reference], "reference_confirmed_date": dates[reference + 1]})
    raw = pd.DataFrame(rows).sample(frac=1, random_state=2)
    independent = study.independent_events(raw, cooldown=21)
    accepted = independent[independent.independent]
    primary = accepted[(accepted.ticker == "A") & (accepted.series_kind == "price")]
    assert primary.signal_date.tolist() == [dates[20], dates[41]]
    assert len(accepted) == 4
    excluded = independent[~independent.independent]
    assert set(excluded.deduplication_reason) == {"repeated_reference_low", "cooldown"}
    no_cooldown = study.independent_events(raw, cooldown=0)
    assert no_cooldown.independent.sum() == 5
    pd.testing.assert_frame_equal(independent, study.independent_events(raw.sample(frac=1, random_state=7), cooldown=21))


def _match_inputs():
    day = pd.Timestamp("2025-06-02")
    rows = []
    for ticker, below in [("A", True), ("B", True), ("C", False), ("D", False)]:
        rows.append({"date": day, "ticker": ticker, "market": "US", "series_kind": "price",
                     "momentum_3m": .2, "volatility": .1, "below_low": below,
                     "observations": 300, "common_position": 299, "reference_low": 80.})
    event = dict(rows[0], event_id="event-a", signal_date=day,
                 reference_pivot_date=day-pd.Timedelta(days=20), reference_confirmed_date=day-pd.Timedelta(days=13), benchmark="SPY")
    return pd.DataFrame([event]), pd.DataFrame(rows)


def test_controls_are_same_date_no_alert_and_ties_are_deterministic():
    events, daily = _match_inputs()
    result = study.match_controls(events, daily)
    assert result.iloc[0].control_ticker == "C"
    assert result.iloc[0].control_ticker not in {"A", "B"}
    pd.testing.assert_frame_equal(result, study.match_controls(events, daily.sample(frac=1, random_state=9)))


def test_future_features_do_not_change_controls_and_no_candidate_is_unmatched():
    events, daily = _match_inputs()
    original = study.match_controls(events, daily)
    future = daily.copy()
    future["date"] += pd.Timedelta(days=100)
    future["momentum_3m"] = [-.99, 50, -100, 200]
    future["below_low"] = ~future.below_low
    pd.testing.assert_frame_equal(original, study.match_controls(events, pd.concat([daily, future], ignore_index=True)))
    unmatched = study.match_controls(events, daily[daily.ticker.isin(["A", "B"])])
    assert unmatched.iloc[0].status == "no_control"
    assert pd.isna(unmatched.iloc[0].control_ticker)


def test_matching_enforces_market_series_and_prespecified_caliper():
    events, daily = _match_inputs()
    wrong_pool = daily.copy()
    wrong_pool.loc[wrong_pool.ticker == "C", "market"] = "JP"
    wrong_pool.loc[wrong_pool.ticker == "D", "series_kind"] = "relative"
    result = study.match_controls(events, wrong_pool)
    assert result.iloc[0].status == "no_control"
    far = daily[daily.ticker.isin(["A", "C"])].copy()
    far.loc[far.ticker == "C", "momentum_3m"] = 100.
    result = study.match_controls(events, far, caliper=.5)
    assert result.iloc[0].status == "no_control"
    assert pd.isna(result.iloc[0].control_ticker)


def test_outcomes_enter_next_observation_exit_entry_plus_h_and_keep_censoring():
    index = pd.bdate_range("2025-01-02", periods=20)
    frame = pd.DataFrame({"A": np.arange(100., 120.), "B": np.arange(200., 220.), "SPY": 100.}, index=index)
    event = pd.DataFrame([{"event_id": "e", "ticker": "A", "market": "US", "series_kind": "price", "signal_date": index[2], "benchmark": "SPY", "momentum_3m": .2}])
    match = pd.DataFrame([{"event_id": "e", "control_ticker": "B", "status": "matched", "signal_date": index[2]}])
    outcomes = study.evaluate_outcomes(event, match, frame, ["A", "B"], [], horizons=(5, 21, 63))
    observed = outcomes[(outcomes.group == "alert") & (outcomes.horizon == 5)].iloc[0]
    assert observed.status == "observed"
    assert observed.entry_date == index[3]
    assert observed.exit_date == index[8]
    assert observed.stock_return == pytest.approx(108 / 103 - 1)
    assert observed.benchmark_return == 0
    assert observed.relative_return == pytest.approx(108 / 103 - 1)
    assert observed.max_drawdown == 0
    censored = outcomes[(outcomes.group == "alert") & (outcomes.horizon == 63)].iloc[0]
    assert censored.status == "right_censored"
    assert pd.isna(censored.stock_return)
    assert pd.isna(censored.relative_return)
    assert len(outcomes) == 6


def test_outcomes_use_common_calendar_and_geometric_relative_return():
    index = pd.bdate_range("2025-01-02", periods=20)
    frame = pd.DataFrame({"A": np.arange(100., 120.), "B": np.arange(200., 220.), "SPY": 100.+.5*np.arange(20)}, index=index)
    frame.loc[index[4], "A"] = np.nan
    frame.loc[index[6], "SPY"] = np.nan
    event = pd.DataFrame([{"event_id": "e", "ticker": "A", "market": "US", "series_kind": "relative", "signal_date": index[2], "benchmark": "SPY", "momentum_3m": .2}])
    match = pd.DataFrame([{"event_id": "e", "control_ticker": "B", "status": "matched", "signal_date": index[2]}])
    outcome = study.evaluate_outcomes(event, match, frame, ["A", "B"], [], horizons=(5,)).query("group == 'alert'").iloc[0]
    common = frame[["A", "SPY"]].dropna()
    entry = common.index.get_loc(index[3])
    assert outcome.entry_date == index[3]
    assert outcome.exit_date == common.index[entry + 5]
    stock = common.A.iloc[entry + 5] / common.A.iloc[entry] - 1
    benchmark = common.SPY.iloc[entry + 5] / common.SPY.iloc[entry] - 1
    assert outcome.relative_return == pytest.approx((1 + stock) / (1 + benchmark) - 1)
    assert not np.isclose(outcome.relative_return, stock-benchmark)


def test_outcome_max_drawdown_includes_peak_to_trough_within_window():
    index = pd.bdate_range("2025-01-02", periods=10)
    frame = pd.DataFrame({"A": [90, 95, 100, 120, 90, 110, 130, 140, 145, 150], "SPY": 100.}, index=index)
    events = pd.DataFrame([{"event_id": "e", "ticker": "A", "market": "US", "series_kind": "price", "signal_date": index[1], "benchmark": "SPY", "momentum_3m": .2}])
    matches = pd.DataFrame([{"event_id": "e", "control_ticker": None, "status": "unmatched", "signal_date": index[1]}])
    outcome = study.evaluate_outcomes(events, matches, frame, ["A"], [], horizons=(5,)).query("group == 'alert'").iloc[0]
    assert outcome.entry_date == index[2]
    assert outcome.exit_date == index[7]
    assert outcome.max_drawdown == pytest.approx(-.25)
    assert outcome.stock_return == pytest.approx(.4)


def test_summary_reports_paired_difference_positive_subgroup_and_seeded_clusters():
    index = pd.bdate_range("2025-01-02", periods=100)
    t = np.arange(len(index))
    frame = pd.DataFrame({"A": 100*np.exp(.001*t), "B": 100*np.exp(-.001*t), "C": 100., "SPY": 100.}, index=index)
    events = pd.DataFrame([
        {"event_id": "e1", "ticker": "A", "market": "US", "series_kind": "price", "signal_date": index[10], "momentum_3m": .2},
        {"event_id": "e2", "ticker": "B", "market": "US", "series_kind": "price", "signal_date": index[15], "momentum_3m": -.1},
        {"event_id": "e3", "ticker": "A", "market": "US", "series_kind": "price", "signal_date": index[20], "momentum_3m": .3},
    ])
    matches = events[["event_id", "signal_date"]].assign(control_ticker="C", status="matched")
    outcomes = study.evaluate_outcomes(events, matches, frame, ["A", "B", "C"], [], horizons=(21,))
    summary, uncertainty = study.summarize_outcomes(outcomes, bootstrap_samples=24, seed=48071)
    again_summary, again_uncertainty = study.summarize_outcomes(outcomes, bootstrap_samples=24, seed=48071)
    pd.testing.assert_frame_equal(summary, again_summary)
    pd.testing.assert_frame_equal(uncertainty, again_uncertainty)
    overall = summary.query("subgroup == 'all' and market == 'ALL' and period == 'ALL' and group == 'paired_difference'").iloc[0]
    expected = (2*np.expm1(.021) + np.expm1(-.021))/3
    assert overall.valid_count == overall.total_count == 3
    assert bool(overall.primary_horizon)
    assert overall.mean_relative_return == pytest.approx(expected)
    assert overall.underperformance_probability == pytest.approx(1/3)
    positive = summary.query("subgroup == 'positive_momentum_3m' and market == 'ALL' and period == 'ALL' and group == 'paired_difference'").iloc[0]
    assert positive.valid_count == 2
    assert positive.mean_relative_return == pytest.approx(np.expm1(.021))
    clustered = uncertainty.query("subgroup == 'all' and market == 'ALL' and period == 'ALL'").iloc[0]
    assert clustered.ticker_cluster_count == 2
    assert clustered.date_cluster_count == 3
    assert clustered.seed == 48071


def test_benchmark_correction_is_guarded_and_original_is_never_changed():
    index = pd.bdate_range("2026-03-16", "2026-04-14")
    source = pd.DataFrame({"1306.T": 1000. + np.arange(len(index)), "OTHER": 55.}, index=index)
    suspect = pd.to_datetime(["2026-03-30", "2026-03-31"])
    source.loc[suspect, "1306.T"] /= 10
    original = source.copy(deep=True)
    corrected, log = study.apply_benchmark_correction(source)
    pd.testing.assert_frame_equal(source, original, check_exact=True)
    expected = original.copy(deep=True)
    expected.loc[suspect, "1306.T"] *= 10
    pd.testing.assert_frame_equal(corrected, expected, check_exact=True)
    assert len(log) >= 2
    assert abs(corrected["1306.T"].pct_change().dropna()).max() < .01
    with pytest.raises(ValueError):
        study.apply_benchmark_correction(corrected)
    unexpected = original.copy()
    unexpected.loc[suspect[0], "1306.T"] = 5000.
    with pytest.raises(ValueError):
        study.apply_benchmark_correction(unexpected)


def test_correction_dates_absent_is_explicit_no_change(prices):
    corrected, log = study.apply_benchmark_correction(prices)
    pd.testing.assert_frame_equal(corrected, prices, check_exact=True)
    assert len(log) == 0


def test_study_retains_source_bytes_verifies_outputs_and_is_reproducible(tmp_path, snapshot):
    original = _hashes(snapshot)
    first = study.run_study(snapshot, tmp_path / "analysis", bootstrap_samples=12, seed=48071)
    before = _hashes(first)
    second = study.run_study(snapshot, tmp_path / "analysis", bootstrap_samples=12, seed=48071)
    assert first != second
    assert _hashes(snapshot) == original
    assert _hashes(first) == before
    required = {"daily_features.parquet", "swing_events.csv", "alerts_raw.csv", "alerts_deduplicated.csv",
                "control_matches.csv", "event_outcomes.csv", "summary_statistics.csv", "sensitivity_analysis.csv",
                "data_quality_warnings.csv", "study_metadata.json", "study_report.md"}
    assert {p.name for p in first.iterdir()} >= required
    assert (first / "charts").is_dir()
    required_charts = {"exploratory_SNDK.svg", "exploratory_DELL.svg", "exploratory_4062.T.svg",
                       "alert_frequency.svg", "relative_return_distribution.svg", "momentum_only.svg",
                       "alert_momentum_outcomes.svg"}
    assert {p.name for p in (first / "charts").iterdir()} >= required_charts
    assert all((first / "charts" / name).stat().st_size > 100 for name in required_charts)
    metadata = _json(first / "study_metadata.json")
    assert metadata["status"] == "completed"
    assert metadata["validation_status"] == "passed"
    assert metadata["input_unchanged"] is True
    assert metadata["source_snapshot"] == str(snapshot.resolve())
    assert metadata["candidate_counts"] == {"US": len(US), "JP": len(JP), "total": len(US)+len(JP), "snapshot_ranked": len(US)+len(JP)}
    assert metadata["analyzed_ticker_count"] == len(US)+len(JP)
    assert metadata["excluded_ticker_count"] == 0
    assert len(metadata["input_hashes"]) == 6
    for name, digest in metadata["input_hashes"].items():
        assert digest == original[name]
    assert metadata["parameters"]["primary_horizon"] == 21
    assert metadata["parameters"]["horizons"] == [5, 21, 63]
    assert metadata["parameters"]["span"] == 5
    assert metadata["parameters"]["threshold"] == 0
    assert metadata["parameters"]["cooldown"] == 21
    assert metadata["parameters"]["seed"] == 48071
    assert metadata["momentum_reproduction"] == {"matched": 3*(len(US)+len(JP))}
    assert metadata["correction_history"] == []
    events = pd.read_csv(first / "alerts_deduplicated.csv")
    assert metadata["event_counts"]["independent"] == len(events)
    matches = pd.read_csv(first / "control_matches.csv")
    assert metadata["event_counts"]["matched"] == (matches.status == "matched").sum()
    sensitivity = pd.read_csv(first / "sensitivity_analysis.csv")
    assert set(sensitivity.specification) == {"primary_corrected", "uncorrected", "exclude_influence", "span3", "span10", "threshold_minus1pct", "threshold_minus3pct", "no_cooldown"}
    manifest = _json(first / "file_manifest.json")
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((first / name).read_bytes()).hexdigest() == digest
    for name in required - {"study_metadata.json", "study_report.md", "daily_features.parquet"}:
        pd.testing.assert_frame_equal(pd.read_csv(first / name), pd.read_csv(second / name))
    pd.testing.assert_frame_equal(pd.read_parquet(first / "daily_features.parquet"), pd.read_parquet(second / "daily_features.parquet"))


def test_changed_snapshot_hash_stops_without_repair_or_redownload(tmp_path, snapshot):
    source = snapshot / "price_data" / "price_cache.parquet"
    source.write_bytes(source.read_bytes() + b"changed-input")
    damaged = _hashes(snapshot)
    with pytest.raises((ValueError, study.StudyRunError)):
        study.run_study(snapshot, tmp_path / "analysis", bootstrap_samples=12)
    assert _hashes(snapshot) == damaged


def test_exclusion_window_covers_all_252_returns_and_clears_after_253_prices():
    index = pd.bdate_range("2025-01-02", periods=800)
    t = np.arange(len(index))
    frame = pd.DataFrame({"J": 100*np.exp(.001*t), "1306.T": 1000., "SPY": 100.}, index=index)
    frame.loc[pd.to_datetime(["2026-03-30", "2026-03-31"]), "1306.T"] = 100.
    daily, _, _ = study.compute_features(frame, [], ["J"])
    row = daily[(daily.ticker == "J") & (daily.series_kind == "price")].set_index("date")
    bad_position = index.get_loc(pd.Timestamp("2026-03-31"))
    # No stock-price pivot exists in this monotone fixture, so this isolates the
    # volatility's 252 returns (253 input prices) rather than pivot support.
    assert row.reference_low.isna().all()
    assert bool(row.loc[index[bad_position+252], "excluded_influence"])
    assert not bool(row.loc[index[bad_position+253], "excluded_influence"])


def test_anomaly_sensitivity_excludes_future_outcomes_without_changing_controls():
    index = pd.bdate_range("2024-11-01", periods=450)
    t = np.arange(len(index))
    frame = pd.DataFrame({
        "J1": 100*np.exp(.0003*t-.0012*np.maximum(t-240, 0)+.12*np.sin(t/6)),
        "J2": 100*np.exp(.0003*t-.0012*np.maximum(t-240, 0)+.12*np.sin(t/6+.5)),
        "1306.T": 1000.+.05*t, "SPY": 100.,
    }, index=index)
    frame.loc[pd.to_datetime(["2026-03-30", "2026-03-31"]), "1306.T"] /= 10
    original = frame.copy(deep=True)
    _, _, _, accepted, ordinary_matches, ordinary_outcomes = study._case(frame, [], ["J1", "J2"])
    _, _, _, excluded, safe_matches, safe_outcomes = study._case(frame, [], ["J1", "J2"], exclude_influence=True)
    affected = excluded[excluded.excluded_influence]
    assert not affected.empty
    assert not affected.analysis_eligible.any()
    marked = safe_outcomes[safe_outcomes.status == "excluded_influence"]
    assert not marked.empty
    assert marked[["stock_return", "benchmark_return", "relative_return", "max_drawdown"]].isna().all().all()
    # Future outcome contamination is assessed AFTER matching. It must not
    # replace a control using knowledge of which future windows touch the dates.
    expected = ordinary_matches[ordinary_matches.event_id.isin(safe_matches.event_id)].reset_index(drop=True)
    pd.testing.assert_frame_equal(safe_matches.reset_index(drop=True), expected)
    summary, _ = study.summarize_outcomes(safe_outcomes, bootstrap_samples=0)
    primary = summary.query("subgroup == 'all' and market == 'ALL' and period == 'ALL' and group == 'alert'")
    alert = safe_outcomes[safe_outcomes.group == "alert"]
    for row in primary.itertuples():
        horizon = alert[(alert.horizon == row.horizon) & (alert.series_kind == row.series_kind)]
        assert row.excluded_influence_count == (horizon.status == "excluded_influence").sum()
        assert row.right_censored_count == (horizon.status == "right_censored").sum()
        assert row.missing_signal_observation_count == (horizon.status == "missing_signal_observation").sum()
        assert row.total_count == (row.valid_count + row.right_censored_count + row.unmatched_count +
                                   row.excluded_influence_count + row.missing_signal_observation_count)
    pd.testing.assert_frame_equal(frame, original, check_exact=True)


def test_zero_alert_snapshot_completes_without_inventing_events(tmp_path, snapshot):
    data = snapshot / "price_data"
    original_prices = pd.read_parquet(data / "price_cache.parquet")
    index = original_prices.index
    t = np.arange(len(index))
    monotone = pd.DataFrame({ticker: 100*np.exp((.002+i*.0001)*t) for i, ticker in enumerate(US+JP)}, index=index)
    benchmark = pd.DataFrame({"SPY": 100*np.exp(.0001*t), "1306.T": 200*np.exp(.0001*t)}, index=index)
    monotone.to_parquet(data / "price_cache.parquet")
    benchmark.to_parquet(data / "benchmark_prices.parquet")
    complete = pd.concat([monotone, benchmark], axis=1)
    meta = _json(data / "price_cache_metadata.json")
    meta["sha256"] = {name: hashlib.sha256((data / name).read_bytes()).hexdigest()
                      for name in ("price_cache.parquet", "benchmark_prices.parquet")}
    (data / "price_cache_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    scores = pd.read_csv(snapshot / "engine_outputs" / "score_components.csv")
    for label, n in (("3m",63),("6m",126),("12m",252)):
        scores[f"momentum_{label}"] = [alpha._window_return(complete, ticker, n) for ticker in scores.ticker]
    scores.to_csv(snapshot / "engine_outputs" / "score_components.csv", index=False)
    before = _hashes(snapshot)
    out = study.run_study(snapshot, tmp_path / "analysis", bootstrap_samples=0, seed=48071)
    assert _json(out / "study_metadata.json")["status"] == "completed"
    assert pd.read_csv(out / "alerts_raw.csv").empty
    assert pd.read_csv(out / "alerts_deduplicated.csv").empty
    assert pd.read_csv(out / "event_outcomes.csv").empty
    assert _hashes(snapshot) == before


def test_output_inside_snapshot_is_rejected_without_adding_or_changing_files(snapshot):
    original = _hashes(snapshot)
    with pytest.raises((ValueError, study.StudyRunError)):
        study.run_study(snapshot, snapshot / "analysis", bootstrap_samples=12)
    assert _hashes(snapshot) == original


@pytest.mark.parametrize("defect", ["duplicate_dates", "invalid_price", "overlapping_universe", "incomplete_snapshot"])
def test_resealed_but_invalid_snapshot_still_fails_data_validation(tmp_path, snapshot, defect):
    if defect in {"duplicate_dates", "invalid_price"}:
        path = snapshot / "price_data" / "price_cache.parquet"
        frame = pd.read_parquet(path)
        if defect == "duplicate_dates":
            index = frame.index.to_list()
            index[1] = index[0]
            frame.index = pd.DatetimeIndex(index, name=frame.index.name)
        else:
            frame.iloc[0, 0] = 0.
        frame.to_parquet(path)
        cache_meta = _json(snapshot / "price_data" / "price_cache_metadata.json")
        cache_meta["sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        (snapshot / "price_data" / "price_cache_metadata.json").write_text(json.dumps(cache_meta), encoding="utf-8")
    else:
        path = snapshot / "research_run_metadata.json"
        meta = _json(path)
        if defect == "overlapping_universe":
            meta["jp_universe"].append(US[0])
        else:
            meta["status"] = "incomplete"
        path.write_text(json.dumps(meta), encoding="utf-8")
    original = _hashes(snapshot)
    with pytest.raises((ValueError, study.StudyRunError)):
        study.run_study(snapshot, tmp_path / "analysis", bootstrap_samples=12)
    assert _hashes(snapshot) == original
