"""Offline acceptance tests for market-clock Dow structure research v0.2."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import alpha_engine_research_runner as research
import dow_momentum_study as v01
import dow_structure_study as study
from tests.test_dow_momentum_study import (
    JP, US, _hashes, _json, prices, snapshot,
    prohibit_engine_execution_or_downloads,
)


STATES = {"HH/HL", "LH/LL", "LH/HL", "HH/LL"}


@pytest.mark.parametrize("previous_high,last_high,previous_low,last_low,expected", [
    (10, 11, 5, 6, "HH/HL"), (10, 9, 5, 4, "LH/LL"),
    (10, 9, 5, 6, "LH/HL"), (10, 11, 5, 4, "HH/LL"),
    (10, 10, 5, 6, "unclassified"), (10, 11, 5, 5, "unclassified"),
    (None, 11, 5, 6, "unclassified"), (10, np.nan, 5, 6, "unclassified"),
    (10, 11, None, 6, "unclassified"), (10, 11, 5, np.nan, "unclassified"),
])
def test_four_states_are_strict_and_ties_or_missing_are_unclassified(previous_high, last_high, previous_low, last_low, expected):
    assert study.classify_state(previous_high, last_high, previous_low, last_low) == expected


def _ordered(frame):
    return frame.sort_values(["ticker", "series_kind", "date"]).reset_index(drop=True)


@pytest.mark.parametrize("cut", [260, 315, 350])
def test_daily_states_and_freshness_are_prefix_causal(prices, cut):
    original = prices.copy(deep=True)
    full = study.compute_structure_features(prices, US, JP)
    cutoff = prices.index[cut]
    prefix = study.compute_structure_features(prices.loc[:cutoff], US, JP)
    pd.testing.assert_frame_equal(_ordered(full[full.date <= cutoff]), _ordered(prefix), check_exact=False, rtol=1e-12, atol=1e-12)
    pd.testing.assert_frame_equal(prices, original, check_exact=True)
    for column in ["reference_high_confirmed_date", "reference_low_confirmed_date"]:
        known = full[full[column].notna()]
        assert (known[column] <= known.date).all()


def test_equal_recent_highs_remain_unclassified_until_new_high_is_confirmed():
    values = np.r_[np.linspace(1., 9., 250), [10., 8., 12., 7., 12., 6., 13., 5., 11., 4.]]
    index = pd.bdate_range("2024-01-02", periods=len(values))
    frame = pd.DataFrame({"X": values, "SPY": 100., "1306.T": 100.}, index=index)
    daily = study.compute_structure_features(frame, ["X"], [], span=1)
    stock = daily.query("ticker == 'X' and series_kind == 'price'").set_index("date")
    # High12 at position254 is confirmed255. High13 at256 is still future
    # information until257; the equal12/12 pair must not be labelled HH or LH.
    assert stock.loc[index[255], "state"] == "unclassified"
    assert stock.loc[index[256], "state"] == "unclassified"
    assert stock.loc[index[257], "state"] == "HH/LL"


def test_structure_uses_latest_two_confirmed_pivots_and_common_clock_features(prices):
    daily = study.compute_structure_features(prices, US, JP)
    common = prices[["4062.T", "1306.T"]].dropna()
    series = common["4062.T"]
    row = daily[(daily.ticker == "4062.T") & (daily.series_kind == "price")].sort_values("date").iloc[-1]
    pivots = v01.detect_swings(series, span=5)
    highs = pivots[(pivots.kind == "high") & (pivots.confirmed_date <= row.date)].tail(2)
    lows = pivots[(pivots.kind == "low") & (pivots.confirmed_date <= row.date)].tail(2)
    assert len(highs) == len(lows) == 2
    assert row.previous_confirmed_high == highs.value.iloc[0]
    assert row.previous_confirmed_low == lows.value.iloc[0]
    assert row.reference_high == highs.value.iloc[1]
    assert row.reference_low == lows.value.iloc[1]
    assert row.high_age == row.common_position-highs.confirmed_position.iloc[1]
    assert row.low_age == row.common_position-lows.confirmed_position.iloc[1]
    assert row.distance_to_high == pytest.approx(row.series_value / highs.value.iloc[1] - 1)
    assert row.distance_to_low == pytest.approx(row.series_value / lows.value.iloc[1] - 1)
    for label, window in [("3m", 63), ("6m", 126), ("12m", 252)]:
        assert row[f"momentum_{label}"] == pytest.approx(series.iloc[-1]/series.iloc[-window]-1)
    relative = common["4062.T"] / common["1306.T"]
    assert row.relative_momentum_12m == pytest.approx(relative.iloc[-1]/relative.iloc[-252]-1)
    assert not np.isclose(row.momentum_3m, row.engine_momentum_3m)


def test_landmark_calendar_is_fixed_by_market_benchmark_and_phase():
    index = pd.bdate_range("2023-12-01", periods=150)
    benchmark = pd.Series(100., index=index)
    benchmark.iloc[[29, 42, 65]] = np.nan
    valid = benchmark.loc["2024-01-01":].dropna().index
    assert study.landmark_dates(benchmark).tolist() == valid[::21].tolist()
    assert study.landmark_dates(benchmark, interval=21, phase=7).tolist() == valid[7::21].tolist()
    pd.testing.assert_index_equal(study.landmark_dates(benchmark), study.landmark_dates(benchmark.copy()))


def test_snapshot_without_post2024_market_dates_fails_explicitly(prices):
    historical = prices.copy(deep=True)
    historical.index = pd.bdate_range("2022-01-03", periods=len(historical), name=historical.index.name)
    assert study.landmark_dates(historical.SPY).empty
    daily = study.compute_structure_features(historical, US, JP)
    with pytest.raises(ValueError, match="No market landmark dates"):
        study.build_landmarks(daily, historical, US, JP)


def test_scheduled_missing_stock_is_retained_and_cannot_shift_its_clock(prices):
    frame = prices.copy(deep=True)
    scheduled = study.landmark_dates(frame.SPY)
    missing_date = scheduled[-1]
    frame.loc[missing_date, US[0]] = np.nan
    daily = study.compute_structure_features(frame, US, JP)
    landmarks = study.build_landmarks(daily, frame, US, JP)
    for ticker in US:
        for kind in ("price", "relative"):
            rows = landmarks[(landmarks.ticker == ticker) & (landmarks.series_kind == kind)]
            assert rows.date.tolist() == scheduled.tolist()
    missing = landmarks[(landmarks.ticker == US[0]) & (landmarks.date == missing_date)]
    assert len(missing) == 2
    assert set(missing.state) == {"unclassified"}
    assert not missing.history_eligible.any()
    assert missing.exclusion_reason.notna().all() and missing.exclusion_reason.str.len().gt(0).all()
    assert landmarks.groupby(["market", "date"]).market_landmark_number.nunique().eq(1).all()


def test_253_common_observations_and_two_pivots_per_side_are_required(prices):
    daily = study.compute_structure_features(prices, US, JP)
    landmarks = study.build_landmarks(daily, prices, US, JP, interval=1)
    stock = landmarks[(landmarks.ticker == US[0]) & (landmarks.series_kind == "price")].set_index("date")
    assert not bool(stock.loc[prices.index[251], "history_eligible"])
    assert stock.loc[prices.index[251], "state"] == "unclassified"
    assert bool(stock.loc[prices.index[252], "history_eligible"])
    assert stock.loc[prices.index[252], "state"] in STATES
    # A monotonically increasing candidate has enough prices but no confirmed
    # high/low pair; it must remain in the count as unclassified.
    monotone = prices.copy()
    monotone[US[0]] = np.arange(100., 100.+len(monotone))
    monotone_daily = study.compute_structure_features(monotone, US, JP)
    observed = study.build_landmarks(monotone_daily, monotone, US, JP)
    candidate = observed[(observed.ticker == US[0]) & (observed.series_kind == "price") & observed.history_eligible]
    assert not candidate.empty
    assert set(candidate.state) == {"unclassified"}
    assert not candidate.classified.any()


@pytest.mark.parametrize("positive", [True, False])
def test_top10_ranks_all_history_eligible_names_including_unclassified_and_uses_ceil(positive):
    index = pd.bdate_range("2024-01-02", periods=400)
    tickers = [f"U{i:02d}" for i in range(11)]
    t = np.arange(len(index))
    sign = 1 if positive else -1
    data = {ticker: 100*np.exp(sign*.001*(min(i, 9)+1)*t) for i, ticker in enumerate(tickers)}
    frame = pd.DataFrame({**data, "SPY": 100., "1306.T": 100.}, index=index)
    daily = study.compute_structure_features(frame, tickers, [])
    landmarks = study.build_landmarks(daily, frame, tickers, [])
    day = study.landmark_dates(frame.SPY)[-1]
    pool = landmarks[(landmarks.date == day) & (landmarks.series_kind == "price")]
    assert len(pool) == 11
    assert set(pool.state) == {"unclassified"}
    assert set(pool.rank_population) == {11}
    if positive:
        assert pool.loc[pool.raw_top10, "ticker"].tolist() == ["U09", "U10"]
        assert pool.set_index("ticker").loc["U09", "rank_12m"] == 1
        assert pool.set_index("ticker").loc["U10", "rank_12m"] == 2
    else:
        assert not pool.raw_top10.any()
    paired = landmarks[landmarks.date == day].pivot(index="ticker", columns="series_kind", values="raw_top10")
    assert paired.price.equals(paired.relative)


def test_historical_top10_and_landmarks_are_unchanged_by_future_tail(prices):
    cutoff = prices.index[294]
    full_daily = study.compute_structure_features(prices, US, JP)
    full = study.build_landmarks(full_daily, prices, US, JP)
    changed = prices.copy()
    changed.loc[changed.index > cutoff, US[0]] *= 100
    changed.loc[changed.index > cutoff, US[1]] /= 100
    changed_daily = study.compute_structure_features(changed, US, JP)
    again = study.build_landmarks(changed_daily, changed, US, JP)
    pd.testing.assert_frame_equal(_ordered(full[full.date <= cutoff]), _ordered(again[again.date <= cutoff]), check_exact=False, rtol=1e-12, atol=1e-12)


def _matching_pool():
    day = pd.Timestamp("2025-06-02")
    rows = []
    for ticker, state in [("A", "HH/HL"), ("B", "HH/HL"), ("C", "LH/LL"), ("D", "HH/LL"), ("E", "unclassified")]:
        rows.append({"date": day, "signal_date": day, "event_id": ticker+"-price", "sample_id": ticker,
                     "ticker": ticker, "market": "US", "series_kind": "price", "state": state,
                     "momentum_3m": .2, "momentum_6m": .3, "momentum_12m": .4, "volatility": .1,
                     "observations": 300, "history_eligible": True, "classified": state in STATES,
                     "raw_top10": False, "relative_top10": False, "exclusion_reason": "",
                     "momentum_3m_sign": "positive", "market_landmark_number": 12})
    return pd.DataFrame(rows)


def test_matching_requires_different_state_and_all_four_features_with_deterministic_ties():
    pool = _matching_pool()
    matches, balance = study.match_structure_controls(pool)
    focal = matches[matches.ticker == "A"].iloc[0]
    assert focal.status == "matched"
    assert focal.control_ticker == "C"
    assert focal.control_state != focal.state
    again, again_balance = study.match_structure_controls(pool.sample(frac=1, random_state=7))
    pd.testing.assert_frame_equal(matches, again)
    pd.testing.assert_frame_equal(balance, again_balance)
    assert set(balance.feature) == {"momentum_3m", "momentum_6m", "momentum_12m", "volatility"}


@pytest.mark.parametrize("feature", ["momentum_3m", "momentum_6m", "momentum_12m", "volatility"])
def test_featurewise_caliper_cannot_ignore_any_control_variable(feature):
    pool = _matching_pool().query("ticker in ['A','C']").copy()
    pool.loc[pool.ticker == "C", feature] = 100.
    matches, _ = study.match_structure_controls(pool, caliper=.5)
    focal = matches[matches.ticker == "A"].iloc[0]
    assert focal.status == "no_control"
    assert pd.isna(focal.control_ticker)


def test_future_features_cannot_change_current_matching_and_unclassified_is_not_control():
    pool = _matching_pool()
    matches, balance = study.match_structure_controls(pool)
    future = pool.copy()
    future["date"] += pd.Timedelta(days=100)
    future["signal_date"] = future.date
    future["event_id"] += "-future"
    future["sample_id"] += "-future"
    future["momentum_12m"] = [100., -100., 5., 9., -3.]
    again, _ = study.match_structure_controls(pd.concat([pool, future], ignore_index=True))
    day = pool.date.iloc[0]
    pd.testing.assert_frame_equal(matches, again[again.signal_date == day].reset_index(drop=True))
    unavailable, _ = study.match_structure_controls(pool.query("ticker in ['A','B','E']"))
    assert unavailable.loc[unavailable.ticker == "A", "status"].iloc[0] == "no_control"


def test_outcomes_separate_landmark_close_from_entry_and_preserve_censoring():
    index = pd.bdate_range("2024-01-02", periods=280)
    t = np.arange(len(index))
    frame = pd.DataFrame({"A": 100.+t, "B": 200.+t, "SPY": 100.+.5*t}, index=index)
    landmark = _matching_pool().query("ticker == 'A'").copy()
    landmark["date"] = landmark["signal_date"] = index[260]
    landmark["raw_top10"] = True
    matches = pd.DataFrame([{"event_id": "A-price", "ticker": "A", "state": "HH/HL", "control_ticker": "B", "control_state": "LH/LL", "status": "matched",
                             "momentum_3m":.2, "momentum_6m":.3, "momentum_12m":.4, "volatility":.1,
                             "control_momentum_3m":.21, "control_momentum_6m":.32, "control_momentum_12m":.43, "control_volatility":.11}])
    outcomes = study.structure_outcomes(landmark, matches, frame, ["A", "B"], [])
    observed = outcomes[(outcomes.group == "alert") & (outcomes.horizon == 5)].iloc[0]
    assert observed.entry_date == index[261]
    assert observed.exit_date == index[266]
    stock = frame.A.iloc[266]/frame.A.iloc[261]-1
    benchmark = frame.SPY.iloc[266]/frame.SPY.iloc[261]-1
    assert observed.stock_return == pytest.approx(stock)
    assert observed.relative_return == pytest.approx((1+stock)/(1+benchmark)-1)
    assert observed.state == "HH/HL"
    assert bool(observed.raw_top10)
    control = outcomes[(outcomes.group == "control") & (outcomes.horizon == 5)].iloc[0]
    assert control.focal_state == control.state == "HH/HL"
    assert control.observed_ticker_state == control.control_state == "LH/LL"
    assert control.observed_ticker_state != control.focal_state
    paired = study._paired_comparisons(outcomes, matches).set_index("horizon")
    assert paired.loc[5, "pair_status"] == "observed"
    assert paired.loc[5, "state_focal"] == "HH/HL"
    assert paired.loc[5, "state_control"] == paired.loc[5, "control_state"] == "LH/LL"
    for feature in ("momentum_3m","momentum_6m","momentum_12m","volatility"):
        assert paired.loc[5,feature+"_focal"] == pytest.approx(matches[feature].iloc[0])
        assert paired.loc[5,feature+"_control"] == pytest.approx(matches["control_"+feature].iloc[0])
        assert paired.loc[5,feature+"_control"] != paired.loc[5,feature+"_focal"]
    assert bool(paired.loc[5,"raw_top10_focal"])
    for copied_context in ("raw_top10_control","relative_top10_control","momentum_3m_sign_control","event_momentum_3m_control"):
        assert copied_context not in paired.columns
    censored = outcomes[(outcomes.group == "alert") & (outcomes.horizon == 63)].iloc[0]
    assert censored.status == "right_censored"
    assert pd.isna(censored.relative_return)
    assert len(outcomes) == 6


def _summary_outcomes():
    """Small hand-labelled cohort: missing and intentional exclusions differ."""
    events = [
        ("e1", "A", "US", "price", "HH/HL", "observed", "observed", True, False, .10, .05),
        ("e2", "B", "US", "price", "HH/HL", "observed", "right_censored", False, False, -.05, np.nan),
        ("e3", "C", "US", "price", "HH/HL", "right_censored", "excluded_influence", True, False, np.nan, np.nan),
        ("e4", "D", "US", "price", "HH/HL", "excluded_influence", "unmatched", False, False, np.nan, np.nan),
        ("e5", "J.T", "JP", "relative", "LH/LL", "observed", "observed", False, True, -.02, .01),
    ]
    rows = []
    for i, (event_id, ticker, market, kind, state, focal_status, control_status, raw_top10, relative_top10, focal_return, control_return) in enumerate(events):
        for horizon in (5,21,63):
            for group, status, value in (("alert", focal_status, focal_return), ("control", control_status, control_return)):
                rows.append({"event_id":event_id, "ticker":ticker if group=="alert" else (None if status=="unmatched" else "CONTROL"),
                             "event_ticker":ticker, "signal_date":pd.Timestamp("2025-01-02")+pd.Timedelta(days=i),
                             "market":market, "series_kind":kind, "state":state, "group":group, "horizon":horizon,
                             "status":status, "raw_top10":raw_top10, "relative_top10":relative_top10,
                             "momentum_3m_sign":"positive" if i%2==0 else "negative",
                             "stock_return":value, "relative_return":value, "max_drawdown":-.03 if status=="observed" else np.nan})
    return pd.DataFrame(rows)


def test_reduced_sensitivity_summary_equals_prespecified_slice_of_full_summary():
    outcomes = _summary_outcomes()
    full, full_uncertainty = study.summarize_structure(outcomes, bootstrap_samples=12, seed=48071)
    sensitivity, uncertainty = study.summarize_structure(outcomes, bootstrap_samples=12, seed=48071, sensitivity=True)
    expected = full.loc[full.cohort.isin(["all","raw_top10"]) & full.momentum_3m_sign.eq("all") &
                        full.period.eq("ALL") & full.horizon.eq(21)].reset_index(drop=True)
    pd.testing.assert_frame_equal(sensitivity, expected)
    pd.testing.assert_frame_equal(uncertainty, full_uncertainty)
    assert set(sensitivity.horizon) == {21}
    assert set(full.horizon) == {5,21,63}
    assert "relative_top10" in set(full.cohort)
    assert "relative_top10" not in set(sensitivity.cohort)
    assert set(sensitivity.series_kind) == {"price","relative"}
    assert set(sensitivity.market) == {"ALL","US","JP"}


def test_summary_distinguishes_intentional_exclusion_from_right_censoring():
    summary, _ = study.summarize_structure(_summary_outcomes(), bootstrap_samples=0, sensitivity=True)
    cohort = summary.query("cohort == 'all' and market == 'US' and series_kind == 'price' and state == 'HH/HL'").set_index("group")
    raw = cohort.loc["unadjusted"]
    assert raw.total_count == 4
    assert raw.valid_count == 2
    assert raw.right_censored_count == 1
    assert raw.excluded_influence_count == 1
    # e3 has a censored focal and an intentionally excluded control; the pair
    # is explicitly excluded, rather than counted twice or called censoring.
    for group in ("focal_matched","control_matched","paired_difference"):
        row = cohort.loc[group]
        assert row.assigned_count == 3
        assert row.valid_count == row.right_censored_count == row.excluded_influence_count == row.unmatched_count == 1
        assert row.other_unobserved_count == 0
        assert row.total_count == row.valid_count+row.right_censored_count+row.excluded_influence_count+row.unmatched_count+row.other_unobserved_count
    assert cohort.loc["paired_difference", "mean_relative_return"] == pytest.approx(.05)


def test_read_only_pipeline_is_reproducible_and_preserves_v01_and_engine_files(tmp_path, snapshot):
    source_hashes = _hashes(snapshot)
    repo = Path(study.__file__).resolve().parent
    protected = [*research.ENGINE_FILES, "alpha_engine_research_runner.py", "dow_momentum_study.py"]
    engine_hashes = {name: hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in protected}
    first = study.run_study(snapshot, tmp_path/"analysis", bootstrap_samples=12, seed=48071)
    preserved = _hashes(first)
    second = study.run_study(snapshot, tmp_path/"analysis", bootstrap_samples=12, seed=48071)
    assert first != second
    assert _hashes(snapshot) == source_hashes
    assert _hashes(first) == preserved
    assert {name: hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in protected} == engine_hashes
    required = {"daily_structure_features.parquet", "landmark_observations.csv", "structure_distribution.csv",
                "structure_transitions.csv", "structure_outcomes.csv", "matched_structure_comparisons.csv",
                "matching_balance.csv", "extreme_momentum_analysis.csv", "freshness_analysis.csv",
                "freshness_outcomes.csv", "daily_state_durations.csv",
                "sensitivity_analysis.csv", "uncertainty.csv", "data_quality_warnings.csv",
                "study_metadata.json", "study_report.md"}
    assert {p.name for p in first.iterdir()} >= required
    assert (first/"charts").is_dir()
    chart_names = {f"{name}_{kind}.svg" for name in ("state_composition", "state_transitions", "state_outcomes", "momentum_structure")
                   for kind in ("price", "relative")}
    assert {p.name for p in (first/"charts").iterdir()} >= chart_names
    assert all((first/"charts"/name).stat().st_size > 100 for name in chart_names)
    meta = _json(first/"study_metadata.json")
    assert meta["status"] == "completed"
    assert meta["validation_status"] == "passed"
    assert meta["input_unchanged"] and meta["protected_source_unchanged"]
    assert meta["protected_source_sha256"] == engine_hashes
    assert meta["source_snapshot"] == str(snapshot.resolve())
    assert len(meta["input_hashes"]) == 6
    for name, digest in meta["input_hashes"].items():
        assert digest == source_hashes[name]
    assert meta["candidate_counts"] == {"US":len(US), "JP":len(JP), "total":len(US)+len(JP), "snapshot_ranked":len(US)+len(JP)}
    assert meta["unique_tickers"] == len(US)+len(JP)
    assert meta["momentum_reproduction"] == {"matched":3*(len(US)+len(JP))}
    settings = meta["parameters"]
    assert settings["span"] == 5
    assert settings["landmark_start"] == "2024-01-01"
    assert settings["landmark_interval"] == 21
    assert settings["landmark_phase"] == 0
    assert settings["primary_series"] == "price"
    assert settings["primary_horizon"] == 21
    assert settings["horizons"] == [5,21,63]
    assert settings["momentum_price_windows"] == [63,126,252]
    assert settings["minimum_common_prices"] == 253
    assert settings["minimum_highs"] == settings["minimum_lows"] == 2
    assert settings["matching_features"] == ["momentum_3m", "momentum_6m", "momentum_12m", "volatility"]
    assert settings["caliper_per_feature"] == .5
    assert settings["seed"] == 48071
    loaded = research.load_research_prices(snapshot)
    expected_rows = 2*(len(US)*len(study.landmark_dates(loaded.SPY)) + len(JP)*len(study.landmark_dates(loaded["1306.T"])))
    observations = pd.read_csv(first/"landmark_observations.csv")
    assert len(observations) == expected_rows == meta["observation_counts"]["scheduled_rows"]
    assert meta["observation_counts"]["classified_rows"]+meta["observation_counts"]["unclassified_rows"] == expected_rows
    distribution = pd.read_csv(first/"structure_distribution.csv")
    assert set(distribution.state) >= STATES
    comparisons = pd.read_csv(first/"matched_structure_comparisons.csv")
    assigned = comparisons[comparisons.control_state.notna()]
    assert assigned.state_control.eq(assigned.control_state).all()
    assert assigned.state_control.ne(assigned.state_focal).all()
    sensitivity = pd.read_csv(first/"sensitivity_analysis.csv")
    assert set(sensitivity.specification) == {"primary_corrected", "span3", "span10", "phase7", "phase14", "uncorrected", "exclude_influence"}
    manifest = _json(first/"file_manifest.json")
    for name,digest in manifest["files"].items():
        assert hashlib.sha256((first/name).read_bytes()).hexdigest() == digest
    for name in required-{"study_metadata.json", "study_report.md", "daily_structure_features.parquet"}:
        pd.testing.assert_frame_equal(pd.read_csv(first/name), pd.read_csv(second/name))
    pd.testing.assert_frame_equal(pd.read_parquet(first/"daily_structure_features.parquet"), pd.read_parquet(second/"daily_structure_features.parquet"))


def test_changed_input_hash_fails_without_redownload_or_repair(tmp_path, snapshot):
    path = snapshot/"price_data"/"price_cache.parquet"
    path.write_bytes(path.read_bytes()+b"changed")
    before = _hashes(snapshot)
    with pytest.raises((ValueError, study.StudyRunError)):
        study.run_study(snapshot, tmp_path/"analysis", bootstrap_samples=0)
    assert _hashes(snapshot) == before


def test_output_inside_source_snapshot_is_refused(snapshot):
    before = _hashes(snapshot)
    with pytest.raises((ValueError, study.StudyRunError)):
        study.run_study(snapshot, snapshot/"analysis", bootstrap_samples=0)
    assert _hashes(snapshot) == before


def test_known_benchmark_correction_guard_is_reused_without_mutating_source():
    index = pd.bdate_range("2026-03-16", "2026-04-14")
    frame = pd.DataFrame({"1306.T": 1000.+np.arange(len(index)), "X": 55.}, index=index)
    bad = pd.to_datetime(["2026-03-30", "2026-03-31"])
    frame.loc[bad, "1306.T"] /= 10
    before = frame.copy(deep=True)
    corrected, log = v01.apply_benchmark_correction(frame)
    pd.testing.assert_frame_equal(frame, before, check_exact=True)
    assert np.allclose(corrected.loc[bad, "1306.T"], before.loc[bad, "1306.T"]*10)
    assert len(log) == 2
    with pytest.raises(ValueError):
        v01.apply_benchmark_correction(corrected)


def test_already_corrected_snapshot_stops_v02_and_retains_failed_status(tmp_path, snapshot):
    data = snapshot/"price_data"
    cache_meta = _json(data/"price_cache_metadata.json")
    for name in ("price_cache.parquet", "benchmark_prices.parquet"):
        path = data/name
        frame = pd.read_parquet(path)
        # Move a synthetic already-continuous dataset across the declared
        # anomaly dates. Its normal prices must not receive another 10x pass.
        frame.index = pd.bdate_range("2025-01-02", periods=len(frame), name=frame.index.name)
        frame.to_parquet(path)
        cache_meta["sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (data/"price_cache_metadata.json").write_text(json.dumps(cache_meta), encoding="utf-8")
    meta = _json(snapshot/"research_run_metadata.json")
    meta["observed_start"] = frame.index[0].date().isoformat()
    meta["observed_end"] = frame.index[-1].date().isoformat()
    (snapshot/"research_run_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    before = _hashes(snapshot)
    with pytest.raises(study.StudyRunError) as caught:
        study.run_study(snapshot, tmp_path/"analysis", bootstrap_samples=0)
    failed = _json(caught.value.run_dir/"study_metadata.json")
    assert failed["status"] == "failed"
    assert failed["validation_status"] == "failed"
    assert _hashes(snapshot) == before
