"""Dow Momentum Study v0.1: offline, causal, retrospective event study.

Reads immutable research snapshots only. Never downloads market data, invokes
screening/selection, or changes existing engine logic or saved prices.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_engine_research_runner import load_research_prices

BASE_COMMIT = "57d99dbc75fe69fd59dd031fc4df78ce3dff18a2"
SNAPSHOT_NAME = "20261104T124443Z_residual100_n6_9b9d99b92dcf"
DEFAULT_SNAPSHOT = f"/content/drive/MyDrive/alpha_engine/research_runs/{SNAPSHOT_NAME}"
DEFAULT_OUTPUT = "/content/drive/MyDrive/alpha_engine/analysis/dow_momentum_v01"
BENCHMARKS = {"US": "SPY", "JP": "1306.T"}
SUSPECT_DATES = pd.DatetimeIndex(["2026-03-30", "2026-03-31"])
HORIZONS = (5, 21, 63)


class StudyRunError(RuntimeError):
    def __init__(self, message, run_dir):
        super().__init__(message)
        self.run_dir = run_dir


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1048576), b""):
            digest.update(part)
    return digest.hexdigest()


def _json(path, value):
    # Convert missing numeric values to JSON null, never nonstandard NaN tokens.
    text = json.dumps(value, default=str, ensure_ascii=False)
    cleaned = json.loads(text, parse_constant=lambda _: None)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(cleaned, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def detect_swings(series: pd.Series, span: int = 5) -> pd.DataFrame:
    """Strict extrema, no pivots on ties; confirmation after span valid bars.

    Missing values are omitted from the observation clock. Caller supplies the
    pairwise-valid stock/benchmark calendar. Same-kind successive pivots remain
    separate identities; we do not merge them using future prices.
    """
    if not isinstance(span, int) or isinstance(span, bool) or span < 1:
        raise ValueError("span must be a positive integer")
    series = series.dropna()
    columns = ["pivot_date", "confirmed_date", "kind", "value", "pivot_position", "confirmed_position"]
    if len(series) < 2 * span + 1:
        return pd.DataFrame(columns=columns)
    values = series.to_numpy(dtype=float)
    windows = np.lib.stride_tricks.sliding_window_view(values, 2 * span + 1)
    others = np.concatenate([windows[:, :span], windows[:, span+1:]], axis=1)
    rows = []
    for kind, mask in (("low", (windows[:, span, None] < others).all(axis=1)),
                       ("high", (windows[:, span, None] > others).all(axis=1))):
        for offset in np.flatnonzero(mask):
            position = int(offset + span)
            rows.append({"pivot_date": series.index[position], "confirmed_date": series.index[position+span],
                         "kind": kind, "value": float(values[position]), "pivot_position": position,
                         "confirmed_position": position + span})
    return pd.DataFrame(rows, columns=columns).sort_values(["confirmed_date", "kind"]).reset_index(drop=True)


def apply_benchmark_correction(prices: pd.DataFrame):
    """Tentative, explicitly guarded 10x adjustment on a deep research copy."""
    adjusted = prices.copy(deep=True)
    rows = []
    if "1306.T" not in prices:
        raise ValueError("1306.T benchmark is missing")
    present = [date for date in SUSPECT_DATES if date in prices.index and pd.notna(prices.at[date, "1306.T"])]
    if not present:
        return adjusted, pd.DataFrame(columns=["ticker", "date", "original", "adjusted", "multiplier", "anchor", "reason"])
    if len(present) != 2:
        raise ValueError("Incomplete suspect 1306.T date pair; refusing a partial correction")
    clean = prices["1306.T"].drop(index=SUSPECT_DATES, errors="ignore").dropna()
    before = clean.loc[clean.index < SUSPECT_DATES.min()].tail(5)
    after = clean.loc[clean.index > SUSPECT_DATES.max()].head(5)
    if len(before) < 3 or len(after) < 3:
        raise ValueError("Not enough unaffected 1306.T anchors to validate the correction")
    anchor = float(pd.concat([before, after]).median())
    for date in present:
        original = float(prices.at[date, "1306.T"])
        ratio = original / anchor
        if not (.05 <= ratio <= .2 and .5 <= ratio * 10 <= 2):
            raise ValueError("1306.T outside declared raw anomaly range; possibly already corrected. No correction applied.")
        adjusted.at[date, "1306.T"] = original * 10
        rows.append({"ticker": "1306.T", "date": date, "original": original,
                     "adjusted": original * 10, "multiplier": 10, "anchor": anchor,
                     "reason": "tentative continuity correction; official adjusted-price verification incomplete"})
    return adjusted, pd.DataFrame(rows)


def compute_features(prices, us, jp, benchmarks=None, span=5, threshold=0.0):
    """Features/pivots/raw below-low states, all based on information at t."""
    benchmarks = benchmarks or BENCHMARKS
    if not (-1 < threshold <= 0):
        raise ValueError("threshold must be in (-1, 0]")
    union = prices.ffill()  # Only engine-reproduction features use union ffill.
    daily_frames, pivot_frames = [], []
    for market, universe in (("US", us), ("JP", jp)):
        benchmark = benchmarks[market]
        for ticker in universe:
            if ticker not in prices:
                continue
            common = prices[[ticker, benchmark]].dropna()
            if common.empty:
                continue
            stock = common[ticker]
            base = pd.DataFrame(index=common.index)
            base["date"] = common.index
            base["ticker"], base["market"], base["benchmark"] = ticker, market, benchmark
            base["common_position"] = np.arange(len(common))
            base["observations"] = np.arange(1, len(common)+1)
            engine_series = union[ticker].dropna()
            for name, window in (("3m", 63), ("6m", 126), ("12m", 252)):
                base[f"momentum_{name}"] = stock / stock.shift(window-1) - 1
                # The existing _window_return guard requires len > window.
                engine_momentum = (engine_series / engine_series.shift(window-1)-1).where(np.arange(len(engine_series)) >= window)
                base[f"engine_momentum_{name}"] = engine_momentum.reindex(common.index)
            base["volatility"] = stock.pct_change(fill_method=None).rolling(252, min_periods=252).std(ddof=1) * np.sqrt(252)
            base["engine_volatility"] = (engine_series.pct_change(fill_method=None).rolling(252, min_periods=2).std(ddof=1) * np.sqrt(252)).reindex(common.index)
            base["stock_price"] = stock
            base["benchmark_price"] = common[benchmark]
            # Conservative contamination marker: a known suspect observation
            # in the 253-price input window (252 returns) or pivot support.
            bad = pd.Series(common.index.isin(SUSPECT_DATES) if market == "JP" else False, index=common.index)
            trailing_bad = bad.rolling(253, min_periods=1).max().astype(bool)
            for kind, series in (("price", stock), ("relative", stock / common[benchmark])):
                pivots = detect_swings(series, span)
                pivots["ticker"], pivots["market"], pivots["series_kind"] = ticker, market, kind
                pivot_frames.append(pivots)
                frame = base.copy()
                frame["series_kind"], frame["series_value"] = kind, series
                for pivot_kind, label in (("low", "low"), ("high", "high")):
                    subset = pivots[pivots.kind == pivot_kind]
                    mapping = subset.set_index("confirmed_date")
                    for source, target in (("value", f"reference_{label}"), ("pivot_date", f"reference_{label}_pivot_date"),
                                           ("confirmed_position", f"reference_{label}_confirmed_position"),
                                           ("pivot_position", f"reference_{label}_pivot_position")):
                        frame[target] = mapping[source].reindex(common.index).ffill() if len(mapping) else (pd.NaT if source == "pivot_date" else np.nan)
                    frame[f"reference_{label}_confirmed_date"] = pd.Series(subset.confirmed_date.to_numpy(), index=subset.confirmed_date).reindex(common.index).ffill() if len(subset) else pd.NaT
                    # Compare newest two CONFIRMED pivots only; no retrospective
                    # trend labelling using later swings.
                    movement = subset.value.diff()
                    frame[f"{label}_structure"] = pd.Series(np.sign(movement).to_numpy(), index=subset.confirmed_date).reindex(common.index).ffill() if len(subset) else np.nan
                frame["reference_pivot_date"] = frame.reference_low_pivot_date
                frame["reference_confirmed_date"] = frame.reference_low_confirmed_date
                frame["below_low"] = frame.series_value < frame.reference_low * (1 + threshold)
                frame["break_distance"] = frame.series_value / frame.reference_low - 1
                support_bad = np.zeros(len(frame), dtype=bool)
                if market == "JP":
                    positions = frame.reference_low_pivot_position.to_numpy(dtype=float)
                    known = np.isfinite(positions)
                    centers = positions[known].astype(int)
                    prefix = np.r_[0,np.cumsum(bad.to_numpy(dtype=int))]
                    left, right = np.maximum(0,centers-span), np.minimum(len(frame),centers+span+1)
                    support_bad[known] = prefix[right] > prefix[left]
                frame["excluded_influence"] = trailing_bad.to_numpy() | support_bad
                daily_frames.append(frame.reset_index(drop=True))
    if not daily_frames:
        raise ValueError("No common valid stock/benchmark observations")
    daily = pd.concat(daily_frames, ignore_index=True).sort_values(["ticker", "series_kind", "date"]).reset_index(drop=True)
    raw = daily.loc[daily.below_low].copy()
    raw["signal_date"] = raw.date
    raw["event_id"] = [f"{r.ticker}|{r.series_kind}|{pd.Timestamp(r.date).date()}|{pd.Timestamp(r.reference_pivot_date).date()}" for r in raw.itertuples()]
    pivots = pd.concat(pivot_frames, ignore_index=True)
    return daily, pivots, raw.reset_index(drop=True)


def independent_events(raw, cooldown=21):
    """First breach per low, then cooldown from last accepted breach.

    Duplicate/cooldown selection occurs before eligibility filtering. A consumed
    pivot cannot reappear when momentum history later becomes sufficient.
    """
    if cooldown < 0:
        raise ValueError("cooldown must be nonnegative")
    work = raw.sort_values(["ticker", "series_kind", "signal_date", "reference_pivot_date"]).copy()
    work["first_per_reference"] = ~work.duplicated(["ticker", "series_kind", "reference_pivot_date"])
    accepted, reasons, last = [], [], {}
    for row in work.itertuples():
        key = (row.ticker, row.series_kind)
        if not row.first_per_reference:
            use, reason = False, "repeated_reference_low"
        elif key in last and row.common_position - last[key] < cooldown:
            use, reason = False, "cooldown"
        else:
            use, reason = True, "accepted"
            last[key] = row.common_position
        accepted.append(use)
        reasons.append(reason)
    work["independent"] = accepted
    work["deduplication_reason"] = reasons
    return work.reset_index(drop=True)


def _reproduction(original, us, jp, scores):
    """Final-date audit only; snapshot rankings never become past features."""
    filled = original.ffill()
    rows = []
    for row in scores.itertuples():
        if row.ticker not in filled:
            rows.append({"ticker": row.ticker, "status": "missing_ticker"})
            continue
        series = filled[row.ticker].dropna()
        for name, width in (("3m", 63), ("6m", 126), ("12m", 252)):
            actual = float(series.iloc[-1]/series.iloc[-width]-1) if len(series) > width else np.nan
            expected = getattr(row, "momentum_"+name, np.nan)
            match = bool(np.isclose(actual, expected, atol=1e-10, rtol=1e-10, equal_nan=True))
            rows.append({"ticker": row.ticker, "window": name, "engine_value": actual, "snapshot_value": expected,
                         "difference": actual - expected, "status": "matched" if match else "mismatch",
                         "definition": "full union calendar, past-only ffill, endpoint window-1; require len>window"})
    return pd.DataFrame(rows)


def _input_snapshot(snapshot):
    required = ["price_data/price_cache.parquet", "price_data/benchmark_prices.parquet",
                "price_data/price_cache_metadata.json", "research_run_metadata.json", "engine_outputs/score_components.csv"]
    for relative in required:
        if not (snapshot / relative).is_file():
            raise ValueError(f"Required snapshot input missing: {relative}")
    if (snapshot / "price_data/price_data_quality.csv").exists():
        required.append("price_data/price_data_quality.csv")
    hashes = {name: _hash(snapshot/name) for name in required}
    prices = load_research_prices(snapshot)
    meta = json.loads((snapshot/"research_run_metadata.json").read_text(encoding="utf-8"))
    if meta.get("status") != "completed" or meta.get("validation_status") != "passed":
        raise ValueError("Input research snapshot is not completed and validated")
    us, jp = list(meta["us_universe"]), list(meta["jp_universe"])
    if len(set(us+jp)) != len(us)+len(jp):
        raise ValueError("Duplicate or overlapping snapshot universe tickers")
    if prices.index.has_duplicates or prices.columns.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("Duplicate or unordered snapshot observations")
    if prices.index.tz is not None or prices.index.hasnans:
        raise ValueError("Expected timezone-naive daily snapshot dates")
    for field, observed in (("observed_start", prices.index.min()),("observed_end",prices.index.max())):
        if field in meta and str(pd.Timestamp(meta[field]).date()) != str(observed.date()):
            raise ValueError(f"Snapshot metadata {field} does not match actual observations")
    for ticker in prices:
        if not pd.api.types.is_numeric_dtype(prices[ticker]):
            raise ValueError(f"Non-numeric snapshot column: {ticker}")
        values = prices[ticker].dropna().to_numpy(dtype=float)
        if ((values <= 0) | ~np.isfinite(values)).any():
            raise ValueError(f"Invalid snapshot price: {ticker}")
    for benchmark in BENCHMARKS.values():
        if benchmark not in prices or prices[benchmark].notna().sum() < 253:
            raise ValueError(f"Missing/short benchmark: {benchmark}")
    scores = pd.read_csv(snapshot/"engine_outputs/score_components.csv")
    return prices, us, jp, scores, hashes


_MATCH_COLUMNS = [
    "event_id", "signal_date", "ticker", "market", "series_kind", "status",
    "momentum_3m", "volatility", "control_ticker", "control_momentum_3m",
    "control_volatility", "distance", "momentum_scale", "volatility_scale",
    "eligible_control_count", "within_caliper_count", "caliper",
]


def match_controls(events, daily, caliper=0.5):
    """Match one same-date, same-market, same-series non-alerting stock.

    Population standard deviations from that date's diagnosed, history-eligible
    cross-section scale momentum and volatility. Each scaled absolute difference
    must be <= caliper; Euclidean distance then ticker name breaks ties. Controls
    may be reused by different events. No forward prices enter this function.
    """
    if not np.isfinite(caliper) or caliper < 0:
        raise ValueError("Matching caliper must be finite and nonnegative.")
    if events.empty:
        return pd.DataFrame(columns=_MATCH_COLUMNS)
    needed = ["date", "market", "series_kind", "ticker", "momentum_3m",
              "volatility", "observations", "reference_low", "below_low"]
    missing = sorted(set(needed) - set(daily.columns))
    if missing:
        raise ValueError(f"Missing matching features: {missing}")
    frame = daily[needed].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    # Build only contemporaneous pools that actually have an event, while
    # retaining every eligible name in each selected cross-section.
    keys = events[["signal_date","market","series_kind"]].drop_duplicates().rename(columns={"signal_date":"date"})
    keys["date"] = pd.to_datetime(keys.date)
    frame = frame.merge(keys,on=["date","market","series_kind"],how="inner",validate="many_to_one")
    pools = {}
    for key, group in frame.groupby(["date", "market", "series_kind"], sort=False):
        finite = np.isfinite(group[["momentum_3m", "volatility"]].to_numpy(dtype=float)).all(axis=1)
        eligible = group.loc[finite & (group.observations >= 252) &
                             (group.volatility > 0) & group.reference_low.notna()].copy()
        if eligible.empty:
            pools[key] = (eligible, np.array([1e-12, 1e-12]))
            continue
        scales = eligible[["momentum_3m", "volatility"]].std(ddof=0).to_numpy(dtype=float)
        scales = np.maximum(scales, 1e-12)
        pools[key] = (eligible, scales)
    ordered = events.copy()
    ordered["signal_date"] = pd.to_datetime(ordered["signal_date"])
    ordered = ordered.sort_values(["signal_date", "market", "series_kind", "ticker", "event_id"], kind="stable")
    rows = []
    for event in ordered.to_dict("records"):
        momentum = event.get("momentum_3m", np.nan)
        volatility = event.get("volatility", np.nan)
        row = {"event_id": event["event_id"], "signal_date": event["signal_date"],
               "ticker": event["ticker"], "market": event["market"],
               "series_kind": event["series_kind"], "momentum_3m": momentum,
               "volatility": volatility, "status": "no_control", "control_ticker": None,
               "control_momentum_3m": np.nan, "control_volatility": np.nan,
               "distance": np.nan, "momentum_scale": np.nan, "volatility_scale": np.nan,
               "eligible_control_count": 0, "within_caliper_count": 0, "caliper": float(caliper)}
        if (event.get("observations", 0) < 252 or not np.isfinite(momentum) or
                not np.isfinite(volatility) or volatility <= 0):
            row["status"] = "insufficient_history"
            rows.append(row)
            continue
        pool, scales = pools.get((event["signal_date"], event["market"], event["series_kind"]),
                                 (pd.DataFrame(), np.array([np.nan, np.nan])))
        row["momentum_scale"], row["volatility_scale"] = map(float, scales)
        if pool.empty:
            rows.append(row)
            continue
        controls = pool.loc[(pool.ticker != event["ticker"]) & (pool.below_low == False)].copy()
        row["eligible_control_count"] = len(controls)
        if not controls.empty:
            differences = np.abs(controls[["momentum_3m", "volatility"]].to_numpy(dtype=float) -
                                 np.array([momentum, volatility])) / scales
            keep = (differences <= caliper + 1e-14).all(axis=1)
            controls = controls.loc[keep].copy()
            row["within_caliper_count"] = len(controls)
            if not controls.empty:
                controls["_distance"] = np.sqrt(np.square(differences[keep]).sum(axis=1))
                best = controls.sort_values(["_distance", "ticker"], kind="stable").iloc[0]
                row.update({"status": "matched", "control_ticker": best.ticker,
                            "control_momentum_3m": float(best.momentum_3m),
                            "control_volatility": float(best.volatility),
                            "distance": float(best._distance)})
        rows.append(row)
    return pd.DataFrame(rows, columns=_MATCH_COLUMNS)


_OUTCOME_COLUMNS = [
    "event_id", "group", "ticker", "event_ticker", "market", "series_kind",
    "signal_date", "horizon", "primary_horizon", "momentum_3m", "event_momentum_3m",
    "status", "entry_date", "exit_date", "last_available_date",
    "available_future_observations", "stock_return", "benchmark_return",
    "relative_return", "return_difference", "max_drawdown", "min_return_from_entry",
]


def evaluate_outcomes(events, matches, prices, us, jp, horizons=(5, 21, 63)):
    """Measure entry at t+1 close, exit H common valid intervals after entry.

    Horizon censoring is independent: an event may have observed H=5 and missing
    H=21/63. Partial forward paths never become zero returns or full outcomes.
    """
    horizons = tuple(horizons)
    if any(isinstance(h, bool) or int(h) != h or h <= 0 for h in horizons):
        raise ValueError("Outcome horizons must be positive integers.")
    if events.empty:
        return pd.DataFrame(columns=_OUTCOME_COLUMNS)
    benchmark_by_market = {"US": "SPY", "JP": "1306.T"}
    membership = {**{ticker: "US" for ticker in us}, **{ticker: "JP" for ticker in jp}}
    if len(membership) != len(set(us)) + len(set(jp)):
        raise ValueError("US and JP memberships must not overlap.")
    match_lookup = matches.set_index("event_id").to_dict("index") if not matches.empty else {}
    paired_cache = {}

    def paired(ticker, market):
        key = ticker, market
        if key not in paired_cache:
            benchmark = benchmark_by_market[market]
            if ticker not in prices or benchmark not in prices:
                paired_cache[key] = pd.DataFrame(columns=[ticker, benchmark])
            else:
                pair = prices[[ticker, benchmark]].dropna()
                values = pair.to_numpy(dtype=float)
                pair = pair.loc[np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)]
                if pair.index.has_duplicates or not pair.index.is_monotonic_increasing:
                    raise ValueError("Outcomes require unique ordered observation dates.")
                paired_cache[key] = pair
        return paired_cache[key]

    ordered = events.copy()
    ordered["signal_date"] = pd.to_datetime(ordered["signal_date"])
    ordered = ordered.sort_values(["signal_date", "market", "series_kind", "ticker", "event_id"], kind="stable")
    rows = []
    for event in ordered.to_dict("records"):
        match = match_lookup.get(event["event_id"], {})
        control = match.get("control_ticker") if match.get("status") == "matched" else None
        for group, ticker in (("alert", event["ticker"]), ("control", control)):
            if ticker is not None and membership.get(ticker) != event["market"]:
                raise ValueError("Outcome ticker belongs to a different market or unknown universe.")
            frame = paired(ticker, event["market"]) if ticker is not None else pd.DataFrame()
            position = frame.index.get_indexer([event["signal_date"]])[0] if not frame.empty else -1
            entry = position + 1
            for horizon in horizons:
                row = {"event_id": event["event_id"], "group": group, "ticker": ticker,
                       "event_ticker": event["ticker"], "market": event["market"],
                       "series_kind": event["series_kind"], "signal_date": event["signal_date"],
                       "horizon": int(horizon), "primary_horizon": horizon == 21,
                       "momentum_3m": event.get("momentum_3m", np.nan),
                       "event_momentum_3m": event.get("momentum_3m", np.nan),
                       "status": "right_censored", "entry_date": pd.NaT, "exit_date": pd.NaT,
                       "last_available_date": frame.index[-1] if not frame.empty else pd.NaT,
                       "available_future_observations": max(0, len(frame) - entry - 1) if position >= 0 else 0,
                       "stock_return": np.nan, "benchmark_return": np.nan, "relative_return": np.nan,
                       "return_difference": np.nan, "max_drawdown": np.nan, "min_return_from_entry": np.nan}
                if ticker is None:
                    row["status"] = "unmatched"
                elif position < 0:
                    row["status"] = "missing_signal_observation"
                elif entry < len(frame):
                    row["entry_date"] = frame.index[entry]
                    exit_position = entry + int(horizon)
                    if exit_position < len(frame):
                        stock = frame.iloc[:, 0]
                        benchmark = frame.iloc[:, 1]
                        stock_return = float(stock.iloc[exit_position] / stock.iloc[entry] - 1)
                        benchmark_return = float(benchmark.iloc[exit_position] / benchmark.iloc[entry] - 1)
                        path = stock.iloc[entry:exit_position + 1].to_numpy(dtype=float)
                        row.update({"status": "observed", "exit_date": frame.index[exit_position],
                                    "stock_return": stock_return, "benchmark_return": benchmark_return,
                                    "relative_return": (1 + stock_return) / (1 + benchmark_return) - 1,
                                    "return_difference": stock_return - benchmark_return,
                                    "max_drawdown": float(np.min(path / np.maximum.accumulate(path) - 1)),
                                    "min_return_from_entry": float(np.min(path / path[0] - 1))})
                rows.append(row)
    return pd.DataFrame(rows, columns=_OUTCOME_COLUMNS)


_SUMMARY_COLUMNS = [
    "subgroup", "market", "period", "series_kind", "horizon", "primary_horizon",
    "group", "total_count", "valid_count", "right_censored_count", "unmatched_count",
    "excluded_influence_count", "missing_signal_observation_count",
    "mean_stock_return", "median_stock_return", "loss_probability", "mean_benchmark_return",
    "mean_relative_return", "median_relative_return", "underperformance_probability",
    "mean_max_drawdown", "median_max_drawdown",
]


def _cluster_mean_interval(values, clusters, samples, rng):
    """Resample complete clusters and recompute the event-weighted mean."""
    frame = pd.DataFrame({"value": values, "cluster": clusters}).dropna()
    grouped = frame.groupby("cluster", sort=True).value.agg(["sum", "count"])
    n = len(grouped)
    if n < 2 or samples < 2:
        return np.nan, np.nan, n
    sums, counts = grouped["sum"].to_numpy(), grouped["count"].to_numpy()
    draws = rng.integers(0, n, size=(samples, n))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    lower, upper = np.quantile(means, [.025, .975])
    return float(lower), float(upper), n


def summarize_outcomes(outcomes, bootstrap_samples=500, seed=20261104):
    """Describe all and positive-momentum events, including paired A-minus-B.

    Intervals are exploratory ticker/date cluster sensitivity estimates, not an
    independence-based significance test. Their envelope is not a calibrated
    two-way-cluster confidence interval. No p-values are produced.
    """
    uncertainty_columns = ["subgroup", "market", "period", "series_kind", "horizon",
                           "primary_horizon", "paired_count", "mean_relative_difference",
                           "median_relative_difference", "ticker_cluster_count", "date_cluster_count",
                           "ticker_ci_low", "ticker_ci_high", "date_ci_low", "date_ci_high",
                           "envelope_low", "envelope_high", "bootstrap_samples", "seed", "warning"]
    if int(bootstrap_samples) != bootstrap_samples or bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be a nonnegative integer.")
    if outcomes.empty:
        return pd.DataFrame(columns=_SUMMARY_COLUMNS), pd.DataFrame(columns=uncertainty_columns)
    frame = outcomes.copy()
    frame["signal_date"] = pd.to_datetime(frame.signal_date)
    frame["_year"] = frame.signal_date.dt.year.astype(str)
    years = sorted(frame._year.unique())
    rng = np.random.default_rng(seed)
    summary_rows, uncertainty_rows = [], []

    def measure(group, base, group_name):
        observed = group.loc[group.status == "observed"]
        row = {**base, "group": group_name, "total_count": len(group), "valid_count": len(observed),
               "right_censored_count": int((group.status == "right_censored").sum()),
               "unmatched_count": int((group.status == "unmatched").sum()),
               "excluded_influence_count": int((group.status == "excluded_influence").sum()),
               "missing_signal_observation_count": int((group.status == "missing_signal_observation").sum()),
               "mean_stock_return": observed.stock_return.mean(), "median_stock_return": observed.stock_return.median(),
               "loss_probability": float((observed.stock_return < 0).mean()) if len(observed) else np.nan,
               "mean_benchmark_return": observed.benchmark_return.mean(),
               "mean_relative_return": observed.relative_return.mean(), "median_relative_return": observed.relative_return.median(),
               "underperformance_probability": float((observed.relative_return < 0).mean()) if len(observed) else np.nan,
               "mean_max_drawdown": observed.max_drawdown.mean(), "median_max_drawdown": observed.max_drawdown.median()}
        return row

    for subgroup in ("all", "positive_momentum_3m"):
        sub = frame if subgroup == "all" else frame.loc[frame.event_momentum_3m > 0]
        for market in ("ALL", "US", "JP"):
            market_frame = sub if market == "ALL" else sub.loc[sub.market == market]
            for period in ("ALL", *years):
                period_frame = market_frame if period == "ALL" else market_frame.loc[market_frame._year == period]
                for (series_kind, horizon), sample in period_frame.groupby(["series_kind", "horizon"], sort=True):
                    base = {"subgroup": subgroup, "market": market, "period": period,
                            "series_kind": series_kind, "horizon": int(horizon), "primary_horizon": horizon == 21}
                    alert = sample.loc[sample.group == "alert"]
                    control = sample.loc[sample.group == "control"]
                    summary_rows.extend([measure(alert, base, "alert"), measure(control, base, "control")])
                    paired = alert.loc[alert.status == "observed"].merge(
                        control.loc[control.status == "observed"], on="event_id", suffixes=("_alert", "_control"), validate="one_to_one")
                    differences = pd.DataFrame({"status": ["observed"] * len(paired)})
                    for col in ("stock_return", "benchmark_return", "relative_return", "max_drawdown"):
                        differences[col] = paired[f"{col}_alert"].to_numpy() - paired[f"{col}_control"].to_numpy()
                    paired_row = measure(differences, base, "paired_difference")
                    # Count the originally eligible alert cohort in the pairing
                    # denominator; incomplete/unmatched pairs remain explicit.
                    paired_row["total_count"] = len(alert)
                    all_pairs = alert[["event_id","status"]].merge(control[["event_id","status"]],on="event_id",suffixes=("_alert","_control"),validate="one_to_one")
                    pair_status = pd.Series("observed",index=all_pairs.index)
                    # Explicit failure priority yields disjoint denominators.
                    for state in ("right_censored","missing_signal_observation","excluded_influence","unmatched"):
                        mask = (all_pairs.status_alert == state) | (all_pairs.status_control == state)
                        pair_status.loc[mask] = state
                    for state in ("right_censored","missing_signal_observation","excluded_influence","unmatched"):
                        paired_row[state+"_count"] = int((pair_status==state).sum())
                    summary_rows.append(paired_row)
                    relative_difference = differences.relative_return.to_numpy(dtype=float)
                    t_low, t_high, nt = _cluster_mean_interval(relative_difference, paired.get("event_ticker_alert", pd.Series(dtype=str)), int(bootstrap_samples), rng)
                    d_low, d_high, nd = _cluster_mean_interval(relative_difference, paired.get("signal_date_alert", pd.Series(dtype="datetime64[ns]")), int(bootstrap_samples), rng)
                    lower = [x for x in (t_low, d_low) if np.isfinite(x)]
                    upper = [x for x in (t_high, d_high) if np.isfinite(x)]
                    warning = "Exploratory separate-cluster bootstrap; envelope is not a calibrated two-way interval."
                    if min(nt, nd) < 10:
                        warning += " Fewer than 10 ticker/date clusters; uncertainty is weakly identified."
                    uncertainty_rows.append({**base, "paired_count": len(paired),
                                             "mean_relative_difference": float(np.mean(relative_difference)) if len(paired) else np.nan,
                                             "median_relative_difference": float(np.median(relative_difference)) if len(paired) else np.nan,
                                             "ticker_cluster_count": nt, "date_cluster_count": nd,
                                             "ticker_ci_low": t_low, "ticker_ci_high": t_high,
                                             "date_ci_low": d_low, "date_ci_high": d_high,
                                             "envelope_low": min(lower) if lower else np.nan,
                                             "envelope_high": max(upper) if upper else np.nan,
                                             "bootstrap_samples": int(bootstrap_samples), "seed": seed, "warning": warning})
    return (pd.DataFrame(summary_rows, columns=_SUMMARY_COLUMNS),
            pd.DataFrame(uncertainty_rows, columns=uncertainty_columns))


def _case(prices, us, jp, *, span=5, threshold=0.0, cooldown=21, exclude_influence=False):
    daily, pivots, raw = compute_features(prices, us, jp, span=span, threshold=threshold)
    dedup = independent_events(raw, cooldown)
    accepted = dedup.loc[dedup.independent].copy()
    valid = (accepted.observations >= 252) & np.isfinite(accepted[["momentum_3m", "momentum_6m", "momentum_12m", "volatility"]]).all(axis=1) & (accepted.volatility > 0)
    accepted["analysis_eligible"] = valid
    accepted["analysis_exclusion_reason"] = np.where(valid, "", "insufficient_feature_history_or_zero_volatility")
    if exclude_influence:
        affected = accepted.excluded_influence
        accepted.loc[affected, "analysis_eligible"] = False
        accepted.loc[affected, "analysis_exclusion_reason"] = "known_benchmark_anomaly_in_input_or_pivot_support"
        pool = daily.loc[~daily.excluded_influence]
    else:
        pool = daily
    events = accepted.loc[accepted.analysis_eligible].copy()
    matches = match_controls(events, pool)
    outcomes = evaluate_outcomes(events, matches, prices, us, jp)
    if exclude_influence:
        for index, row in outcomes.loc[(outcomes.market == "JP") & (outcomes.status == "observed")].iterrows():
            pair = prices[[row.ticker, "1306.T"]].dropna()
            touched = pair.loc[row.entry_date:row.exit_date].index.isin(SUSPECT_DATES).any()
            if touched:
                outcomes.at[index, "status"] = "excluded_influence"
                outcomes.loc[index, ["stock_return", "benchmark_return", "relative_return", "return_difference", "max_drawdown", "min_return_from_entry"]] = np.nan
    return daily, pivots, dedup, accepted, matches, outcomes


def _charts(folder, daily, alerts, outcomes, baseline):
    """Standalone figures; individual examples have deterministic criteria."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.hashsalt": "dow-momentum-v01", "font.size": 9})
    folder.mkdir()

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(folder/f"{name}.svg", metadata={"Date": None})
        plt.close(fig)

    for ticker in ("SNDK", "DELL", "4062.T"):
        frame = daily.loc[(daily.ticker == ticker) & (daily.series_kind == "price")]
        fig, ax = plt.subplots(figsize=(10, 3))
        if frame.empty:
            ax.text(.5, .5, f"{ticker}: absent from input", ha="center", transform=ax.transAxes)
        else:
            ax.plot(frame.date, frame.stock_price, label="Adjusted Close")
            points = alerts.loc[(alerts.ticker == ticker) & (alerts.series_kind == "price") & alerts.independent]
            ax.scatter(points.signal_date, points.stock_price, marker="v", color="red", label="Independent alert")
            ax.legend()
        ax.set_title(f"{ticker}: post-selected exploratory example, not validation")
        save(fig, f"exploratory_{ticker}")
    fig, ax = plt.subplots(figsize=(10, 3))
    selected = alerts.loc[alerts.independent].copy()
    if not selected.empty:
        counts = selected.groupby([pd.to_datetime(selected.signal_date).dt.to_period("M"), "series_kind"]).size().unstack(fill_value=0)
        counts.index = counts.index.to_timestamp()
        counts.plot(ax=ax)
    ax.set_title("Independent alerts by month; price/relative duplicates retained")
    ax.set_ylabel("Alerts")
    save(fig, "alert_frequency")
    fig, ax = plt.subplots(figsize=(9, 3))
    all_values = outcomes.loc[(outcomes.horizon==21)&(outcomes.status=="observed"),"relative_return"].dropna()
    bins = np.histogram_bin_edges(all_values,bins=30) if len(all_values) else 30
    for group, color in (("alert", "red"), ("control", "steelblue")):
        values = outcomes.loc[(outcomes.horizon == 21) & (outcomes.status == "observed") & (outcomes.group == group), "relative_return"]
        if len(values):
            ax.hist(values, bins=bins, alpha=.45, label=f"{group} n={len(values)}", color=color)
    ax.set_title("21-observation relative returns: descriptive, unmatched cohorts may differ")
    ax.set_xlabel("Geometric benchmark-relative return")
    if ax.has_data():
        ax.legend()
    save(fig, "relative_return_distribution")
    fig, ax = plt.subplots(figsize=(9, 3))
    if not baseline.empty:
        view = baseline.loc[baseline.horizon == 21].groupby("momentum_bucket").relative_return.mean()
        view.plot.bar(ax=ax)
    ax.set_title("Momentum-only fixed 21-bar landmarks: 21-bar future relative mean")
    save(fig, "momentum_only")
    fig, ax = plt.subplots(figsize=(10,3))
    observed = outcomes.loc[(outcomes.horizon==21)&(outcomes.status=="observed")]
    if len(observed):
        paired = observed.pivot(index="event_id",columns="group",values="relative_return").reindex(columns=["alert","control"]).dropna()
        momenta = observed.loc[observed.group=="alert"].set_index("event_id").event_momentum_3m
        paired["momentum_bucket"] = pd.cut(momenta.reindex(paired.index),[-np.inf,0,.1,.3,np.inf],labels=["nonpositive","0_to_10pct","10_to_30pct","above_30pct"])
        paired["paired_difference"] = paired.alert-paired.control
        if len(paired):
            paired.groupby("momentum_bucket",observed=False)[["alert","control","paired_difference"]].mean().plot.bar(ax=ax)
        else:
            ax.text(.5,.5,"No observed matched 21-bar pairs",ha="center",transform=ax.transAxes)
    ax.set_title("21-bar relative returns by momentum AT ALERT (observed matched pairs)")
    save(fig,"alert_momentum_outcomes")
    # Worst/best primary observed price-alert outcomes are explicitly post-hoc
    # illustrative examples. They never enter event or control selection.
    examples = outcomes.loc[(outcomes.group == "alert") & (outcomes.horizon == 21) & (outcomes.status == "observed") & (outcomes.series_kind == "price")].sort_values(["relative_return", "event_id"])
    chosen = [] if examples.empty else [("lowest_relative_return", examples.iloc[0]), ("highest_relative_return", examples.iloc[-1])]
    counterexamples = examples.loc[examples.stock_return > 0]
    if len(counterexamples):
        chosen.append(("rising_after_alert_counterexample",counterexamples.sort_values(["stock_return","event_id"]).iloc[-1]))
    for label, row in chosen:
        view = daily.loc[(daily.ticker == row.ticker) & (daily.series_kind == "price")]
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(view.date, view.stock_price)
        ax.axvline(row.signal_date, color="red", linestyle="--")
        ax.set_title(f"Post-hoc {label}: {row.ticker}, {pd.Timestamp(row.signal_date).date()}, relative={row.relative_return:.3f}")
        save(fig, label)


def run_study(snapshot_dir=DEFAULT_SNAPSHOT, output_root=DEFAULT_OUTPUT, *, bootstrap_samples=500, seed=20261104):
    """Analyze a hash-verified snapshot without modifying it or downloading."""
    if isinstance(bootstrap_samples, bool) or int(bootstrap_samples) != bootstrap_samples or bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be a nonnegative integer")
    snapshot, root = Path(snapshot_dir).expanduser().resolve(), Path(output_root).expanduser().resolve()
    if root == snapshot or snapshot in root.parents:
        raise ValueError("Analysis output must be outside the source snapshot")
    stamp = datetime.now(timezone.utc)
    out = root/f"{stamp.strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:12]}"
    out.mkdir(parents=True, exist_ok=False)
    warnings = [
        "Exploratory retrospective study: current-universe survivorship and post-selection bias; not an untouched out-of-sample validation.",
        "1306.T correction is tentative; official adjusted-price verification is incomplete.",
        "Case/control clocks use each stock's common valid benchmark observations; future missing bars may produce different calendar exit dates.",
        "Controls may be reused; price/relative alerts overlap. Ticker/date cluster bootstrap envelopes are not calibrated two-way confidence intervals or significance tests.",
        "Main horizon21, span5, threshold0 and cooldown21 fixed before analysis; sensitivity results are not a parameter selection contest.",
    ]
    parameters = {"primary_horizon": 21, "horizons": list(HORIZONS), "span": 5, "threshold": 0,
                  "cooldown": 21, "matching_caliper": .5, "matching_scale": "same-date diagnosed cross-section population SD(ddof=0), floor1e-12",
                  "matching_features": ["common_stock_momentum_3m", "common_stock_252_return_volatility"],
                  "matching_rule": "same date/market/series, reference low present, no active raw alert in same series; nearest L2 then ticker; controls reusable",
                  "entry": "next common valid close", "exit": "entry + H common valid intervals",
                  "bootstrap_samples": int(bootstrap_samples), "seed": int(seed),
                  "correction": {"ticker": "1306.T", "dates": [str(d.date()) for d in SUSPECT_DATES], "multiplier": 10,
                                 "anchor": "median nearest5 clean before and5 after; minimum3 each", "raw_ratio_range": [.05,.2], "corrected_ratio_range": [.5,2]},
                  "influence_exclusion": "uncorrected JP signals/controls: trailing253common prices (252 returns) or low pivot support touches suspect dates; outcomes: entry..exit path touches suspect dates; never select controls using future availability",
                  "momentum_only_baseline": "every21st common observation (position%21==0), eligible253+history; fixed bins<=0,0to10%,10to30%,above30%; descriptive unpaired cohort"}
    meta = {"version": "0.1", "status": "incomplete", "validation_status": "incomplete", "started_at": stamp.isoformat(),
            "finished_at": None, "source_snapshot": str(snapshot), "output_folder": str(out), "parameters": parameters,
            "warnings": warnings, "base_commit": BASE_COMMIT, "code_sha256": _hash(Path(__file__)),
            "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__}}
    _json(out/"study_metadata.json", meta)
    try:
        for package in ("pyarrow","matplotlib"):
            meta["versions"][package] = importlib.metadata.version(package)
        try:
            meta["git_commit_sha"] = subprocess.check_output(["git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            meta["git_commit_sha"] = "unavailable"
        original, us, jp, scores, hashes = _input_snapshot(snapshot)
        meta["input_hashes"] = hashes
        meta["analysis_period"] = {"start": str(original.index.min().date()), "end": str(original.index.max().date())}
        meta["candidate_counts"] = {"US": len(us), "JP": len(jp), "total": len(us)+len(jp), "snapshot_ranked": len(scores)}
        if (len(us), len(jp), len(scores)) != (503,108,607):
            warnings.append("Input counts differ from stated fixed snapshot US503/JP108/ranked607; inspect input identity (synthetic tests intentionally differ).")
        corrected, correction = apply_benchmark_correction(original)
        correction.to_csv(out/"correction_log.csv", index=False)
        reproduction = _reproduction(original, us, jp, scores)
        reproduction.to_csv(out/"momentum_reproduction.csv", index=False)
        meta["momentum_reproduction"] = reproduction.status.value_counts().to_dict()
        if (reproduction.status != "matched").any():
            warnings.append("Snapshot momentum does not fully reproduce; see momentum_reproduction.csv. Check source definition/calendar/NaN or source artifact identity; never silently equate common and union indicators.")
        daily, pivots, raw, accepted, matches, outcomes = _case(corrected, us, jp)
        if len(raw) and (pd.to_datetime(raw.reference_confirmed_date) > pd.to_datetime(raw.signal_date)).any():
            raise AssertionError("Future pivot leaked into an alert")
        summary, uncertainty = summarize_outcomes(outcomes, int(bootstrap_samples), int(seed))
        daily.to_parquet(out/"daily_features.parquet", index=False)
        for name, frame in (("swing_events.csv", pivots), ("alerts_raw.csv", raw), ("alerts_deduplicated.csv", accepted),
                            ("control_matches.csv", matches), ("event_outcomes.csv", outcomes),
                            ("summary_statistics.csv", summary), ("uncertainty.csv", uncertainty)):
            frame.to_csv(out/name, index=False)
        eligible = daily.loc[(daily.series_kind == "price") & (daily.observations >= 253) & (daily.common_position % 21 == 0) & np.isfinite(daily.volatility)].copy()
        eligible["signal_date"] = eligible.date
        eligible["event_id"] = [f"momentum|{r.ticker}|{pd.Timestamp(r.date).date()}" for r in eligible.itertuples()]
        eligible["series_kind"] = "momentum_only"
        baseline = evaluate_outcomes(eligible, pd.DataFrame(), corrected, us, jp)
        baseline = baseline.loc[baseline.group == "alert"].copy()
        baseline["momentum_bucket"] = pd.cut(baseline.momentum_3m, [-np.inf,0,.1,.3,np.inf], labels=["nonpositive","0_to_10pct","10_to_30pct","above_30pct"]).astype(str)
        baseline.to_csv(out/"momentum_only_outcomes.csv", index=False)
        baseline_summary = baseline.groupby(["market","horizon","momentum_bucket","status"], dropna=False).agg(count=("event_id","size"), mean_relative_return=("relative_return","mean"), median_relative_return=("relative_return","median")).reset_index()
        baseline_summary.to_csv(out/"momentum_only_summary.csv", index=False)
        sensitivity_rows = []
        cases = [("primary_corrected",5,0.,21,False), ("uncorrected",5,0.,21,False),
                 ("exclude_influence",5,0.,21,True), ("span3",3,0.,21,False), ("span10",10,0.,21,False),
                 ("threshold_minus1pct",5,-.01,21,False), ("threshold_minus3pct",5,-.03,21,False), ("no_cooldown",5,0.,0,False)]
        meta["parameters"]["sensitivity_specifications"] = [{"name":label,"span":span,"threshold":threshold,"cooldown":cooldown,"exclude_influence":exclude} for label,span,threshold,cooldown,exclude in cases]
        sensitivity_dir = out/"sensitivity"
        sensitivity_dir.mkdir()
        for label, span, threshold, cooldown, exclude in cases:
            if label == "primary_corrected":
                alt_accepted, alt_matches, alt_outcomes = accepted, matches, outcomes
                alt_summary = summary
            else:
                _, _, _, alt_accepted, alt_matches, alt_outcomes = _case(original if label in ("uncorrected","exclude_influence") else corrected, us, jp,
                                                                       span=span, threshold=threshold, cooldown=cooldown, exclude_influence=exclude)
                alt_summary, _ = summarize_outcomes(alt_outcomes, 0, int(seed))
            for filename, frame in (("alerts",alt_accepted),("matches",alt_matches),("outcomes",alt_outcomes)):
                frame.to_csv(sensitivity_dir/f"{label}_{filename}.csv",index=False)
            part = alt_summary.copy()
            part["specification"], part["span"], part["threshold"], part["cooldown"] = label, span, threshold, cooldown
            sensitivity_rows.append(part)
        pd.concat(sensitivity_rows, ignore_index=True).to_csv(out/"sensitivity_analysis.csv", index=False)
        quality = []
        for market, tickers in (("US",us),("JP",jp)):
            for ticker in tickers:
                observations = int(original[[ticker,BENCHMARKS[market]]].dropna().shape[0]) if ticker in original else 0
                if observations < 253:
                    quality.append({"ticker":ticker,"market":market,"warning":"insufficient_common_history","observations":observations})
                if ticker in original and original[ticker].dropna().pct_change(fill_method=None).abs().gt(.5).any():
                    quality.append({"ticker":ticker,"market":market,"warning":"absolute_daily_return_above50pct_not_automatically_corrected","observations":observations})
        for note in warnings:
            quality.append({"ticker":"","market":"ALL","warning":note,"observations":None})
        pd.DataFrame(quality).to_csv(out/"data_quality_warnings.csv", index=False)
        meta["analyzed_ticker_count"] = int(daily.loc[daily.observations >= 253,"ticker"].nunique())
        meta["excluded_ticker_count"] = len(us)+len(jp)-meta["analyzed_ticker_count"]
        meta["event_counts"] = {"raw":len(raw),"first_per_reference":int(raw.first_per_reference.sum()),"independent":len(accepted),
                                "analysis_eligible":int(accepted.analysis_eligible.sum()),"matched":int((matches.status=="matched").sum()),
                                "price_relative_duplicate_ticker_dates":int(accepted.duplicated(["ticker","signal_date"]).sum()),
                                "reused_control_assignments":int(matches.loc[matches.status=="matched"].duplicated(["control_ticker","signal_date"]).sum())}
        meta["observed_outcome_counts"] = {str(h):int(((outcomes.horizon==h)&(outcomes.status=="observed")&(outcomes.group=="alert")).sum()) for h in HORIZONS}
        meta["correction_history"] = correction.to_dict("records")
        concentration = accepted.loc[accepted.analysis_eligible].groupby(["signal_date","market","series_kind"]).size().rename("alert_count").reset_index()
        concentration["benchmark_daily_return"] = [corrected[BENCHMARKS[row.market]].dropna().pct_change(fill_method=None).get(row.signal_date,np.nan) for row in concentration.itertuples()]
        concentration["market_drop_below_minus3pct"] = concentration.benchmark_daily_return < -.03
        concentration.to_csv(out/"event_concentration.csv",index=False)
        meta["event_concentration"] = {"maximum_same_date_market_series_count":int(concentration.alert_count.max()) if len(concentration) else 0,
                                       "alerts_on_benchmark_drop_below_minus3pct":int(concentration.loc[concentration.market_drop_below_minus3pct,"alert_count"].sum())}
        _charts(out/"charts",daily,raw,outcomes,baseline.loc[baseline.status=="observed"])
        after = {name:_hash(snapshot/name) for name in hashes}
        if after != hashes:
            raise AssertionError("Input snapshot changed during analysis")
        meta.update({"status":"completed","validation_status":"passed","finished_at":datetime.now(timezone.utc).isoformat(),"input_unchanged":True})
        (out/"study_report.md").write_text(_report(meta,summary,uncertainty),encoding="utf-8")
        _json(out/"study_metadata.json",meta)
        _json(out/"file_manifest.json",{"algorithm":"sha256","self_excluded":"file_manifest.json","files":{p.relative_to(out).as_posix():_hash(p) for p in sorted(out.rglob('*')) if p.is_file()}})
    except Exception as exc:
        meta.update({"status":"failed","validation_status":"failed","finished_at":datetime.now(timezone.utc).isoformat(),"error":f"{type(exc).__name__}: {exc}"})
        _json(out/"study_metadata.json",meta)
        (out/"study_report.md").write_text(f"# Incomplete Dow Momentum Study\n\nFailed: {meta['error']}\n\nNo completed study or superiority claim.\n",encoding="utf-8")
        raise StudyRunError(meta["error"],out) from exc
    return out


def _report(meta, summary, uncertainty):
    primary = summary.loc[(summary.horizon==21)&(summary.period=="ALL")&(summary.market=="ALL")&(summary.subgroup=="all")]
    text = ["# Dow Momentum Study v0.1", "", "Exploratory retrospective event study; this report does not establish predictive superiority.",
            "", "Primary horizon: 21 common valid intervals AFTER next-close entry. Main specification: corrected, span5, threshold0, cooldown21.",
            "", "## Primary descriptive results", "", "```",primary.to_string(index=False),"```", "", "## Counts", "",json.dumps(meta["event_counts"],indent=2),
            "", "## Interpretation and limitations", ""]
    text += [f"- {warning}" for warning in meta["warnings"]]
    text += ["- Consult uncertainty.csv for separate ticker/date cluster intervals and their non-calibrated envelope; small clusters or no matched pairs cannot support significance.",
             "- Positive-momentum subanalysis, markets and years are in summary_statistics.csv. Paired differences use only pairs observed at that horizon; marginal case/control means can cover different cohorts.",
             "- Momentum-only descriptive landmarks are separate from matched comparisons; missing future outcomes remain unobserved. Exclusion sensitivity preserves matching's information time.",
             "- event_concentration.csv records same-date market/series counts and benchmark daily changes (flag below-3%). It documents market shocks without claiming to remove all common-factor confounding.",
             "- Preliminary SNDK/DELL/4062.T charts are post-selected. Lowest/highest observed21 relative examples and the highest positive-stock-return counterexample (if any) are chosen only for illustration after analysis.",
             "- This is a close-only simplified confirmed-swing detector inspired by Dow theory, not a strict reproduction of classical Dow theory or an executable trading simulation."]
    return "\n".join(text)+"\n"


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261104)
    args=parser.parse_args(argv)
    try:
        result=run_study(**vars(args))
    except StudyRunError as exc:
        print(f"Study failed: {exc}\nIncomplete output: {exc.run_dir}")
        return 1
    print(f"Study output: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
