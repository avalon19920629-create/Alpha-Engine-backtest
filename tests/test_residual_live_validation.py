import json
from pathlib import Path

import numpy as np
import pandas as pd

import alpha_engine_backtest as aeb


def _synthetic_prices():
    idx = pd.bdate_range("2018-01-01", "2021-12-31")
    rng = np.random.default_rng(7)
    names = [*(f"US{i}" for i in range(8)), *(f"JP{i}.T" for i in range(8)), "SPY", "QQQ", "1306.T", "^N225", "^GSPC", "^TOPX"]
    data = {}
    for i, n in enumerate(names):
        drift = 0.00015 + (i % 4) * 0.00004
        vol = 0.008 + (i % 5) * 0.001
        data[n] = 100 * np.exp(np.cumsum(rng.normal(drift, vol, len(idx))))
    return pd.DataFrame(data, index=idx)


def test_live_variants_keep_baseline_and_weights_sum_to_one():
    variants = aeb.build_residual_live_variants()
    baseline = variants[0]
    assert baseline["name"] == "Baseline"
    assert baseline["base_weight"] == 1.0
    assert baseline["residual_weight"] == 0.0
    assert all(abs(v["base_weight"] + v["residual_weight"] - 1.0) < 1e-12 for v in variants)
    assert all(v["vcp_weight"] == 0.0 for v in variants)


def test_simple_residual_score_calculates_stock_minus_benchmark():
    idx = pd.bdate_range("2020-01-01", periods=300)
    prices = pd.DataFrame({"AAA": np.linspace(100, 150, 300), "SPY": np.linspace(100, 120, 300)}, index=idx)
    score = aeb.compute_residual_momentum_score(prices, ["AAA"], idx[-1], "US", benchmark_mode={"US": ("SPY",)})
    expected_values = []
    for n, w in ((63, 1), (126, 2), (252, 3)):
        expected_values.append(w * ((prices["AAA"].iloc[-1] / prices["AAA"].iloc[-n] - 1) - (prices["SPY"].iloc[-1] / prices["SPY"].iloc[-n] - 1)))
    expected = sum(expected_values) / 6
    assert score.loc["AAA", "residual_raw"] == expected
    assert score.loc["AAA", "residual_score"] == 1.0


def test_missing_benchmark_does_not_crash_and_neutralizes_score():
    idx = pd.bdate_range("2020-01-01", periods=300)
    prices = pd.DataFrame({"AAA": np.linspace(100, 150, 300)}, index=idx)
    score = aeb.compute_residual_momentum_score(prices, ["AAA"], idx[-1], "US", benchmark_mode={"US": ("MISSING",)})
    assert np.isnan(score.loc["AAA", "residual_raw"])
    base = aeb.score_universe(prices, ["AAA"], idx[-1])
    combined = aeb.combine_residual_score(base, score, {"base_weight": 0.8, "residual_weight": 0.2})
    assert combined.loc["AAA", "residual_score"] == 0.5


def test_download_failure_is_logged_without_crashing():
    def failing_downloader(*args, **kwargs):
        raise RuntimeError("temporary yfinance outage")

    prices, failures, requested = aeb.download_live_universe_prices(["BRK.B", "7203.T"], "2020-01-01", "2020-12-31", downloader=failing_downloader)
    assert prices.empty
    assert requested == ["BRK-B", "7203.T"]
    assert set(failures["ticker"]) == {"BRK-B", "7203.T"}
    assert failures["reason"].str.contains("download_exception").all()


def test_residual_is_not_hard_filter_and_no_exit_regime_vcp_flags(tmp_path):
    prices = _synthetic_prices()
    us = [f"US{i}" for i in range(8)]
    jp = [f"JP{i}.T" for i in range(8)]
    summary = aeb.run_residual_live_validation(prices, us, jp, "2019-01-01", "2021-12-31", tmp_path)
    assert "Baseline" in summary.index
    meta = json.loads((tmp_path / "audit_metadata.json").read_text())
    assert meta["whether_exit_protocol_used"] is False
    assert meta["whether_regime_filter_used"] is False
    assert meta["whether_vcp_used"] is False
    assert meta["ttl_days"] == 90
    scores = pd.read_csv(tmp_path / "score_components.csv")
    selected = pd.read_csv(tmp_path / "selected_tickers.csv")
    assert "vcp_score" not in scores.columns
    assert set(scores["residual_method"]) == {"simple"}
    # Weak residual names can still appear when base score and rank combine; residual is a score component, not a filter.
    assert selected["ticker"].nunique() > 0


def test_residual_live_validation_outputs_required_files(tmp_path):
    prices = _synthetic_prices()
    us = [f"US{i}" for i in range(8)]
    jp = [f"JP{i}.T" for i in range(8)]
    aeb.run_residual_live_validation(prices, us, jp, "2019-01-01", "2021-12-31", tmp_path)
    required = [
        "variant_summary.csv", "variant_summary.json", "annual_returns.csv", "monthly_returns.csv",
        "drawdown_series.csv", "turnover.csv", "selected_tickers.csv", "selection_diff.csv",
        "score_components.csv", "benchmark_sensitivity.csv", "data_quality.csv", "download_failures.csv",
        "audit_metadata.json",
    ]
    assert all((tmp_path / name).is_file() for name in required)
    assert Path("reports/residual_live_validation_report.md").is_file()
