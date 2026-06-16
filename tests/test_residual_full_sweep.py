import json
from pathlib import Path

import numpy as np
import pandas as pd

import alpha_engine_backtest as aeb


def _synthetic_prices():
    idx = pd.bdate_range("2018-01-01", "2021-12-31")
    rng = np.random.default_rng(11)
    names = [*(f"US{i}" for i in range(8)), *(f"JP{i}.T" for i in range(8)), "SPY", "QQQ", "1306.T", "^N225", "^GSPC", "^TOPX"]
    data = {}
    for i, n in enumerate(names):
        data[n] = 100 * np.exp(np.cumsum(rng.normal(0.00018 + (i % 4) * 0.00004, 0.009 + (i % 5) * 0.001, len(idx))))
    return pd.DataFrame(data, index=idx)


def test_full_sweep_variants_baseline_and_five_percent_grid():
    variants = aeb.build_residual_full_sweep_variants()
    assert len(variants) == 21
    assert variants[0]["name"] == "Baseline"
    assert variants[0]["base_weight"] == 1.0
    assert variants[0]["residual_weight"] == 0.0
    assert variants[-1]["name"] == "Residual_100"
    assert variants[-1]["base_weight"] == 0.0
    assert variants[-1]["residual_weight"] == 1.0
    assert [v["residual_weight"] for v in variants] == [round(x / 100, 2) for x in range(0, 101, 5)]
    assert all(abs(v["base_weight"] + v["residual_weight"] - 1.0) < 1e-12 for v in variants)
    assert all(v["vcp_weight"] == 0.0 for v in variants)


def test_simple_residual_and_missing_benchmark_neutral_handling():
    idx = pd.bdate_range("2020-01-01", periods=300)
    prices = pd.DataFrame({"AAA": np.linspace(100, 150, 300), "SPY": np.linspace(100, 120, 300)}, index=idx)
    score = aeb.compute_residual_momentum_score(prices, ["AAA"], idx[-1], "US", benchmark_mode={"US": ("SPY",)})
    expected = sum(w * ((prices["AAA"].iloc[-1] / prices["AAA"].iloc[-n] - 1) - (prices["SPY"].iloc[-1] / prices["SPY"].iloc[-n] - 1)) for n, w in ((63, 1), (126, 2), (252, 3))) / 6
    assert score.loc["AAA", "residual_raw"] == expected

    missing = aeb.compute_residual_momentum_score(prices.drop(columns=["SPY"]), ["AAA"], idx[-1], "US", benchmark_mode={"US": ("MISSING",)})
    base = aeb.score_universe(prices.drop(columns=["SPY"]), ["AAA"], idx[-1])
    combined = aeb.combine_residual_score(base, missing, {"base_weight": 0.0, "residual_weight": 1.0})
    assert combined.loc["AAA", "residual_score"] == 0.5


def test_download_failure_and_cache_corruption_are_non_fatal(tmp_path):
    def failing_downloader(*args, **kwargs):
        raise RuntimeError("blocked")

    prices, failures, requested, meta = aeb.build_price_cache(["BRK.B", "7203.T"], ["SPY"], "2020-01-01", "2020-12-31", tmp_path / "cache", downloader=failing_downloader)
    assert prices.empty
    assert set(failures["ticker"]) == {"BRK-B", "7203.T", "SPY"}
    assert meta["cache_used"] is False
    ok, reason = aeb.validate_price_cache(tmp_path / "cache", "2020-01-01", "2020-12-31", ["BRK-B", "7203.T"], ["SPY"])
    assert ok
    (tmp_path / "cache" / "prices.pkl").write_text("not a pickle")
    ok, reason = aeb.validate_price_cache(tmp_path / "cache", "2020-01-01", "2020-12-31", ["BRK-B", "7203.T"], ["SPY"])
    assert not ok
    assert "cache_corrupt" in reason


def test_peak_plateau_and_full_sweep_outputs(tmp_path):
    prices = _synthetic_prices()
    us = [f"US{i}" for i in range(8)]
    jp = [f"JP{i}.T" for i in range(8)]
    summary = aeb.run_residual_full_sweep(prices, us, jp, "2019-01-01", "2021-12-31", tmp_path)
    assert "Baseline" in summary.index
    assert "Residual_100" in summary.index
    required = [
        "variant_summary.csv", "variant_summary.json", "annual_returns.csv", "monthly_returns.csv",
        "drawdown_series.csv", "turnover.csv", "selected_tickers.csv", "selection_diff.csv",
        "score_components.csv", "benchmark_sensitivity.csv", "data_quality.csv", "download_failures.csv",
        "peak_ratio_diagnostics.csv", "plateau_analysis.csv", "audit_metadata.json",
        "cache/prices.pkl", "cache/benchmarks.pkl", "cache/universe.csv", "cache/cache_metadata.json",
    ]
    assert all((tmp_path / name).is_file() for name in required)
    peak = pd.read_csv(tmp_path / "peak_ratio_diagnostics.csv")
    assert {"best_cagr_ratio", "best_calmar_ratio", "lowest_maxdd_ratio"}.issubset(set(peak["diagnostic"]))
    plateau = pd.read_csv(tmp_path / "plateau_analysis.csv")
    assert not plateau.empty
    meta = json.loads((tmp_path / "audit_metadata.json").read_text())
    assert meta["whether_exit_protocol_used"] is False
    assert meta["whether_regime_filter_used"] is False
    assert meta["whether_vcp_used"] is False
    assert meta["ttl_days"] == 90
    scores = pd.read_csv(tmp_path / "score_components.csv")
    assert "vcp_score" not in scores.columns
    assert set(scores["residual_method"]) == {"simple"}
    assert Path("reports/residual_full_sweep_report.md").is_file()
