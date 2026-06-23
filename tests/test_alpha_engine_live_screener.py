import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import alpha_engine_live_screener as live


def _prices():
    idx = pd.bdate_range("2024-01-01", periods=390)
    names = [*(f"US{i}" for i in range(8)), *(f"JP{i}.T" for i in range(8)), "SPY", "1306.T", "^GSPC", "^TOPX"]
    data = {}
    for i, n in enumerate(names):
        drift = 0.0002 + (i % 6) * 0.00008
        vol = 0.006 + (i % 4) * 0.001
        data[n] = 100 * np.exp(np.cumsum(np.random.default_rng(i).normal(drift, vol, len(idx))))
    return pd.DataFrame(data, index=idx)


def _run(tmp_path, residual=60, n=12, prices=None):
    us = [f"US{i}" for i in range(8)]
    jp = [f"JP{i}.T" for i in range(8)]
    return live.run_live_screening(prices=prices if prices is not None else _prices(), us=us, jp=jp, residual_ratio=residual, total_holdings=n, output_root=tmp_path)


def test_n12_selects_us6_jp6_and_outputs_artifacts(tmp_path):
    out = _run(tmp_path, 60, 12)
    selected = pd.read_csv(out / "selected_tickers.csv")
    assert selected.groupby("region").size().to_dict() == {"JP": 6, "US": 6}
    assert {p.name for p in out.iterdir()} >= {"ranked_candidates_us.csv", "ranked_candidates_jp.csv", "selected_tickers.csv", "adopted_weights.csv", "score_components.csv", "data_quality.csv", "download_failures.csv", "metadata.json", "screen_report.md"}


def test_n6_selects_us3_jp3(tmp_path):
    out = _run(tmp_path, 100, 6)
    selected = pd.read_csv(out / "selected_tickers.csv")
    assert selected.groupby("region").size().to_dict() == {"JP": 3, "US": 3}


def test_odd_n_is_explicit_error():
    with pytest.raises(ValueError, match="must be even"):
        live.build_live_variant(60, 11)


def test_metadata_records_residual60_and_residual100(tmp_path):
    out60 = _run(tmp_path / "r60", 60, 12)
    out100 = _run(tmp_path / "r100", 100, 6)
    assert json.loads((out60 / "metadata.json").read_text())["residual_ratio"] == 60
    assert json.loads((out100 / "metadata.json").read_text())["residual_ratio"] == 100
    assert json.loads((out100 / "metadata.json").read_text())["total_holdings"] == 6


def test_selected_weights_sum_to_one(tmp_path):
    out = _run(tmp_path, 60, 12)
    weights = pd.read_csv(out / "adopted_weights.csv")
    assert weights["Weight"].sum() == pytest.approx(1.0)


def test_does_not_call_ttl_or_backtest_functions(tmp_path):
    with patch("alpha_engine_backtest.run_ttl_renewal_variant", side_effect=AssertionError("ttl called")), patch("alpha_engine_backtest.run_backtest", side_effect=AssertionError("backtest called")):
        _run(tmp_path, 60, 12)


def test_jp_benchmark_unavailable_stops_without_silent_substitution(tmp_path):
    prices = _prices().drop(columns=["1306.T", "^TOPX"])
    with pytest.raises(RuntimeError, match="JP benchmark unavailable"):
        _run(tmp_path, 60, 12, prices=prices)
