import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import alpha_engine_live_screener as live
import alpha_engine_order_planner as planner


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


def test_reference_prices_are_numeric_scalars_and_dates(tmp_path):
    out = _run(tmp_path, 60, 12)
    selected = pd.read_csv(out / "selected_tickers.csv")
    report = (out / "screen_report.md").read_text()
    assert "reference_price_date" in selected.columns
    assert pd.api.types.is_numeric_dtype(selected["reference_price_local"])
    assert selected["reference_price_local"].notna().all()
    assert (selected["reference_price_local"] > 0).all()
    assert selected["reference_price_local"].map(np.isscalar).all()
    assert "last valid closing price (`reference_price_local`)" in report


def test_order_planner_accepts_live_screener_run_and_generates_costs(tmp_path):
    screen_out = _run(tmp_path / "screen", 60, 12)
    plan_out = planner.build_buy_order_plan(
        screen_out, total_capital_jpy=3_000_000, usd_jpy_rate=158.0,
        cash_buffer_pct=0.01, price_buffer_pct=0.02, us_order_unit=1, jp_order_unit=1,
    )
    plan = pd.read_csv(plan_out / "buy_order_plan.csv")
    summary = pd.read_csv(plan_out / "buy_order_summary.csv").iloc[0]
    assert (plan["action"] == "BUY").all()
    assert (plan["planned_shares"] > 0).all()
    assert (plan["planned_cost_jpy"] > 0).all()
    assert summary["planned_total_cost_jpy"] > 0
    assert summary["residual_cash_jpy"] >= 0


def test_reference_price_uses_last_valid_close_when_latest_is_nan(tmp_path):
    prices = _prices()
    prices.loc[prices.index[-1], "US0"] = np.nan
    out = _run(tmp_path, 60, 12, prices=prices)
    selected = pd.read_csv(out / "selected_tickers.csv")
    if "US0" in set(selected["ticker"]):
        row = selected[selected["ticker"] == "US0"].iloc[0]
        expected = prices["US0"].dropna().iloc[-1]
        expected_date = prices["US0"].dropna().index[-1].date().isoformat()
        assert row["reference_price_local"] == pytest.approx(expected)
        assert row["reference_price_date"] == expected_date


def test_reference_price_extractor_handles_yfinance_multiindex_close():
    idx = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame(
        {
            ("Close", "AAA"): [10.0, np.nan, 12.5],
            ("Open", "AAA"): [9.0, 11.0, 12.0],
            ("Close", "BBB"): [20.0, 21.0, np.nan],
        },
        index=idx,
    )
    price, date = live._latest_valid_close_for_ticker(prices, "BBB")
    assert price == pytest.approx(21.0)
    assert date == idx[1].date().isoformat()


def test_does_not_call_ttl_or_backtest_functions(tmp_path):
    with patch("alpha_engine_backtest.run_ttl_renewal_variant", side_effect=AssertionError("ttl called")), patch("alpha_engine_backtest.run_backtest", side_effect=AssertionError("backtest called")):
        _run(tmp_path, 60, 12)


def test_jp_benchmark_unavailable_stops_without_silent_substitution(tmp_path):
    prices = _prices().drop(columns=["1306.T", "^TOPX"])
    with pytest.raises(RuntimeError, match="JP benchmark unavailable"):
        _run(tmp_path, 60, 12, prices=prices)
