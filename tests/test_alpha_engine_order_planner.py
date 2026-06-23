import json

import pandas as pd
import pytest

import alpha_engine_order_planner as planner


def _screen_run(tmp_path, missing_price=False):
    rows = []
    for i in range(6):
        rows.append({
            "ticker": f"US{i}", "region": "US", "rank_in_region": i + 1,
            "Weight": 1 / 12, "reference_price_local": None if missing_price and i == 0 else 100 + i,
        })
    for i in range(6):
        rows.append({
            "ticker": f"JP{i}.T", "region": "JP", "rank_in_region": i + 1,
            "Weight": 1 / 12, "reference_price_local": 1000 + i * 10,
        })
    pd.DataFrame(rows).to_csv(tmp_path / "selected_tickers.csv", index=False)
    return tmp_path


def test_generates_plan_for_twelve_weighted_tickers(tmp_path):
    out = planner.build_buy_order_plan(
        _screen_run(tmp_path), total_capital_jpy=3_000_000, usd_jpy_rate=158.0,
        cash_buffer_pct=0.01, price_buffer_pct=0.02, us_order_unit=1, jp_order_unit=1,
    )
    plan = pd.read_csv(out / "buy_order_plan.csv")
    summary = pd.read_csv(out / "buy_order_summary.csv").iloc[0]
    report = (out / "buy_order_report.md").read_text()
    metadata = json.loads((out / "order_plan_metadata.json").read_text())

    assert len(plan) == 12
    assert set(planner.REQUIRED_PLAN_COLUMNS).issubset(plan.columns)
    assert (plan["planned_shares"] > 0).all()
    assert (plan["planned_shares"] % plan["order_unit"] == 0).all()
    assert summary["planned_total_cost_jpy"] <= summary["deployable_capital_jpy"]
    assert summary["residual_cash_jpy"] >= 0
    assert "ORDER PLAN ONLY — not automatic trading" in report
    assert "Source Live Screener run folder" in report
    assert metadata["no_automatic_trading"] is True


def test_order_units_are_respected(tmp_path):
    out = planner.build_buy_order_plan(
        _screen_run(tmp_path), total_capital_jpy=3_000_000, usd_jpy_rate=158.0,
        us_order_unit=5, jp_order_unit=100,
    )
    plan = pd.read_csv(out / "buy_order_plan.csv")
    assert (plan.loc[plan.region == "US", "planned_shares"] % 5 == 0).all()
    assert (plan.loc[plan.region == "JP", "planned_shares"] % 100 == 0).all()


def test_usd_jpy_unspecified_and_unavailable_stops_safely(tmp_path, monkeypatch):
    monkeypatch.setattr(planner, "fetch_usd_jpy_rate", lambda: (_ for _ in ()).throw(RuntimeError("no rate")))
    with pytest.raises(RuntimeError, match="no rate"):
        planner.build_buy_order_plan(_screen_run(tmp_path), usd_jpy_rate=None)


def test_missing_price_does_not_continue_silently(tmp_path):
    with pytest.raises(ValueError, match="Invalid reference_price_local"):
        planner.build_buy_order_plan(_screen_run(tmp_path, missing_price=True), usd_jpy_rate=158.0)


def test_series_string_reference_price_stops_with_explicit_contract_error(tmp_path):
    run = _screen_run(tmp_path)
    selected = pd.read_csv(run / "selected_tickers.csv")
    selected["reference_price_local"] = selected["reference_price_local"].astype(object)
    selected.loc[0, "reference_price_local"] = (
        "Date\n"
        "2023-06-12 NaN\n"
        "2026-06-23 1988.660034\n"
        "Name: SNDK, Length: 788, dtype: float64"
    )
    selected.to_csv(run / "selected_tickers.csv", index=False)
    with pytest.raises(
        ValueError,
        match=(
            "Invalid reference_price_local for US0.\n"
            "Expected one numeric latest price from Live Screener.\n"
            "Re-run the Live Screener with a valid price output."
        ),
    ):
        planner.build_buy_order_plan(run, usd_jpy_rate=158.0)


def test_audit_artifacts_and_no_execution_integration(tmp_path):
    out = planner.build_buy_order_plan(_screen_run(tmp_path), usd_jpy_rate=158.0)
    assert {"buy_order_plan.csv", "buy_order_summary.csv", "buy_order_report.md", "order_plan_metadata.json"}.issubset({p.name for p in out.iterdir()})
    source = open("alpha_engine_order_planner.py", encoding="utf-8").read().lower()
    forbidden = ["alpaca", "interactivebrokers", "ib_insync", "kabustation", "place_order", "submit_order"]
    assert not any(term in source for term in forbidden)


def _rounding_screen_run(tmp_path):
    rows = [
        {"ticker": "AAA", "region": "JP", "rank_in_region": 1, "Weight": 0.50, "reference_price_local": 5100},
        {"ticker": "BBB", "region": "JP", "rank_in_region": 2, "Weight": 0.30, "reference_price_local": 2900},
        {"ticker": "CCC", "region": "JP", "rank_in_region": 3, "Weight": 0.20, "reference_price_local": 1900},
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "selected_tickers.csv", index=False)
    return tmp_path


def test_floor_mode_preserves_simple_floor_behavior(tmp_path):
    out = planner.build_buy_order_plan(
        _rounding_screen_run(tmp_path), total_capital_jpy=10_000, usd_jpy_rate=158.0,
        cash_buffer_pct=0, price_buffer_pct=0, rounding_mode="floor",
    )
    plan = pd.read_csv(out / "buy_order_plan.csv")
    summary = pd.read_csv(out / "buy_order_summary.csv").iloc[0]
    assert plan["planned_shares"].tolist() == plan["floor_planned_shares"].tolist()
    assert summary["rounding_mode"] == "floor"
    assert summary["optimization_step_count"] == 0


def test_target_tracking_improves_floor_without_exceeding_deployable(tmp_path):
    out = planner.build_buy_order_plan(
        _rounding_screen_run(tmp_path), total_capital_jpy=10_000, usd_jpy_rate=158.0,
        cash_buffer_pct=0, price_buffer_pct=0, rounding_mode="target_tracking",
        min_one_unit_per_selected=True,
    )
    plan = pd.read_csv(out / "buy_order_plan.csv")
    summary = pd.read_csv(out / "buy_order_summary.csv").iloc[0]
    assert summary["optimized_total_cost_jpy"] <= summary["deployable_capital_jpy"]
    assert summary["optimized_weight_tracking_error"] < summary["initial_weight_tracking_error"]
    assert summary["optimized_residual_cash_jpy"] < summary["deployable_capital_jpy"] - summary["initial_floor_total_cost_jpy"]
    assert (plan["planned_shares"] >= plan["order_unit"]).all()
    assert (plan["planned_shares"] % plan["order_unit"] == 0).all()


def test_minimum_one_unit_errors_when_budget_is_too_small(tmp_path):
    with pytest.raises(ValueError, match="Insufficient deployable capital to buy at least one order unit"):
        planner.build_buy_order_plan(
            _rounding_screen_run(tmp_path), total_capital_jpy=4_000, usd_jpy_rate=158.0,
            cash_buffer_pct=0, price_buffer_pct=0, rounding_mode="target_tracking",
            min_one_unit_per_selected=True,
        )


def test_target_tracking_is_deterministic(tmp_path):
    run = _rounding_screen_run(tmp_path)
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    kwargs = dict(
        screen_run_dir=run, total_capital_jpy=10_000, usd_jpy_rate=158.0,
        cash_buffer_pct=0, price_buffer_pct=0, rounding_mode="target_tracking",
        min_one_unit_per_selected=True,
    )
    planner.build_buy_order_plan(output_dir=out1, **kwargs)
    planner.build_buy_order_plan(output_dir=out2, **kwargs)
    pd.testing.assert_frame_equal(
        pd.read_csv(out1 / "buy_order_plan.csv"),
        pd.read_csv(out2 / "buy_order_plan.csv"),
    )


def test_live_screener_selection_columns_are_not_rewritten(tmp_path):
    run = _rounding_screen_run(tmp_path)
    before = pd.read_csv(run / "selected_tickers.csv")
    planner.build_buy_order_plan(run, total_capital_jpy=10_000, usd_jpy_rate=158.0, cash_buffer_pct=0, price_buffer_pct=0)
    after = pd.read_csv(run / "selected_tickers.csv")
    pd.testing.assert_frame_equal(before, after)
