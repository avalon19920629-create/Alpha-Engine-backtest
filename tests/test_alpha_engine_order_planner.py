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
    with pytest.raises(ValueError, match="Invalid reference price"):
        planner.build_buy_order_plan(_screen_run(tmp_path, missing_price=True), usd_jpy_rate=158.0)


def test_audit_artifacts_and_no_execution_integration(tmp_path):
    out = planner.build_buy_order_plan(_screen_run(tmp_path), usd_jpy_rate=158.0)
    assert {"buy_order_plan.csv", "buy_order_summary.csv", "buy_order_report.md", "order_plan_metadata.json"}.issubset({p.name for p in out.iterdir()})
    source = open("alpha_engine_order_planner.py", encoding="utf-8").read().lower()
    forbidden = ["alpaca", "interactivebrokers", "ib_insync", "kabustation", "place_order", "submit_order"]
    assert not any(term in source for term in forbidden)
