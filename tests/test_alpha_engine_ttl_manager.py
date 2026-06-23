import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import alpha_engine_ttl_manager as ttl


def prices():
    idx = pd.bdate_range(end=pd.Timestamp.now('UTC').normalize().tz_localize(None), periods=320)
    data = {"AAA": np.linspace(100, 150, len(idx)), "BBB": np.linspace(100, 50, len(idx)), "SPY": np.linspace(400, 430, len(idx)), "1306.T": np.linspace(2000, 2100, len(idx))}
    return pd.DataFrame(data, index=idx)


def ledger(tmp_path, age, status="ACTIVE", ticker="AAA", region="US"):
    today = pd.Timestamp.now('UTC').normalize().tz_localize(None)
    start = today - pd.Timedelta(days=age)
    p = tmp_path / "live_holdings_ledger.csv"
    pd.DataFrame([{c:"" for c in ttl.LEDGER_COLUMNS}]).assign(
        ticker=ticker, region=region, actual_shares=10, actual_entry_date=start.date().isoformat(),
        actual_entry_price_local=100, currency="USD" if region=="US" else "JPY", source_screen_run="s", source_order_plan="o",
        status=status, cycle_start_date=start.date().isoformat(), ttl_review_due_date=(start+pd.Timedelta(days=90)).date().isoformat(),
        renewal_expiry_date=(start+pd.Timedelta(days=120)).date().isoformat() if status=="RENEWED" else "", last_review_date="", last_decision="", notes=""
    )[ttl.LEDGER_COLUMNS].to_csv(p, index=False)
    return p


def patch_health(monkeypatch, passed=True):
    def fake(*args, **kwargs):
        return {"market_rank": 1 if passed else 99, "rank_pass": passed, "residual_score": 0.1 if passed else -0.1,
                "residual_pass": passed, "price_above_50dma": passed, "composite_count": 3 if passed else 0,
                "composite_pass": passed}
    monkeypatch.setattr(ttl.alpha, "_health_row", fake)


def run(tmp_path, p, monkeypatch, passed=True):
    patch_health(monkeypatch, passed)
    return ttl.run_ttl_manager(p, tmp_path / "runs", prices=prices(), us=["AAA","BBB"], jp=[])


def read(out, name):
    return pd.read_csv(Path(out) / name)


def test_day89_no_sell_or_renew(tmp_path, monkeypatch):
    out = run(tmp_path, ledger(tmp_path, 89), monkeypatch, passed=False)
    dec = read(out, "ttl_review_decisions.csv")
    assert dec.loc[0, "review_type"] == "not_due"
    assert dec.loc[0, "status_after"] == "ACTIVE"
    assert read(out, "sell_order_plan.csv").empty


def test_day90_composite_pass_renews(tmp_path, monkeypatch):
    out = run(tmp_path, ledger(tmp_path, 90), monkeypatch, passed=True)
    dec = read(out, "ttl_review_decisions.csv")
    assert dec.loc[0, "status_after"] == "RENEWED"
    assert read(out, "sell_order_plan.csv").empty


def test_day90_composite_fail_sell_pending_no_replenishment(tmp_path, monkeypatch):
    out = run(tmp_path, ledger(tmp_path, 90), monkeypatch, passed=False)
    dec = read(out, "ttl_review_decisions.csv")
    sells = read(out, "sell_order_plan.csv")
    assert dec.loc[0, "status_after"] == "SELL_PENDING"
    assert sells.loc[0, "recommended_action"] == "SELL_MANUAL_MARKET_NEXT_TRADABLE_SESSION"
    assert "cash" in sells.loc[0, "reason"].lower()


def test_renewal_weekly_fail_becomes_pending(tmp_path, monkeypatch):
    out = run(tmp_path, ledger(tmp_path, 100, status="RENEWED"), monkeypatch, passed=False)
    dec = read(out, "ttl_review_decisions.csv")
    assert dec.loc[0, "status_after"] == "RENEWAL_FAILED_SELL_PENDING"
    assert not read(out, "renewal_decisions.csv").empty


def test_day120_reconstitution_not_second_renewal(tmp_path, monkeypatch):
    out = run(tmp_path, ledger(tmp_path, 120, status="RENEWED"), monkeypatch, passed=True)
    dec = read(out, "ttl_review_decisions.csv")
    assert dec.loc[0, "status_after"] == "RECONSTITUTION_REQUIRED"
    assert not read(out, "reconstitution_required.csv").empty


def test_missing_ledger_safety_stops(tmp_path):
    with pytest.raises(RuntimeError, match="DATA_BLOCKED"):
        ttl.run_ttl_manager(tmp_path / "missing.csv", tmp_path / "runs", prices=prices(), us=["AAA"], jp=[])


def test_initialize_ledger_requires_actual_fills_not_plan(tmp_path):
    fills = tmp_path / "fills.csv"
    pd.DataFrame([{"ticker":"AAA","region":"US","planned_shares":10,"actual_shares":"","actual_entry_date":"","actual_entry_price_local":"","currency":"USD"}]).to_csv(fills,index=False)
    with pytest.raises(RuntimeError, match="Actual fill fields"):
        ttl.initialize_ledger(fills, tmp_path / "ledger.csv")


def test_apply_confirmation_does_not_zero_without_confirmation():
    with pytest.raises(RuntimeError, match="automatic zeroing"):
        ttl.apply_execution_confirmation()


def test_jp_benchmark_missing_blocks(tmp_path, monkeypatch):
    p = prices().drop(columns=["1306.T"])
    patch_health(monkeypatch, True)
    with pytest.raises(RuntimeError, match="Benchmark unavailable"):
        ttl.run_ttl_manager(ledger(tmp_path, 90, ticker="BBB", region="JP"), tmp_path / "runs", prices=p, us=[], jp=["BBB"])


def test_outputs_and_metadata_are_deterministic_shape(tmp_path, monkeypatch):
    out = run(tmp_path, ledger(tmp_path, 90), monkeypatch, passed=True)
    required = ["ttl_review_decisions.csv","renewal_decisions.csv","sell_order_plan.csv","reconstitution_required.csv","live_holdings_ledger_proposed.csv","data_quality.csv","download_failures.csv","metadata.json","ttl_review_report.md"]
    assert all((Path(out)/x).exists() for x in required)
    meta = json.loads((Path(out)/"metadata.json").read_text())
    assert meta["mode"] == "live_ttl_manager"


def test_no_auto_trading_or_full_backtest_calls_present():
    src = Path("alpha_engine_ttl_manager.py").read_text()
    assert "run_ttl_renewal_variant" not in src
    assert "broker" not in src.lower() or "brokerage" in src.lower()
    assert "send_order" not in src and "place_order" not in src
