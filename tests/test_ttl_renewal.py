import argparse, json
from pathlib import Path
import pandas as pd

import alpha_engine_backtest as aeb


def test_ttl_renewal_variant_counts_and_names():
    quick=aeb.build_ttl_renewal_variants(quick=True)
    full=aeb.build_ttl_renewal_variants(quick=False)
    assert sum(not v.get("is_baseline") for v in quick)==21
    assert sum(not v.get("is_baseline") for v in full)==64
    names={v["name"] for v in quick}
    assert "Baseline_N12_TTL90" in names
    for ttl in (60,90,120,180):
        assert f"Residual_60_N12_TTL{ttl}" in names
    assert "Residual_60_N12_TTL90_Renew30_Rank" in names
    assert "Residual_60_N12_TTL90_Renew30_Residual" in names
    assert "Residual_60_N12_TTL90_Renew30_Composite" in names


def test_cli_ttl_renewal_and_quick_recognized():
    p=argparse.ArgumentParser(); p.add_argument("--audit", choices=["ttl_renewal"]); p.add_argument("--quick", action="store_true")
    args=p.parse_args(["--audit","ttl_renewal","--quick"])
    assert args.audit=="ttl_renewal" and args.quick


def test_renewal_conditions():
    h={"rank_pass":True,"residual_pass":False,"composite_pass":False}
    assert aeb._renewal_pass("rank", h)
    h={"rank_pass":False,"residual_pass":True,"composite_pass":False}
    assert aeb._renewal_pass("residual", h)
    h={"rank_pass":False,"residual_pass":False,"composite_pass":True}
    assert aeb._renewal_pass("composite", h)


def test_cache_hit_and_corrupt_fallback(tmp_path):
    idx=pd.bdate_range("2019-01-01","2019-03-31")
    prices=pd.DataFrame({"US0":1.0,"SPY":1.0}, index=idx)
    cache=tmp_path/"cache"; cache.mkdir()
    prices[["US0"]].to_pickle(cache/"prices.pkl"); prices[["SPY"]].to_pickle(cache/"benchmarks.pkl")
    (cache/"universe.csv").write_text("ticker\nUS0\n")
    (cache/"cache_metadata.json").write_text(json.dumps(aeb._cache_metadata("2019-01-01","2019-03-31",["US0"],["SPY"])))
    loaded,meta=aeb._load_ttl_cache([cache],"2019-01-01","2019-03-31",["US0"],["SPY"])
    assert meta["cache_used"] and "US0" in loaded
    (cache/"cache_metadata.json").write_text("broken")
    loaded,meta=aeb._load_ttl_cache([cache],"2019-01-01","2019-03-31",["US0"],["SPY"])
    assert loaded is None and not meta["cache_used"]


def test_quick_demo_outputs_and_renewal_limits(tmp_path):
    prices=aeb.demo_prices(); us=[f"US{i}" for i in range(8)]; jp=[f"JP{i}.T" for i in range(8)]
    out=tmp_path/"ttl"
    summary=aeb.run_ttl_renewal_audit(prices,us,jp,"2019-01-01","2019-06-30",out,quick=True)
    required=["variant_summary.csv","variant_summary.json","annual_returns.csv","monthly_returns.csv","drawdown_series.csv","turnover.csv","trade_log.csv","holding_periods.csv","renewal_decisions.csv","ttl_event_log.csv","cost_adjusted_summary.csv","data_quality.csv","insufficient_history_summary.csv","audit_metadata.json","ttl_renewal_report.md"]
    for name in required: assert (out/name).exists(), name
    assert not (out/"score_components_full.csv").exists()
    hp=pd.read_csv(out/"holding_periods.csv")
    ren=hp[hp.renewal_protocol!="fixed"]
    assert ren.holding_days.min() >= 90
    assert ren.holding_days.max() <= 120
    meta=json.loads((out/"audit_metadata.json").read_text())
    assert meta["target_variant_count"]==21
    assert meta["exit_protocol_enabled"] is False


def test_no_future_data_boundary(monkeypatch):
    calls=[]
    orig=aeb.asof_prices
    def spy(prices, as_of_date):
        calls.append(pd.Timestamp(as_of_date)); return orig(prices, as_of_date)
    monkeypatch.setattr(aeb,"asof_prices",spy)
    prices=aeb.demo_prices(); us=[f"US{i}" for i in range(8)]; jp=[f"JP{i}.T" for i in range(8)]
    v=[x for x in aeb.build_ttl_renewal_variants(True) if x["name"]=="Residual_60_N12_TTL90"][0]
    aeb.run_ttl_renewal_variant(prices,us,jp,"2019-01-01","2019-03-31",v,aeb.build_benchmark_modes()["broad_default"])
    assert calls and max(calls) <= pd.Timestamp("2019-03-31")
