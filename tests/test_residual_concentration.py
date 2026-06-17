import json
from pathlib import Path

import numpy as np
import pandas as pd

import alpha_engine_backtest as ae


def test_baseline_n12_and_variant_matrix():
    sizes=ae.build_portfolio_size_configs()
    assert [(x["total_holdings"],x["us_holdings"],x["jp_holdings"]) for x in sizes]==[(40,20,20),(30,15,15),(24,12,12),(20,10,10),(16,8,8),(12,6,6),(10,5,5),(8,4,4),(6,3,3),(4,2,2)]
    variants=ae.build_residual_concentration_variants()
    assert len(variants)==70
    base=[v for v in variants if v["name"]=="Baseline_N12"][0]
    assert base["base_weight"]==1.0 and base["residual_weight"]==0.0
    assert (base["total_holdings"],base["us_holdings"],base["jp_holdings"])==(12,6,6)
    assert all(abs(v["base_weight"]+v["residual_weight"]-1.0)<1e-9 for v in variants)


def test_simple_residual_and_missing_benchmark_neutral():
    idx=pd.bdate_range("2020-01-01",periods=300)
    prices=pd.DataFrame({"A":np.linspace(100,160,len(idx)),"SPY":np.linspace(100,130,len(idx))},index=idx)
    scored=ae.compute_residual_momentum_score(prices,["A"],idx[-1],"US",benchmark_mode={"US":("SPY",)})
    assert "residual_score" in scored.columns
    assert scored.loc["A","residual_raw"]>0
    neutral=ae.compute_residual_momentum_score(prices[["A"]],["A"],idx[-1],"US",benchmark_mode={"US":("MISSING",)})
    assert neutral.loc["A","residual_score"]==0.5


def test_download_failure_and_cache_corruption(tmp_path):
    def bad_downloader(*args, **kwargs):
        raise RuntimeError("boom")
    prices,failures,requested=ae.download_live_universe_prices(["A","BRK.B"],"2020-01-01","2020-12-31",downloader=bad_downloader)
    assert prices.empty
    assert set(failures["ticker"])=={"A","BRK-B"}
    c=tmp_path/"cache"; c.mkdir()
    (c/"prices.pkl").write_text("bad")
    (c/"benchmarks.pkl").write_text("bad")
    (c/"cache_metadata.json").write_text(json.dumps({"start":"2020-01-01","end":"2020-12-31","requested_tickers":["A"],"benchmark_tickers":["SPY"]}))
    ok,reason=ae.validate_price_cache(c,"2020-01-01","2020-12-31",["A"],["SPY"])
    assert not ok and "cache_corrupt" in reason


def test_residual_not_hard_filter_and_no_forbidden_flags():
    base=pd.DataFrame({"Total_Score":[3.0,2.0],"Volatility":[0.2,0.2]},index=["A","B"])
    residual=pd.DataFrame({"residual_score":[0.0,1.0],"residual_raw":[-1,1]},index=["A","B"])
    v={"base_weight":0.9,"residual_weight":0.1}
    combined=ae.combine_residual_score(base,residual,v)
    assert set(combined.index)=={"A","B"}
    variants=ae.build_residual_concentration_variants()
    assert all(x.get("vcp_weight",0.0)==0.0 for x in variants)


def test_demo_audit_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(ae, "RESIDUAL_CONCENTRATION_WEIGHTS", (0.0, 0.6))
    monkeypatch.setattr(ae, "RESIDUAL_CONCENTRATION_SIZES", ((12,6,6),(8,4,4)))
    prices=ae.demo_prices()
    us=[f"US{i}" for i in range(8)]
    jp=[f"JP{i}.T" for i in range(8)]
    out=tmp_path/"residual_concentration"
    summary=ae.run_residual_concentration_audit(prices,us,jp,"2019-01-01","2020-12-31",out)
    assert len(summary)==4
    required=["variant_summary.csv","selected_tickers.csv","selection_diff.csv","score_components.csv","data_quality.csv","download_failures.csv","concentration_diagnostics.csv","best_by_portfolio_size.csv","best_by_residual_ratio.csv","sweet_spot_analysis.csv","diversification_reference_analysis.csv","audit_metadata.json"]
    for name in required:
        assert (out/name).exists(), name
    meta=json.loads((out/"audit_metadata.json").read_text())
    assert meta["whether_exit_protocol_used"] is False
    assert meta["whether_regime_filter_used"] is False
    assert meta["whether_vcp_used"] is False
    assert meta["whether_sector_residual_used"] is False
    assert meta["whether_downside_penalty_used"] is False
    assert meta["whether_correlation_penalty_used"] is False


def test_quick_mode_variant_matrix():
    portfolio_configs=tuple(c for c in ae.build_portfolio_size_configs() if c["total_holdings"] in (24,20,16,12,10,8,6))
    baseline_cfg=next(c for c in ae.build_portfolio_size_configs() if c["total_holdings"]==12)
    variants=(ae.build_residual_concentration_variants(residual_weights=(0.0,),portfolio_configs=(baseline_cfg,))
              + ae.build_residual_concentration_variants(residual_weights=(0.60,0.65,1.00),portfolio_configs=portfolio_configs))
    quick_targets=[v for v in variants if not v["name"].startswith("Baseline_")]
    assert len(quick_targets)==21
    assert len(variants)==22
    assert any(v["name"]=="Baseline_N12" and v["base_weight"]==1.0 and v["residual_weight"]==0.0 for v in variants)


def test_concentration_selection_diff_uses_baseline_n12_and_fast_lookup(monkeypatch):
    selected=pd.DataFrame([
        {"screen_date":"2020-01-01","variant":"Baseline_N12","ticker":"A"},
        {"screen_date":"2020-01-01","variant":"Residual_60_N12","ticker":"B"},
    ])
    scores=pd.DataFrame([
        {"date":"2020-01-01","variant":"Baseline_N12","ticker":"A","base_score":1,"residual_score":.5,"final_score":1},
        {"date":"2020-01-01","variant":"Residual_60_N12","ticker":"B","base_score":2,"residual_score":.8,"final_score":2},
    ])
    def forbidden_to_datetime(*args, **kwargs):
        return pd.to_datetime(*args, **kwargs)
    diff=ae.compare_concentration_selection_diff(selected,scores,baseline_variant="Baseline_N12")
    assert set(diff["change"])=={"added","removed"}
    assert set(diff["variant"])=={"Residual_60_N12"}
    assert "Baseline" not in set(selected["variant"])


def test_selection_diff_failure_keeps_primary_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(ae, "RESIDUAL_CONCENTRATION_WEIGHTS", (0.0, 0.6))
    monkeypatch.setattr(ae, "RESIDUAL_CONCENTRATION_SIZES", ((12,6,6),))
    monkeypatch.setattr(ae, "compare_concentration_selection_diff", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    prices=ae.demo_prices(); us=[f"US{i}" for i in range(8)]; jp=[f"JP{i}.T" for i in range(8)]
    out=tmp_path/"failure_fallback"
    ae.run_residual_concentration_audit(prices,us,jp,"2019-01-01","2020-12-31",out)
    for name in ["variant_summary.csv","variant_summary.json","annual_returns.csv","monthly_returns.csv","drawdown_series.csv","turnover.csv","selected_tickers.csv","score_components.csv","data_quality.csv","selection_diff.csv","audit_metadata.json"]:
        assert (out/name).exists(), name
    meta=json.loads((out/"audit_metadata.json").read_text())
    assert meta["selection_diff_status"]=="failed"


def test_insufficient_history_warnings_are_aggregated():
    ae.reset_insufficient_history_warnings()
    idx=pd.bdate_range("2020-01-01",periods=10)
    prices=pd.DataFrame({"A":range(10)},index=idx)
    for _ in range(3):
        ae.score_universe(prices,["A"],idx[-1],min_history=252)
    summary=ae.get_insufficient_history_summary()
    assert summary.loc[0,"ticker"]=="A"
    assert summary.loc[0,"count"]==3
