import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import alpha_engine_backtest as aeb


def test_build_r100_default_variants_is_limited_to_eight():
    variants = aeb.build_r100_composite_variants()
    names = [v["name"] for v in variants]
    assert len(names) == 8
    assert names == list(aeb.R100_DEFAULT_VARIANTS)
    assert all(v["residual_weight"] == 1.0 for v in variants)
    assert {v["total_holdings"] for v in variants} == {12, 10, 8, 6}


def test_cli_recognizes_r100_audit():
    parser_help = subprocess.run(
        [sys.executable, "alpha_engine_backtest.py", "--help"],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "r100_composite_experiment" in parser_help.stdout
    assert "--only-variant" in parser_help.stdout
    assert "--no-detail-logs" in parser_help.stdout


def test_cache_missing_without_force_refresh_fails_fast(tmp_path):
    with pytest.raises(SystemExit, match="cache miss"):
        aeb.run_r100_composite_experiment_audit(
            prices=None,
            us=["AAA"],
            jp=["BBB.T"],
            start="2020-01-01",
            end="2020-12-31",
            output_dir=tmp_path,
            cache_dir=tmp_path / "missing_cache",
            source_dir=tmp_path / "missing_source",
            variants="Residual_100_N6_TTL90",
            force_refresh_cache=False,
        )


def test_future_boundary_classifies_end_of_sample_partial_cycle():
    rd = pd.DataFrame({
        "variant": ["Residual_100_N6_TTL90_Renew30_Composite"],
        "screen_date": ["2026-05-01"],
        "trade_date": ["2026-05-07"],
        "health_check_date": ["2026-06-18"],
    })
    dates = pd.date_range("2026-06-16", "2026-06-18", freq="D")
    artifacts = {"drawdown_series.csv": pd.DataFrame({"x": [0, 0, 0]}, index=dates)}
    review = aeb._future_boundary_review(rd, artifacts)
    row = review[review["check"].str.contains("health_check")].iloc[0]
    assert row["status"] == "end_of_sample_partial_cycle"


def test_demo_r100_outputs_and_resume(tmp_path):
    prices = aeb.demo_prices()
    us = [f"US{i}" for i in range(8)]
    jp = [f"JP{i}.T" for i in range(8)]
    out = tmp_path / "r100"
    summary = aeb.run_r100_composite_experiment_audit(
        prices=prices,
        us=us,
        jp=jp,
        start="2018-01-01",
        end="2020-12-31",
        output_dir=out,
        variants="Residual_100_N6_TTL90,Residual_100_N6_TTL90_Renew30_Composite",
    )
    assert "Residual_100_N6_TTL90" in summary.index
    assert "Residual_100_N6_TTL90_Renew30_Composite" in summary.index
    for name in [
        "r100_active_exposure_summary.csv",
        "r100_stress_year_2022.csv",
        "r100_concentration_risk_summary.csv",
        "r100_overdrive_recommendation.csv",
        "future_data_boundary_review.csv",
        "r100_experiment_report.md",
        "variant_summary_partial.csv",
        "cost_adjusted_summary_partial.csv",
        "completed_variants.csv",
        "audit_metadata.json",
    ]:
        assert (out / name).exists(), name
    meta = json.loads((out / "audit_metadata.json").read_text())
    assert meta["default_full_variant_recalculation_allowed"] is False
    assert meta["default_download_allowed"] is False
    assert meta["score_components_full_output"] is False
    completed_before = pd.read_csv(out / "completed_variants.csv")
    resumed = aeb.run_r100_composite_experiment_audit(
        prices=prices,
        us=us,
        jp=jp,
        start="2018-01-01",
        end="2020-12-31",
        output_dir=out,
        variants="Residual_100_N6_TTL90,Residual_100_N6_TTL90_Renew30_Composite",
        resume=True,
    )
    assert list(resumed.index) == [
        "Residual_100_N6_TTL90",
        "Residual_100_N6_TTL90_Renew30_Composite",
    ]
    completed_after = pd.read_csv(out / "completed_variants.csv")
    assert len(completed_after) == len(completed_before)


def test_r100_resume_rehydrates_completed_variants_into_final_outputs(tmp_path):
    prices = aeb.demo_prices()
    us = [f"US{i}" for i in range(8)]
    jp = [f"JP{i}.T" for i in range(8)]
    out = tmp_path / "r100_resume_full"
    n6_variants = "Residual_100_N6_TTL90,Residual_100_N6_TTL90_Renew30_Composite"

    aeb.run_r100_composite_experiment_audit(
        prices=prices,
        us=us,
        jp=jp,
        start="2018-01-01",
        end="2022-12-31",
        output_dir=out,
        variants=n6_variants,
    )

    summary = aeb.run_r100_composite_experiment_audit(
        prices=prices,
        us=us,
        jp=jp,
        start="2018-01-01",
        end="2022-12-31",
        output_dir=out,
        resume=True,
    )

    final_summary = pd.read_csv(out / "r100_variant_summary.csv", index_col=0)
    assert list(final_summary.index) == list(aeb.R100_DEFAULT_VARIANTS)
    assert list(summary.index) == list(aeb.R100_DEFAULT_VARIANTS)

    overdrive = pd.read_csv(out / "r100_overdrive_recommendation.csv")
    assert "Residual_100_N6_TTL90_Renew30_Composite" in set(overdrive["variant"])

    stress = pd.read_csv(out / "r100_stress_year_2022.csv")
    n6_stress = stress[stress["variant"].isin([
        "Residual_100_N6_TTL90",
        "Residual_100_N6_TTL90_Renew30_Composite",
    ])]
    assert len(n6_stress) == 2
    assert not n6_stress.drop(columns=["variant"]).isna().all(axis=1).any()
