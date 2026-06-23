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
        start="2020-01-01",
        end="2022-12-31",
        output_dir=out,
        variants=n6_variants,
    )

    stale_stress = pd.read_csv(out / "r100_stress_year_2022.csv")
    n6_mask = stale_stress["variant"].isin(n6_variants.split(","))
    stale_stress.loc[n6_mask, ["return_2022", "max_drawdown_2022", "worst_month_2022", "average_active_exposure_2022"]] = 0.0
    stale_stress.loc[n6_mask, "stress_observation_status"] = "neutral_filled_no_observations"
    stale_stress.to_csv(out / "r100_stress_year_2022.csv", index=False)

    summary = aeb.run_r100_composite_experiment_audit(
        prices=prices,
        us=us,
        jp=jp,
        start="2020-01-01",
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
    assert list(stress["variant"]) == list(aeb.R100_DEFAULT_VARIANTS)
    stress_metric_columns = [
        "return_2022",
        "max_drawdown_2022",
        "worst_month_2022",
        "monthly_win_rate_2022",
        "average_active_exposure_2022",
    ]
    assert not stress[stress_metric_columns].isna().any().any()
    n6_stress = stress[stress["variant"].isin(n6_variants.split(","))]
    assert set(n6_stress["stress_observation_status"]) == {"observed"}
    assert not (n6_stress[["return_2022", "max_drawdown_2022", "worst_month_2022", "average_active_exposure_2022"]] == 0.0).all(axis=1).any()
    assert not final_summary.loc[n6_variants.split(","), "Worst_Month"].isna().any()

    composite_overdrive = overdrive[
        overdrive["variant"].isin([v for v in aeb.R100_DEFAULT_VARIANTS if v.endswith("Composite")])
    ]
    assert len(composite_overdrive) == 4
    assert not composite_overdrive["max_drawdown_2022"].isna().any()
    n6_comp_overdrive = composite_overdrive[composite_overdrive["variant"] == "Residual_100_N6_TTL90_Renew30_Composite"].iloc[0]
    assert n6_comp_overdrive["stress_observation_status"] == "observed"
    assert n6_comp_overdrive["max_drawdown_2022"] != 0.0


def test_r100_frozen_cache_resume_rehydrates_n6_outputs(tmp_path):
    prices = aeb.demo_prices()
    us = [f"US{i}" for i in range(8)]
    jp = [f"JP{i}.T" for i in range(8)]
    out = tmp_path / "r100_frozen_resume"
    cache = tmp_path / "frozen_cache"
    cache.mkdir()
    start = "2018-01-01"
    frozen_end = "2022-12-30"
    requested_end = "2023-01-31"
    n6_variants = "Residual_100_N6_TTL90,Residual_100_N6_TTL90_Renew30_Composite"
    modes = aeb.build_benchmark_modes()
    bench_tickers = sorted({x for mode in modes.values() for vals in mode.values() for x in vals})
    requested = [*us, *jp]
    frozen_prices = prices.loc[:frozen_end]
    frozen_prices[[c for c in requested if c in frozen_prices.columns]].to_pickle(cache / "prices.pkl")
    frozen_prices[[c for c in bench_tickers if c in frozen_prices.columns]].to_pickle(cache / "benchmarks.pkl")
    (cache / "cache_metadata.json").write_text(
        json.dumps(aeb._cache_metadata(start, frozen_end, requested, bench_tickers), indent=2),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="cache miss"):
        aeb.run_r100_composite_experiment_audit(
            prices=None,
            us=us,
            jp=jp,
            start=start,
            end=requested_end,
            output_dir=out,
            cache_dir=cache,
            source_dir=tmp_path / "missing_source",
            variants=n6_variants,
        )

    aeb.run_r100_composite_experiment_audit(
        prices=None,
        us=us,
        jp=jp,
        start=start,
        end=requested_end,
        output_dir=out,
        cache_dir=cache,
        source_dir=tmp_path / "missing_source",
        variants=n6_variants,
        allow_frozen_cache=True,
    )

    stale_stress = pd.read_csv(out / "r100_stress_year_2022.csv")
    stale_stress.loc[
        stale_stress["variant"] == "Residual_100_N6_TTL90_Renew30_Composite",
        ["return_2022", "max_drawdown_2022", "worst_month_2022", "average_active_exposure_2022"],
    ] = 0.0
    stale_stress.loc[
        stale_stress["variant"] == "Residual_100_N6_TTL90_Renew30_Composite",
        "stress_observation_status",
    ] = "neutral_filled_no_observations"
    stale_stress.to_csv(out / "r100_stress_year_2022.csv", index=False)
    partial = pd.read_csv(out / "variant_summary_partial.csv")
    partial.loc[partial["Variant"] == "Residual_100_N6_TTL90_Renew30_Composite", "Worst_Month"] = pd.NA
    partial.to_csv(out / "variant_summary_partial.csv", index=False)

    aeb.run_r100_composite_experiment_audit(
        prices=None,
        us=us,
        jp=jp,
        start=start,
        end=requested_end,
        output_dir=out,
        cache_dir=cache,
        source_dir=tmp_path / "missing_source",
        variants=n6_variants,
        resume=True,
        allow_frozen_cache=True,
    )

    meta = json.loads((out / "audit_metadata.json").read_text())
    assert meta["frozen_cache_used"] is True
    assert meta["frozen_cache_end"] == frozen_end
    assert meta["reproducibility_mode"] is True
    assert meta["note"] == aeb.FROZEN_CACHE_REPRO_NOTE

    final_summary = pd.read_csv(out / "r100_variant_summary.csv", index_col=0)
    stress = pd.read_csv(out / "r100_stress_year_2022.csv").set_index("variant")
    composite = "Residual_100_N6_TTL90_Renew30_Composite"
    assert stress.loc[composite, "stress_observation_status"] == "observed"
    assert stress.loc[composite, "max_drawdown_2022"] != 0.0
    assert not pd.isna(final_summary.loc[composite, "Worst_Month"])
