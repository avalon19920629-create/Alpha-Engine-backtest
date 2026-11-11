"""Phase 1 R×N six-cell smoke comparison on *deterministic synthetic prices*.

Not historical evidence or a model-selection result. This verifies that the
repository's six settings are executable with exactly identical input data,
dates, signal implementation, and inverse-volatility allocation convention.
"""
from pathlib import Path
import json
import pandas as pd

import alpha_engine_backtest as engine


def main():
    output = Path("phase1_artifacts")
    output.mkdir(exist_ok=True)
    prices = engine.demo_prices()
    us = [f"US{i}" for i in range(8)]
    jp = [f"JP{i}.T" for i in range(8)]
    start, end = "2019-01-01", "2021-12-31"
    modes = engine.build_benchmark_modes()
    rows, picks = [], []
    for ratio in (0, 60, 100):
        for n in (6, 12):
            variant = {
                "name": f"R{ratio}_N{n}",
                "base_weight": 1 - ratio / 100,
                "residual_weight": ratio / 100,
                "total_holdings": n,
                "us_holdings": n // 2,
                "jp_holdings": n // 2,
            }
            returns, selected, _, turn = engine.run_residual_concentration_variant(
                prices, us, jp, start, end, variant,
                modes["broad_default"], method="simple"
            )
            m = engine.metrics(returns)
            rows.append({
                "variant": variant["name"],
                "R": ratio,
                "N": n,
                **m,
                "turnover": float(turn["turnover"].mean()) if len(turn) else None,
                "selected_rows": len(selected),
            })
            if not selected.empty:
                picks.append(selected.assign(config_R=ratio, config_N=n))
    out = pd.DataFrame(rows).sort_values(["N", "R"]).reset_index(drop=True)
    assert len(out) == 6
    assert out["CAGR"].notna().all() and out["Max_Drawdown"].notna().all()
    out.to_csv(output / "synthetic_six_cell_comparison.csv", index=False)
    pd.concat(picks, ignore_index=True).to_csv(
        output / "synthetic_selected_tickers.csv", index=False
    )
    (output / "metadata.json").write_text(json.dumps({
        "data_kind": "synthetic_demo_only",
        "is_investment_evidence": False,
        "commit": "research/phase1-ttl-causal-fix-20261111",
        "data": "engine.demo_prices()",
        "start": start, "end": end,
        "signal": "simple residual; present engine logic",
        "benchmark_mode": "broad_default",
        "source": "alpha_engine_backtest.run_residual_concentration_variant",
        "exit_or_ttl_renewal": False,
        "R": [0, 60, 100], "N": [6, 12],
    }, indent=2), encoding="utf-8")
    print(out[["variant", "CAGR", "Annualized_Volatility",
               "Max_Drawdown", "Sharpe", "Calmar", "selected_rows"]].to_string(index=False))


if __name__ == "__main__":
    main()
