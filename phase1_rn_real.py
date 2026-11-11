"""Research Phase 1: six-cell R/N comparison on newly downloaded market data.

Research-only, no broker, no orders, no production changes. An exploratory
comparison, NOT a reproduction of the June 18 audit or a performance forecast.
"""
from __future__ import annotations
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

import alpha_engine_backtest as ae


OUT = Path("phase1_real_results")
START = "2022-01-01"
END = "2026-11-11"
R_VALUES = (0, 60, 100)
N_VALUES = (6, 12)


def _digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {
        "status": "started",
        "run_started_utc": datetime.now(timezone.utc).isoformat(),
        "source_kind": "newly_downloaded_market_data",
        "source": "yfinance auto_adjust=True adjusted Close",
        "price_data_from_original_june_audit": False,
        "start": START,
        "end_requested": END,
        "configurations": [{"R": r, "N": n} for r in R_VALUES for n in N_VALUES],
        "limitations": [
            "Exploratory retrospective comparison, NOT independent out-of-sample evidence.",
            "Contemporary US/JP universe is backfilled historically; survivorship and membership bias.",
            "yfinance adjusted prices are not immutable point-in-time historical source records.",
            "No trade commissions, slippage, taxes, currency conversion or real broker lots.",
            "US and JP returns are weighted in their local currencies without explicit FX conversion.",
            "Daily return aggregation uses fixed initial inverse-vol weights, not passive drifting weights.",
            "Quarterly selection date and next trading day trade date; no TTL extension.",
            "The legacy metric field Sharpe is CAGR/annualized volatility, not conventional Sharpe.",
            "End dates are calendar controls; observed last trade date may differ.",
        ],
    }
    try:
        us_raw, jp_raw = ae.get_live_universe()
        us = list(dict.fromkeys(ae.normalize_yfinance_ticker(t) for t in us_raw))
        jp = list(dict.fromkeys(ae.normalize_yfinance_ticker(t) for t in jp_raw))
        meta["universe_us_requested"] = us
        meta["universe_jp_requested"] = jp
        meta["universe_us_count"] = len(us)
        meta["universe_jp_count"] = len(jp)
        if len(us) < 250 or len(jp) < 70:
            raise ValueError(
                f"Dynamic universe fallback or shortage: US={len(us)} JP={len(jp)};"
                " refuse a misleading small-universe result."
            )

        mode = ae.build_benchmark_modes()["broad_default"]
        benches = list(dict.fromkeys([*mode["US"], *mode["JP"]]))
        tickers = list(dict.fromkeys([*us, *jp, *benches]))
        prices = ae.download_live_prices(tickers, START, END, batch_size=75)
        if prices.empty:
            raise ValueError("All live adjusted-price downloads failed.")
        prices = prices.sort_index()
        if prices.index.has_duplicates or prices.columns.has_duplicates:
            raise ValueError("Duplicate dates/columns in downloaded prices.")
        # Keep raw NaN patterns and do not forward-fill the saved data.
        us_valid = [t for t in us if t in prices and prices[t].notna().sum() >= 252]
        jp_valid = [t for t in jp if t in prices and prices[t].notna().sum() >= 252]
        meta["universe_us_usable"] = us_valid
        meta["universe_jp_usable"] = jp_valid
        meta["universe_us_usable_count"] = len(us_valid)
        meta["universe_jp_usable_count"] = len(jp_valid)
        meta["downloaded_columns"] = len(prices.columns)
        meta["observed_start"] = str(prices.index.min().date())
        meta["observed_end"] = str(prices.index.max().date())
        meta["benchmark_used"] = {
            region: ae._resolve_benchmark(prices, region, mode)
            for region in ("US", "JP")
        }
        if len(us_valid) < 250 or len(jp_valid) < 70:
            raise ValueError(
                f"Usable history insufficient: US={len(us_valid)} JP={len(jp_valid)}."
            )
        if not all(meta["benchmark_used"].values()):
            raise ValueError("US/JP benchmarks unavailable.")
        if len(prices.loc[START:END]) < 252:
            raise ValueError("Less than 252 trading dates in comparison interval.")

        price_path = OUT / "prices_original.parquet"
        prices.to_parquet(price_path)
        pd.testing.assert_frame_equal(
            prices, pd.read_parquet(price_path), check_freq=False, check_exact=True
        )
        (OUT / "universe.json").write_text(
            json.dumps({"US": us, "JP": jp}, indent=2), encoding="utf-8"
        )
        meta["price_file_sha256"] = _digest(price_path)
        meta["universe_file_sha256"] = _digest(OUT / "universe.json")
        meta["numpy_version"] = np.__version__
        meta["pandas_version"] = pd.__version__
        meta["yfinance_version"] = yf.__version__
        meta["python_version"] = platform.python_version()

        results, holdings = [], []
        for r in R_VALUES:
            for n in N_VALUES:
                variant = {
                    "name": f"R{r}_N{n}",
                    "base_weight": 1.0 - r / 100.0,
                    "residual_weight": r / 100.0,
                    "total_holdings": n,
                    "us_holdings": n // 2,
                    "jp_holdings": n // 2,
                }
                returns, selected, _, turn = ae.run_residual_concentration_variant(
                    prices, us_valid, jp_valid, START, END, variant, mode,
                    method="simple",
                )
                if selected.empty:
                    raise ValueError(f"Empty results for {variant['name']}")
                if not np.isfinite(returns.to_numpy()).all():
                    raise ValueError(f"Nonfinite returns for {variant['name']}")
                m = ae.metrics(returns)
                daily_vol = returns.std(ddof=1)
                conventional_sharpe_rf0 = float(
                    returns.mean() / daily_vol * np.sqrt(252)
                ) if daily_vol > 0 else None
                results.append({
                    "variant": variant["name"], "R": r, "N": n,
                    **m,
                    "Conventional_Sharpe_RF0": conventional_sharpe_rf0,
                    "Mean_Turnover": float(turn.turnover.mean()) if len(turn) else None,
                    "Selected_Rows": len(selected),
                    "Unique_Screen_Dates": selected.screen_date.nunique(),
                })
                holdings.append(selected.assign(config_R=r, config_N=n))
                print(
                    f"{variant['name']} CAGR={m['CAGR']:.3%}"
                    f" MDD={m['Max_Drawdown']:.3%}", flush=True
                )

        df = pd.DataFrame(results).sort_values(["N", "R"])
        if len(df) != 6 or not all(df["Unique_Screen_Dates"].eq(df["Unique_Screen_Dates"].iloc[0])):
            raise ValueError("Comparison does not have six comparable rebalance schedules.")
        df.to_csv(OUT / "real_six_cell_comparison.csv", index=False)
        pd.concat(holdings, ignore_index=True).to_csv(
            OUT / "real_selected_tickers.csv", index=False
        )
        meta["status"] = "completed_exploratory"
        meta["result_count"] = len(df)
        print(df[["variant", "CAGR", "Annualized_Volatility",
                  "Max_Drawdown", "Conventional_Sharpe_RF0", "Selected_Rows"]].to_string(index=False))
    except Exception as exc:
        meta["status"] = "failed_no_claims"
        meta["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (OUT / "metadata.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
