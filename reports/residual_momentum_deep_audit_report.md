# Residual Momentum Deep Audit Report

## Executive Summary
Best non-baseline by Calmar was **Residual_20**. Baseline CAGR 10.57%, MaxDD -5.89%, Calmar 1.79; Residual_20 CAGR 10.48%, MaxDD -5.21%, Calmar 2.01. Residual_20 is evaluated as one point inside the Residual_10-25 zone, not as a production rule. Production use still requires live/free-data validation.

## Baseline Reminder
TTL 90 days, quarterly / four trades per year, no Exit Protocol, no Regime Filter, no VCP, no stop loss, no discretionary cash retreat.

## CLI
`python alpha_engine_backtest.py --demo --audit residual_momentum_deep --output-dir artifacts/residual_momentum_deep`

## Residual Weight Sweep Summary
| Variant | CAGR | Annualized_Volatility | Max_Drawdown | Sharpe | Sortino | Calmar | Turnover | Judgment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 0.1057 | 0.0517 | -0.0589 | 2.0450 | 3.3256 | 1.7938 | 0.1235 | baseline |
| Residual_05 | 0.1057 | 0.0517 | -0.0589 | 2.0450 | 3.3256 | 1.7938 | 0.1235 | no improvement |
| Residual_10 | 0.1057 | 0.0517 | -0.0589 | 2.0450 | 3.3256 | 1.7938 | 0.1235 | no improvement |
| Residual_15 | 0.1048 | 0.0517 | -0.0589 | 2.0270 | 3.3046 | 1.7788 | 0.1258 | worse |
| Residual_20 | 0.1048 | 0.0517 | -0.0521 | 2.0276 | 3.3013 | 2.0108 | 0.1258 | clear improvement |
| Residual_25 | 0.1023 | 0.0515 | -0.0521 | 1.9854 | 3.2196 | 1.9620 | 0.1283 | clear improvement |
| Residual_30 | 0.0987 | 0.0516 | -0.0521 | 1.9115 | 3.1075 | 1.8926 | 0.1313 | clear improvement |
| Residual_40 | 0.0942 | 0.0514 | -0.0521 | 1.8319 | 2.9735 | 1.8068 | 0.1526 | clear improvement |

Improvement zone Residual_10-25 average Calmar 1.89 versus Baseline 1.79; evaluate stability across the CSVs rather than selecting a single lucky point.

## Residual Method Comparison
Both simple benchmark-adjusted residual and beta-adjusted residual are implemented. See `artifacts/residual_momentum_deep/residual_method_comparison.csv`; beta-adjusted falls back safely when benchmark variance/history is insufficient.

## Benchmark Sensitivity
Modes tested: broad_default, growth_adjusted_us, index_alt_jp, strict_available_default. See `benchmark_sensitivity.csv`; this checks whether Residual_15/20/25 depend excessively on SPY/QQQ or TOPIX/Nikkei choices.

## Year / Period Review
Annual returns include 2020, 2022, 2023, 2024 and 2025 where present in `annual_returns.csv`. This is post-analysis only; no regime trading filter or cash retreat is introduced.

## Selection Difference Review
`selection_diff.csv` compares Baseline against Residual_10/15/20/25/30 and other residual weights by rebalance date. `score_components.csv` lists base_score, residual_score, final_score, stock/benchmark returns, benchmark_used, method, selected_flag, and weights.

## Risk Review
Use MaxDD, Worst Year, monthly returns, drawdown_series, turnover, and 2022 annual returns. Calmar is emphasized because the research question is return efficiency versus maximum loss.

## Recommendation
Continue researching Residual Momentum only if the Residual_10-25 zone improves Calmar/MaxDD without unacceptable CAGR or turnover cost. Do not change production Alpha Engine from this demo/free-data audit alone; live/free-data validation is required.

## Safety Notes
This is not investment advice. Historical yfinance/Wikipedia/free-data tests do not guarantee future returns. Free data can contain missing values, delays, adjusted-price issues, index membership bias, and survivorship bias; this is framed as an individual-investor free-data audit.
