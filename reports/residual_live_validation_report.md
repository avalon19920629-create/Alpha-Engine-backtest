# Residual Momentum Live Validation Report

## 1. Executive Summary
Best non-baseline by Calmar was **Residual_20**. Baseline CAGR 10.57%, MaxDD -5.89%, Calmar 1.79; Residual_20 CAGR 10.48%, MaxDD -5.21%, Calmar 2.01. Residual_20 / Residual_25 are candidates only if their rows improve Calmar/MaxDD without unacceptable CAGR or turnover cost. Improved Calmar variants in the main 15-30% zone: Residual_20, Residual_25, Residual_30. This is not a production adoption decision.

## 2. Baseline Reminder
TTL 90 days, quarterly / four trades per year, no Exit Protocol, no Regime Filter, no VCP, no stop loss, no discretionary cash retreat. Baseline remains current Alpha Engine base score only.

## CLI / Colab Command
`python alpha_engine_backtest.py --audit residual_live_validation --output-dir artifacts/residual_live_validation`

Colab dependencies: `python -m pip install -r requirements.txt` (includes pandas/numpy/yfinance if declared; otherwise install `yfinance`).

## 3. Data Quality Summary
| value |
| --- |
| requested_tickers | 22 |
| successfully_downloaded_tickers | 21 |
| failed_tickers | 0 |
| insufficient_history_tickers | 0 |
| excluded_tickers | 1 |
| final_usable_universe_size | 21 |
| us_usable_count | 8 |
| jp_usable_count | 8 |
| data_start | 2018-01-01 |
| data_end | 2025-12-31 |
| usable_start | 2015-01-01 |
| usable_end | 2026-06-16 |
| benchmark_broad_default | {'US': 'SPY', 'JP': '1306.T'} |
| benchmark_growth_adjusted_us | {'US': 'QQQ', 'JP': '1306.T'} |
| benchmark_index_alt_jp | {'US': 'SPY', 'JP': '^N225'} |
| benchmark_strict_available_default | {'US': 'SPY', 'JP': '1306.T'} |

Missing downloads and insufficient history can reduce breadth; current constituents used historically can introduce survivorship / historical constituent bias.

## 4. Variant Summary
| CAGR | Annualized_Volatility | Max_Drawdown | Sharpe | Sortino | Calmar | Turnover | Judgment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 0.1057 | 0.0517 | -0.0589 | 2.0450 | 3.3256 | 1.7938 | 0.1235 | baseline |
| Residual_10 | 0.1057 | 0.0517 | -0.0589 | 2.0450 | 3.3256 | 1.7938 | 0.1235 | no improvement |
| Residual_15 | 0.1048 | 0.0517 | -0.0589 | 2.0270 | 3.3046 | 1.7788 | 0.1258 | worse |
| Residual_20 | 0.1048 | 0.0517 | -0.0521 | 2.0276 | 3.3013 | 2.0108 | 0.1258 | clear improvement |
| Residual_25 | 0.1023 | 0.0515 | -0.0521 | 1.9854 | 3.2196 | 1.9620 | 0.1283 | clear improvement |
| Residual_30 | 0.0987 | 0.0516 | -0.0521 | 1.9115 | 3.1075 | 1.8926 | 0.1313 | clear improvement |

## 5. Benchmark Sensitivity
| status | CAGR | Annualized_Volatility | Max_Drawdown | Sharpe | Calmar | Sortino | Total_Return | Best_Year | Worst_Year | Monthly_Win_Rate | Turnover |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ('broad_default', 'Residual_15') | ok | 0.1048 | 0.0517 | -0.0589 | 2.0270 | 1.7788 | 3.3046 | 1.2838 | 0.2358 | 0.0000 | 0.6250 | 0.1258 |
| ('broad_default', 'Residual_20') | ok | 0.1048 | 0.0517 | -0.0521 | 2.0276 | 2.0108 | 3.3013 | 1.2841 | 0.2358 | 0.0000 | 0.6250 | 0.1258 |
| ('broad_default', 'Residual_25') | ok | 0.1023 | 0.0515 | -0.0521 | 1.9854 | 1.9620 | 3.2196 | 1.2408 | 0.2358 | 0.0000 | 0.6250 | 0.1283 |
| ('broad_default', 'Residual_30') | ok | 0.0987 | 0.0516 | -0.0521 | 1.9115 | 1.8926 | 3.1075 | 1.1807 | 0.2320 | 0.0000 | 0.6042 | 0.1313 |
| ('growth_adjusted_us', 'Residual_15') | ok | 0.1048 | 0.0517 | -0.0589 | 2.0270 | 1.7788 | 3.3046 | 1.2838 | 0.2358 | 0.0000 | 0.6250 | 0.1258 |
| ('growth_adjusted_us', 'Residual_20') | ok | 0.1048 | 0.0517 | -0.0521 | 2.0276 | 2.0108 | 3.3013 | 1.2841 | 0.2358 | 0.0000 | 0.6250 | 0.1258 |
| ('growth_adjusted_us', 'Residual_25') | ok | 0.1023 | 0.0515 | -0.0521 | 1.9854 | 1.9620 | 3.2196 | 1.2408 | 0.2358 | 0.0000 | 0.6250 | 0.1283 |
| ('growth_adjusted_us', 'Residual_30') | ok | 0.0987 | 0.0516 | -0.0521 | 1.9115 | 1.8926 | 3.1075 | 1.1807 | 0.2320 | 0.0000 | 0.6042 | 0.1313 |
| ('index_alt_jp', 'Residual_15') | ok | 0.1048 | 0.0517 | -0.0589 | 2.0270 | 1.7788 | 3.3046 | 1.2838 | 0.2358 | 0.0000 | 0.6250 | 0.1258 |
| ('index_alt_jp', 'Residual_20') | ok | 0.1048 | 0.0517 | -0.0521 | 2.0276 | 2.0108 | 3.3013 | 1.2841 | 0.2358 | 0.0000 | 0.6250 | 0.1258 |
| ('index_alt_jp', 'Residual_25') | ok | 0.1023 | 0.0515 | -0.0521 | 1.9854 | 1.9620 | 3.2196 | 1.2408 | 0.2358 | 0.0000 | 0.6250 | 0.1283 |
| ('index_alt_jp', 'Residual_30') | ok | 0.0987 | 0.0516 | -0.0521 | 1.9115 | 1.8926 | 3.1075 | 1.1807 | 0.2320 | 0.0000 | 0.6042 | 0.1313 |
| ('strict_available_default', 'Residual_15') | ok | 0.1048 | 0.0517 | -0.0589 | 2.0270 | 1.7788 | 3.3046 | 1.2838 | 0.2358 | 0.0000 | 0.6250 | 0.1258 |
| ('strict_available_default', 'Residual_20') | ok | 0.1048 | 0.0517 | -0.0521 | 2.0276 | 2.0108 | 3.3013 | 1.2841 | 0.2358 | 0.0000 | 0.6250 | 0.1258 |
| ('strict_available_default', 'Residual_25') | ok | 0.1023 | 0.0515 | -0.0521 | 1.9854 | 1.9620 | 3.2196 | 1.2408 | 0.2358 | 0.0000 | 0.6250 | 0.1283 |
| ('strict_available_default', 'Residual_30') | ok | 0.0987 | 0.0516 | -0.0521 | 1.9115 | 1.8926 | 3.1075 | 1.1807 | 0.2320 | 0.0000 | 0.6042 | 0.1313 |

## 6. Year / Period Review
Review `annual_returns.csv`, `monthly_returns.csv`, and `drawdown_series.csv` for 2020, 2022, 2023, 2024, 2025, and 2026 YTD when present. This is post-analysis only and never triggers cash retreat or trade suspension.

## 7. Selection Difference Review
`selection_diff.csv` compares Baseline with Residual_10/15/20/25/30 by rebalance date and records added/removed tickers plus score context where available. Use it to assess whether residual removes market beta followers and whether US/JP concentration changes.

## 8. Risk Review
Calmar and MaxDD are emphasized alongside CAGR, Sharpe, Sortino, worst year/month, drawdown series, and turnover. 2022 behavior and opportunity cost in 2023-2024 should be inspected before any vNext decision.

## 9. Recommendation
Continue Alpha Engine vNext research only if Residual_15-25 shows stable Calmar/MaxDD improvement across benchmark modes. Do not merge into production from this validation alone; keep Baseline untouched until broader live/free-data and bias checks are complete.

## 10. Safety Notes
This validation is research, not investment advice. Past data does not guarantee future returns. yfinance / Wikipedia / free data may have missing values, delays, adjusted-price issues, current-constituent bias, and survivorship bias. The audit is positioned as a realistic individual-investor free-data validation.
