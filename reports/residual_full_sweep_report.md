# Residual Momentum Full Sweep Audit Report

## 1. Executive Summary
Best non-baseline by Calmar was **Residual_70**. Baseline CAGR 9.79%, MaxDD -5.21%, Calmar 1.88; Residual_70 CAGR 11.39%, MaxDD -3.29%, Calmar 3.46. Concept classification: **Hybrid Core**. This is a research classification only and is not production adoption.

## 2. Baseline Reminder
TTL 90 days, quarterly / four trades per year, no Exit Protocol, no Regime Filter, no VCP, no stop loss, no discretionary cash retreat. Baseline is Residual_00 / base_weight=1.0 / residual_weight=0.0.

## CLI / Colab Commands
`python alpha_engine_backtest.py --audit residual_full_sweep --output-dir artifacts/residual_full_sweep`

`python alpha_engine_backtest.py --demo --audit residual_full_sweep --output-dir artifacts/residual_full_sweep`

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
| usable_start | 2019-01-01 |
| usable_end | 2021-12-31 |
| benchmark_broad_default | {'US': 'SPY', 'JP': '1306.T'} |
| benchmark_growth_adjusted_us | {'US': 'QQQ', 'JP': '1306.T'} |
| benchmark_index_alt_jp | {'US': 'SPY', 'JP': '^N225'} |
| benchmark_strict_available_default | {'US': 'SPY', 'JP': '1306.T'} |
| cache_used | False |
| cache_path | artifacts/residual_full_sweep/cache |
| cache_created_at |  |

Cache used: False / cache path: `artifacts/residual_full_sweep/cache`. Price cache files (`prices.pkl`, `benchmarks.pkl`) are generated at runtime and intentionally ignored by git; committed artifacts keep only text metadata such as `cache_metadata.json` and `universe.csv`. Missing downloads, insufficient history, current-constituent use, and survivorship bias can affect results. Demo artifacts are not live results.

## 4. Full Sweep Summary
| CAGR | Annualized_Volatility | Max_Drawdown | Sharpe | Sortino | Calmar | Turnover | Judgment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 0.0979 | 0.0544 | -0.0521 | 1.8000 | 2.9967 | 1.8771 | 0.1526 | baseline |
| Residual_05 | 0.0979 | 0.0544 | -0.0521 | 1.8000 | 2.9967 | 1.8771 | 0.1526 | no improvement |
| Residual_10 | 0.0979 | 0.0544 | -0.0521 | 1.8000 | 2.9967 | 1.8771 | 0.1526 | no improvement |
| Residual_15 | 0.0955 | 0.0544 | -0.0521 | 1.7551 | 2.9389 | 1.8324 | 0.1587 | worse |
| Residual_20 | 0.0927 | 0.0547 | -0.0521 | 1.6953 | 2.8270 | 1.7791 | 0.1585 | worse |
| Residual_25 | 0.0908 | 0.0546 | -0.0521 | 1.6613 | 2.7730 | 1.7416 | 0.1584 | worse |
| Residual_30 | 0.0892 | 0.0548 | -0.0521 | 1.6289 | 2.7429 | 1.7118 | 0.1653 | worse |
| Residual_35 | 0.0855 | 0.0541 | -0.0521 | 1.5804 | 2.6436 | 1.6406 | 0.1924 | worse |
| Residual_40 | 0.0855 | 0.0541 | -0.0521 | 1.5804 | 2.6436 | 1.6406 | 0.1924 | worse |
| Residual_45 | 0.0866 | 0.0544 | -0.0442 | 1.5911 | 2.6591 | 1.9609 | 0.1796 | clear improvement |
| Residual_50 | 0.0866 | 0.0544 | -0.0442 | 1.5911 | 2.6591 | 1.9609 | 0.1796 | clear improvement |
| Residual_55 | 0.0979 | 0.0538 | -0.0400 | 1.8204 | 3.0471 | 2.4474 | 0.1800 | clear improvement |
| Residual_60 | 0.0979 | 0.0538 | -0.0400 | 1.8204 | 3.0471 | 2.4474 | 0.1800 | clear improvement |
| Residual_65 | 0.1095 | 0.0541 | -0.0400 | 2.0234 | 3.4121 | 2.7363 | 0.1734 | clear improvement |
| Residual_70 | 0.1139 | 0.0543 | -0.0329 | 2.0964 | 3.5396 | 3.4607 | 0.1665 | clear improvement |
| Residual_75 | 0.1139 | 0.0543 | -0.0329 | 2.0964 | 3.5396 | 3.4607 | 0.1665 | clear improvement |
| Residual_80 | 0.1035 | 0.0544 | -0.0362 | 1.9013 | 3.0981 | 2.8561 | 0.1810 | clear improvement |
| Residual_85 | 0.1035 | 0.0544 | -0.0362 | 1.9013 | 3.0981 | 2.8561 | 0.1810 | clear improvement |
| Residual_90 | 0.1001 | 0.0546 | -0.0362 | 1.8327 | 3.0307 | 2.7622 | 0.1880 | clear improvement |
| Residual_95 | 0.1001 | 0.0546 | -0.0362 | 1.8327 | 3.0307 | 2.7622 | 0.1880 | clear improvement |
| Residual_100 | 0.1001 | 0.0546 | -0.0362 | 1.8327 | 3.0307 | 2.7622 | 0.1880 | clear improvement |

The sweep covers Residual_00 through Residual_100 in 5% increments. Inspect whether improvement continues above 30%, where metrics peak, and whether pure residual remains viable.

## 5. Peak Ratio Diagnostics
| variant | ratio | value |
| --- | --- | --- |
| best_cagr_ratio | Residual_70 | 70 | 0.1139 |
| best_sharpe_ratio | Residual_70 | 70 | 2.0964 |
| best_sortino_ratio | Residual_70 | 70 | 3.5396 |
| best_calmar_ratio | Residual_70 | 70 | 3.4607 |
| lowest_maxdd_ratio | Residual_70 | 70 | -0.0329 |
| lowest_volatility_ratio | Residual_55 | 55 | 0.0538 |
| lowest_turnover_ratio_among_improving_variants | Residual_70 | 70 | 0.1665 |

## 6. Plateau / Band Analysis
| ratio_range | count |
| --- | --- |
| cagr_improved_range | 55-100% | 10 |
| maxdd_improved_range | 15%; 25-100% | 17 |
| calmar_improved_range | 45-100% | 12 |
| sharpe_improved_range | 55-100% | 10 |
| sortino_improved_range | 55-100% | 10 |
| cagr_maxdd_calmar_simultaneous_improvement_range | 55-100% | 10 |
| best_plateau_near_best_calmar | 65-85% | 5 |

## 7. Concept Classification
Classification: **Hybrid Core**. Auxiliary Lens is 5-25%, Hybrid Core is 30-70%, Residual Dominant is 75-95%, Pure Residual is 100%/near-best. This is research framing only.

## 8. Benchmark Sensitivity
| status | CAGR | Annualized_Volatility | Max_Drawdown | Sharpe | Calmar | Sortino | Total_Return | Best_Year | Worst_Year | Monthly_Win_Rate | Turnover |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ('broad_default', 'Baseline') | ok | 0.0979 | 0.0544 | -0.0521 | 1.8000 | 1.8771 | 2.9967 | 0.3370 | 0.1053 | 0.0962 | 0.5833 | 0.1526 |
| ('broad_default', 'Residual_25') | ok | 0.0908 | 0.0546 | -0.0521 | 1.6613 | 1.7416 | 2.7730 | 0.3104 | 0.0980 | 0.0876 | 0.5833 | 0.1584 |
| ('broad_default', 'Residual_30') | ok | 0.0892 | 0.0548 | -0.0521 | 1.6289 | 1.7118 | 2.7429 | 0.3046 | 0.0980 | 0.0828 | 0.5556 | 0.1653 |
| ('broad_default', 'Residual_50') | ok | 0.0866 | 0.0544 | -0.0442 | 1.5911 | 1.9609 | 2.6591 | 0.2948 | 0.0974 | 0.0828 | 0.5833 | 0.1796 |
| ('broad_default', 'Residual_70') | ok | 0.1139 | 0.0543 | -0.0329 | 2.0964 | 3.4607 | 3.5396 | 0.3989 | 0.1692 | 0.0909 | 0.6389 | 0.1665 |
| ('broad_default', 'Residual_75') | ok | 0.1139 | 0.0543 | -0.0329 | 2.0964 | 3.4607 | 3.5396 | 0.3989 | 0.1692 | 0.0909 | 0.6389 | 0.1665 |
| ('broad_default', 'Residual_100') | ok | 0.1001 | 0.0546 | -0.0362 | 1.8327 | 2.7622 | 3.0307 | 0.3455 | 0.1692 | 0.0671 | 0.6111 | 0.1880 |
| ('growth_adjusted_us', 'Baseline') | ok | 0.0979 | 0.0544 | -0.0521 | 1.8000 | 1.8771 | 2.9967 | 0.3370 | 0.1053 | 0.0962 | 0.5833 | 0.1526 |
| ('growth_adjusted_us', 'Residual_25') | ok | 0.0908 | 0.0546 | -0.0521 | 1.6613 | 1.7416 | 2.7730 | 0.3104 | 0.0980 | 0.0876 | 0.5833 | 0.1584 |
| ('growth_adjusted_us', 'Residual_30') | ok | 0.0892 | 0.0548 | -0.0521 | 1.6289 | 1.7118 | 2.7429 | 0.3046 | 0.0980 | 0.0828 | 0.5556 | 0.1653 |
| ('growth_adjusted_us', 'Residual_50') | ok | 0.0866 | 0.0544 | -0.0442 | 1.5911 | 1.9609 | 2.6591 | 0.2948 | 0.0974 | 0.0828 | 0.5833 | 0.1796 |
| ('growth_adjusted_us', 'Residual_70') | ok | 0.1139 | 0.0543 | -0.0329 | 2.0964 | 3.4607 | 3.5396 | 0.3989 | 0.1692 | 0.0909 | 0.6389 | 0.1665 |
| ('growth_adjusted_us', 'Residual_75') | ok | 0.1139 | 0.0543 | -0.0329 | 2.0964 | 3.4607 | 3.5396 | 0.3989 | 0.1692 | 0.0909 | 0.6389 | 0.1665 |
| ('growth_adjusted_us', 'Residual_100') | ok | 0.1001 | 0.0546 | -0.0362 | 1.8327 | 2.7622 | 3.0307 | 0.3455 | 0.1692 | 0.0671 | 0.6111 | 0.1880 |
| ('index_alt_jp', 'Baseline') | ok | 0.0979 | 0.0544 | -0.0521 | 1.8000 | 1.8771 | 2.9967 | 0.3370 | 0.1053 | 0.0962 | 0.5833 | 0.1526 |
| ('index_alt_jp', 'Residual_25') | ok | 0.0908 | 0.0546 | -0.0521 | 1.6613 | 1.7416 | 2.7730 | 0.3104 | 0.0980 | 0.0876 | 0.5833 | 0.1584 |
| ('index_alt_jp', 'Residual_30') | ok | 0.0892 | 0.0548 | -0.0521 | 1.6289 | 1.7118 | 2.7429 | 0.3046 | 0.0980 | 0.0828 | 0.5556 | 0.1653 |
| ('index_alt_jp', 'Residual_50') | ok | 0.0866 | 0.0544 | -0.0442 | 1.5911 | 1.9609 | 2.6591 | 0.2948 | 0.0974 | 0.0828 | 0.5833 | 0.1796 |
| ('index_alt_jp', 'Residual_70') | ok | 0.1139 | 0.0543 | -0.0329 | 2.0964 | 3.4607 | 3.5396 | 0.3989 | 0.1692 | 0.0909 | 0.6389 | 0.1665 |
| ('index_alt_jp', 'Residual_75') | ok | 0.1139 | 0.0543 | -0.0329 | 2.0964 | 3.4607 | 3.5396 | 0.3989 | 0.1692 | 0.0909 | 0.6389 | 0.1665 |
| ('index_alt_jp', 'Residual_100') | ok | 0.1001 | 0.0546 | -0.0362 | 1.8327 | 2.7622 | 3.0307 | 0.3455 | 0.1692 | 0.0671 | 0.6111 | 0.1880 |
| ('strict_available_default', 'Baseline') | ok | 0.0979 | 0.0544 | -0.0521 | 1.8000 | 1.8771 | 2.9967 | 0.3370 | 0.1053 | 0.0962 | 0.5833 | 0.1526 |
| ('strict_available_default', 'Residual_25') | ok | 0.0908 | 0.0546 | -0.0521 | 1.6613 | 1.7416 | 2.7730 | 0.3104 | 0.0980 | 0.0876 | 0.5833 | 0.1584 |
| ('strict_available_default', 'Residual_30') | ok | 0.0892 | 0.0548 | -0.0521 | 1.6289 | 1.7118 | 2.7429 | 0.3046 | 0.0980 | 0.0828 | 0.5556 | 0.1653 |
| ('strict_available_default', 'Residual_50') | ok | 0.0866 | 0.0544 | -0.0442 | 1.5911 | 1.9609 | 2.6591 | 0.2948 | 0.0974 | 0.0828 | 0.5833 | 0.1796 |
| ('strict_available_default', 'Residual_70') | ok | 0.1139 | 0.0543 | -0.0329 | 2.0964 | 3.4607 | 3.5396 | 0.3989 | 0.1692 | 0.0909 | 0.6389 | 0.1665 |
| ('strict_available_default', 'Residual_75') | ok | 0.1139 | 0.0543 | -0.0329 | 2.0964 | 3.4607 | 3.5396 | 0.3989 | 0.1692 | 0.0909 | 0.6389 | 0.1665 |
| ('strict_available_default', 'Residual_100') | ok | 0.1001 | 0.0546 | -0.0362 | 1.8327 | 2.7622 | 3.0307 | 0.3455 | 0.1692 | 0.0671 | 0.6111 | 0.1880 |

## 9. Year / Period Review
Use `annual_returns.csv`, `monthly_returns.csv`, and `drawdown_series.csv` for 2020, 2022, 2023, 2024, 2025, and 2026 YTD if present. This is post-analysis only and never changes trades.

## 10. Selection Difference Review
`selection_diff.csv` compares Baseline against Residual_25, Residual_30, Residual_50, Residual_75, Residual_100, Residual_20, and the best ratio where available. Review added/removed tickers, base/residual/final scores, US/JP mix, and whether residual is removing beta followers or merely adding volatility.

## 11. Risk Review
Focus on MaxDD, Worst Year, Worst Month/monthly returns, drawdowns, turnover, 2022 behavior, opportunity cost in 2023-2024, and Residual_100 risk. Calmar is emphasized over Sharpe because the question is return versus maximum loss.

## 12. Recommendation
Conservative candidate: first stable plateau ratio that improves Calmar/MaxDD. Balanced candidate: best Calmar ratio if nearby ratios also improve. Aggressive candidate: highest residual ratio that remains in the stable plateau. Do **not** productionize without year-by-year, benchmark, turnover, and selection-difference review.

## 13. Safety Notes
This is not investment advice. Historical tests do not guarantee future returns. yfinance/Wikipedia/free data can have missing values, delays, adjusted-price issues, survivorship bias, and historical constituent bias. The audit remains within individual-investor free-data constraints.
