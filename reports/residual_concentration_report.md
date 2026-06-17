# Residual Core Portfolio Concentration Audit Report

## 1. Executive Summary
Best Calmar variant was **Baseline_N40**. Baseline_N12 CAGR 9.60%, MaxDD -3.58%, Calmar 2.68; Baseline_N40 CAGR 10.26%, MaxDD -2.57%, Calmar 3.99. This audit fixes Residual Momentum as the tested core signal and varies only residual ratio and balanced US/JP portfolio size. N30/N40 are high-diversification reference points, not practical production recommendations. If improvements are not robust across neighboring variants, production adoption should be deferred.

## 2. Baseline Reminder
TTL 90 days, quarterly/four trades per year, no Exit Protocol, no Regime Filter, no VCP Proxy, no Sector Residual, no Downside penalty, no Correlation penalty. Baseline_N12 is base_weight=1.0, residual_weight=0.0, US 6 / JP 6.

## CLI
`python alpha_engine_backtest.py --audit residual_concentration --output-dir artifacts/residual_concentration`

`python alpha_engine_backtest.py --demo --audit residual_concentration --output-dir artifacts/residual_concentration`

## 3. Why Portfolio Size Matters
The current 12-stock structure is a practical balance, not a proof of optimality. This audit compares N4/N6 concentration limits, N8/N10 concentration candidates, current N12, N16/N20/N24 practical diversification, and N30/N40 high-diversification references to study Residual signal purity versus single-name accident risk.

## 4. Data Quality Summary
| value |
| --- |
| requested_tickers | 16 |
| successfully_downloaded_tickers | 16 |
| failed_tickers | 0 |
| insufficient_history_tickers | 0 |
| excluded_tickers | 0 |
| final_usable_universe_size | 16 |
| us_usable_count | 8 |
| jp_usable_count | 8 |
| data_start | 2018-01-01 |
| data_end | 2025-12-31 |
| usable_start | 2019-01-01 |
| usable_end | 2020-12-31 |
| benchmark_broad_default | {'US': 'SPY', 'JP': '1306.T'} |
| benchmark_growth_adjusted_us | {'US': 'QQQ', 'JP': '1306.T'} |
| benchmark_index_alt_jp | {'US': 'SPY', 'JP': '^N225'} |
| benchmark_strict_available_default | {'US': 'SPY', 'JP': '1306.T'} |
| cache_used | False |
| cache_source | provided_prices |
| cache_path | artifacts/residual_concentration/cache |
| cache_created_at | nan |

Cache used: False / source: `provided_prices` / path: `artifacts/residual_concentration/cache`. Survivorship and current-constituent bias remain when historical constituents are not reconstructed.

## 5. Variant Summary
| CAGR | Annualized_Volatility | Max_Drawdown | Sharpe | Sortino | Calmar | Turnover | Judgment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline_N40 | 0.1026 | 0.0454 | -0.0257 | 2.2595 | 3.9001 | 3.9917 | 0.0721 | clear improvement |
| Baseline_N30 | 0.1026 | 0.0454 | -0.0257 | 2.2595 | 3.9001 | 3.9917 | 0.0721 | clear improvement |
| Baseline_N24 | 0.1026 | 0.0454 | -0.0257 | 2.2595 | 3.9001 | 3.9917 | 0.0721 | clear improvement |
| Baseline_N20 | 0.1026 | 0.0454 | -0.0257 | 2.2595 | 3.9001 | 3.9917 | 0.0721 | clear improvement |
| Baseline_N16 | 0.1026 | 0.0454 | -0.0257 | 2.2595 | 3.9001 | 3.9917 | 0.0721 | clear improvement |
| Baseline_N12 | 0.0960 | 0.0534 | -0.0358 | 1.7972 | 2.9571 | 2.6818 | 0.1631 | baseline |
| Baseline_N10 | 0.1040 | 0.0591 | -0.0367 | 1.7596 | 2.8508 | 2.8379 | 0.2193 | clear improvement |
| Baseline_N8 | 0.1163 | 0.0650 | -0.0408 | 1.7874 | 2.7944 | 2.8497 | 0.2857 | clear improvement |
| Baseline_N6 | 0.1405 | 0.0724 | -0.0365 | 1.9408 | 3.0649 | 3.8478 | 0.2348 | clear improvement |
| Baseline_N4 | 0.0956 | 0.0869 | -0.0620 | 1.0999 | 1.8143 | 1.5424 | 0.3813 | worse |
| Residual_50_N40 | 0.1026 | 0.0454 | -0.0257 | 2.2595 | 3.9001 | 3.9917 | 0.0721 | clear improvement |
| Residual_50_N30 | 0.1026 | 0.0454 | -0.0257 | 2.2595 | 3.9001 | 3.9917 | 0.0721 | clear improvement |
| Residual_50_N24 | 0.1026 | 0.0454 | -0.0257 | 2.2595 | 3.9001 | 3.9917 | 0.0721 | clear improvement |
| Residual_50_N20 | 0.1026 | 0.0454 | -0.0257 | 2.2595 | 3.9001 | 3.9917 | 0.0721 | clear improvement |
| Residual_50_N16 | 0.1026 | 0.0454 | -0.0257 | 2.2595 | 3.9001 | 3.9917 | 0.0721 | clear improvement |
| Residual_50_N12 | 0.0866 | 0.0540 | -0.0360 | 1.6047 | 2.6660 | 2.4041 | 0.1730 | worse |
| Residual_50_N10 | 0.1129 | 0.0580 | -0.0362 | 1.9459 | 3.2092 | 3.1181 | 0.2319 | clear improvement |
| Residual_50_N8 | 0.1180 | 0.0648 | -0.0343 | 1.8216 | 2.7928 | 3.4376 | 0.2395 | clear improvement |
| Residual_50_N6 | 0.1324 | 0.0741 | -0.0386 | 1.7865 | 2.7631 | 3.4283 | 0.3587 | clear improvement |
| Residual_50_N4 | 0.1027 | 0.0900 | -0.0705 | 1.1414 | 1.9008 | 1.4555 | 0.5033 | mixed |
| Residual_55_N40 | 0.1026 | 0.0454 | -0.0257 | 2.2595 | 3.9001 | 3.9917 | 0.0721 | clear improvement |
| Residual_55_N30 | 0.1026 | 0.0454 | -0.0257 | 2.2595 | 3.9001 | 3.9917 | 0.0721 | clear improvement |
| Residual_55_N24 | 0.1026 | 0.0454 | -0.0257 | 2.2595 | 3.9001 | 3.9917 | 0.0721 | clear improvement |
| Residual_55_N20 | 0.1026 | 0.0454 | -0.0257 | 2.2595 | 3.9001 | 3.9917 | 0.0721 | clear improvement |
| Residual_55_N16 | 0.1026 | 0.0454 | -0.0257 | 2.2595 | 3.9001 | 3.9917 | 0.0721 | clear improvement |
| Residual_55_N12 | 0.0879 | 0.0537 | -0.0400 | 1.6369 | 2.6895 | 2.1977 | 0.1735 | worse |
| Residual_55_N10 | 0.1017 | 0.0582 | -0.0382 | 1.7476 | 2.7789 | 2.6594 | 0.2568 | mixed |
| Residual_55_N8 | 0.1189 | 0.0648 | -0.0378 | 1.8345 | 2.8289 | 3.1479 | 0.2713 | clear improvement |
| Residual_55_N6 | 0.1089 | 0.0722 | -0.0492 | 1.5070 | 2.2536 | 2.2127 | 0.3780 | mixed |
| Residual_55_N4 | 0.0704 | 0.0880 | -0.0705 | 0.7997 | 1.2399 | 0.9980 | 0.5038 | worse |
| Residual_60_N40 | 0.1026 | 0.0454 | -0.0257 | 2.2595 | 3.9001 | 3.9917 | 0.0721 | clear improvement |
| Residual_60_N30 | 0.1026 | 0.0454 | -0.0257 | 2.2595 | 3.9001 | 3.9917 | 0.0721 | clear improvement |
| Residual_60_N24 | 0.1026 | 0.0454 | -0.0257 | 2.2595 | 3.9001 | 3.9917 | 0.0721 | clear improvement |
| Residual_60_N20 | 0.1026 | 0.0454 | -0.0257 | 2.2595 | 3.9001 | 3.9917 | 0.0721 | clear improvement |
| Residual_60_N16 | 0.1026 | 0.0454 | -0.0257 | 2.2595 | 3.9001 | 3.9917 | 0.0721 | clear improvement |
| Residual_60_N12 | 0.0879 | 0.0537 | -0.0400 | 1.6369 | 2.6895 | 2.1977 | 0.1735 | worse |
| Residual_60_N10 | 0.0948 | 0.0586 | -0.0382 | 1.6184 | 2.5963 | 2.4796 | 0.2570 | worse |
| Residual_60_N8 | 0.1189 | 0.0648 | -0.0378 | 1.8345 | 2.8289 | 3.1479 | 0.2713 | clear improvement |
| Residual_60_N6 | 0.1104 | 0.0721 | -0.0543 | 1.5317 | 2.2786 | 2.0320 | 0.3581 | mixed |
| Residual_60_N4 | 0.0704 | 0.0880 | -0.0705 | 0.7997 | 1.2399 | 0.9980 | 0.5038 | worse |
| Residual_65_N40 | 0.1026 | 0.0454 | -0.0257 | 2.2595 | 3.9001 | 3.9917 | 0.0721 | clear improvement |
| Residual_65_N30 | 0.1026 | 0.0454 | -0.0257 | 2.2595 | 3.9001 | 3.9917 | 0.0721 | clear improvement |
| Residual_65_N24 | 0.1026 | 0.0454 | -0.0257 | 2.2595 | 3.9001 | 3.9917 | 0.0721 | clear improvement |
| Residual_65_N20 | 0.1026 | 0.0454 | -0.0257 | 2.2595 | 3.9001 | 3.9917 | 0.0721 | clear improvement |
| Residual_65_N16 | 0.1026 | 0.0454 | -0.0257 | 2.2595 | 3.9001 | 3.9917 | 0.0721 | clear improvement |
| Residual_65_N12 | 0.0879 | 0.0537 | -0.0400 | 1.6369 | 2.6895 | 2.1977 | 0.1735 | worse |
| Residual_65_N10 | 0.0948 | 0.0586 | -0.0382 | 1.6184 | 2.5963 | 2.4796 | 0.2570 | worse |
| Residual_65_N8 | 0.1189 | 0.0648 | -0.0378 | 1.8345 | 2.8289 | 3.1479 | 0.2713 | clear improvement |
| Residual_65_N6 | 0.0824 | 0.0734 | -0.0543 | 1.1226 | 1.6792 | 1.5169 | 0.3800 | worse |
| Residual_65_N4 | 0.0704 | 0.0880 | -0.0705 | 0.7997 | 1.2399 | 0.9980 | 0.5038 | worse |
| Residual_70_N40 | 0.1026 | 0.0454 | -0.0257 | 2.2595 | 3.9001 | 3.9917 | 0.0721 | clear improvement |
| Residual_70_N30 | 0.1026 | 0.0454 | -0.0257 | 2.2595 | 3.9001 | 3.9917 | 0.0721 | clear improvement |
| Residual_70_N24 | 0.1026 | 0.0454 | -0.0257 | 2.2595 | 3.9001 | 3.9917 | 0.0721 | clear improvement |
| Residual_70_N20 | 0.1026 | 0.0454 | -0.0257 | 2.2595 | 3.9001 | 3.9917 | 0.0721 | clear improvement |
| Residual_70_N16 | 0.1026 | 0.0454 | -0.0257 | 2.2595 | 3.9001 | 3.9917 | 0.0721 | clear improvement |
| Residual_70_N12 | 0.0903 | 0.0537 | -0.0329 | 1.6804 | 2.7542 | 2.7421 | 0.1835 | clear improvement |
| Residual_70_N10 | 0.0905 | 0.0583 | -0.0442 | 1.5529 | 2.4377 | 2.0458 | 0.2686 | worse |
| Residual_70_N8 | 0.0854 | 0.0639 | -0.0389 | 1.3361 | 2.1443 | 2.1974 | 0.3176 | worse |
| Residual_70_N6 | 0.0762 | 0.0737 | -0.0543 | 1.0343 | 1.5860 | 1.4026 | 0.3594 | worse |
| Residual_70_N4 | 0.0600 | 0.0898 | -0.0705 | 0.6679 | 1.0280 | 0.8501 | 0.5352 | worse |
| Residual_100_N40 | 0.1026 | 0.0454 | -0.0257 | 2.2595 | 3.9001 | 3.9917 | 0.0721 | clear improvement |
| Residual_100_N30 | 0.1026 | 0.0454 | -0.0257 | 2.2595 | 3.9001 | 3.9917 | 0.0721 | clear improvement |
| Residual_100_N24 | 0.1026 | 0.0454 | -0.0257 | 2.2595 | 3.9001 | 3.9917 | 0.0721 | clear improvement |
| Residual_100_N20 | 0.1026 | 0.0454 | -0.0257 | 2.2595 | 3.9001 | 3.9917 | 0.0721 | clear improvement |
| Residual_100_N16 | 0.1026 | 0.0454 | -0.0257 | 2.2595 | 3.9001 | 3.9917 | 0.0721 | clear improvement |
| Residual_100_N12 | 0.0700 | 0.0541 | -0.0362 | 1.2946 | 2.0600 | 1.9325 | 0.2158 | worse |
| Residual_100_N10 | 0.0899 | 0.0580 | -0.0480 | 1.5499 | 2.4215 | 1.8739 | 0.2572 | worse |
| Residual_100_N8 | 0.0823 | 0.0631 | -0.0517 | 1.3039 | 2.1028 | 1.5929 | 0.3019 | worse |
| Residual_100_N6 | 0.0911 | 0.0727 | -0.0395 | 1.2524 | 2.0550 | 2.3057 | 0.3801 | worse |
| Residual_100_N4 | 0.0600 | 0.0898 | -0.0705 | 0.6679 | 1.0280 | 0.8501 | 0.5352 | worse |

## 6. Best by Portfolio Size
| best_cagr_ratio | best_sharpe_ratio | best_sortino_ratio | best_calmar_ratio | lowest_maxdd_ratio |
| --- | --- | --- | --- | --- |
| 40 | Baseline_N40 | Baseline_N40 | Residual_65_N40 | Baseline_N40 | Baseline_N40 |
| 30 | Baseline_N30 | Baseline_N30 | Residual_65_N30 | Baseline_N30 | Baseline_N30 |
| 24 | Baseline_N24 | Baseline_N24 | Residual_65_N24 | Baseline_N24 | Baseline_N24 |
| 20 | Baseline_N20 | Baseline_N20 | Residual_65_N20 | Baseline_N20 | Baseline_N20 |
| 16 | Baseline_N16 | Baseline_N16 | Residual_65_N16 | Baseline_N16 | Baseline_N16 |
| 12 | Baseline_N12 | Baseline_N12 | Baseline_N12 | Residual_70_N12 | Residual_70_N12 |
| 10 | Residual_50_N10 | Residual_50_N10 | Residual_50_N10 | Residual_50_N10 | Residual_50_N10 |
| 8 | Residual_55_N8 | Residual_55_N8 | Residual_55_N8 | Residual_50_N8 | Residual_50_N8 |
| 6 | Baseline_N6 | Baseline_N6 | Baseline_N6 | Baseline_N6 | Baseline_N6 |
| 4 | Residual_50_N4 | Residual_50_N4 | Residual_50_N4 | Baseline_N4 | Baseline_N4 |

## 7. Best by Residual Ratio
| best_cagr_portfolio_size | best_sharpe_portfolio_size | best_sortino_portfolio_size | best_calmar_portfolio_size | lowest_maxdd_portfolio_size |
| --- | --- | --- | --- | --- |
| Baseline | 6 | 40 | 40 | 40 | 40 |
| Residual_100 | 40 | 40 | 40 | 40 | 40 |
| Residual_50 | 6 | 40 | 40 | 40 | 40 |
| Residual_55 | 8 | 40 | 40 | 40 | 40 |
| Residual_60 | 8 | 40 | 40 | 40 | 40 |
| Residual_65 | 8 | 40 | 40 | 40 | 40 |
| Residual_70 | 40 | 40 | 40 | 40 | 40 |

## 8. Sweet Spot Analysis
| total_holdings | cagr_delta_vs_baseline_n12 | maxdd_abs_delta_vs_baseline_n12 | calmar_delta_vs_baseline_n12 | simultaneous_cagr_maxdd_calmar_improvement | zone | judgment |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline_N40 | 40 | 0.0066 | -0.0101 | 1.3099 | True | high_diversification_reference | strong improvement |
| Baseline_N30 | 30 | 0.0066 | -0.0101 | 1.3099 | True | high_diversification_reference | strong improvement |
| Baseline_N24 | 24 | 0.0066 | -0.0101 | 1.3099 | True | diversification | strong improvement |
| Baseline_N20 | 20 | 0.0066 | -0.0101 | 1.3099 | True | diversification | strong improvement |
| Baseline_N16 | 16 | 0.0066 | -0.0101 | 1.3099 | True | diversification | strong improvement |
| Baseline_N12 | 12 | 0.0000 | 0.0000 | 0.0000 | False | current | worse |
| Baseline_N10 | 10 | 0.0080 | 0.0008 | 0.1561 | False | concentration | clear improvement |
| Baseline_N8 | 8 | 0.0202 | 0.0050 | 0.1679 | False | concentration | mixed |
| Baseline_N6 | 6 | 0.0445 | 0.0007 | 1.1660 | False | concentration | clear improvement |
| Baseline_N4 | 4 | -0.0005 | 0.0262 | -1.1394 | False | concentration | worse |
| Residual_50_N40 | 40 | 0.0066 | -0.0101 | 1.3099 | True | high_diversification_reference | strong improvement |
| Residual_50_N30 | 30 | 0.0066 | -0.0101 | 1.3099 | True | high_diversification_reference | strong improvement |
| Residual_50_N24 | 24 | 0.0066 | -0.0101 | 1.3099 | True | diversification | strong improvement |
| Residual_50_N20 | 20 | 0.0066 | -0.0101 | 1.3099 | True | diversification | strong improvement |
| Residual_50_N16 | 16 | 0.0066 | -0.0101 | 1.3099 | True | diversification | strong improvement |
| Residual_50_N12 | 12 | -0.0094 | 0.0002 | -0.2776 | False | current | worse |
| Residual_50_N10 | 10 | 0.0168 | 0.0004 | 0.4364 | False | concentration | clear improvement |
| Residual_50_N8 | 8 | 0.0220 | -0.0015 | 0.7558 | True | concentration | clear improvement |
| Residual_50_N6 | 6 | 0.0363 | 0.0028 | 0.7466 | False | concentration | mixed |
| Residual_50_N4 | 4 | 0.0066 | 0.0347 | -1.2263 | False | concentration | mixed |
| Residual_55_N40 | 40 | 0.0066 | -0.0101 | 1.3099 | True | high_diversification_reference | strong improvement |
| Residual_55_N30 | 30 | 0.0066 | -0.0101 | 1.3099 | True | high_diversification_reference | strong improvement |
| Residual_55_N24 | 24 | 0.0066 | -0.0101 | 1.3099 | True | diversification | strong improvement |
| Residual_55_N20 | 20 | 0.0066 | -0.0101 | 1.3099 | True | diversification | strong improvement |
| Residual_55_N16 | 16 | 0.0066 | -0.0101 | 1.3099 | True | diversification | strong improvement |
| Residual_55_N12 | 12 | -0.0081 | 0.0042 | -0.4841 | False | current | worse |
| Residual_55_N10 | 10 | 0.0056 | 0.0024 | -0.0223 | False | concentration | mixed |
| Residual_55_N8 | 8 | 0.0228 | 0.0020 | 0.4661 | False | concentration | mixed |
| Residual_55_N6 | 6 | 0.0128 | 0.0134 | -0.4691 | False | concentration | mixed |
| Residual_55_N4 | 4 | -0.0256 | 0.0347 | -1.6838 | False | concentration | worse |

## 9. Diversification Reference Review
| variant | cagr_delta | maxdd_abs_delta | vol_delta | calmar_delta | role |
| --- | --- | --- | --- | --- | --- |
| ('Baseline', 'N12_vs_N4') | Baseline_N4 | -0.0005 | 0.0262 | 0.0335 | -1.1394 | concentration_reference |
| ('Baseline', 'N12_vs_N6') | Baseline_N6 | 0.0445 | 0.0007 | 0.0190 | 1.1660 | concentration_reference |
| ('Baseline', 'N12_vs_N8') | Baseline_N8 | 0.0202 | 0.0050 | 0.0116 | 0.1679 | concentration_reference |
| ('Baseline', 'N12_vs_N10') | Baseline_N10 | 0.0080 | 0.0008 | 0.0057 | 0.1561 | concentration_reference |
| ('Baseline', 'N12_vs_N16') | Baseline_N16 | 0.0066 | -0.0101 | -0.0080 | 1.3099 | practical_diversification |
| ('Baseline', 'N12_vs_N20') | Baseline_N20 | 0.0066 | -0.0101 | -0.0080 | 1.3099 | practical_diversification |
| ('Baseline', 'N12_vs_N24') | Baseline_N24 | 0.0066 | -0.0101 | -0.0080 | 1.3099 | practical_diversification |
| ('Baseline', 'N12_vs_N30') | Baseline_N30 | 0.0066 | -0.0101 | -0.0080 | 1.3099 | high_diversification_reference |
| ('Baseline', 'N12_vs_N40') | Baseline_N40 | 0.0066 | -0.0101 | -0.0080 | 1.3099 | high_diversification_reference |
| ('Residual_100', 'N12_vs_N4') | Residual_100_N4 | -0.0101 | 0.0343 | 0.0357 | -1.0825 | concentration_reference |
| ('Residual_100', 'N12_vs_N6') | Residual_100_N6 | 0.0211 | 0.0033 | 0.0187 | 0.3732 | concentration_reference |
| ('Residual_100', 'N12_vs_N8') | Residual_100_N8 | 0.0123 | 0.0154 | 0.0090 | -0.3397 | concentration_reference |
| ('Residual_100', 'N12_vs_N10') | Residual_100_N10 | 0.0199 | 0.0117 | 0.0039 | -0.0586 | concentration_reference |
| ('Residual_100', 'N12_vs_N16') | Residual_100_N16 | 0.0326 | -0.0105 | -0.0087 | 2.0591 | practical_diversification |
| ('Residual_100', 'N12_vs_N20') | Residual_100_N20 | 0.0326 | -0.0105 | -0.0087 | 2.0591 | practical_diversification |
| ('Residual_100', 'N12_vs_N24') | Residual_100_N24 | 0.0326 | -0.0105 | -0.0087 | 2.0591 | practical_diversification |
| ('Residual_100', 'N12_vs_N30') | Residual_100_N30 | 0.0326 | -0.0105 | -0.0087 | 2.0591 | high_diversification_reference |
| ('Residual_100', 'N12_vs_N40') | Residual_100_N40 | 0.0326 | -0.0105 | -0.0087 | 2.0591 | high_diversification_reference |
| ('Residual_50', 'N12_vs_N4') | Residual_50_N4 | 0.0160 | 0.0345 | 0.0360 | -0.9486 | concentration_reference |
| ('Residual_50', 'N12_vs_N6') | Residual_50_N6 | 0.0457 | 0.0026 | 0.0201 | 1.0242 | concentration_reference |
| ('Residual_50', 'N12_vs_N8') | Residual_50_N8 | 0.0314 | -0.0017 | 0.0108 | 1.0335 | concentration_reference |
| ('Residual_50', 'N12_vs_N10') | Residual_50_N10 | 0.0263 | 0.0002 | 0.0040 | 0.7140 | concentration_reference |
| ('Residual_50', 'N12_vs_N16') | Residual_50_N16 | 0.0160 | -0.0103 | -0.0086 | 1.5875 | practical_diversification |
| ('Residual_50', 'N12_vs_N20') | Residual_50_N20 | 0.0160 | -0.0103 | -0.0086 | 1.5875 | practical_diversification |
| ('Residual_50', 'N12_vs_N24') | Residual_50_N24 | 0.0160 | -0.0103 | -0.0086 | 1.5875 | practical_diversification |
| ('Residual_50', 'N12_vs_N30') | Residual_50_N30 | 0.0160 | -0.0103 | -0.0086 | 1.5875 | high_diversification_reference |
| ('Residual_50', 'N12_vs_N40') | Residual_50_N40 | 0.0160 | -0.0103 | -0.0086 | 1.5875 | high_diversification_reference |
| ('Residual_55', 'N12_vs_N4') | Residual_55_N4 | -0.0175 | 0.0305 | 0.0343 | -1.1997 | concentration_reference |
| ('Residual_55', 'N12_vs_N6') | Residual_55_N6 | 0.0209 | 0.0092 | 0.0185 | 0.0149 | concentration_reference |
| ('Residual_55', 'N12_vs_N8') | Residual_55_N8 | 0.0310 | -0.0022 | 0.0111 | 0.9502 | concentration_reference |
| ('Residual_55', 'N12_vs_N10') | Residual_55_N10 | 0.0138 | -0.0018 | 0.0045 | 0.4617 | concentration_reference |
| ('Residual_55', 'N12_vs_N16') | Residual_55_N16 | 0.0147 | -0.0143 | -0.0083 | 1.7939 | practical_diversification |
| ('Residual_55', 'N12_vs_N20') | Residual_55_N20 | 0.0147 | -0.0143 | -0.0083 | 1.7939 | practical_diversification |
| ('Residual_55', 'N12_vs_N24') | Residual_55_N24 | 0.0147 | -0.0143 | -0.0083 | 1.7939 | practical_diversification |
| ('Residual_55', 'N12_vs_N30') | Residual_55_N30 | 0.0147 | -0.0143 | -0.0083 | 1.7939 | high_diversification_reference |
| ('Residual_55', 'N12_vs_N40') | Residual_55_N40 | 0.0147 | -0.0143 | -0.0083 | 1.7939 | high_diversification_reference |
| ('Residual_60', 'N12_vs_N4') | Residual_60_N4 | -0.0175 | 0.0305 | 0.0343 | -1.1997 | concentration_reference |
| ('Residual_60', 'N12_vs_N6') | Residual_60_N6 | 0.0225 | 0.0143 | 0.0184 | -0.1658 | concentration_reference |
| ('Residual_60', 'N12_vs_N8') | Residual_60_N8 | 0.0310 | -0.0022 | 0.0111 | 0.9502 | concentration_reference |
| ('Residual_60', 'N12_vs_N10') | Residual_60_N10 | 0.0069 | -0.0018 | 0.0049 | 0.2819 | concentration_reference |

## 10. Concentration Risk Review
| average_number_of_holdings | average_max_single_name_weight | max_observed_single_name_weight | average_portfolio_herfindahl_index | average_top_3_weight | highest_concentration_date |
| --- | --- | --- | --- | --- | --- |
| Baseline_N10 | 10.0000 | 0.1058 | 0.1099 | 0.1001 | 0.3114 | 2019-03-29 00:00:00 |
| Baseline_N12 | 12.0000 | 0.0884 | 0.0916 | 0.0835 | 0.2616 | 2019-03-29 00:00:00 |
| Baseline_N16 | 16.0000 | 0.0665 | 0.0695 | 0.0626 | 0.1970 | 2019-03-29 00:00:00 |
| Baseline_N20 | 16.0000 | 0.0665 | 0.0695 | 0.0626 | 0.1970 | 2019-03-29 00:00:00 |
| Baseline_N24 | 16.0000 | 0.0665 | 0.0695 | 0.0626 | 0.1970 | 2019-03-29 00:00:00 |
| Baseline_N30 | 16.0000 | 0.0665 | 0.0695 | 0.0626 | 0.1970 | 2019-03-29 00:00:00 |
| Baseline_N4 | 4.0000 | 0.2602 | 0.2717 | 0.2502 | 0.7591 | 2019-03-29 00:00:00 |
| Baseline_N40 | 16.0000 | 0.0665 | 0.0695 | 0.0626 | 0.1970 | 2019-03-29 00:00:00 |
| Baseline_N6 | 6.0000 | 0.1746 | 0.1814 | 0.1668 | 0.5129 | 2019-03-29 00:00:00 |
| Baseline_N8 | 8.0000 | 0.1310 | 0.1365 | 0.1251 | 0.3865 | 2019-03-29 00:00:00 |
| Residual_100_N10 | 10.0000 | 0.1063 | 0.1115 | 0.1002 | 0.3129 | 2019-03-29 00:00:00 |
| Residual_100_N12 | 12.0000 | 0.0888 | 0.0927 | 0.0835 | 0.2619 | 2019-03-29 00:00:00 |
| Residual_100_N16 | 16.0000 | 0.0665 | 0.0695 | 0.0626 | 0.1970 | 2019-03-29 00:00:00 |
| Residual_100_N20 | 16.0000 | 0.0665 | 0.0695 | 0.0626 | 0.1970 | 2019-03-29 00:00:00 |
| Residual_100_N24 | 16.0000 | 0.0665 | 0.0695 | 0.0626 | 0.1970 | 2019-03-29 00:00:00 |
| Residual_100_N30 | 16.0000 | 0.0665 | 0.0695 | 0.0626 | 0.1970 | 2019-03-29 00:00:00 |
| Residual_100_N4 | 4.0000 | 0.2625 | 0.2717 | 0.2503 | 0.7599 | 2019-03-29 00:00:00 |
| Residual_100_N40 | 16.0000 | 0.0665 | 0.0695 | 0.0626 | 0.1970 | 2019-03-29 00:00:00 |
| Residual_100_N6 | 6.0000 | 0.1761 | 0.1847 | 0.1669 | 0.5128 | 2019-03-29 00:00:00 |
| Residual_100_N8 | 8.0000 | 0.1324 | 0.1383 | 0.1252 | 0.3883 | 2019-03-29 00:00:00 |
| Residual_50_N10 | 10.0000 | 0.1060 | 0.1099 | 0.1001 | 0.3125 | 2019-03-29 00:00:00 |
| Residual_50_N12 | 12.0000 | 0.0886 | 0.0930 | 0.0835 | 0.2614 | 2019-03-29 00:00:00 |
| Residual_50_N16 | 16.0000 | 0.0665 | 0.0695 | 0.0626 | 0.1970 | 2019-03-29 00:00:00 |
| Residual_50_N20 | 16.0000 | 0.0665 | 0.0695 | 0.0626 | 0.1970 | 2019-03-29 00:00:00 |
| Residual_50_N24 | 16.0000 | 0.0665 | 0.0695 | 0.0626 | 0.1970 | 2019-03-29 00:00:00 |
| Residual_50_N30 | 16.0000 | 0.0665 | 0.0695 | 0.0626 | 0.1970 | 2019-03-29 00:00:00 |
| Residual_50_N4 | 4.0000 | 0.2617 | 0.2717 | 0.2503 | 0.7595 | 2019-03-29 00:00:00 |
| Residual_50_N40 | 16.0000 | 0.0665 | 0.0695 | 0.0626 | 0.1970 | 2019-03-29 00:00:00 |
| Residual_50_N6 | 6.0000 | 0.1746 | 0.1831 | 0.1668 | 0.5115 | 2019-03-29 00:00:00 |
| Residual_50_N8 | 8.0000 | 0.1313 | 0.1365 | 0.1251 | 0.3869 | 2019-03-29 00:00:00 |
| Residual_55_N10 | 10.0000 | 0.1062 | 0.1115 | 0.1001 | 0.3126 | 2019-03-29 00:00:00 |
| Residual_55_N12 | 12.0000 | 0.0887 | 0.0930 | 0.0835 | 0.2617 | 2019-03-29 00:00:00 |
| Residual_55_N16 | 16.0000 | 0.0665 | 0.0695 | 0.0626 | 0.1970 | 2019-03-29 00:00:00 |
| Residual_55_N20 | 16.0000 | 0.0665 | 0.0695 | 0.0626 | 0.1970 | 2019-03-29 00:00:00 |
| Residual_55_N24 | 16.0000 | 0.0665 | 0.0695 | 0.0626 | 0.1970 | 2019-03-29 00:00:00 |
| Residual_55_N30 | 16.0000 | 0.0665 | 0.0695 | 0.0626 | 0.1970 | 2019-03-29 00:00:00 |
| Residual_55_N4 | 4.0000 | 0.2628 | 0.2717 | 0.2503 | 0.7597 | 2019-03-29 00:00:00 |
| Residual_55_N40 | 16.0000 | 0.0665 | 0.0695 | 0.0626 | 0.1970 | 2019-03-29 00:00:00 |
| Residual_55_N6 | 6.0000 | 0.1753 | 0.1820 | 0.1668 | 0.5121 | 2019-03-29 00:00:00 |
| Residual_55_N8 | 8.0000 | 0.1318 | 0.1383 | 0.1251 | 0.3875 | 2019-03-29 00:00:00 |
| Residual_60_N10 | 10.0000 | 0.1062 | 0.1115 | 0.1001 | 0.3126 | 2019-03-29 00:00:00 |
| Residual_60_N12 | 12.0000 | 0.0887 | 0.0930 | 0.0835 | 0.2617 | 2019-03-29 00:00:00 |
| Residual_60_N16 | 16.0000 | 0.0665 | 0.0695 | 0.0626 | 0.1970 | 2019-03-29 00:00:00 |
| Residual_60_N20 | 16.0000 | 0.0665 | 0.0695 | 0.0626 | 0.1970 | 2019-03-29 00:00:00 |
| Residual_60_N24 | 16.0000 | 0.0665 | 0.0695 | 0.0626 | 0.1970 | 2019-03-29 00:00:00 |
| Residual_60_N30 | 16.0000 | 0.0665 | 0.0695 | 0.0626 | 0.1970 | 2019-03-29 00:00:00 |
| Residual_60_N4 | 4.0000 | 0.2628 | 0.2717 | 0.2503 | 0.7597 | 2019-03-29 00:00:00 |
| Residual_60_N40 | 16.0000 | 0.0665 | 0.0695 | 0.0626 | 0.1970 | 2019-03-29 00:00:00 |
| Residual_60_N6 | 6.0000 | 0.1754 | 0.1820 | 0.1668 | 0.5118 | 2019-03-29 00:00:00 |
| Residual_60_N8 | 8.0000 | 0.1318 | 0.1383 | 0.1251 | 0.3875 | 2019-03-29 00:00:00 |
| Residual_65_N10 | 10.0000 | 0.1062 | 0.1115 | 0.1001 | 0.3126 | 2019-03-29 00:00:00 |
| Residual_65_N12 | 12.0000 | 0.0887 | 0.0930 | 0.0835 | 0.2617 | 2019-03-29 00:00:00 |
| Residual_65_N16 | 16.0000 | 0.0665 | 0.0695 | 0.0626 | 0.1970 | 2019-03-29 00:00:00 |
| Residual_65_N20 | 16.0000 | 0.0665 | 0.0695 | 0.0626 | 0.1970 | 2019-03-29 00:00:00 |
| Residual_65_N24 | 16.0000 | 0.0665 | 0.0695 | 0.0626 | 0.1970 | 2019-03-29 00:00:00 |
| Residual_65_N30 | 16.0000 | 0.0665 | 0.0695 | 0.0626 | 0.1970 | 2019-03-29 00:00:00 |
| Residual_65_N4 | 4.0000 | 0.2628 | 0.2717 | 0.2503 | 0.7597 | 2019-03-29 00:00:00 |
| Residual_65_N40 | 16.0000 | 0.0665 | 0.0695 | 0.0626 | 0.1970 | 2019-03-29 00:00:00 |
| Residual_65_N6 | 6.0000 | 0.1757 | 0.1820 | 0.1668 | 0.5124 | 2019-03-29 00:00:00 |
| Residual_65_N8 | 8.0000 | 0.1318 | 0.1383 | 0.1251 | 0.3875 | 2019-03-29 00:00:00 |
| Residual_70_N10 | 10.0000 | 0.1062 | 0.1115 | 0.1001 | 0.3125 | 2019-03-29 00:00:00 |
| Residual_70_N12 | 12.0000 | 0.0887 | 0.0930 | 0.0835 | 0.2615 | 2019-03-29 00:00:00 |
| Residual_70_N16 | 16.0000 | 0.0665 | 0.0695 | 0.0626 | 0.1970 | 2019-03-29 00:00:00 |
| Residual_70_N20 | 16.0000 | 0.0665 | 0.0695 | 0.0626 | 0.1970 | 2019-03-29 00:00:00 |
| Residual_70_N24 | 16.0000 | 0.0665 | 0.0695 | 0.0626 | 0.1970 | 2019-03-29 00:00:00 |
| Residual_70_N30 | 16.0000 | 0.0665 | 0.0695 | 0.0626 | 0.1970 | 2019-03-29 00:00:00 |
| Residual_70_N4 | 4.0000 | 0.2625 | 0.2717 | 0.2503 | 0.7599 | 2019-03-29 00:00:00 |
| Residual_70_N40 | 16.0000 | 0.0665 | 0.0695 | 0.0626 | 0.1970 | 2019-03-29 00:00:00 |
| Residual_70_N6 | 6.0000 | 0.1758 | 0.1820 | 0.1668 | 0.5121 | 2019-03-29 00:00:00 |
| Residual_70_N8 | 8.0000 | 0.1322 | 0.1383 | 0.1252 | 0.3880 | 2019-03-29 00:00:00 |

## 11. Year / Period Review
Use `annual_returns.csv`, `monthly_returns.csv`, and `drawdown_series.csv` for 2020, 2022, 2023, 2024, 2025, and 2026 YTD where present. This is post-analysis only and does not introduce any regime rule.

## 12. Selection Difference Review
`selection_diff.csv` compares Baseline_N12 with Residual_60/65/100 concentration and diversification variants. `score_components.csv` records base_score, residual_score, final_score, stock/benchmark/residual returns, benchmark_used, selected_flag, and weights.

## 13. Risk Review
Prioritize MaxDD, Worst Year, Worst Month/monthly returns, drawdowns, turnover, average/max single-name weight, Herfindahl index, and 2022 behavior. N4/N6 are limit tests; N30/N40 test whether diversification dilutes Residual signal.

## 14. Recommendation
Conservative candidate: strongest N12/N16/N20 variant with improved Calmar and acceptable MaxDD. Balanced candidate: best stable neighborhood around Residual_55/60/65. Aggressive candidate: best N8/N10 only if MaxDD and single-name weights remain acceptable. High-diversification reference candidate: best N30/N40 by Calmar, treated as reference only. Use actual CSVs before production; do not adopt a single isolated best point.

## 15. Safety Notes
This is not investment advice. Historical yfinance/Wikipedia/free-data tests do not guarantee future returns. Free data can contain missing values, delays, adjusted-price issues, survivorship bias, and historical constituent bias. Alpha Engine is an alpha sleeve, not an all-asset portfolio.
