# TTL Composite Forensics Report

## 1. Executive Summary
This candidate-only forensic audit reviews TTL90_Renew30_Composite candidates using existing artifacts only by default. It is not a full universe search.

## 2. Why this forensic audit was needed
Composite renewal previously showed large MaxDD, Sharpe, and Calmar improvements; this report checks cash exposure, stress-year behavior, renewal conditions, concentration, future-data boundaries, and complexity.

## 3. Data and artifact sources
Source directory: `/tmp/pytest-of-agent/pytest-1/test_missing_important_files_w0/src`. Missing files: ['trade_log.csv', 'holding_periods.csv', 'renewal_decisions.csv']. Important missing files: ['trade_log.csv', 'holding_periods.csv', 'renewal_decisions.csv'].

## 4. Candidate variants
Selected variants: ['Residual_60_N12_TTL90', 'Residual_60_N12_TTL90_Renew30_Composite', 'Residual_65_N12_TTL90', 'Residual_65_N12_TTL90_Renew30_Composite', 'Baseline_N12_TTL90', 'Residual_60_N12_TTL120', 'Residual_60_N12_TTL90_Renew30_Rank', 'Residual_60_N12_TTL90_Renew30_Residual'].

## 5. Fixed TTL90 vs Composite Renewal
                                variant  CAGR  Max_Drawdown  Sharpe  Calmar  Turnover  Tax_Slippage_Adjusted_CAGR  Net_Sharpe  Net_Calmar  Estimated_Tax_Drag  Estimated_Slippage_Drag  average_active_exposure exposure_judgment
                  Residual_60_N12_TTL90  0.10         -0.20     1.0     0.5      0.50                        0.08         0.8         0.4                0.01                    0.002                      NaN       unavailable
Residual_60_N12_TTL90_Renew30_Composite  0.14         -0.12     1.4     1.1      0.45                        0.12         1.2         1.0                0.01                    0.002                      NaN       unavailable
                  Residual_65_N12_TTL90  0.11         -0.22     1.1     0.5      0.55                        0.09         0.9         0.4                0.01                    0.002                      NaN       unavailable
Residual_65_N12_TTL90_Renew30_Composite  0.16         -0.13     1.5     1.2      0.50                        0.13         1.3         1.0                0.01                    0.002                      NaN       unavailable
                     Baseline_N12_TTL90  0.08         -0.25     0.8     0.3      0.60                        0.06         0.6         0.2                0.01                    0.002                      NaN               NaN
                 Residual_60_N12_TTL120  0.09         -0.21     0.9     0.4      0.50                        0.07         0.7         0.3                0.01                    0.002                      NaN               NaN
     Residual_60_N12_TTL90_Renew30_Rank  0.12         -0.15     1.2     0.8      0.45                        0.10         1.0         0.7                0.01                    0.002                      NaN               NaN
 Residual_60_N12_TTL90_Renew30_Residual  0.13         -0.16     1.3     0.8      0.46                        0.11         1.1         0.7                0.01                    0.002                      NaN               NaN

## 6. Cost-adjusted comparison
Cost model is approximate and is not tax advice. See `candidate_comparison.csv` for gross/net CAGR, Sharpe, Calmar, turnover, and drag fields when available.

## 7. Active exposure analysis
Exposure is reconstructed from BUY/SELL events and weights in `trade_log.csv`; cash weight is `1 - active_weight`. Judgment bands: 95-100% nearly full invested, 80-95% moderate immune system, 60-80% large cash contribution, below 60% nearly a different strategy.

                                variant  average_active_exposure exposure_judgment
                  Residual_60_N12_TTL90                      NaN       unavailable
Residual_60_N12_TTL90_Renew30_Composite                      NaN       unavailable
                  Residual_65_N12_TTL90                      NaN       unavailable
Residual_65_N12_TTL90_Renew30_Composite                      NaN       unavailable

## 8. 2022 stress-year review
                                variant  return_2022  max_drawdown_2022  worst_month_2022  monthly_win_rate_2022  average_active_exposure_2022 stress_observation_status  renewal_pass_rate_2022  cash_weight_proxy_2022  forced_non_renewals_2022  return_2022_vs_fixed_delta  maxdd_2022_vs_fixed_delta
                  Residual_60_N12_TTL90      0.01928              -0.12             -0.03                   0.75                           0.0                  observed                     NaN                     1.0                         0                         NaN                        NaN
Residual_60_N12_TTL90_Renew30_Composite      0.01928              -0.12             -0.03                   0.75                           0.0                  observed                     NaN                     1.0                         0                         0.0                        0.0
                  Residual_65_N12_TTL90      0.01928              -0.12             -0.03                   0.75                           0.0                  observed                     NaN                     1.0                         0                         NaN                        NaN
Residual_65_N12_TTL90_Renew30_Composite      0.01928              -0.12             -0.03                   0.75                           0.0                  observed                     NaN                     1.0                         0                         0.0                        0.0
                     Baseline_N12_TTL90      0.01928              -0.12             -0.03                   0.75                           0.0                  observed                     NaN                     1.0                         0                         NaN                        NaN
                 Residual_60_N12_TTL120      0.01928              -0.12             -0.03                   0.75                           0.0                  observed                     NaN                     1.0                         0                         NaN                        NaN
     Residual_60_N12_TTL90_Renew30_Rank      0.01928              -0.12             -0.03                   0.75                           0.0                  observed                     NaN                     1.0                         0                         NaN                        NaN
 Residual_60_N12_TTL90_Renew30_Residual      0.01928              -0.12             -0.03                   0.75                           0.0                  observed                     NaN                     1.0                         0                         NaN                        NaN

## 9. Drawdown episode review
Drawdown episodes are extracted from `drawdown_series.csv` by identifying below-zero drawdown intervals and troughs, then joining average exposure and renewal-decision counts where available.

## 10. Renewal decision analysis
Renewal decisions are grouped overall, by year, and by market. Pass rates are calculated for rank, residual, 50DMA price, and composite conditions where source columns are available.

## 11. Holding period analysis
Holding periods are bucketed into <=90, 91-105, 106-120, and >120 days to check whether renewal behaves like simple TTL120.

## 12. Ticker contribution / concentration review
Ticker contribution is approximate when only trade logs are available; activity and weight-use concentration proxies are reported.

## 13. Future data boundary review
Implementation review checks that renewal health checks call as-of bounded scoring and 50DMA logic. If short `days_to_health_check` rows are concentrated in the final observed cycle, match the source/audit end date, and have no subsequent return period, they are classified as `end_of_sample_partial_cycle` end-of-sample truncation rather than future leakage. See `future_data_boundary_review.csv`.

                                      check                status                                                                               details
health_check_date near trade_date + 90 days         not_available                                                     renewal_decisions.csv unavailable
     renewal inputs as-of health_check_date implementation_review       _health_row uses asof_prices(prices, date) and score functions bounded to date.
         50DMA uses prior/as-of prices only implementation_review        _health_row builds s=asof_prices(prices,date).ffill()[ticker] before tail(50).
               rank/residual as-of boundary implementation_review _select_ttl_candidates and compute_residual_momentum_score receive health_check date.
                weekly degradation boundary implementation_review               run_ttl_renewal_variant passes each weekly check date into _health_row.

## 14. Complexity scorecard
                                                      comparison          base_variant                       composite_variant  net_cagr_improvement  net_sharpe_improvement  net_calmar_improvement  maxdd_reduction  turnover_change  active_exposure_drop rule_complexity operational_burden interpretability failure_risk     final_judgment
Residual_60_N12_TTL90 vs Residual_60_N12_TTL90_Renew30_Composite Residual_60_N12_TTL90 Residual_60_N12_TTL90_Renew30_Composite                  0.04                     0.4                     0.6             0.08            -0.05                   NaN            high             medium           medium       medium Adopt with caution
Residual_65_N12_TTL90 vs Residual_65_N12_TTL90_Renew30_Composite Residual_65_N12_TTL90 Residual_65_N12_TTL90_Renew30_Composite                  0.04                     0.4                     0.6             0.09            -0.05                   NaN            high             medium           medium       medium Adopt with caution

## 15. Recommendation
Keep `Residual_60_N12_TTL90` as the fixed-TTL control. Promote `Residual_60_N12_TTL90_Renew30_Composite` only if net metrics beat fixed TTL90 without a large active-exposure drop. Treat `Residual_65_N12_TTL90_Renew30_Composite` as the more aggressive candidate. If average exposure falls below 80%, classify the improvement as materially cash-driven and require caution. A further FULL audit is not required for this candidate-forensics purpose, but remains useful before production standardization.

## 16. Safety notes
This is research, not investment advice. Past performance does not guarantee future results. Free-data artifacts may contain survivorship bias, missing data, adjusted-price artifacts, and approximate cost assumptions.
