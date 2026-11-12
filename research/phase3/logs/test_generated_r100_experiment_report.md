# R100 Composite Experiment Report

## Executive Summary
No data available.

## Why R100 Composite experiment was run
No data available.

## Experimental scope and anti-overheat design
Default scope is exactly 8 R100 variants, cache-first, no full sweep, no full score components, no per-health-check detail logs, partial save per variant, and resume support.

## Candidate variants
['Residual_100_N6_TTL90', 'Residual_100_N6_TTL90_Renew30_Composite']

## Fixed R100 TTL90 vs R100 Composite
                                variant     CAGR  Annualized_Volatility  Max_Drawdown  Sharpe   Calmar  Sortino  Total_Return  Best_Year  Worst_Year  Monthly_Win_Rate  Turnover  Worst_Month  Slippage_Adjusted_CAGR  Tax_Adjusted_CAGR  Tax_Slippage_Adjusted_CAGR  Estimated_Tax_Drag  Estimated_Slippage_Drag  Net_Sharpe  Net_Calmar
                  Residual_100_N6_TTL90 0.123642               0.069355     -0.087917 1.78274 1.406354 2.848412      0.828859   0.242711         0.0          0.616667  0.257574    -0.028975                0.122470           0.123593                    0.122422            0.001220                 0.005409    1.765268    1.380454
Residual_100_N6_TTL90_Renew30_Composite 0.192681               0.058037     -0.028582 3.31996 6.741224 5.265002      1.490495   0.387332         0.0          0.683333  0.313563    -0.015583                0.191596           0.192601                    0.191517            0.001164                 0.004703    3.297406    6.659805

## R100_N6 Ferrari test
No data available.

## R100_N8/N10/N12 comparison
No data available.

## Comparison against Residual_60/65 Composite standard candidates
No data available.

## Cost-adjusted results
                                Variant  Slippage_Adjusted_CAGR  Tax_Adjusted_CAGR  Tax_Slippage_Adjusted_CAGR  Estimated_Tax_Drag  Estimated_Slippage_Drag  Net_Sharpe  Net_Calmar
                  Residual_100_N6_TTL90                0.122470           0.123593                    0.122422            0.001220                 0.005409    1.765268    1.380454
Residual_100_N6_TTL90_Renew30_Composite                0.191596           0.192601                    0.191517            0.001164                 0.004703    3.297406    6.659805

## Active exposure and cash-drag analysis
                                variant  full_invested_days_ratio  below_90pct_exposure_days_ratio  below_80pct_exposure_days_ratio  below_70pct_exposure_days_ratio  average_active_exposure  median_active_exposure  min_active_exposure  average_active_position_count  average_cash_weight exposure_judgment
                  Residual_100_N6_TTL90                  0.786973                         0.213027                         0.213027                         0.213027                 0.786973                     1.0                  0.0                       4.721839             0.213027  キャッシュ寄与が大きい。慎重評価
Residual_100_N6_TTL90_Renew30_Composite                  0.623755                         0.376245                         0.310345                         0.310345                 0.729240                     1.0                  0.0                       4.375479             0.270760  キャッシュ寄与が大きい。慎重評価

## 2022 stress-year analysis
                                variant  return_2022  max_drawdown_2022  worst_month_2022  monthly_win_rate_2022  average_active_exposure_2022 stress_observation_status  renewal_pass_rate_2022  cash_weight_proxy_2022  forced_non_renewals_2022
                  Residual_100_N6_TTL90     0.242711          -0.042722         -0.019929               0.916667                      0.980769                  observed                     NaN                0.019231                         0
Residual_100_N6_TTL90_Renew30_Composite     0.387332          -0.022070         -0.001649               0.916667                      0.929646                  observed                0.944444                0.070354                         1

## Drawdown episode analysis
                                variant drawdown_start_date drawdown_trough_date recovery_date  max_drawdown_depth  days_to_trough  days_to_recovery  return_during_drawdown  active_exposure_during_drawdown  renewal_decisions_around_drawdown
                  Residual_100_N6_TTL90          2019-02-01           2019-08-02    2019-12-06           -0.087917             182             308.0               -0.087917                         0.986425                                0.0
                  Residual_100_N6_TTL90          2020-12-30           2021-01-15    2021-05-19           -0.050750              16             140.0               -0.050750                         0.990099                                0.0
                  Residual_100_N6_TTL90          2022-02-15           2022-03-23    2022-05-04           -0.042722              36              78.0               -0.042722                         0.982456                                0.0
                  Residual_100_N6_TTL90          2021-07-05           2021-08-27    2021-09-14           -0.035516              53              71.0               -0.035516                         1.000000                                0.0
                  Residual_100_N6_TTL90          2022-05-30           2022-06-13    2022-06-24           -0.031118              14              25.0               -0.031118                         1.000000                                0.0
Residual_100_N6_TTL90_Renew30_Composite          2020-12-30           2021-01-13    2021-02-25           -0.028582              14              57.0               -0.028582                         0.976190                                0.0
Residual_100_N6_TTL90_Renew30_Composite          2020-07-28           2020-10-15    2020-11-10           -0.026931              79             105.0               -0.026931                         0.844258                                6.0
Residual_100_N6_TTL90_Renew30_Composite          2021-08-20           2021-08-27    2021-09-07           -0.025133               7              18.0               -0.025133                         0.829305                                0.0
Residual_100_N6_TTL90_Renew30_Composite          2022-02-16           2022-03-07    2022-04-07           -0.022070              19              50.0               -0.022070                         0.942864                                6.0
Residual_100_N6_TTL90_Renew30_Composite          2022-07-05           2022-07-13    2022-07-27           -0.021341               8              22.0               -0.021341                         1.000000                                0.0

## Renewal decision analysis
                                variant  decisions  renewed_count  renewal_rate  rank_pass_rate  residual_pass_rate  price_above_50dma_pass_rate  composite_pass_rate  average_composite_count
Residual_100_N6_TTL90_Renew30_Composite         72             60      0.833333        0.944444            0.777778                     0.611111             0.833333                 2.333333

## Holding period analysis
                                variant  count  mean_holding_days  median_holding_days  max_holding_days  min_holding_days  percent_held_around_90_days  percent_extended_beyond_90_days  percent_reached_max_120_days  bucket_le_90  bucket_91_105  bucket_106_120  bucket_gt_120
                  Residual_100_N6_TTL90    102          84.882353                 90.0              90.0               3.0                     0.941176                         0.000000                      0.000000           102              0               0              0
Residual_100_N6_TTL90_Renew30_Composite     72         111.055556                120.0             122.0              90.0                     0.305556                         0.722222                      0.694444            20              2              43              7

## Concentration and ticker contribution risk
                                variant  unique_tickers_used  top_5_contribution_share_approx  top_10_contribution_share_approx  largest_single_ticker_activity_share  repeat_winner_count  repeat_loser_count concentration_risk
                  Residual_100_N6_TTL90                   15                         0.531857                          0.874534                              0.137544                    9                   0         controlled
Residual_100_N6_TTL90_Renew30_Composite                   14                         0.571209                          0.903964                              0.151778                   10                   0           moderate

## Future data boundary review
                                      check                status                                                                               details
health_check_date near trade_date + 90 days                  pass                                   observed_min_days=90, observed_max_days=91, rows=72
     renewal inputs as-of health_check_date implementation_review       _health_row uses asof_prices(prices, date) and score functions bounded to date.
         50DMA uses prior/as-of prices only implementation_review        _health_row builds s=asof_prices(prices,date).ffill()[ticker] before tail(50).
               rank/residual as-of boundary implementation_review _select_ttl_candidates and compute_residual_momentum_score receive health_check date.
                weekly degradation boundary implementation_review               run_ttl_renewal_variant passes each weekly check date into _health_row.

## Overdrive recommendation
                                variant      recommendation  net_cagr  max_drawdown  average_active_exposure  max_drawdown_2022 concentration_risk stress_observation_status                                                                           criteria_note
Residual_100_N6_TTL90_Renew30_Composite Overdrive Candidate  0.191517     -0.028582                  0.72924           -0.02207           moderate                  observed Classified using net CAGR, MaxDD, exposure, 2022 stress, concentration, and complexity.

## Safety notes
This is research/backtest output, not investment advice. R100 concentrated variants may be psychologically difficult to hold. High CAGR does not automatically imply standard adoption. R100_N6 should be treated as Overdrive / Satellite unless risk metrics are unexpectedly robust. Default L.U.M.U.S.-8 standard candidate remains Residual_60_N12_TTL90_Renew30_Composite unless clearly proven otherwise.
