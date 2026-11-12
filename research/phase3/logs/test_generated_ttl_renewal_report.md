# TTL Renewal Audit Report

## 1. Executive Summary
Best Calmar variant in this run: **Baseline_N12_TTL90**. Comparison baseline for current residual standard: **Residual_60_N12_TTL90**. Recommendation buckets to review: Keep TTL90, Move to Fixed TTL120, Use TTL90 + Renewal30, Research only, Reject.

## 2. Audit Scope
Residual Momentum selection with US/JP balanced N12/N10, fixed TTL variants, and TTL90 Renewal30 variants only. No Exit Protocol, Regime Filter, VCP, Sector Residual, Downside penalty, or Correlation penalty.

## 3. Data Quality
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
| usable_end | 2019-06-30 |
| cache_used | True |
| cache_source | provided_prices |
| prices_cache_found |  |
| benchmarks_cache_found |  |
| cache_loaded_at |  |
| universe_size | 16 |
| usable_universe_size | 16 |

## 4. Cache Usage
cache_used=True; cache_source=provided_prices; cache_loaded_at=.

## 5. Fixed TTL Results
Review TTL30/60/90/120/180 rows in `variant_summary.csv` for CAGR, MaxDD, Calmar, holding days, and turnover.

## 6. TTL Renewal Protocol Results
Rank uses same-market rank <= market_slots*2. Residual uses residual_raw > 0. Composite requires at least two of rank buffer, residual_raw > 0, and price above 50DMA. Weekly degradation applies only during the extension window and is an extension cutoff, not a general exit.

## 7. Cost-adjusted Results
Tax/slippage are approximate relative-comparison estimates, not tax advice. See `cost_adjusted_summary.csv`.

## 8. Turnover / Tax / Slippage Review
Compare Annualized_Turnover, Trade_Count, Estimated_Tax_Drag, and Estimated_Slippage_Drag.

## 9. Holding Period Review
Renewal variants enforce no initial exit before 90 days and max holding days of 120 days.

## 10. 2022 Stress Year Review
Use `annual_returns.csv`, `monthly_returns.csv`, and `drawdown_series.csv` for 2022 if present.

## 11. Comparison vs Current TTL90
Primary row: Residual_60_N12_TTL90 versus Residual_60_N12_TTL120 and Residual_60_N12_TTL90_Renew30_*.

## 12. Recommendation
Keep TTL90 if it remains Calmar/MaxDD competitive. Move to Fixed TTL120 only if it clearly improves net CAGR/Calmar without larger drawdowns. Use TTL90 + Renewal30 only if net metrics and turnover improve enough to justify complexity. Otherwise Research only or Reject.

## 13. Safety Notes
Research only; no automatic trading. Free data may be incomplete and biased. Cost model is approximate and not tax advice.

## Summary Preview
| CAGR | Max_Drawdown | Calmar | Turnover | Tax_Slippage_Adjusted_CAGR |
| --- | --- | --- | --- | --- |
| Baseline_N12_TTL90 | 0.1115 | -0.0175 | 6.3884 | 0.2981 | 0.1102 |
| Residual_55_N12_TTL60 | 0.0351 | -0.0256 | 1.3713 | 0.2264 | 0.0337 |
| Residual_55_N12_TTL90 | -0.0010 | -0.0388 | -0.0269 | 0.3394 | -0.0024 |
| Residual_55_N12_TTL120 | 0.0256 | -0.0261 | 0.9794 | 0.3386 | 0.0242 |
| Residual_55_N12_TTL180 | 0.0242 | -0.0261 | 0.9252 | 0.5000 | 0.0231 |
| Residual_55_N12_TTL90_Renew30_Rank | 0.0256 | -0.0261 | 0.9794 | 0.3386 | 0.0242 |
| Residual_55_N12_TTL90_Renew30_Residual | 0.0464 | -0.0213 | 2.1729 | 0.3386 | 0.0450 |
| Residual_55_N12_TTL90_Renew30_Composite | 0.0836 | -0.0179 | 4.6813 | 0.3386 | 0.0821 |
| Residual_60_N12_TTL60 | 0.0351 | -0.0256 | 1.3713 | 0.2264 | 0.0337 |
| Residual_60_N12_TTL90 | -0.0010 | -0.0388 | -0.0269 | 0.3394 | -0.0024 |
| Residual_60_N12_TTL120 | 0.0256 | -0.0261 | 0.9794 | 0.3386 | 0.0242 |
| Residual_60_N12_TTL180 | 0.0242 | -0.0261 | 0.9252 | 0.5000 | 0.0231 |
| Residual_60_N12_TTL90_Renew30_Rank | 0.0256 | -0.0261 | 0.9794 | 0.3386 | 0.0242 |
| Residual_60_N12_TTL90_Renew30_Residual | 0.0464 | -0.0213 | 2.1729 | 0.3386 | 0.0450 |
| Residual_60_N12_TTL90_Renew30_Composite | 0.0836 | -0.0179 | 4.6813 | 0.3386 | 0.0821 |
| Residual_65_N12_TTL60 | 0.0445 | -0.0206 | 2.1589 | 0.2559 | 0.0429 |
| Residual_65_N12_TTL90 | -0.0010 | -0.0388 | -0.0269 | 0.3394 | -0.0024 |
| Residual_65_N12_TTL120 | 0.0256 | -0.0261 | 0.9794 | 0.3386 | 0.0242 |
| Residual_65_N12_TTL180 | 0.0242 | -0.0261 | 0.9252 | 0.5000 | 0.0231 |
| Residual_65_N12_TTL90_Renew30_Rank | 0.0256 | -0.0261 | 0.9794 | 0.3386 | 0.0242 |
| Residual_65_N12_TTL90_Renew30_Residual | 0.0464 | -0.0213 | 2.1729 | 0.3386 | 0.0450 |
| Residual_65_N12_TTL90_Renew30_Composite | 0.0836 | -0.0179 | 4.6813 | 0.3386 | 0.0821 |
