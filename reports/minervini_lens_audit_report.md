# Minervini Lens Audit Report

## Executive Summary
Best non-baseline by Calmar was **Residual_20**. Baseline CAGR 10.57%, Vol 5.17%, MaxDD -5.89%, Calmar 1.79; Residual_20 CAGR 10.48%, Vol 5.17%, MaxDD -5.21%, Calmar 2.01. The judgment is **clear improvement**; no overly optimistic production recommendation is made from demo/free data alone.

## Baseline Reminder
TTL is fixed at 90 days / quarterly rebalancing, trades occur four times per year, no Exit Protocol, no Regime Filter, no stop loss, no discretionary cash retreat. Baseline remains the current Alpha Engine score only.

## CLI
`python alpha_engine_backtest.py --demo --audit minervini_lens --output-dir artifacts/minervini_lens`

## Variant Summary Table
| Variant | CAGR | Annualized_Volatility | Max_Drawdown | Sharpe | Sortino | Calmar | Turnover | Judgment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 0.1057 | 0.0517 | -0.0589 | 2.0450 | 3.3256 | 1.7938 | 0.1235 | baseline |
| Residual_20 | 0.1048 | 0.0517 | -0.0521 | 2.0276 | 3.3013 | 2.0108 | 0.1258 | clear improvement |
| Residual_30 | 0.0987 | 0.0516 | -0.0521 | 1.9115 | 3.1075 | 1.8926 | 0.1313 | clear improvement |
| VCP_10 | 0.1057 | 0.0517 | -0.0589 | 2.0450 | 3.3256 | 1.7938 | 0.1235 | no improvement |
| VCP_20 | 0.1057 | 0.0517 | -0.0589 | 2.0450 | 3.3256 | 1.7938 | 0.1235 | no improvement |
| Residual_20_VCP_10 | 0.1013 | 0.0515 | -0.0521 | 1.9647 | 3.1991 | 1.9428 | 0.1311 | clear improvement |
| Residual_30_VCP_10 | 0.0970 | 0.0516 | -0.0521 | 1.8785 | 3.0566 | 1.8609 | 0.1447 | clear improvement |
| Residual_20_VCP_20 | 0.0966 | 0.0517 | -0.0521 | 1.8706 | 3.0295 | 1.8534 | 0.1313 | clear improvement |

## Improvement Judgment
Classifications use CAGR, Vol, MaxDD, Calmar, and turnover versus Baseline: clear improvement / mixed / no improvement / worse.

## Residual Momentum Findings
Residual Momentum uses stock return minus US benchmark (SPY/^GSPC) or JP benchmark (1306.T/^TOPX/^N225), then rank-percentile scoring. Compare Residual_20 and Residual_30 in the table; higher residual weight is not assumed better unless Calmar and downside behavior improve.

## VCP Proxy Findings
VCP Proxy combines trend alignment, near-52-week-high behavior, MA20 extension penalty, volatility contraction, and range contraction. It is not a low-volatility strategy; it only prioritizes tighter action among already strong candidates. VCP is a scoring lens, not a hard filter.

## Combined Lens Findings
Combined variants test whether Residual + VCP improves beyond single lenses without changing the 90-day mechanical trade cadence.

## Risk Review
Review MaxDD, Worst Year, 2022 annual returns when present, and turnover in the CSV artifacts. Sector data is not newly fetched; concentration review is limited to selected tickers.

## Recommendation
Production Alpha Engine should not be changed unless a non-baseline variant shows durable Calmar improvement without excessive turnover and without materially worsening adverse years. Candidate from this run: **Residual_20** only if its judgment is clear improvement or acceptable mixed after live-data validation.

## Safety Notes
This is research, not investment advice. Past data does not guarantee future returns. yfinance/Wikipedia/free data can have missing values, delays, index membership bias, and survivorship bias; results are framed within free-data constraints for individual investors.
