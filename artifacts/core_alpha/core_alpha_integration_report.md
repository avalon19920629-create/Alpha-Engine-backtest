# Core + Alpha Integration Backtest Report

## 1. 監査目的
Alpha EngineをL.U.M.U.S.-8の10〜15%補助Alpha枠として採用する価値があるかを検証する初期監査です。

## 2. Core設定
| ticker | weight |
|---|---:|
| VT | 85.00% |
| BNDX | 0.00% |
| TLT | 0.00% |
| TIP | 0.00% |
| GLDM | 0.00% |
| DBC | 0.00% |
| XLRE | 0.00% |
| CASH | 15.00% |

## 3. 比較結果
| Strategy | CAGR | MaxDD | Sharpe | Calmar |
|---|---:|---:|---:|---:|
| Core_Only | 3.82% | -36.50% | 0.24 | 0.10 |
| Alpha_Always_Only | 11.58% | -5.89% | 2.14 | 1.96 |
| Core90_Alpha10 | 4.71% | -32.61% | 0.33 | 0.14 |
| Core85_Alpha15 | 5.14% | -30.60% | 0.38 | 0.17 |
| Core80_Alpha20 | 5.57% | -28.55% | 0.43 | 0.20 |
| Core90_AlphaRegime10 | 4.28% | -33.01% | 0.30 | 0.13 |
| Core85_AlphaRegime15 | 4.50% | -31.19% | 0.33 | 0.14 |
| SPY | 0.58% | -37.79% | 0.03 | 0.02 |
| QQQ | 4.39% | -45.76% | 0.23 | 0.10 |
| VT | 4.25% | -41.61% | 0.23 | 0.10 |
| TOPIX | 9.86% | -44.28% | 0.53 | 0.22 |

## 4. 採用判定
**1. L.U.M.U.S.-8内15%補助枠として採用候補**

Core85_Alpha15について、Core_Only比のCAGR、MaxDD、Sharpe/Calmar、およびTurnoverを確認した仮判定です。コスト控除後の有用性とCoreの防御思想維持は追加検証が必要です。主判定はAlpha_Alwaysであり、Alpha_Regime_Filterは参考です。

## 5. 重要な限界
- Alpha側は現在ユニバースによる生存者バイアスを含む
- 過去S&P500構成、日本株の上場廃止、銘柄変更を完全再現していない
- yfinance価格品質に依存する
- 税・スリッページは簡易または未考慮
- Core比率は `config/lumus8_core_weights.csv` に依存する
- 投資助言ではない
- 自動売買、自動売却、自動配分変更には接続しない
