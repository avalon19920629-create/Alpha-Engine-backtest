# L.U.M.U.S.-8 Core Profiles + Alpha Integration Backtest

## 1. 監査目的
Alpha EngineをMAX_CAGR_MDD25およびROBUST_MEDIANのL.U.M.U.S.-8 Coreに10% / 15% / 20%混ぜた場合の有効性を検証する初期監査です。

## 2. Coreプロファイル
| Profile | Ticker | Weight |
|---|---|---:|
| MAX_CAGR_MDD25 | VT | 40.00% |
| MAX_CAGR_MDD25 | BNDX | 0.00% |
| MAX_CAGR_MDD25 | TLT | 7.00% |
| MAX_CAGR_MDD25 | TIP | 14.00% |
| MAX_CAGR_MDD25 | GLD | 19.00% |
| MAX_CAGR_MDD25 | XLRE | 3.00% |
| MAX_CAGR_MDD25 | DBC | 5.00% |
| MAX_CAGR_MDD25 | SHY | 7.00% |
| MAX_CAGR_MDD25 | BTC-USD | 5.00% |
| ROBUST_MEDIAN | VT | 27.00% |
| ROBUST_MEDIAN | BNDX | 7.00% |
| ROBUST_MEDIAN | TLT | 7.00% |
| ROBUST_MEDIAN | TIP | 16.00% |
| ROBUST_MEDIAN | GLD | 17.00% |
| ROBUST_MEDIAN | XLRE | 2.00% |
| ROBUST_MEDIAN | DBC | 6.00% |
| ROBUST_MEDIAN | SHY | 13.00% |
| ROBUST_MEDIAN | BTC-USD | 5.00% |

## 3. GLD / GLDM と SHY / CASH の扱い
- GLDはGLDMの長期履歴プロキシとして使用している
- SHY / CASH は短期債または現金相当枠であり、SHYは短期債または現金相当枠として使用している
- 実運用ではGLDMや現金に置換する可能性がある

## 4. 比較結果
| Strategy | CAGR | MaxDD | Sharpe | Calmar | Turnover |
|---|---:|---:|---:|---:|---:|
| Alpha_Always_Only | 11.58% | -5.89% | 2.14 | 1.96 | 13.83% |
| MAX_CAGR_MDD25_Core_Only | 4.98% | -18.56% | 0.61 | 0.27 | 0.00% |
| MAX_CAGR_MDD25_Core90_Alpha10 | 5.67% | -15.77% | 0.77 | 0.36 | 1.38% |
| MAX_CAGR_MDD25_Core85_Alpha15 | 6.01% | -14.36% | 0.86 | 0.42 | 2.07% |
| MAX_CAGR_MDD25_Core80_Alpha20 | 6.35% | -12.95% | 0.97 | 0.49 | 2.77% |
| MAX_CAGR_MDD25_Core85_AlphaRegime15 | 5.36% | -15.10% | 0.77 | 0.36 | 0.00% |
| ROBUST_MEDIAN_Core_Only | 5.30% | -12.35% | 0.87 | 0.43 | 0.00% |
| ROBUST_MEDIAN_Core90_Alpha10 | 5.94% | -10.07% | 1.08 | 0.59 | 1.38% |
| ROBUST_MEDIAN_Core85_Alpha15 | 6.26% | -9.05% | 1.20 | 0.69 | 2.07% |
| ROBUST_MEDIAN_Core80_Alpha20 | 6.58% | -8.03% | 1.33 | 0.82 | 2.77% |
| ROBUST_MEDIAN_Core85_AlphaRegime15 | 5.62% | -9.65% | 1.08 | 0.58 | 0.00% |
| SPY | 0.58% | -37.79% | 0.03 | 0.02 | 0.00% |
| QQQ | 4.39% | -45.76% | 0.23 | 0.10 | 0.00% |
| VT | 4.25% | -41.61% | 0.23 | 0.10 | 0.00% |

## 5. 採用判定
- **MAX_CAGR_MDD25: 1. 15%補助Alpha枠として採用候補**
- **ROBUST_MEDIAN: 1. 15%補助Alpha枠として採用候補**

Core85_Alpha15のCAGR改善、限定的なMaxDD悪化、Sharpe/Calmar改善、Coreの防御思想、Turnoverとコスト控除後の可能性に基づく仮判定です。Alpha_Regimeは参考であり主判定には使用しません。

## 6. 最終コメント
Alpha Engine本体が有望か、15%混合が有効か、10%混合の方が美しいか、攻撃型Coreとロバスト型CoreのどちらにAlphaが合うかは上表と同一Core比較で判断します。現段階では独立した風ユニットではなく、補助Alpha枠としての採用可能性を監査するものです。

## 7. 重要な限界
- Alpha側は現在ユニバースによる生存者バイアスを含む
- 過去S&P500構成、日本株の上場廃止、銘柄変更を完全再現していない
- yfinance価格品質に依存する
- 税・スリッページは簡易または未考慮
- GLDはGLDMの履歴プロキシである
- SHYは現金相当枠の履歴プロキシである
- BTC-USDは暗号資産価格系列であり、ETFとは取引日・市場時間が異なる可能性がある
- 投資助言ではない
- 自動売買、自動売却、自動配分変更には接続しない
