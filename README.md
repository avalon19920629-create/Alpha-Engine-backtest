# Alpha Engine Backtest Audit v0.1

日米株モメンタム Alpha Engine のポイントインタイム簡易監査です。スクリーニング日以前のデータのみを使い、翌取引日から保有します。投資助言ではなく、自動売買・自動売却・自動配分変更には接続しません。

## 実行方法

```bash
python -m pip install -r requirements.txt
python alpha_engine_backtest.py --demo --start 2018-01-01 --end 2025-12-31 --output-dir artifacts/demo
```

依存関係をインストールできないネットワーク制限環境ではデモとテストは実行できません。
その場合は、PyPIへ接続できるユーザー環境で上記インストールを実行してから、次を実行してください。

```bash
python -m unittest -v
python alpha_engine_backtest.py --demo --start 2018-01-01 --end 2025-12-31 --output-dir artifacts/demo
```

実データ利用時は、調整済み終値を列=ティッカー、行=取引日とした `DataFrame` と現在ユニバースを用意し、`run_backtest` と `write_outputs` を呼び出してください。現在ユニバースには生存者バイアスがあります。

## 出力ファイル

- `selected_tickers_by_period.csv`
- `backtest_summary.csv`
- `annual_returns.csv`
- `monthly_returns.csv`
- `drawdown_report.csv`
- `turnover_report.csv`
- `momentum_alpha_backtest_report.md`

## テスト

```bash
python -m py_compile alpha_engine_backtest.py tests/test_alpha_engine_backtest.py
python -m unittest -v
```

Colabでの確認記録には、実行日、コミットID、および `python -m unittest -v` の最終行（例: `OK`）を記載してください。

## Live mode（yfinance 実データ）

`--demo` を付けない場合は、Alpha-Engine と同じ現在の US / JP ユニバース、レジーム指数、ベンチマークを yfinance から取得します。スコア計算のため開始日の400営業日前から取得しますが、評価期間は `--start` 以降です。

```bash
python alpha_engine_backtest.py --start 2015-01-01 --end 2026-06-15 --rebalance quarterly --output-dir artifacts/live
```

Live mode は yfinance と外部ネットワークの可用性に依存します。現在の構成銘柄を過去にも適用するため生存者バイアスがあり、上場廃止・銘柄変更・過去の指数構成を完全には再現しません。また、企業行動の調整、欠損、配信元の訂正など価格データ品質にも限界があります。結果は調査用途であり、取引判断には追加検証が必要です。
