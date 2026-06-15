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
python -m unittest -v
```
