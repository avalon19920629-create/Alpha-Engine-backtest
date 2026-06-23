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

## Core + Alpha Integration Backtest

L.U.M.U.S.-8 Core と point-in-time `Alpha_Always` を比較し、10〜15%補助枠としての価値を監査します。

```bash
python core_alpha_integration_backtest.py --start 2015-01-01 --end 2026-06-15 --core-weights config/lumus8_core_weights.csv --output-dir artifacts/core_alpha
```

`config/lumus8_core_weights.csv` は Core のティッカーと比率を定義し、合計は必ず1.0にしてください。初期値は正式比率を仮定しない全ゼロであるため、実行前に利用者が設定する必要があります。`CASH` は日次0%リターンとして扱います。

出力先には summary、年次/月次リターン、ドローダウン、資産曲線、回転率、Core比較、選定履歴、およびMarkdown監査レポートが生成されます。Alpha側には現在ユニバース由来の生存者バイアスがあり、Core結果は設定ファイルの比率に依存します。本機能は投資助言ではなく、自動売買・自動売却・自動配分変更には接続しません。

## L.U.M.U.S.-8 Core Profiles + Alpha Integration Backtest

複数のCore候補を同時に読み込み、各Core単体と `Alpha_Always` 10% / 15% / 20% 混合を比較します。`MAX_CAGR_MDD25` はCAGRを重視しつつ最大ドローダウン25%を意識した攻撃型候補、`ROBUST_MEDIAN` は配分の頑健性を重視した候補です。

```bash
python core_alpha_integration_backtest.py --start 2015-01-01 --end 2026-06-15 --core-profiles config/lumus8_core_profiles.csv --output-dir artifacts/core_alpha_profiles
```

`config/lumus8_core_profiles.csv` は `profile,ticker,weight` 形式で、各profileのweight合計は必ず1.0です。GLDはGLDMの長期履歴プロキシ、SHYは短期債または現金相当枠として使用します（`CASH` 指定時は日次0%）。BTC-USDはETFと取引日・市場時間が異なる可能性があります。Alpha側には現在ユニバースによる生存者バイアスがあります。本監査は投資助言ではなく、自動売買・自動売却・自動配分変更には接続しません。

### 評価期間とメトリクス計算
価格取得はAlphaスコア計算とリバランス準備のため `--start` より前のwarmup期間を含みますが、CAGR、MaxDD、Sharpe、Calmar、Total_Returnなどの成績指標は `--start` 以降、`--end` 以前の評価期間リターンのみで計算します。CAGRの年数分母にwarmup期間は含めず、Equity Curveも評価開始時点でリセットします。本バックテストは戦略検証・コード品質確認を目的とし、投資助言ではありません。

## Colab Production Gearbox

`alpha_engine_production.py` is the Colab-oriented production entry point for running one Alpha Engine body with a profile switch instead of separate ROBUST/FERRARI code paths. Defaults are intentionally conservative: `ENGINE_PROFILE="ROBUST"`, TTL=90 days, Renew=30 days, and Composite health checks.

### Minimal Colab cells

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
%cd /content/drive/MyDrive
!git clone https://github.com/<your-org>/Alpha-Engine-backtest.git || true
%cd Alpha-Engine-backtest
!git pull --ff-only
!python -m pip install -r requirements.txt
```

```python
# Profile setting: ROBUST is the default standard machine.
ENGINE_PROFILE = "ROBUST"      # "ROBUST" / "FERRARI" / "CUSTOM"
ALLOW_OVERDRIVE = False         # FERRARI requires True explicitly.
ALLOW_CUSTOM_PROFILE = False    # CUSTOM requires True explicitly.
```

```bash
!python alpha_engine_production.py --profile ROBUST --output-root /content/drive/MyDrive/alpha_engine_runs
# Overdrive/satellite only, not the standard machine:
# !python alpha_engine_production.py --profile FERRARI --allow-overdrive --output-root /content/drive/MyDrive/alpha_engine_runs
# Research-only custom profile:
# !python alpha_engine_production.py --profile CUSTOM --allow-custom-profile --custom-residual-ratio 60 --custom-portfolio-n 12 --output-root /content/drive/MyDrive/alpha_engine_runs
```

```bash
!find /content/drive/MyDrive/alpha_engine_runs -maxdepth 2 -name metadata.json -o -name run_report.md
```

```python
from pathlib import Path
latest = sorted(Path('/content/drive/MyDrive/alpha_engine_runs').glob('*'))[-1]
print(latest)
print((latest / 'run_report.md').read_text()[:4000])
```

Each run writes a timestamped folder containing `metadata.json`, `run_report.md`, selected tickers, adopted weights, renewal decisions, sell/extend/cash reasons, Git commit hash, data timestamps, and a copied or explicitly referenced price cache with `cache_metadata.json` when live data is used. FERRARI emits an explicit warning in console output and in the report because it is `overdrive_satellite_alpha`, not the ROBUST standard profile.

## Lightweight Live Screener（Residual Momentum）

`alpha_engine_live_screener.py` は、バックテストを再実行せず、直近約18か月の市場データだけで現在のResidual Momentumランキングを作成する実運用確認用の軽量スクリーナーです。TTL90 / Renew30、過去売買履歴、CAGR、Sharpe、MDD、年次成績、自動発注は実装していません。

### Colab 最小手順

1. Google Driveをmountします。

```python
from google.colab import drive
drive.mount('/content/drive')
```

2. repositoryをcloneまたはpullします。

```bash
%cd /content/drive/MyDrive
!git clone https://github.com/<your-org>/Alpha-Engine-backtest.git || true
%cd Alpha-Engine-backtest
!git pull --ff-only
```

3. dependenciesをinstallします。

```bash
!python -m pip install -r requirements.txt
```

4. `RESIDUAL_RATIO` と `TOTAL_HOLDINGS` を設定します。標準は Residual 60% / N12（US6・JP6）、フェラーリ設定は Residual 100% / N6（US3・JP3）です。

```python
RESIDUAL_RATIO = 60
TOTAL_HOLDINGS = 12
# Ferrari / satellite setting:
# RESIDUAL_RATIO = 100
# TOTAL_HOLDINGS = 6
```

5. Live Screenerを実行します。

```bash
!python alpha_engine_live_screener.py \
  --residual-ratio 60 \
  --total-holdings 12 \
  --output-root /content/drive/MyDrive/alpha_engine/live_screening_runs
```

6. `screen_report.md` と `selected_tickers.csv` を確認します。

```python
from pathlib import Path
latest = sorted(Path('/content/drive/MyDrive/alpha_engine/live_screening_runs').glob('*'))[-1]
print(latest)
print((latest / 'screen_report.md').read_text()[:4000])
```

各実行は日時付きフォルダに `ranked_candidates_us.csv`、`ranked_candidates_jp.csv`、`selected_tickers.csv`、`adopted_weights.csv`、`score_components.csv`、`data_quality.csv`、`download_failures.csv`、`metadata.json`、`screen_report.md` を保存します。日本株Residualのベンチマーク取得状態は `metadata.json` と `screen_report.md` に明示され、JPベンチマークが取得不能な場合は無言で代替せず停止します。
