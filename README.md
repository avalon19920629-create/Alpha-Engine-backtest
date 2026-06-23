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

### Order Plannerの整数株丸め最適化

`alpha_engine_order_planner.py` は、Live Screenerが出力した採用銘柄、順位、Residual Score、Weight、目標金額を変更せず、証券会社画面で人間が確認して手動発注するための買付数量案だけを作ります。自動発注、証券API接続、注文送信は実装していません。

整数株または指定注文単位への変換は、次の2モードを選べます。

- `floor`: 各銘柄の目標金額をバッファ込み価格と注文単位で単純に切り捨てます。従来挙動の監査・比較用です。
- `target_tracking`: デフォルトかつ実運用推奨です。バッファ込み価格で利用可能買付資金を超えないことを守りながら、追加の1注文単位ごとに実効Weightと目標Weightの二乗誤差が最も小さくなる銘柄へ決定論的に配分します。

主な設定とCLIは以下です。

```python
ROUNDING_MODE = "target_tracking"   # "floor" or "target_tracking"
MIN_ONE_UNIT_PER_SELECTED = True
MAX_RESIDUAL_CASH_PCT_WARNING = 0.03
```

```bash
python alpha_engine_order_planner.py \
  --screen-run-dir /path/to/live_screening_run \
  --rounding-mode target_tracking \
  --min-one-unit-per-selected \
  --max-residual-cash-pct-warning 0.03
```

`MIN_ONE_UNIT_PER_SELECTED=True` の場合、全採用銘柄について最低1注文単位を買えるか最初に判定します。買える場合は全銘柄へ最低1単位を確保してから追加配分します。買えない場合は一部銘柄を無言で0株にせず、資金追加、`TOTAL_HOLDINGS`削減、端株利用、または `--no-min-one-unit-per-selected` による最低単位強制の解除を求める明示エラーで停止します。

予算が小さすぎる場合は、次のいずれかを検討してください。

- 投入資金を増やす。
- `TOTAL_HOLDINGS` を減らして採用銘柄数を絞る。
- 証券会社が対応している場合は端株・単元未満株を別途検討する。
- 監査目的でのみ `--rounding-mode floor` または `--no-min-one-unit-per-selected` を使い、0株銘柄の警告を確認する。

`buy_order_plan.csv`、`buy_order_summary.csv`、`buy_order_report.md` には、単純切り捨て数量、最適化後数量、実効Weight、Weight乖離、未投資資金、最適化ステップ数、残余現金警告が出力されます。発注直前には証券会社画面で最新価格、注文単位、手数料、税金、為替、必要資金を必ず再確認してください。

## Alpha Engine Live TTL Manager

`alpha_engine_ttl_manager.py` は、実際に約定した保有銘柄だけを対象に TTL90 / Renew30 / Composite 判定を行う実運用補助ツールです。自動売買システムではなく、証券会社 API、口座連携、自動発注、注文送信は実装していません。TTL Manager は過去の `buy_order_plan.csv` を唯一の事実源にせず、ユーザーが証券会社画面で確認した `live_holdings_ledger.csv` を唯一の事実源として扱います。

### 1. 初回保有台帳作成

まず Order Planner の注文案から、実約定入力用テンプレートだけを作成します。この段階では `planned_shares` を実保有数量として採用しません。

```bash
python alpha_engine_ttl_manager.py \
  --create-fill-template \
  --order-plan-dir "/path/to/order_plan_run" \
  --output-root "artifacts/ttl_manager_runs"
```

生成された `actual_fills_template.csv` に、ユーザーが証券会社の約定画面で確認した `actual_shares`、`actual_entry_date`、`actual_entry_price_local` を入力します。入力済みファイルから初期台帳を作成します。

```bash
python alpha_engine_ttl_manager.py \
  --initialize-ledger \
  --actual-fills "/path/to/actual_fills_confirmed.csv" \
  --ledger-path "/path/to/live_holdings_ledger.csv"
```

`actual_shares`、`actual_entry_date`、`actual_entry_price_local` が欠損している場合、TTL Manager は安全停止します。注文案だけから実保有台帳を無言で確定することはありません。

### 2. Day90 TTL 判定

通常の TTL 判定は次のように実行します。

```bash
python alpha_engine_ttl_manager.py \
  --ledger-path "/path/to/live_holdings_ledger.csv" \
  --output-root "artifacts/ttl_manager_runs"
```

デフォルトは `TTL_DAYS=90`、`RENEWAL_DAYS=30`、`WEEKLY_REVIEW_DAY=FRI`、`DATA_LOOKBACK_MONTHS=18` です。Day89 以前の `ACTIVE` 銘柄には売却・延命判定を出さず、Day90 到達銘柄だけ既存監査済みロジックに基づく Composite 判定を行います。Composite は既存バックテスト実装の Rank Buffer、Residual、50DMA 条件を再利用し、3 条件中 2 条件以上を合格とします。

Composite 合格時は `RENEWED` として最大 30 日延命し、売却注文案は出しません。Composite 不合格時は `SELL_PENDING` として `sell_order_plan.csv` に手動売却注文案を出力します。売却した銘柄は即時補充せず、現金化して次回再選定まで空席を維持します。

### 3. 延命中の週次判定

`RENEWED` 銘柄は Day91–Day120 の延命期間中、原則として毎週金曜日の各市場終値確定後に TTL Manager を再実行して監視します。出力には `decision_as_of_date`、`decision_market` 相当の市場注記、`sell_not_before_date` が含まれます。延命中に Composite 失格となった場合は `RENEWAL_FAILED_SELL_PENDING` とし、手動売却注文案を出します。ただし自動発注は行わず、同じ終値で売れたものとして扱いません。

### 4. Day120 再選定

Day120 到達時は二度目の Renew30 を認めず、`RECONSTITUTION_REQUIRED` を出力します。新しい上位 N 銘柄への再選定と注文計画は TTL Manager では行わず、既存の Live Screener と Order Planner を使います。

```bash
python alpha_engine_live_screener.py --output-root artifacts/live_screening_runs
python alpha_engine_order_planner.py --help
```

旧保有銘柄が新しい上位 N に再び含まれる場合は、新サイクルで再選定されたものとして扱います。不必要な売却→再購入は強制しませんが、連続 Renew30 としては記録しません。

### 5. 実約定後の台帳更新

TTL Manager は初期状態では `live_holdings_ledger.csv` を上書きせず、`live_holdings_ledger_proposed.csv` だけを出力します。売却・買付が実際に完了した後、ユーザーが約定内容を確認したファイルを用意してから明示的に更新してください。

```bash
python alpha_engine_ttl_manager.py \
  --apply-execution-confirmation \
  --ledger-path "/path/to/live_holdings_ledger.csv" \
  --execution-confirmation "/path/to/execution_confirmation.csv"
```

安全のため、売却未約定・一部約定・注文取消がある場合に TTL Manager が実保有数量を勝手に 0 にすることはありません。実運用では必ず証券会社画面で株数、価格、約定状況を確認してから台帳へ反映してください。

各実行は日時付きフォルダに `ttl_review_decisions.csv`、`renewal_decisions.csv`、`sell_order_plan.csv`、`reconstitution_required.csv`、`live_holdings_ledger_proposed.csv`、`data_quality.csv`、`download_failures.csv`、`metadata.json`、`ttl_review_report.md` を保存します。データ不足、ベンチマーク取得不能、JP ベンチマークの無言代替が必要になる状況、Composite 構成要素の計算不能、未来日付、不正な台帳、既に売却待ちの銘柄の二重売却処理は `DATA_BLOCKED — no trading recommendation generated.` として安全停止します。
