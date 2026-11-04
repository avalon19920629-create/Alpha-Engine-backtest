# Alpha Engine Research Operations v1.0

既存Alpha Engineのスコア・順位・採用銘柄・ウェイトを変更せず、スクリーニングに使った日次調整後価格と結果を保存する研究用の実行機構です。データ取得・保存・検証だけを追加します。既存の運用ノート、本番実行履歴、エンジン本体は変更しません。

このランナーはLive Screenerを使用します。バックテスト、TTL判定、証券API接続、自動注文、売却、資産配分変更、追加の選定ロジックは実行しません。

## 最初に使う方法

1. 専用ブランチ `codex/alpha-engine-research-operations-v1` の `Alpha_Engine_Research_Operations.ipynb` をGoogle Colabで開きます。
2. セル1でGoogle Driveを接続します。
3. セル2にGit参照と研究設定を集約しています。`CODE_REF` は研究用ランナーの参照、`SOURCE_REF` は既存エンジン本体の参照です。初期SOURCE_REFは `075f0985c76c2212552732534d7f3c741270a80d`、初期設定はResidual 100・N6（US3・JP3）です。
4. セル3で新しい `/content` 作業ディレクトリへcloneし、両参照をSHAへ解決して表示します。研究コードは解決したコミットへdetachし、以後pullや参照更新をしません。研究専用の依存関係を導入します。
5. セル4で研究用ランナーを起動します。セル5で実行ステータス、検査レポート、キャッシュハッシュ、採用銘柄を確認します。
6. セル6に表示されたGoogle Driveの保存先を開きます。正常完了以外のフォルダは、完成済みデータセットとして扱いません。

同じチェックアウトで再実行する場合はセル4以降を実行できます。取得処理からやり直し、別の実行フォルダを作成します。CODE_REFを変える場合はセル3も再実行してください。新しいチェックアウトを作るため、既存チェックアウトの作業内容をreset・上書きしません。ランナーは別Pythonプロセスで起動するので、以前のコードのimportも流用しません。

ノートの新規ブランチをGitHubから取得できること、Google Driveへ書き込めること、PyPIと価格データ配信元へ接続できることが必要です。GitHub資格情報をノートや保存ファイルへ書かないでください。

## 設定とCLI

| 項目 | 初期値・意味 |
| --- | --- |
| `SOURCE_REF` / `--source-ref` | 必須。既存エンジンのGit参照。実行開始時にコミットSHAへ固定 |
| `expected_source_sha` / `--expected-source-sha` | Notebookが事前に解決したSHA。参照が変わっていれば停止 |
| `RESIDUAL_RATIO` / `--residual-ratio` | 100。60も既存スクリーナーのまま使用可能 |
| `TOTAL_HOLDINGS` / `--total-holdings` | 6（US3・JP3）。12ならUS6・JP6 |
| `LOOKBACK_MONTHS` / `--lookback-months` | 18。現行スクリーナーの固定値と一致させる |
| `RESEARCH_OUTPUT_ROOT` / `--output-root` | `/content/drive/MyDrive/alpha_engine/research_runs` |
| `CACHE_POLICY` | `new_snapshot` のみ。既存キャッシュの暗黙再利用はしない |
| `RUN_MODE` | `research_only` のみ |

ローカル環境では、リポジトリの専用ブランチ上で次を実行します。

```bash
python -m pip install -r requirements-research.txt
python alpha_engine_research_runner.py \
  --source-ref 075f0985c76c2212552732534d7f3c741270a80d \
  --residual-ratio 100 \
  --total-holdings 6 \
  --output-root /absolute/path/to/alpha_engine/research_runs
```

Residual 60・N12では `--residual-ratio 60 --total-holdings 12` に変更します。既存の `requirements.txt` は変更せず、Parquet保存用のpyarrowを研究専用依存関係へ追加します。

SOURCE_REFは現在読み込むエンジンのコードと一致する必要があります。ランナーは指定参照をSHAへ解決して既存エンジンのファイル内容を確認します。別コミットのエンジンへ無断で切り替えたり、最新ブランチへpullしたりはしません。研究用の追加ファイルがあるため、研究ランナーのHEADと比較対象のエンジンコミットが異なることは許容します。対象エンジン本体が違えば停止します。

## 実装前の確認結果と設計判断

実装開始時のHEADは比較基準の `075f0985c76c2212552732534d7f3c741270a80d` と同一でした。リポジトリの既存Colab起動手順はREADMEにあり、`Alpha_Engine_Operations.ipynb` はこのチェックアウトには存在しません。既存ファイルへの変更は行わず、新しい研究ノートを追加します。

- `alpha_engine_backtest.get_live_universe()` と `normalize_yfinance_ticker()` で既存と同じUS/JPユニバースを取得・正規化します。
- `build_benchmark_modes()['broad_default']` から既存Live Screenerと同じベンチマーク候補を使います。
- `download_live_universe_prices()` が返す調整後終値、取得失敗情報、要求銘柄を利用します。1回の研究実行でこの取得結果を共用します。内部のバッチ取得や配信元リトライは既存関数の処理です。
- `run_live_screening(prices=..., us=..., jp=...)` をそのまま呼び出すため、スコア算式・順位・ウェイトを複製しません。`build_live_variant()`、`_score_region()`、`_benchmark_status()` の既存の計算経路を利用します。
- 既存 `build_price_cache()` / `validate_price_cache()` は日付引数や銘柄集合、ファイルが読めるかを中心に確認します。研究用の地域別ユニバース、厳密なParquet往復一致、ハッシュ、品質監査には不十分なため、新しい保存・検証処理を外側へ追加します。

現行Live ScreenerのLOOKBACK_MONTHSは18か月の固定値で、引数では変更できません。ランナーはこの値に合わせます。別の値を実現するためのモンキーパッチや既存コードの再実装は行いません。

## 価格期間と保存内容

取得の指定終了日は既存スクリーナーと同じUTCの日付です。指定開始日は18か月前ですが、既存 `download_live_universe_prices()` はさらに約400営業日前へ取得開始を広げ、終了日を含めるため翌日を取得APIの上限に指定します。**指定期間と実際の観測期間は異なります。** メタデータでは指定開始・終了日、実観測開始日・最終観測日を区別します。

価格種別は、yfinanceの `auto_adjust=True` で調整された `Close` です。未調整終値や出来高ではありません。取得元のDataFrameをそのまま保存し、休日由来のNaNを前方補完して原本を変更しません。既存エンジンが内部で行う補完は従来どおりです。

`price_cache.parquet` はUS/JP株式、`benchmark_prices.parquet` はベンチマークを日付行・ティッカー列の形式で保持します。取得した列の順序、日付インデックス、型、値、NaN位置を保存・再読込で照合します。銘柄集合だけの照合ではありません。重複日、重複列、非数値、ゼロ以下、無限大、スクリーニング時点後の観測を検出し、不正なデータを修復して成功扱いしません。

ベンチマークは存在だけでなく、実際に使用する系列が必要な履歴を持つことを確認します。欠損系列を使って残差スコアが無言で中立値となるケースを検証済みデータとして扱いません。株式の取得失敗・履歴不足は銘柄別に記録し、必要候補数を満たさない場合は停止します。

## 実行フォルダ

```text
research_runs/
  <UTC時刻>_residual100_n6.../
    engine_outputs/
      ranked_candidates_us.csv
      ranked_candidates_jp.csv
      selected_tickers.csv
      adopted_weights.csv
      score_components.csv
      data_quality.csv
      download_failures.csv
      metadata.json
      screen_report.md
    price_data/
      price_cache.parquet
      benchmark_prices.parquet
      price_cache_metadata.json
      requested_universe.csv
      research_download_failures.csv
      price_data_quality.csv
    audit/
      validation_report.json
      validation_report.md
      file_manifest.json
    research_run_metadata.json
```

各実行は独立した新規フォルダです。既存実行履歴を削除・上書きしません。途中失敗でも、作成済みのフォルダや部分成果物は調査用に残します。実行中・失敗・未完了を成功と混同しないよう、`research_run_metadata.json` と検査レポートを確認してください。

正常完了は `status == "completed"` かつ `validation_status == "passed"` です。実行時刻は `run_started_at` / `run_finished_at`、エンジンSHAは `git_commit_sha`、元の指定参照は `source_ref`、価格観測範囲は `observed_start` / `observed_end`、価格キャッシュのハッシュは `price_cache_sha256` にあります。

既存エンジンが作成した9種類の成果物は、内容を書き換えず `engine_outputs` へ移します。このため、エンジンの `metadata.json` やレポート内に記録された元の出力パスと、研究フォルダ内の実際の保存先が異なる場合があります。研究用メタデータと監査注記に移動元・実保存先を記録し、エンジン側のメタデータを研究用に書き換えません。

## 各研究用ファイルの意味

| ファイル | 内容 |
| --- | --- |
| `price_cache_metadata.json` | 指定期間と実観測期間、価格種別、取得元、auto_adjust、保存列・型、取得日時、価格ファイルのSHA-256など |
| `requested_universe.csv` | 要求したUS/JP株式ユニバースと地域。取得成功銘柄だけの一覧ではない |
| `research_download_failures.csv` | 研究用の実際の取得失敗。既存ダウンローダーから返された情報を保持 |
| `price_data_quality.csv` | 銘柄別の取得状況、観測数、履歴不足、異常、除外を説明する品質記録 |
| `research_run_metadata.json` | 実行開始・終了UTC、モード、設定、指定Git参照・SHA、銘柄数、成功・不足・除外数、使用ベンチマーク、Python/pandas/yfinance/pyarrow版、ハッシュ、実行・検査ステータス、保存先 |
| `validation_report.json` / `.md` | データ、スクリーン結果、回帰比較の検査結果と注意事項 |
| `file_manifest.json` | 保存成果物のファイルハッシュ。独立した保管・再検証の参照情報 |

**取得失敗の正しい情報は `research_download_failures.csv` と `price_data_quality.csv` を確認してください。** 既存スクリーナーに `prices` を渡すとダウンロード処理が省略されるため、元の `engine_outputs/download_failures.csv` は空の場合があります。これは実際の取得失敗がゼロだったことを意味しません。研究用ランナーは既存CSVを変更せず、この差を監査レポートにも記録します。

## 必須検査

1. Parquetを読み戻し、保存前のDataFrameと日付・列順・型・値・NaN位置が一致することを確認します。キャッシュ原本の補完・ソート・値修正で差を隠しません。
2. ベンチマークの所要履歴、株式候補数・除外理由、観測がスクリーニング時点以前であることを検査します。
3. US/JPの採用数、ウェイト合計1、地域ランキングの採用対象、スコア構成値と選定結果の整合を確認します。
4. 保存前の価格による既存スクリーナー実行と、再読込した同じ価格・ユニバース・パラメータによる実行を比較します。ランキング、採用銘柄、ウェイト、スコアを比較し、実行日時・実行時間・出力フォルダ名は比較から除外します。数値は浮動小数点誤差を許容しますが、順位・銘柄の違いは許容しません。
5. 実行後も入力価格が変更されていないこと、成果物と検査結果が揃っていることを確認します。検査失敗を成功扱いしません。

既存エンジンのソースがSOURCE_REFと同一であることは、過去の実行結果自体が再現できることを意味しません。過去当日の価格キャッシュとユニバースが別途必要です。後日の再取得価格による結果を、当時のスクリーニングの再現成功とは記載しないでください。

保存価格を取得し直さずに読み戻すには、同じ研究コードの環境で次を使います。価格ファイルのSHA-256を検証し、株式・ベンチマークを結合して元の列順で返します。読み戻しは新たなスクリーニングや注文を実行しません。

```python
from alpha_engine_research_runner import load_research_prices

prices = load_research_prices('/absolute/path/to/research_runs/<run-folder>')
print(prices.shape, prices.index.min(), prices.index.max())
```

ハッシュ照合は、保存後の破損や不一致を検出するものです。価格ファイルとメタデータ全体を第三者が書き換えた場合の真正性を証明する署名ではありません。

## 自動テストと検証範囲

研究用テストはネットワークを使わず、モック価格・取得失敗情報を使って保存・再読込、NaN、履歴不足、ベンチマーク欠損、重複・非数値、R100/N6・R60/N12、既存結果との一致、途中失敗、上書き防止を確認します。

```bash
python -m pip install -r requirements-research.txt pytest
python -m pytest -q tests/test_alpha_engine_research_runner.py
```

既存テストには作業ディレクトリ相対の `reports/*.md` を書くものがあるため、全体テストは一時コピーで実行してください。次はリポジトリのルートから実行する例です。既存レポートを変更せず、研究ランナーのSOURCE_REF検証に必要なGitオブジェクトは元のチェックアウトから読み取ります。Gitメタデータへのリンクは読み取り用であり、テスト用コピーでGitの更新操作はしません。

```bash
RESEARCH_TEST_REPO="$PWD"
RESEARCH_TEST_DIR=$(mktemp -d)
cp "$RESEARCH_TEST_REPO"/*.py "$RESEARCH_TEST_DIR/"
cp -R "$RESEARCH_TEST_REPO/tests" "$RESEARCH_TEST_REPO/config" "$RESEARCH_TEST_DIR/"
ln -s "$(git rev-parse --absolute-git-dir)" "$RESEARCH_TEST_DIR/.git"
RESEARCH_TEST_STATUS=0
(cd "$RESEARCH_TEST_DIR" && python -m pytest -q) || RESEARCH_TEST_STATUS=$?
rm -rf "$RESEARCH_TEST_DIR"
test "$RESEARCH_TEST_STATUS" -eq 0
```

オンボーディング時のヘルパーを使う場合も、この研究ランナーの追加後はGit参照を検証できる一時コピーであることを確認してください。Gitメタデータを含まない単純なコードコピーではSOURCE_REFの確認ができません。

オフライン自動テストの合格と、Google Colab実機でのDrive保存・ネットワーク接続・大規模ユニバース取得の成功は別です。このノートのColab実機・外部データ動作をオフラインテストのみで検証済みとはしません。最初のColab実行では、表示されたSHA、保存先、検査結果、株式・ベンチマークのキャッシュを確認してください。クラウド環境の通信許可はColabの通信可否とは別です。

## 研究上の限界

現在のUS/JPユニバースを取得するため、過去の任意時点のポイントインタイム・ユニバースや上場廃止銘柄を完全には復元しません。現在取得した過去の調整後価格も、過去当時の情報そのものではありません。配信元の訂正や企業行動で同じ期間の価格が後日変わることがあります。取得日時とハッシュを持つ各スナップショットを保持し、再ダウンロードを同一データとみなしません。

将来のバックテストでは生存者バイアス、企業行動、先読み、銘柄変更、欠損を別途監査してください。β推定、純粋α、波形解析、移動平均判定、モメンタム加減速分類、新規最適化、既存順位補正はv1.0の対象外です。
