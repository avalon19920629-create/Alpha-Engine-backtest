# Dow Structure Predictive Study v0.3

モメンタム・ボラティリティ・市場内順位を既知としたとき、確定スイング4状態が予測に追加する情報を固定入力で評価する独立研究です。主仮説は63観測区間の大幅下落、副仮説は順位悪化・3Mモメンタム負転換です。副目的で有利な結果が出ても主仮説の不支持を覆しません。

v0.2で強モメンタム群の構造分布差が見えた一方、安定した追加予測力は確認できていないという記述は依頼者の研究背景です。この実装作業で固定Driveデータの実績を再解析して確認したという意味ではありません。成功条件は有利な結果の発見ではなく、追加情報の有無・標本不足を同じ手順で判定できることです。

## 保護範囲と固定入力

起点は `1b2ecb4267d176b327d3995ed0bbc3d7a2b273f6`、専用ブランチは `codex/dow-structure-predictive-v0-3` です。既存αエンジン、Research Runner、v0.1/v0.2、既存ノート・依存関係・成果物を変更しません。価格取得、本番スクリーニング、発注、TTL、配分変更、既存スコアへの自動反映は行いません。

固定価格：

```text
/content/drive/MyDrive/alpha_engine/research_runs/
20261104T124443Z_residual100_n6_9b9d99b92dcf
```

固定v0.2分析：

```text
/content/drive/MyDrive/alpha_engine/analysis/dow_structure_v02/
20261111T010459Z_34a3f71bec76
```

v0.2の `daily_structure_features.parquet`、`landmark_observations.csv`、`structure_outcomes.csv`、`structure_outcome_summary.csv`、`extreme_transitions.csv`、メタデータ、マニフェストを確認します。保存価格のSHA-256とv0.2のマニフェストを照合し、終了時にも入力不変を確認します。固定入力が不足しても再ダウンロードや新規v0.2実行で補いません。

v0.2の分類可能な定点を再利用し、マッチした標本だけを学習母集団にしません。価格原本は目的変数の計算・品質照合用に読み直しますが、書き換えません。1306.Tの試験的補正、スイング確定、日付・将来評価は既存研究の定義を保持します。

## Colabでの実行

1. 専用ブランチの `Dow_Structure_Predictive_Study.ipynb` を開き、セル1でDriveを接続します。
2. セル2で固定入力、`CODE_REF`、起点SHA、保存先、反復数とseedを確認します。最初は `RUN_SENSITIVITIES=False` にします。
3. セル3で新しい `/content` チェックアウトを作り、参照をSHAへ固定します。起点の子孫であることと、起点の既存追跡ファイルが変更・削除・置換されていないことを確認します。既存チェックアウトへpull/resetしません。
4. セル4で特徴量・目的変数・逐次予測・独立再実行照合・指標計算を実行します。入力読込、特徴量、ラベル、日付ごとの予測、再実行監査、指標、終了状態をログで確認します。
5. セル5でModel A対Bの主比較と除外・予測不能を確認します。全候補、上位10%、US/JP、3M符号別を区別してください。
6. 主分析が正常完了してから必要に応じ `RUN_SENSITIVITIES=True` とし、セル6を実行します。副分析は別の新規フォルダに保存します。
7. セル7でGate I、監査、図、今回の主・副分析保存先を確認します。

GitコードとPython依存関係を取得しますが市場価格は取得しません。新しい `requirements-dow-predictive.txt` が既存研究依存を参照し、scikit-learn>=1.4,<2を追加します。既存requirementsは変更しません。

各実行は別Pythonプロセス、結果通知も毎回新規です。import失敗時も前回の成功フォルダを今回の結果に使いません。新しいCODE_REFへ切り替える場合はセル3から再実行します。保存先は原価格・v0.2入力の外側を指定してください。

## APIとCLI

主分析：

```bash
python -m pip install -r requirements-dow-predictive.txt
python dow_structure_predictive_study.py \
  --snapshot-dir /absolute/path/to/fixed-price-snapshot \
  --v02-dir /absolute/path/to/fixed-v02-analysis \
  --output-root /absolute/path/to/analysis/dow_predictive_v03 \
  --bootstrap-samples 500 \
  --seed 20261111
```

```python
from dow_structure_predictive_study import run_study, run_sensitivities

main_dir = run_study(
    snapshot_dir='/absolute/path/to/fixed-price-snapshot',
    v02_dir='/absolute/path/to/fixed-v02-analysis',
    output_root='/absolute/path/to/analysis/dow_predictive_v03',
    bootstrap_samples=500,
    seed=20261111,
)
# 主分析のcompleted・validation passedとマニフェストを再確認してから実行する。
sensitivity_dir = run_sensitivities(main_dir)
```

`run_sensitivities(main_run_dir, output_root=None)` は正常完了した主分析とその入力・マニフェストを検査し、主分析と並ぶ別の新規フォルダを作ります。完成済み主分析のファイルやGateを更新しません。主分析の独立ウォークフォワード再実行照合は自動で行います。単なる再実行一致と、真正な未使用データでの予測性能を混同しないでください。

副段階のCLIは次です。seedと反復数は主分析に保存した値を継承し、副段階で都合よく再設定しません。

```bash
python dow_structure_predictive_study.py \
  --sensitivity-of /absolute/path/to/completed-main-run
# 必要なら別の副分析保存ルートを --output-root で指定する。
```

完成判定は終了コードやAPIの保存先返却だけではなく、メタデータの `status=completed` かつ `validation_status=passed` で行います。目的変数・モデル計算に一つでも実際の失敗があれば `status=partial`、検査failedとし、Gateを `evaluation_unavailable` にします。他の正常な目的の診断値は保存しますが、全分析の正常完了と偽装しません。訓練不足・負転換のリスク対象外・右側打切りは想定された状態で、それだけでpartialにはしません。

## 観測母集団と特徴量

主系列は調整後株価の四状態です。相対価格構造は別の感度分析で、同一銘柄の観測を独立標本のようにプールしません。定点はv0.2と同じ市場ベンチマーク21観測日間隔・位相0、SPAN5です。

学習母集団は必要履歴・五つの数値特徴・確定構造が揃う全候補です。未マッチの銘柄も含みます。上位10%研究群は実際のαエンジン選定銘柄そのものではなく、候補を近似する研究集団です。学習は全候補、評価は全候補と上位群を分けます。

数値特徴は共通有効日上の3/6/12M株価端点モメンタム、252リターン年率ボラティリティ、同日・市場内の生12M順位パーセンタイルです。価格窓63/126/252は端点間62/125/251区間で、253価格観測値が252リターンに必要です。既存エンジンの市場横断カレンダー・前方補完方式と混同しません。

v0.3では有限なゼロボラティリティも既知の回帰入力として受け入れ、負値は不適格とします。五つの数値特徴はすべて有限である必要があり、訓練中に一定の特徴量はStandardScalerのscale=1で処理します。既存v0.1/v0.2の適格条件は変更しません。

順位は必要履歴と有限な12M値を持つ当日の全有効候補から再計算し、構造分類不能を順位母集団から除いて順位を作り直しません。12M降順、同値ティッカー順で、1位を0・最下位を1とする `(rank-1)/(N-1)` を使います。N<2は未定義・特徴不適格です。生12M上位10%は `ceil(0.1*N)` 位以内かつ生12Mプラスで、v0.2と同じ時点別定義です。最終日の順位を過去へ適用しません。

## 目的変数と未観測理由

| 対象 | 定義・終了時点 |
| --- | --- |
| `deep_drawdown_63`（主） | 次の株式／ベンチマーク共通有効日終値から63区間後までの、累積高値比による最大DD≤−20% |
| 12M順位変化（副） | 次の予定市場定点での順位パーセンタイル−現在値。悪化がプラス |
| 3M負転換（副） | 現在3M>0の銘柄で、次の予定市場定点の3M≤0 |
| 63日相対リターン（補助） | 主目的と同じ開始・終了、`(1+R_stock)/(1+R_benchmark)-1` |

63日評価はv0.2と同じ、当日tの次の共通有効終値で開始し、そこから63観測区間後を終了とします。当日終値で事前に売買できたことにはしません。

数値境界の丸めだけを吸収する固定許容差 `drawdown_boundary_atol=1e-12` を使い、−20%を含む判定を安定させます。Gateの3%以上・Bが単純基準に劣後しないという包含比較も `inclusive_comparison_atol=1e-12` です。閾値の最適化ではなく、区間下限が厳密に0を上回る条件などを有利に変更するものでもありません。

順位変化と負転換の終了は**次の予定された市場定点**です。個別株で21共通観測値を数えて終了日をずらしません。未来の波形が分類できることをラベルの条件にしません。現在3M≤0は `not_at_risk_nonpositive` として、負転換の0（陰性例）に混ぜません。現在値を定義できない場合も `invalid_current_momentum` と分けます。

ラベルごとに `observed`、`right_censored`、`price_missing`、`known_anomaly_influence`、`computation_failed` などの理由を保持します。既知異常の影響フラグと、影響除外の感度分析で対象値を未観測とすることを区別します。計算失敗を0や正常観測に置換せず、ある目的変数の失敗で他の正常結果を偽装しません。将来実績が未知でも、特徴が適格で成熟訓練集合が十分ならその時点の予測を保存し、評価可能になった行だけで指標を計算します。

1306.T補正は公式調整済み価格との完全照合未了の試験的修正です。既知異常に関わる観測・目的変数の扱いを品質監査へ残します。原本Parquetは不変です。

## Model 0・A・Bの公平な比較

Model 0は同じ成熟訓練集合の市場別の既往イベント発生率を基準予測とします。Model Aは五つの数値特徴とJPダミー（USを基準）、Model BはそれにHH/HL基準の残り三つの構造ダミーだけを加えます。鮮度・交互作用・追加の探索特徴を入れません。

数値五特徴だけを同じ訓練データで適合したStandardScalerで変換します。市場・構造ダミーを同じ数値スケーラーへ混ぜません。欠損値補完はせず、共通の特徴不適格行をA/B両方から同じように除きます。A/Bで訓練行・評価対象IDを完全一致させ、その集合とハッシュを監査へ保存します。Model 0も比較可能な同じ対象で評価します。

分類はL2ロジスティック回帰、C=1、lbfgs、max_iter=1000、class_weight=Noneです。順位変化・相対リターンはRidge、alpha=1です。ランダム分割、全期間スケーリング、特徴選択、結果に応じた正則化変更はしません。

## 拡大型ウォークフォワードと学習条件

評価日tの学習集合は、**特徴日<tかつ実際のラベル終了日<t** の観測だけです。t当日に終わる結果は使いません。63日ラベルがまだ確定しない過去行も使いません。学習末特徴日・最終ラベル確定日・件数・予測対象を日付ごとに記録します。

固定した開始条件は、成熟訓練集合でプールしたユニーク過去日が12以上、さらに予測する市場でも成熟した過去市場定点が12以上です。二値目的はプールした陽性・陰性各30以上、数値目的は成熟行60以上も必要です。これは本実装の保守的な固定規則で、良い結果を得るため後から緩めません。不足期間は `insufficient_training_data` として残し、無理な適合や定数を正常モデルとして偽装しません。

予測を評価実績と比較する前に保存します。主分析では同じ固定入力・seedでウォークフォワードを独立に再実行し、予測と監査の再現性を照合します。これは処理の再現性であり、同じ2024〜2026年が未使用アウトオブサンプルに変わるわけではありません。

## 指標と不確実性

主指標は同じ評価IDでのBrier Scoreです。改善率は `(Brier_A-Brier_B)/Brier_A` で、高いほど良く、AのBrierが0など改善率を定義できない場合を隠しません。Model 0も同じ評価集合で比較します。

補助的にLog Loss、ROC-AUC、PR-AUC、較正、イベント率、検出漏れを保存します。副目的ではMAE・RMSE・順位相関、負転換Brier、実績事象率などを別々に報告します。主・副指標を単一の都合のよいスコアへ合算しません。単純的中率だけで判断しません。

予測損失差を同じ市場・暦月のブロックでまとめて再標本化する分析を主な探索的不確実性評価とし、銘柄クラスタ区間は別の診断にします。市場月ブロック数と、Gateに必要な**ユニーク暦月数**を混同しません。同じ月のUS/JPはGateの評価月として2カ月と数えません。較正済み二方向クラスタ推論や独立標本の厳密な有意差検定とは呼びません。

## Gate Iの読み方

主判断は上位10%群の63日大幅下落予測です。Aを認めるには、BのAに対するBrier相対改善が3%以上、BがModel 0に劣後しない、時系列の前半・後半でも改善方向がプラス、主不確実性区間の下限が0より大きいことを確認します。閾値は事前固定で、最良の副目的や感度設定から選びません。

前半・後半はプールした主評価対象のユニーク予測日を時系列順に分け、最初の `ceil(日付数/2)` 日と残りを使います。同じ日の行を分断せず、US/JPにも同じ境界日を適用します。必要な評価条件を満たす場合、本実装では3%に届かなければC、3%に達しても単純基準・両半期の方向・区間条件のいずれかが足りなければBです。

主評価で陽性30件未満、または有効なユニーク暦月が8未満なら `evaluation_unavailable` です。標本不足をC（研究凍結）や肯定的なAへ置き換えません。予測不能や比較対象不足も理由付きで記録します。

- A：工学的検証候補。主条件・安定性・標本・監査を満たす研究上の候補で、本番導入許可ではありません。
- B：条件付き保留。方向は示唆されても、市場・期間・不確実性などに確証が足りません。
- C：研究凍結。評価可能な範囲で追加価値がほぼない、または不安定な場合です。
- `evaluation_unavailable`：標本や評価条件が足りず、肯定・否定を確定しません。

US/JPを個別に報告します。プール全体がA条件を満たしても、どちらの市場もAでなければ総合Bとします。プール条件に加えて片市場だけがAなら `scope=market_limited` とし、その市場だけを `applicable_markets` に表示します。個別市場がAでもプール主判定がAでなければ、総合判定をAへ昇格しません。主仮説が支持されないとき、副仮説だけでGate Aにせず、独立した確認研究を要求します。

原本不変・先読み防止・A/B入力一致・予測保存と結果照合・独立再実行一致・実行時の検査をGate D監査として確認します。コードの自動テスト結果と、個別実行でのランタイム監査の範囲は区別します。いずれも未来の本番成績を保証しません。

Gate Dの実行時検査は、実際に保存した次の六つの真偽値です。単に正常終了したことから一括でtrueとはみなしません。

| キー | 確認範囲 |
| --- | --- |
| `input_unchanged` | 固定入力の終了時ハッシュ不変 |
| `protected_unchanged` | 既存保護対象ソースの不変 |
| `training_causal` | 訓練特徴日・ラベル確定日が予測日より前 |
| `input_parity` | Model A/Bの訓練・評価対象の一致 |
| `replay_verified` | 独立再実行による予測一致 |
| `predictions_saved` | 評価前の予測保存と結果照合 |

これはその実行の監査であり、Colabがリポジトリの全pytestを再実行したという表示ではありません。

## 感度分析は独立した第2段階

主分析の正常完了後、SPAN3/10、位相7/14、相対価格構造、日本ベンチマーク補正感度を、各設定で同じA/B標本・訓練規則により再評価します。系列は別に扱い、独立標本数を水増ししません。

副分析の新規フォルダから主分析・原入力を参照します。完成した主分析のマニフェストやGateは書き換えず、最良設定の結果を主結果へ昇格しません。実行しなければ「未実行」であり、主分析の成功が感度分析の成功を意味することはありません。

主分析の `sensitivity_analysis.csv` は `status=not_run` と別API実行の案内を持つプレースホルダーです。副分析を実行しても、原本不変のため主分析側のこの案内は書き換えません。実際の副分析状態は、新しい兄弟フォルダのメタデータと集計表で確認します。

七つの設定（span3、span10、phase7、phase14、relative_structure、uncorrected、exclude_anomaly_influence）ごとに特徴・目的・予測・訓練監査・主副指標を保存します。副分析では設定ごとの独立再実行を再度行わないため、各設定のGateは `evaluation_unavailable` であり主分析承認ではありません。副段階全体のcompletedは、全設定の処理・入力保護が正常だったという意味で、Gate Aを得たことではありません。

## 保存成果物

```text
analysis/dow_predictive_v03/<main_experiment_id>/
  study_metadata.json
  study_report.md
  file_manifest.json
  feature_dataset.parquet
  target_dataset.parquet
  walkforward_schedule.csv
  training_audit.csv
  baseline_predictions.csv
  structure_predictions.csv
  paired_prediction_comparison.csv
  risk_metrics.csv
  momentum_persistence_metrics.csv
  subgroup_metrics.csv
  uncertainty.csv
  calibration.csv
  time_half_metrics.csv
  top10_retention.csv
  target_status_counts.csv
  correction_log.csv
  sensitivity_analysis.csv
  quality_warnings.csv
  gate1_assessment.json
  gate1_assessment.md
  charts/

analysis/dow_predictive_v03/<sensitivity_experiment_id>/
  study_metadata.json
  study_report.md
  sensitivity_analysis.csv
  file_manifest.json
  <specification>/
    feature_dataset.parquet
    target_dataset.parquet
    baseline_predictions.csv
    structure_predictions.csv
    training_audit.csv
    ...主副指標・設定別Gate・マニフェスト...
```

特徴・目的・訓練監査・予測・結果照合を観測IDと日付で追えます。評価日、銘柄、市場、時点情報、学習末日、実際の将来終了日、実績、予測、予測不能理由を確認してください。主分析と副分析の保存先はNotebookで別々に表示します。

`target_status_counts.csv` と `quality_warnings.csv` の `target_error` で計算失敗も追跡できます。保存予測CSVの読み戻しは数値許容差1e−12で照合し、独立再実行では予測・訓練監査・日程を比較します。副段階の集約マニフェストには設定別マニフェストも含み、マニフェスト自身だけを除外します。

## 検証手順と研究限界

```bash
python -m pip install -r requirements-dow-predictive.txt pytest
python -m pytest -q tests/test_dow_structure_predictive_study.py
```

追加テストは合成データで先読み、成熟ラベル・訓練のみの標準化、A/B行一致、B追加特徴、時点順位、未観測・負転換リスク群、二系列分離、原本・過去研究不変、再現性、Gate標本不足、部分失敗、上書き防止を確認します。

既存テストには作業ディレクトリ相対の追跡済みレポートを書くものがあるため、全体は一時コピーで実行します。Research RunnerのGit参照確認に元のGitメタデータを読み取り参照します。

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

現在ユニバースの過去適用による生存者バイアス、v0.1/v0.2で探索済みの2024〜2026年、研究上位群と実採用群の違い、63日区間重複、銘柄・市場日付依存、短い期間とJP標本不足、1306.T公式価格照合未了、簡易終値スイングという制約があります。波形の優位性や投資戦略改善が確立したとは主張しません。

固定Drive入力がクラウドにない場合、合成データテストとColabノートの完成までを検証範囲とし、実データの指標・Gateを計算したとは報告しません。Colabで入力ハッシュ、成熟条件、対象件数、モデル比較、監査・Gateを確認してください。v0.4へ進む場合も工学的バックテストと真正な未使用データ検証は別で、本番採否は別途監査します。
