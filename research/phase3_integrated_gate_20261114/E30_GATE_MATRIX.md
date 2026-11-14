# E30合格基準と証拠の対応表

2026-11-14。基準S0 §7のE30合格文言：**「入力ハッシュ一致、旧系列を旧仕様で再現、同点・時間軸・企業行動・将来切断テスト成功」**。不一致/漏洩は不合格、原票・依存・重要データ不足は判定不能。同§5.2の履歴・ID・企業行動台帳を併せて適用。S1の数値絶対1e-10、順位/集合100%を変更しない。

「合格」は表に記した検査範囲の合格であって、E30全体合格ではない。限定合格と判定不能が同居する項目は分けて記す。CとDを新しいE30義務として加算しない。

| 項目/区分 | 要求基準と所在 | 実施済み検証・証拠 | 判定 | 未解決リスク | 最小追加作業 |
|---|---|---|---|---|---|
| 1 選定ロジック A | S0§7、S1旧仕様再現 | S2選定1,026行/指標一致、S5旧228地域再現、T独立ソート | 旧再現合格／新選定認証未達 | R0複合とM0通常モメンタムは別 | ID/規則固定後に同一入力で再試験 |
| 2 欠損・停止 A/B | S0§5.2無制限補完禁止、未解決終端遮断 | S3欠損分類/2×2、S4営業日欠損遮断、T17テスト | 限定合格 | 生存者価格表に営業日内部欠損がないことは完全性保証でない | 歴史候補・終端を入力し停止理由を検証 |
| 3 選定/執行時点 A | S0§4.1翌現地引け、S4別承認の待機台帳 | S4 UTC台帳・prefix切断、T購入>資金確定 | 限定合格 | 旧系列含む購入前リターンは残る。S4の全売却待機はE32月次ラベルとは別 | データknown_atを加え、E32実行前に元ラベル仕様へ接続確認 |
| 4 現金/数量 A | S4 PROTOCOL承認モデル、数量固定/予約は内訳 | S4 run_studyとledger、T17件 | 限定合格 | 仮想単位/費用0/T+0、実株数でない | B整備後も同じ仮想仕様の回帰検証 |
| 5 資金保存/NAV A | S4 1e-10、借入なし、売買でNAV不変 | S4独立差2.66e-14、T保存62,955イベント検算 | 限定合格 | 未モデル化イベント、原価格同一性 | イベント原典を入力した会計検証 |
| 6 恒久ID/クラス B/A | S0§3.1企業ID、§5.2クラス/S1 ID必須 | S5欠落・null受理特性、T未認証影響169採用銘柄 | 判定不能（null拒否には実装欠落） | 発行体/証券/株式クラス混同、再使用、有効期間 | ID表＋方針承認。security_id順は未採用提案 |
| 7 同点順位 A | S0§3.1/§7、S1完全一致 | S5行順6種、T4/228集合差・38/114全順位差 | 旧行順非依存性は不合格。診断2規則は標本内合格 | ticker安定でも恒久IDゲート未解決 | 承認されたID規則の行順/環境/期間端点試験 |
| 8 PIT/可用時刻 B | S0§5.2構成日/利用可能日 | S5年度SHA/19日被覆、T全19日未カバー、16日集計残差 | 判定不能 | 事後構成、発表と発効混同、JP母集団不明 | 原構成集合と時刻付き原典を照合 |
| 9 退出/終端 B/A | S0§4.1対価現金化/§5.2恣意補完禁止 | S5分類6件、T終端検証0/6 | 判定不能 | 未知事件総数、終端損益欠落 | 当該権利/対価と価格・数量を原典照合 |
| 10 価格来歴/調整 B | S0§5.2取得時点/調整方式/ハッシュ | S2固定hash、S3 CSV/Parquet一致 | バイト保全合格／歴史的妥当性判定不能 | 後日調整/ティッカー再使用、配当二重計上 | 原株価・調整係数・企業行動の追跡可能な入力 |
| 11 環境 A | S1二環境・明示欠損・同一順位 | S3最大3.55e-14、S4最大2.22e-16、S5一致、T再集計一致 | 固定二環境で限定合格 | 旧unpinned既定差は残存、将来全環境保証なし | lockと警告を保全、実装変更時のみ同じ試験 |
| 12 E30/E31/E32 A/B/D | S0§7/§9、S2 gate | E30上記。E31 T34,905行再検算。E32未実行 | E30未通過／E31限定合格／E32 NO-GO | 数学同値性と予測力の混同 | Bと同点を解決後、総合再審査。E32実行には別承認 |

## E30へ追加しない条件

- 実通貨のJPY換算、実口座の受渡・端株・実スプレッド・税制の完全再現はCの将来課題。S0§6はJPY統合を保留するため、これをE30失格の独立理由にしない。
- E32/E33の真正OOS、36成熟月末/12四半期/30候補、ΔIC≥0.02・95%下限>0・Holm p<.05等はD。E34の費用10bp等も下流条件。今回は実行しない。
- ただし研究対象の価格系列の企業行動・終端対価Bを「実口座の問題」としてP2へ逃がさない。研究ラベルに必要な時刻Aも同様。
- E31は共通履歴のM1=M0−共通定数。Lβは同窓β控除の複利、M2は過去252日推定による逐次予測誤差の和。後二者を同一モデルとして扱わない。M2未実行で追加性を判定しない。

## 証拠索引（リポジトリ根からのパス）

- S0 `research/phase3/design_v1.0.md`、固定52531ed。§3.1順位、§4.1時刻/終端、§5.2データ、§7実験合否、§9工程終了と予測認証。
- S1 `research/phase3/execution_contract.json`、固定52531ed。既存比較範囲、厳密253価格、ID必須。
- S2 `research/phase3/gate_decision.json`、`results/pinned/{summary.json,input_manifest.json,legacy_selection_comparison.csv,research_m0_m1_rows.csv,research_exclusions.csv,real_future_cutoff_checks.csv}`、`signals.py`、`run_audit.py`、`logs/research_tests_pinned_final.log`。旧価格SHA ab480ba3a1ce8a864aaad9d0dfeb4df16ed115d2bcf5e682e57c96c3de29c67d。
- S3 `research/phase3_repair_20261114/{REPORT.md,protocol.md,repair.py,test_repair.py,gate_decision.json}`、`results_pinned_v1/*`、`verification.log`、`existing_tests_isolated.log`。commit fc499c873e10f4b4f331f3ec10032a656cd381dc。
- S4 `research/phase3_cash_ledger_20261114/{REPORT.md,PROTOCOL.md,ledger.py,run_study.py,test_ledger.py,combined_tests.log,environment_comparison.json,gate_decision.json}`、`results_pinned/*`。commit27144bc。承認設計は`research/phase3_accounting_design_20261114/ACCOUNTING_DESIGN.md`（e0d8c8755d26b402e0c6748b88add87c0686b1ec）。
- S5 `research/phase3_identity_pit_20261114/{P13_REPORT.md,P14_REPORT.md,NEXT_PLAN.md,PROTOCOL.md,tie_audit.py,summarize.py,tie_pinned.log,tie_initial.log,diagnostic_tests.log,source_manifest.json}`、`results_*/*`、`sources/*`。commit80af7f。Phase2参照コード9532f87127a7222dc63d0717e75f79d7c9d34a10、構成元SHA6ff2a610543d12e56b0290c83ed972c8954b8b41。
- T 今回`research/phase3_integrated_gate_20261114/{DIAGNOSTIC_PROTOCOL.md,diagnose.py,diagnostic.log,diagnostic_summary.json,ledger_tests.log,preservation.json,PIT_scope_by_date.csv,exclusion_scope.csv,E31_recheck.csv,ledger_recheck.csv,tie_sensitive_cases.csv}`。新commitは配布COMMIT.txt/TRANSFER.mdの方法で確認。

入力・旧証跡のハッシュはprotected_before.jsonと旧manifestに保存。新証跡はevidence_manifest.json。今回の再検査は保存値と研究合成テストが対象で、価格再取得/新予測ラベル/旧全バックテスト再実行ではない。
