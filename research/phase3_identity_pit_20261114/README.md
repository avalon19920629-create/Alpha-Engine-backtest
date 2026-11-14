# Phase3 P1-3/P1-4監査成果 2026-11-14

保全合格。旧順位規則の行順非依存性は不合格。恒久ID・PIT・企業イベントの完全性は判定不能。E30未通過、E31限定的数学合格、E32 NO-GO。監査完了で停止。

成果物：PRESERVATION_REPORT.md、P13_REPORT.md、P14_REPORT.md、NEXT_PLAN.md、gate_decision.json。診断の根拠はdiagnostic_summary.json、boundary_ties.csv、二環境results、ログ、sources、各manifest。

## 再現手順

リポジトリ根から実行。旧価格は同梱しない。既存凍結データ（prices_original.parquet、universe.json）をDATAに指定する。依存環境はresults_*/environment.jsonを参照。

```bash
python research/phase3_identity_pit_20261114/tie_audit.py --data "$DATA" --out /tmp/p13_new_run
python research/phase3_identity_pit_20261114/summarize.py
```

最初のコマンドは既存出力ディレクトリを拒否する。summarize.pyは保存済二環境出力を照合し、新研究領域の要約だけ再生成する。全旧297ファイルの保護ハッシュを検証する。audit_preservation.pyは保全時remote refsを照合するため、将来ブランチ更新後の実行は保全時と異なる結果になりうる。

既存48件テストを今回の新規テスト数へ加算しない。今回：二環境で各228旧選定・699群再現、4,104診断行、二環境比較、strict_order合成4ケース、旧297件不変。null受理を観測した特性テストのPASSはID安全性合格ではない。

GitHub CLIは未認証、credential helper未設定。認証情報を要求せず、pushを繰り返さず、同一コミットのbundleを転送する。詳細は配布ZIPのTRANSFER.md。研究ブランチ：research/phase3-identity-pit-20261114。
