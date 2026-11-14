# GitHub保全・引き継ぎ

リポジトリ avalon19920629-create/Alpha-Engine-backtest。
新branch research/phase3-integrated-gate-20261114。
前提commit 80af7f14446c2f33285478ee7ecb48946a6af937。
main期待値075f0985c76c2212552732534d7f3c741270a80d。
新commitは配布COMMIT.txtとgit bundle list-headsの一致で確認する。自己commit IDを含むファイルはcommitの外で生成するため、本ファイルに未確定SHAを埋め込まない。

現環境のghは未認証、credential helper未設定。GitHubへのpushは未実施。読み取り用の連携をGit objectのpush認証と混同せず、認証情報を要求しない。既存commitの再作成、cherry-pick、force pushはしない。

配布bundleは**前提80af7fを含む全祖先と今回新commitを保存した自己完結bundle**。差分bundleと異なりclone可能。新コードは研究領域のみ、旧354ファイル不変。

認証済Codespacesの既存リポジトリにbundleとCOMMIT.txtを置き、以下を順番に実行する。GitHubは新研究branchだけ更新。エラーや予想外の既存branchがあれば停止。

```bash
git bundle verify ./alpha_engine_phase3_integrated_gate_20261114.bundle
git bundle list-heads ./alpha_engine_phase3_integrated_gate_20261114.bundle
cat COMMIT.txt
git fetch ./alpha_engine_phase3_integrated_gate_20261114.bundle refs/heads/research/phase3-integrated-gate-20261114:refs/heads/research/phase3-integrated-gate-20261114
git rev-parse research/phase3-integrated-gate-20261114
git merge-base --is-ancestor 80af7f14446c2f33285478ee7ecb48946a6af937 research/phase3-integrated-gate-20261114
git diff --name-status 80af7f14446c2f33285478ee7ecb48946a6af937 research/phase3-integrated-gate-20261114
git push origin refs/heads/research/phase3-integrated-gate-20261114:refs/heads/research/phase3-integrated-gate-20261114
git ls-remote origin refs/heads/research/phase3-integrated-gate-20261114 refs/heads/main refs/heads/research/phase3-identity-pit-20261114
```

新branchのSHAがCOMMIT.txt、mainが075f098…、旧研究branchが80af7f…に一致したら保全完了。差分はresearch/phase3_integrated_gate_20261114/への追加のみであること。ユーザーの作業ツリーを切替えずpushできる。mainへマージしない。

成果物：PHASE3_INTEGRATED_AUDIT.md、E30_GATE_MATRIX.md、REMAINING_ISSUES.md、PHASE3_EXIT_OPTIONS.md、NEXT_ACTION_PLAN.md、本書、診断コード・ログ・CSV・manifest。E30未通過／E31限定数学合格／E32 NO-GOを維持。

再開には、NEXT_ACTION_PLANの限定調査の承認が必要。有料契約・仕様変更・E32は別承認。今回の仕事は統合監査・評価完了で停止する。
