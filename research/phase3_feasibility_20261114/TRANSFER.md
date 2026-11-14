# 保全・再現手順

基点：2c246a33115daa8ff16305d2771540b16ac9324a。
新branch：research/phase3-feasibility-20261114。新SHAは配布COMMIT.txt（コミット自身に自己SHAを埋め込まない）。main期待値075f0985c76c2212552732534d7f3c741270a80d。

gh認証なし、git credential helper設定なし。最終push結果は配布PRESERVATION_STATUS.json。成功と記載されていなければリモート保全未完了。force push/コミット再作成は禁止。

## Codespacesで既存コミットを転送

ZIP内のbundleをアップロードし、対象リポジトリ内で実施する。既存branchがある場合は先にSHAを比較し、異なれば停止。

```bash
git bundle verify /path/alpha_engine_phase3_feasibility_20261114.bundle
git fetch /path/alpha_engine_phase3_feasibility_20261114.bundle refs/heads/research/phase3-feasibility-20261114:refs/heads/research/phase3-feasibility-20261114
git rev-parse research/phase3-feasibility-20261114
git rev-parse research/phase3-feasibility-20261114^
git diff --name-status 2c246a33115daa8ff16305d2771540b16ac9324a research/phase3-feasibility-20261114
git push origin research/phase3-feasibility-20261114:research/phase3-feasibility-20261114
git ls-remote origin refs/heads/research/phase3-feasibility-20261114 refs/heads/main
```

新SHAをCOMMIT.txtと、親を基点と照合。差分は新研究領域の追加のみ。mainは期待値一致。bundleは全祖先を含む。git clone --branch research/phase3-feasibility-20261114 <bundle> alpha-engine-phase3でも復元可能。

## 読み取り診断の再現

配布ZIPのreplay_inputsのe30_dataとp1_inputsをclone先alpha-engine-phase3の隣に置く。収録価格は既存固定標本のコピーで再取得ではない。reconciliation_inputs.jsonのSHAを確認する。Python依存はenvironment.txt参照（pandas/numpy/PyYAML/pyarrow）。
保全成果を上書きしないため、別のコピー/作業用checkoutでのみreconcile.pyを再実行する。取得済み5YAMLとAST抽出した旧関数のみ使用し、旧コードのmainやネットワーク取得を実行しない。

```bash
python -B research/phase3_feasibility_20261114/reconcile.py
python -B research/phase3_feasibility_20261114/validate.py
```

-Oは禁止。validate.py単独は読み取り専用。manifestは生成後成果の照合、reconcile.logは19日旧表一致。旧テスト群・旧バックテスト全体を今回再実行したとは称さない。外部Web応答全文は配布しておらず、その本文ハッシュの第三者再照合には同一抽出応答が別途必要（SOURCE_METHOD.md）。
