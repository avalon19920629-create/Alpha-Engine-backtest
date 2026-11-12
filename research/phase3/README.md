# Phase 3 E30 / E31 audit

[Japanese audit report](https://chatgpt.com/library?file_id=libfile_46658e76506081918caa54618fe30388&account_id=personal)

Outcome: E31 passes mathematical equivalence on the common eligible snapshot.
E30 reproduces the legacy results under pinned dependencies but its full gate does
not pass: legacy execution alignment fails and PIT/identity/event completeness is
indeterminate. E32 transition is NO-GO under design v1.0. No E32+ experiment ran.

Research only. The user-approved `design_v1.0.md` is unchanged. No E32+ experiments,
future-return labels, parameter search, orders, or production configuration changes.

## Inputs

Unpack these two existing Library snapshots outside the repository:

- `phase1-frozen-price-csv-and-sector-map.zip`
- `phase1-rn-realdata-20261112.zip`

The first contains `prices_original.parquet`, `prices_original.csv.gz`,
`metadata.json`, `universe.json`, `real_selected_tickers.csv`, and
`real_six_cell_comparison.csv`. The second provides independent archive-member
identity checks. The required Parquet SHA256 is
`ab480ba3a1ce8a864aaad9d0dfeb4df16ed115d2bcf5e682e57c96c3de29c67d`.
Do not replace this input with freshly downloaded prices.

## Reproduce

Use a dedicated environment and the pinned analytical dependencies:

```bash
python3 -m venv .venv-phase3
.venv-phase3/bin/python -m pip install -r research/phase3/requirements-audit.txt
.venv-phase3/bin/python -m pytest tests research/phase3/test_signals.py -q
.venv-phase3/bin/python research/phase3/run_audit.py \
  --data /absolute/path/to/frozen-price-bundle \
  --original /absolute/path/to/phase1-rn-bundle \
  --out /absolute/path/to/fresh-audit-output
.venv-phase3/bin/python research/phase3/finalize_audit.py \
  --data /absolute/path/to/frozen-price-bundle \
  --out /absolute/path/to/fresh-audit-output
```

Existing upstream tests write demonstration reports under `reports/`; run them in
an expendable worktree or restore only those generated reports after preserving
their diff. Audit code writes only to the explicit output directory. The legacy
module is imported unchanged. `phase1_runner_reference.py` is an archival reference
retrieved from commit `9532f87127a7222dc63d0717e75f79d7c9d34a10`, not executed:
its original entry point downloads live data and is not the reproduction command.

## Separation of semantics

- Legacy: reproduce `iloc[-n]`, full-snapshot usable universe, current dependency
  `pct_change()` defaults, unstable legacy tie rules and daily fixed weights.
- Research: local exchange sessions; exactly 63/126/252 intervals; 253 valid
  prices; no missing-price imputation; M0/M1 mathematical comparison only.
- The ticker-based top-set diagnostic is not a verified permanent-ID selection.
  Missing permanent IDs remain an explicit gate, not an implicit fallback.
- `saved_holdings_accounting_replay.csv` isolates missing-return defaults from
  stock selection. `pad` is expressed explicitly by ffill before pct_change;
  neither mode modifies the original module.
- L-beta independent checks use centered sums to recompute beta and reproduce the
  legacy compounding. They are not M2, do not fit predictive models, and do not
  establish investable alpha.

See the audit report for pass/fail/indeterminate judgments. Environment-specific
results and failed attempts are retained, including harness errors. Thresholds
and model definitions were not revised after observing results.
