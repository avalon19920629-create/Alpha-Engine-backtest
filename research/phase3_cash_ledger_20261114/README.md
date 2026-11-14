# E30 cash ledger — synthetic common numeraire

Read REPORT.md and PROTOCOL.md first. User-approved reference: research/phase3_accounting_design_20261114/ACCOUNTING_DESIGN.md. Historical documents are preserved without editing their then-current approval labels.

This model waits for all regional sales, reserves realized cash, buys at each region's strictly subsequent close, and holds fixed synthetic units. No broker, orders, prediction experiment, FX model, or production change.

## Reproduce

Run from repository root in an isolated Python 3.12 environment. Dependency pins are preserved in research/phase3_repair_20261114/requirements_pinned.txt and requirements_initial.txt. Exact observed versions and input/protocol hashes are recorded under results_*/environment.json.

```bash
python -m unittest discover -s research/phase3_cash_ledger_20261114 -p test_ledger.py -v
python research/phase3_cash_ledger_20261114/run_study.py --data /absolute/path/to/frozen-data --output /absolute/path/to/NEW-output
```

Input directory must contain the original prices_original.parquet. Saved holdings are read from research/phase3/results/pinned/legacy_rerun_holdings.csv. Missing frozen files are not downloaded automatically. Do not substitute a fresh price download.

Output directory must not exist. To reproduce the two-environment comparison, use a separate copy of this new study folder and generate results_pinned/results_initial there; do not overwrite this archived study. finalize.py expects those directories and writes derived reports/checks; use a separate copy to reproduce it. It compares data values and missing masks, not gzip container timestamps.

The old production tests can generate reports. They were not rerun in the protected checkout. The 48-test combined run in combined_tests.log includes new 17 + previous repair 9 + original signal 22. The historical 98-test production pass is inherited with unchanged source hashes, not newly claimed.

## Evidence map

- Preregistered contract: PROTOCOL.md, protocol.sha256
- Engine: ledger.py (chronological events, normalized units, cash/reservations)
- Replay/reference: run_study.py (six fixed variants, independent raw-unit oracle, truncated-prefix checks)
- Tests: test_ledger.py, tests_pinned.log, tests_initial.log, combined_tests.log
- Each environment: daily.csv.gz, events.csv.gz, metrics.csv, accounting_checks.csv, order_bridge.csv, prefix_checks.json, final_states.json, legacy_comparison.csv, environment.json
- Cross-environment and financial identities: environment_comparison.json, telescoping_checks.json
- Protection: protected_before.json, protected_after.json; source hashes are on every event row
- Gate: gate_decision.json. E30 NOT_PASSED, E31 limited, E32 NO_GO.

Events at the same timestamp are ordered as specified. Purchases strictly follow funding. A reservation is a cash subset; it is never added to NAV again. Floating-point residual cash is recorded without clipping and tested against absolute tolerance 1e-10.

Blocked required prices stop the path and keep remaining daily observations as NaN; missing days are not dropped to compute performance. The independent oracle is an offline validation on the complete fixed input, not an executable trading strategy and not a data-quality certification.

Old D/H returns and new ledger returns use different accounting/execution assumptions. Their difference is a compound accounting change, not evidence of alpha. Actual currency/settlement, permanent identity, PIT membership and terminal events remain outside this model's certification.
