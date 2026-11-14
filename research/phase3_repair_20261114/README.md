# E30 repair diagnostics — 2026-11-14

Start with [REPORT.md](REPORT.md), [protocol.md](protocol.md), and the input-format clarification. The protocol and its original hash are preserved, including the failed input-identifier preflight log. No threshold or price value changed.

Baseline: 52531ed2d9ce1ea275fb77bbc7e18bc582655af0. Old 195 tracked files remain unchanged. New branch: research/phase3-e30-repair-20261114. No merge or push of this repair work has been performed.

## Deliverables

- GitHub final audit: preservation_audit.json, preservation_audit_final.json, preservation_final.log, protected_final.json, remote_refs_final.txt
- Preregistered conditions: protocol.md, protocol.sha256, protocol_input_clarification.md
- Code: repair.py, test_repair.py, verify.py, audit_preservation.py, summarize.py
- Data quality: results_*/missing_summary.csv, missing_cells.csv.gz, absent_columns.json; suppression_causes.csv
- Execution accounting: execution_ledger.csv, independent_ownership_oracle.json, funding_exceptions.csv
- 2x2: factorial_CAGR.csv, factorial_logwealth_total.csv, calendar_bridge.csv; results_*/metrics.csv, daily.csv.gz, quarterly_logwealth.csv, factorial_daily.csv.gz
- Legacy reconciliation: legacy_reproduction.csv, all_legacy_metric_checks.csv
- Tests and environment: *_tests_*.log, tests_*.log, verification.log, environment_comparison.json, requirements_*.txt, results_*/environment.json
- Updated status: gate_decision.json (does not replace the old gate)

## Replay

Use Python 3.12 and isolated environments described by requirements_initial.txt and requirements_pinned.txt. Install dependencies in the isolated environment, not in production. Place the already frozen data under the workspace's e30_data/phase1-frozen-price-csv-and-sector-map/ directory or specify another path with --data. Inputs: prices_original.parquet, prices_original.csv.gz, universe.json. The script reads the preserved holdings and outputs from research/phase3/.

From the repository root:

```bash
python -m unittest discover -s research/phase3_repair_20261114 -p test_repair.py -v
python research/phase3_repair_20261114/repair.py --data /absolute/path/to/frozen-data --output /absolute/path/to/new-output
python research/phase3_repair_20261114/audit_preservation.py
```

Output directories must not exist. Do not overwrite results_pinned_v1/results_initial_v1. To recreate their named comparison layout, use a separate checkout/copy and new output directories there. verify.py expects the two preserved results directories plus frozen data in the adjacent workspace e30_data directory; it verifies the results without rerunning selection. summarize.py regenerates derived report/summary outputs; run it only in a separate copy when validating the frozen artifacts.

The old production test suite can write example reports: execute it in a separate git archive of the baseline, as done here. Never run it over the protected research checkout without isolating those writes.

C/D/H_strict are diagnostic return aggregates, NOT certified self-financing portfolios. Local-close ownership can pass while aggregate budget feasibility fails. No price prediction, IC or E32+ experiments.
