import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

import alpha_engine_production as prod


class AlphaEngineProductionGearboxTests(unittest.TestCase):
    def test_robust_profile_resolves_to_residual60_n12(self):
        cfg = prod.resolve_engine_config("ROBUST")
        self.assertEqual(cfg.residual_ratio, 60)
        self.assertEqual(cfg.portfolio_n, 12)
        self.assertEqual(cfg.role, "standard_robust_alpha")
        self.assertEqual(cfg.variant_name, "Residual_60_N12_TTL90_Renew30_Composite")

    def test_ferrari_profile_resolves_to_residual100_n6_when_allowed(self):
        cfg = prod.resolve_engine_config("FERRARI", allow_overdrive=True)
        self.assertEqual(cfg.residual_ratio, 100)
        self.assertEqual(cfg.portfolio_n, 6)
        self.assertEqual(cfg.role, "overdrive_satellite_alpha")
        self.assertEqual(cfg.variant_name, "Residual_100_N6_TTL90_Renew30_Composite")

    def test_ferrari_requires_overdrive_lock(self):
        with self.assertRaises(PermissionError):
            prod.resolve_engine_config("FERRARI", allow_overdrive=False)

    def test_custom_requires_safety_lock(self):
        with self.assertRaises(PermissionError):
            prod.resolve_engine_config("CUSTOM", allow_custom_profile=False, custom_residual_ratio=55, custom_portfolio_n=10)
        cfg = prod.resolve_engine_config("CUSTOM", allow_custom_profile=True, custom_residual_ratio=55, custom_portfolio_n=10)
        self.assertEqual(cfg.run_mode, "research")
        self.assertEqual(cfg.variant_name, "Residual_55_N10_TTL90_Renew30_Composite")

    def test_metadata_and_report_include_profile_parameters_and_git_hash(self):
        cfg = prod.resolve_engine_config("ROBUST")
        dates = pd.bdate_range("2024-01-01", periods=5)
        selected = pd.DataFrame([{"variant": cfg.variant_name, "screen_date": dates[0], "trade_date": dates[1], "ticker": "AAA", "exit_date": dates[-1], "holding_days": 3, "Weight": 1.0, "Region": "US"}])
        empty = pd.DataFrame()
        returns = pd.Series(0.0, index=dates)
        prices = pd.DataFrame({"US0": range(100, 105), "JP0.T": range(100, 105)}, index=dates)
        with TemporaryDirectory() as td:
            with patch.object(prod.alpha, "get_live_universe", return_value=(["US0"], ["JP0.T"])), \
                 patch.object(prod.alpha, "build_live_data_quality_report", return_value=(pd.DataFrame(), pd.DataFrame(), [], ["US0", "JP0.T"])), \
                 patch.object(prod.alpha, "run_ttl_renewal_variant", return_value=(returns, selected, empty, empty, empty, empty, empty, empty)), \
                 patch.object(prod, "git_commit_hash", return_value="abc123"):
                out = prod.run_production(cfg, start="2024-01-01", end="2024-01-05", output_root=td, demo=True)
            meta = json.loads((out / "metadata.json").read_text())
            report = (out / "run_report.md").read_text()
        self.assertEqual(meta["profile"], "ROBUST")
        self.assertEqual(meta["residual_ratio"], 60)
        self.assertEqual(meta["portfolio_n"], 12)
        self.assertEqual(meta["git_commit"], "abc123")
        self.assertIn("Profile: ROBUST", report)
        self.assertIn("residual_ratio: 60", report)
        self.assertIn("Git commit hash: abc123", report)


if __name__ == "__main__":
    unittest.main()
