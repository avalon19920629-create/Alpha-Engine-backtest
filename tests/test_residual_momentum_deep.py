import tempfile, unittest
from pathlib import Path
import numpy as np, pandas as pd
import alpha_engine_backtest as a

class TestResidualMomentumDeep(unittest.TestCase):
    def setUp(self):
        self.p=a.demo_prices(); self.us=[f"US{i}" for i in range(8)]; self.jp=[f"JP{i}.T" for i in range(8)]

    def test_baseline_and_weights(self):
        variants=a.build_residual_weight_variants()
        self.assertEqual(variants[0]["name"],"Baseline")
        self.assertEqual(variants[0]["base_weight"],1.0)
        self.assertEqual(variants[0]["residual_weight"],0.0)
        for v in variants:
            self.assertAlmostEqual(v["base_weight"]+v["residual_weight"],1.0)
            self.assertEqual(v["vcp_weight"],0.0)

    def test_simple_residual_and_missing_benchmark(self):
        idx=pd.bdate_range("2020-01-01",periods=300)
        p=pd.DataFrame({"AAA":np.linspace(100,200,300),"SPY":np.linspace(100,150,300)},index=idx)
        got=a.compute_residual_momentum_score(p,["AAA"],idx[-1],"US")
        self.assertIn("residual_score",got.columns)
        no_bench=a.compute_residual_momentum_score(p.drop(columns=["SPY"]),["AAA"],idx[-1],"US")
        self.assertFalse(no_bench.empty)

    def test_beta_adjusted_fallbacks(self):
        idx=pd.bdate_range("2020-01-01",periods=20)
        stock=pd.Series(np.linspace(100,110,20),index=idx)
        flat=pd.Series(100.0,index=idx)
        self.assertEqual(a.compute_beta_adjusted_residual_score(stock,flat,20),0.0)
        moving=pd.Series(np.linspace(100,105,20),index=idx)
        self.assertTrue(np.isfinite(a.compute_beta_adjusted_residual_score(stock,moving,20)))

    def test_residual_is_not_hard_filter_and_no_vcp_column(self):
        variant={"name":"Residual_40","base_weight":0.6,"residual_weight":0.4,"vcp_weight":0.0}
        p=a._select_residual_deep(self.p,self.us,self.jp,"2022-12-30",variant,a.build_benchmark_modes()["broad_default"],"simple")
        self.assertGreater(len(p),0)
        self.assertNotIn("vcp_score",p.columns)

    def test_outputs_and_determinism(self):
        with tempfile.TemporaryDirectory() as d:
            summary1=a.run_residual_momentum_deep_audit(self.p,self.us,self.jp,"2020-01-01","2021-12-31",d)
            summary2=a.run_residual_momentum_deep_audit(self.p,self.us,self.jp,"2020-01-01","2021-12-31",d)
            pd.testing.assert_series_equal(summary1["CAGR"],summary2["CAGR"])
            for name in ["variant_summary.csv","selected_tickers.csv","selection_diff.csv","score_components.csv","benchmark_sensitivity.csv","residual_method_comparison.csv","audit_metadata.json"]:
                self.assertTrue((Path(d)/name).exists(), name)
            self.assertTrue(Path("reports/residual_momentum_deep_audit_report.md").exists())

if __name__=="__main__": unittest.main()
