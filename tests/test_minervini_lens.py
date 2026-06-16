import tempfile, unittest
from pathlib import Path
import pandas as pd
import alpha_engine_backtest as a

class TestMinerviniLens(unittest.TestCase):
    def setUp(self):
        self.p=a.demo_prices()
        self.us=[f"US{i}" for i in range(8)]
        self.jp=[f"JP{i}.T" for i in range(8)]

    def test_baseline_weights_and_variant_sums(self):
        b=a.MINERVINI_VARIANTS[0]
        self.assertEqual((b["base_weight"],b["residual_weight"],b["vcp_weight"]),(1.0,0.0,0.0))
        for v in a.MINERVINI_VARIANTS:
            self.assertAlmostEqual(v["base_weight"]+v["residual_weight"]+v["vcp_weight"],1.0)

    def test_residual_score_and_missing_benchmark(self):
        d=a.compute_residual_momentum_score(self.p,self.us,"2022-12-30","US")
        self.assertIn("residual_score",d.columns)
        self.assertGreater(len(d),0)
        no_bench=self.p.drop(columns=[c for c in ["SPY","^GSPC"] if c in self.p])
        d2=a.compute_residual_momentum_score(no_bench,self.us,"2022-12-30","US")
        self.assertTrue(d2.residual_score.notna().all())

    def test_vcp_score_is_not_hard_filter(self):
        d=a.compute_vcp_proxy_score(self.p,self.us,"2022-12-30")
        self.assertIn("vcp_score",d.columns)
        self.assertGreater(len(d),1)
        self.assertTrue(((d.vcp_score>=0)&(d.vcp_score<=1)).all())

    def test_combined_score_and_no_exit_or_regime_columns(self):
        base=a.score_universe(self.p,self.us,"2022-12-30")
        r=a.compute_residual_momentum_score(self.p,base.index,"2022-12-30","US")
        v=a.compute_vcp_proxy_score(self.p,base.index,"2022-12-30")
        c=a.combine_minervini_lens_score(base,r,v,a.MINERVINI_VARIANTS[5])
        self.assertIn("Final_Score",c.columns)
        self.assertNotIn("Regime_Exposure",c.columns)
        self.assertFalse(any("stop" in col.lower() or "exit" in col.lower() for col in c.columns))

    def test_demo_deterministic_and_outputs(self):
        p2=a.demo_prices()
        pd.testing.assert_frame_equal(self.p,p2)
        with tempfile.TemporaryDirectory() as td:
            summary=a.run_minervini_lens_audit(self.p,self.us,self.jp,"2020-01-01","2021-12-31",td)
            self.assertEqual(len(summary),8)
            for name in ["variant_summary.csv","selected_tickers.csv","score_components.csv"]:
                self.assertTrue((Path(td)/name).exists())
            self.assertTrue(Path("reports/minervini_lens_audit_report.md").exists())

if __name__=="__main__": unittest.main()
