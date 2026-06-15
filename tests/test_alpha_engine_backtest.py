import tempfile, unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np, pandas as pd
import alpha_engine_backtest as a
class TestAudit(unittest.TestCase):
 def setUp(self): self.p=a.demo_prices()
 def test_future_blocked(self):
  d=a.asof_prices(self.p,"2020-01-02"); self.assertLessEqual(d.index.max(),pd.Timestamp("2020-01-02"))
 def test_shortage_and_missing(self): self.assertLessEqual(len(a.select_portfolio(self.p,["US0","NOPE"],["JP0.T"],"2022-12-30")),2)
 def test_maxdd(self): self.assertAlmostEqual(a.max_drawdown(pd.Series([.1,-.2,.1])), -.2)
 def test_cagr(self): self.assertAlmostEqual(a.cagr(pd.Series([.1]*252)),1.1**252-1)
 def test_exposure(self): self.assertEqual([a.exposure_for_regimes("BULL","BULL"),a.exposure_for_regimes("BULL","BEAR"),a.exposure_for_regimes("BEAR","BEAR")],[1,.6,.2])
 def test_trade_after_screen_and_csv(self):
  s,x,t=a.run_backtest(self.p,[f"US{i}" for i in range(8)],[f"JP{i}.T" for i in range(8)],"2020-01-01","2021-12-31"); self.assertTrue((pd.to_datetime(x.trade_date)>pd.to_datetime(x.screen_date)).all())
  with tempfile.TemporaryDirectory() as d: a.write_outputs(d,s,x,t); self.assertTrue((Path(d)/"selected_tickers_by_period.csv").exists())

 def test_live_download_multiindex_and_warmup(self):
  idx=pd.bdate_range("2013-01-01","2016-01-01"); raw=pd.DataFrame({("Close","AAA"):np.arange(len(idx))+100.,("Close","BAD"):np.nan},index=idx)
  with patch("yfinance.download",return_value=raw) as download:
   got=a.download_live_prices(["AAA","BAD"],"2015-01-01","2015-12-31")
  self.assertEqual(list(got.columns),["AAA"]); self.assertLessEqual(pd.Timestamp(download.call_args.kwargs["start"]),pd.Timestamp("2013-06-20"))
 def test_summary_strategy_and_benchmark_turnover(self):
  s,x,t=a.run_backtest(self.p,[f"US{i}" for i in range(8)],[f"JP{i}.T" for i in range(8)],"2020-01-01","2021-12-31")
  with tempfile.TemporaryDirectory() as d:
   a.write_outputs(d,s,x,t,{"SPY":self.p.SPY.pct_change().loc["2020-01-01":"2021-12-31"]}); summary=pd.read_csv(Path(d)/"backtest_summary.csv")
   self.assertEqual(summary.columns[0],"Strategy"); self.assertTrue(pd.isna(summary.loc[summary.Strategy=="SPY","Turnover"]).all())
if __name__=="__main__": unittest.main()
