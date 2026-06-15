import tempfile, unittest
from pathlib import Path
import numpy as np, pandas as pd
import core_alpha_integration_backtest as c

class TestCoreAlpha(unittest.TestCase):
 def setUp(self):
  self.idx=pd.bdate_range('2020-01-01',periods=520); rng=np.random.default_rng(4)
  names=['VT','BNDX','TLT','TIP','GLDM','DBC','XLRE','SPY','QQQ','1306.T']
  self.p=pd.DataFrame({n:100*np.exp(np.cumsum(rng.normal(.0002,.005,len(self.idx)))) for n in names},index=self.idx)
  self.w=pd.Series({'VT':.4,'BNDX':.1,'TLT':.1,'TIP':.1,'GLDM':.1,'DBC':.05,'XLRE':.05,'CASH':.1})
 def test_bad_weights(self):
  with tempfile.TemporaryDirectory() as d:
   f=Path(d)/'w.csv'; pd.DataFrame({'ticker':['VT','CASH'],'weight':[.5,.4]}).to_csv(f,index=False)
   with self.assertRaises(ValueError): c.load_core_weights(f)
 def test_core_cash_and_combinations_outputs(self):
  core,assets=c.core_returns(self.p,self.w,'2020-02-01','2021-12-31'); self.assertTrue((assets.CASH==0).all()); self.assertGreater(len(core),0)
  alpha=pd.Series(.001,index=core.index); r=c.combine(core,alpha); self.assertIn('Core90_Alpha10',r); self.assertIn('Core85_Alpha15',r)
  with tempfile.TemporaryDirectory() as d:
   summary,_=c.write_outputs(d,r,self.w,pd.DataFrame(),pd.DataFrame())
   self.assertIn('CAGR_improvement_vs_Core_Only',summary); self.assertTrue((Path(d)/'core_alpha_comparison_vs_core.csv').exists()); self.assertTrue((Path(d)/'core_alpha_integration_report.md').exists())
   for f in c.FILES[:7]: self.assertTrue((Path(d)/f).exists())

if __name__=='__main__': unittest.main()
