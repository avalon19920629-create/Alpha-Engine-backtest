import tempfile, unittest
from pathlib import Path
import numpy as np, pandas as pd
import core_alpha_integration_backtest as c

class TestProfiles(unittest.TestCase):
 def setUp(self):
  self.idx=pd.bdate_range('2019-01-01',periods=800); rng=np.random.default_rng(2)
  names=['VT','TLT','TIP','GLD','XLRE','DBC','SHY','BTC-USD','SPY','QQQ','^GSPC','^N225',*[f'US{i}' for i in range(8)],*[f'JP{i}.T' for i in range(8)]]
  self.p=pd.DataFrame({n:100*np.exp(np.cumsum(rng.normal(.0002,.008,len(self.idx)))) for n in names},index=self.idx)
  self.profiles={'A':pd.Series({'VT':.5,'BNDX':0.,'GLD':.2,'SHY':.2,'BTC-USD':.1}),'B':pd.Series({'VT':.4,'TLT':.2,'TIP':.1,'CASH':.2,'BTC-USD':.1})}
 def test_load_validation_and_zero_weight_not_required(self):
  with tempfile.TemporaryDirectory() as d:
   good=Path(d)/'good.csv'; pd.DataFrame([['A','VT',.5],['A','BNDX',0],['A','CASH',.5]],columns=['profile','ticker','weight']).to_csv(good,index=False)
   profiles=c.load_core_profiles(good); self.assertEqual(c.required_core_tickers(profiles),['VT']); c.core_returns(self.p,profiles['A'],'2020-01-01','2021-01-01')
   bad=Path(d)/'bad.csv'; pd.DataFrame([['A','VT',.9]],columns=['profile','ticker','weight']).to_csv(bad,index=False)
   with self.assertRaises(ValueError): c.load_core_profiles(bad)
 def test_btc_shy_cash_and_two_profile_outputs(self):
  _,assets=c.core_returns(self.p,self.profiles['A'],'2020-01-01','2021-01-01'); self.assertIn('BTC-USD',assets); self.assertIn('SHY',assets)
  _,assets=c.core_returns(self.p,self.profiles['B'],'2020-01-01','2021-01-01'); self.assertTrue((assets.CASH==0).all())
  with tempfile.TemporaryDirectory() as d:
   summary,_=c.run_profiles(self.p,[f'US{i}' for i in range(8)],[f'JP{i}.T' for i in range(8)],self.profiles,'2020-01-01','2021-12-31',d)
   for p in self.profiles:
    for n in ('Core_Only','Core90_Alpha10','Core85_Alpha15','Core80_Alpha20'): self.assertIn(f'{p}_{n}',summary.index)
   self.assertTrue((Path(d)/'core_alpha_profile_summary.csv').exists()); self.assertTrue((Path(d)/'core_alpha_profile_report.md').exists())
 def test_evaluation_window_controls_all_metrics_and_equity(self):
  idx=pd.bdate_range('2014-01-01','2016-01-01'); r=pd.Series(.001,index=idx)
  r.loc['2014-06-02']=-.8; r.loc['2015-06-01']=-.1
  with tempfile.TemporaryDirectory() as d:
   returns=pd.DataFrame({'A_Core_Only':r,'A_Core90_Alpha10':r,'A_Core85_Alpha15':r,'A_Core80_Alpha20':r})
   summary,_=c.write_profile_outputs(d,returns,{'A':pd.Series({'CASH':1.})},pd.DataFrame(),pd.DataFrame(),'2015-01-01','2016-01-01','2014-01-01')
   row=summary.loc['A_Core_Only']; equity=pd.read_csv(Path(d)/'core_alpha_profile_equity_curves.csv',index_col=0)
   self.assertGreaterEqual(pd.Timestamp(row.Evaluation_Start),pd.Timestamp('2015-01-01'))
   self.assertAlmostEqual(row.Evaluation_Years,1.0,places=2)
   self.assertAlmostEqual(row.CAGR,(1+row.Total_Return)**(1/row.Evaluation_Years)-1)
   self.assertGreater(row.Max_Drawdown,-.8)
   self.assertAlmostEqual(row.Calmar,row.CAGR/abs(row.Max_Drawdown) if row.Max_Drawdown else np.nan)
   self.assertGreaterEqual(pd.Timestamp(equity.index.min()),pd.Timestamp('2015-01-01'))
   expected=(1+r.loc['2015-01-01':'2016-01-01']).cumprod()
   self.assertAlmostEqual(equity['A_Core_Only'].iloc[0],expected.iloc[0])
if __name__=='__main__': unittest.main()
