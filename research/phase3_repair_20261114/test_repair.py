import unittest
import numpy as np
import pandas as pd
import exchange_calendars as xc
from repair import aggregate, classify, make_weights, metrics

class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.cals={r:xc.get_calendar(c) for r,c in [('US','XNYS'),('JP','XTKS')]}
    def test_pre_entry_jump_excluded(self):
        idx=pd.to_datetime(['2022-03-31','2022-04-01','2022-04-04'])
        p=pd.DataFrame({'X':[100.,200.,200.]},index=idx)
        h=pd.DataFrame([dict(variant='v',screen_date=idx[0],trade_date=idx[1],ticker='X',Weight=1.,Region='US')])
        w,_=make_weights(p,h,self.cals);r=p.pct_change(fill_method=None)
        old,_=aggregate(r,w['v']['old']);new,_=aggregate(r,w['v']['local'])
        self.assertEqual((1+old).prod()-1,1.);self.assertEqual((1+new).prod()-1,0.)
    def test_old_owner_keeps_exit_day_new_owner_excluded(self):
        idx=pd.to_datetime(['2022-03-31','2022-04-01','2022-06-30','2022-07-01','2022-07-05'])
        p=pd.DataFrame({'X':[100,100,100,110,110],'Y':[100,100,100,200,220]},index=idx)
        h=pd.DataFrame([dict(variant='v',screen_date=idx[0],trade_date=idx[1],ticker='X',Weight=1.,Region='US'),dict(variant='v',screen_date=idx[2],trade_date=idx[3],ticker='Y',Weight=1.,Region='US')])
        w,_=make_weights(p,h,self.cals);r,_=aggregate(p.pct_change(fill_method=None),w['v']['local'])
        self.assertAlmostEqual(r.iloc[3],.1);self.assertAlmostEqual(r.iloc[4],.1)
        self.assertEqual(w['v']['local'].iloc[0].sum(),0)
        self.assertEqual(w['v']['local'].loc[idx[3],'X'],1)
        self.assertEqual(w['v']['local'].loc[idx[3],'Y'],0)
    def test_local_holiday_delay_and_signal_time(self):
        idx=pd.to_datetime(['2022-12-30','2023-01-03','2023-01-04','2023-01-05'])
        p=pd.DataFrame({'X.T':[100,np.nan,200,220]},index=idx)
        h=pd.DataFrame([dict(variant='v',screen_date=idx[0],trade_date=idx[1],ticker='X.T',Weight=1.,Region='JP')])
        w,led=make_weights(p,h,self.cals)
        self.assertEqual(str(led.entry_date.iloc[0].date()),'2023-01-04')
        self.assertEqual(w['v']['local'].loc[idx[2]].sum(),0)
        self.assertEqual(w['v']['local'].loc[idx[3]].sum(),1)
        self.assertGreater(pd.Timestamp(led.entry_close_utc.iloc[0]),pd.Timestamp(led.signal_available_utc.iloc[0]))
    def test_holiday_fill_restores_return(self):
        idx=pd.to_datetime(['2022-07-01','2022-07-04','2022-07-05'])
        p=pd.DataFrame({'X':[100,np.nan,110]},index=idx)
        safe,_,_=classify(p,{'X':'US'},self.cals)
        self.assertEqual(safe.iloc[1,0],100)
        self.assertAlmostEqual(safe.pct_change(fill_method=None).iloc[2,0],.1)
    def test_open_missing_never_filled(self):
        idx=pd.to_datetime(['2022-07-05','2022-07-06','2022-07-07'])
        p=pd.DataFrame({'X':[100,np.nan,110]},index=idx)
        safe,_,counts=classify(p,{'X':'US'},self.cals)
        self.assertTrue(np.isnan(safe.iloc[1,0]))
        self.assertEqual(counts.category.iloc[0],'open_session_internal')
        w=pd.DataFrame(1.,index=idx,columns=['X']);r,missing=aggregate(safe.pct_change(fill_method=None),w,True)
        self.assertTrue(r.isna().all());self.assertIsNone(metrics(r)['CAGR'])
    def test_no_fill_before_listing_or_after_terminal(self):
        idx=pd.to_datetime(['2022-07-04','2022-07-05','2022-07-06','2022-07-07'])
        p=pd.DataFrame({'X':[np.nan,100,np.nan,np.nan]},index=idx)
        safe,_,_=classify(p,{'X':'US'},self.cals)
        self.assertEqual(int(safe.X.isna().sum()),3)
    def test_zero_holdings_do_not_block(self):
        ret=pd.DataFrame({'X':[np.nan,.1]});w=pd.DataFrame({'X':[0.,1.]})
        r,_=aggregate(ret,w,True);self.assertEqual(r.tolist(),[0.,.1])
    def test_asynchronous_gross_exposure_detected(self):
        idx=pd.to_datetime(['2022-09-30','2022-10-03','2022-12-30','2023-01-03','2023-01-04','2023-01-05'])
        p=pd.DataFrame(100.,index=idx,columns=['X','Y.T']);rows=[]
        for d,t,ws in [(idx[0],idx[1],[.2,.8]),(idx[2],idx[3],[.8,.2])]:
            for ticker,reg,w in zip(p.columns,['US','JP'],ws):rows.append(dict(variant='v',screen_date=d,trade_date=t,ticker=ticker,Weight=w,Region=reg))
        w,_=make_weights(p,pd.DataFrame(rows),self.cals)
        self.assertAlmostEqual(w['v']['local'].loc['2023-01-04'].sum(),1.6)
    def test_log_factorial_identity(self):
        a,b,c,d=np.log1p([.1,.2,.3,.4]);self.assertAlmostEqual((d-b)-(c-a),d-b-c+a)

if __name__=='__main__':unittest.main(verbosity=2)
