import unittest
import numpy as np
import pandas as pd
from ledger import run, metrics, calendars, signal_time

class LedgerTests(unittest.TestCase):
    def data(self,start='2022-03-30',end='2022-04-08',names=('X',)):
        d=pd.bdate_range(start,end);return pd.DataFrame(100.,index=d,columns=list(names))
    def h(self,rows):
        return pd.DataFrame(rows,columns=['screen_date','ticker','Region','Weight'])
    def test_pre_entry_jump(self):
        p=self.data();p.loc['2022-04-01':,'X']=200
        a=run(p,self.h([('2022-03-31','X','US',1.)]),p.index)
        self.assertEqual(a['status'],'PASS');self.assertAlmostEqual(a['daily'].nav.iloc[-1],1)
    def test_sale_return_and_waiting_cash(self):
        p=self.data(names=('X','Y'));p.loc['2022-04-05':,'X']=110
        a=run(p,self.h([('2022-03-31','X','US',1.),('2022-04-04','Y','US',1.)]),p.index)
        self.assertAlmostEqual(a['daily'].nav.iloc[-1],1.1)
        self.assertAlmostEqual(a['daily'].set_index('date').loc['2022-04-05','cash_fraction'],1)
        l=a['ledger'];self.assertGreater(l[l.event_type=='BUY'].iloc[-1].event_time_utc,l[l.event_type=='SELL'].iloc[-1].event_time_utc)
    def test_asynchronous_no_borrowing(self):
        p=self.data('2022-12-27','2023-01-09',('X','Y.T'))
        h=self.h([('2022-12-28','X','US',.2),('2022-12-28','Y.T','JP',.8),('2022-12-30','X','US',.8),('2022-12-30','Y.T','JP',.2)])
        a=run(p,h,p.index);self.assertEqual(a['status'],'PASS')
        self.assertLessEqual(a['ledger'].investment_fraction.max(),1+1e-10)
        self.assertGreaterEqual(a['ledger'].cash_available.min(),-1e-10)
        self.assertTrue(np.allclose(a['daily'].nav,1,atol=1e-10,rtol=0))
        l=a['ledger'];last=l[l.cycle_id=='2022-12-30'];s=last[last.event_type=='SELL'];b=last[last.event_type=='BUY']
        self.assertTrue((b.event_time_utc>s.event_time_utc.max()).all())
        self.assertEqual(str(s[s.market=='JP'].iloc[0].event_time_utc.date()),'2023-01-04')
    def test_reserved_cash_not_double_counted(self):
        p=self.data(names=('X','Y.T'));a=run(p,self.h([('2022-03-31','X','US',.4),('2022-03-31','Y.T','JP',.6)]),p.index)
        l=a['ledger'];s=l[l.event_type=='RESERVE'].iloc[0]
        self.assertAlmostEqual(s.cash_total,1);self.assertAlmostEqual(s.cash_reserved,1);self.assertAlmostEqual(s.nav_after,1)
    def test_unknown_held_price_blocks_without_dropping_days(self):
        p=self.data();p.loc['2022-04-04','X']=np.nan
        a=run(p,self.h([('2022-03-31','X','US',1.)]),p.index)
        self.assertEqual(a['status'],'BLOCKED_DATA');self.assertEqual(len(a['daily']),len(p));self.assertEqual(metrics(a['daily'])['status'],'INDETERMINATE')
        self.assertTrue(a['daily'].set_index('date').loc['2022-04-04':,'nav'].isna().all())
    def test_deleted_open_day_blocks(self):
        p=self.data();d=p.index;p=p.drop(pd.Timestamp('2022-04-04'))
        a=run(p,self.h([('2022-03-31','X','US',1.)]),d)
        self.assertEqual(a['status'],'BLOCKED_DATA')
    def test_missing_purchase_price_blocks(self):
        p=self.data();p.loc['2022-04-01','X']=np.nan
        a=run(p,self.h([('2022-03-31','X','US',1.)]),p.index);self.assertEqual(a['status'],'BLOCKED_DATA')
    def test_holiday_no_fill_orders(self):
        p=self.data('2022-06-29','2022-07-07');p.loc['2022-07-04','X']=np.nan
        a=run(p,self.h([('2022-06-30','X','US',1.)]),p.index)
        self.assertEqual(a['status'],'PASS');self.assertAlmostEqual(a['daily'].nav.iloc[-1],1)
    def test_future_prices_and_rows_cannot_change_prefix(self):
        p=self.data();h=self.h([('2022-03-31','X','US',1.)]);a=run(p,h,p.index)
        b=p.copy();b.loc['2022-04-06':,'X']=900
        aa=run(b,h,p.index);cut=pd.Timestamp('2022-04-05 23:59:59',tz='UTC')
        pd.testing.assert_frame_equal(a['ledger'][a['ledger'].event_time_utc<=cut].reset_index(drop=True),aa['ledger'][aa['ledger'].event_time_utc<=cut].reset_index(drop=True))
        cc=run(p.loc[:'2022-04-05'],h,p.index[p.index<=pd.Timestamp('2022-04-05')])
        pd.testing.assert_frame_equal(a['daily'][a['daily'].date<=pd.Timestamp('2022-04-05')].reset_index(drop=True),cc['daily'])
    def test_price_scale_invariance(self):
        p=self.data(names=('X','Y'));p.loc['2022-04-04':,'X']=120
        h=self.h([('2022-03-31','X','US',.5),('2022-03-31','Y','US',.5)])
        a=run(p,h,p.index);b=run(p*pd.Series({'X':7.,'Y':.01}),h,p.index)
        np.testing.assert_allclose(a['daily'].nav,b['daily'].nav,atol=1e-10,rtol=0)
    def test_order_row_permutation(self):
        p=self.data(names=('X','Y'));h=self.h([('2022-03-31','X','US',.4),('2022-03-31','Y','US',.6)])
        a=run(p,h,p.index);b=run(p,h.iloc[::-1],p.index)
        pd.testing.assert_frame_equal(a['ledger'],b['ledger'])
    def test_initial_nav_in_drawdown(self):
        d=pd.DataFrame(dict(nav=[.9,.95],return_=[-.1,.95/.9-1],cash_fraction=[0.,0.],investment_fraction=[1.,1.],cash_total=[0.,0.]))
        self.assertAlmostEqual(metrics(d)['Max_Drawdown'],-.1);self.assertEqual(metrics(d)['Legacy_compatible_MDD'],0)
    def test_end_pending_orders_remain_cash(self):
        p=self.data(end='2022-03-31');a=run(p,self.h([('2022-03-31','X','US',1.)]),p.index)
        self.assertEqual(a['pending_orders'],1);self.assertAlmostEqual(a['daily'].nav.iloc[-1],1);self.assertEqual(a['cash'],1)
    def test_unfinished_cycle_blocks_next_signal(self):
        p=self.data('2022-12-29','2023-01-06',('X','Y.T'))
        h=self.h([('2022-12-30','Y.T','JP',1.),('2023-01-03','X','US',1.)])
        a=run(p,h,p.index);self.assertEqual(a['status'],'BLOCKED_DATA');self.assertIn('previous_cycle_incomplete',a['blocked']['reason'])
    def test_invalid_weight_sum_rejected(self):
        p=self.data()
        with self.assertRaises(AssertionError):run(p,self.h([('2022-03-31','X','US',1.1)]),p.index)
    def test_dst_calendar_and_close_cutoff(self):
        c=calendars('2022-03-01','2022-03-31');times={str(d.date()):t for t,d in c['US']}
        self.assertEqual(times['2022-03-11'].hour,21);self.assertEqual(times['2022-03-14'].hour,20)
    def test_fixed_units_differ_from_daily_weight_rebalancing(self):
        p=self.data(names=('X','Y'));p.loc['2022-04-04':,'X']=200;p.loc['2022-04-05':,'X']=100
        a=run(p,self.h([('2022-03-31','X','US',.5),('2022-03-31','Y','US',.5)]),p.index)
        self.assertAlmostEqual(a['daily'].nav.iloc[-1],1.)
        # Same prices: daily 50/50 would give 1.5 * .75 = 1.125.
        self.assertNotAlmostEqual(a['daily'].nav.iloc[-1],1.125)

if __name__=='__main__':unittest.main(verbosity=2)
