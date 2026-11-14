"""Self-financing synthetic total-return units; no live orders or network."""
from bisect import bisect_right
import math
import numpy as np
import pandas as pd
import exchange_calendars as xc

TOL=1e-10
class Blocked(Exception): pass

def calendars(start,end):
    out={}
    for region,name in [('US','XNYS'),('JP','XTKS')]:
        cal=xc.get_calendar(name)
        ss=cal.sessions_in_range(pd.Timestamp(start)-pd.Timedelta(days=10),pd.Timestamp(end)+pd.Timedelta(days=40))
        out[region]=[(cal.session_close(s),pd.Timestamp(s).tz_localize(None).normalize()) for s in ss]
    return out

def signal_time(date,schedules):
    date=pd.Timestamp(date).normalize()
    return max(max(t for t,d in rows if d<=date) for rows in schedules.values())

def run(prices,holdings,eval_dates,schedules=None):
    """One variant. Holdings include only archived targets; timestamps are UTC."""
    dates=pd.DatetimeIndex(eval_dates).tz_localize(None).astype('datetime64[ns]')
    assert dates.is_monotonic_increasing and not dates.has_duplicates
    schedules=schedules or calendars(dates[0],dates[-1])
    end=dates[-1].tz_localize('UTC')+pd.Timedelta(hours=23,minutes=59,seconds=59)
    first=dates[0].tz_localize('UTC')
    h=holdings.copy();h['screen_date']=pd.to_datetime(h.screen_date)
    targets={};events={}
    def add(t,typ,val):events.setdefault(t,[]).append((typ,val))
    for reg,ss in schedules.items():
        for t,d in ss:
            if first<=t<=end:add(t,'close',(reg,d))
    for d,g in h.groupby('screen_date'):
        g=g.sort_values('ticker');assert not g.ticker.duplicated().any()
        weights=g.Weight.to_numpy(float)
        assert np.isfinite(weights).all() and (weights>=0).all()
        assert abs(math.fsum(weights)-1)<=TOL,'target weights not one'
        st=signal_time(d,schedules)
        assert st>=first
        targets[str(d.date())]=g
        if st<=end:add(st,'signal',(str(d.date()),g))
    for d in dates:add(d.tz_localize('UTC')+pd.Timedelta(hours=23,minutes=59,seconds=59),'eval',d)
    close_times={r:[t for t,d in ss] for r,ss in schedules.items()}
    def next_close(reg,t):
        i=bisect_right(close_times[reg],t)
        if i==len(close_times[reg]):raise Blocked('no_future_calendar')
        return close_times[reg][i]
    cash=1.;q={};scale={};last={};observed={};region={};reserved={};buy_due={};sell_due={}
    cycle=None;sig=None;tf=None;target=None;state='IDLE';blocked=None;ledger=[];daily=[]
    def nav():return cash+math.fsum(q[t]*last[t] for t in q if q[t]!=0)
    def invariant():
        res=math.fsum(reserved.values());v=nav();ex=v-cash
        assert np.isfinite(v) and v>0
        assert cash>=-TOL and cash-res>=-TOL and res>=-TOL
        assert all(x>=-TOL for x in q.values())
        assert ex/v<=1+TOL
    def record(now,kind,ticker='',before=None,qbefore=None,pnl=0.,budget=0.,due=None):
        invariant();v=nav();res=math.fsum(reserved.values())
        ledger.append(dict(event_time_utc=now,event_type=kind,cycle_id=cycle,signal_time=sig,funding_time=tf,market=region.get(ticker,''),ticker=ticker,order_id=f'{cycle}:{kind}:{ticker}',quantity_before=q.get(ticker,0.) if qbefore is None else qbefore,quantity_after=q.get(ticker,0.),price=last.get(ticker,np.nan),price_observed_at=observed.get(ticker,pd.NaT),scale=scale.get(ticker,np.nan),budget=budget,cash_total=cash,cash_available=cash-res,cash_reserved=res,nav_before=v if before is None else before,nav_after=v,market_pnl=pnl,investment_fraction=(v-cash)/v,scheduled_fill=due,valuation_status='OK'))
    def quote(ticker,day):
        if day not in prices.index or ticker not in prices.columns:raise Blocked(f'missing_quote:{ticker}:{day}')
        x=float(prices.loc[day,ticker])
        if not math.isfinite(x) or x<=0:raise Blocked(f'invalid_quote:{ticker}:{day}')
        return x
    def reserve(now):
        nonlocal tf,state,reserved,buy_due
        assert not any(x!=0 for x in q.values()) and not sell_due
        tf=now;reserved={row.ticker:cash*float(row.Weight) for row in target.itertuples()}
        for row in target.itertuples():region[row.ticker]=row.Region
        buy_due={t:next_close(region[t],tf) for t in reserved}
        state='BUYING';record(now,'RESERVE')
    for now,evs in sorted(events.items()):
        if blocked:
            for typ,d in evs:
                if typ=='eval':daily.append(dict(date=d,nav=np.nan,return_=np.nan,cash_total=np.nan,cash_fraction=np.nan,investment_fraction=np.nan,pending_orders=len(buy_due)+len(sell_due),status='BLOCKED_DATA'))
            continue
        try:
            closes=sorted([x for typ,x in evs if typ=='close'])
            # Update all held assets before any sale at this timestamp.
            for reg,day in closes:
                for t in sorted(q):
                    if q[t]!=0 and region[t]==reg:
                        raw=quote(t,day);b=nav();old=last[t];last[t]=raw/scale[t];observed[t]=now
                        record(now,'MARK',t,b,pnl=q[t]*(last[t]-old))
            for reg,day in closes:
                for t in sorted(list(sell_due)):
                    if sell_due[t]==now:
                        raw=quote(t,day);assert observed[t]==now
                        b=nav();qb=q[t];cash+=qb*last[t];q[t]=0.;del sell_due[t]
                        record(now,'SELL',t,b,qb);assert abs(nav()-b)<=TOL
            if state=='SELLING' and not sell_due:reserve(now)
            for reg,day in closes:
                for t in sorted(list(buy_due)):
                    if buy_due[t]==now:
                        assert now>tf and region[t]==reg
                        raw=quote(t,day)
                        if t not in scale:scale[t]=raw
                        last[t]=raw/scale[t];observed[t]=now
                        b=nav();qb=q.get(t,0.);budget=reserved.pop(t);cash-=budget;q[t]=qb+budget/last[t];del buy_due[t]
                        record(now,'BUY',t,b,qb,budget=budget,due=now);assert abs(nav()-b)<=TOL
            if state=='BUYING' and not buy_due:state='HOLDING';record(now,'CYCLE_COMPLETE')
            for typ,data in evs:
                if typ=='signal':
                    if buy_due or sell_due:raise Blocked('previous_cycle_incomplete')
                    cycle,target=data;sig=now;tf=None;record(now,'SIGNAL')
                    sell_due={t:next_close(region[t],now) for t in q if q[t]!=0}
                    if sell_due:state='SELLING'
                    else:reserve(now)
            for typ,d in evs:
                if typ=='eval':
                    invariant();v=nav();prev=daily[-1]['nav'] if daily else 1.
                    daily.append(dict(date=d,nav=v,return_=v/prev-1,cash_total=cash,cash_fraction=cash/v,investment_fraction=(v-cash)/v,pending_orders=len(buy_due)+len(sell_due),status='OK'))
        except Blocked as e:
            blocked=dict(time=str(now),reason=str(e));state='BLOCKED_DATA'
            ledger.append(dict(event_time_utc=now,event_type='BLOCKED',cycle_id=cycle,valuation_status='BLOCKED_DATA',reason=str(e)))
            for typ,d in evs:
                if typ=='eval':daily.append(dict(date=d,nav=np.nan,return_=np.nan,cash_total=np.nan,cash_fraction=np.nan,investment_fraction=np.nan,pending_orders=len(buy_due)+len(sell_due),status='BLOCKED_DATA'))
    return dict(daily=pd.DataFrame(daily),ledger=pd.DataFrame(ledger),status='BLOCKED_DATA' if blocked else 'PASS',blocked=blocked,pending_orders=len(buy_due)+len(sell_due),quantities=q,cash=cash,reserved=reserved)

def metrics(daily):
    if daily.nav.isna().any():return dict(status='INDETERMINATE')
    r=daily.return_;v=daily.nav.to_numpy();peak=np.maximum.accumulate(np.r_[1.,v])[1:]
    vol=float(r.std(ddof=1)*np.sqrt(252));c=float(v[-1]**(252/len(v))-1)
    return dict(status='PASS',CAGR=c,Annualized_Volatility=vol,Conventional_Sharpe_RF0=float(r.mean()/r.std(ddof=1)*np.sqrt(252)) if vol else None,Legacy_CAGR_over_vol=c/vol if vol else None,Max_Drawdown=float((v/peak-1).min()),Legacy_compatible_MDD=float((v/np.maximum.accumulate(v)-1).min()),Total_Return=float(v[-1]-1),observations=len(v),mean_cash_fraction=float(daily.cash_fraction.mean()),max_investment_fraction=float(daily.investment_fraction.max()),minimum_cash=float(daily.cash_total.min()))
