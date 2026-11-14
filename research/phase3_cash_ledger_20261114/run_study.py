"""Frozen six-cell replay. No selection or parameter optimization."""
from pathlib import Path
import argparse,json,hashlib,platform
import numpy as np
import pandas as pd
import exchange_calendars as xc
from ledger import run,metrics,calendars,TOL
H=Path(__file__).resolve().parent;ROOT=H.parents[1]

def oracle(p,h,ds,schedules):
    """Independent offline scalar cash/raw-price-units reference, cycle by cycle.
    Does not use event ledger, run(), normalized units, or engine schedule helpers.
    """
    trades=[];cash=1.;qty={};regs={};plan=[]
    for cycle,g in h.groupby('screen_date'):
        st=max(max(t for t,d in ss if d<=cycle) for ss in schedules.values())
        sales=[]
        for t,quantity in sorted(qty.items()):
            tm,day=next((tm,d) for tm,d in schedules[regs[t]] if tm>st)
            value=quantity*float(p.loc[day,t]);cash+=value
            trades.append((tm,t,-quantity,value,regs[t]));sales.append(tm)
        qty={};tf=max(sales) if sales else st
        capital=cash
        for row in g.sort_values('ticker').itertuples():
            tm,day=next((tm,d) for tm,d in schedules[row.Region] if tm>tf)
            budget=capital*row.Weight;quantity=budget/float(p.loc[day,row.ticker]);qty[row.ticker]=quantity;regs[row.ticker]=row.Region;cash-=budget
            trades.append((tm,row.ticker,quantity,-budget,row.Region));plan.append((str(cycle.date()),row.ticker,tm,tf,budget))
    # Reconstruction uses only trades already executed and the last scheduled close.
    rawq={};balance=1.;events=sorted(trades,key=lambda x:(x[0],x[2]>0,x[1]));i=0;navs=[]
    for day in ds:
        cut=day.tz_localize('UTC')+pd.Timedelta(hours=23,minutes=59,seconds=59)
        while i<len(events) and events[i][0]<=cut:
            _,t,dq,dc,reg=events[i];rawq[t]=rawq.get(t,0.)+dq;balance+=dc;i+=1
        value=balance
        for t,qq in rawq.items():
            if qq!=0:
                lastday=max(d for tm,d in schedules[regs[t]] if tm<=cut)
                value+=qq*float(p.loc[lastday,t])
        navs.append(value)
    return np.array(navs),pd.DataFrame(plan,columns=['cycle_id','ticker','oracle_buy_time','oracle_funding_time','oracle_budget'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);data=Path(args.data)/'prices_original.parquet'
    pricehash=hashlib.sha256(data.read_bytes()).hexdigest();assert pricehash=='ab480ba3a1ce8a864aaad9d0dfeb4df16ed115d2bcf5e682e57c96c3de29c67d'
    p=pd.read_parquet(data);p.index=p.index.astype('datetime64[ns]');assert p.index.is_unique and p.columns.is_unique
    h=pd.read_csv(ROOT/'research/phase3/results/pinned/legacy_rerun_holdings.csv',parse_dates=['screen_date','trade_date'],float_precision='round_trip');assert len(h)==1026
    ds=p.loc['2022-01-03':'2026-11-11'].index;assert len(ds)==1263
    ss=calendars(ds[0],ds[-1]);summ=[];ledgers=[];days=[];bridges=[];checks=[];cutchecks=[];states={}
    for variant,g in h.groupby('variant'):
        a=run(p,g,ds,ss);states[variant]={k:a[k] for k in ['status','blocked','pending_orders','cash','quantities','reserved']}
        l=a['ledger'];d=a['daily'];l.insert(0,'variant',variant);d.insert(0,'variant',variant)
        ledgers.append(l);days.append(d)
        if a['status']!='PASS':
            summ.append(dict(variant=variant,**metrics(d)));continue
        inv=l[l.event_type!='BLOCKED']
        transfers=inv[inv.event_type.isin(['BUY','SELL','RESERVE'])]
        conserving=float((transfers.nav_after-transfers.nav_before).abs().max())
        market=inv[inv.event_type=='MARK'];pnlerr=float((market.nav_after-market.nav_before-market.market_pnl).abs().max())
        assert conserving<=TOL and pnlerr<=TOL
        nv,plan=oracle(p,g,ds,ss);err=float(np.max(np.abs(nv-d.nav.to_numpy())));assert err<=TOL
        buys=l[l.event_type=='BUY'].copy();buys['cycle_id']=buys.cycle_id.astype(str)
        joined=buys.merge(plan,on=['cycle_id','ticker'],validate='one_to_one')
        assert len(joined)==len(g)
        assert (joined.event_time_utc==joined.oracle_buy_time).all() and (joined.funding_time==joined.oracle_funding_time).all()
        assert (joined.budget-joined.oracle_budget).abs().max()<=TOL
        # Truncate price rows and evaluation after cutoff; future schedules are allowed,
        # future prices and future target rows are not used for a completed prefix.
        cutoff=pd.Timestamp('2024-01-05');short=run(p.loc[:cutoff],g[g.screen_date<=cutoff],ds[ds<=cutoff],ss)
        pd.testing.assert_frame_equal(d[d.date<=cutoff].drop(columns='variant').reset_index(drop=True),short['daily'],check_exact=False,atol=TOL,rtol=0)
        cuttm=cutoff.tz_localize('UTC')+pd.Timedelta(hours=23,minutes=59,seconds=59)
        pd.testing.assert_frame_equal(l[l.event_time_utc<=cuttm].drop(columns='variant').reset_index(drop=True),short['ledger'],check_exact=False,atol=TOL,rtol=0)
        cutchecks.append(dict(variant=variant,cutoff=str(cutoff.date()),status='PASS'))
        checks.append(dict(variant=variant,oracle_nav_max_error=err,event_conservation_max_error=conserving,mark_pnl_max_error=pnlerr,minimum_event_available=float(inv.cash_available.min()),maximum_event_investment=float(inv.investment_fraction.max()),buy_count=int((l.event_type=='BUY').sum()),sell_count=int((l.event_type=='SELL').sum()),status='PASS'))
        target=g.copy();target['cycle_id']=target.screen_date.dt.strftime('%Y-%m-%d')
        b=buys.merge(target[['cycle_id','ticker','trade_date','Weight']],on=['cycle_id','ticker'],validate='one_to_one')
        b['wait_signal_hours']=(b.event_time_utc-b.signal_time).dt.total_seconds()/3600
        b['wait_funding_hours']=(b.event_time_utc-b.funding_time).dt.total_seconds()/3600
        b['actual_weight_at_fill']=b.budget/b.nav_after
        b['target_weight_difference_at_fill']=b.actual_weight_at_fill-b.Weight
        bridges.append(b)
        summ.append(dict(variant=variant,**metrics(d),mean_wait_signal_hours=float(b.wait_signal_hours.mean()),max_wait_signal_hours=float(b.wait_signal_hours.max()),pending_orders=a['pending_orders']))
        print(variant,'PASS',metrics(d)['CAGR'],'oracle_error',err,flush=True)
    pd.concat(days).to_csv(out/'daily.csv.gz',index=False);pd.concat(ledgers).assign(source_hash=pricehash).to_csv(out/'events.csv.gz',index=False)
    if bridges:pd.concat(bridges).to_csv(out/'order_bridge.csv',index=False)
    pd.DataFrame(summ).to_csv(out/'metrics.csv',index=False);pd.DataFrame(checks).to_csv(out/'accounting_checks.csv',index=False)
    (out/'prefix_checks.json').write_text(json.dumps(cutchecks,indent=2));(out/'final_states.json').write_text(json.dumps(states,indent=2))
    old=pd.read_csv(ROOT/'research/phase3_repair_20261114/results_pinned_v1/metrics.csv')
    comp=old[old.cell.isin(['D','H_strict'])].merge(pd.DataFrame(summ),on='variant',suffixes=('_diagnostic','_cash_ledger'))
    comp['comparison_scope']='COMPOUND_ACCOUNTING_CHANGE_NOT_ALPHA_OR_IMPROVEMENT_TEST'
    comp.to_csv(out/'legacy_comparison.csv',index=False)
    (out/'environment.json').write_text(json.dumps(dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,exchange_calendars=xc.__version__,price_sha256=pricehash,protocol_sha256=hashlib.sha256((H/'PROTOCOL.md').read_bytes()).hexdigest()),indent=2))
    assert len(checks)==6,'one or more paths blocked; inspect before proceeding'

if __name__=='__main__':main()
