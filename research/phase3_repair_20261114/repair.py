"""Frozen-holdings accounting diagnostics. No network or production mutation."""
import argparse, hashlib, json, platform
from pathlib import Path
import numpy as np
import pandas as pd
import exchange_calendars as xc

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
TOL=1e-10

def metrics(r):
    if r.isna().any():
        return {k:None for k in ['CAGR','Annualized_Volatility','Max_Drawdown','Sharpe','Conventional_Sharpe_RF0','Total_Return','logwealth']}
    w=(1+r).cumprod(); vol=float(r.std(ddof=1)*np.sqrt(252)); c=float(w.iloc[-1]**(252/len(r))-1)
    return dict(CAGR=c,Annualized_Volatility=vol,Max_Drawdown=float((w/w.cummax()-1).min()),Sharpe=c/vol if vol else None,Conventional_Sharpe_RF0=float(r.mean()/r.std(ddof=1)*np.sqrt(252)) if vol else None,Total_Return=float(w.iloc[-1]-1),logwealth=float(np.log1p(r).sum()))

def classify(p,regions,cals):
    rows=[]; safe=p.copy(); summary=[]
    for t in p:
        s=p[t]; reg=regions[t]; sessions=cals[reg].sessions.tz_localize(None); op=p.index.isin(sessions)
        valid=s.notna(); first=s.first_valid_index();last=s.last_valid_index()
        if first is None: kinds=np.repeat('all_missing',len(p))
        else:
            kinds=np.where(~op,'market_closed',np.where(p.index<first,'pre_history',np.where(p.index>last,'terminal_unknown','open_session_internal')))
        miss=~valid.to_numpy()
        for kind in np.unique(kinds[miss]): summary.append(dict(ticker=t,region=reg,category=kind,count=int((miss&(kinds==kind)).sum())))
        for d,k in zip(p.index[miss],kinds[miss]): rows.append((d,t,reg,k))
        if first is not None:
            fill=miss&~op&(p.index>=first)&(p.index<=last)
            safe.loc[fill,t]=s.ffill().loc[fill]
    return safe,pd.DataFrame(rows,columns=['date','ticker','region','category']),pd.DataFrame(summary)

def make_weights(p,h,cals):
    idx=p.loc['2022-01-01':'2026-11-11'].index; cols=sorted(h.ticker.unique())
    result={}; logs=[]
    for v,g in h.groupby('variant'):
        dates=sorted(g.screen_date.unique())
        mats={k:pd.DataFrame(0.,index=idx,columns=cols) for k in ['old','union','local']}
        for j,d in enumerate(dates):
            z=g[g.screen_date==d];trade=z.trade_date.iloc[0];end=dates[j+1] if j+1<len(dates) else idx[-1]
            oldmask=(idx>=trade)&(idx<=end)
            mats['old'].loc[oldmask,z.ticker]=np.broadcast_to(z.Weight.to_numpy(),(oldmask.sum(),len(z)))
            nexttrade=g[g.screen_date==dates[j+1]].trade_date.iloc[0] if j+1<len(dates) else idx[-1]
            umask=(idx>trade)&(idx<=nexttrade)
            mats['union'].loc[umask,z.ticker]=np.broadcast_to(z.Weight.to_numpy(),(umask.sum(),len(z)))
            for reg,zr in z.groupby('Region'):
                cal=cals[reg]; entry=cal.date_to_session(pd.Timestamp(d),direction='next')
                if entry<=pd.Timestamp(d): entry=cal.next_session(entry)
                if j+1<len(dates):
                    exit_=cal.date_to_session(pd.Timestamp(dates[j+1]),direction='next')
                    if exit_<=pd.Timestamp(dates[j+1]):exit_=cal.next_session(exit_)
                else: exit_=idx[-1]
                mask=(idx>entry)&(idx<=exit_)
                mats['local'].loc[mask,zr.ticker]=np.broadcast_to(zr.Weight.to_numpy(),(mask.sum(),len(zr)))
                closes=[c.session_close(c.date_to_session(pd.Timestamp(d),direction='previous')) for c in cals.values()]
                available=max(closes); close=cal.session_close(entry)
                assert close>available
                for row in zr.itertuples():
                    logs.append(dict(variant=v,screen_date=d,signal_available_utc=str(available),region=reg,ticker=row.ticker,weight=row.Weight,legacy_trade_date=trade,entry_date=entry,entry_close_utc=str(close),exit_date=exit_,entry_excluded=True,exit_included=True))
        result[v]=mats
    return result,pd.DataFrame(logs)

def aggregate(ret,w,strict=False):
    active=w!=0; missing=active&ret.isna(); contribution=ret*w
    rr=contribution.sum(axis=1,skipna=True)
    if strict:rr=rr.mask(missing.any(axis=1))
    return rr,missing

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    data=Path(a.data);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    pp=data/'prices_original.parquet'
    assert hashlib.sha256(pp.read_bytes()).hexdigest()=='ab480ba3a1ce8a864aaad9d0dfeb4df16ed115d2bcf5e682e57c96c3de29c67d'
    p=pd.read_parquet(pp);p.index=p.index.astype('datetime64[ns]')
    csv=pd.read_csv(data/'prices_original.csv.gz',index_col=0,parse_dates=True,float_precision='round_trip');csv.index=csv.index.astype('datetime64[ns]');csv.columns=p.columns.copy()
    pd.testing.assert_frame_equal(p,csv,check_exact=True,check_freq=False)
    h=pd.read_csv(ROOT/'research/phase3/results/pinned/legacy_rerun_holdings.csv',parse_dates=['screen_date','trade_date'],float_precision='round_trip')
    assert len(h)==1026
    cals={r:xc.get_calendar(n) for r,n in [('US','XNYS'),('JP','XTKS')]}
    regions={t:'JP' if t.endswith('.T') else 'US' for t in p}
    safe,detail,counts=classify(p,regions,cals);counts.to_csv(out/'missing_summary.csv',index=False);detail.to_csv(out/'missing_cells.csv.gz',index=False)
    universe=json.loads((data/'universe.json').read_text());absent={r:sorted(set(ts)-set(p.columns)) for r,ts in universe.items()}
    (out/'absent_columns.json').write_text(json.dumps(absent,indent=2))
    weights,ledger=make_weights(p,h,cals);ledger.to_csv(out/'execution_ledger.csv',index=False)
    rets={'none':p.pct_change(fill_method=None),'pad':p.ffill().pct_change(fill_method=None),'holiday':safe.pct_change(fill_method=None)}
    specs={'A':('none','old',False),'B':('pad','old',False),'C':('none','local',False),'D':('pad','local',False),'C_union':('none','union',False),'D_union':('pad','union',False),'H_strict':('holiday','local',True)}
    daily=[];sums=[];exposure=[];missingrows=[];contrib=[];cross=[]
    for v,mats in weights.items():
        series={}
        for cell,(fill,timing,strict) in specs.items():
            w=mats[timing];ret=rets[fill].reindex(index=w.index,columns=w.columns);rr,missing=aggregate(ret,w,strict)
            series[cell]=rr
            daily.append(pd.DataFrame(dict(date=rr.index,variant=v,cell=cell,return_=rr.to_numpy(),nav=(1+rr).cumprod(skipna=False).to_numpy(),missing_active=missing.sum(axis=1).to_numpy(),missing_weight=w.where(missing,0).abs().sum(axis=1).to_numpy(),gross_weight=w.sum(axis=1).to_numpy())))
            sums.append(dict(variant=v,cell=cell,observations=len(rr),valid=int(rr.notna().sum()),missing_active_cells=int(missing.to_numpy().sum()),days_gross_above_one=int((w.sum(axis=1)>1+TOL).sum()),**metrics(rr)))
            ii,jj=np.where(missing.to_numpy())
            for i,j in zip(ii,jj):missingrows.append((v,cell,w.index[i],w.columns[j],float(w.iloc[i,j])))
            # Quarterly log wealth; missing quarters remain indeterminate.
            for q,rs in rr.groupby(rr.index.to_period('Q')):
                contrib.append(dict(variant=v,cell=cell,quarter=str(q),logwealth=float(np.log1p(rs).sum()) if rs.notna().all() else None,observations=len(rs)))
        A,B,C,D=[series[k] for k in 'ABCD'];cu=series['C_union'];du=series['D_union']
        for d in A.index:
            logs={k:float(np.log1p(series[k].loc[d])) for k in ['A','B','C','D','C_union','D_union']}
            cross.append(dict(variant=v,date=d,missing_log=logs['B']-logs['A'],timing_log=logs['C']-logs['A'],pure_boundary_log=logs['C_union']-logs['A'],calendar_log=logs['C']-logs['C_union'],interaction_log=logs['D']-logs['B']-logs['C']+logs['A']))
    df=pd.concat(daily,ignore_index=True);df.to_csv(out/'daily.csv.gz',index=False)
    sm=pd.DataFrame(sums);sm.to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(missingrows,columns=['variant','cell','date','ticker','weight']).to_csv(out/'suppressed_or_blocked_returns.csv.gz',index=False)
    pd.DataFrame(contrib).to_csv(out/'quarterly_logwealth.csv',index=False)
    pd.DataFrame(cross).to_csv(out/'factorial_daily.csv.gz',index=False)
    replay=[]
    for cell,env in [('A','pinned'),('B','initial')]:
        old=pd.read_csv(ROOT/f'research/phase3/results/{env}/legacy_daily_returns.csv',index_col=0,parse_dates=True,float_precision='round_trip')
        for v in weights:
            r=df[(df.variant==v)&(df.cell==cell)].set_index('date').return_
            assert r.index.equals(old.index)
            delta=float(np.max(np.abs(r.to_numpy()-old[v].to_numpy())))
            replay.append(dict(cell=cell,variant=v,max_abs_daily_delta=delta,status='PASS' if delta<=TOL else 'FAIL'))
    pd.DataFrame(replay).to_csv(out/'legacy_reproduction.csv',index=False)
    assert all(x['status']=='PASS' for x in replay)
    (out/'environment.json').write_text(json.dumps(dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,exchange_calendars=xc.__version__,protocol_sha256=hashlib.sha256((HERE/'protocol.md').read_bytes()).hexdigest(),price_sha256=hashlib.sha256(pp.read_bytes()).hexdigest()),indent=2))
    print(sm[['variant','cell','CAGR','missing_active_cells','days_gross_above_one']].to_string(index=False))

if __name__=='__main__':main()
