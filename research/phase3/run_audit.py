"""Frozen-data E30/E31 audit. Execute from repository root. Never downloads prices."""
import argparse, hashlib, json, platform, sys, subprocess, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import exchange_calendars as xc

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import alpha_engine_backtest as ae
from research.phase3.signals import strict_signals, legacy_simple_independent, legacy_beta_independent, TOL

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,obj): p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str,allow_nan=False)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--original',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();data=Path(a.data);orig=Path(a.original);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((data/'metadata.json').read_text());u=json.loads((data/'universe.json').read_text())
    expected_hash='ab480ba3a1ce8a864aaad9d0dfeb4df16ed115d2bcf5e682e57c96c3de29c67d'
    hashes={p.name:sha(p) for p in data.iterdir() if p.is_file()}
    originals={p.name:sha(p) for p in orig.iterdir() if p.is_file()}
    dump(out/'input_manifest.json',{'data':hashes,'original_archive_members':originals,'expected_price_sha256':expected_hash,
        'price_hash_pass':hashes['prices_original.parquet']==expected_hash,'cross_bundle_exact':{n:hashes[n]==h for n,h in originals.items() if n in hashes}})
    if hashes['prices_original.parquet']!=expected_hash: raise ValueError('frozen hash mismatch')
    p=pd.read_parquet(data/'prices_original.parquet')
    csv=pd.read_csv(data/'prices_original.csv.gz',index_col=0,parse_dates=True,float_precision='round_trip')
    # CSV parser date-unit/string dtype metadata can differ across pandas versions.
    # Check labels before normalizing representation; never alter any price values.
    assert p.index.tolist()==csv.index.tolist() and p.columns.tolist()==csv.columns.tolist()
    csv.index=csv.index.astype(p.index.dtype)
    csv.columns=p.columns.copy()
    pd.testing.assert_frame_equal(p,csv,check_exact=True,check_freq=False)
    if p.index.has_duplicates or p.columns.has_duplicates or not p.index.is_monotonic_increasing: raise ValueError('invalid table axes')
    env={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'exchange_calendars':xc.__version__,
         'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         'design_sha256':sha(ROOT/'research/phase3/design_v1.0.md'),'legacy_code_sha256':sha(ROOT/'alpha_engine_backtest.py'),
         'command':sys.argv,'frozen_metadata_versions':{k:v for k,v in meta.items() if k.endswith('_version')}}
    dump(out/'environment.json',env)
    selected=pd.read_csv(data/'real_selected_tickers.csv',parse_dates=['screen_date','trade_date'],float_precision='round_trip')
    refs=pd.read_csv(data/'real_six_cell_comparison.csv').set_index('variant')
    usable={k:[t for t in v if t in p and p[t].notna().sum()>=252] for k,v in u.items()}
    mode=ae.build_benchmark_modes()['broad_default']
    metrics,holdings,diffs,retframes=[],[],[],[]
    # Exactly the six archived variants. This is reproduction, not R/N search.
    for r in (0,60,100):
        for n in (6,12):
            name=f'R{r}_N{n}';v={'name':name,'base_weight':1-r/100,'residual_weight':r/100,'total_holdings':n,'us_holdings':n//2,'jp_holdings':n//2}
            returns,sel,_,turn=ae.run_residual_concentration_variant(p,usable['US'],usable['JP'],'2022-01-01','2026-11-11',v,mode)
            holdings.append(sel);retframes.append(returns.rename(name))
            actual=ae.metrics(returns);actual['Conventional_Sharpe_RF0']=float(returns.mean()/returns.std()*np.sqrt(252))
            actual['Mean_Turnover']=float(turn.turnover.mean())
            for k,x in actual.items():
                if k in refs: metrics.append({'variant':name,'metric':k,'reference':refs.loc[name,k],'actual':x,'delta':x-refs.loc[name,k],'pass_1e10':abs(x-refs.loc[name,k])<=TOL})
            old=selected[selected.variant==name]
            for d,g in old.groupby('screen_date'):
                ng=sel[sel.screen_date==d]
                diffs.append({'variant':name,'screen_date':d,'set_equal':set(g.ticker)==set(ng.ticker),'order_equal':g.ticker.tolist()==ng.ticker.tolist(),
                              'only_saved':'|'.join(sorted(set(g.ticker)-set(ng.ticker))),'only_rerun':'|'.join(sorted(set(ng.ticker)-set(g.ticker)))})
            print('legacy',name,'complete',flush=True)
    pd.DataFrame(metrics).to_csv(out/'legacy_metric_comparison.csv',index=False)
    pd.concat(holdings).to_csv(out/'legacy_rerun_holdings.csv',index=False)
    pd.DataFrame(diffs).to_csv(out/'legacy_selection_comparison.csv',index=False)
    pd.concat(retframes,axis=1).to_csv(out/'legacy_daily_returns.csv')
    # Replay archived holdings, explicit no-fill/pad, to isolate selection vs accounting.
    replay=[]
    for fill in (None,'pad'):
        rets=(p.ffill() if fill=='pad' else p).pct_change(fill_method=None)
        for name,g in selected.groupby('variant'):
            rr=pd.Series(0.,index=p.loc['2022-01-01':'2026-11-11'].index)
            dates=sorted(g.screen_date.unique())
            for i,d in enumerate(dates):
                z=g[g.screen_date==d];end=dates[i+1] if i+1<len(dates) else pd.Timestamp('2026-11-11')
                idx=rr.index[(rr.index>=z.trade_date.iloc[0])&(rr.index<=end)]
                rr.loc[idx]=rets.reindex(idx)[z.ticker].mul(z.set_index('ticker').Weight).sum(axis=1)
            m=ae.metrics(rr)
            replay.append({'variant':name,'fill_method':str(fill),'CAGR':m['CAGR'],'delta_CAGR':m['CAGR']-refs.loc[name,'CAGR'],
                           'Max_Drawdown':m['Max_Drawdown'],'delta_MDD':m['Max_Drawdown']-refs.loc[name,'Max_Drawdown']})
    pd.DataFrame(replay).to_csv(out/'saved_holdings_accounting_replay.csv',index=False)
    # Old implementation same-window invariance and independent L-beta on every eligible candidate.
    oldrows,beta_rows,ties=[],[],[]
    for d in sorted(selected.screen_date.unique()):
        hist=p.loc[:d].ffill()
        for region,ts in usable.items():
            bname=ae._resolve_benchmark(hist,region,mode);b=hist[bname].dropna()
            base=ae.score_universe(p,ts,d);res=ae.compute_residual_momentum_score(p,base.index,d,region,benchmark_mode=mode)
            for t,row in base.iterrows():
                s=hist[t].dropna();full=len(s)>252
                raw=legacy_simple_independent(s,b)
                oldrows.append({'date':d,'region':region,'ticker':t,'full_windows':full,'m0':row.Composite_Ret,'m1':res.loc[t,'residual_raw'],'independent_error':abs(raw-res.loc[t,'residual_raw'])})
                for h in (63,126,252):
                    x=ae.compute_beta_adjusted_residual_score(s,b,h);y=legacy_beta_independent(s,b,h)
                    beta_rows.append({'date':d,'region':region,'ticker':t,'window':h,'legacy':x,'independent':y,'absolute_error':abs(x-y)})
            for r in (0,60,100):
                score=(1-r/100)*ae._rank01(base.Total_Score)+r/100*res.residual_score
                for scoreval,g in score.groupby(score):
                    if len(g)>1: ties.append({'date':d,'region':region,'R':r,'score':scoreval,'tickers':'|'.join(g.index)})
    old=pd.DataFrame(oldrows);old.to_csv(out/'legacy_signal_rows.csv',index=False)
    pd.DataFrame(beta_rows).to_csv(out/'legacy_beta_independent.csv',index=False)
    pd.DataFrame(ties).to_csv(out/'legacy_ties.csv',index=False)
    oldsum=[]
    for (d,reg),g in old.groupby(['date','region']):
        for full in (False,True):
            q=g[g.full_windows] if full else g
            oldsum.append({'date':d,'region':reg,'common_full_windows_only':full,'rows':len(q),
                           'rank_mismatches':int((q.m0.rank()!=q.m1.rank()).sum()),'max_independent_error':q.independent_error.max()})
    pd.DataFrame(oldsum).to_csv(out/'legacy_equivalence_summary.csv',index=False)
    print('legacy formulas complete',flush=True)
    # Strict research: local sessions, 253 observations, no imputation, no forward labels.
    allrows,exclusions,summaries,calendar_rows,quality=[],[],[],[],[]
    for region,bname,calname in [('US','SPY','XNYS'),('JP','1306.T','XTKS')]:
        cal=xc.get_calendar(calname,start='2020-06-22',end='2026-11-30')
        sessions=cal.sessions_in_range(p.index.min(),p.index.max())
        for d in p.index:
            if d not in sessions:
                present=[t for t in u[region] if t in p and pd.notna(p.loc[d,t])]
                calendar_rows.append({'region':region,'date':d,'reason':'non_session_input_row','non_null_stock_count':len(present)})
        for d in sessions.difference(p.index): calendar_rows.append({'region':region,'date':d,'reason':'exchange_session_absent_in_input','non_null_stock_count':0})
        for t in u[region]:
            if t not in p:
                quality.append({'region':region,'ticker':t,'missing_column':True});continue
            s=p[t].reindex(sessions);valid=s.dropna()
            within=s.loc[valid.index.min():valid.index.max()] if len(valid) else s
            quality.append({'region':region,'ticker':t,'missing_column':False,'first_valid':str(valid.index.min()),'last_valid':str(valid.index.max()),
                            'observations':len(valid),'internal_missing_sessions':int(within.isna().sum()),'nonpositive':int((s<=0).sum()),
                            'tail_missing_sessions':int(s.loc[valid.index.max():].isna().sum()) if len(valid) else len(s)})
        month_dates=pd.Series(sessions,index=sessions).groupby(sessions.to_period('M')).max()
        month_dates=month_dates[(month_dates>='2022-01-01')&(month_dates<'2026-11-01')] # last partial month not a month end
        for d in month_dates:
            f,e=strict_signals(p,u[region],bname,sessions,d)
            e=e.assign(date=d,region=region);exclusions.append(e)
            if f.empty:
                summaries.append({'date':d,'region':region,'eligible':0,'status':'NO_VALID_ROWS'});continue
            f=f.assign(date=d,region=region);allrows.append(f)
            rankerr=int((f.m0.rank(method='average')!=f.m1.rank(method='average')).sum())
            k=int(np.ceil(len(f)*.1))
            # Ticker order is a diagnostic only, never substituted for permanent IDs.
            a0=f.sort_values(['m0','ticker'],ascending=[False,True]);a1=f.sort_values(['m1','ticker'],ascending=[False,True])
            boundary_tie=(len(f)>k and (a0.m0.iloc[k-1]==a0.m0.iloc[k] or a1.m1.iloc[k-1]==a1.m1.iloc[k]))
            err=float(((f.m0-f.m1)-f.benchmark_constant).abs().max())
            summaries.append({'date':d,'region':region,'eligible':len(f),'rank_mismatches':rankerr,'spearman':float(f.m0.rank().corr(f.m1.rank())),
                              'constant_error':err,'top10_set_equal':set(a0.head(k).ticker)==set(a1.head(k).ticker),'boundary_tie':bool(boundary_tie),
                              'permanent_id_selection_certified':False,'status':'PASS_MATH' if rankerr==0 and err<=TOL else 'FAIL'})
    pd.concat(allrows).to_csv(out/'research_m0_m1_rows.csv',index=False)
    pd.concat(exclusions).to_csv(out/'research_exclusions.csv',index=False)
    pd.DataFrame(summaries).to_csv(out/'research_equivalence_summary.csv',index=False)
    pd.DataFrame(calendar_rows).to_csv(out/'calendar_discrepancies.csv',index=False)
    pd.DataFrame(quality).to_csv(out/'price_quality_by_ticker.csv',index=False)
    ms=pd.DataFrame(metrics);ds=pd.DataFrame(diffs);ss=pd.DataFrame(summaries)
    summary={'input_hash':'PASS','csv_parquet_exact':'PASS','legacy_metrics_all_match':bool(ms.pass_1e10.all()),
             'legacy_selection_set_mismatch_count':int((~ds.set_equal).sum()),'legacy_selection_order_mismatch_count':int((~ds.order_equal).sum()),
             'legacy_beta_max_error':float(pd.DataFrame(beta_rows).absolute_error.max()),
             'research_month_market_count':len(ss),'research_rows':sum(ss.eligible),'research_all_math_pass':bool(ss.status.eq('PASS_MATH').all()),
             'research_constant_error_max':float(ss.constant_error.max()),'research_boundary_tie_count':int(ss.boundary_tie.sum()),
             'E32_executed':False,'forward_labels_computed':False,'permanent_company_ids_available':False,
             'PIT_membership_available_in_inputs':False,'terminal_event_ledger_available':False,'unresolved_data_gate':True}
    dump(out/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    main()
