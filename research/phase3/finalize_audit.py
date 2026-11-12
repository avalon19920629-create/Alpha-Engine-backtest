"""Supplemental evidence from completed E30/E31 outputs; no prediction analysis."""
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import alpha_engine_backtest as ae
from research.phase3.signals import strict_signals

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    data,out=Path(a.data),Path(a.out)
    p=pd.read_parquet(data/'prices_original.parquet')
    old=pd.read_csv(out/'legacy_signal_rows.csv',parse_dates=['date'],float_precision='round_trip')
    # Prior E01 used available-window ordinary momentum; base Composite_Ret always
    # uses all three terms even at len==252. Keep both definitions, do not relabel.
    old['m0_available_windows']=old.m0.copy()
    for idx,row in old[~old.full_windows].iterrows():
        s=p.loc[:row.date,row.ticker].ffill().dropna()
        terms=[(w,s.iloc[-1]/s.iloc[-h]-1) for h,w in [(63,1),(126,2),(252,3)] if len(s)>h]
        old.loc[idx,'m0_available_windows']=sum(w*r for w,r in terms)/sum(w for w,r in terms)
    rows=[]
    for (d,r),g in old.groupby(['date','region']):
        for label,col in [('base_Composite_Ret','m0'),('available_window_ordinary','m0_available_windows')]:
            rows.append({'date':d,'region':r,'definition':label,'rows':len(g),'rank_mismatches':int((g[col].rank()!=g.m1.rank()).sum())})
    old[~old.full_windows].to_csv(out/'legacy_incomplete_window_definition_bridge.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'legacy_m0_definition_bridge.csv',index=False)
    saved=pd.read_csv(data/'real_selected_tickers.csv',float_precision='round_trip')
    rerun=pd.read_csv(out/'legacy_rerun_holdings.csv',float_precision='round_trip')
    keys=['variant','screen_date','ticker'];joined=saved.merge(rerun,on=keys,suffixes=('_saved','_rerun'),validate='one_to_one')
    numeric=[]
    for c in ['Weight','Final_Score','residual_raw','residual_score','base_score','Composite_Ret','Total_Score','Volatility']:
        diff=(joined[c+'_saved']-joined[c+'_rerun']).abs()
        numeric.append({'field':c,'rows':len(joined),'max_abs_error':float(diff.max()),'pass_1e10':bool((diff<=1e-10).all())})
    pd.DataFrame(numeric).to_csv(out/'legacy_selected_numeric_comparison.csv',index=False)
    # Real-data cutoff check: append/mutate future values and retain identical asof.
    future=[];u=json.loads((data/'universe.json').read_text())
    mode=ae.build_benchmark_modes()['broad_default']
    for d in ['2022-03-31','2024-06-28','2026-09-30']:
        mutated=p.copy();mutated.loc[mutated.index>pd.Timestamp(d)]*=7
        for region in ['US','JP']:
            names=[t for t in u[region] if t in p]
            b0=ae.score_universe(p,names,d);b1=ae.score_universe(mutated,names,d)
            r0=ae.compute_residual_momentum_score(p,b0.index,d,region,benchmark_mode=mode)
            r1=ae.compute_residual_momentum_score(mutated,b1.index,d,region,benchmark_mode=mode)
            pd.testing.assert_frame_equal(b0,b1,check_exact=True);pd.testing.assert_frame_equal(r0,r1,check_exact=True)
            future.append({'date':d,'region':region,'base_and_simple_exact_invariant':True})
    pd.DataFrame(future).to_csv(out/'real_future_cutoff_checks.csv',index=False)
    print(json.dumps({'matched_saved_rows':len(joined),'all_selected_numeric_match':all(r['pass_1e10'] for r in numeric),
       'available_window_rank_mismatches':int(pd.DataFrame(rows).query("definition=='available_window_ordinary'").rank_mismatches.sum()),'real_future_cutoff_checks':len(future)}))

if __name__=='__main__':main()
