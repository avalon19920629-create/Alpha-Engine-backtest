"""Read-only verification of baseline, replay and cross-environment outputs."""
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
import alpha_engine_backtest as ae
from repair import make_weights,aggregate
import exchange_calendars as xc
TOL=1e-10
c='52531ed2d9ce1ea275fb77bbc7e18bc582655af0';parent='075f0985c76c2212552732534d7f3c741270a80d'
def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args])
def blob(path):return git('show',c+':'+path)
assert git('rev-parse',c+'^').decode().strip()==parent
manifest=json.loads(blob('research/phase3/evidence_manifest.json'))
assert len(manifest)==73
for f,h in manifest.items():assert hashlib.sha256((ROOT/'research/phase3'/f).read_bytes()).hexdigest()==h
old=git('ls-tree','-r','--name-only',c).decode().splitlines()
assert all((ROOT/f).read_bytes()==blob(f) for f in old)
lines=blob('research/phase3/protected_before.sha256').decode().splitlines();assert len(lines)==121
for line in lines:
 h,f=line.split('  ',1);assert hashlib.sha256(blob(f)).hexdigest()==h;assert blob(f)==git('show',parent+':'+f)
refs=git('ls-remote','origin','refs/heads/main','refs/heads/research/phase3-selection-20261113').decode();assert parent+'\trefs/heads/main' in refs;assert c+'\trefs/heads/research/phase3-selection-20261113' in refs
(HERE/'remote_refs_final.txt').write_text(refs)
(HERE/'protected_final.json').write_text(json.dumps(dict(status='PASS',all_baseline_files=len(old),protected=121,manifest=73,remote_main=parent,remote_research=c),indent=2))
checks=[]
for name in ['daily.csv.gz','metrics.csv','execution_ledger.csv','missing_summary.csv','factorial_daily.csv.gz','quarterly_logwealth.csv','suppressed_or_blocked_returns.csv.gz']:
 a=pd.read_csv(HERE/'results_pinned_v1'/name,float_precision='round_trip');b=pd.read_csv(HERE/'results_initial_v1'/name,float_precision='round_trip')
 assert a.shape==b.shape and a.columns.tolist()==b.columns.tolist()
 maxerr=0.
 for col in a:
  if pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(b[col]):
   assert np.array_equal(a[col].isna(),b[col].isna());av=a[col].dropna().to_numpy(dtype=float);bv=b[col].dropna().to_numpy(dtype=float)
   delta=float(np.max(abs(av-bv))) if len(av) else 0.;maxerr=max(maxerr,delta);assert delta<=TOL,(name,col,delta)
  else:assert a[col].fillna('<NA>').tolist()==b[col].fillna('<NA>').tolist(),(name,col)
 checks.append(dict(file=name,rows=len(a),max_abs_delta=maxerr,status='PASS'))
(HERE/'environment_comparison.json').write_text(json.dumps(checks,indent=2))
daily=pd.read_csv(HERE/'results_pinned_v1/daily.csv.gz',parse_dates=['date'],float_precision='round_trip')
metric_checks=[]
for cell,env in [('A','pinned'),('B','initial')]:
 ref=pd.read_csv(ROOT/f'research/phase3/results/{env}/legacy_metric_comparison.csv',float_precision='round_trip')
 for v,g in daily[daily.cell==cell].groupby('variant'):
  rr=g.set_index('date').return_;m=ae.metrics(rr);m['Conventional_Sharpe_RF0']=float(rr.mean()/rr.std()*np.sqrt(252))
  for row in ref[ref.variant==v].itertuples():
   if row.metric not in m:continue
   delta=abs(m[row.metric]-row.actual);assert delta<=TOL,(cell,v,row.metric,delta)
   metric_checks.append(dict(cell=cell,variant=v,metric=row.metric,max_abs_delta=delta,status='PASS'))
pd.DataFrame(metric_checks).to_csv(HERE/'all_legacy_metric_checks.csv',index=False)
# Independent scalar ownership oracle, not matrix construction: last entry strictly before d.
p=pd.read_parquet(ROOT.parent/'e30_data/phase1-frozen-price-csv-and-sector-map/prices_original.parquet')
p.index=p.index.astype('datetime64[ns]');returns=p.ffill().pct_change(fill_method=None)
ledger=pd.read_csv(HERE/'results_pinned_v1/execution_ledger.csv',parse_dates=['entry_date','exit_date','screen_date'])
oracle=[]
for v,g in daily[daily.cell=='D'].groupby('variant'):
 l=ledger[ledger.variant==v];parts=[]
 for d in g.date:
  active=l[(l.entry_date<d)&(l.exit_date>=d)]
  assert not active.duplicated(['region','ticker']).any()
  val=sum(float(returns.loc[d,r.ticker])*r.weight if pd.notna(returns.loc[d,r.ticker]) else 0. for r in active.itertuples())
  parts.append(val)
 delta=float(np.max(np.abs(g.return_.to_numpy()-np.array(parts))));assert delta<=TOL
 oracle.append(dict(variant=v,max_abs_delta=delta,status='PASS'))
(HERE/'independent_ownership_oracle.json').write_text(json.dumps(oracle,indent=2))
print('PASS baseline protection, remote refs, two-environment outputs, all legacy metrics, scalar ownership oracle')
