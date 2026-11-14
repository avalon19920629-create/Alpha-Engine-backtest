"""Read-only evidence audit; output only beside this script. No network or returns simulation."""
from pathlib import Path
import subprocess,json,hashlib,math,sys
import pandas as pd,numpy as np
H=Path(__file__).resolve().parent;R=H.parents[1];P=R/'research/phase3_identity_pit_20261114';C=R/'research/phase3_cash_ledger_20261114';S=R/'research/phase3'
BASE='80af7f14446c2f33285478ee7ecb48946a6af937';PARENT='27144bc66d8e5ba583c960107c70f4c89b7334a9';MAIN='075f0985c76c2212552732534d7f3c741270a80d';TOL=1e-10
sha=lambda b:hashlib.sha256(b).hexdigest()
def git(*a):return subprocess.check_output(['git','-C',str(R),*a])
def read(p):return pd.read_csv(p,float_precision='round_trip')
assert git('rev-parse','origin/research/phase3-identity-pit-20261114').decode().strip()==BASE
assert git('rev-parse','origin/main').decode().strip()==MAIN
assert git('rev-parse',BASE+'^').decode().strip()==PARENT
assert git('merge-base',BASE,'52531ed2d9ce1ea275fb77bbc7e18bc582655af0').decode().strip()=='52531ed2d9ce1ea275fb77bbc7e18bc582655af0'
changes=git('diff','--name-status',PARENT,BASE).decode().splitlines();assert len(changes)==57 and all(x.startswith('A\tresearch/phase3_identity_pit_20261114/') for x in changes)
paths=git('ls-tree','-r','--name-only',BASE).decode().splitlines();assert len(paths)==354
protected={}
for p in paths:
 blob=git('show',BASE+':'+p);assert blob==(R/p).read_bytes();protected[p]=sha(blob)
for p in git('ls-tree','-r','--name-only',PARENT).decode().splitlines():assert git('show',PARENT+':'+p)==(R/p).read_bytes()
counts={}
for folder,name in [('phase3','evidence_manifest.json'),('phase3_repair_20261114','repair_manifest.json'),('phase3_accounting_design_20261114','design_manifest.json'),('phase3_cash_ledger_20261114','evidence_manifest.json'),('phase3_identity_pit_20261114','evidence_manifest.json')]:
 root=R/'research'/folder;m=json.loads((root/name).read_text());assert all(sha((root/f).read_bytes())==v for f,v in m.items());counts[folder]=len(m)
(H/'protected_before.json').write_text(json.dumps(protected,indent=2))
(H/'preservation.json').write_text(json.dumps(dict(status='PASS',remote_branch=BASE,parent=PARENT,main=MAIN,prior_files_unchanged=297,protected_now=354,new_files=57,manifest_counts=counts),indent=2))
# Independent aggregation and diagnostic deterministic sorting from archived scores.
p=read(P/'results_pinned/permutations.csv.gz');q=read(P/'results_initial/permutations.csv.gz');assert p.equals(q)
s=read(P/'results_pinned/score_frames.csv.gz');s2=read(P/'results_initial/score_frames.csv.gz');assert s.equals(s2)
t=read(P/'results_pinned/ties.csv');keys=['date','region','R','k'];rules={};case=[]
for rule,g in p.groupby('rule'):
 vary=g.groupby(keys).top_set.nunique().gt(1)
 rules[rule]={'selection_set_sensitive':int(vary.sum()),'full_order_sensitive_regional_selections':int(g.groupby(keys).ranked_all.nunique().gt(1).sum()),'full_order_sensitive_frames':int(g.groupby(['date','region','R']).ranked_all.nunique().gt(1).sum())}
 for key in vary[vary].index:case.append(dict(zip(keys,key),rule=rule,sets=' / '.join(sorted(g.set_index(keys).loc[key].top_set.unique()))))
pd.DataFrame(case).to_csv(H/'tie_sensitive_cases.csv',index=False)
checks=0
for (date,region,ratio),g in s.groupby(['date','region','R']):
 for rule,cols,asc in [('diagnostic_ticker',['Final_Score','ticker'],[False,True]),('phase2_components',['Final_Score','residual_score','base_score','ticker'],[False,False,False,True])]:
  order=g.sort_values(cols,ascending=asc).ticker.tolist();saved=p[(p.date==date)&(p.region==region)&(p.R==ratio)&(p.rule==rule)]
  assert (saved.ranked_all=='|'.join(order)).all();checks+=1
assert len(t)==699 and int(t.boundary3.sum()+t.boundary6.sum())==4
hold=read(S/'results/pinned/legacy_rerun_holdings.csv');eq=read(S/'results/pinned/research_m0_m1_rows.csv');ex=read(S/'results/pinned/research_exclusions.csv')
identity={'legacy_score_rows':len(s),'legacy_unique_tickers':s.ticker.nunique(),'legacy_region_dates':len(s[['date','region']].drop_duplicates()),'selected_rows':len(hold),'selected_unique_tickers':hold.ticker.nunique(),'selected_region_dates':len(hold[['screen_date','Region']].drop_duplicates()),'selected_portfolios':len(hold[['screen_date','variant']].drop_duplicates()),'E31_rows':len(eq),'E31_unique_tickers':eq.ticker.nunique(),'E31_market_months':len(eq[['date','region']].drop_duplicates()),'certified_identity_table_present':False,'interpretation':'all these rows are uncertified, not proved erroneous; overlapping model rows are not independent samples'}
ex.groupby(['region','reason']).agg(rows=('ticker','size'),tickers=('ticker','nunique'),dates=('date','nunique')).reset_index().to_csv(H/'exclusion_scope.csv',index=False)
cov=read(P/'sources/historical_membership_coverage_extended.csv');cov['count_residual_not_explained']=cov.official_historical_member_count_proxy-cov.historical_eligible_total_partial-cov.historical_names_uncovered_total
cov.to_csv(H/'PIT_scope_by_date.csv',index=False)
rec=read(P/'sources/historical_symbol_quote_recovery.csv');freq=read(P/'sources/phase2_priority/priority/unrecoverable_ticker_frequency.csv');un=read(P/'sources/next_gate_unresolved_54_priority.csv');events=read(P/'sources/blind_control_six_independently_verified_events.csv')
assert len(rec)==102 and (rec.status=='historical_price_usable_some_dates').sum()==48 and len(un)==54
assert set(freq.ticker)-set(un.ticker)=={'OGN'}
summary={'preservation':'PASS','tie_rules':rules,'independent_sorted_frames_checked':checks,'identity_scope':identity,'PIT':{'screen_dates':len(cov),'dates_with_uncovered':int((cov.historical_names_uncovered_total>0).sum()),'uncovered_min':int(cov.historical_names_uncovered_total.min()),'uncovered_max':int(cov.historical_names_uncovered_total.max()),'coverage_min':float(cov.fraction_members_with_historical_quotes_partial.min()),'coverage_mean':float(cov.fraction_members_with_historical_quotes_partial.mean()),'nonzero_count_residual_dates':int((cov.count_residual_not_explained!=0).sum()),'count_residual_values':sorted(cov.count_residual_not_explained.unique().tolist()),'information_availability_verified_dates':0,'missing_original_YAML_in_archived_P1':True},'former_symbols':{'total':102,'some_usable':48,'none_usable':54,'frequency_table':len(freq),'extra_symbol':'OGN'},'events':{'sample_size':len(events),'classified':int(events.classification_done.sum()),'terminal_PnL_verified':int(events.historical_PIT_price_and_termination_PnL_validated.sum()),'full_population_event_count':None}}
# E31 independent numeric, rank, and top-set test; diagnostic ticker only.
echecks=[]
for (date,reg),g in eq.groupby(['date','region']):
 err=float((g.m0-g.m1-g.benchmark_constant).abs().max());ranks=g.m0.rank().equals(g.m1.rank());k=math.ceil(len(g)*.1)
 a=g.sort_values(['m0','ticker'],ascending=[False,True]).head(k).ticker;b=g.sort_values(['m1','ticker'],ascending=[False,True]).head(k).ticker
 assert err<=TOL and ranks and set(a)==set(b)
 echecks.append({'date':date,'region':reg,'rows':len(g),'constant_error':err,'ranks_equal':ranks,'top_set_equal':True})
pd.DataFrame(echecks).to_csv(H/'E31_recheck.csv',index=False);summary['E31']={'rows':len(eq),'market_months':len(echecks),'max_constant_error':max(x['constant_error'] for x in echecks),'excluded_rows':len(ex)}
# Saved ledger chronology/conservation, numeric and missing masks cross env.
errs=[]
for f in ['daily.csv.gz','events.csv.gz','metrics.csv','accounting_checks.csv','order_bridge.csv','legacy_comparison.csv']:
 a=read(C/'results_pinned'/f);b=read(C/'results_initial'/f);assert a.shape==b.shape and a.columns.equals(b.columns);err=0.
 for col in a:
  assert a[col].isna().equals(b[col].isna())
  if pd.api.types.is_numeric_dtype(a[col]):
   x=(a[col].astype(float)-b[col].astype(float)).abs().max();err=max(err,float(x) if pd.notna(x) else 0.)
  else:assert a[col].fillna('<NA>').equals(b[col].fillna('<NA>'))
 assert err<=TOL;errs.append({'file':f,'rows':len(a),'max_error':err})
l=read(C/'results_pinned/events.csv.gz');d=read(C/'results_pinned/daily.csv.gz');lc=[]
for v,g in l.groupby('variant'):
 tr=g[g.event_type.isin(['BUY','SELL','RESERVE'])];er=float((tr.nav_after-tr.nav_before).abs().max());end=d[d.variant==v].nav.iloc[-1];tel=float(abs(end-1-g[g.event_type=='MARK'].market_pnl.sum()))
 buy=g[g.event_type=='BUY'];sell=g[g.event_type=='SELL'];assert (pd.to_datetime(buy.event_time_utc)>pd.to_datetime(buy.funding_time)).all();assert (pd.to_datetime(sell.event_time_utc)>pd.to_datetime(sell.signal_time)).all();assert er<=TOL and tel<=TOL
 assert g.cash_available.min()>=-TOL and g.investment_fraction.max()<=1+TOL
 lc.append({'variant':v,'trade_NAV_error':er,'telescoping_error':tel,'minimum_cash_available':g.cash_available.min(),'maximum_investment_fraction':g.investment_fraction.max()})
pd.DataFrame(lc).to_csv(H/'ledger_recheck.csv',index=False);summary['ledger_environment']=errs
(H/'diagnostic_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2));print('PASS: diagnostic assertions (not E30 pass)')
