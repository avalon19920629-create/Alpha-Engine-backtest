from pathlib import Path
import ast,types,json,hashlib
import pandas as pd,numpy as np,yaml
H=Path(__file__).resolve().parent;R=H.parents[1];src=H/'sources';d=R.parent/'e30_data/phase1-frozen-price-csv-and-sector-map';q=R.parent/'p1_inputs/alpha_engine_phase2_survivorship_quantification_bundle_20261112/phase2_verified_former_member_quotes_expanded/downloaded_historical_former_quotes.parquet'
assert hashlib.sha256((d/'prices_original.parquet').read_bytes()).hexdigest()=='ab480ba3a1ce8a864aaad9d0dfeb4df16ed115d2bcf5e682e57c96c3de29c67d'
assert hashlib.sha256(q.read_bytes()).hexdigest()=='017f8b99ffbae836dba44ce0b361d86aef28fdc4eaf5e779a0403fd49ecaedf8'
for y in range(2022,2027):(src/f'sp500-ticker-changes-{y}.yaml').write_bytes((src/f'{y}.yaml').read_bytes())
def funcs(path,names,ns):
 tree=ast.parse(path.read_text());code=ast.Module(body=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names],type_ignores=[]);exec(compile(code,str(path),'exec'),ns)
ns={'pd':pd,'np':np,'yaml':yaml,'Path':Path,'SRC':src,'sha256':lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()};funcs(src/'phase2_pit_at_recovery.py',{'canonical','membership_asof'},ns);pit=types.SimpleNamespace(canonical=ns['canonical'])
rn={'pd':pd,'np':np,'pit':pit,'MAX_STALE_CALENDAR_DAYS':8};funcs(src/'phase2_restore_former_members.py',{'is_available_at','build_calendar_coverage'},rn)
f=pd.read_parquet(d/'prices_original.parquet');f.index=f.index.astype('datetime64[ns]');rr=pd.read_parquet(q);rr.index=rr.index.astype('datetime64[ns]');data=pd.concat([f,rr.reindex(f.index)],axis=1);assert data.columns.is_unique
u=json.loads((d/'universe.json').read_text());orig=[x for x in u['US'] if x in f and f[x].notna().sum()>=252];a=pd.read_csv(R/'research/phase3_identity_pit_20261114/sources/historical_symbol_quote_recovery.csv');rec=a[a.eligible_screen_dates>0].ticker.tolist()
old=pd.read_csv(R/'research/phase3_identity_pit_20261114/sources/historical_membership_coverage_extended.csv');cache={};results=[];sets=[];reasons=[]
for row in old.itertuples():
 t=pd.Timestamp(row.screen_date);m=ns['membership_asof'](t,cache);calc,_=rn['build_calendar_coverage'](data,orig,rec,[t],{t:m});got=calc.iloc[0]
 for col in old.columns:
  x=got[col];y=getattr(row,col)
  if isinstance(x,(float,np.floating)):assert abs(x-y)<1e-12,(t,col,x,y)
  else:assert x==y,(t,col,x,y)
 eligible={x for x in orig+rec if ns['canonical'](x) in m and rn['is_available_at'](data,x,t)};uncovered=set(got.uncovered_symbols.split(';'));rest=m-eligible-uncovered
 original_ineligible={x for x in orig if ns['canonical'](x) in m and not rn['is_available_at'](data,x,t)}
 for symbol in sorted(original_ineligible):
  hist=data.loc[:t,symbol].dropna();reasons.append({'date':str(t.date()),'symbol':symbol,'observations':len(hist),'stale_calendar_days':int((t-hist.index[-1]).days) if len(hist) else None,'under_252':len(hist)<252})
 assert rest==original_ineligible|({'FB'} if 'FB' in m and 'META' in orig else set())
 results.append({'date':row.screen_date,'proxy':len(m),'eligible':len(eligible),'uncovered':len(uncovered),'residual':len(rest),'FB_alias_exception':int('FB' in rest),'original_pool_ineligible':';'.join(sorted(original_ineligible)),'residual_symbols':';'.join(sorted(rest)),'status':'PASS_EXACT_OLD_TABLE_REPRODUCTION'})
 sets.append({'date':row.screen_date,'members':sorted(m),'eligible':sorted(eligible),'uncovered':sorted(uncovered),'residual':sorted(rest)})
 print(row.screen_date,sorted(rest),flush=True)
pd.DataFrame(results).to_csv(H/'set_reconciliation.csv',index=False);(H/'set_evidence.json').write_text(json.dumps(sets,indent=2));(H/'reconciliation_inputs.json').write_text(json.dumps({'price_sha256':hashlib.sha256((d/'prices_original.parquet').read_bytes()).hexdigest(),'recovered_sha256':hashlib.sha256(q.read_bytes()).hexdigest(),'code_commit':'fe24e43d97004c32fef376a18b692e7fbccc6959','PIT_commit':'6ff2a610543d12e56b0290c83ed972c8954b8b41','no_downloads_or_simulation':True},indent=2));print('PASS all19 exact old coverage reproduction; residual partition explained')

pd.DataFrame(reasons).to_csv(H/'ineligible_reasons.csv',index=False)
