"""Recompute diagnostic summary from saved outputs; never changes source artifacts."""
from pathlib import Path
import json,hashlib,sys
import pandas as pd,numpy as np
H=Path(__file__).resolve().parent;R=H.parents[1];sys.path.insert(0,str(R))
from research.phase3.signals import strict_order
p=pd.read_csv(H/'results_pinned/permutations.csv.gz');q=pd.read_csv(H/'results_initial/permutations.csv.gz');t=pd.read_csv(H/'results_pinned/ties.csv')
keys=['date','region','R','k'];out={'tie_groups':len(t),'boundary3':int(t.boundary3.sum()),'boundary6':int(t.boundary6.sum()),'regional_selections':228,'permutation_rows_per_environment':len(p),'rules':{}}
assert p.equals(q),'cross-environment permutation mismatch'
for rule,g in p.groupby('rule'):
 out['rules'][rule]={'row_order_sensitive_selection_sets':int(g.groupby(keys).top_set.nunique().gt(1).sum()),'row_order_sensitive_full_rankings':int(g.groupby(keys).ranked_all.nunique().gt(1).sum()),'distinct_selection_sets_different_from_legacy':len(g[g.set_changed].drop_duplicates(keys))}
a=pd.read_csv(H/'results_pinned/score_frames.csv.gz');b=pd.read_csv(H/'results_initial/score_frames.csv.gz');assert a[['date','region','R','ticker','legacy_rank']].equals(b[['date','region','R','ticker','legacy_rank']]);delta=float((a[['Final_Score','base_score','residual_score']]-b[['Final_Score','base_score','residual_score']]).abs().max().max());assert delta<=1e-10
out['cross_environment_score_max_delta']=delta
rev=pd.read_csv(H/'results_pinned/reverse_input_recalculation.csv');out['reverse_universe_set_changes']=int(rev.set_changed.sum());out['reverse_universe_max_score_delta']=float(rev.max_score_delta.max())
f=pd.DataFrame({'ticker':['A','B'],'score':[1.,1.]});cases=[]
for name,ids,expected in [('missing',{'A':'1'},True),('duplicate_issuer',{'A':'1','B':'1'},True),('null_id',{'A':None,'B':'2'},False),('distinct_synthetic',{'A':'1','B':'2'},False)]:
 try: result=strict_order(f,'score',ids);raised=False
 except ValueError:result=None;raised=True
 assert raised==expected
 cases.append({'case':name,'raises':raised,'result':result,'interpretation':'characterization only; acceptance does not certify identity'})
out['strict_order_characterization']=cases
(H/'diagnostic_summary.json').write_text(json.dumps(out,indent=2));t[t.boundary3|t.boundary6].to_csv(H/'boundary_ties.csv',index=False)
old=json.loads((H/'protected_before.json').read_text());assert len(old)==297
assert all(hashlib.sha256((R/p).read_bytes()).hexdigest()==s for p,s in old.items())
print(json.dumps(out,indent=2));print('PASS: cross-environment comparisons, four helper characterizations, protected297')
