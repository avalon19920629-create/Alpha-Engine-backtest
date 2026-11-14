"""Read-only validation. Does not execute selection or alter a protected artifact."""
from pathlib import Path
import subprocess,json,hashlib,csv
H=Path(__file__).resolve().parent;R=H.parents[1]
def sha(b):return hashlib.sha256(b).hexdigest()
base='2c246a33115daa8ff16305d2771540b16ac9324a'
protected=json.loads((H/'protected_before.json').read_text())
assert len(protected)==379
for name,digest in protected.items():
 assert sha((R/name).read_bytes())==digest,name
 assert sha(subprocess.check_output(['git','show',base+':'+name],cwd=R))==digest,name
print('PASS protected 379 working bytes and base Git blobs')
rows=list(csv.DictReader((H/'set_reconciliation.csv').open()))
sets=json.loads((H/'set_evidence.json').read_text())
assert len(rows)==len(sets)==19
for row,s in zip(rows,sets):
 m,e,u,r=(set(s[x]) for x in ['members','eligible','uncovered','residual'])
 assert e<=m and u<=m and r==m-e-u and not(e&u)
 assert [len(x) for x in (m,e,u,r)]==[int(row[x]) for x in ['proxy','eligible','uncovered','residual']]
assert sets[0]['residual']==['CEG','FB']
assert sum(int(x['residual'])>0 for x in rows)==16
assert all(x['under_252']=='True' for x in csv.DictReader((H/'ineligible_reasons.csv').open()))
print('PASS 19 disjoint set partitions, 16 residuals, CEG/FB first case; original-ineligible histories under252')
for x in json.loads((H/'pit_source_provenance.json').read_text())['files']:
 assert sha((H/x['local_file']).read_bytes())==x['expected_sha256']==x['actual_sha256']
print('PASS 5 original YAML blob hashes')
g=json.loads((H/'gate_decision.json').read_text())
assert (g['E30'],g['E31'],g['E32'])==('NOT_PASSED','LIMITED_MATHEMATICAL_PASS','NO_GO')
events=json.loads((H/'event_evidence.json').read_text()); assert len(events)==9
assert all(x['terminal_market_price'] is None and x['price_adjustment_method'] is None for x in events)
sources=list(csv.DictReader((H/'SOURCE_REGISTER.csv').open()));assert len(sources)==29
assert all(len(x['sha256'])==64 for x in sources)
print('PASS 9 event unknowns,29 source records,gates preserved')
manifest=H/'evidence_manifest.json'
if manifest.exists():
 for p,h in json.loads(manifest.read_text()).items():assert sha((H/p).read_bytes())==h,p
 print('PASS deliverable manifest')
