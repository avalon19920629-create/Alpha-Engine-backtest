from pathlib import Path
import subprocess,json,hashlib,datetime
H=Path(__file__).resolve().parent;R=H.parents[1]
def g(*a):return subprocess.check_output(['git','-C',str(R),*a])
c='27144bc66d8e5ba583c960107c70f4c89b7334a9';p='e0d8c8755d26b402e0c6748b88add87c0686b1ec';old='52531ed2d9ce1ea275fb77bbc7e18bc582655af0';main='075f0985c76c2212552732534d7f3c741270a80d'
blob=lambda f:g('show',c+':'+f)
assert g('rev-parse','origin/research/phase3-cash-ledger-20261114').decode().strip()==c
assert g('rev-parse','origin/main').decode().strip()==main
assert g('rev-parse',c+'^').decode().strip()==p
assert g('merge-base',old,c).decode().strip()==old
chain=g('rev-list','--reverse',old+'..'+c).decode().splitlines();assert chain==['fc499c873e10f4b4f331f3ec10032a656cd381dc',p,c]
d=g('diff','--name-status',p,c).decode().splitlines();assert len(d)==41 and all(x.startswith('A\tresearch/phase3_cash_ledger_20261114/') for x in d)
all_diff=g('diff','--name-status',old,c).decode().splitlines();assert len(all_diff)==102 and all(x.startswith('A\tresearch/') for x in all_diff)
prot=json.loads(blob('research/phase3_cash_ledger_20261114/protected_before.json'));assert len(prot)==256
for f,h in prot.items():assert hashlib.sha256(blob(f)).hexdigest()==h and blob(f)==g('show',p+':'+f)
counts={}
for folder,mfile,n in [('phase3','evidence_manifest.json',73),('phase3_repair_20261114','repair_manifest.json',57),('phase3_accounting_design_20261114','design_manifest.json',2),('phase3_cash_ledger_20261114','evidence_manifest.json',40)]:
 m=json.loads(blob('research/'+folder+'/'+mfile));assert len(m)==n
 for f,h in m.items():assert hashlib.sha256(blob('research/'+folder+'/'+f)).hexdigest()==h
 counts[folder]=len(m)
gate=json.loads(blob('research/phase3_cash_ledger_20261114/gate_decision.json'));assert gate['E30']=='NOT_PASSED' and gate['E31']=='LIMITED_MATHEMATICAL_PASS_UNCHANGED' and gate['E32']=='NO_GO' and gate['E32_executed']==False
assert b'48 passed' in blob('research/phase3_cash_ledger_20261114/combined_tests.log')
allfiles=g('ls-tree','-r','--name-only',c).decode().splitlines();assert len(allfiles)==297
assert all((R/f).read_bytes()==blob(f) for f in allfiles)
(H/'protected_before.json').write_text(json.dumps({f:hashlib.sha256(blob(f)).hexdigest() for f in allfiles},indent=2))
r=dict(status='PASS',verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),research_commit=c,parent=p,main=main,ancestor=old,chain=chain,parent_diff=d,ancestor_diff=all_diff,protected_256='PASS',local_baseline_297='PASS',manifest_counts=counts,gate=gate,log_48_passed=True)
(H/'preservation_audit.json').write_text(json.dumps(r,indent=2));print('PASS remote 27144bc; parent e0d8c87; 41 additions / 102 since 52531ed; 256 protected; 297 local; all manifests and gates')
