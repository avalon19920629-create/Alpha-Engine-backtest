"""Fetch only; verify exact preserved research revision and unchanged main."""
import hashlib,json,subprocess,datetime
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[1]
def g(*a):return subprocess.check_output(['git','-C',str(R),*a])
c='52531ed2d9ce1ea275fb77bbc7e18bc582655af0';p='075f0985c76c2212552732534d7f3c741270a80d';b='research/phase3-selection-20261113'
g('fetch','origin','refs/heads/'+b+':refs/remotes/origin/'+b,'refs/heads/main:refs/remotes/origin/main')
assert g('rev-parse','origin/'+b).decode().strip()==c
assert g('rev-parse','origin/main').decode().strip()==p
assert g('rev-parse',c+'^').decode().strip()==p
blob=lambda f:g('show',c+':'+f)
d=g('diff','--name-status',p,c).decode().splitlines();assert len(d)==74 and all(x.startswith('A\tresearch/phase3/') for x in d)
m=json.loads(blob('research/phase3/evidence_manifest.json'));assert len(m)==73
for f,h in m.items():assert hashlib.sha256(blob('research/phase3/'+f)).hexdigest()==h
pr=blob('research/phase3/protected_before.sha256').decode().splitlines();assert len(pr)==121
for l in pr:
 h,f=l.split('  ',1);assert hashlib.sha256(blob(f)).hexdigest()==h;assert blob(f)==g('show',p+':'+f)
gate=json.loads(blob('research/phase3/gate_decision.json'))
assert gate['E30']['overall']=='NOT_PASSED'
assert gate['E31']['overall']=='PASS_COMMON_ELIGIBLE_SNAPSHOT_MATHEMATICAL_EQUIVALENCE'
assert gate['E32_transition']['decision']=='NO_GO_UNDER_V1_0_FULL_GATE'
files=g('ls-tree','-r','--name-only',c).decode().splitlines()
assert all((R/f).read_bytes()==blob(f) for f in files)
report=dict(status='PASS',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),remote_commit=c,remote_main=p,parent=p,additions=74,manifest_hashes=73,protected=121,all_baseline_files_unchanged=len(files),gates_unchanged=True)
(H/'preservation_audit_final.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
