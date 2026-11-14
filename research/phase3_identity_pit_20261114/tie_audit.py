"""Read-only reproduction/permutation diagnostics. No portfolio return replay."""
import argparse,json,hashlib,sys,platform
from pathlib import Path
import numpy as np
import pandas as pd
H=Path(__file__).resolve().parent;R=H.parents[1];sys.path.insert(0,str(R))
import alpha_engine_backtest as ae
TOL=1e-10

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
 pfile=Path(a.data)/'prices_original.parquet';assert hashlib.sha256(pfile.read_bytes()).hexdigest()=='ab480ba3a1ce8a864aaad9d0dfeb4df16ed115d2bcf5e682e57c96c3de29c67d'
 p=pd.read_parquet(pfile);p.index=p.index.astype('datetime64[ns]')
 u=json.loads((Path(a.data)/'universe.json').read_text());usable={r:[t for t in ts if t in p and p[t].notna().sum()>=252] for r,ts in u.items()}
 old=pd.read_csv(R/'research/phase3/results/pinned/legacy_rerun_holdings.csv',parse_dates=['screen_date'],float_precision='round_trip')
 ties=[];scores=[];runs=[];baselines=[];reverse=[]
 for date in sorted(old.screen_date.unique()):
  for reg,ts in usable.items():
   b=ae.score_universe(p,ts,date);res=ae.compute_residual_momentum_score(p,b.index,date,reg,benchmark_mode=ae.build_benchmark_modes()['broad_default'])
   br=ae.score_universe(p,list(reversed(ts)),date);rr=ae.compute_residual_momentum_score(p,br.index,date,reg,benchmark_mode=ae.build_benchmark_modes()['broad_default'])
   for ratio in [0,60,100]:
    spec={'base_weight':1-ratio/100,'residual_weight':ratio/100}
    baseline=ae.combine_residual_score(b,res,spec)
    for t,row in baseline.iterrows():scores.append(dict(date=str(pd.Timestamp(date).date()),region=reg,R=ratio,ticker=t,Final_Score=row.Final_Score,base_score=row.base_score,residual_score=row.residual_score,legacy_rank=baseline.index.get_loc(t)+1))
    for score,g in baseline.groupby('Final_Score'):
     if len(g)>1:
      greater=int((baseline.Final_Score>score).sum());ties.append(dict(date=str(pd.Timestamp(date).date()),region=reg,R=ratio,score=score,tickers='|'.join(sorted(g.index)),size=len(g),higher=greater,boundary3=greater<3<greater+len(g),boundary6=greater<6<greater+len(g)))
    for n in [6,12]:
     o=old[(old.screen_date==date)&(old.variant==f'R{ratio}_N{n}')&(old.Region==reg)]
     got=baseline.head(n//2);err=float(np.max(np.abs(got.Final_Score.to_numpy()-o.Final_Score.to_numpy())))
     ok=got.index.tolist()==o.ticker.tolist() and err<=TOL
     baselines.append(dict(date=date,region=reg,R=ratio,N=n,order_equal=got.index.tolist()==o.ticker.tolist(),max_error=err,status='PASS' if ok else 'FAIL'))
     assert ok,('baseline mismatch',date,reg,ratio,n)
    variants={'original':b,'reverse':b.iloc[::-1],'ticker_asc':b.sort_index(),'ticker_desc':b.sort_index(ascending=False),'seed20261114':b.sample(frac=1,random_state=20261114),'seed20261115':b.sample(frac=1,random_state=20261115)}
    for order,bb in variants.items():
     legacy=ae.combine_residual_score(bb,res,spec)
     ticker=legacy.assign(ticker_key=legacy.index).sort_values(['Final_Score','ticker_key'],ascending=[False,True],kind='stable')
     phase2=legacy.assign(ticker_key=legacy.index).sort_values(['Final_Score','residual_score','base_score','ticker_key'],ascending=[False,False,False,True],kind='stable')
     for rule,frame in [('legacy',legacy),('diagnostic_ticker',ticker),('phase2_components',phase2)]:
      for k in [3,6]:
       rows=frame.head(k).index.tolist();orig=baseline.head(k).index.tolist()
       runs.append(dict(date=date,region=reg,R=ratio,k=k,order=order,rule=rule,ranked_all='|'.join(frame.index),top='|'.join(rows),top_set='|'.join(sorted(rows)),baseline_top='|'.join(orig),set_changed=set(rows)!=set(orig),top_order_changed=rows!=orig,all_order_changed=frame.index.tolist()!=baseline.index.tolist()))
    rev=ae.combine_residual_score(br,rr,spec)
    aligned=rev.reindex(baseline.index);err=float((aligned.Final_Score-baseline.Final_Score).abs().max())
    for k in [3,6]:reverse.append(dict(date=date,region=reg,R=ratio,k=k,max_score_delta=err,set_changed=set(rev.head(k).index)!=set(baseline.head(k).index),order_changed=rev.head(k).index.tolist()!=baseline.head(k).index.tolist()))
  print(str(date),'complete',flush=True)
 t=pd.DataFrame(ties);assert len(t)==699,len(t)
 prev=pd.read_csv(R/'research/phase3/results/pinned/legacy_ties.csv',float_precision='round_trip');prev.tickers=prev.tickers.map(lambda x:'|'.join(sorted(x.split('|'))))
 keys=['date','region','R','tickers'];pd.testing.assert_frame_equal(t[keys].sort_values(keys).reset_index(drop=True),prev[keys].sort_values(keys).reset_index(drop=True),check_dtype=False)
 t.to_csv(out/'ties.csv',index=False);pd.DataFrame(scores).to_csv(out/'score_frames.csv.gz',index=False)
 pd.DataFrame(runs).to_csv(out/'permutations.csv.gz',index=False);pd.DataFrame(reverse).to_csv(out/'reverse_input_recalculation.csv',index=False);pd.DataFrame(baselines).to_csv(out/'baseline_checks.csv',index=False)
 (out/'environment.json').write_text(json.dumps(dict(numpy=np.__version__,pandas=pd.__version__,python=platform.python_version(),protocol_sha256=hashlib.sha256((H/'PROTOCOL.md').read_bytes()).hexdigest()),indent=2))
 print('699 ties and 228 regional selections reproduced')
if __name__=='__main__':main()
