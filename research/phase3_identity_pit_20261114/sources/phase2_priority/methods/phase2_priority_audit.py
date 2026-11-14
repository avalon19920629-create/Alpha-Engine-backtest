import io,zipfile, numpy as np, pandas as pd
from pathlib import Path
root=Path('/mnt/data')
a=zipfile.ZipFile(root/'phase2-former-member-recovery-verified-20261112.zip')
b=zipfile.ZipFile(root/'phase2-historical-membership-audit-20261112.zip')
x=pd.read_csv(a.open('holdings_partial_recovery.csv'))
y=pd.read_csv(b.open('selected_historical_membership_comparison.csv'))
c=pd.read_csv(a.open('historical_membership_coverage_extended.csv'))
z=pd.read_csv(a.open('historical_symbol_quote_recovery.csv'))
print('US recovered holdings grouped:')
g=x[x.is_recovered_former_member]
print(g.groupby('ticker').agg(sessions=('screen_date','nunique'),holdings=('ticker','count'),models=('variant','nunique'),first=('screen_date','min'),last=('screen_date','max')).sort_values('holdings',ascending=False).to_string())
print('Total selected recovered',len(g),'distinct',g.ticker.nunique())
print('by variant',g.groupby('variant').agg(records=('ticker','size'),screen_dates=('screen_date','nunique')).to_string())
print('postrestore vs restricted overlap')
rows=[]
for (variant,date),r in x.groupby(['variant','screen_date']):
  m=y[(y['mode']=='historical_members_within_current_pool')&(y['variant']==variant)&(y['screen_date']==date)]
  all_changes=set(r.ticker)^set(m.ticker)
  rows.append(dict(variant=variant,screen_date=date,changed=bool(all_changes),added_from_recovery=';'.join(sorted(set(r.ticker)-set(m.ticker))),removed_from_old=';'.join(sorted(set(m.ticker)-set(r.ticker))),n_changes=len(all_changes)//2,n_recovered=int(r.is_recovered_former_member.sum())))
d=pd.DataFrame(rows)
print(d.groupby('variant').agg(changed_quarters=('changed','sum'),sum_switches=('n_changes','sum'),added_recovered=('n_recovered','sum')).to_string())
print('nonrestorable 20 high frequency missing across quarter:')
missing=pd.read_csv(a.open('uncovered_historical_symbols_by_screen.csv'))
remaining=missing[~missing.recovered].groupby('ticker').screen_date.nunique().sort_values(ascending=False)
print(remaining.head(28).to_string())
print('Per quarter coverage:')
print(c[['screen_date','historical_eligible_total_partial','official_historical_member_count_proxy','historical_names_uncovered_total','fraction_members_with_historical_quotes_partial']].to_string(index=False))
print('recovered actual candidate count by screen')
print(d.groupby('screen_date').agg(changed=('changed','sum'),new_selected=('n_recovered','sum'),total_switches=('n_changes','sum')).sort_values(['changed','total_switches'],ascending=False).to_string())
out=root/'phase2_priority_pilot'
out.mkdir(exist_ok=True)
d.to_csv(out/'selection_changes_caused_by_recovered_members.csv',index=False)
g.groupby('ticker').agg(sessions=('screen_date','nunique'),holdings=('ticker','count'),models=('variant','nunique'),first=('screen_date','min'),last=('screen_date','max')).sort_values('holdings',ascending=False).to_csv(out/'recovered_winners_priority.csv')
remaining.rename('missing_screen_dates').to_csv(out/'unrecoverable_ticker_frequency.csv')
