"""Research-only: measure 2016-2026 membership drift relative to the
frozen November-2026 S&P500 candidate universe. Not a performance test."""
import json
from pathlib import Path
import pandas as pd
import phase2_point_in_time_membership as pit

def main():
    out=Path("phase2_decade_membership")
    out.mkdir(exist_ok=True)
    original=json.loads(Path("frozen_prices/universe.json").read_text())
    # historical selection used 2026 stock pool; universe codes may include
    # names with missing stock quotes. Membership-only audit uses all US symbols.
    us={pit.canonical(x) for x in original["US"]}
    assert 490<=len(us)<=520
    cache={}
    dates=list(pd.date_range("2016-03-31","2026-09-30",freq="QE-DEC"))
    rows=[]
    for d in dates:
        members=pit.membership_asof(d,cache)
        matched=members&us
        rows.append({
            "quarter_end":str(d.date()),
            "year":d.year,
            "historical_member_count":len(members),
            "2026_snapshot_candidates":len(us),
            "intersection":len(matched),
            "historical_members_absent_from_2026_pool":len(members-us),
            "2026_pool_not_yet_historical_members":len(us-members),
            "missing_historical_fraction":len(members-us)/len(members),
            "future_member_fraction_of_2026_pool":len(us-members)/len(us),
        })
    df=pd.DataFrame(rows)
    df.to_csv(out/"quarterly_2016_2026_membership_gap.csv",index=False)
    annual=df.groupby("year",as_index=False).agg({
       "missing_historical_fraction":"mean",
       "future_member_fraction_of_2026_pool":"mean",
       "historical_members_absent_from_2026_pool":"mean",
       "2026_pool_not_yet_historical_members":"mean"})
    annual.to_csv(out/"annual_2016_2026_membership_gap.csv",index=False)
    summary={
       "period":"2016-Q1 through 2026-Q3",
       "quarter_count":len(dates),
       "member_source_commit":"6ff2a610543d12e56b0290c83ed972c8954b8b41",
       "price_data":"none (universe definition only)",
       "current_reference_snapshot":"2026-11-11, same frozen pool as Phase 1",
       "current_universe_count":len(us),
       "oldest_quarter":df.iloc[0].to_dict(),
       "latest_quarter":df.iloc[-1].to_dict(),
       "mean_historical_missing_fraction":float(df.missing_historical_fraction.mean()),
       "max_historical_missing_fraction":float(df.missing_historical_fraction.max()),
       "cautions":[
          "This measures only historical membership mismatch, NOT return inflation or standalone survivorship-adjusted CAGR.",
          "Old member tickers can differ due to renamed corporations/share classes and symbol reuse.",
          "Community-maintained member history requires vendor/official corroboration.",
          "2026 November pool differs from June 2026 pool used in original 10-year backtest.",
          "The absent constituents may or may not have been selectable under actual strategy eligibility."
       ]
    }
    (out/"metadata.json").write_text(json.dumps(summary,indent=2,default=str))
    print("2016–2026 historical member / 2026 pool mismatch:",flush=True)
    print(df.loc[(df.year.isin([2016,2018,2020,2022,2024,2026]))&(pd.to_datetime(df.quarter_end).dt.quarter==3),
       ["quarter_end","historical_member_count","intersection",
        "historical_members_absent_from_2026_pool","2026_pool_not_yet_historical_members"]].to_string(index=False),flush=True)
    print("mean gap",summary["mean_historical_missing_fraction"],flush=True)

if __name__=="__main__":
    main()
