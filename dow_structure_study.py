"""Dow Structure Study v0.2: fixed-date observations of confirmed structure.

Offline research only. Existing engines, Research Operations and Dow v0.1 are
imported without modification. No screening, trading, or data acquisition runs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import dow_momentum_study as v1

BASE_COMMIT = "59db288274608ed8f6b8753dcfbe790e37ffe5cb"
DEFAULT_SNAPSHOT = v1.DEFAULT_SNAPSHOT
DEFAULT_OUTPUT = "/content/drive/MyDrive/alpha_engine/analysis/dow_structure_v02"
STATES = ("HH/HL", "LH/LL", "LH/HL", "HH/LL")
ALL_STATES = (*STATES, "unclassified")
FEATURES = ("momentum_3m", "momentum_6m", "momentum_12m", "volatility")
PROTECTED = ("Alpha-Engine.py", "alpha_engine_backtest.py", "alpha_engine_live_screener.py",
             "alpha_engine_production.py", "alpha_engine_order_planner.py", "alpha_engine_ttl_manager.py",
             "core_alpha_integration_backtest.py", "alpha_engine_research_runner.py", "dow_momentum_study.py")


class StudyRunError(RuntimeError):
    def __init__(self, message, run_dir):
        super().__init__(message)
        self.run_dir = run_dir


def classify_state(previous_high, last_high, previous_low, last_low):
    """Use the newest two confirmed highs/lows, never a future pivot."""
    try:
        values = np.array([previous_high, last_high, previous_low, last_low], dtype=float)
    except (ValueError, TypeError):
        return "unclassified"
    if not np.isfinite(values).all():
        return "unclassified"
    high, low = np.sign(values[1]-values[0]), np.sign(values[3]-values[2])
    return {(1,1):"HH/HL", (-1,-1):"LH/LL", (-1,1):"LH/HL", (1,-1):"HH/LL"}.get((high,low),"unclassified")


def compute_structure_features(prices, us, jp, span=5):
    """Daily common-calendar features for P and P/B, including unclassified.

    v0.1 supplies causal confirmed pivots and endpoint momentum. Its alert
    results are discarded: an alert is never a condition for this study.
    """
    daily, pivots, _ = v1.compute_features(prices, us, jp, span=span)
    frames = []
    pivot_groups = {key:group for key,group in pivots.groupby(["ticker","market","series_kind"],sort=False)}
    for key, group in daily.groupby(["ticker","market","series_kind"],sort=False):
        frame = group.sort_values("date").copy()
        index = pd.DatetimeIndex(frame.date)
        available = pivot_groups.get(key,pivots.iloc[:0])
        for kind, label in (("high","high"),("low","low")):
            selected = available.loc[available.kind==kind].sort_values("confirmed_date")
            count = pd.Series(np.arange(1,len(selected)+1),index=selected.confirmed_date)
            previous = pd.Series(selected.value.shift(1).to_numpy(),index=selected.confirmed_date)
            previous_position = pd.Series(selected.pivot_position.shift(1).to_numpy(),index=selected.confirmed_date)
            frame[f"confirmed_{label}_count"] = count.reindex(index).ffill().fillna(0).to_numpy(dtype=int)
            frame[f"previous_confirmed_{label}"] = previous.reindex(index).ffill().to_numpy()
            frame[f"previous_{label}_pivot_position"] = previous_position.reindex(index).ffill().to_numpy()
        h0, h1 = frame.previous_confirmed_high, frame.reference_high
        l0, l1 = frame.previous_confirmed_low, frame.reference_low
        frame["state"] = np.select([(h1>h0)&(l1>l0),(h1<h0)&(l1<l0),(h1<h0)&(l1>l0),(h1>h0)&(l1<l0)],STATES,default="unclassified")
        enough = (frame.confirmed_high_count>=2)&(frame.confirmed_low_count>=2)
        frame["state_reason"] = np.select([~enough,frame.state=="unclassified"],["insufficient_confirmed_pivots","equal_pivots"],default="classified")
        frame["high_age"] = frame.common_position-frame.reference_high_confirmed_position
        frame["low_age"] = frame.common_position-frame.reference_low_confirmed_position
        frame["distance_to_high"] = frame.series_value/frame.reference_high-1
        frame["distance_to_low"] = frame.series_value/frame.reference_low-1
        both = frame.reference_high.notna()&frame.reference_low.notna()
        above, below = frame.series_value>frame.reference_high, frame.series_value<frame.reference_low
        frame["range_position"] = np.select([~both,above&below,above,below],["unavailable","above_high_and_below_low","above_high","below_low"],default="within_range")
        relative = frame.stock_price/frame.benchmark_price
        frame["relative_momentum_12m"] = relative/relative.shift(251)-1
        # Four-state comparisons depend on TWO highs and TWO lows. An old
        # anomalous reference can remain relevant beyond the momentum window.
        if key[1]=="JP":
            prefix=np.r_[0,np.cumsum(index.isin(v1.SUSPECT_DATES).astype(int))]
            affected=frame.excluded_influence.to_numpy(dtype=bool).copy()
            for column in ("reference_high_pivot_position","reference_low_pivot_position","previous_high_pivot_position","previous_low_pivot_position"):
                positions=frame[column].to_numpy(dtype=float)
                valid=np.isfinite(positions)
                centers=positions[valid].astype(int)
                a,b=np.maximum(0,centers-span),np.minimum(len(frame),centers+span+1)
                affected[valid] |= prefix[b]>prefix[a]
            frame["excluded_influence"]=affected
        frames.append(frame)
    return pd.concat(frames,ignore_index=True).sort_values(["ticker","series_kind","date"]).reset_index(drop=True)


def landmark_dates(benchmark, start="2024-01-01", interval=21, phase=0):
    """One market-wide clock; do not restart the clock for each stock."""
    if isinstance(interval,bool) or int(interval)!=interval or interval<=0:
        raise ValueError("interval must be a positive integer")
    if isinstance(phase,bool) or int(phase)!=phase or not 0<=phase<interval:
        raise ValueError("phase must lie within the interval")
    valid = benchmark.dropna()
    if not isinstance(valid.index,pd.DatetimeIndex) or valid.index.has_duplicates or not valid.index.is_monotonic_increasing:
        raise ValueError("Benchmark calendar must contain unique ordered dates")
    valid = valid.loc[(valid.index>=pd.Timestamp(start))&np.isfinite(valid)&(valid>0)]
    return pd.DatetimeIndex(valid.index[int(phase)::int(interval)])


def build_landmarks(daily, prices, us, jp, interval=21, phase=0):
    """Retain every scheduled candidate, including missing and unclassified."""
    frames = []
    for market,universe in (("US",us),("JP",jp)):
        dates = landmark_dates(prices[v1.BENCHMARKS[market]],interval=interval,phase=phase)
        grid = pd.MultiIndex.from_product([dates,universe,("price","relative")],names=["date","ticker","series_kind"]).to_frame(index=False)
        grid["market"] = market
        number = {date:i for i,date in enumerate(dates)}
        grid["market_landmark_number"] = grid.date.map(number)
        selected = grid.merge(daily,on=["date","ticker","series_kind","market"],how="left",validate="one_to_one")
        selected["history_eligible"] = selected.observations.ge(253)&selected.stock_price.notna()&selected.benchmark_price.notna()
        selected["daily_state_before_history_filter"] = selected.state.fillna("unclassified")
        missing = selected.common_position.isna()
        selected["exclusion_reason"] = np.select([missing,~selected.history_eligible,selected.state.eq("unclassified")],
                                                   ["missing_common_observation","insufficient_253_price_history",selected.state_reason.fillna("insufficient_confirmed_pivots")],default="")
        selected.loc[~selected.history_eligible,"state"] = "unclassified"
        selected["classified"] = selected.history_eligible&selected.state.isin(STATES)
        selected["state_reason"] = selected.exclusion_reason.where(~selected.classified,"classified")
        selected["signal_date"] = selected.date
        selected["sample_id"] = [f"{market}|{r.ticker}|{r.series_kind}|{pd.Timestamp(r.date).date()}" for r in selected.itertuples()]
        selected["event_id"] = selected.sample_id
        selected["phase"] = phase
        frames.append(selected)
    result = pd.concat(frames,ignore_index=True)
    if result.empty:
        raise ValueError("No market landmark dates on or after 2024-01-01 in the snapshot")
    # Rank before removing unclassified states: structure must not change the
    # cross-section defining extreme momentum. Price and relative rows share it.
    ranks = []
    for (date,market), group in result.loc[result.series_kind=="price"].groupby(["date","market"],sort=True):
        eligible = group.loc[group.history_eligible&np.isfinite(group.momentum_12m)].copy()
        population = len(eligible)
        quota = int(np.ceil(population*.1))
        eligible = eligible.sort_values(["momentum_12m","ticker"],ascending=[False,True],kind="stable")
        rank_by_ticker = dict(zip(eligible.ticker,np.arange(1,population+1)))
        relative = eligible.loc[np.isfinite(eligible.relative_momentum_12m)].sort_values(["relative_momentum_12m","ticker"],ascending=[False,True],kind="stable")
        relative_rank = dict(zip(relative.ticker,np.arange(1,len(relative)+1)))
        for row in group.itertuples():
            rank = rank_by_ticker.get(row.ticker,np.nan)
            rs_rank = relative_rank.get(row.ticker,np.nan)
            ranks.append({"date":date,"market":market,"ticker":row.ticker,"rank_12m":rank,"rank_population":population,"top10_quota":quota,
                          "raw_top10":bool(rank<=quota and row.momentum_12m>0),
                          "relative_rank_12m":rs_rank,"relative_rank_population":len(relative),
                          "relative_top10":bool(rs_rank<=int(np.ceil(len(relative)*.1)) and row.relative_momentum_12m>0)})
    rank_columns=["date","market","ticker","rank_12m","rank_population","top10_quota","raw_top10","relative_rank_12m","relative_rank_population","relative_top10"]
    result = result.merge(pd.DataFrame(ranks,columns=rank_columns),on=["date","market","ticker"],how="left",validate="many_to_one")
    for name in ("raw_top10","relative_top10"):
        result[name] = result[name].fillna(False).astype(bool)
    result["momentum_3m_sign"] = np.select([result.momentum_3m.isna(),result.momentum_3m>0,result.momentum_3m<0],["missing","positive","negative"],default="zero")
    return result.sort_values(["date","market","ticker","series_kind"]).reset_index(drop=True)


def structure_outcomes(landmarks, matches, prices, us, jp):
    """Reuse v0.1 next-valid-close entry and horizon-specific censoring."""
    eligible = landmarks.loc[landmarks.history_eligible].copy()
    measured = v1.evaluate_outcomes(eligible,matches,prices,us,jp)
    dimensions = ["event_id","state","classified","history_eligible","raw_top10","relative_top10","momentum_3m_sign","market_landmark_number"]
    measured = measured.merge(eligible[dimensions],on="event_id",how="left",validate="many_to_one")
    if "control_state" in matches:
        measured = measured.merge(matches[["event_id","control_state"]],on="event_id",how="left",validate="many_to_one")
    else:
        measured["control_state"] = None
    measured["focal_state"] = measured.state
    measured["observed_ticker_state"] = measured.state.where(measured.group.eq("alert"),measured.control_state)
    return measured


def _verify_code():
    root = Path(__file__).resolve().parent
    def git(*args):
        return subprocess.check_output(["git","-C",str(root),*args],stderr=subprocess.PIPE)
    sha = git("rev-parse","HEAD").decode().strip()
    git("merge-base","--is-ancestor",BASE_COMMIT,sha)
    hashes = {}
    for name in PROTECTED:
        actual = (root/name).read_bytes()
        if actual != git("show",f"{BASE_COMMIT}:{name}"):
            raise ValueError(f"Protected source changed from v0.1 baseline: {name}")
        hashes[name] = hashlib.sha256(actual).hexdigest()
    return {"git_commit_sha":sha,"protected_source_sha256":hashes,"study_source_sha256":v1._hash(Path(__file__))}


def _case(prices, us, jp, span=5, phase=0, exclude_influence=False):
    daily = compute_structure_features(prices,us,jp,span)
    observations = build_landmarks(daily,prices,us,jp,phase=phase)
    if exclude_influence:
        affected = observations.excluded_influence.fillna(False).astype(bool)
        observations.loc[affected,"history_eligible"] = False
        observations.loc[affected,"classified"] = False
        observations.loc[affected,"state"] = "unclassified"
        observations.loc[affected,"exclusion_reason"] = "known_benchmark_anomaly_in_253_price_window_or_pivot_support"
        observations.loc[affected,"state_reason"] = observations.loc[affected,"exclusion_reason"]
        # Cohort ranks remain frozen before this exclusion, rather than
        # replacing dropped top names with lower-ranked survivors.
    matches,balance = match_structure_controls(observations)
    outcomes = structure_outcomes(observations,matches,prices,us,jp)
    if exclude_influence:
        touched = pd.Series(False,index=outcomes.index)
        for date in v1.SUSPECT_DATES:
            if date in prices.index:
                usable = [t for t in jp if t in prices and pd.notna(prices.at[date,t]) and pd.notna(prices.at[date,"1306.T"])]
                touched |= outcomes.market.eq("JP")&outcomes.ticker.isin(usable)&outcomes.entry_date.le(date)&outcomes.exit_date.ge(date)
        touched &= outcomes.status.eq("observed")
        outcomes.loc[touched,"status"] = "excluded_influence"
        outcomes.loc[touched,["stock_return","benchmark_return","relative_return","return_difference","max_drawdown","min_return_from_entry"]] = np.nan
    return daily,observations,matches,balance,outcomes


def _paired_comparisons(outcomes, matches):
    keys=["event_id","horizon"]
    a=outcomes.loc[outcomes.group=="alert"].drop(columns=["group"])
    b=outcomes.loc[outcomes.group=="control"].drop(columns=["group"])
    # v0.1 attaches focal covariates/cohort flags to both measurement rows.
    # Do not export those copies as if they were the control's own features.
    context=["classified","history_eligible","raw_top10","relative_top10","momentum_3m_sign","market_landmark_number","momentum_3m","event_momentum_3m"]
    a=a.rename(columns={name:name+"_focal" for name in context if name in a})
    b=b.drop(columns=[name for name in context if name in b])
    # Event-level 'state' is the focal comparison stratum for both sides;
    # exported paired sides instead show each ticker's actual state.
    b=b.copy()
    b["state"]=b.observed_ticker_state
    joined=a.merge(b,on=keys,how="outer",suffixes=("_focal","_control"),validate="one_to_one")
    complete=joined.status_focal.eq("observed")&joined.status_control.eq("observed")
    joined["pair_status"]=np.select([complete,joined.status_control.eq("unmatched"),joined.status_focal.eq("excluded_influence")|joined.status_control.eq("excluded_influence")],
                                   ["observed","unmatched","excluded_influence"],default="right_censored_or_unavailable")
    for feature in ("relative_return","stock_return","max_drawdown"):
        joined[feature+"_difference"]=(joined[feature+"_focal"]-joined[feature+"_control"]).where(complete)
    match_cols=[c for c in ("event_id","distance","status","control_state","eligible_control_count","within_caliper_count") if c in matches]
    additional={}
    for feature in FEATURES:
        if feature in matches and feature+"_focal" not in joined:
            additional[feature]=feature+"_focal"
        if "control_"+feature in matches:
            additional["control_"+feature]=feature+"_control"
    selection=matches[[*match_cols,*additional]].rename(columns=additional)
    return joined.merge(selection,on="event_id",how="left",validate="many_to_one")


def _daily_durations(daily):
    records=[]
    for (ticker,market,series_kind),group in daily.groupby(["ticker","market","series_kind"],sort=True):
        group=group.sort_values("date")
        run=group.state.ne(group.state.shift()).cumsum()
        parts=list(group.groupby(run,sort=True))
        for i,(_,part) in enumerate(parts):
            records.append({"ticker":ticker,"market":market,"series_kind":series_kind,"state":part.state.iloc[0],
                            "start_date":part.date.iloc[0],"end_date":part.date.iloc[-1],"common_observations":len(part),
                            "calendar_days":int((part.date.iloc[-1]-part.date.iloc[0]).days),"left_boundary_censored":i==0,"right_boundary_censored":i==len(parts)-1,
                            "definition":"daily model-state run on common valid observations, not classical trend duration"})
    return pd.DataFrame(records)


def _freshness_outcomes(landmarks,outcomes):
    """Exploratory forward distributions; never a primary-sample filter."""
    frame=outcomes.loc[outcomes.group=="alert"].merge(landmarks[["event_id","high_age","low_age","range_position"]],on="event_id",how="left",validate="many_to_one")
    for source in ("high_age","low_age"):
        frame[source+"_bin"]=pd.cut(frame[source],[0,21,63,np.inf],right=False,labels=["0-20","21-62","63+"]).astype(object).fillna("missing")
    rows=[]
    keys=["market","series_kind","state","horizon","high_age_bin","low_age_bin","range_position"]
    for cohort in ("all","raw_top10","relative_top10"):
        selected=frame if cohort=="all" else frame.loc[frame[cohort]]
        for values,part in selected.groupby(keys,dropna=False,sort=True):
            observed=part.loc[part.status=="observed"]
            rows.append(dict(zip(keys,values))|{"cohort":cohort,"total_count":len(part),"valid_count":len(observed),
                         "right_censored_count":int(part.status.eq("right_censored").sum()),
                         "mean_relative_return":observed.relative_return.mean(),"median_relative_return":observed.relative_return.median(),
                         "analysis_scope":"additional exploratory freshness/position analysis; not a primary filter"})
    return pd.DataFrame(rows,columns=[*keys,"cohort","total_count","valid_count","right_censored_count","mean_relative_return","median_relative_return","analysis_scope"])


def _charts(folder, observations, matches, outcomes, tables):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.hashsalt":"dow-structure-v02","font.size":9})
    folder.mkdir()
    def save(fig,name):
        fig.tight_layout()
        fig.savefig(folder/(name+".svg"),metadata={"Date":None})
        plt.close(fig)
    for kind in ("price","relative"):
        sample=observations.loc[observations.series_kind==kind]
        fig,ax=plt.subplots(figsize=(9,3))
        fractions={}
        for cohort,group in (("all",sample),("raw_top10",sample.loc[sample.raw_top10])):
            count=group.state.value_counts().reindex(ALL_STATES,fill_value=0)
            fractions[cohort]=count/len(group) if len(group) else count*np.nan
        pd.DataFrame(fractions).plot.bar(ax=ax)
        ax.set_title(f"{kind}: scheduled state shares, unclassified retained")
        ax.set_ylabel("Share of scheduled cohort observations")
        save(fig,"state_composition_"+kind)
        fig,ax=plt.subplots(figsize=(7,4))
        ordered=sample.sort_values(["ticker","market_landmark_number"])
        prior=ordered.groupby("ticker").state.shift()
        matrix=pd.crosstab(prior,ordered.state).reindex(index=ALL_STATES,columns=ALL_STATES,fill_value=0)
        probabilities=matrix.div(matrix.sum(axis=1).replace(0,np.nan),axis=0).fillna(0)
        im=ax.imshow(probabilities.to_numpy(),vmin=0,vmax=1,cmap="Blues")
        ax.set_xticks(range(5),ALL_STATES,rotation=30)
        ax.set_yticks(range(5),ALL_STATES)
        for i in range(5):
            for j in range(5):
                ax.text(j,i,f"{probabilities.iloc[i,j]:.0%}\n(n={matrix.iloc[i,j]})",ha="center",va="center",fontsize=7)
        ax.set_title(f"{kind}: consecutive scheduled-landmark transitions")
        fig.colorbar(im,ax=ax)
        save(fig,"state_transitions_"+kind)
        fig,ax=plt.subplots(figsize=(9,3))
        observed=outcomes.loc[outcomes.group.eq("alert")&outcomes.status.eq("observed")&outcomes.horizon.eq(21)&outcomes.series_kind.eq(kind)]
        arrays=[observed.loc[observed.state==state,"relative_return"].to_numpy() for state in ALL_STATES]
        if any(len(array) for array in arrays):
            ax.boxplot(arrays,tick_labels=ALL_STATES,showfliers=False)
        else:
            ax.text(.5,.5,"No observed 21-bar outcomes",ha="center",transform=ax.transAxes)
        ax.set_title(f"{kind}: unadjusted 21-bar relative returns (repeated observations)")
        save(fig,"state_outcomes_"+kind)
        fig,ax=plt.subplots(figsize=(9,3))
        band=np.select([sample.raw_top10,sample.momentum_12m.isna(),sample.momentum_12m<=0],["raw_top10","missing","nonpositive"],default="positive_other")
        matrix=pd.crosstab(band,sample.state).reindex(columns=ALL_STATES,fill_value=0)
        if len(matrix):
            matrix.div(matrix.sum(axis=1),axis=0).plot.bar(stacked=True,ax=ax)
        ax.set_title(f"{kind}: structure by stock-momentum strength, not an optimization")
        save(fig,"momentum_structure_"+kind)


def run_study(snapshot_dir=DEFAULT_SNAPSHOT,output_root=DEFAULT_OUTPUT,*,bootstrap_samples=500,seed=20261104):
    if isinstance(bootstrap_samples,bool) or int(bootstrap_samples)!=bootstrap_samples or bootstrap_samples<0:
        raise ValueError("bootstrap_samples must be a nonnegative integer")
    snapshot,root=Path(snapshot_dir).expanduser().resolve(),Path(output_root).expanduser().resolve()
    if root==snapshot or snapshot in root.parents:
        raise ValueError("Output must lie outside the immutable snapshot")
    now=datetime.now(timezone.utc)
    output=root/f"{now.strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:12]}"
    output.mkdir(parents=True,exist_ok=False)
    parameters={"span":5,"landmark_start":"2024-01-01","landmark_interval":21,"landmark_phase":0,
                "primary_series":"price","secondary_series":"relative","primary_horizon":21,"horizons":[5,21,63],
                "minimum_common_prices":253,"minimum_highs":2,"minimum_lows":2,"states":list(ALL_STATES),
                "momentum_price_windows":[63,126,252],"volatility_return_window":252,
                "matching_features":list(FEATURES),"caliper_per_feature":.5,"scale":"same-date classified feature-eligible cross-section SD ddof0, floor1e-12",
                "match_distance":"L2 scaled differences; tie by ticker; different classified state; controls reusable; never use future outcomes",
                "extreme_definition":"same-market same-date eligible stock m12 descending then ticker; ceil10% quota; positive m12 required; unclassified included in rank population",
                "relative_extreme_definition":"separate reference, relative_m12 top ceil10% and positive relative_m12",
                "phaseII_adjusted_controls":"main_population_controls: retain fixed PhaseI pairs, subset focal rawtop10; not internally top10-matched",
                "outcome_state_fields":"state/focal_state denote comparison stratum; observed_ticker_state denotes the measured stock's own state; paired state_control denotes actual control state",
                "outcome_feature_fields":"structure_outcomes momentum/cohort labels describe the focal landmark on both rows; paired *_control covariates come from actual selected control in structure_matches.csv",
                "extreme_transitions":"from-date cohort membership; retain next state even if exits cohort",
                "uncertainty_scope":"primaryH21,periodALL,signall,4classifiedstates,all/rawtop10,marketsALL/US/JP,series separately; other horizons descriptive only",
                "sensitivity_summary_scope":"primaryH21,periodALL,signall,all/rawtop10,all states,marketsALL/US/JP,series separately; complete5/21/63 outcome rows retained per specification",
                "bootstrap_samples":int(bootstrap_samples),"seed":int(seed),
                "influence_exclusion":"uncorrected JP trailing253 prices or support of either of the latest TWO highs and TWO lows; outcomes touching anomaly excluded after matching; cohort ranks frozen before exclusion"}
    warnings=["Exploratory retrospective fixed-current-universe study; survivorship, post-selection, repeated-ticker and shared-date dependence remain.",
              "User-reported v0.1 empirical findings are background, not independently verified by this implementation.",
              "P primary and P/B secondary are separate views of the same stocks, not independent observations.",
              "1306.T 10x correction remains tentative and not fully checked against official adjusted prices.",
              "Four-feature calipers are fixed; unmatched observations are not representative proof about the full population.",
              "Separate ticker/date bootstrap intervals are not a calibrated combined two-way confidence interval. Multiple comparisons are exploratory.",
              "Focal and control stocks use their own common valid benchmark clocks; future missing bars can produce different calendar entry/exit dates.",
              "Landmark runs and sampled transitions do not reveal all hidden daily transitions; daily model-run table is separate."]
    meta={"version":"0.2","status":"incomplete","validation_status":"incomplete","started_at":now.isoformat(),"finished_at":None,
          "base_commit":BASE_COMMIT,"source_snapshot":str(snapshot),"output_folder":str(output),"parameters":parameters,"warnings":warnings,
          "versions":{"python":platform.python_version(),"pandas":pd.__version__,"numpy":np.__version__}}
    v1._json(output/"study_metadata.json",meta)
    try:
        meta.update(_verify_code())
        if not (snapshot/"price_data/price_data_quality.csv").is_file():
            raise ValueError("Saved price_data_quality.csv is required; never download a replacement")
        original,us,jp,scores,hashes=v1._input_snapshot(snapshot)
        meta["input_hashes"]=hashes
        meta["candidate_counts"]={"US":len(us),"JP":len(jp),"total":len(us)+len(jp),"snapshot_ranked":len(scores)}
        meta["analysis_period"]={"start":str(original.index.min().date()),"end":str(original.index.max().date())}
        if (len(us),len(jp),len(scores))!=(503,108,607):
            warnings.append("Input counts differ from the named US503/JP108/ranked607 snapshot; synthetic test fixtures intentionally differ.")
        for package in ("pyarrow","matplotlib"):
            meta["versions"][package]=importlib.metadata.version(package)
        corrected,correction=v1.apply_benchmark_correction(original)
        correction.to_csv(output/"correction_log.csv",index=False)
        meta["correction_history"]=correction.to_dict("records")
        reproduction=v1._reproduction(original,us,jp,scores)
        reproduction.to_csv(output/"momentum_reproduction.csv",index=False)
        meta["momentum_reproduction"]=reproduction.status.value_counts().to_dict()
        if reproduction.status.ne("matched").any():
            warnings.append("Saved final momentum does not fully reproduce; see original-union-calendar momentum_reproduction.csv. Cause unresolved; common-calendar values are a distinct definition.")
        daily,landmarks,matches,balance,outcomes=_case(corrected,us,jp)
        if (daily.reference_high_confirmed_date.gt(daily.date)|daily.reference_low_confirmed_date.gt(daily.date)).any():
            raise AssertionError("A future confirmed pivot entered structure features")
        summary,uncertainty=summarize_structure(outcomes,int(bootstrap_samples),int(seed))
        tables=structure_tables(landmarks)
        daily.to_parquet(output/"daily_structure_features.parquet",index=False)
        landmarks.to_csv(output/"landmark_observations.csv",index=False)
        outcomes.to_csv(output/"structure_outcomes.csv",index=False)
        _paired_comparisons(outcomes,matches).to_csv(output/"matched_structure_comparisons.csv",index=False)
        matches.to_csv(output/"structure_matches.csv",index=False)
        balance.to_csv(output/"matching_balance.csv",index=False)
        summary.to_csv(output/"structure_outcome_summary.csv",index=False)
        uncertainty.to_csv(output/"uncertainty.csv",index=False)
        _freshness_outcomes(landmarks,outcomes).to_csv(output/"freshness_outcomes.csv",index=False)
        _daily_durations(daily).to_csv(output/"daily_state_durations.csv",index=False)
        filenames={"distribution":"structure_distribution.csv","transitions":"structure_transitions.csv","durations":"state_durations.csv",
                   "agreement":"structure_agreement.csv","extreme":"extreme_momentum_analysis.csv","extreme_transitions":"extreme_transitions.csv",
                   "freshness":"freshness_analysis.csv","concentration":"observation_concentration.csv","feature_distribution":"structure_feature_distribution.csv","repetitions":"ticker_repetitions.csv"}
        for name,frame in tables.items():
            frame.to_csv(output/filenames[name],index=False)
        cases=[("primary_corrected",5,0,False),("span3",3,0,False),("span10",10,0,False),("phase7",5,7,False),("phase14",5,14,False),
               ("uncorrected",5,0,False),("exclude_influence",5,0,True)]
        meta["parameters"]["sensitivity_specifications"]=[{"name":name,"span":span,"phase":phase,"exclude_influence":exclude} for name,span,phase,exclude in cases]
        sensitivity=[]
        directory=output/"sensitivity"
        directory.mkdir()
        for name,span,phase,exclude in cases:
            if name=="primary_corrected":
                alternative,alt_matches,alt_outcomes,alt_summary=landmarks,matches,outcomes,summary
            else:
                _,alternative,alt_matches,_,alt_outcomes=_case(original if name in ("uncorrected","exclude_influence") else corrected,us,jp,span=span,phase=phase,exclude_influence=exclude)
                alt_summary,_=summarize_structure(alt_outcomes,0,int(seed),sensitivity=True)
            part=alt_summary.copy()
            part=part.loc[part.horizon.eq(21)&part.period.eq("ALL")&part.momentum_3m_sign.eq("all")&part.cohort.isin(["all","raw_top10"])]
            part["specification"],part["span"],part["phase"]=name,span,phase
            sensitivity.append(part)
            for label,frame in (("observations",alternative),("matches",alt_matches),("outcomes",alt_outcomes)):
                frame.to_csv(directory/f"{name}_{label}.csv",index=False)
        pd.concat(sensitivity,ignore_index=True).to_csv(output/"sensitivity_analysis.csv",index=False)
        reasons=landmarks.loc[~landmarks.classified].groupby(["market","series_kind","exclusion_reason"],dropna=False).size().rename("count").reset_index()
        reasons["warning"]=reasons.exclusion_reason
        quality=pd.concat([reasons,pd.DataFrame([{"market":"ALL","series_kind":"separate","warning":note} for note in warnings])],ignore_index=True)
        quality.to_csv(output/"data_quality_warnings.csv",index=False)
        meta["observation_counts"]={"scheduled_rows":len(landmarks),"scheduled_stock_dates":int(landmarks.loc[landmarks.series_kind=="price"].shape[0]),
                                    "classified_rows":int(landmarks.classified.sum()),"unclassified_rows":int((~landmarks.classified).sum()),
                                    "state_counts":landmarks.groupby(["series_kind","state"]).size().to_dict()}
        # JSON object keys cannot be tuples.
        meta["observation_counts"]["state_counts"]={f"{kind}:{state}":int(count) for (kind,state),count in landmarks.groupby(["series_kind","state"]).size().items()}
        meta["matching_counts"]=matches.status.value_counts().to_dict()
        meta["outcome_counts"]=outcomes.groupby(["horizon","status"]).size().rename_axis(["horizon","status"]).reset_index(name="count").to_dict("records")
        meta["unique_tickers"]=int(landmarks.ticker.nunique())
        meta["eligible_unique_tickers"]=int(landmarks.loc[landmarks.history_eligible,"ticker"].nunique())
        meta["classified_unique_tickers"]=int(landmarks.loc[landmarks.classified,"ticker"].nunique())
        meta["never_classified_tickers"]=len(us)+len(jp)-meta["classified_unique_tickers"]
        meta["observation_counts_by_series"]={kind:{"scheduled":len(group),"history_eligible":int(group.history_eligible.sum()),"classified":int(group.classified.sum()),"unclassified":int((~group.classified).sum())} for kind,group in landmarks.groupby("series_kind")}
        meta["unique_market_dates"]=int(landmarks[["market","date"]].drop_duplicates().shape[0])
        _charts(output/"charts",landmarks,matches,outcomes,tables)
        if {name:v1._hash(snapshot/name) for name in hashes}!=hashes:
            raise AssertionError("Input snapshot changed during analysis")
        if _verify_code()["protected_source_sha256"]!=meta["protected_source_sha256"]:
            raise AssertionError("Protected source changed during analysis")
        meta.update({"status":"completed","validation_status":"passed","finished_at":datetime.now(timezone.utc).isoformat(),"input_unchanged":True,"protected_source_unchanged":True})
        v1._json(output/"study_metadata.json",meta)
        (output/"study_report.md").write_text(_report(meta,summary),encoding="utf-8")
        v1._json(output/"file_manifest.json",{"algorithm":"sha256","self_excluded":"file_manifest.json","files":{p.relative_to(output).as_posix():v1._hash(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    except Exception as exc:
        meta.update({"status":"failed","validation_status":"failed","finished_at":datetime.now(timezone.utc).isoformat(),"error":f"{type(exc).__name__}: {exc}"})
        v1._json(output/"study_metadata.json",meta)
        (output/"study_report.md").write_text(f"# Incomplete Dow Structure Study\n\n{meta['error']}\n\nNo completed study or superiority claim.\n",encoding="utf-8")
        raise StudyRunError(meta["error"],output) from exc
    return output


def _report(meta,summary):
    primary=summary.loc[(summary.horizon==21)&(summary.period=="ALL")&(summary.series_kind=="price")&(summary.cohort=="all")&(summary.market=="ALL")&(summary.momentum_3m_sign=="all")]
    lines=["# Dow Structure Study v0.2","","Exploratory retrospective research; no law or predictive superiority is presumed.",
           "","## Purpose and fixed methods","","Observe every saved candidate on shared market landmarks, without requiring a low-break alert. Primary: price structure, span5, market interval21, horizon21 after next-close entry.",
           "","Four-feature fixed-caliper comparisons supplement unadjusted distributions; matched subsets do not represent all unmatched names. PhaseII raw-top10 labels use only contemporaneous raw stock momentum; its adjusted controls remain the frozen main population controls.",
           "","## Primary descriptive results","","```",primary.to_string(index=False),"```","","## Integrity and interpretation","",
           f"Input: {meta['source_snapshot']}",f"Code SHA: {meta['git_commit_sha']}","Input and protected source hashes verified unchanged.",""]
    lines.extend(f"- {warning}" for warning in meta["warnings"])
    lines += ["- See full summaries for equal treatment of all four states, secondary relative structure, markets, years, censoring, quantiles and trimmed means.",
              "- Primary21 uncertainty uses separate ticker/date cluster resampling only. Secondary horizons and year/sign tables are descriptive, with no claim their intervals were computed.",
              "- Structure agreement must distinguish both-classified observations from shared unclassified labels. Sampled landmark transitions and daily model runs have different denominators.",
              "- Sensitivity spans3/10, phases7/14, uncorrected and influence-excluded data are separate; no favorable specification is promoted to primary."]
    return "\n".join(lines)+"\n"


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir",default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output-root",default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples",type=int,default=500)
    parser.add_argument("--seed",type=int,default=20261104)
    args=parser.parse_args(argv)
    try:
        output=run_study(**vars(args))
    except StudyRunError as exc:
        print(f"Study failed: {exc}\nIncomplete output: {exc.run_dir}")
        return 1
    print(f"Study output: {output}")
    return 0


_STRUCTURE_MATCH_FEATURES = ("momentum_3m", "momentum_6m", "momentum_12m", "volatility")


def _structure_frame(landmarks):
    frame = landmarks.copy()
    if "date" not in frame and "signal_date" in frame:
        frame["date"] = frame.signal_date
    if "signal_date" not in frame and "date" in frame:
        frame["signal_date"] = frame.date
    frame["date"] = pd.to_datetime(frame["date"])
    frame["signal_date"] = pd.to_datetime(frame["signal_date"])
    frame["state"] = frame.state.where(frame.state.isin(STATES), "unclassified")
    return frame


def _structure_flag(frame, name):
    return frame[name].fillna(False).astype(bool) if name in frame else pd.Series(False, index=frame.index)


def _structure_cohorts(frame, include_signs=False):
    yield "all", "all", frame
    raw = frame.loc[_structure_flag(frame, "raw_top10")]
    yield "raw_top10", "all", raw
    if include_signs:
        for sign in ("positive", "negative", "zero"):
            if "momentum_3m_sign" in raw:
                sub = raw.loc[raw.momentum_3m_sign == sign]
            else:
                momentum = raw.momentum_3m
                sub = raw.loc[(momentum > 0) if sign == "positive" else (momentum < 0) if sign == "negative" else (momentum == 0)]
            yield "raw_top10", sign, sub
    yield "relative_top10", "all", frame.loc[_structure_flag(frame, "relative_top10")]


def _structure_slices(frame):
    years = sorted(frame.date.dt.year.dropna().astype(int).unique())
    for market in ("ALL", "US", "JP"):
        market_frame = frame if market == "ALL" else frame.loc[frame.market == market]
        yield market, "ALL", market_frame
        for year in years:
            yield market, str(year), market_frame.loc[market_frame.date.dt.year == year]


def _structure_standardized_difference(case, control):
    case = np.asarray(case, dtype=float)
    control = np.asarray(control, dtype=float)
    case, control = case[np.isfinite(case)], control[np.isfinite(control)]
    if not len(case) or not len(control):
        return np.nan
    denominator = np.sqrt((np.var(case) + np.var(control)) / 2)
    difference = float(np.mean(case) - np.mean(control))
    return difference / denominator if denominator > 1e-12 else 0.0 if abs(difference) <= 1e-12 else np.nan


def match_structure_controls(landmarks, caliper=0.5):
    """Match different-state stocks using only each landmark's cross-section.

    All four featurewise absolute standardized differences must be within the
    fixed caliper. Population SD from the contemporaneous classified and
    feature-eligible cross-section is used (floor 1e-12). Nearest Euclidean
    distance, then ticker, chooses one control; controls may be reused.
    """
    if not np.isfinite(caliper) or caliper < 0:
        raise ValueError("Matching caliper must be finite and nonnegative.")
    match_columns = ["event_id", "sample_id", "signal_date", "ticker", "market", "series_kind", "state",
                     "status", "control_ticker", "control_state", "distance", "caliper",
                     "eligible_control_count", "within_caliper_count"]
    for feature in _STRUCTURE_MATCH_FEATURES:
        match_columns.extend([feature, f"control_{feature}", f"scale_{feature}", f"standardized_difference_{feature}"])
    balance_columns = ["date", "market", "series_kind", "state", "feature", "case_count", "eligible_case_count",
                       "matched_count", "no_control_count", "insufficient_features_count", "attrition_fraction",
                       "potential_control_count", "control_unique_count", "control_state_counts",
                       "case_mean_before", "case_std_before", "control_mean_before", "control_std_before",
                       "case_mean_after", "case_std_after", "control_mean_after", "control_std_after",
                       "standardized_difference_before", "standardized_difference_after"]
    if landmarks.empty:
        return pd.DataFrame(columns=match_columns), pd.DataFrame(columns=balance_columns)
    frame = _structure_frame(landmarks)
    if frame.event_id.duplicated().any():
        raise ValueError("Landmark event_id must be unique before matching.")
    finite = np.isfinite(frame[list(_STRUCTURE_MATCH_FEATURES)].to_numpy(dtype=float)).all(axis=1)
    frame["_eligible"] = (finite & (frame.volatility > 0) & (frame.observations >= 253) &
                          _structure_flag(frame, "classified") & _structure_flag(frame, "history_eligible") & frame.state.isin(STATES))
    rows, balance_rows = [], []
    for (date, market, kind), group in frame.groupby(["date", "market", "series_kind"], sort=True):
        eligible = group.loc[group._eligible].sort_values("ticker", kind="stable")
        scales = eligible[list(_STRUCTURE_MATCH_FEATURES)].std(ddof=0).to_numpy(dtype=float)
        scales = np.maximum(np.nan_to_num(scales, nan=0.0), 1e-12)
        group_rows = []
        for case in group.sort_values(["ticker", "event_id"], kind="stable").to_dict("records"):
            row = {"event_id": case["event_id"], "sample_id": case.get("sample_id", case["event_id"]),
                   "signal_date": case["signal_date"], "ticker": case["ticker"], "market": market,
                   "series_kind": kind, "state": case["state"], "status": "no_control", "control_ticker": None,
                   "control_state": None, "distance": np.nan, "caliper": float(caliper),
                   "eligible_control_count": 0, "within_caliper_count": 0}
            for i, feature in enumerate(_STRUCTURE_MATCH_FEATURES):
                row.update({feature: case[feature], f"control_{feature}": np.nan,
                            f"scale_{feature}": float(scales[i]), f"standardized_difference_{feature}": np.nan})
            if case["state"] not in STATES or not case.get("classified", False):
                row["status"] = "unclassified"
            elif not case["_eligible"]:
                row["status"] = "insufficient_features"
            else:
                controls = eligible.loc[(eligible.state != case["state"]) & (eligible.ticker != case["ticker"])].copy()
                row["eligible_control_count"] = len(controls)
                if not controls.empty:
                    delta = ((controls[list(_STRUCTURE_MATCH_FEATURES)].to_numpy(dtype=float) -
                              np.array([case[f] for f in _STRUCTURE_MATCH_FEATURES])) / scales)
                    within = (np.abs(delta) <= caliper + 1e-14).all(axis=1)
                    controls = controls.loc[within].copy()
                    row["within_caliper_count"] = len(controls)
                    if not controls.empty:
                        controls["_distance"] = np.sqrt(np.square(delta[within]).sum(axis=1))
                        best = controls.sort_values(["_distance", "ticker"], kind="stable").iloc[0]
                        row.update({"status": "matched", "control_ticker": best.ticker,
                                    "control_state": best.state, "distance": float(best._distance)})
                        for i, feature in enumerate(_STRUCTURE_MATCH_FEATURES):
                            row[f"control_{feature}"] = float(best[feature])
                            row[f"standardized_difference_{feature}"] = float((case[feature] - best[feature]) / scales[i])
            rows.append(row)
            group_rows.append(row)
        group_matches = pd.DataFrame(group_rows)
        for state in (*STATES, "unclassified"):
            cases = group.loc[group.state == state]
            before_cases = eligible.loc[eligible.state == state]
            before_controls = eligible.loc[eligible.state != state]
            assignments = group_matches.loc[group_matches.state == state]
            matched = assignments.loc[assignments.status == "matched"]
            composition = {s: int((matched.control_state == s).sum()) for s in (*STATES, "unclassified")}
            common = {"date": date, "market": market, "series_kind": kind, "state": state,
                      "case_count": len(cases), "eligible_case_count": len(before_cases), "matched_count": len(matched),
                      "no_control_count": int((assignments.status == "no_control").sum()),
                      "insufficient_features_count": int((assignments.status == "insufficient_features").sum()),
                      "attrition_fraction": 1 - len(matched) / len(cases) if len(cases) else np.nan,
                      "potential_control_count": len(before_controls), "control_unique_count": int(matched.control_ticker.nunique()),
                      "control_state_counts": json.dumps(composition, sort_keys=True)}
            for feature in _STRUCTURE_MATCH_FEATURES:
                case_before = before_cases[feature]
                control_before = before_controls[feature]
                case_after = matched[feature]
                control_after = matched[f"control_{feature}"]
                balance_rows.append({**common, "feature": feature,
                                     "case_mean_before": case_before.mean(), "case_std_before": case_before.std(ddof=0),
                                     "control_mean_before": control_before.mean(), "control_std_before": control_before.std(ddof=0),
                                     "case_mean_after": case_after.mean(), "case_std_after": case_after.std(ddof=0),
                                     "control_mean_after": control_after.mean(), "control_std_after": control_after.std(ddof=0),
                                     "standardized_difference_before": _structure_standardized_difference(case_before, control_before),
                                     "standardized_difference_after": _structure_standardized_difference(case_after, control_after)})
    return pd.DataFrame(rows, columns=match_columns), pd.DataFrame(balance_rows, columns=balance_columns)


def _structure_trimmed_mean(values):
    values = np.sort(np.asarray(values, dtype=float))
    values = values[np.isfinite(values)]
    trim = int(np.floor(0.1 * len(values)))
    return float(np.mean(values[trim:len(values) - trim])) if len(values) else np.nan


def _structure_return_measure(observed):
    def average(name):
        return float(observed[name].mean()) if len(observed) else np.nan
    relative = observed.relative_return.to_numpy(dtype=float)
    return {"mean_stock_return": average("stock_return"), "median_stock_return": float(observed.stock_return.median()) if len(observed) else np.nan,
            "mean_relative_return": average("relative_return"), "median_relative_return": float(np.median(relative)) if len(relative) else np.nan,
            "trimmed_mean_relative_return": _structure_trimmed_mean(relative),
            "relative_q10": float(np.quantile(relative, .1)) if len(relative) else np.nan,
            "relative_q90": float(np.quantile(relative, .9)) if len(relative) else np.nan,
            "loss_probability": float((observed.stock_return < 0).mean()) if len(observed) else np.nan,
            "underperformance_probability": float((relative < 0).mean()) if len(relative) else np.nan,
            "mean_max_drawdown": average("max_drawdown"),
            "median_max_drawdown": float(observed.max_drawdown.median()) if len(observed) else np.nan}


def summarize_structure(outcomes, bootstrap_samples=500, seed=20261104, *, sensitivity=False):
    """Report every state equally, with exploratory separate cluster intervals.

    Phase II adjusted rows subset the existing all-population matches: they do
    not rematch inside the top-decile cohort. Both sides of a pair must have an
    observed horizon before either matched return enters its summary.
    """
    if isinstance(bootstrap_samples, bool) or int(bootstrap_samples) != bootstrap_samples or bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be a nonnegative integer.")
    summary_columns = ["cohort", "momentum_3m_sign", "market", "period", "series_kind", "state", "horizon", "primary_horizon",
                       "group", "comparison_population", "total_count", "assigned_count", "valid_count", "right_censored_count",
                       "unmatched_count", "excluded_influence_count", "other_unobserved_count", "unique_ticker_count", "unique_date_count",
                       "cohort_unique_ticker_count", "cohort_unique_date_count", "mean_stock_return", "median_stock_return",
                       "mean_relative_return", "median_relative_return", "trimmed_mean_relative_return", "relative_q10", "relative_q90",
                       "loss_probability", "underperformance_probability", "mean_max_drawdown", "median_max_drawdown",
                       "loss_probability_difference", "underperformance_probability_difference"]
    uncertainty_columns = ["cohort", "momentum_3m_sign", "market", "period", "series_kind", "state", "horizon", "primary_horizon",
                           "group", "valid_count", "mean_relative_return", "ticker_cluster_count", "date_cluster_count",
                           "ticker_ci_low", "ticker_ci_high", "date_ci_low", "date_ci_high", "bootstrap_samples", "seed", "warning"]
    if outcomes.empty:
        return pd.DataFrame(columns=summary_columns), pd.DataFrame(columns=uncertainty_columns)
    frame = _structure_frame(outcomes)
    rng = np.random.default_rng(seed)
    summary_rows, uncertainty_rows = [], []
    empty_measure = {name: np.nan for name in ("mean_stock_return", "median_stock_return", "mean_relative_return",
                                              "median_relative_return", "trimmed_mean_relative_return", "relative_q10", "relative_q90",
                                              "loss_probability", "underperformance_probability", "mean_max_drawdown", "median_max_drawdown")}
    for cohort, sign, cohort_frame in _structure_cohorts(frame, include_signs=not sensitivity):
        if sensitivity and cohort=="relative_top10":
            continue
        for market, period, sample in _structure_slices(cohort_frame):
            if sensitivity and period!="ALL":
                continue
            for kind in ("price", "relative"):
                for horizon in ((21,) if sensitivity else (5, 21, 63)):
                    part = sample.loc[(sample.series_kind == kind) & (sample.horizon == horizon)]
                    for state in (*STATES, "unclassified"):
                        subset = part.loc[part.state == state]
                        if subset.empty:
                            base = {"cohort": cohort, "momentum_3m_sign": sign, "market": market, "period": period,
                                    "series_kind": kind, "state": state, "horizon": horizon, "primary_horizon": horizon == 21,
                                    "total_count": 0, "assigned_count": 0, "cohort_unique_ticker_count": 0, "cohort_unique_date_count": 0}
                            for group_name in ("unadjusted", "focal_matched", "control_matched", "paired_difference"):
                                row = {**base, "group": group_name,
                                       "comparison_population": "main_population_controls" if group_name != "unadjusted" else "all_focal_landmarks",
                                       "valid_count": 0, "right_censored_count": 0, "unmatched_count": 0,"excluded_influence_count":0,
                                       "other_unobserved_count": 0, "unique_ticker_count": 0, "unique_date_count": 0,
                                       **empty_measure, "loss_probability_difference": np.nan, "underperformance_probability_difference": np.nan}
                                summary_rows.append(row)
                                if (bootstrap_samples and period == "ALL" and sign == "all" and horizon == 21 and
                                        cohort in ("all", "raw_top10") and state in STATES and
                                        group_name in ("unadjusted", "paired_difference")):
                                    uncertainty_rows.append({key: row[key] for key in ("cohort", "momentum_3m_sign", "market", "period", "series_kind", "state",
                                                                                      "horizon", "primary_horizon", "group", "valid_count", "mean_relative_return")} |
                                                            {"ticker_cluster_count": 0, "date_cluster_count": 0, "ticker_ci_low": np.nan, "ticker_ci_high": np.nan,
                                                             "date_ci_low": np.nan, "date_ci_high": np.nan, "bootstrap_samples": int(bootstrap_samples),
                                                             "seed": int(seed), "warning": "No observed sample; uncertainty unavailable."})
                            continue
                        focal = subset.loc[subset.group == "alert"]
                        controls = subset.loc[subset.group == "control"]
                        paired = focal.merge(controls, on="event_id", suffixes=("_focal", "_control"), how="left", validate="one_to_one")
                        assigned_mask = paired.ticker_control.notna() & (paired.status_control != "unmatched")
                        complete = paired.loc[(paired.status_focal == "observed") & (paired.status_control == "observed")]
                        assigned = int(assigned_mask.sum())
                        intentionally_excluded = assigned_mask & (paired.status_focal.eq("excluded_influence")|paired.status_control.eq("excluded_influence"))
                        matched_censored = int((assigned_mask & ~intentionally_excluded & ((paired.status_focal == "right_censored") |
                                                               (paired.status_control == "right_censored"))).sum())
                        base = {"cohort": cohort, "momentum_3m_sign": sign, "market": market, "period": period,
                                "series_kind": kind, "state": state, "horizon": horizon, "primary_horizon": horizon == 21,
                                "total_count": len(focal), "assigned_count": assigned,
                                "cohort_unique_ticker_count": int(focal.ticker.nunique()),
                                "cohort_unique_date_count": int(focal.signal_date.nunique())}
                        raw_observed = focal.loc[focal.status == "observed"]
                        datasets = {"unadjusted": raw_observed}
                        for name, side in (("focal_matched", "focal"), ("control_matched", "control")):
                            datasets[name] = pd.DataFrame({column: complete[f"{column}_{side}"].to_numpy()
                                                          for column in ("ticker", "signal_date", "stock_return", "relative_return", "max_drawdown")})
                        difference = pd.DataFrame({column: datasets["focal_matched"][column].copy()
                                                   for column in ("ticker", "signal_date")})
                        for column in ("stock_return", "relative_return", "max_drawdown"):
                            difference[column] = datasets["focal_matched"][column] - datasets["control_matched"][column]
                        datasets["paired_difference"] = difference
                        for group_name, observed in datasets.items():
                            if group_name == "unadjusted":
                                censored = int((focal.status == "right_censored").sum())
                                unmatched = 0
                                excluded = int(focal.status.eq("excluded_influence").sum())
                                other = len(focal) - len(observed) - censored - excluded
                            else:
                                censored = matched_censored
                                unmatched = len(focal) - assigned
                                excluded = int(intentionally_excluded.sum())
                                other = assigned - len(observed) - censored - excluded
                            measure = _structure_return_measure(observed)
                            loss_difference = under_difference = np.nan
                            if group_name == "paired_difference":
                                measure["loss_probability"] = measure["underperformance_probability"] = np.nan
                                if len(complete):
                                    loss_difference = float((complete.stock_return_focal < 0).astype(float).mean() -
                                                            (complete.stock_return_control < 0).astype(float).mean())
                                    under_difference = float((complete.relative_return_focal < 0).astype(float).mean() -
                                                             (complete.relative_return_control < 0).astype(float).mean())
                            row = {**base, "group": group_name,
                                   "comparison_population": "main_population_controls" if group_name != "unadjusted" else "all_focal_landmarks",
                                   "valid_count": len(observed), "right_censored_count": censored, "unmatched_count": unmatched,
                                   "excluded_influence_count":excluded,"other_unobserved_count": other, "unique_ticker_count": int(observed.ticker.nunique()),
                                   "unique_date_count": int(observed.signal_date.nunique()), **measure,
                                   "loss_probability_difference": loss_difference, "underperformance_probability_difference": under_difference}
                            summary_rows.append(row)
                            if (bootstrap_samples and period == "ALL" and sign == "all" and horizon == 21 and
                                    cohort in ("all", "raw_top10") and state in STATES and
                                    group_name in ("unadjusted", "paired_difference")):
                                low_t, high_t, count_t = v1._cluster_mean_interval(observed.relative_return.to_numpy(), observed.ticker,
                                                                               int(bootstrap_samples), rng)
                                low_d, high_d, count_d = v1._cluster_mean_interval(observed.relative_return.to_numpy(), observed.signal_date,
                                                                               int(bootstrap_samples), rng)
                                warning = "Exploratory separate ticker/date cluster intervals; not a calibrated two-way interval; no significance claim."
                                if min(count_t, count_d) < 10:
                                    warning += " Fewer than 10 ticker/date clusters."
                                uncertainty_rows.append({key: row[key] for key in ("cohort", "momentum_3m_sign", "market", "period", "series_kind", "state",
                                                                                  "horizon", "primary_horizon", "group", "valid_count", "mean_relative_return")} |
                                                        {"ticker_cluster_count": count_t, "date_cluster_count": count_d,
                                                         "ticker_ci_low": low_t, "ticker_ci_high": high_t, "date_ci_low": low_d, "date_ci_high": high_d,
                                                         "bootstrap_samples": int(bootstrap_samples), "seed": int(seed), "warning": warning})
    return pd.DataFrame(summary_rows, columns=summary_columns), pd.DataFrame(uncertainty_rows, columns=uncertainty_columns)


def _structure_distribution_tables(frame):
    distribution, feature_rows, extreme = [], [], []
    for cohort, sign, cohort_frame in _structure_cohorts(frame, include_signs=True):
        for market, period, sample in _structure_slices(cohort_frame):
            for kind in ("price", "relative"):
                part = sample.loc[sample.series_kind == kind]
                for state in (*STATES, "unclassified"):
                    state_frame = part.loc[part.state == state]
                    row = {"cohort": cohort, "momentum_3m_sign": sign, "market": market, "period": period,
                           "series_kind": kind, "state": state, "count": len(state_frame), "denominator": len(part),
                           "proportion": len(state_frame) / len(part) if len(part) else np.nan,
                           "history_eligible_count": int(_structure_flag(state_frame, "history_eligible").sum()),
                           "unique_ticker_count": int(state_frame.ticker.nunique()), "unique_date_count": int(state_frame.date.nunique())}
                    if sign == "all":
                        distribution.append(row)
                    extreme.append(row)
                    if sign == "all":
                        for feature in (*_STRUCTURE_MATCH_FEATURES, "relative_momentum_12m"):
                            values = state_frame[feature].dropna() if feature in state_frame else pd.Series(dtype=float)
                            values = values.loc[np.isfinite(values.to_numpy(dtype=float))]
                            feature_rows.append({**{k: row[k] for k in ("cohort", "market", "period", "series_kind", "state")},
                                                 "feature": feature, "valid_count": len(values), "mean": values.mean(), "std": values.std(ddof=0),
                                                 "q10": values.quantile(.1), "median": values.median(), "q90": values.quantile(.9)})
    return pd.DataFrame(distribution), pd.DataFrame(feature_rows), pd.DataFrame(extreme)


def _structure_transition_tables(frame):
    observations = frame.sort_values(["market", "series_kind", "ticker", "market_landmark_number"], kind="stable").copy()
    group_keys = ["market", "series_kind", "ticker"]
    grouped = observations.groupby(group_keys, sort=False)
    observations["from_state"] = grouped.state.shift()
    observations["from_date"] = grouped.date.shift()
    observations["from_landmark_number"] = grouped.market_landmark_number.shift()
    for flag in ("raw_top10", "relative_top10"):
        observations[flag] = _structure_flag(observations, flag)
        observations[f"from_{flag}"] = observations.groupby(group_keys, sort=False)[flag].shift().fillna(False).astype(bool)
    transitions = observations.loc[(observations.market_landmark_number - observations.from_landmark_number == 1)].copy()
    rows = []
    for cohort in ("all", "raw_top10", "relative_top10"):
        # Membership is fixed at the FROM landmark. Requiring membership at
        # the next landmark would select on future momentum and hide exits.
        cohort_transitions = transitions if cohort == "all" else transitions.loc[transitions[f"from_{cohort}"]]
        for market, period, sample in _structure_slices(cohort_transitions):
            for kind in ("price", "relative"):
                part = sample.loc[sample.series_kind == kind]
                counts = part.groupby(["from_state", "state"]).size()
                source_counts = part.groupby("from_state").size()
                for before in (*STATES, "unclassified"):
                    for after in (*STATES, "unclassified"):
                        count = int(counts.get((before, after), 0))
                        denominator = int(source_counts.get(before, 0))
                        cell = part.loc[(part.from_state == before) & (part.state == after)]
                        retained = int(cell[cohort].sum()) if cohort != "all" else len(cell)
                        rows.append({"cohort": cohort, "market": market, "period": period, "series_kind": kind,
                                     "from_state": before, "to_state": after, "count": count, "from_count": denominator,
                                     "transition_probability": count / denominator if denominator else np.nan,
                                     "to_cohort_count": retained, "cohort_exit_count": count - retained if cohort != "all" else 0,
                                     "cohort_rule": "all consecutive landmark pairs" if cohort == "all" else "FROM-date top10 membership only; next-date exits retained",
                                     "observation_scope": "sampled consecutive market landmarks; daily transitions are not observed"})
    duration_rows = []
    for (market, kind, ticker), group in observations.groupby(group_keys, sort=True):
        group = group.sort_values("market_landmark_number", kind="stable")
        new_run = ((group.state != group.state.shift()) | (group.market_landmark_number.diff() != 1)).cumsum()
        for run_number, run in group.groupby(new_run, sort=True):
            duration_rows.append({"market": market, "series_kind": kind, "ticker": ticker, "state": run.state.iloc[0],
                                  "sampled_run_number": int(run_number), "start_date": run.date.iloc[0], "end_date": run.date.iloc[-1],
                                  "observed_landmark_count": len(run), "calendar_span_days": int((run.date.iloc[-1] - run.date.iloc[0]).days),
                                  "first_landmark_number": int(run.market_landmark_number.iloc[0]),
                                  "last_landmark_number": int(run.market_landmark_number.iloc[-1]),
                                  "interpretation": "same state at consecutive sampled landmarks; intervening hidden state changes remain possible"})
    result = pd.DataFrame(rows)
    return result.loc[result.cohort == "all"].copy(), result.loc[result.cohort != "all"].copy(), pd.DataFrame(duration_rows)


def _structure_agreement_table(frame):
    columns = ["ticker", "market", "date", "state", "raw_top10", "relative_top10"]
    work = frame.copy()
    for flag in ("raw_top10", "relative_top10"):
        work[flag] = _structure_flag(work, flag)
    price = work.loc[work.series_kind == "price", columns]
    relative = work.loc[work.series_kind == "relative", columns]
    pairs = price.merge(relative, on=["ticker", "market", "date"], how="outer", suffixes=("_price", "_relative"), validate="one_to_one")
    for side in ("price", "relative"):
        pairs[f"state_{side}"] = pairs[f"state_{side}"].fillna("unclassified")
    for flag in ("raw_top10", "relative_top10"):
        pairs[flag] = pairs[f"{flag}_price"].fillna(False).astype(bool) | pairs[f"{flag}_relative"].fillna(False).astype(bool)
    rows = []
    for cohort in ("all", "raw_top10", "relative_top10"):
        cohort_pairs = pairs if cohort == "all" else pairs.loc[pairs[cohort]]
        for market, period, sample in _structure_slices(cohort_pairs):
            both = sample.state_price.isin(STATES) & sample.state_relative.isin(STATES)
            agreeing = sample.state_price == sample.state_relative
            counts = sample.groupby(["state_price", "state_relative"]).size()
            for price_state in (*STATES, "unclassified"):
                for relative_state in (*STATES, "unclassified"):
                    count = int(counts.get((price_state, relative_state), 0))
                    rows.append({"cohort": cohort, "market": market, "period": period, "price_state": price_state,
                                 "relative_state": relative_state, "count": count, "denominator": len(sample),
                                 "joint_proportion": count / len(sample) if len(sample) else np.nan,
                                 "both_classified_count": int(both.sum()), "matching_classified_count": int((both & agreeing).sum()),
                                 "agreement_rate": float(agreeing.loc[both].mean()) if both.any() else np.nan,
                                 "including_unclassified_agreement_rate": float(agreeing.mean()) if len(sample) else np.nan})
    return pd.DataFrame(rows)


def _structure_freshness_table(frame):
    work = frame.copy()
    for source, target in (("high_age", "high_age_bin"), ("low_age", "low_age_bin")):
        values = work[source] if source in work else pd.Series(np.nan, index=work.index)
        work[target] = pd.cut(values, bins=[0, 21, 63, np.inf], right=False,
                              labels=["0-20", "21-62", "63+"]).astype(object).fillna("missing")
    high = work.distance_to_high if "distance_to_high" in work else pd.Series(np.nan, index=work.index)
    low = work.distance_to_low if "distance_to_low" in work else pd.Series(np.nan, index=work.index)
    available = high.notna() & low.notna()
    work["position"] = "unavailable"
    work.loc[available, "position"] = "within"
    work.loc[available & (high > 0), "position"] = "above_high"
    work.loc[available & (low < 0), "position"] = "below_low"
    work.loc[available & (high > 0) & (low < 0), "position"] = "ambiguous"
    rows = []
    for cohort, sign, cohort_frame in _structure_cohorts(work):
        for (market, kind, state, high_bin, low_bin, position), sample in cohort_frame.groupby(
                ["market", "series_kind", "state", "high_age_bin", "low_age_bin", "position"], sort=True):
            rows.append({"cohort": cohort, "market": market, "series_kind": kind, "state": state,
                         "high_age_bin": high_bin, "low_age_bin": low_bin, "position": position, "count": len(sample),
                         "mean_momentum_3m": sample.momentum_3m.mean(), "analysis_scope": "exploratory freshness; not a post-hoc primary filter"})
    return pd.DataFrame(rows, columns=["cohort", "market", "series_kind", "state", "high_age_bin", "low_age_bin", "position", "count", "mean_momentum_3m", "analysis_scope"])


def structure_tables(landmarks):
    """Descriptive tables retain unclassified and report sample-based continuity."""
    if landmarks.empty:
        return {name: pd.DataFrame() for name in ("distribution", "feature_distribution", "transitions", "durations", "agreement",
                                                 "extreme", "extreme_transitions", "freshness", "concentration", "repetitions")}
    frame = _structure_frame(landmarks)
    keys = ["ticker", "market", "series_kind", "market_landmark_number"]
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate ticker/market/series landmark observations.")
    distribution, features, extreme = _structure_distribution_tables(frame)
    transitions, extreme_transitions, durations = _structure_transition_tables(frame)
    concentration = frame.groupby(["market", "date", "series_kind"], sort=True).agg(
        sample_count=("ticker", "size"), unique_ticker_count=("ticker", "nunique"),
        classified_count=("classified", "sum"), history_eligible_count=("history_eligible", "sum")).reset_index()
    repetitions = frame.groupby(["market", "ticker", "series_kind"], sort=True).agg(
        observation_count=("date", "size"), unique_date_count=("date", "nunique"),
        classified_count=("classified", "sum"), first_date=("date", "min"), last_date=("date", "max")).reset_index()
    return {"distribution": distribution, "feature_distribution": features, "transitions": transitions,
            "durations": durations, "agreement": _structure_agreement_table(frame), "extreme": extreme,
            "extreme_transitions": extreme_transitions, "freshness": _structure_freshness_table(frame),
            "concentration": concentration, "repetitions": repetitions}


if __name__ == "__main__":
    raise SystemExit(main())
