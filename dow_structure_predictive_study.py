"""Dow Structure Predictive Study v0.3: retrospective, expanding forecasts.

All inputs are immutable saved research artifacts. No market-data acquisition,
production screening, order, allocation, or TTL code is executed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import dow_momentum_study as v1
import dow_structure_study as v2

BASE_COMMIT = "1b2ecb4267d176b327d3995ed0bbc3d7a2b273f6"
DEFAULT_SNAPSHOT = v1.DEFAULT_SNAPSHOT
DEFAULT_V02 = "/content/drive/MyDrive/alpha_engine/analysis/dow_structure_v02/20261111T010459Z_34a3f71bec76"
DEFAULT_OUTPUT = "/content/drive/MyDrive/alpha_engine/analysis/dow_predictive_v03"
TARGET_NAMES = ("deep_drawdown_63", "rank_change_21", "negative_flip_21", "relative_return_63")
NUMERIC_FEATURES = ("momentum_3m", "momentum_6m", "momentum_12m", "volatility", "rank_percentile")
INTEGRITY_KEYS = ("input_unchanged", "protected_unchanged", "training_causal", "input_parity", "replay_verified", "predictions_saved")


class StudyRunError(RuntimeError):
    def __init__(self, message, run_dir):
        super().__init__(message)
        self.run_dir = run_dir


def _progress(stage, **counts):
    print(f"[v0.3] {stage}: " + ", ".join(f"{key}={value}" for key,value in counts.items()), flush=True)


def _boolean(series):
    return series.map(lambda x: str(x).lower() in ("true","1","1.0")).fillna(False).astype(bool)


def verify_manifest(folder):
    """Verify all declared files, refusing paths outside the supplied artifact."""
    folder = Path(folder).resolve()
    manifest_path = folder/"file_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files,dict) or not files:
        raise ValueError("Artifact manifest has no files")
    hashes = {"file_manifest.json":v1._hash(manifest_path)}
    for relative, expected in files.items():
        path = (folder/relative).resolve()
        if Path(relative).is_absolute() or folder not in path.parents:
            raise ValueError("Manifest path escapes the artifact folder")
        actual = v1._hash(path)
        digest = expected.get("sha256") if isinstance(expected,dict) else expected
        if actual != digest:
            raise ValueError(f"Manifest hash mismatch: {relative}")
        hashes[relative] = actual
    return hashes


def _verify_code():
    root = Path(__file__).resolve().parent
    def git(*args):
        return subprocess.check_output(["git","-C",str(root),*args],stderr=subprocess.PIPE)
    sha = git("rev-parse","HEAD").decode().strip()
    git("merge-base","--is-ancestor",BASE_COMMIT,sha)
    hashes = {}
    for name in (*v2.PROTECTED,"dow_structure_study.py"):
        actual = (root/name).read_bytes()
        if actual != git("show",f"{BASE_COMMIT}:{name}"):
            raise ValueError(f"Protected code differs from v0.2: {name}")
        hashes[name] = hashlib.sha256(actual).hexdigest()
    return {"git_commit_sha":sha,"protected_source_sha256":hashes,"study_source_sha256":v1._hash(Path(__file__))}


def build_features(landmarks, series_kind="price"):
    """Recompute ranks BEFORE classification filtering, from each past cross-section."""
    if series_kind not in ("price","relative"):
        raise ValueError("One price or relative series must be selected")
    frame = landmarks.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if "signal_date" not in frame:
        frame["signal_date"] = frame.date
    else:
        frame["signal_date"] = pd.to_datetime(frame.signal_date)
    for field in ("reference_high_confirmed_date","reference_low_confirmed_date"):
        if field in frame and pd.to_datetime(frame[field]).gt(frame.date).any():
            raise ValueError("Future confirmed pivot in saved v0.2 features")
    for field in ("classified","history_eligible"):
        frame[field] = _boolean(frame[field])
    for field in ("momentum_3m","momentum_6m","momentum_12m","volatility"):
        frame[field] = pd.to_numeric(frame[field],errors="coerce")
    if frame.duplicated(["date","market","ticker","series_kind"]).any():
        raise ValueError("Duplicate stock/market/series landmark")
    price = frame.loc[frame.series_kind=="price"]
    if price.empty:
        price = frame.loc[frame.series_kind==series_kind]
    rank_rows = []
    for (date,market),group in price.groupby(["date","market"],sort=True):
        eligible = group.loc[group.history_eligible&np.isfinite(group.momentum_12m)].sort_values(["momentum_12m","ticker"],ascending=[False,True],kind="stable")
        n = len(eligible)
        ranks = dict(zip(eligible.ticker,range(1,n+1)))
        for row in group.itertuples():
            rank = ranks.get(row.ticker,np.nan)
            percentile = (rank-1)/(n-1) if n>1 else np.nan
            rank_rows.append({"date":date,"market":market,"ticker":row.ticker,"rank_percentile":percentile,
                              "recomputed_rank_12m":rank,"recomputed_rank_population":n,
                              "recomputed_raw_top10":bool(rank<=int(np.ceil(.1*n)) and row.momentum_12m>0)})
    result = frame.loc[frame.series_kind==series_kind].merge(pd.DataFrame(rank_rows),on=["date","market","ticker"],how="left",validate="many_to_one")
    if "raw_top10" in result:
        saved = _boolean(result.raw_top10)
        result["saved_raw_top10"] = saved
        result["source_cohort_agreement"] = saved.eq(result.recomputed_raw_top10.astype(bool))
    result["raw_top10"] = result.recomputed_raw_top10.astype(bool)
    result["obs_id"] = result["event_id"] if "event_id" in result else [f"{r.market}|{r.ticker}|{series_kind}|{r.date.date()}" for r in result.itertuples()]
    if result.obs_id.duplicated().any():
        raise ValueError("Predictive observation IDs must be unique")
    finite = np.isfinite(result[list(NUMERIC_FEATURES)].to_numpy(dtype=float)).all(axis=1)
    result["eligible"] = result.classified&result.history_eligible&result.state.isin(v2.STATES)&finite&(result.volatility>=0)&result.market.isin(["US","JP"])
    result["exclusion_reason"] = np.select([~result.history_eligible,~result.classified,~finite,result.volatility.lt(0)],
                                            ["insufficient_history_or_missing_price","unclassified_structure","missing_or_invalid_predictor","negative_volatility"],default="")
    result["momentum_3m_sign"] = np.select([result.momentum_3m.isna(),result.momentum_3m>0,result.momentum_3m<0],["missing","positive","negative"],default="zero")
    return result.sort_values(["date","market","ticker"]).reset_index(drop=True)


def _risk_targets(features, prices, us, jp):
    events = features.copy()
    events["event_id"] = events.obs_id
    events["signal_date"] = events.date
    # The v0.1 evaluator is the unchanged v0.2 entry/path measurement mechanism.
    measured = v1.evaluate_outcomes(events,pd.DataFrame(),prices,us,jp,horizons=(63,))
    measured = measured.loc[measured.group=="alert"].set_index("event_id")
    rows = []
    for feature in features.itertuples():
        row = measured.loc[feature.obs_id]
        status = row.status
        if status == "missing_signal_observation":
            status = "price_missing"
        if status == "right_censored":
            benchmark = prices[v1.BENCHMARKS[feature.market]].dropna()
            if int((benchmark.index>feature.date).sum())>=64:
                status = "price_missing"
        anomaly = False
        if feature.market=="JP" and pd.notna(row.entry_date):
            pair = prices[[feature.ticker,"1306.T"]].dropna() if feature.ticker in prices else pd.DataFrame()
            if pd.notna(row.exit_date):
                anomaly = bool(pair.loc[row.entry_date:row.exit_date].index.isin(v1.SUSPECT_DATES).any())
        for name in ("deep_drawdown_63","relative_return_63"):
            # 100 -> 80 produces -0.19999999999999996 in binary floating
            # point. A fixed tiny tolerance preserves the inclusive -20% rule.
            value = (float(row.max_drawdown<=-.2+1e-12) if name=="deep_drawdown_63" else float(row.relative_return)) if status=="observed" else np.nan
            rows.append({"obs_id":feature.obs_id,"target":name,"target_value":value,"target_status":status,
                         "label_end_date":row.exit_date,"entry_date":row.entry_date,"exit_date":row.exit_date,
                         "path_max_drawdown":row.max_drawdown,"anomaly_influence":anomaly,"top10_retained":np.nan})
    return rows


def build_targets(features, all_landmarks, prices, us, jp, exclude_influence=False):
    """Future outcomes are labels only; every label retains its actual maturity."""
    rows = []
    errors = {}
    try:
        rows.extend(_risk_targets(features,prices,us,jp))
    except Exception as exc:
        errors["risk_path"] = f"{type(exc).__name__}: {exc}"
        for obs_id in features.obs_id:
            for target in ("deep_drawdown_63","relative_return_63"):
                rows.append({"obs_id":obs_id,"target":target,"target_value":np.nan,"target_status":"computation_failed","label_end_date":pd.NaT,"entry_date":pd.NaT,"exit_date":pd.NaT,"error":errors["risk_path"]})
    try:
        full = build_features(all_landmarks,features.series_kind.iloc[0]) if len(features) else features.copy()
        lookup = full.set_index(["market","market_landmark_number","ticker"])
        market_dates = full[["market","market_landmark_number","date"]].drop_duplicates().set_index(["market","market_landmark_number"]).date.to_dict()
        for feature in features.itertuples():
            next_number = int(feature.market_landmark_number)+1
            next_date = market_dates.get((feature.market,next_number),pd.NaT)
            key = (feature.market,next_number,feature.ticker)
            future = lookup.loc[key] if key in lookup.index else None
            status = "right_censored" if pd.isna(next_date) else "price_missing"
            change, future_m3, retained = np.nan,np.nan,np.nan
            future_price_available = future is not None and bool(future.history_eligible)
            if future_price_available:
                future_m3 = float(future.momentum_3m)
            if future is not None and future.history_eligible and np.isfinite(future.rank_percentile) and np.isfinite(feature.rank_percentile):
                change = float(future.rank_percentile-feature.rank_percentile)
                future_m3 = float(future.momentum_3m)
                retained = float(bool(future.raw_top10)) if feature.raw_top10 else np.nan
                status = "observed"
            rows.append({"obs_id":feature.obs_id,"target":"rank_change_21","target_value":change,"target_status":status,
                         "label_end_date":next_date,"entry_date":feature.date,"exit_date":next_date,"top10_retained":retained,"anomaly_influence":False})
            if not np.isfinite(feature.momentum_3m) or feature.momentum_3m<=0:
                flip_status = "not_at_risk_nonpositive" if np.isfinite(feature.momentum_3m) else "invalid_current_momentum"
                flip = np.nan
            else:
                flip_status = "observed" if future_price_available and np.isfinite(future_m3) else "right_censored" if pd.isna(next_date) else "price_missing"
                flip = float(future_m3<=0) if flip_status=="observed" else np.nan
            rows.append({"obs_id":feature.obs_id,"target":"negative_flip_21","target_value":flip,"target_status":flip_status,
                         "label_end_date":next_date,"entry_date":feature.date,"exit_date":next_date,"anomaly_influence":False,"top10_retained":np.nan})
    except Exception as exc:
        errors["persistence"] = f"{type(exc).__name__}: {exc}"
        rows = [row for row in rows if row["target"] not in ("rank_change_21","negative_flip_21")]
        for obs_id in features.obs_id:
            for target in ("rank_change_21","negative_flip_21"):
                rows.append({"obs_id":obs_id,"target":target,"target_value":np.nan,"target_status":"computation_failed","label_end_date":pd.NaT,"entry_date":pd.NaT,"exit_date":pd.NaT,"error":errors["persistence"]})
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=["obs_id","target","target_value","target_status","label_end_date","entry_date","exit_date","anomaly_influence","top10_retained","error"])
    result["anomaly_influence"] = result.anomaly_influence.fillna(False).astype(bool)
    feature_influence = features.set_index("obs_id").get("excluded_influence",pd.Series(False,index=features.obs_id)).fillna(False).astype(bool)
    result["anomaly_influence"] |= result.obs_id.map(feature_influence).fillna(False).astype(bool)
    if exclude_influence:
        affected = result.anomaly_influence&result.target_status.eq("observed")
        result.loc[affected,"target_status"] = "known_anomaly_influence"
        result.loc[affected,"target_value"] = np.nan
    result.attrs["computation_errors"] = errors
    if result.duplicated(["obs_id","target"]).any():
        raise ValueError("Duplicate target IDs")
    return result.sort_values(["obs_id","target"]).reset_index(drop=True)


def load_inputs(snapshot_dir,v02_dir):
    snapshot,v02 = Path(snapshot_dir).resolve(),Path(v02_dir).resolve()
    required=("daily_structure_features.parquet","landmark_observations.csv","structure_outcomes.csv","structure_outcome_summary.csv","extreme_transitions.csv","study_metadata.json")
    hashes = verify_manifest(v02)
    if any(name not in hashes for name in required):
        raise ValueError("Required v0.2 artifacts are missing from its manifest")
    meta=json.loads((v02/"study_metadata.json").read_text(encoding="utf-8"))
    if meta.get("status")!="completed" or meta.get("validation_status")!="passed":
        raise ValueError("v0.2 study is not completed and validated")
    if meta.get("git_commit_sha")!=BASE_COMMIT:
        raise ValueError("v0.2 study commit differs from the specified baseline")
    params=meta.get("parameters",{})
    if (params.get("span"),params.get("landmark_interval"),params.get("landmark_phase"))!=(5,21,0):
        raise ValueError("v0.2 primary span/calendar definition differs")
    prices,us,jp,scores,source_hashes=v1._input_snapshot(snapshot)
    if meta.get("input_hashes")!=source_hashes:
        raise ValueError("v0.2 provenance does not match the supplied fixed price snapshot")
    landmarks=pd.read_csv(v02/"landmark_observations.csv")
    landmarks["date"]=pd.to_datetime(landmarks.date)
    for market,universe in (("US",us),("JP",jp)):
        part=landmarks.loc[(landmarks.market==market)&(landmarks.series_kind=="price")]
        dates=v2.landmark_dates(prices[v1.BENCHMARKS[market]])
        if set(part.date)!=set(dates) or len(part)!=len(dates)*len(universe) or set(part.ticker)!=set(universe):
            raise ValueError("Saved v0.2 landmarks do not cover the fixed candidate/calendar grid")
    return prices,us,jp,landmarks,{"snapshot":source_hashes,"v02":hashes},meta


_PREDICTIVE_NUMERIC_FEATURES = ("momentum_3m", "momentum_6m", "momentum_12m", "volatility", "rank_percentile")
_PREDICTIVE_STATE_FEATURES = ("state_LH_LL", "state_LH_HL", "state_HH_LL")
_PREDICTIVE_STATES = ("HH/HL", "LH/LL", "LH/HL", "HH/LL")
_PREDICTIVE_TARGETS = ("deep_drawdown_63", "rank_change_21", "negative_flip_21", "relative_return_63")
_PREDICTIVE_BINARY_TARGETS = ("deep_drawdown_63", "negative_flip_21")


def _predictive_ids_hash(ids):
    return hashlib.sha256(json.dumps([str(value) for value in ids], ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _predictive_matrix_hash(values):
    values = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(str(values.shape).encode("ascii") + values.tobytes()).hexdigest()


def _predictive_dummy_matrix(frame):
    return np.column_stack([(frame.market == "JP").to_numpy(dtype=float),
                            (frame.state == "LH/LL").to_numpy(dtype=float),
                            (frame.state == "LH/HL").to_numpy(dtype=float),
                            (frame.state == "HH/LL").to_numpy(dtype=float)])


def walk_forward(features, targets, *, seed=20261111, progress=False):
    """Forecast each landmark using labels fully matured strictly before it.

    US/JP are pooled within a series only. Price and relative structures never
    share fitting data. A and B use exactly the same rows and five train-scaled
    numeric features; B's only additions are three fixed state dummies. Test
    label status/value/end date never determines whether a forecast is made.
    """
    import warnings
    import inspect
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler

    prediction_columns = ["obs_id", "event_id", "date", "prediction_date", "ticker", "market", "series_kind", "state",
                          "raw_top10", "momentum_3m_sign", "eligible", "exclusion_reason", "target", "target_status",
                          "target_value", "label_end_date", "entry_date", "exit_date", "model", "status", "prediction_status",
                          "prediction", "error", "train_count", "unique_training_dates", "market_training_dates",
                          "training_feature_max_date", "max_label_end_date", "train_ids_hash", "eval_ids_hash"]
    prediction_columns += list(_PREDICTIVE_NUMERIC_FEATURES)
    audit_columns = ["target", "series_kind", "prediction_date", "model", "status", "error", "train_count",
                     "positive_count", "negative_count", "unique_training_dates", "us_training_dates", "jp_training_dates",
                     "training_feature_max_date", "max_label_end_date", "train_ids_hash", "eval_ids_hash", "train_target_hash",
                     "shared_train_matrix_hash", "shared_eval_matrix_hash", "eval_count", "forecastable_eval_count",
                     "scaler_mean_json", "scaler_scale_json", "numeric_feature_names_json", "feature_names_json",
                     "dummy_baseline", "dummy_scaling", "algorithm", "hyperparameters_json", "estimator_parameters_json", "seed",
                     "invalid_observed_training_label_count", "feature_input_parity"]
    schedule_columns = ["target", "series_kind", "prediction_date", "market", "row_count", "feature_eligible_count",
                        "not_at_risk_count", "forecastable_count", "train_count", "unique_training_dates", "market_training_dates",
                        "positive_count", "negative_count", "training_feature_max_date", "max_label_end_date",
                        "train_ids_hash", "eval_ids_hash", "status"]
    if features.empty:
        return (pd.DataFrame(columns=prediction_columns), pd.DataFrame(columns=audit_columns),
                pd.DataFrame(columns=schedule_columns))
    frame = features.copy()
    if "obs_id" not in frame and "event_id" in frame:
        frame["obs_id"] = frame.event_id
    if "date" not in frame and "signal_date" in frame:
        frame["date"] = frame.signal_date
    required = {"obs_id", "date", "ticker", "market", "series_kind", "state", "eligible", *_PREDICTIVE_NUMERIC_FEATURES}
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"Missing walk-forward features: {missing}")
    if frame.obs_id.isna().any():
        raise ValueError("Walk-forward obs_id must not be missing.")
    frame["obs_id"] = frame.obs_id.astype(str)
    if frame.obs_id.duplicated().any():
        raise ValueError("Walk-forward obs_id must be unique.")
    frame["event_id"] = frame.get("event_id", frame.obs_id).astype(str)
    frame["date"] = pd.to_datetime(frame.date)
    if frame.date.isna().any():
        raise ValueError("Walk-forward evaluation dates must not be missing.")
    frame["raw_top10"] = frame.get("raw_top10", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    if "momentum_3m_sign" not in frame:
        frame["momentum_3m_sign"] = np.select([frame.momentum_3m > 0, frame.momentum_3m < 0, frame.momentum_3m == 0],
                                             ["positive", "negative", "zero"], default="missing")
    frame["exclusion_reason"] = frame.get("exclusion_reason", pd.Series("", index=frame.index)).fillna("")
    finite = np.isfinite(frame[list(_PREDICTIVE_NUMERIC_FEATURES)].to_numpy(dtype=float)).all(axis=1)
    frame["_feature_valid"] = (frame.eligible.fillna(False).astype(bool) & finite & (frame.volatility >= 0) &
                                frame.state.isin(_PREDICTIVE_STATES) & frame.market.isin(("US", "JP")))
    frame = frame.sort_values(["date", "series_kind", "market", "ticker", "obs_id"], kind="stable").reset_index(drop=True)
    labels = targets.copy()
    if "obs_id" not in labels and "event_id" in labels:
        labels["obs_id"] = labels.event_id
    target_required = {"obs_id", "target", "target_value", "target_status", "label_end_date"}
    missing = sorted(target_required - set(labels))
    if missing:
        raise ValueError(f"Missing walk-forward target columns: {missing}")
    labels["obs_id"] = labels.obs_id.astype(str)
    if labels.duplicated(["obs_id", "target"]).any():
        raise ValueError("Each obs_id/target must occur once.")
    for column in ("label_end_date", "entry_date", "exit_date"):
        labels[column] = pd.to_datetime(labels[column]) if column in labels else pd.NaT
    joined = {}
    for target in _PREDICTIVE_TARGETS:
        selected = labels.loc[labels.target == target, ["obs_id", "target_value", "target_status", "label_end_date", "entry_date", "exit_date"]]
        dataset = frame.merge(selected, on="obs_id", how="left", validate="one_to_one", sort=False)
        dataset["target_status"] = dataset.target_status.fillna("computation_failed")
        dataset["target_value"] = pd.to_numeric(dataset.target_value, errors="coerce")
        joined[target] = dataset
    prediction_rows, audit_rows, schedule_rows = [], [], []
    dates = sorted(frame.date.unique())
    for date_number, evaluation_date in enumerate(dates, start=1):
        evaluation_date = pd.Timestamp(evaluation_date)
        if progress:
            print(f"Walk-forward {date_number}/{len(dates)}: {evaluation_date.date()} (4 targets; independent series)", flush=True)
        for series_kind in sorted(frame.loc[frame.date == evaluation_date, "series_kind"].unique()):
            for target in _PREDICTIVE_TARGETS:
                dataset = joined[target]
                test = dataset.loc[(dataset.date == evaluation_date) & (dataset.series_kind == series_kind)].copy()
                binary = target in _PREDICTIVE_BINARY_TARGETS
                at_risk = (dataset.momentum_3m > 0) if target == "negative_flip_21" else pd.Series(True, index=dataset.index)
                finite_label = np.isfinite(dataset.target_value.to_numpy(dtype=float))
                allowed_label = dataset.target_value.isin((0.0, 1.0)) if binary else pd.Series(True, index=dataset.index)
                prior_observed = ((dataset.date < evaluation_date) & (dataset.target_status == "observed") &
                                  (dataset.series_kind == series_kind))
                valid_end_date = dataset.label_end_date.notna() & (dataset.label_end_date > dataset.date)
                mature = prior_observed & valid_end_date & (dataset.label_end_date < evaluation_date)
                invalid_label_count = int((prior_observed & dataset._feature_valid & at_risk &
                                           (~finite_label | ~allowed_label | ~valid_end_date)).sum())
                train = dataset.loc[mature & dataset._feature_valid & at_risk & finite_label & allowed_label].copy()
                train = train.sort_values(["date", "market", "ticker", "obs_id"], kind="stable")
                if not train.empty and not ((train.date < evaluation_date).all() and (train.label_end_date < evaluation_date).all()):
                    raise AssertionError("Training information was not fully mature before evaluation.")
                train_count = len(train)
                date_count = int(train.date.nunique())
                market_dates = {market: int(train.loc[train.market == market, "date"].nunique()) for market in ("US", "JP")}
                positive = int((train.target_value == 1).sum()) if binary else 0
                negative = int((train.target_value == 0).sum()) if binary else 0
                global_sufficient = date_count >= 12 and ((positive >= 30 and negative >= 30) if binary else train_count >= 60)
                train_hash = _predictive_ids_hash(train.obs_id)
                test_risk = test.momentum_3m > 0 if target == "negative_flip_21" else pd.Series(True, index=test.index)
                score_mask = test._feature_valid & test_risk
                score = test.loc[score_mask].copy()
                eval_hash = _predictive_ids_hash(score.obs_id)
                forecastable = score.loc[score.market.map(market_dates) >= 12] if global_sufficient else score.iloc[:0]
                shared = {"target": target, "series_kind": series_kind, "prediction_date": evaluation_date,
                          "train_count": train_count, "positive_count": positive, "negative_count": negative,
                          "unique_training_dates": date_count, "us_training_dates": market_dates["US"], "jp_training_dates": market_dates["JP"],
                          "training_feature_max_date": train.date.max() if train_count else pd.NaT,
                          "max_label_end_date": train.label_end_date.max() if train_count else pd.NaT,
                          "train_ids_hash": train_hash, "eval_ids_hash": eval_hash,
                          "train_target_hash": _predictive_matrix_hash(train.target_value.to_numpy(dtype=float)),
                          "eval_count": len(score), "forecastable_eval_count": len(forecastable),
                          "invalid_observed_training_label_count": invalid_label_count}
                for market, market_test in test.groupby("market", sort=True):
                    mask = market_test._feature_valid & ((market_test.momentum_3m > 0) if target == "negative_flip_21" else True)
                    count = int(mask.sum())
                    schedule_rows.append({**{key: shared[key] for key in ("target", "series_kind", "prediction_date", "train_count", "unique_training_dates",
                                                                          "positive_count", "negative_count", "training_feature_max_date", "max_label_end_date", "train_ids_hash", "eval_ids_hash")},
                                          "market": market, "row_count": len(market_test), "feature_eligible_count": int(market_test._feature_valid.sum()),
                                          "not_at_risk_count": int((market_test._feature_valid & (market_test.momentum_3m <= 0)).sum()) if target == "negative_flip_21" else 0,
                                          "forecastable_count": count if global_sufficient and market_dates.get(market, 0) >= 12 else 0,
                                          "market_training_dates": market_dates.get(market, 0),
                                          "status": "training_available" if global_sufficient and market_dates.get(market, 0) >= 12 else "insufficient_training_data"})
                scaled_train = scaled_score = None
                scaler_mean, scaler_scale, scaling_error = [], [], None
                shared_train_hash = shared_score_hash = None
                if train_count:
                    try:
                        scaler = StandardScaler()
                        scaled_train = scaler.fit_transform(train[list(_PREDICTIVE_NUMERIC_FEATURES)].to_numpy(dtype=float))
                        scaled_score = scaler.transform(score[list(_PREDICTIVE_NUMERIC_FEATURES)].to_numpy(dtype=float)) if len(score) else np.empty((0, 5))
                        scaler_mean = scaler.mean_.astype(float).tolist()
                        scaler_scale = scaler.scale_.astype(float).tolist()
                        common_train = np.column_stack([scaled_train, _predictive_dummy_matrix(train)[:, :1]])
                        common_score = np.column_stack([scaled_score, _predictive_dummy_matrix(score)[:, :1]])
                        shared_train_hash = _predictive_matrix_hash(common_train)
                        shared_score_hash = _predictive_matrix_hash(common_score)
                    except Exception as exc:
                        scaling_error = f"{type(exc).__name__}: {exc}"
                for model_name in ("0", "A", "B"):
                    feature_names = [] if model_name == "0" else [*_PREDICTIVE_NUMERIC_FEATURES, "market_JP"]
                    if model_name == "B":
                        feature_names += list(_PREDICTIVE_STATE_FEATURES)
                    hyperparameters = ({"type": "market_training_target_mean"} if model_name == "0" else
                                       {"C": 1.0, "penalty": "l2", "class_weight": None, "solver": "lbfgs", "max_iter": 1000} if binary else {"alpha": 1.0})
                    algorithm = "market_training_target_mean" if model_name == "0" else "LogisticRegression" if binary else "Ridge"
                    estimator_parameters = {"alpha": 1.0} if not binary else {"C": 1.0, "class_weight": None, "solver": "lbfgs",
                                                                                          "max_iter": 1000, "random_state": int(seed)}
                    if binary:
                        penalty_parameter = inspect.signature(LogisticRegression).parameters.get("penalty")
                        # sklearn >=1.8 exposes pure L2 through l1_ratio=0;
                        # earlier supported versions use penalty='l2'. Both
                        # express the same preregistered regularization.
                        if penalty_parameter is not None and penalty_parameter.default == "deprecated":
                            estimator_parameters["l1_ratio"] = 0.0
                        else:
                            estimator_parameters["penalty"] = "l2"
                    model_status, error, predicted = "insufficient_training_data", "", {}
                    if global_sufficient and len(forecastable):
                        try:
                            if model_name == "0":
                                means = train.groupby("market", sort=True).target_value.mean().to_dict()
                                predicted = {obs_id: float(means[market]) for obs_id, market in zip(forecastable.obs_id, forecastable.market)}
                            else:
                                if scaling_error is not None:
                                    raise ValueError(f"Train-only scaler failed: {scaling_error}")
                                if scaled_train is None or scaled_score is None:
                                    raise ValueError("Train-only scaler unavailable.")
                                extra_train = _predictive_dummy_matrix(train)
                                extra_score = _predictive_dummy_matrix(score)
                                train_matrix = np.column_stack([scaled_train, extra_train[:, :1] if model_name == "A" else extra_train])
                                test_matrix = np.column_stack([scaled_score, extra_score[:, :1] if model_name == "A" else extra_score])
                                if (not np.array_equal(train_matrix[:, :6], common_train) or
                                        not np.array_equal(test_matrix[:, :6], common_score)):
                                    raise AssertionError("A/B common numeric/market inputs differ.")
                                estimator = LogisticRegression(**estimator_parameters) if binary else Ridge(**estimator_parameters)
                                with warnings.catch_warnings(record=True) as recorded:
                                    warnings.simplefilter("always", ConvergenceWarning)
                                    estimator.fit(train_matrix, train.target_value.to_numpy(dtype=float))
                                if any(issubclass(warning.category, ConvergenceWarning) for warning in recorded):
                                    raise RuntimeError("ConvergenceWarning: fixed model did not converge.")
                                values = estimator.predict_proba(test_matrix)[:, 1] if binary else estimator.predict(test_matrix)
                                if not np.isfinite(values).all() or (binary and ((values < 0) | (values > 1)).any()):
                                    raise ValueError("Nonfinite or invalid probability prediction.")
                                predicted = {obs_id: float(value) for obs_id, value, market in zip(score.obs_id, values, score.market)
                                             if market_dates.get(market, 0) >= 12}
                            if not all(np.isfinite(value) for value in predicted.values()):
                                raise ValueError("Nonfinite baseline prediction.")
                            model_status = "fitted"
                        except Exception as exc:
                            model_status = "model_fit_failed"
                            error = f"{type(exc).__name__}: {exc}"
                            predicted = {}
                    audit_rows.append({**shared, "model": model_name, "status": model_status, "error": error,
                                       "shared_train_matrix_hash": shared_train_hash, "shared_eval_matrix_hash": shared_score_hash,
                                       "scaler_mean_json": json.dumps(scaler_mean), "scaler_scale_json": json.dumps(scaler_scale),
                                       "numeric_feature_names_json": json.dumps(list(_PREDICTIVE_NUMERIC_FEATURES)),
                                       "feature_names_json": json.dumps(feature_names), "dummy_baseline": "HH/HL",
                                       "dummy_scaling": "unscaled binary market and state indicators", "algorithm": algorithm,
                                       "hyperparameters_json": json.dumps(hyperparameters, sort_keys=True),
                                       "estimator_parameters_json": json.dumps(estimator_parameters if model_name != "0" else {}, sort_keys=True), "seed": int(seed),
                                       "feature_input_parity": "passed" if scaling_error is None else "scaler_failed"})
                    for row in test.to_dict("records"):
                        if not row["_feature_valid"]:
                            row_status = "ineligible_feature_row"
                        elif target == "negative_flip_21" and row["momentum_3m"] <= 0:
                            row_status = "not_at_risk_nonpositive"
                        elif not global_sufficient or market_dates.get(row["market"], 0) < 12:
                            row_status = "insufficient_training_data"
                        elif model_status == "model_fit_failed":
                            row_status = "model_fit_failed"
                        elif row["obs_id"] in predicted:
                            row_status = "predicted"
                        else:
                            row_status = "computation_failed"
                        prediction_rows.append({key: row.get(key, np.nan) for key in ("obs_id", "event_id", "date", "ticker", "market", "series_kind", "state",
                                                                                    "raw_top10", "momentum_3m_sign", "eligible", "exclusion_reason", "target_status",
                                                                                    "target_value", "label_end_date", "entry_date", "exit_date", *_PREDICTIVE_NUMERIC_FEATURES)} |
                                               {"prediction_date": evaluation_date, "target": target, "model": model_name, "status": row_status,
                                                "prediction_status": row_status, "prediction": predicted.get(row["obs_id"], np.nan),
                                                "error": (error if row_status == "model_fit_failed" else
                                                          str(row.get("exclusion_reason") or "Required finite numeric features, nonnegative volatility, classified state or supported market missing.")
                                                          if row_status == "ineligible_feature_row" else ""),
                                                "train_count": train_count, "unique_training_dates": date_count,
                                                "market_training_dates": market_dates.get(row["market"], 0),
                                                "training_feature_max_date": shared["training_feature_max_date"], "max_label_end_date": shared["max_label_end_date"],
                                                "train_ids_hash": train_hash, "eval_ids_hash": eval_hash})
    return (pd.DataFrame(prediction_rows, columns=prediction_columns), pd.DataFrame(audit_rows, columns=audit_columns),
            pd.DataFrame(schedule_rows, columns=schedule_columns))


_EVAL_BINARY_TARGETS = ("deep_drawdown_63", "negative_flip_21")
_EVAL_NUMERIC_TARGETS = ("rank_change_21", "relative_return_63")
_EVAL_MODELS = ("0", "A", "B")
_EVAL_INTEGRITY_KEYS = ("input_unchanged", "protected_unchanged", "training_causal",
                        "input_parity", "replay_verified", "predictions_saved")
_EVAL_DIMENSIONS = ("series_kind", "cohort", "market", "momentum_3m_sign", "target")


def _evaluation_boolean(values):
    """Preserve false flags even when a saved CSV supplies string booleans."""
    def convert(value):
        if pd.isna(value):
            return False
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if value in (0, 1):
            return bool(value)
        if isinstance(value, str) and value.lower().strip() in ("true", "false"):
            return value.lower().strip() == "true"
        raise ValueError("raw_top10 must contain boolean flags, not arbitrary labels")
    return values.map(convert).astype(bool)


def _evaluation_paired(predictions):
    """Retain every objective/observation; scoring always uses the shared trio."""
    required = {"obs_id", "target", "model", "date", "ticker", "market",
                "target_status", "target_value", "prediction"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction evaluation missing fields: {missing}")
    frame = predictions.copy()
    if "prediction_status" not in frame:
        if "status" not in frame:
            raise ValueError("Prediction evaluation requires prediction_status or status")
        frame["prediction_status"] = frame.status
    frame["model"] = frame.model.astype(str)
    if not frame.model.isin(_EVAL_MODELS).all():
        raise ValueError("Only models 0, A and B may enter the paired evaluation")
    if not frame.target.isin((*_EVAL_BINARY_TARGETS, *_EVAL_NUMERIC_TARGETS)).all():
        raise ValueError("Unknown prediction target; never silently omit an objective")
    if not frame.market.isin(("US", "JP")).all() or frame.obs_id.isna().any():
        raise ValueError("Forecast rows require known markets and nonmissing observation identifiers")
    if frame.duplicated(["obs_id", "target", "model"]).any():
        raise ValueError("Duplicate observation/target/model forecast rows")
    frame["date"] = pd.to_datetime(frame.date)
    if frame.date.isna().any():
        raise ValueError("Evaluation dates cannot be missing")
    if "series_kind" not in frame:
        frame["series_kind"] = "price"
    kinds = frame.series_kind.dropna().unique()
    if len(frame) and (len(kinds) != 1 or kinds[0] not in ("price", "relative")):
        raise ValueError("Evaluate price and relative structure in separate runs; never pool them")
    if "raw_top10" not in frame:
        frame["raw_top10"] = False
    frame["raw_top10"] = _evaluation_boolean(frame.raw_top10)
    for column, default in (("state", "unclassified"), ("momentum_3m_sign", "missing")):
        if column not in frame:
            frame[column] = default
    keys = ["obs_id", "target"]
    shared = ["date", "ticker", "market", "series_kind", "state", "raw_top10",
              "momentum_3m_sign", "target_status", "target_value"]
    # Model identity can change predictions, never the actual objective/cohort.
    for column in shared:
        if frame.groupby(keys, dropna=False)[column].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Models disagree on shared evaluation field: {column}")
    optional = [c for c in ("top10_retained", "top10_retained_status", "retention_status",
                            "top10_retention_status", "rank_change_21_status") if c in frame]
    for column in optional:
        if frame.groupby(keys, dropna=False)[column].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Models disagree on retention field: {column}")
    base = frame.drop_duplicates(keys).set_index(keys)[shared + optional]
    prediction = frame.pivot(index=keys, columns="model", values="prediction")
    statuses = frame.pivot(index=keys, columns="model", values="prediction_status")
    for model in _EVAL_MODELS:
        base[f"prediction_{model}"] = pd.to_numeric(prediction[model], errors="coerce") if model in prediction else np.nan
        base[f"prediction_status_{model}"] = statuses[model] if model in statuses else "missing_prediction"
    paired = base.reset_index().sort_values(["date", "market", "ticker", "target", "obs_id"], kind="stable").reset_index(drop=True)
    paired["target_value"] = pd.to_numeric(paired.target_value, errors="coerce")
    observed = paired.target_status.eq("observed")
    valid_target = np.isfinite(paired.target_value.to_numpy(dtype=float))
    binary = paired.target.isin(_EVAL_BINARY_TARGETS)
    valid_target &= (~binary | paired.target_value.isin((0, 1))).to_numpy()
    predictions_available = np.ones(len(paired), dtype=bool)
    valid_prediction = np.ones(len(paired), dtype=bool)
    for model in _EVAL_MODELS:
        values = paired[f"prediction_{model}"].to_numpy(dtype=float)
        predictions_available &= paired[f"prediction_status_{model}"].eq("predicted").to_numpy() & np.isfinite(values)
        valid_prediction &= (~binary | paired[f"prediction_{model}"].between(0, 1)).to_numpy()
    paired["pair_status"] = np.select(
        [~observed, ~valid_target, ~predictions_available, ~valid_prediction],
        ["target_unobserved", "invalid_target_value", "prediction_unavailable", "invalid_probability"],
        default="eligible")
    paired["loss_type"] = np.where(binary, "brier", "squared_error")
    eligible = paired.pair_status.eq("eligible")
    for model in _EVAL_MODELS:
        paired[f"loss_{model}"] = ((paired[f"prediction_{model}"] - paired.target_value) ** 2).where(eligible)
    paired["loss_difference_A_minus_B"] = paired.loss_A - paired.loss_B
    paired["relative_improvement_observation"] = (paired.loss_difference_A_minus_B / paired.loss_A).where(paired.loss_A > 0)
    return paired


def _evaluation_counts(sample):
    valid = sample.loc[sample.pair_status == "eligible"]
    primary_binary = valid.target.isin(_EVAL_BINARY_TARGETS)
    statuses = sample.target_status.fillna("").astype(str)
    repetitions = valid.groupby(["market", "ticker"]).size()
    return {"total_count": len(sample), "valid_count": len(valid),
            "target_unobserved_count": int(sample.pair_status.eq("target_unobserved").sum()),
            "right_censored_count": int(statuses.eq("right_censored").sum()),
            "price_missing_count": int(statuses.isin(("price_missing", "missing_price", "missing_signal_observation")).sum()),
            "known_anomaly_count": int(statuses.isin(("excluded_influence", "known_anomaly", "known_anomaly_influence")).sum()),
            "not_at_risk_count": int(statuses.isin(("not_at_risk","not_at_risk_nonpositive")).sum()),
            "computation_failed_count": int(statuses.eq("computation_failed").sum()),
            "prediction_unavailable_count": int(sample.pair_status.eq("prediction_unavailable").sum()),
            "invalid_target_count": int(sample.pair_status.eq("invalid_target_value").sum()),
            "invalid_probability_count": int(sample.pair_status.eq("invalid_probability").sum()),
            "positive_count": int((primary_binary & valid.target_value.eq(1)).sum()),
            "negative_count": int((primary_binary & valid.target_value.eq(0)).sum()),
            "unique_ticker_count": int(valid.ticker.nunique()),
            "repeated_ticker_count": int(repetitions.gt(1).sum()),
            "maximum_observations_per_ticker": int(repetitions.max()) if len(repetitions) else 0,
            "unique_date_count": int(valid.date.nunique()),
            "unique_month_count": int(valid.date.dt.to_period("M").nunique())}


def _evaluation_cohorts(paired, include_states=False):
    kind = paired.series_kind.iloc[0] if len(paired) else "price"
    for cohort in ("all", "raw_top10"):
        cohort_frame = paired if cohort == "all" else paired.loc[paired.raw_top10]
        for market in ("ALL", "US", "JP"):
            market_frame = cohort_frame if market == "ALL" else cohort_frame.loc[cohort_frame.market == market]
            for sign in ("all", "positive", "negative", "zero"):
                sign_frame = market_frame if sign == "all" else market_frame.loc[market_frame.momentum_3m_sign == sign]
                states = ("ALL", *sorted(paired.state.dropna().unique())) if include_states else ("ALL",)
                for state in states:
                    sample = sign_frame if state == "ALL" else sign_frame.loc[sign_frame.state == state]
                    for target in (*_EVAL_BINARY_TARGETS, *_EVAL_NUMERIC_TARGETS):
                        yield {"series_kind": kind, "cohort": cohort, "market": market,
                               "momentum_3m_sign": sign, "state": state, "target": target}, sample.loc[sample.target == target]


def _evaluation_risk(y, p):
    """Fixed probability scores; discrimination is undefined for one class."""
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, auc
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    names = ("brier_score", "log_loss", "roc_auc", "pr_auc", "average_precision", "event_rate",
             "mean_prediction", "false_negative_rate", "false_positive_rate", "false_negative_count",
             "true_positive_count", "false_positive_count", "true_negative_count", "expected_calibration_error")
    if not len(y):
        return {name: np.nan for name in names}
    clipped = np.clip(p, 1e-15, 1 - 1e-15)
    positive, negative = y == 1, y == 0
    predicted_positive = p >= .5
    false_negative = int((positive & ~predicted_positive).sum())
    false_positive = int((negative & predicted_positive).sum())
    bins = np.minimum((p * 10).astype(int), 9)
    ece = 0.0
    for number in range(10):
        keep = bins == number
        if keep.any():
            ece += keep.mean() * abs(float(p[keep].mean() - y[keep].mean()))
    discriminates = positive.any() and negative.any()
    roc = float(roc_auc_score(y, p)) if discriminates else np.nan
    precision, recall, _ = precision_recall_curve(y, p) if discriminates else (None, None, None)
    return {"brier_score": float(np.mean(np.square(p-y))),
            "log_loss": float(-np.mean(y*np.log(clipped) + (1-y)*np.log1p(-clipped))),
            "roc_auc": roc, "pr_auc": float(auc(recall, precision)) if discriminates else np.nan,
            "average_precision": float(average_precision_score(y, p)) if discriminates else np.nan,
            "event_rate": float(y.mean()), "mean_prediction": float(p.mean()),
            "false_negative_rate": false_negative / int(positive.sum()) if positive.any() else np.nan,
            "false_positive_rate": false_positive / int(negative.sum()) if negative.any() else np.nan,
            "false_negative_count": false_negative, "true_positive_count": int((positive & predicted_positive).sum()),
            "false_positive_count": false_positive, "true_negative_count": int((negative & ~predicted_positive).sum()),
            "expected_calibration_error": float(ece)}


def _evaluation_rank_correlation(y, p):
    if len(y) < 2 or pd.Series(y).nunique() < 2 or pd.Series(p).nunique() < 2:
        return np.nan
    return float(pd.Series(y).rank(method="average").corr(pd.Series(p).rank(method="average")))


def _evaluation_numeric(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if not len(y):
        return {name: np.nan for name in ("mae", "rmse", "spearman_correlation", "direction_accuracy", "direction_correlation")}
    signs_y, signs_p = np.sign(y), np.sign(p)
    direction_correlation = float(pd.Series(signs_y).corr(pd.Series(signs_p))) if np.unique(signs_y).size > 1 and np.unique(signs_p).size > 1 else np.nan
    return {"mae": float(np.mean(np.abs(p-y))), "rmse": float(np.sqrt(np.mean(np.square(p-y)))),
            "spearman_correlation": _evaluation_rank_correlation(y, p),
            "direction_accuracy": float(np.mean(signs_y == signs_p)), "direction_correlation": direction_correlation}


def _evaluation_calibration(sample, dimensions, model):
    valid = sample.loc[sample.pair_status == "eligible"]
    p = valid[f"prediction_{model}"].to_numpy(dtype=float)
    y = valid.target_value.to_numpy(dtype=float)
    assignments = np.minimum((p * 10).astype(int), 9)
    rows = []
    for number in range(10):
        keep = assignments == number
        n = int(keep.sum())
        rows.append({**dimensions, "model": model, "bin": number, "bin_left": number/10,
                     "bin_right": (number+1)/10, "right_inclusive": number == 9,
                     "count": n, "valid_count": len(valid),
                     "mean_prediction": float(p[keep].mean()) if n else np.nan,
                     "event_rate": float(y[keep].mean()) if n else np.nan,
                     "calibration_gap": float(p[keep].mean()-y[keep].mean()) if n else np.nan})
    return rows


def _evaluation_cluster_interval(sample, method, samples, rng):
    """Whole-cluster resampling, event-weighted Brier difference and ratio."""
    work = sample[["date", "market", "ticker", "loss_A", "loss_B"]].copy()
    if method == "market_month":
        work["cluster"] = work.market.astype(str) + ":" + work.date.dt.to_period("M").astype(str)
    elif method == "ticker":
        work["cluster"] = work.market.astype(str) + ":" + work.ticker.astype(str)
    else:
        raise ValueError("Unknown uncertainty cluster method")
    grouped = work.groupby("cluster", sort=True).agg(loss_A=("loss_A", "sum"), loss_B=("loss_B", "sum"), count=("loss_A", "size"))
    cluster_count = len(grouped)
    result = {"cluster_count": cluster_count, "ci_loss_low": np.nan, "ci_loss_high": np.nan,
              "ci_relative_low": np.nan, "ci_relative_high": np.nan}
    if cluster_count < 2 or samples < 2:
        return result
    sums_a, sums_b, counts = (grouped[c].to_numpy(dtype=float) for c in ("loss_A", "loss_B", "count"))
    draw = rng.integers(0, cluster_count, size=(samples, cluster_count))
    a, b, n = sums_a[draw].sum(axis=1), sums_b[draw].sum(axis=1), counts[draw].sum(axis=1)
    loss_difference = (a-b)/n
    relative = np.divide(a-b, a, out=np.full_like(a, np.nan), where=a>0)
    result["ci_loss_low"], result["ci_loss_high"] = map(float, np.quantile(loss_difference, [.025,.975]))
    finite = relative[np.isfinite(relative)]
    if len(finite) == samples:
        result["ci_relative_low"], result["ci_relative_high"] = map(float, np.quantile(finite, [.025,.975]))
    return result


def _evaluation_uncertainty(paired, samples, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for dimensions, sample in _evaluation_cohorts(paired):
        if dimensions["target"] != "deep_drawdown_63":
            continue
        valid = sample.loc[sample.pair_status == "eligible"]
        counts = _evaluation_counts(sample)
        loss_a, loss_b = valid.loss_A.mean(), valid.loss_B.mean()
        improvement = float((loss_a-loss_b)/loss_a) if len(valid) and loss_a > 0 else np.nan
        for method in ("market_month", "ticker"):
            interval = _evaluation_cluster_interval(valid, method, samples, rng)
            warning = "Exploratory whole-cluster resampling; overlapping63-bar outcomes remain dependent; not an iid significance test or combined two-way interval."
            if counts["unique_month_count"] < 8:
                warning += " Fewer than8 distinct evaluation months."
            if interval["cluster_count"] < 2 or samples < 2:
                warning += " Interval unavailable: fewer than2 clusters or bootstrap replicates."
            rows.append({**dimensions, **counts, "count": len(valid), "cluster_method": method,
                         "loss_difference": float(loss_a-loss_b) if len(valid) else np.nan,
                         "relative_improvement": improvement, **interval,
                         "bootstrap_samples": samples, "seed": seed, "warning": warning})
    return pd.DataFrame(rows)


def _evaluation_stability(paired):
    """One pooled primary top10 date cutoff, shared by market assessments."""
    primary = paired.loc[(paired.target == "deep_drawdown_63") & paired.raw_top10 & paired.pair_status.eq("eligible")]
    dates = sorted(primary.date.unique())
    cutoff = pd.Timestamp(dates[(len(dates)+1)//2-1]) if len(dates) >= 2 else pd.NaT
    rows = []
    for cohort in ("all", "raw_top10"):
        frame = paired.loc[paired.target.eq("deep_drawdown_63") & paired.pair_status.eq("eligible")]
        if cohort == "raw_top10":
            frame = frame.loc[frame.raw_top10]
        for market in ("ALL", "US", "JP"):
            group = frame if market == "ALL" else frame.loc[frame.market == market]
            for half in ("first", "second"):
                sample = group.iloc[:0] if pd.isna(cutoff) else group.loc[group.date.le(cutoff) if half == "first" else group.date.gt(cutoff)]
                a, b = sample.loss_A.mean(), sample.loss_B.mean()
                rows.append({"series_kind": paired.series_kind.iloc[0] if len(paired) else "price", "target": "deep_drawdown_63",
                             "cohort": cohort, "market": market, "half": half, "cutoff_date": cutoff,
                             "valid_count": len(sample), "positive_count": int(sample.target_value.eq(1).sum()),
                             "unique_date_count": int(sample.date.nunique()), "unique_month_count": int(sample.date.dt.to_period("M").nunique()),
                             "brier_A": a, "brier_B": b,
                             "relative_improvement": float((a-b)/a) if len(sample) and a>0 else np.nan,
                             "definition": "global paired primary top10 unique dates; first ceil(n/2), remainder second; shared cutoff across markets"})
    return pd.DataFrame(rows)


def _evaluation_retention(paired):
    columns = ["series_kind", "market", "state", "initial_top10_count", "observed_count", "unobserved_count", "retained_count", "retention_rate", "definition"]
    if "top10_retained" not in paired:
        return pd.DataFrame(columns=columns)
    # This is an observed auxiliary target, not a model score or Gate condition.
    initial = paired.loc[paired.raw_top10].copy()
    initial["_retention_target_priority"] = np.where(initial.target.eq("rank_change_21"), 0, 1)
    initial = initial.sort_values(["obs_id", "_retention_target_priority"], kind="stable").drop_duplicates(["obs_id"]).copy()
    status_name = next((c for c in ("top10_retained_status", "top10_retention_status", "retention_status", "rank_change_21_status") if c in initial), None)
    observed = initial.top10_retained.notna()
    if status_name:
        observed &= initial[status_name].eq("observed")
    else:
        # The rank-change target supplies the next-landmark observation status.
        rank_status = paired.loc[paired.target == "rank_change_21"].set_index("obs_id").target_status
        observed &= initial.obs_id.map(rank_status).eq("observed")
    initial["_observed_retention"] = observed
    rows = []
    for market in ("ALL", "US", "JP"):
        group = initial if market == "ALL" else initial.loc[initial.market == market]
        for state in ("ALL", *sorted(initial.state.dropna().unique())):
            sample = group if state == "ALL" else group.loc[group.state == state]
            valid = sample.loc[sample._observed_retention]
            values = _evaluation_boolean(valid.top10_retained)
            rows.append({"series_kind": paired.series_kind.iloc[0] if len(paired) else "price", "market": market,
                         "state": state, "initial_top10_count": len(sample), "observed_count": len(valid),
                         "unobserved_count": len(sample)-len(valid), "retained_count": int(values.sum()),
                         "retention_rate": float(values.mean()) if len(valid) else np.nan,
                         "definition": "initial rawtop10, observed next-market-landmark membership; auxiliary unmodeled label, never Gate evidence"})
    return pd.DataFrame(rows, columns=columns)


def evaluate_predictions(predictions, *, bootstrap_samples=500, seed=20261111, integrity=None):
    """Compare 0/A/B only on identical observed objectives and forecast rows.

    Prediction/label failures remain in ``paired`` and availability counts. A
    good secondary objective cannot override the predeclared primary Gate.
    """
    if isinstance(bootstrap_samples, bool) or int(bootstrap_samples) != bootstrap_samples or bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be a nonnegative integer")
    paired = _evaluation_paired(predictions)
    risk_rows, numeric_rows, calibration, subgroup_rows = [], [], [], []
    for dimensions, sample in _evaluation_cohorts(paired):
        counts = _evaluation_counts(sample)
        valid = sample.loc[sample.pair_status == "eligible"]
        for model in _EVAL_MODELS:
            base = {**dimensions, **counts, "model": model, "evaluation_sample": "identical observed rows with finite predicted models0/A/B"}
            y, p = valid.target_value.to_numpy(dtype=float), valid[f"prediction_{model}"].to_numpy(dtype=float)
            if dimensions["target"] in _EVAL_BINARY_TARGETS:
                risk_rows.append({**base, **_evaluation_risk(y,p), "diagnostic_threshold": .5,
                                  "pr_auc_definition": "trapezoidal precision-recall curve area; average_precision is separately reported"})
                calibration.extend(_evaluation_calibration(sample, dimensions, model))
            else:
                numeric_rows.append({**base, **_evaluation_numeric(y,p)})
    for dimensions, sample in _evaluation_cohorts(paired, include_states=True):
        counts = _evaluation_counts(sample)
        row = {**dimensions, **counts}
        valid = sample.loc[sample.pair_status == "eligible"]
        for model in _EVAL_MODELS:
            values = valid[f"loss_{model}"]
            row[f"mean_squared_loss_{model}"] = values.mean()
        a, b = valid.loss_A.mean(), valid.loss_B.mean()
        row["loss_difference_A_minus_B"] = float(a-b) if len(valid) else np.nan
        row["relative_improvement"] = float((a-b)/a) if len(valid) and a>0 else np.nan
        row["interpretation"] = "Brier improvement" if dimensions["target"] in _EVAL_BINARY_TARGETS else "squared-error improvement; separate from primary classification Gate"
        subgroup_rows.append(row)
    uncertainty = _evaluation_uncertainty(paired, int(bootstrap_samples), int(seed))
    gate = gate_assessment(paired, uncertainty, integrity or {})
    return {"risk_metrics": pd.DataFrame(risk_rows), "momentum_metrics": pd.DataFrame(numeric_rows),
            "subgroup_metrics": pd.DataFrame(subgroup_rows), "paired": paired,
            "uncertainty": uncertainty, "calibration": pd.DataFrame(calibration),
            "stability_metrics": _evaluation_stability(paired),
            "retention_metrics": _evaluation_retention(paired), "gate": gate}


"""Predeclared Gate I assessment for paired v0.3 research predictions."""

import numpy as np
import pandas as pd


_GATE_REQUIRED_INTEGRITY = (
    "input_unchanged", "protected_unchanged", "training_causal",
    "input_parity", "replay_verified", "predictions_saved",
)


def _gate_counts(frame):
    dates = pd.to_datetime(frame["date"], errors="coerce")
    target = pd.to_numeric(frame["target_value"], errors="coerce")
    return {
        "valid_prediction_count": int(len(frame)),
        "positive_count": int(target.eq(1).sum()),
        "negative_count": int(target.eq(0).sum()),
        "unique_month_count": int(dates.dt.strftime("%Y-%m").nunique()),
        "unique_date_count": int(dates.nunique()),
        "unique_ticker_count": int(frame["ticker"].nunique()),
    }


def _gate_half(frame):
    result = _gate_counts(frame)
    a = float(frame["loss_A"].mean()) if len(frame) else np.nan
    b = float(frame["loss_B"].mean()) if len(frame) else np.nan
    result.update({
        "brier_A": a, "brier_B": b,
        "loss_difference_A_minus_B": a - b,
        "relative_improvement": (a - b) / a if np.isfinite(a) and a > 0 else np.nan,
        "improvement_positive": bool(np.isfinite(a) and np.isfinite(b) and a > b),
    })
    return result


def _gate_monthly_interval(uncertainty, market):
    fields = ("ci_loss_low", "ci_loss_high", "ci_relative_low", "ci_relative_high")
    result = {name: np.nan for name in fields}
    result.update({"available": False, "cluster_count": 0, "bootstrap_samples": 0})
    if uncertainty.empty:
        return result
    keys = {"target", "cohort", "market", "series_kind", "momentum_3m_sign", "cluster_method"}
    if not keys.issubset(uncertainty.columns):
        raise ValueError("Gate uncertainty lacks primary-cohort identification columns")
    selected = uncertainty.loc[
        uncertainty["target"].eq("deep_drawdown_63")
        & uncertainty["cohort"].eq("raw_top10")
        & uncertainty["market"].eq(market)
        & uncertainty["series_kind"].eq("price")
        & uncertainty["momentum_3m_sign"].eq("all")
        & uncertainty["cluster_method"].eq("market_month")
    ]
    if len(selected) > 1:
        raise ValueError("Gate has duplicate primary market-month uncertainty rows")
    if selected.empty:
        return result
    row = selected.iloc[0]
    for name in fields:
        value = pd.to_numeric(pd.Series([row.get(name, np.nan)]), errors="coerce").iloc[0]
        result[name] = float(value) if pd.notna(value) else np.nan
    for name in ("cluster_count", "bootstrap_samples"):
        value = pd.to_numeric(pd.Series([row.get(name, 0)]), errors="coerce").iloc[0]
        result[name] = int(value) if pd.notna(value) and np.isfinite(value) else 0
    result["available"] = bool(
        result["cluster_count"] >= 2 and result["bootstrap_samples"] >= 2
        and all(np.isfinite(result[name]) for name in fields)
        and result["ci_loss_low"] <= result["ci_loss_high"]
        and result["ci_relative_low"] <= result["ci_relative_high"]
    )
    return result


def _gate_market_assessment(frame, uncertainty, market, cutoff, integrity_checks):
    counts = _gate_counts(frame)
    losses = frame[["loss_0", "loss_A", "loss_B"]].to_numpy(dtype=float)
    numeric_ok = bool(
        len(frame) and np.isfinite(losses).all() and (losses >= 0).all()
        and pd.to_datetime(frame.date, errors="coerce").notna().all()
        and pd.to_numeric(frame.target_value, errors="coerce").isin([0, 1]).all()
    )
    means = {model: float(frame["loss_" + model].mean()) if len(frame) else np.nan
             for model in ("0", "A", "B")}
    a, b, zero = means["A"], means["B"], means["0"]
    relative = (a - b) / a if np.isfinite(a) and a > 0 else np.nan
    dates = pd.to_datetime(frame.date, errors="coerce")
    first = frame.loc[dates.le(cutoff)] if cutoff is not None else frame.iloc[:0]
    second = frame.loc[dates.gt(cutoff)] if cutoff is not None else frame.iloc[:0]
    halves = {"first": _gate_half(first), "second": _gate_half(second)}
    interval = _gate_monthly_interval(uncertainty, market)
    criteria = {
        "nonempty_valid_paired_predictions": bool(len(frame)),
        "finite_binary_paired_losses": numeric_ok,
        "at_least_30_positive_events": counts["positive_count"] >= 30,
        "at_least_8_unique_evaluation_months": counts["unique_month_count"] >= 8,
        "integrity_verified": all(integrity_checks.values()),
        "primary_market_month_interval_available": interval["available"],
        "relative_improvement_defined": bool(np.isfinite(relative)),
        "relative_improvement_at_least_3pct": bool(np.isfinite(relative) and relative >= .03-1e-12),
        "model_B_not_worse_than_model_0": bool(np.isfinite(b) and np.isfinite(zero) and b <= zero+1e-12),
        "positive_improvement_in_both_time_halves": bool(
            halves["first"]["improvement_positive"] and halves["second"]["improvement_positive"]),
        "monthly_relative_interval_strictly_above_zero": bool(
            interval["available"] and interval["ci_relative_low"] > 0),
    }
    requirements = (
        "nonempty_valid_paired_predictions", "finite_binary_paired_losses",
        "at_least_30_positive_events", "at_least_8_unique_evaluation_months",
        "integrity_verified", "primary_market_month_interval_available",
        "relative_improvement_defined",
    )
    unavailable = [name for name in requirements if not criteria[name]]
    reasons = []
    if unavailable:
        status, grade = "evaluation_unavailable", None
        reasons = ["Required evaluation condition failed: " + name for name in unavailable]
    elif not criteria["relative_improvement_at_least_3pct"]:
        status, grade = "assessed", "C"
        reasons = ["The predeclared 3% relative Brier improvement was not reached."]
    else:
        support = (
            "model_B_not_worse_than_model_0", "positive_improvement_in_both_time_halves",
            "monthly_relative_interval_strictly_above_zero",
        )
        failed = [name for name in support if not criteria[name]]
        status, grade = "assessed", "B" if failed else "A"
        reasons = (["Candidate condition failed: " + name for name in failed] if failed
                   else ["All predeclared pooled/market-level candidate conditions passed."])
    return {
        "status": status, "grade": grade, "market": market, "counts": counts,
        "brier_0": zero, "brier_A": a, "brier_B": b,
        "loss_difference_A_minus_B": a - b, "relative_improvement": relative,
        "time_half_cutoff": cutoff.date().isoformat() if cutoff is not None else None,
        "time_halves": halves, "market_month_uncertainty": interval,
        "criteria": criteria, "reasons": reasons,
    }


def gate_assessment(paired, uncertainty, integrity=None):
    """Assess only primary price/top10/63-bar risk; never infer deployment permission.

    Market-month resampling is the predeclared uncertainty criterion. Ticker
    intervals remain a separate sensitivity analysis and are never combined.
    Lack of required integrity, labels, months or an interval yields an explicit
    unavailable result, never an A/B/C assertion based on secondary targets.
    """
    columns = ["target", "series_kind", "raw_top10", "pair_status", "date", "ticker",
               "market", "target_value", "loss_0", "loss_A", "loss_B"]
    if paired.empty:
        primary = pd.DataFrame(columns=columns)
    else:
        missing = sorted(set(columns) - set(paired.columns))
        if missing:
            raise ValueError("Gate paired data lacks columns: " + ", ".join(missing))
        flags = paired["raw_top10"].fillna(False)
        if not flags.isin([True, False, 0, 1]).all():
            raise ValueError("raw_top10 must be boolean rather than truthy text")
        primary = paired.loc[
            paired.target.eq("deep_drawdown_63") & paired.series_kind.eq("price")
            & flags.astype(bool) & paired.pair_status.eq("eligible")
        ].copy()
    primary["date"] = pd.to_datetime(primary["date"], errors="coerce")
    if len(primary) and not primary.market.isin(["US", "JP"]).all():
        raise ValueError("Gate primary predictions must identify US or JP markets")
    dates = pd.DatetimeIndex(primary.date.dropna().unique()).sort_values()
    # The same date cannot straddle halves; use the pooled primary cutoff also
    # for each individual market so the stability periods remain comparable.
    cutoff = dates[(len(dates) + 1) // 2 - 1] if len(dates) > 1 else None
    supplied = integrity or {}
    checks = {name: isinstance(supplied.get(name), (bool, np.bool_)) and bool(supplied[name])
              for name in _GATE_REQUIRED_INTEGRITY}
    pooled = _gate_market_assessment(primary, uncertainty, "ALL", cutoff, checks)
    markets = {
        market: _gate_market_assessment(primary.loc[primary.market.eq(market)], uncertainty,
                                        market, cutoff, checks)
        for market in ("US", "JP")
    }
    status, grade = pooled["status"], pooled["grade"]
    reasons = list(pooled["reasons"])
    qualified = [market for market, result in markets.items() if result["grade"] == "A"]
    scope = "evaluation_unavailable" if status == "evaluation_unavailable" else "not_qualified"
    if grade == "A":
        if not qualified:
            grade, scope = "B", "market_support_unconfirmed"
            reasons.append("Pooled conditions passed, but neither individual market qualified as A.")
        elif len(qualified) == 1:
            scope = "market_limited"
            reasons.append("Candidate support is limited to the qualifying market; the other market is not an A result.")
        else:
            scope = "US_and_JP"
    elif qualified:
        reasons.append("A market-only A result does not promote the non-A pooled primary decision.")
    return {
        "status": status, "grade": grade, "scope": scope,
        "applicable_markets": qualified if grade == "A" else [],
        "primary_target": "deep_drawdown_63", "primary_cohort": "raw_top10",
        "primary_series_kind": "price", "production_permission": False,
        "reasons": reasons, "integrity_checks": checks,
        "required_integrity_fields": list(_GATE_REQUIRED_INTEGRITY),
        "criteria": {
            "minimum_relative_brier_improvement": .03, "minimum_positive_events": 30,
            "inclusive_comparison_atol":1e-12,
            "minimum_unique_evaluation_months": 8, "minimum_bootstrap_clusters": 2,
            "primary_uncertainty_method": "market_month",
            "interval_condition": "relative improvement lower endpoint strictly greater than zero",
            "time_half_definition": "unique pooled primary evaluation dates split chronologically; first ceil(n/2) dates, same cutoff for markets",
            "time_half_cutoff": cutoff.date().isoformat() if cutoff is not None else None,
            "simple_baseline_condition": "Brier_B <= Brier_0 on the same eligible paired observations",
            "market_support_condition": "pooled A also requires at least one individual-market A",
            "secondary_targets_can_override_primary": False,
        },
        "pooled_assessment": pooled, "market_assessments": markets,
        "limitations": [
            "This is a retrospective research gate, not a calibrated significance test or investment authorization.",
            "Market-month and ticker cluster intervals are separate; no combined two-way interval is asserted.",
            "Test execution must be documented separately; this helper does not manufacture a tests-passed assertion.",
        ],
    }


def _forecast_audit(predictions,audit):
    scored=predictions.loc[predictions.prediction_status.eq("predicted")]
    causal=bool((pd.to_datetime(scored.max_label_end_date)<pd.to_datetime(scored.date)).all() and
                (pd.to_datetime(scored.training_feature_max_date)<pd.to_datetime(scored.date)).all())
    parity=True
    common=("train_ids_hash","eval_ids_hash","shared_train_matrix_hash","shared_eval_matrix_hash","scaler_mean_json","scaler_scale_json")
    for _,group in audit.groupby(["target","series_kind","prediction_date"],sort=False):
        pair=group.loc[group.model.isin(["A","B"])]
        if len(pair)!=2 or pair.model.nunique()!=2:
            parity=False
            continue
        if any(pair[name].nunique(dropna=False)!=1 for name in common):
            parity=False
        names={row.model:json.loads(row.feature_names_json) for row in pair.itertuples()}
        if names["A"]!=[*NUMERIC_FEATURES,"market_JP"] or names["B"]!=[*names["A"],"state_LH_LL","state_LH_HL","state_HH_LL"]:
            parity=False
    ids={model:set(map(tuple,group[["obs_id","target"]].to_numpy())) for model,group in predictions.groupby("model")}
    parity &= set(ids)=={"0","A","B"} and ids["0"]==ids["A"]==ids["B"]
    return causal,bool(parity)


def _charts(folder,evaluation):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.hashsalt":"dow-predictive-v03","font.size":9})
    folder.mkdir()
    def save(fig,name):
        fig.tight_layout(); fig.savefig(folder/(name+".svg"),metadata={"Date":None}); plt.close(fig)
    risk=evaluation["risk_metrics"]
    fig,ax=plt.subplots(figsize=(9,3))
    if len(risk):
        view=risk.loc[risk.target.eq("deep_drawdown_63")&risk.cohort.eq("raw_top10")&risk.momentum_3m_sign.eq("all")]
        table=view.pivot(index="market",columns="model",values="brier_score").reindex(columns=["0","A","B"])
        if table.notna().any().any():
            table.plot.bar(ax=ax)
        else:
            ax.text(.5,.5,"Insufficient observed paired forecasts",ha="center",transform=ax.transAxes)
    ax.set_title("Primary top10: identical-sample Brier scores (lower is better)")
    save(fig,"primary_brier")
    fig,ax=plt.subplots(figsize=(6,4))
    calibration=evaluation["calibration"]
    if len(calibration):
        view=calibration.loc[calibration.target.eq("deep_drawdown_63")&calibration.cohort.eq("raw_top10")&calibration.market.eq("ALL")&calibration.momentum_3m_sign.eq("all")]
        for model,group in view.groupby("model"):
            group=group.loc[group["count"]>0]
            if len(group): ax.plot(group.mean_prediction,group.event_rate,marker="o",label=model)
    ax.plot([0,1],[0,1],linestyle="--",color="grey")
    if ax.get_legend_handles_labels()[0]: ax.legend(title="Model")
    ax.set(xlabel="Forecast probability",ylabel="Observed frequency",title="Top10 risk calibration; fixed probability bins")
    save(fig,"calibration")
    fig,ax=plt.subplots(figsize=(10,3))
    paired=evaluation["paired"]
    view=paired.loc[paired.target.eq("deep_drawdown_63")&paired.raw_top10&paired.pair_status.eq("eligible")].copy()
    if len(view):
        view["month"]=pd.to_datetime(view.date).dt.to_period("M").dt.to_timestamp()
        table=view.groupby(["month","market"]).loss_difference_A_minus_B.mean().unstack()
        table.plot(ax=ax,marker="o")
    ax.axhline(0,color="grey",linestyle="--")
    ax.set_title("Observed monthly top10 Brier loss difference A minus B; positive favors B")
    save(fig,"monthly_loss_difference")


def _parameters(seed,bootstrap_samples):
    return {"span":5,"landmark_interval":21,"landmark_phase":0,"series_kind":"price","primary_target":"deep_drawdown_63",
            "deep_drawdown_threshold":-.2,"drawdown_boundary_atol":1e-12,"risk_entry":"next common valid close","risk_exit":"entry+63 common intervals",
            "persistence_end":"next market landmark (21 benchmark observations), not entry+21 common intervals",
            "numeric_features":list(NUMERIC_FEATURES),"market_encoding":"JP=1,US=0,unscaled",
            "state_reference":"HH/HL","state_dummies":["LH/LL","LH/HL","HH/LL"],"imputation":"none; shared finite-feature eligible rows",
            "scaling":"five numeric predictors, fit only on mature training rows; identical scaler shared by A and B",
            "logistic":{"penalty":"L2","C":1.,"solver":"lbfgs","class_weight":None,"max_iter":1000},"ridge_alpha":1.,
            "minimum_training_dates":12,"minimum_prediction_market_training_dates":12,"minimum_binary_class_count":30,"minimum_numeric_rows":60,
            "training_cutoff":"feature date and actual target end strictly before prediction date",
            "rank_percentile":"(ordinal_rank-1)/(population-1), best0,worst1; fewer2 candidates undefined",
            "rank_population":"same date/market, price present, history sufficient, finite raw12M, before structure filtering; tie ticker",
            "top10":"ceil10% ranked quota and raw12M>0; not actual Alpha selected holdings",
            "bootstrap_samples":int(bootstrap_samples),"seed":int(seed),"uncertainty":"separate market-month block and ticker cluster, event-weighted losses and ratios",
            "sensitivity":"separate second-stage sibling artifact; never changes primary specification or Gate"}


def _gate_text(gate):
    return "# Gate I\n\n"+f"Status: {gate['status']}\n\nGrade: {gate.get('grade')}\n\nScope: {gate.get('scope')}\n\n"+"\n".join("- "+str(reason) for reason in gate.get("reasons",[]))+"\n\nResearch assessment only; no production investment permission.\n"


def _report(meta,evaluation):
    risk=evaluation["risk_metrics"]
    view=risk.loc[risk.target.eq("deep_drawdown_63")&risk.cohort.eq("raw_top10")&risk.momentum_3m_sign.eq("all")] if len(risk) else risk
    return "\n".join(["# Dow Structure Predictive Study v0.3","","## Purpose and fixed comparison","",
        "Test the added predictive information from confirmed structure while keeping Model A/B rows, five numeric predictors and train-only scaling identical. Model B adds only three fixed state indicators.",
        "","## Primary top10 deep-drawdown forecasts","","```",view.to_string(index=False),"```","",
        _gate_text(evaluation["gate"]),"","## Integrity and limitations","",
        *["- "+note for note in meta["warnings"]],
        "- Forecasts are saved before evaluation; replay verifies numerical reproducibility, not truly unused out-of-sample validity.",
        "- Audit rows expose mature training cutoffs, row and matrix hashes, train-only scaler values and exact estimator settings. Test execution is documented separately.",
        "- Main and secondary targets are never blended into one score. Favorable sensitivity or secondary results cannot promote a failed primary Gate."])+"\n"


def _new_output(output_root,protected,prefix="main"):
    root=Path(output_root).expanduser().resolve()
    for source in protected:
        source=Path(source).resolve()
        if root==source or source in root.parents:
            raise ValueError("Output must be outside completed inputs and artifacts")
    now=datetime.now(timezone.utc)
    out=root/f"{now.strftime('%Y%m%dT%H%M%SZ')}_{prefix}_{uuid.uuid4().hex[:12]}"
    out.mkdir(parents=True,exist_ok=False)
    return out


def _case_predictions(output,features,targets,*,seed,bootstrap_samples,verify_replay=True,input_verifier=None):
    _progress("features/targets saved",features=len(features),eligible=int(features.eligible.sum()),targets=len(targets))
    features.to_parquet(output/"feature_dataset.parquet",index=False)
    targets.to_parquet(output/"target_dataset.parquet",index=False)
    predictions,audit,schedule=walk_forward(features,targets,seed=seed,progress=True)
    # Auxiliary retention labels do not enter any predictive feature matrix.
    retention=targets[["obs_id","target","top10_retained"]].copy()
    predictions=predictions.merge(retention,on=["obs_id","target"],how="left",validate="many_to_one")
    for name,frame in (("baseline_predictions.csv",predictions.loc[predictions.model.isin(["0","A"])]),
                       ("structure_predictions.csv",predictions.loc[predictions.model.eq("B")]),
                       ("training_audit.csv",audit),("walkforward_schedule.csv",schedule)):
        frame.to_csv(output/name,index=False)
    saved=pd.concat([pd.read_csv(output/"baseline_predictions.csv",dtype={"model":str,"obs_id":str}),
                     pd.read_csv(output/"structure_predictions.csv",dtype={"model":str,"obs_id":str})],ignore_index=True)
    compare=["obs_id","target","model","prediction_status","prediction","target_status","target_value","train_count","train_ids_hash","eval_ids_hash"]
    order=["obs_id","target","model"]
    assert_frame_equal(predictions[compare].sort_values(order).reset_index(drop=True),saved[compare].sort_values(order).reset_index(drop=True),
                       check_exact=False,rtol=1e-12,atol=1e-12,check_dtype=False)
    causal,parity=_forecast_audit(predictions,audit)
    if not causal or not parity:
        raise AssertionError("Training information maturity or A/B row/input parity failed")
    replay=False
    if verify_replay:
        _progress("independent walk-forward replay")
        other,other_audit,other_schedule=walk_forward(features,targets,seed=seed,progress=False)
        assert_frame_equal(predictions.drop(columns="top10_retained"),other,check_exact=False,rtol=1e-12,atol=1e-12)
        assert_frame_equal(audit,other_audit,check_exact=False,rtol=1e-12,atol=1e-12)
        assert_frame_equal(schedule,other_schedule,check_exact=False,rtol=1e-12,atol=1e-12)
        replay=True
    checked=input_verifier() if input_verifier else {"input_unchanged":False,"protected_unchanged":False}
    integrity={**checked,"training_causal":causal,"input_parity":parity,"replay_verified":replay,"predictions_saved":True}
    evaluation=evaluate_predictions(predictions,bootstrap_samples=bootstrap_samples,seed=seed,integrity=integrity)
    failures=targets.target_status.eq("computation_failed").any() or predictions.prediction_status.isin(["model_fit_failed","computation_failed"]).any()
    if failures:
        evaluation["gate"].update({"status":"evaluation_unavailable","grade":None,"scope":"partial_computation_failure","applicable_markets":[]})
        evaluation["gate"]["reasons"].append("A required target or predictive model calculation failed; valid other-objective results are preserved separately.")
        for assessment in [evaluation["gate"].get("pooled_assessment",{}),*evaluation["gate"].get("market_assessments",{}).values()]:
            assessment.update({"status":"evaluation_unavailable","grade":None,"provisional_due_to_partial_failure":True})
            assessment.setdefault("criteria",{})["all_required_computations_succeeded"]=False
            assessment.setdefault("reasons",[]).append("Required target/model failure prevents Gate approval; numeric diagnostics are provisional.")
    names={"risk_metrics":"risk_metrics.csv","momentum_metrics":"momentum_persistence_metrics.csv","subgroup_metrics":"subgroup_metrics.csv",
           "paired":"paired_prediction_comparison.csv","uncertainty":"uncertainty.csv","calibration":"calibration.csv","stability_metrics":"time_half_metrics.csv","retention_metrics":"top10_retention.csv"}
    for key,name in names.items(): evaluation[key].to_csv(output/name,index=False)
    v1._json(output/"gate1_assessment.json",evaluation["gate"])
    (output/"gate1_assessment.md").write_text(_gate_text(evaluation["gate"]),encoding="utf-8")
    _charts(output/"charts",evaluation)
    label_columns=[name for name in ("obs_id","target","target_status","label_end_date","anomaly_influence","error") if name in targets]
    label_warnings=targets.loc[~targets.target_status.eq("observed"),label_columns].rename(columns={"error":"target_error"})
    quality=pd.concat([predictions.loc[~predictions.prediction_status.eq("predicted"),["obs_id","target","model","prediction_status","error"]],label_warnings],ignore_index=True)
    quality.to_csv(output/"quality_warnings.csv",index=False)
    targets.groupby(["target","target_status"]).size().rename("count").reset_index().to_csv(output/"target_status_counts.csv",index=False)
    return evaluation,integrity,bool(failures),predictions


def _manifest(output):
    v1._json(output/"file_manifest.json",{"algorithm":"sha256","self_excluded":"file_manifest.json","files":{p.relative_to(output).as_posix():v1._hash(p) for p in sorted(output.rglob('*')) if p.is_file() and p!=output/"file_manifest.json"}})


def _check_inputs(snapshot,v02,input_hashes):
    if {name:v1._hash(Path(snapshot)/name) for name in input_hashes["snapshot"]}!=input_hashes["snapshot"]:
        raise AssertionError("Fixed snapshot changed")
    if verify_manifest(v02)!=input_hashes["v02"]:
        raise AssertionError("v0.2 analysis changed")


def run_study(snapshot_dir=DEFAULT_SNAPSHOT,v02_dir=DEFAULT_V02,output_root=DEFAULT_OUTPUT,*,bootstrap_samples=500,seed=20261111):
    if isinstance(bootstrap_samples,bool) or int(bootstrap_samples)!=bootstrap_samples or bootstrap_samples<0:
        raise ValueError("bootstrap_samples must be nonnegative integer")
    output=_new_output(output_root,[snapshot_dir,v02_dir])
    meta={"version":"0.3","stage":"primary","status":"incomplete","validation_status":"incomplete","started_at":datetime.now(timezone.utc).isoformat(),
          "finished_at":None,"snapshot_dir":str(Path(snapshot_dir).resolve()),"v02_dir":str(Path(v02_dir).resolve()),"output_folder":str(output),
          "parameters":_parameters(seed,bootstrap_samples),"warnings":[
              "Retrospective exploration of a current saved universe; survivorship and repeated use of2024-2026 by earlier studies remain.",
              "The research top10 cohort is not the actual Alpha Engine selected portfolio.",
              "63-observation outcomes overlap; repeated ticker, market-date and market-shock dependence remain.",
              "Market-month and ticker uncertainty are separate exploratory intervals, not calibrated iid significance or a combined two-way interval.",
              "Limited history and Japanese top10 samples may make Gate evaluation unavailable, not affirmative or negative evidence.",
              "1306.T adjustment is tentative and not fully verified against official adjusted prices.",
              "Close-based confirmed swings are a simplified structure measure, not classical Dow theory or production trading permission.",
              "User-reported v0.2 empirical findings are background, not independently verified by this implementation.",
              "Primary price and secondary relative structures are never pooled to increase independent sample size."]}
    v1._json(output/"study_metadata.json",meta)
    try:
        meta.update(_verify_code())
        _progress("load and hash fixed inputs")
        original,us,jp,landmarks,hashes,v02_meta=load_inputs(snapshot_dir,v02_dir)
        meta["input_hashes"]=hashes
        meta["v02_commit"]=v02_meta["git_commit_sha"]
        corrected,correction=v1.apply_benchmark_correction(original)
        correction.to_csv(output/"correction_log.csv",index=False)
        meta["correction_history"]=correction.to_dict("records")
        features=build_features(landmarks,"price")
        targets=build_targets(features,landmarks,corrected,us,jp)
        meta["candidate_counts"]={"US":len(us),"JP":len(jp),"feature_rows":len(features),"eligible_rows":int(features.eligible.sum())}
        if "source_cohort_agreement" in features and (~features.source_cohort_agreement).any():
            meta["warnings"].append("Saved cohort flags disagree with freshly recomputed historical cross-sections; only recomputed ranks/cohorts are used.")
        def verifier():
            _check_inputs(snapshot_dir,v02_dir,hashes)
            if _verify_code()["protected_source_sha256"]!=meta["protected_source_sha256"]:
                raise AssertionError("Protected source changed")
            return {"input_unchanged":True,"protected_unchanged":True}
        evaluation,integrity,partial,predictions=_case_predictions(output,features,targets,seed=int(seed),bootstrap_samples=int(bootstrap_samples),input_verifier=verifier)
        _check_inputs(snapshot_dir,v02_dir,hashes)
        if _verify_code()["protected_source_sha256"]!=meta["protected_source_sha256"]:
            raise AssertionError("Protected source changed")
        meta["integrity"]=integrity
        meta["prediction_status_counts"]=predictions.prediction_status.value_counts().to_dict()
        meta["target_status_counts"]=targets.groupby(["target","target_status"]).size().reset_index(name="count").to_dict("records")
        meta["gate1"]={key:evaluation["gate"][key] for key in ("status","grade","scope")}
        meta["versions"]={"python":platform.python_version(),"pandas":pd.__version__,"numpy":np.__version__,**{name:importlib.metadata.version(name) for name in ("scikit-learn","scipy","pyarrow","matplotlib")}}
        pd.DataFrame([{"status":"not_run","reason":"Use run_sensitivities on this completed primary artifact; no alternative can replace the main Gate"}]).to_csv(output/"sensitivity_analysis.csv",index=False)
        meta.update({"status":"partial" if partial else "completed","validation_status":"failed" if partial else "passed","finished_at":datetime.now(timezone.utc).isoformat()})
        v1._json(output/"study_metadata.json",meta)
        (output/"study_report.md").write_text(_report(meta,evaluation),encoding="utf-8")
        _manifest(output)
        _progress("primary finished",status=meta["status"],gate=evaluation["gate"]["grade"],output=output)
    except Exception as exc:
        meta.update({"status":"failed","validation_status":"failed","finished_at":datetime.now(timezone.utc).isoformat(),"error":f"{type(exc).__name__}: {exc}"})
        v1._json(output/"study_metadata.json",meta)
        failed_gate={"status":"evaluation_unavailable","grade":None,"scope":"failed_study","production_permission":False,"reasons":[meta["error"]]}
        v1._json(output/"gate1_assessment.json",failed_gate)
        (output/"gate1_assessment.md").write_text(_gate_text(failed_gate),encoding="utf-8")
        (output/"study_report.md").write_text(f"# Incomplete predictive study\n\n{meta['error']}\n\nNo completed study or Gate approval.\n",encoding="utf-8")
        raise StudyRunError(meta["error"],output) from exc
    return output


def run_sensitivities(main_run_dir,output_root=None):
    main=Path(main_run_dir).resolve()
    main_hashes=verify_manifest(main)
    meta=json.loads((main/"study_metadata.json").read_text(encoding="utf-8"))
    if meta.get("stage")!="primary" or meta.get("status")!="completed" or meta.get("validation_status")!="passed":
        raise ValueError("Sensitivities require a completed, audited primary study")
    output=_new_output(output_root or main.parent,[main,meta["snapshot_dir"],meta["v02_dir"]],prefix="sensitivity")
    record={"version":"0.3","stage":"sensitivity","status":"incomplete","validation_status":"incomplete","parent_primary":str(main),"parent_hashes":main_hashes,
            "started_at":datetime.now(timezone.utc).isoformat(),"finished_at":None,"parameters":meta["parameters"],"warnings":["Exploratory alternatives; never promote the best result over the fixed primary Gate."]}
    v1._json(output/"study_metadata.json",record)
    try:
        original,us,jp,landmarks,hashes,_=load_inputs(meta["snapshot_dir"],meta["v02_dir"])
        if hashes!=meta["input_hashes"]:
            raise ValueError("Primary inputs changed before sensitivity stage")
        corrected,_=v1.apply_benchmark_correction(original)
        def verifier():
            _check_inputs(meta["snapshot_dir"],meta["v02_dir"],hashes)
            current=_verify_code()
            if current["protected_source_sha256"]!=meta["protected_source_sha256"] or current["study_source_sha256"]!=meta["study_source_sha256"]:
                raise AssertionError("Code differs from the completed primary study")
            if verify_manifest(main)!=main_hashes:
                raise AssertionError("Primary study changed")
            return {"input_unchanged":True,"protected_unchanged":True}
        cases=[("span3",3,0,"price",False,False),("span10",10,0,"price",False,False),("phase7",5,7,"price",False,False),
               ("phase14",5,14,"price",False,False),("relative_structure",5,0,"relative",False,False),
               ("uncorrected",5,0,"price",True,False),("exclude_anomaly_influence",5,0,"price",True,True)]
        record["specifications"]=[dict(zip(("name","span","phase","series_kind","uncorrected","exclude_influence"),case)) for case in cases]
        results=[]; failed=False
        for name,span,phase,kind,raw,exclude in cases:
            _progress("sensitivity",specification=name)
            directory=output/name; directory.mkdir()
            price=original if raw else corrected
            if name=="relative_structure":
                alternative=landmarks
            else:
                daily=v2.compute_structure_features(price,us,jp,span)
                alternative=v2.build_landmarks(daily,price,us,jp,phase=phase)
            features=build_features(alternative,kind)
            if exclude:
                affected=features.excluded_influence.fillna(False).astype(bool)
                features.loc[affected,"eligible"]=False
                features.loc[affected,"exclusion_reason"]="known_anomaly_influence"
            targets=build_targets(features,alternative,price,us,jp,exclude_influence=exclude)
            evaluation,_,partial,_=_case_predictions(directory,features,targets,seed=int(meta["parameters"]["seed"]),bootstrap_samples=int(meta["parameters"]["bootstrap_samples"]),verify_replay=False,input_verifier=verifier)
            failed |= partial
            table=evaluation["risk_metrics"].copy(); table["specification"]=name; results.append(table)
            _manifest(directory)
        pd.concat(results,ignore_index=True).to_csv(output/"sensitivity_analysis.csv",index=False)
        _check_inputs(meta["snapshot_dir"],meta["v02_dir"],hashes)
        if verify_manifest(main)!=main_hashes:
            raise AssertionError("Primary artifact changed during sensitivities")
        record.update({"status":"partial" if failed else "completed","validation_status":"failed" if failed else "passed","finished_at":datetime.now(timezone.utc).isoformat(),"primary_unchanged":True})
        v1._json(output/"study_metadata.json",record)
        (output/"study_report.md").write_text("# Sensitivity stage\n\nSeparate exploratory alternatives with identical A/B training rules inside each specification. Main Gate is unchanged.\n\nPer-case replay is not repeated; sensitivity Gate files are not primary approval.\n",encoding="utf-8")
        _manifest(output)
    except Exception as exc:
        record.update({"status":"failed","validation_status":"failed","error":f"{type(exc).__name__}: {exc}"})
        v1._json(output/"study_metadata.json",record)
        raise StudyRunError(record["error"],output) from exc
    return output


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir",default=DEFAULT_SNAPSHOT)
    parser.add_argument("--v02-dir",default=DEFAULT_V02)
    parser.add_argument("--output-root",default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples",type=int,default=500)
    parser.add_argument("--seed",type=int,default=20261111)
    parser.add_argument("--sensitivity-of",help="Completed primary artifact; create a new sibling sensitivity run")
    args=parser.parse_args(argv)
    try:
        if args.sensitivity_of:
            output=run_sensitivities(args.sensitivity_of,None if args.output_root==DEFAULT_OUTPUT else args.output_root)
        else:
            values=vars(args); values.pop("sensitivity_of"); output=run_study(**values)
    except StudyRunError as exc:
        print(f"Study failed: {exc}\nIncomplete output: {exc.run_dir}"); return 1
    print(f"Study output: {output}"); return 0


if __name__=="__main__":
    raise SystemExit(main())
