"""Offline acceptance tests for matured-label, paired predictive research."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import dow_structure_study as v02
import dow_structure_predictive_study as study
from tests.test_dow_momentum_study import (
    JP, US, _hashes, _json, prices, snapshot,
    prohibit_engine_execution_or_downloads,
)


NUMERIC = ["momentum_3m","momentum_6m","momentum_12m","volatility","rank_percentile"]
INTEGRITY = {name:True for name in ("input_unchanged","protected_unchanged","training_causal","input_parity","replay_verified","predictions_saved")}


def _raw_landmarks():
    index = pd.bdate_range("2024-01-02", periods=400)
    names = [f"U{i:02d}" for i in range(11)]
    rows = []
    for position in range(252,400,21):
        for kind in ("price","relative"):
            for i,ticker in enumerate(names):
                state = "HH/HL" if i in (2,5,9) else "unclassified"
                m12 = .01*(min(i,9)+1)
                m3 = .2
                if position>=273 and ticker=="U09":
                    m12,m3,state = -.1,-.01,"unclassified"
                rows.append({"date":index[position],"signal_date":index[position],"ticker":ticker,"market":"US","series_kind":kind,
                             "event_id":f"US|{ticker}|{kind}|{index[position].date()}","market_landmark_number":position//21,
                             "state":state,"classified":state!="unclassified","history_eligible":True,"observations":position+1,
                             "momentum_3m":m3,"momentum_6m":.3,"momentum_12m":m12,"volatility":.1,
                             "rank_12m":1,"rank_population":11,"raw_top10":False,"exclusion_reason":"",
                             "stock_price":100.,"benchmark_price":100.})
    frame = pd.DataFrame({ticker:100. for ticker in names}, index=index)
    frame["SPY"],frame["1306.T"] = 100.+.1*np.arange(len(index)),100.
    return pd.DataFrame(rows),frame,names


def test_features_rank_entire_past_market_before_excluding_unclassified():
    raw,_,_ = _raw_landmarks()
    features = study.build_features(raw)
    first = features.loc[features.date==features.date.min()].set_index("ticker")
    assert len(first)==11
    assert first.loc["U05","rank_percentile"]==pytest.approx(.5)
    assert first.loc["U09","rank_percentile"]==0
    assert first.loc["U10","rank_percentile"]==pytest.approx(.1)
    assert set(first.index[first.raw_top10])=={"U09","U10"}
    assert not bool(first.loc["U10","eligible"])
    assert bool(first.loc["U09","eligible"])
    assert not first.source_cohort_agreement.all()
    assert set(features.series_kind)=={"price"}
    relative = study.build_features(raw,series_kind="relative")
    assert len(relative)==len(features)
    assert set(relative.series_kind)=={"relative"}


def test_future_landmark_changes_cannot_change_current_features_or_ranks():
    raw,_,_ = _raw_landmarks()
    first_date = raw.date.min()
    original = study.build_features(raw)
    changed = raw.copy(deep=True)
    changed.loc[changed.date>first_date,"momentum_12m"] *= -100
    repeated = study.build_features(changed)
    pd.testing.assert_frame_equal(original[original.date==first_date].reset_index(drop=True),repeated[repeated.date==first_date].reset_index(drop=True))
    missing = raw.copy()
    missing.loc[(missing.ticker=="U05")&(missing.date==first_date),"momentum_6m"] = np.nan
    excluded = study.build_features(missing)
    assert not excluded.loc[(excluded.ticker=="U05")&(excluded.date==first_date),"eligible"].any()
    future_pivot = raw.copy()
    future_pivot["reference_high_confirmed_date"] = future_pivot.date+pd.Timedelta(days=1)
    with pytest.raises(ValueError,match="Future confirmed pivot"):
        study.build_features(future_pivot)


def test_zero_volatility_is_valid_and_train_scaler_uses_unit_scale_for_constant_column(model_data):
    raw,_,_ = _raw_landmarks()
    raw["volatility"] = 0.
    zero = study.build_features(raw)
    selected = zero[(zero.ticker=="U05")&(zero.date==zero.date.min())].iloc[0]
    assert selected.eligible
    negative = raw.copy()
    negative["volatility"] = -.01
    rejected = study.build_features(negative)
    rejected = rejected[(rejected.ticker=="U05")&(rejected.date==rejected.date.min())].iloc[0]
    assert not rejected.eligible
    assert rejected.exclusion_reason=="negative_volatility"
    features,targets,dates = model_data
    features = features.copy(deep=True)
    features["volatility"] = 0.
    predictions,audit,_ = study.walk_forward(features,targets,seed=20261111)
    current = predictions[(predictions.target=="deep_drawdown_63")&(predictions.date==dates[12])]
    assert current.prediction_status.eq("predicted").all()
    row = audit[(audit.target=="deep_drawdown_63")&(audit.prediction_date==dates[12])&(audit.model=="A")].iloc[0]
    position = NUMERIC.index("volatility")
    assert json.loads(row.scaler_mean_json)[position]==0.
    assert json.loads(row.scaler_scale_json)[position]==1.


def test_targets_use_next_market_cross_section_including_future_unclassified():
    raw,prices,names = _raw_landmarks()
    features = study.build_features(raw)
    first_date = features.date.min()
    first = features[features.date==first_date]
    targets = study.build_targets(first,raw,prices,names,[])
    obs = first.loc[first.ticker=="U09","obs_id"].iloc[0]
    rows = targets[targets.obs_id==obs].set_index("target")
    assert rows.loc["rank_change_21","target_status"]=="observed"
    assert rows.loc["rank_change_21","target_value"]==1
    assert rows.loc["negative_flip_21","target_status"]=="observed"
    assert rows.loc["negative_flip_21","target_value"]==1
    assert rows.loc["rank_change_21","label_end_date"]==prices.index[273]
    later = features[(features.date==prices.index[273])&(features.ticker=="U09")]
    assert not later.eligible.any()
    existing_negative = study.build_targets(later,raw,prices,names,[]).query("target == 'negative_flip_21'").iloc[0]
    assert existing_negative.target_status=="not_at_risk_nonpositive"
    assert pd.isna(existing_negative.target_value)


def test_targets_distinguish_missing_next_stock_from_end_of_snapshot():
    raw,prices,names = _raw_landmarks()
    first_date = raw.date.min()
    features = study.build_features(raw)
    future_date = prices.index[273]
    absent = raw.copy()
    absent.loc[(absent.ticker=="U05")&(absent.date==future_date),"history_eligible"] = False
    first = features[(features.date==first_date)&(features.ticker=="U05")]
    targets = study.build_targets(first,absent,prices,names,[])
    assert set(targets.loc[targets.target.isin(["rank_change_21","negative_flip_21"]),"target_status"])=={"price_missing"}
    last = features[(features.date==prices.index[-1])&(features.ticker=="U05")]
    targets = study.build_targets(last,raw,prices,names,[])
    assert set(targets.target_status)=={"right_censored"}
    assert targets.target_value.isna().all()


def test_risk_path_starts_next_close_and_uses_63_intervals_geometric_return():
    raw,prices,names = _raw_landmarks()
    features = study.build_features(raw)
    first = features[(features.date==prices.index[252])&features.ticker.isin(["U00","U01"])].copy()
    prices.loc[prices.index[252],"U01"] = 1000.
    prices.loc[prices.index[253]:prices.index[316],"U01"] = np.arange(100.,164.)
    prices.loc[prices.index[253],"U00"] = 100.
    prices.loc[prices.index[254],"U00"] = 125.
    prices.loc[prices.index[260],"U00"] = 99.
    prices.loc[prices.index[316],"U00"] = 130.
    targets = study.build_targets(first,raw,prices,names,[])
    ids = first.set_index("ticker").obs_id
    risky = targets[(targets.obs_id==ids["U00"])].set_index("target")
    assert risky.loc["deep_drawdown_63","target_value"]==1
    assert risky.loc["deep_drawdown_63","path_max_drawdown"]==pytest.approx(99/125-1)
    assert risky.loc["deep_drawdown_63","entry_date"]==prices.index[253]
    assert risky.loc["deep_drawdown_63","label_end_date"]==prices.index[316]
    benchmark = prices.SPY.iloc[316]/prices.SPY.iloc[253]
    assert risky.loc["relative_return_63","target_value"]==pytest.approx(1.3/benchmark-1)
    safe = targets[(targets.obs_id==ids["U01"])&(targets.target=="deep_drawdown_63")].iloc[0]
    assert safe.target_value==0


@pytest.mark.parametrize("trough,expected",[(80.,1.),(80.01,0.),(79.99,1.)])
def test_exact_20pct_drawdown_boundary_is_not_lost_to_float_roundoff(trough,expected):
    raw,prices,names = _raw_landmarks()
    features = study.build_features(raw)
    first = features[(features.date==prices.index[252])&(features.ticker=="U05")]
    prices.loc[prices.index[260],"U05"] = trough
    row = study.build_targets(first,raw,prices,names,[]).query("target == 'deep_drawdown_63'").iloc[0]
    assert row.target_status=="observed"
    assert row.target_value==expected
    assert row.entry_date==prices.index[253]
    assert row.label_end_date==prices.index[316]


def test_one_target_calculation_failure_does_not_falsify_normal_labels(monkeypatch):
    raw,prices,names = _raw_landmarks()
    features = study.build_features(raw)
    first = features[features.date==features.date.min()]
    def broken(*args,**kwargs):
        raise RuntimeError("synthetic risk path failure")
    monkeypatch.setattr(study,"_risk_targets",broken)
    targets = study.build_targets(first,raw,prices,names,[])
    assert set(targets.loc[targets.target.isin(["deep_drawdown_63","relative_return_63"]),"target_status"])=={"computation_failed"}
    assert set(targets.loc[targets.target.isin(["rank_change_21","negative_flip_21"]),"target_status"])=={"observed"}
    assert "risk_path" in targets.attrs["computation_errors"]


@pytest.fixture(scope="module")
def model_data():
    dates = pd.date_range("2024-01-01",periods=16,freq="MS")+pd.Timedelta(days=1)
    rows,labels = [],[]
    for d,date in enumerate(dates):
        for i in range(10):
            obs_id = f"{d:02d}-{i:02d}"
            rows.append({"obs_id":obs_id,"date":date,"ticker":f"T{i}","market":"US" if i<5 else "JP","series_kind":"price",
                         "state":("HH/HL","LH/LL","LH/HL","HH/LL")[i%4],"eligible":True,"exclusion_reason":"",
                         "momentum_3m":.1+.001*d+.01*i,"momentum_6m":.2+.001*d+.02*i,
                         "momentum_12m":.3+.002*d+.01*i,"volatility":.1+.005*i+.0001*d,
                         "rank_percentile":(i%5)/4,"raw_top10":i%5==0,"momentum_3m_sign":"positive"})
            labels.append({"obs_id":obs_id,"target":"deep_drawdown_63","target_status":"observed","target_value":float(i%2),
                           "label_end_date":date+pd.Timedelta(days=1),"entry_date":date,"exit_date":date+pd.Timedelta(days=1)})
    return pd.DataFrame(rows),pd.DataFrame(labels),dates


@pytest.fixture(scope="module")
def fitted(model_data):
    features,targets,_ = model_data
    return study.walk_forward(features,targets,seed=20261111)


def test_walkforward_requires_12_matured_dates_and_records_matching_model_inputs(model_data,fitted):
    features,targets,dates = model_data
    predictions,audit,schedule = fitted
    primary = predictions[predictions.target=="deep_drawdown_63"]
    assert not primary.loc[primary.date<dates[12],"prediction_status"].eq("predicted").any()
    assert primary.loc[primary.date==dates[12],"prediction_status"].eq("predicted").all()
    fits = audit[(audit.target=="deep_drawdown_63")&(audit.status=="fitted")]
    assert (pd.to_datetime(fits.max_label_end_date)<pd.to_datetime(fits.prediction_date)).all()
    assert (pd.to_datetime(fits.training_feature_max_date)<pd.to_datetime(fits.prediction_date)).all()
    assert (fits.unique_training_dates>=12).all()
    for _,group in fits.groupby("prediction_date"):
        models = group.set_index("model")
        for name in ("train_ids_hash","eval_ids_hash","shared_train_matrix_hash","shared_eval_matrix_hash"):
            assert models.loc["A",name]==models.loc["B",name]
        a,b = (json.loads(models.loc[m,"feature_names_json"]) for m in ("A","B"))
        assert a==[*NUMERIC,"market_JP"]
        assert set(b)-set(a)=={"state_LH_LL","state_LH_HL","state_HH_LL"}
        assert "state_HH_HL" not in b
        assert json.loads(models.loc["A","numeric_feature_names_json"])==NUMERIC
        hyper = json.loads(models.loc["A","hyperparameters_json"])
        assert hyper=={"C":1.,"penalty":"l2","class_weight":None,"solver":"lbfgs","max_iter":1000}
    simple = primary[(primary.date==dates[12])&(primary.model=="0")]
    assert np.allclose(simple.loc[simple.market=="US","prediction"],.4)
    assert np.allclose(simple.loc[simple.market=="JP","prediction"],.6)


def test_unmatured_equal_date_label_and_future_features_cannot_fit_scaler(model_data):
    features,targets,dates = model_data
    features,targets = features.copy(deep=True),targets.copy(deep=True)
    leaked = features.iloc[[0]].copy()
    leaked["obs_id"],leaked["ticker"] = "unmatured","LEAK"
    leaked[NUMERIC[:-1]] = 1e8
    leaked["rank_percentile"] = .5
    label = targets.iloc[[0]].copy()
    label["obs_id"],label["target_value"],label["label_end_date"] = "unmatured",1.,dates[12]
    features = pd.concat([features,leaked],ignore_index=True)
    targets = pd.concat([targets,label],ignore_index=True)
    features.loc[features.date>dates[12],NUMERIC[:-1]] = 1e9
    _,audit,_ = study.walk_forward(features,targets,seed=20261111)
    at_date = audit[(audit.target=="deep_drawdown_63")&(audit.prediction_date==dates[12])&(audit.model=="A")].iloc[0]
    train = features[(features.date<dates[12])&(features.obs_id!="unmatured")]
    assert at_date.train_count==120
    assert len(json.loads(at_date.scaler_mean_json))==5
    np.testing.assert_allclose(json.loads(at_date.scaler_mean_json),train[NUMERIC].mean().to_numpy())
    np.testing.assert_allclose(json.loads(at_date.scaler_scale_json),train[NUMERIC].std(ddof=0).to_numpy())


def test_current_future_label_availability_never_controls_forecast(model_data,fitted):
    features,targets,dates = model_data
    changed = targets.copy(deep=True)
    current_ids = set(features.loc[features.date==dates[12],"obs_id"])
    changed.loc[changed.obs_id.isin(current_ids),"target_status"] = "right_censored"
    changed.loc[changed.obs_id.isin(current_ids),["target_value","label_end_date"]] = [np.nan,pd.NaT]
    again,_,_ = study.walk_forward(features,changed,seed=20261111)
    base = fitted[0]
    cols = ["obs_id","target","model","prediction_status","prediction"]
    pd.testing.assert_frame_equal(base.loc[base.date<=dates[12],cols].reset_index(drop=True),again.loc[again.date<=dates[12],cols].reset_index(drop=True))
    assert again.loc[(again.date==dates[12])&(again.target=="deep_drawdown_63"),"prediction_status"].eq("predicted").all()


def test_target_market_needs_its_own_12_matured_dates(model_data):
    features,targets,dates = model_data
    changed = targets.copy()
    unavailable = set(features.loc[(features.market=="JP")&(features.date<dates[8]),"obs_id"])
    changed.loc[changed.obs_id.isin(unavailable),"target_status"] = "right_censored"
    changed.loc[changed.obs_id.isin(unavailable),"target_value"] = np.nan
    predictions,_,_ = study.walk_forward(features,changed)
    current = predictions[(predictions.date==dates[12])&(predictions.target=="deep_drawdown_63")]
    assert current.loc[current.market=="US","prediction_status"].eq("predicted").all()
    assert current.loc[current.market=="JP","prediction_status"].eq("insufficient_training_data").all()


@pytest.mark.parametrize("failure",["one_class","computation_failed"])
def test_failed_or_one_class_target_does_not_disable_normal_numeric_model(model_data,failure):
    features,targets,dates = model_data
    main = targets.copy()
    main["target_value"] = 0.
    if failure=="computation_failed":
        main["target_status"],main["target_value"] = "computation_failed",np.nan
    numeric = targets.copy()
    numeric["target"] = "relative_return_63"
    numeric["target_value"] = np.linspace(-.1,.2,len(numeric))
    predictions,audit,_ = study.walk_forward(features,pd.concat([main,numeric],ignore_index=True))
    current = predictions[predictions.date==dates[12]]
    assert not current.loc[current.target=="deep_drawdown_63","prediction_status"].eq("predicted").any()
    assert current.loc[current.target=="relative_return_63","prediction_status"].eq("predicted").all()
    hyper = audit.loc[(audit.target=="relative_return_63")&(audit.model=="A")&(audit.status=="fitted"),"hyperparameters_json"].iloc[0]
    assert json.loads(hyper)=={"alpha":1.}


def test_negative_flip_does_not_forecast_already_nonpositive_momentum(model_data):
    features,targets,dates = model_data
    features,targets = features.copy(),targets.copy()
    targets["target"] = "negative_flip_21"
    selected = features.index[(features.date==dates[12])&(features.ticker=="T0")][0]
    features.loc[selected,"momentum_3m"] = 0.
    features.loc[selected,"momentum_3m_sign"] = "zero"
    predictions,_,_ = study.walk_forward(features,targets)
    rows = predictions[(predictions.obs_id==features.loc[selected,"obs_id"])&(predictions.target=="negative_flip_21")]
    assert set(rows.prediction_status)=={"not_at_risk_nonpositive"}
    assert rows.prediction.isna().all()


def test_relative_series_cannot_enter_primary_model_and_seed_replays(model_data,fitted):
    features,targets,_ = model_data
    relative = features.copy()
    relative["series_kind"],relative["obs_id"] = "relative",relative.obs_id+"-relative"
    relative[NUMERIC[:-1]] *= 100
    labels = targets.copy()
    labels["obs_id"],labels["target_value"] = labels.obs_id+"-relative",1-labels.target_value
    again,audit,_ = study.walk_forward(pd.concat([features,relative],ignore_index=True),pd.concat([targets,labels],ignore_index=True),seed=20261111)
    primary = again[again.series_kind=="price"].reset_index(drop=True)
    pd.testing.assert_frame_equal(fitted[0],primary)
    repeated = study.walk_forward(features,targets,seed=20261111)
    for before,after in zip(fitted,repeated):
        pd.testing.assert_frame_equal(before,after)


def test_active_forecasts_are_persisted_replayed_and_not_given_unverified_input_proof(tmp_path,model_data):
    features,labels,_ = model_data
    objectives = []
    rank = features.set_index("obs_id").rank_percentile
    for name in ("deep_drawdown_63","negative_flip_21","rank_change_21","relative_return_63"):
        frame = labels.copy(deep=True)
        frame["target"] = name
        frame["top10_retained"] = np.nan
        if name=="rank_change_21":
            frame["target_value"] = frame.obs_id.map(1-2*rank)
            frame["top10_retained"] = 0.
        elif name=="relative_return_63":
            frame["target_value"] = np.linspace(-.1,.2,len(frame))
        objectives.append(frame)
    output = tmp_path/"active_forecasts"
    output.mkdir()
    evaluation,integrity,partial,predictions = study._case_predictions(
        output,features,pd.concat(objectives,ignore_index=True),seed=20261111,
        bootstrap_samples=12,verify_replay=True,input_verifier=None,
    )
    active = predictions[predictions.prediction_status.eq("predicted")]
    assert len(active)>0 and set(active.target)==set(study.TARGET_NAMES)
    assert set(active.model)=={"0","A","B"}
    assert not partial
    assert integrity["training_causal"] and integrity["input_parity"]
    assert integrity["replay_verified"] and integrity["predictions_saved"]
    assert not integrity["input_unchanged"] and not integrity["protected_unchanged"]
    assert evaluation["gate"]["status"]=="evaluation_unavailable"
    assert evaluation["gate"]["grade"] is None
    saved = pd.concat([pd.read_csv(output/name,dtype={"model":str,"obs_id":str})
                       for name in ("baseline_predictions.csv","structure_predictions.csv")],ignore_index=True)
    saved_active = saved[saved.prediction_status.eq("predicted")]
    columns = ["obs_id","target","model","prediction_status","prediction"]
    order = ["obs_id","target","model"]
    pd.testing.assert_frame_equal(active[columns].sort_values(order).reset_index(drop=True),
                                  saved_active[columns].sort_values(order).reset_index(drop=True),
                                  check_exact=False,rtol=1e-12,atol=1e-12)
    assert not pd.read_csv(output/"risk_metrics.csv").empty
    assert not pd.read_csv(output/"momentum_persistence_metrics.csv").empty
    assert _json(output/"gate1_assessment.json")["grade"] is None


def _gate_data(months=8,names=10,positive=4):
    rows = []
    for date in pd.date_range("2025-01-01",periods=months,freq="MS")+pd.Timedelta(days=14):
        for i in range(names):
            rows.append({"target":"deep_drawdown_63","series_kind":"price","raw_top10":True,"pair_status":"eligible",
                         "date":date,"ticker":f"T{i}","market":"US","target_value":float(i<positive),"loss_0":.26,"loss_A":.25,"loss_B":.24})
    uncertainty = pd.DataFrame([{"target":"deep_drawdown_63","cohort":"raw_top10","market":market,"series_kind":"price",
                                 "momentum_3m_sign":"all","cluster_method":"market_month","ci_loss_low":.005,"ci_loss_high":.02,
                                 "ci_relative_low":.01,"ci_relative_high":.1,"cluster_count":months,"bootstrap_samples":20}
                                for market in ("ALL","US")])
    return pd.DataFrame(rows),uncertainty


def test_gate_can_only_qualify_sufficient_supported_primary_research():
    paired,uncertainty = _gate_data()
    gate = study.gate_assessment(paired,uncertainty,INTEGRITY)
    assert gate["grade"]=="A"
    assert gate["scope"]=="market_limited"
    assert gate["applicable_markets"]==["US"]
    assert gate["production_permission"] is False


@pytest.mark.parametrize("improvement,expected",[(.03,"A"),(.02999,"C")])
def test_gate_three_percent_boundary_is_inclusive_without_lowering_threshold(improvement,expected):
    paired,uncertainty = _gate_data()
    paired["loss_A"],paired["loss_B"],paired["loss_0"] = .08,.08*(1-improvement),.09
    uncertainty["ci_loss_low"],uncertainty["ci_loss_high"] = .001,.004
    uncertainty["ci_relative_low"],uncertainty["ci_relative_high"] = .01,.06
    assessment = study.gate_assessment(paired,uncertainty,INTEGRITY)
    assert assessment["grade"]==expected
    assert assessment["criteria"]["minimum_relative_brier_improvement"]==.03


@pytest.mark.parametrize("months,names,positive",[(8,10,1),(7,12,6)])
def test_gate_never_passes_with_too_few_positive_events_or_months(months,names,positive):
    paired,uncertainty = _gate_data(months,names,positive)
    gate = study.gate_assessment(paired,uncertainty,INTEGRITY)
    assert gate["status"]=="evaluation_unavailable"
    assert gate["grade"] is None


@pytest.mark.parametrize("defect",["below_3pct","baseline_better","unstable_halves","crosses_zero","missing_integrity"])
def test_gate_preregistered_improvement_stability_and_uncertainty_cannot_be_skipped(defect):
    paired,uncertainty = _gate_data()
    integrity = dict(INTEGRITY)
    if defect=="below_3pct":
        paired["loss_B"] = .245
    elif defect=="baseline_better":
        paired["loss_0"] = .23
    elif defect=="unstable_halves":
        paired["loss_B"] = np.where(paired.date<=paired.date.sort_values().unique()[3],.20,.26)
    elif defect=="crosses_zero":
        uncertainty["ci_loss_low"],uncertainty["ci_relative_low"] = -.01,-.01
    else:
        integrity["training_causal"] = False
    assert study.gate_assessment(paired,uncertainty,integrity)["grade"]!="A"


def test_secondary_good_result_cannot_override_failed_primary_gate():
    paired,uncertainty = _gate_data()
    paired["loss_B"] = .26
    auxiliary = paired.copy()
    auxiliary["target"],auxiliary["loss_B"] = "rank_change_21",0.
    gate = study.gate_assessment(pd.concat([paired,auxiliary],ignore_index=True),uncertainty,INTEGRITY)
    assert gate["grade"]=="C"


def test_metric_losses_use_identical_available_rows_and_report_nonpositive_exclusions():
    rows = []
    for obs_id,target,target_status,value,sign in [
        ("good","deep_drawdown_63","observed",1.,"positive"),
        ("missing_prediction","deep_drawdown_63","observed",0.,"positive"),
        ("already_negative","negative_flip_21","not_at_risk_nonpositive",np.nan,"negative"),
    ]:
        for model,prediction in [("0",.4),("A",.5),("B",.9)]:
            status = "predicted"
            if obs_id=="already_negative":
                status,prediction = "not_at_risk_nonpositive",np.nan
            elif obs_id=="missing_prediction" and model=="B":
                status,prediction = "model_fit_failed",np.nan
            rows.append({"obs_id":obs_id,"target":target,"model":model,"date":pd.Timestamp("2025-01-15"),
                         "ticker":obs_id,"market":"US","series_kind":"price","state":"HH/HL","raw_top10":True,
                         "momentum_3m_sign":sign,"target_status":target_status,"target_value":value,
                         "prediction":prediction,"prediction_status":status})
    frame = pd.DataFrame(rows)
    result = study.evaluate_predictions(frame,bootstrap_samples=12,seed=20261111,integrity=INTEGRITY)
    risk = result["risk_metrics"]
    primary = risk[(risk.cohort=="all")&(risk.market=="ALL")&(risk.momentum_3m_sign=="all")&risk.target.eq("deep_drawdown_63")].set_index("model")
    assert set(primary.valid_count)=={1}
    assert set(primary.prediction_unavailable_count)=={1}
    assert primary.loc["A","brier_score"]==pytest.approx(.25)
    assert primary.loc["B","brier_score"]==pytest.approx(.01)
    nonpositive = risk[(risk.cohort=="all")&(risk.market=="ALL")&(risk.momentum_3m_sign=="all")&risk.target.eq("negative_flip_21")]
    assert set(nonpositive.not_at_risk_count)=={1}
    assert set(nonpositive.valid_count)=={0}
    assert result["gate"]["grade"] is None
    relative = frame[frame.obs_id=="good"].copy()
    relative["obs_id"],relative["series_kind"] = "good-relative","relative"
    with pytest.raises(ValueError,match="separate"):
        study.evaluate_predictions(pd.concat([frame,relative],ignore_index=True),bootstrap_samples=0,integrity=INTEGRITY)


@pytest.fixture
def v02_artifact(tmp_path,snapshot,prices):
    folder = tmp_path/"fixed_v02"
    folder.mkdir()
    daily = v02.compute_structure_features(prices,US,JP)
    landmarks = v02.build_landmarks(daily,prices,US,JP)
    daily.to_parquet(folder/"daily_structure_features.parquet",index=False)
    landmarks.to_csv(folder/"landmark_observations.csv",index=False)
    pd.DataFrame(columns=["event_id","group","ticker","market","series_kind","signal_date","horizon","status","stock_return","relative_return"]).to_csv(folder/"structure_outcomes.csv",index=False)
    pd.DataFrame(columns=["cohort","series_kind","horizon","state","group","valid_count"]).to_csv(folder/"structure_outcome_summary.csv",index=False)
    pd.DataFrame(columns=["cohort","market","series_kind","state","from_state","count"]).to_csv(folder/"extreme_transitions.csv",index=False)
    meta = {"version":"0.2","status":"completed","validation_status":"passed","git_commit_sha":"1b2ecb4267d176b327d3995ed0bbc3d7a2b273f6",
            "base_commit":"59db288274608ed8f6b8753dcfbe790e37ffe5cb","source_snapshot":str(snapshot),"input_hashes":_hashes(snapshot),
            "parameters":{"span":5,"landmark_interval":21,"landmark_phase":0}}
    (folder/"study_metadata.json").write_text(json.dumps(meta),encoding="utf-8")
    manifest = {"algorithm":"sha256","self_excluded":"file_manifest.json","files":_hashes(folder)}
    (folder/"file_manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
    return folder


def test_small_fixed_input_pipeline_completes_without_faking_predictions_or_gate(tmp_path,snapshot,v02_artifact):
    original_price,original_v02 = _hashes(snapshot),_hashes(v02_artifact)
    repo = Path(study.__file__).resolve().parent
    protected = [*v02.PROTECTED,"dow_structure_study.py"]
    original_code = {name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in protected}
    first = study.run_study(snapshot,v02_artifact,tmp_path/"analysis",bootstrap_samples=12,seed=20261111)
    saved = _hashes(first)
    second = study.run_study(snapshot,v02_artifact,tmp_path/"analysis",bootstrap_samples=12,seed=20261111)
    assert first!=second
    assert _hashes(first)==saved
    assert _hashes(snapshot)==original_price
    assert _hashes(v02_artifact)==original_v02
    assert {name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in protected}==original_code
    required = {"study_metadata.json","study_report.md","file_manifest.json","feature_dataset.parquet","target_dataset.parquet",
                "walkforward_schedule.csv","training_audit.csv","baseline_predictions.csv","structure_predictions.csv",
                "paired_prediction_comparison.csv","risk_metrics.csv","momentum_persistence_metrics.csv","subgroup_metrics.csv",
                "uncertainty.csv","sensitivity_analysis.csv","quality_warnings.csv","gate1_assessment.json","gate1_assessment.md"}
    assert {path.name for path in first.iterdir()}>=required
    assert (first/"charts").is_dir()
    assert _json(first/"study_metadata.json")["status"]=="completed"
    gate = _json(first/"gate1_assessment.json")
    assert gate["status"]=="evaluation_unavailable"
    assert gate["grade"] is None
    for filename in ("baseline_predictions.csv","structure_predictions.csv"):
        rows = pd.read_csv(first/filename)
        assert not rows.prediction_status.eq("predicted").any()
        pd.testing.assert_frame_equal(rows,pd.read_csv(second/filename))
    for filename in ("feature_dataset.parquet","target_dataset.parquet"):
        pd.testing.assert_frame_equal(pd.read_parquet(first/filename),pd.read_parquet(second/filename))
    for name,digest in _json(first/"file_manifest.json")["files"].items():
        assert hashlib.sha256((first/name).read_bytes()).hexdigest()==digest
    sensitivity = study.run_sensitivities(first)
    assert sensitivity.parent==first.parent and sensitivity not in (first,second)
    assert first not in sensitivity.parents
    sensitivity_meta = _json(sensitivity/"study_metadata.json")
    assert sensitivity_meta["stage"]=="sensitivity"
    assert sensitivity_meta["status"]=="completed"
    assert sensitivity_meta["primary_unchanged"] is True
    expected_cases = {"span3","span10","phase7","phase14","relative_structure","uncorrected","exclude_anomaly_influence"}
    assert set(pd.read_csv(sensitivity/"sensitivity_analysis.csv").specification)==expected_cases
    aggregate_manifest = _json(sensitivity/"file_manifest.json")["files"]
    all_files = {path.relative_to(sensitivity).as_posix() for path in sensitivity.rglob("*") if path.is_file()}
    assert set(aggregate_manifest)==all_files-{"file_manifest.json"}
    assert {case+"/file_manifest.json" for case in expected_cases}<=set(aggregate_manifest)
    for name,digest in aggregate_manifest.items():
        assert hashlib.sha256((sensitivity/name).read_bytes()).hexdigest()==digest
    assert _hashes(first)==saved
    assert _hashes(snapshot)==original_price and _hashes(v02_artifact)==original_v02
    for case in expected_cases:
        audit = pd.read_csv(sensitivity/case/"training_audit.csv",dtype={"model":str})
        for _,group in audit.groupby(["target","series_kind","prediction_date"]):
            models = group.set_index("model")
            assert models.loc["A","train_ids_hash"]==models.loc["B","train_ids_hash"]
            assert models.loc["A","eval_ids_hash"]==models.loc["B","eval_ids_hash"]


@pytest.mark.parametrize("which",["price","v02"])
def test_modified_fixed_input_stops_without_repair_or_new_prices(tmp_path,snapshot,v02_artifact,which):
    path = snapshot/"price_data"/"price_cache.parquet" if which=="price" else v02_artifact/"landmark_observations.csv"
    path.write_bytes(path.read_bytes()+b"changed")
    before_price,before_v02 = _hashes(snapshot),_hashes(v02_artifact)
    with pytest.raises((ValueError,study.StudyRunError)):
        study.run_study(snapshot,v02_artifact,tmp_path/"analysis",bootstrap_samples=0)
    assert _hashes(snapshot)==before_price and _hashes(v02_artifact)==before_v02


def test_sensitivity_stage_refuses_unfinished_primary_before_creating_new_output(tmp_path):
    folder = tmp_path/"unfinished_primary"
    folder.mkdir()
    (folder/"study_metadata.json").write_text(json.dumps({"stage":"primary","status":"incomplete","validation_status":"incomplete"}),encoding="utf-8")
    (folder/"file_manifest.json").write_text(json.dumps({"files":_hashes(folder)}),encoding="utf-8")
    before = _hashes(folder)
    with pytest.raises(ValueError,match="completed"):
        study.run_sensitivities(folder)
    assert _hashes(folder)==before
    assert {path.name for path in tmp_path.iterdir()}=={"unfinished_primary"}
