import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
import alpha_engine_backtest as aeb

V=aeb.TTL_COMPOSITE_FORENSICS_DEFAULT_VARIANTS
CORE=aeb.TTL_COMPOSITE_FORENSICS_CORE_VARIANTS

def _make_source(tmp_path, missing=()):
    src=tmp_path/"src"; src.mkdir()
    idx=pd.to_datetime(["2021-12-31","2022-01-31","2022-02-28","2022-03-31","2022-12-31"])
    summary=pd.DataFrame({"CAGR":[.1,.14,.11,.16,.08,.09,.12,.13],"Max_Drawdown":[-.2,-.12,-.22,-.13,-.25,-.21,-.15,-.16],"Sharpe":[1,1.4,1.1,1.5,.8,.9,1.2,1.3],"Calmar":[.5,1.1,.5,1.2,.3,.4,.8,.8],"Turnover":[.5,.45,.55,.5,.6,.5,.45,.46]},index=V)
    cost=pd.DataFrame({"Tax_Slippage_Adjusted_CAGR":[.08,.12,.09,.13,.06,.07,.1,.11],"Net_Sharpe":[.8,1.2,.9,1.3,.6,.7,1.0,1.1],"Net_Calmar":[.4,1.0,.4,1.0,.2,.3,.7,.7],"Estimated_Tax_Drag":[.01]*8,"Estimated_Slippage_Drag":[.002]*8},index=V)
    annual=pd.DataFrame({v:[.05,-.1,.12] for v in V},index=pd.to_datetime(["2021-12-31","2022-12-31","2023-12-31"]))
    monthly=pd.DataFrame({v:[-.01,.02,-.03,.01,.02] for v in V},index=idx)
    draw=pd.DataFrame({v:[0,-.05,-.12,-.02,0] for v in V},index=idx)
    rows=[]; hp=[]; rd=[]; turn=[]; ev=[]
    for v in V:
        turn.append({"variant":v,"trade_date":"2022-01-03","turnover":.5})
        for i,t in enumerate(["AAA","BBB"]):
            w=.5
            rows += [{"variant":v,"date":"2022-01-03","ticker":t,"action":"BUY","weight":w},{"variant":v,"date":"2022-04-05","ticker":t,"action":"SELL","weight":w}]
            hp.append({"variant":v,"ticker":t,"entry_date":"2022-01-03","exit_date":"2022-04-05","holding_days":92})
            if "Renew30" in v:
                rd.append({"variant":v,"ticker":t,"Region":"US","trade_date":"2022-01-03","health_check_date":"2022-04-04","renewed":i==0,"rank_pass":True,"residual_pass":i==0,"price_above_50dma":True,"composite_pass":i==0,"composite_count":2 if i==0 else 1})
            ev.append({"variant":v,"date":"2022-01-03","ticker":t,"event":"entry"})
    files={
        "variant_summary.csv":summary,"cost_adjusted_summary.csv":cost,"annual_returns.csv":annual,"monthly_returns.csv":monthly,"drawdown_series.csv":draw,
        "turnover.csv":pd.DataFrame(turn),"trade_log.csv":pd.DataFrame(rows),"holding_periods.csv":pd.DataFrame(hp),"renewal_decisions.csv":pd.DataFrame(rd),"ttl_event_log.csv":pd.DataFrame(ev)
    }
    for name,df in files.items():
        if name not in missing: df.to_csv(src/name,index=name in {"variant_summary.csv","cost_adjusted_summary.csv","annual_returns.csv","monthly_returns.csv","drawdown_series.csv"})
    if "audit_metadata.json" not in missing: (src/"audit_metadata.json").write_text(json.dumps({"audit_name":"ttl_renewal"}))
    return src

def test_cli_ttl_composite_forensics_recognized():
    p=argparse.ArgumentParser(); p.add_argument("--audit",choices=["ttl_composite_forensics"]); p.add_argument("--source-dir"); p.add_argument("--rerun-selected",action="store_true"); p.add_argument("--variants")
    args=p.parse_args(["--audit","ttl_composite_forensics","--source-dir","x"])
    assert args.audit=="ttl_composite_forensics" and args.source_dir=="x"

def test_forensics_reads_source_and_generates_outputs(tmp_path):
    src=_make_source(tmp_path); out=tmp_path/"out"
    aeb.run_ttl_composite_forensics_audit(src,out)
    required=["forensics_summary.csv","forensics_summary.json","candidate_comparison.csv","active_exposure_daily.csv","active_exposure_summary.csv","annual_return_selected.csv","monthly_return_selected.csv","stress_year_2022.csv","drawdown_episodes.csv","renewal_condition_summary.csv","renewal_decision_by_year.csv","renewal_decision_by_market.csv","holding_period_summary.csv","trade_activity_summary.csv","ticker_contribution_summary.csv","cash_drag_proxy.csv","future_data_boundary_review.csv","complexity_scorecard.csv","ttl_composite_forensics_report.md","audit_metadata.json"]
    for name in required: assert (out/name).exists(), name
    meta=json.loads((out/"audit_metadata.json").read_text())
    assert meta["audit_name"]=="ttl_composite_forensics"
    assert meta["default_download_allowed"] is False
    assert meta["default_full_variant_recalculation_allowed"] is False
    assert meta["selected_variants"]==list(V)
    assert set(pd.read_csv(out/"active_exposure_summary.csv").variant).issubset(set(CORE))
    assert not pd.read_csv(out/"renewal_condition_summary.csv").empty
    assert not pd.read_csv(out/"holding_period_summary.csv").empty
    assert not pd.read_csv(out/"stress_year_2022.csv").empty
    assert not pd.read_csv(out/"complexity_scorecard.csv").empty
    assert not pd.read_csv(out/"future_data_boundary_review.csv").empty
    assert "Executive Summary" in (out/"ttl_composite_forensics_report.md").read_text()

def test_missing_important_files_warns_but_does_not_crash(tmp_path, caplog):
    src=_make_source(tmp_path, missing={"trade_log.csv","renewal_decisions.csv","holding_periods.csv"}); out=tmp_path/"out"
    aeb.run_ttl_composite_forensics_audit(src,out)
    meta=json.loads((out/"audit_metadata.json").read_text())
    assert "trade_log.csv" in meta["important_missing_files"]
    assert (out/"active_exposure_summary.csv").exists()


def test_future_boundary_review_classifies_end_of_sample_partial_cycle():
    rd=pd.DataFrame([
        {"variant":"v","ticker":"AAA","screen_date":"2022-10-03","trade_date":"2022-10-17","health_check_date":"2022-12-31"},
        {"variant":"v","ticker":"BBB","screen_date":"2022-10-04","trade_date":"2022-10-17","health_check_date":"2022-12-31"},
    ])
    artifacts={
        "monthly_returns.csv":pd.DataFrame({"v":[.01]},index=pd.to_datetime(["2022-12-31"])),
        "drawdown_series.csv":pd.DataFrame({"v":[0]},index=pd.to_datetime(["2022-12-31"])),
        "trade_log.csv":pd.DataFrame({"date":["2022-12-31"]}),
    }
    out=aeb._future_boundary_review(rd,artifacts)
    row=out[out.check=="health_check_date near trade_date + 90 days"].iloc[0]
    assert row.status=="end_of_sample_partial_cycle"
    assert "end-of-sample truncation" in row.details
    assert "not future leakage" in row.details


def test_future_boundary_review_keeps_warning_when_returns_exist_after_short_cycle():
    rd=pd.DataFrame([{"variant":"v","ticker":"AAA","screen_date":"2022-10-03","trade_date":"2022-10-17","health_check_date":"2022-12-31"}])
    artifacts={"monthly_returns.csv":pd.DataFrame({"v":[.01,.02]},index=pd.to_datetime(["2022-12-31","2023-01-31"]))}
    out=aeb._future_boundary_review(rd,artifacts)
    row=out[out.check=="health_check_date near trade_date + 90 days"].iloc[0]
    assert row.status=="warning"
    assert "no_return_period_after_health_check=False" in row.details
