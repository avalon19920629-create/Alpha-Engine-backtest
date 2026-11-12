"""Independent behavioral probes, including legacy hazards as characterizations."""
import numpy as np
import pandas as pd
import pytest
import alpha_engine_backtest as ae
from research.phase3.signals import strict_signals, strict_order, legacy_beta_independent, legacy_simple_independent, next_session, value_path


def fixture(n=300):
    idx=pd.bdate_range('2020-01-01',periods=n)
    return pd.DataFrame({'A':100*1.001**np.arange(n),'B':100*1.002**np.arange(n),'SPY':100*1.0005**np.arange(n)},index=idx)


def test_exact_252_intervals_vs_legacy_251():
    p=fixture(); f,e=strict_signals(p,['A'],'SPY',p.index,p.index[-1])
    want=sum(w*(1.001**h-1) for h,w in [(63,1),(126,2),(252,3)])/6
    assert f.m0.iloc[0]==pytest.approx(want,abs=1e-12)
    assert not np.isclose(f.m0.iloc[0],ae.score_universe(p,['A'],p.index[-1]).Composite_Ret.iloc[0])


def test_252_prices_rejected_253_prices_accepted():
    p=fixture(253)
    assert strict_signals(p.iloc[:252],['A'],'SPY',p.index[:252],p.index[251])[0].empty
    assert len(strict_signals(p,['A'],'SPY',p.index,p.index[-1])[0])==1


def test_future_mutation_does_not_change_signals():
    p=fixture(320);d=p.index[280];mut=p.copy();mut.loc[mut.index>d]*=1000
    for source in (p,mut):
        f,e=strict_signals(source,['A','B'],'SPY',source.index,d)
        if source is p: ref=f
        else: pd.testing.assert_frame_equal(ref,f,check_exact=True)
    pd.testing.assert_frame_equal(ae.score_universe(p,['A','B'],d),ae.score_universe(mut,['A','B'],d))


def test_closed_day_removed_not_counted_as_return_interval():
    p=fixture();closed=p.index[200];sessions=p.index.delete(200)
    q=p.copy();q.loc[closed]=np.nan
    f,_=strict_signals(q,['A','B'],'SPY',sessions,p.index[-1])
    g,_=strict_signals(p.drop(index=closed),['A','B'],'SPY',sessions,p.index[-1])
    pd.testing.assert_frame_equal(f,g)


def test_open_session_missing_not_forward_filled():
    p=fixture();p.loc[p.index[100],'A']=np.nan
    f,e=strict_signals(p,['A','B'],'SPY',p.index,p.index[-1])
    assert f.ticker.tolist()==['B']
    assert e.ticker.tolist()==['A']


def test_missing_calendar_row_is_not_silently_dropped():
    p=fixture();sessions=p.index;q=p.drop(index=sessions[120])
    f,e=strict_signals(q,['A','B'],'SPY',sessions,sessions[-1])
    assert f.empty and e.reason.eq('benchmark_missing_or_invalid').all()


def test_missing_benchmark_blocks_instead_of_neutral_scores():
    p=fixture().drop(columns='SPY')
    f,e=strict_signals(p,['A'],'SPY',p.index,p.index[-1])
    assert f.empty
    assert e.reason.iloc[0]=='benchmark_missing_or_invalid'


def test_invalid_prices_excluded():
    p=fixture();p.loc[p.index[90],'A']=0
    assert strict_signals(p,['A'],'SPY',p.index,p.index[-1])[0].empty


def test_same_prices_tie_deterministic_verified_ids():
    p=fixture();p.B=p.A
    f,_=strict_signals(p,['B','A'],'SPY',p.index,p.index[-1]);ids={'A':'002','B':'001'}
    assert strict_order(f,'m0',ids)==strict_order(f,'m1',ids)==['B','A']
    assert strict_order(f.iloc[::-1],'m0',ids)==['B','A']


def test_missing_company_id_and_duplicate_share_classes_block():
    f,_=strict_signals(fixture(),['A','B'],'SPY',fixture().index,fixture().index[-1])
    with pytest.raises(ValueError):strict_order(f,'m0',{'A':'1'})
    with pytest.raises(ValueError):strict_order(f,'m0',{'A':'1','B':'1'})


def test_beta_only_return_removed():
    rng=np.random.default_rng(20261113);rm=rng.normal(0,.01,300)
    b=pd.Series(100*np.cumprod(1+rm));s=pd.Series(100*np.cumprod(1+1.7*rm))
    assert abs(ae.compute_beta_adjusted_residual_score(s,b,252))<1e-10
    assert abs(legacy_beta_independent(s,b,252))<1e-10


def test_original_simple_independent_and_benchmark_mode_keyword():
    p=fixture();mode={'US':('SPY',)}
    res=ae.compute_residual_momentum_score(p,['A','B'],p.index[-1],region='US',benchmark_mode=mode)
    for t in ['A','B']:
        assert res.loc[t,'residual_raw']==pytest.approx(legacy_simple_independent(p[t],p.SPY),abs=1e-10)


def test_beta_future_cut_is_unchanged():
    p=fixture(320);d=p.index[280];q=p.copy();q.loc[q.index>d]*=1000
    kw={'region':'US','benchmark_mode':{'US':('SPY',)},'method':'beta_adjusted'}
    pd.testing.assert_frame_equal(ae.compute_residual_momentum_score(p,['A'],d,**kw),ae.compute_residual_momentum_score(q,['A'],d,**kw))


def test_legacy_first_trade_day_return_is_pre_entry_exposure():
    # Characterize the legacy hold>=trade rule; production is not changed.
    idx=pd.bdate_range('2026-03-30',periods=4)
    p=pd.DataFrame({'A':[100.,100.,200.,200.]},index=idx)
    trade=idx[2];ret=p.pct_change(fill_method=None)
    legacy_taken=ret.loc[ret.index>=trade,'A']
    close_entry_taken=ret.loc[ret.index>trade,'A']
    assert (1+legacy_taken).prod()-1==1.0
    assert (1+close_entry_taken).prod()-1==0.0


def test_legacy_beta_retains_intercept_and_compounds_it():
    rng=np.random.default_rng(20261113);rm=rng.normal(0,.01,300);alpha=.001
    b=pd.Series(100*np.cumprod(1+rm));s=pd.Series(100*np.cumprod(1+alpha+1.7*rm))
    expected=(1+alpha)**252-1
    assert ae.compute_beta_adjusted_residual_score(s,b,252)==pytest.approx(expected,abs=1e-10)
    assert legacy_beta_independent(s,b,252)==pytest.approx(expected,abs=1e-10)


def test_in_sample_ols_residual_sum_zero():
    rng=np.random.default_rng(123);x=rng.normal(0,.01,252);y=.001+1.4*x+rng.normal(0,.002,252)
    X=np.column_stack([np.ones(len(x)),x]);coef=np.linalg.lstsq(X,y,rcond=None)[0]
    assert abs((y-X@coef).sum())<1e-10


def test_legacy_zero_variance_and_short_window_return_zero():
    b=pd.Series(np.ones(300)*100);s=pd.Series(np.linspace(100,200,300))
    assert ae.compute_beta_adjusted_residual_score(s,b,252)==0
    assert legacy_beta_independent(s,b,252)==0
    assert ae.compute_beta_adjusted_residual_score(s.head(4),b.head(4),252)==0


def test_next_session_excludes_signal_day():
    sessions=pd.to_datetime(['2026-01-02','2026-01-05','2026-01-06'])
    assert next_session(sessions,'2026-01-02')==pd.Timestamp('2026-01-05')
    with pytest.raises(ValueError):next_session(sessions,'2026-01-06')


def test_closed_day_valuation_then_reopening_return():
    idx=pd.date_range('2026-01-02',periods=4)
    p=pd.Series([100,np.nan,np.nan,110],index=idx)
    v=value_path(p,idx[[0,3]],quantity=2)
    assert v.tolist()==[200,200,200,220]
    assert v.iloc[-1]/v.iloc[0]-1==pytest.approx(.1)


def test_unknown_terminal_price_blocks_and_documented_cash_event_resolves():
    idx=pd.bdate_range('2026-01-01',periods=4);p=pd.Series([100,101,np.nan,np.nan],index=idx)
    with pytest.raises(ValueError):value_path(p,idx)
    v=value_path(p,idx,quantity=2,cash_event=(idx[2],220))
    assert v.tolist()==[200,202,220,220]


def test_known_split_adjustment_preserves_wealth():
    # A two-for-one split doubles actual units; adjusted prices must not create a loss.
    raw=np.array([100.,50.]);shares=np.array([1.,2.])
    np.testing.assert_array_equal(raw*shares,[100.,100.])
    idx=pd.bdate_range('2026-01-01',periods=2)
    assert value_path(pd.Series([50.,50.],index=idx),idx,quantity=2).tolist()==[100.,100.]


def test_share_exchange_identity_not_ticker_concatenation():
    # Documented 1:2 exchange: holding value is successor units times successor price.
    old_units=3;ratio=2;successor_price=40;cash_per_old=5
    assert old_units*ratio*successor_price+old_units*cash_per_old==255
    with pytest.raises(ValueError):
        strict_order(pd.DataFrame({'ticker':['OLD','NEW'],'m0':[.2,.2]}),'m0',{'OLD':'company','NEW':'company'})
