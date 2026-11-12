"""E30/E31 research primitives. No predictions, trade orders or production edits."""
import numpy as np
import pandas as pd

WINDOWS = (63, 126, 252)
WEIGHTS = (1, 2, 3)
TOL = 1e-10


def strict_signals(prices, tickers, benchmark, sessions, asof):
    """Exact h-session returns; no filling; same calendar/window for every name.

    sessions must be an explicit exchange calendar, not observed price rows.
    Does not claim that ticker is a verified permanent company identifier.
    """
    sessions = pd.DatetimeIndex(sessions)
    if not sessions.is_monotonic_increasing or sessions.has_duplicates:
        raise ValueError('invalid sessions')
    sessions = sessions[sessions <= pd.Timestamp(asof)]
    hist = prices.loc[:pd.Timestamp(asof)].reindex(sessions).tail(253)
    if len(hist) < 253:
        return pd.DataFrame(), pd.DataFrame({'ticker': tickers, 'reason': 'insufficient_sessions'})
    if benchmark not in hist or hist[benchmark].isna().any() or (hist[benchmark] <= 0).any():
        return pd.DataFrame(), pd.DataFrame({'ticker': tickers, 'reason': 'benchmark_missing_or_invalid'})
    b = hist[benchmark].to_numpy()
    br = np.array([b[-1] / b[-h-1] - 1 for h in WINDOWS])
    rows, excluded = [], []
    for t in tickers:
        if t not in hist:
            excluded.append((t, 'missing_column')); continue
        x = hist[t].to_numpy()
        if not np.isfinite(x).all() or (x <= 0).any():
            excluded.append((t, 'incomplete_or_invalid_253_session_window')); continue
        r = np.array([x[-1] / x[-h-1] - 1 for h in WINDOWS])
        m0 = float(np.dot(r, WEIGHTS) / 6)
        m1 = float(np.dot(r-br, WEIGHTS) / 6)
        rows.append((t, m0, m1, float(np.dot(br, WEIGHTS)/6)))
    return (pd.DataFrame(rows, columns=['ticker','m0','m1','benchmark_constant']),
            pd.DataFrame(excluded, columns=['ticker','reason']))


def legacy_simple_independent(stock, bench):
    """Legacy iloc[-n], len>n and partial-window denominator, intentionally."""
    s, b = stock.dropna(), bench.dropna()
    vals = [w*((s.iloc[-1]/s.iloc[-h]-1)-(b.iloc[-1]/b.iloc[-h]-1))
            for h,w in zip(WINDOWS,WEIGHTS) if len(s)>h and len(b)>h]
    return sum(vals)/sum(WEIGHTS[:len(vals)]) if vals else np.nan


def legacy_beta_independent(stock, bench, window):
    """Independent centered-sum covariance implementation of L-beta.

    pad explicitly reflects historical pandas behavior for Series.pct_change.
    Call with aligned, already ffilled histories for frozen legacy replication.
    """
    joined = pd.concat([stock.ffill().pct_change(fill_method=None),
                        bench.ffill().pct_change(fill_method=None)], axis=1).dropna().tail(window)
    if len(joined)<5: return 0.0
    x,y = joined.iloc[:,1].to_numpy(),joined.iloc[:,0].to_numpy()
    dx,dy=x-x.mean(),y-y.mean()
    var=float(dx@dx/(len(x)-1))
    if not np.isfinite(var) or var<=1e-12: return 0.0
    beta=float(dx@dy/(dx@dx))
    return float(np.prod(1+y-beta*x)-1)


def strict_order(frame, score, company_ids):
    if set(frame.ticker)-set(company_ids):
        raise ValueError('verified permanent company IDs required')
    d=frame.assign(company_id=frame.ticker.map(company_ids))
    if d.company_id.duplicated().any():
        raise ValueError('duplicate company IDs require share-class policy')
    return d.sort_values([score,'company_id'],ascending=[False,True],kind='stable').ticker.tolist()


def next_session(sessions, signal_date):
    future=pd.DatetimeIndex(sessions)[pd.DatetimeIndex(sessions)>pd.Timestamp(signal_date)]
    if not len(future): raise ValueError('no next trading session')
    return future[0]


def value_path(prices, sessions, quantity=1, cash_event=None):
    """Tiny audit oracle, not a trading simulator. Unresolved session gaps block.

    Closed days may carry valuations; terminal cash requires an explicit event.
    """
    p=prices.copy().astype(float)
    active=pd.DatetimeIndex(sessions).intersection(p.index)
    if cash_event is not None:
        date,cash=cash_event
        active=active[active<pd.Timestamp(date)]
        p.loc[p.index>=pd.Timestamp(date)]=float(cash)/quantity
    if p.reindex(active).isna().any(): raise ValueError('unresolved open-session missing price')
    return p.ffill()*quantity
