"""因子體檢 + 自動權重.

For every scoring factor (with its intended sign):
1. Neutralize: cross-sectional rank, minus its industry average, then the part explained by size
   (log 60-day average traded value) is removed; re-ranked to 0-1. What is left is the chip signal
   that is not just "big caps" or "hot industry".
2. Health (weekly samples, liquid stocks, no transaction costs): rank IC with the forward 5 / 20 / 60
   trading-day excess return, IC information ratio, share of years with positive IC, top-minus-bottom
   quintile excess return over 20 days, and turnover (1 - 20-day rank autocorrelation).
3. Automatic weights, refreshed every quarter WITHOUT look-ahead: at each quarter start, weight =
   max(0, IC IR of the 20-day IC over the previous 3 years, using only outcomes already known).
   Factors whose IC turned the wrong way get 0. Before 1 year of history the original equal weights
   apply. The composite score then uses neutralized ranks with these weights.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .common import log
from .factors import FactorSet, wide

warnings.filterwarnings("ignore", category=RuntimeWarning, module=__name__)   # empty-slice means on sparse rows

HORIZONS = (5, 20, 60)
SAMPLE_EVERY = 5
LOOKBACK_DAYS = 756          # 3 years of trading days for the weights
MIN_HISTORY = 252


def _size(fs: FactorSet, store) -> pd.DataFrame | None:
    if store is None:
        return None
    try:
        pr = store.load("price")
        close, vol = wide(pr, "close", fs.calendar, fs.codes), wide(pr, "volume", fs.calendar, fs.codes)
        tv = (close * vol).where(vol > 0)
        s = np.log(tv.rolling(60, min_periods=20).mean())
        return s if s.notna().values.any() else None
    except Exception as e:
        log.warning("health: size proxy unavailable (%s)", e)
        return None


def _industry(fs: FactorSet) -> np.ndarray | None:
    try:
        ind = fs.meta.reindex(fs.codes)["industry"].fillna("").astype(str).values
    except Exception:
        return None
    return ind if (ind != "").sum() > len(ind) * 0.5 else None


def neutralize(rk: pd.DataFrame, ind: np.ndarray | None, size_rk: np.ndarray | None) -> pd.DataFrame:
    """rk: 0-1 ranks. Remove industry means and the linear size effect per date, re-rank to 0-1."""
    v = rk.values.astype("float64")
    ok = np.isfinite(v)
    if ind is not None:
        codes = pd.Series(ind).astype("category").cat.codes.values
        k = codes.max() + 1
        out = v.copy()
        for g in range(k):
            cols = np.where((codes == g) & (np.asarray(ind) != ""))[0]
            if len(cols) < 5:
                continue
            with np.errstate(all="ignore"):
                m = np.nanmean(v[:, cols], axis=1, keepdims=True)
            out[:, cols] = v[:, cols] - np.nan_to_num(m) + 0.5
        v = np.where(ok, out, np.nan)
    if size_rk is not None:
        s = np.where(ok, size_rk, np.nan)
        with np.errstate(all="ignore"):
            sm, vm = np.nanmean(s, axis=1, keepdims=True), np.nanmean(v, axis=1, keepdims=True)
            cov = np.nanmean((s - sm) * (v - vm), axis=1, keepdims=True)
            var = np.nanvar(s, axis=1, keepdims=True)
            beta = np.where(var > 1e-9, cov / var, 0)
        v = v - np.nan_to_num(beta) * np.nan_to_num(s - sm)
    return pd.DataFrame(v, index=rk.index, columns=rk.columns).rank(axis=1, pct=True)


def _row_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = np.where(ok, a, np.nan), np.where(ok, b, np.nan)
    with np.errstate(all="ignore"):
        a = a - np.nanmean(a, axis=1, keepdims=True)
        b = b - np.nanmean(b, axis=1, keepdims=True)
        c = np.nansum(a * b, axis=1) / np.sqrt(np.nansum(a * a, axis=1) * np.nansum(b * b, axis=1))
    n = ok.sum(axis=1)
    return np.where(n >= 30, c, np.nan)


def compute(fs: FactorSet, store=None) -> dict:
    """Returns {"factors": {name: health}, "signs": {name: sign}, "weights": DataFrame(date x factor), ...}."""
    cal, px = fs.calendar, fs.price
    T = len(cal)
    liq = (fs.liquid if fs.liquid is not None else pd.DataFrame(True, index=cal, columns=fs.codes)).fillna(False).values
    ind = _industry(fs)
    size = _size(fs, store)
    size_rk = size.where(liq).rank(axis=1, pct=True).values if size is not None else None
    samp = np.arange(0, T, SAMPLE_EVERY)
    # forward excess-return ranks at the sampled dates, per horizon
    fwd = {}
    pv = px.values
    for h in HORIZONS:
        with np.errstate(all="ignore"):
            f = np.full_like(pv, np.nan, dtype="float64")
            f[:-h] = pv[h:] / pv[:-h] - 1
        f = np.where(liq, f, np.nan)[samp]
        fwd[h] = pd.DataFrame(f).rank(axis=1, pct=True).values
        if h == 20:
            with np.errstate(all="ignore"):
                fx = np.clip(f, -0.9, 5)
                fwd["exc20"] = fx - np.nanmean(fx, axis=1, keepdims=True)
    factors, neutral, ic20 = {}, {}, {}
    years = cal[samp].year.values
    for g, items in fs.score_spec.items():
        for name, sign in items:
            if name not in fs.factors or name in neutral:
                continue
            rk = (fs.factors[name] * sign).rank(axis=1, pct=True)
            nr = neutralize(rk, ind, size_rk)
            neutral[name] = sign
            x = np.where(liq[samp], nr.values[samp], np.nan)
            del nr, rk
            res = {"group": g, "sign": sign, "label": fs.labels.get(name, name)}
            for h in HORIZONS:
                ic = _row_corr(x, fwd[h])
                if h == 20:
                    ic20[name] = ic
                known = samp + h < T
                v = ic[known & np.isfinite(ic)]
                res[f"ic{h}"] = round(float(np.mean(v)), 4) if len(v) >= 20 else None
                if h == 20 and len(v) >= 20:
                    # samples overlap (20-day returns, 5-day spacing): IR on non-overlapping every 4th sample
                    vv = v[::4]
                    res["ir20"] = round(float(np.mean(vv) / np.std(vv)), 2) if len(vv) > 5 and np.std(vv) > 0 else None
                    yr = pd.Series(ic[known & np.isfinite(ic)], index=years[known & np.isfinite(ic)]).groupby(level=0).mean()
                    res["pos_years"] = int((yr > 0).sum()); res["years"] = int(len(yr))
            # quintile spread of 20-day excess return and turnover
            q = pd.DataFrame(x).rank(axis=1, pct=True).values
            e = fwd["exc20"]
            known = samp + 20 < T
            with np.errstate(all="ignore"):
                top = np.nanmean(np.where(q >= 0.8, e, np.nan), axis=1)
                bot = np.nanmean(np.where(q <= 0.2, e, np.nan), axis=1)
            spr = (top - bot)[known]
            res["q_spread20"] = round(float(np.nanmean(spr)) * 100, 2) if np.isfinite(spr).sum() >= 20 else None
            ac = _row_corr(x[:-4], x[4:]) if len(x) > 8 else np.array([np.nan])
            res["turnover"] = round(float(1 - np.nanmean(ac)), 2) if np.isfinite(ac).any() else None
            factors[name] = res
    # quarterly weights without look-ahead
    names = list(neutral)
    qstarts = pd.DatetimeIndex(sorted({pd.Timestamp(d).to_period("Q").start_time for d in cal}))
    W = pd.DataFrame(np.nan, index=cal, columns=names, dtype="float32")
    sdates = cal[samp]
    for i, qs in enumerate(qstarts):
        rows = (cal >= qs) & (cal < (qstarts[i + 1] if i + 1 < len(qstarts) else cal[-1] + pd.Timedelta(days=1)))
        if not rows.any():
            continue
        q0 = int(np.argmax(rows))
        if q0 < MIN_HISTORY:
            continue
        # outcomes known before the quarter starts: sample date + 20 < q0, within the lookback
        m = (samp + 20 < q0) & (samp >= q0 - LOOKBACK_DAYS)
        if m.sum() < 40:
            continue
        w = {}
        for n in names:
            v = ic20[n][m]
            v = v[np.isfinite(v)][::4]
            w[n] = max(0.0, float(np.mean(v) / np.std(v))) if len(v) > 5 and np.std(v) > 0 else 0.0
        if sum(w.values()) <= 0:
            continue
        W.loc[rows, :] = [w[n] for n in names]
    latest = W.iloc[-1]
    tot = float(np.nansum(latest.values)) or 1.0
    for n in names:
        factors[n]["weight"] = round(float(latest[n]) / tot * 100, 1) if np.isfinite(latest[n]) else None
    log.info("health: %d factors, neutralized by %s%s; weights active from %s", len(names),
             "industry" if ind is not None else "-", "+size" if size_rk is not None else "",
             W.dropna(how="all").index.min())
    return {"factors": factors, "signs": neutral, "weights": W, "ind": ind, "size_rk": size_rk,
            "neutralized_by": [x for x, ok in (("industry", ind is not None), ("size", size_rk is not None)) if ok]}


def weighted_score(fs: FactorSet, h: dict, base_score: pd.DataFrame) -> pd.DataFrame:
    """Composite from neutralized ranks with the quarterly weights; rows without weights keep the base score."""
    W = h["weights"]
    num = np.zeros((len(fs.calendar), len(fs.codes)))
    den = np.zeros_like(num)
    for n, sign in h["signs"].items():
        w = W[n].values[:, None].astype("float64")
        if not (np.isfinite(w) & (w > 0)).any():
            continue
        v = neutralize((fs.factors[n] * sign).rank(axis=1, pct=True), h["ind"], h["size_rk"]).values
        ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
        num += np.where(ok, v * w, 0)
        den += np.where(ok, w, 0)
    with np.errstate(all="ignore"):
        s = np.where(den > 0, num / den, np.nan)
    s = pd.DataFrame(s, index=fs.calendar, columns=fs.codes).rank(axis=1, pct=True) * 100
    has_w = W.notna().any(axis=1).values[:, None]
    out = np.where(has_w & np.isfinite(s.values), s.values, base_score.values)
    return pd.DataFrame(out, index=fs.calendar, columns=fs.codes).astype("float32")
