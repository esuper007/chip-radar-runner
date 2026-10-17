"""「突破型起漲」: which chip paths make a breakout to a 1-year high actually turn into a big winner?

1. Event (observable on the day itself, no hindsight): the close is above the highest close of the
   previous 250 trading days, and the stock had not made such a new high in the previous 60 days
   (a fresh breakout from a base, not every new high of an ongoing rally). Liquid stocks only.
2. Success, two modes (the forward return ranked against ALL liquid stocks on the breakout day, so
   market swings cancel out):
     h126q10  return over the next 126 trading days (~half a year) in the top 10%
     h252q20  return over the next 252 trading days (~one year)    in the top 20%
3. Observation window: the L trading days before the breakout, L chosen per market and mode from
   {123, 189, 255} (quarterly, see 6). Features (cross-sectional percentiles 0-100) for the
   composite score, every sub-score and every scoring factor:
     late   = mean over the last 123 days before the breakout
     early  = mean over days L..124 before the breakout (when L > 123)
     chg    = late - early
     recent = mean over the last 20 days
   Price is not a feature (every event is a breakout by construction).
4. Profile: successful breakouts vs ALL breakouts (failed ones included): standardized difference
   per feature, kept when |d| >= 0.1 and most events have the data. Score = weighted agreement.
5. Validation (walk-forward by year, no look-ahead): profile from breakouts whose outcome was known
   before the test year; in the test year, the top 20% of breakouts by score vs all breakouts:
   success rate (lift = ratio) and average forward return difference.
6. Window optimization once per quarter, with the same honest check as the surge similarity:
   each year uses the L that was best on earlier years only.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .common import log
from .factors import FactorSet, xrank
from .surge import GROUPS, _Win, _fit, _score, _series, quarter_end

HIGH_LOOKBACK = 250
FRESH_DAYS = 60          # no new 1-year high in this many days before the breakout
RECENT_EVENT_DAYS = 5    # breakouts in the last N sessions are listed on the dashboard
NEAR_PCT = 0.03          # "about to break out": within 3% of the 1-year high
LATE, RECENT = 123, 20
GRID = [123, 189, 255]
MODES = {"h126q10": (126, 0.10), "h252q20": (252, 0.20)}
PAT_SHORT, PAT_LONG = 22, 123      # pattern windows (trading days before the breakout)
RANGE_MAX = 0.40                    # 波動收斂: 22-day close range < 40% of the 123-day range
MODE_NAMES = {"h126q10": "突破後半年報酬前 10%", "h252q20": "突破後一年報酬前 20%"}
TOP_SHARE = 0.2          # validation: the top 20% of breakouts by score
MAX_EVENTS = 250_000
OPT_META = "breakout_opt"
SEG = {"early": "突破前段", "late": "突破前半年", "chg": "前後段變化", "recent": "突破前 20 日"}


def _label(key: str, fs: FactorSet) -> str:
    seg, name = key.split(":", 1)
    if name == "score":
        base = "籌碼綜合分數"
    elif name.startswith("s_"):
        base = GROUPS.get(name[2:], name[2:]) + "子分數"
    else:
        base = fs.labels.get(name[2:], name[2:])
    return SEG[seg] + "・" + base


def _features(wins: dict, rows: np.ndarray, cols: np.ndarray, L: int):
    """Features for events on (rows, cols); windows end the day BEFORE the breakout."""
    end = rows - 1
    keys, mats = [], []
    for name, w in wins.items():
        late = w.mean(end - LATE + 1, end, cols)
        rec = w.mean(end - RECENT + 1, end, cols)
        keys += [f"late:{name}", f"recent:{name}"]
        mats += [late, rec]
        if L > LATE:
            early = w.mean(end - L + 1, end - LATE, cols)
            keys += [f"early:{name}", f"chg:{name}"]
            mats += [early, late - early]
    return keys, np.column_stack(mats).astype("float32")


class _Ctx:
    """Everything shared by all windows and modes for one market."""

    def __init__(self, fs: FactorSet, scores: dict):
        self.fs = fs
        cal, px = fs.calendar, fs.price
        self.T = len(cal)
        self.liq = (fs.liquid if fs.liquid is not None else pd.DataFrame(True, index=cal, columns=fs.codes)).fillna(False).values
        has = scores["score"].notna().values
        self.first = int(np.argmax(has.any(axis=1)))
        self.wins = {k: _Win(v) for k, v in _series(fs, scores).items()}
        pv = px.values.astype("float64")
        self.pv = pv
        prior_hi = px.shift(1).rolling(HIGH_LOOKBACK, min_periods=HIGH_LOOKBACK).max().values
        self.prior_hi = prior_hi
        with np.errstate(invalid="ignore"):
            newhi = pv > prior_hi
        recent_hi = pd.DataFrame(newhi).shift(1).rolling(FRESH_DAYS, min_periods=1).max().fillna(0).values > 0
        self.event = newhi & ~recent_hi & self.liq & has
        # forward returns and their same-day percentile among liquid stocks, per horizon
        self.fwd, self.fpct = {}, {}
        for h in {h for h, _ in MODES.values()}:
            with np.errstate(all="ignore"):
                f = pv[h:] / pv[:-h] - 1 if h < self.T else np.empty((0, pv.shape[1]))
            f = np.vstack([f, np.full((min(h, self.T), pv.shape[1]), np.nan)])
            f = np.where(self.liq, f, np.nan)
            self.fwd[h] = f.astype("float32")
            self.fpct[h] = pd.DataFrame(f).rank(axis=1, pct=True).values.astype("float32")
        r, c = np.where(self.event)
        keep = r - 1 - max(GRID) + 1 >= self.first          # full observation window for the longest L
        r, c = r[keep], c[keep]
        if len(r) > MAX_EVENTS:
            pick = np.sort(np.random.default_rng(0).choice(len(r), MAX_EVENTS, replace=False))
            r, c = r[pick], c[pick]
        self.ev_r, self.ev_c = r, c
        self.ev_year = cal[r].year.values if len(r) else np.array([], int)
        self._feat = {}
        # price patterns before the breakout (closes only; every window ends the day BEFORE row t)
        p1 = px.shift(1)
        with np.errstate(all="ignore"):
            rng_s = (p1.rolling(PAT_SHORT, min_periods=PAT_SHORT).max() - p1.rolling(PAT_SHORT, min_periods=PAT_SHORT).min()) / p1
            rng_l = (p1.rolling(PAT_LONG, min_periods=PAT_LONG).max() - p1.rolling(PAT_LONG, min_periods=PAT_LONG).min()) / p1
            ret1 = np.log(p1 / p1.shift(1))
            vol_s = ret1.rolling(PAT_SHORT, min_periods=PAT_SHORT - 2).std()
            vol_l = ret1.rolling(PAT_LONG, min_periods=PAT_LONG - 5).std()
            self.range_ratio = (rng_s / rng_l).values.astype("float32")
            self.vol_ratio = (vol_s / vol_l).values.astype("float32")
            low_recent = p1.rolling(PAT_SHORT, min_periods=PAT_SHORT).min()
            low_before = px.shift(PAT_SHORT + 1).rolling(PAT_LONG - PAT_SHORT, min_periods=PAT_LONG - PAT_SHORT).min()
            hl = (low_recent > low_before).values
            self.higher_low = np.where(np.isfinite(low_recent.values) & np.isfinite(low_before.values), hl, np.nan).astype("float32")

    def feats(self, L: int):
        if L not in self._feat:
            self._feat[L] = _features(self.wins, self.ev_r, self.ev_c, L)
        return self._feat[L]

    def outcome(self, mode: str):
        h, q = MODES[mode]
        known = self.ev_r + h < self.T
        p = self.fpct[h][self.ev_r, self.ev_c]
        f = self.fwd[h][self.ev_r, self.ev_c]
        succ = np.where(np.isfinite(p), p >= 1 - q, np.nan)
        return known & np.isfinite(p), succ, f


def _validate(ctx: _Ctx, mode: str, L: int) -> dict:
    h, _ = MODES[mode]
    keys, X = ctx.feats(L)
    known, succ, fwd = ctx.outcome(mode)
    cal = ctx.fs.calendar
    per_year, buckets = [], []
    n_top = n_all = s_top = s_all = f_top = f_all = 0.0
    for y in sorted(set(ctx.ev_year)):
        test = (ctx.ev_year == y) & known
        y0 = np.searchsorted(cal.values, np.datetime64(f"{y}-01-01"))
        train = known & (ctx.ev_r + h < y0)
        if test.sum() < 30 or train.sum() < 300 or np.nansum(succ[train]) < 30:
            continue
        tr = np.where(train)[0]
        model = _fit(X[tr[succ[tr] == 1]], X[tr])
        te = np.where(test)[0]
        s, _ = _score(X[te], model)
        # ties (e.g. no feature passed the threshold) are broken at random, so "top 20%" is never empty
        tie = np.random.default_rng(int(y)).random(len(s)) * 1e-9
        pct = pd.Series(s + tie).rank(pct=True, method="first").values
        top = pct > 1 - TOP_SHARE
        st, sa = float(np.mean(succ[te][top])), float(np.mean(succ[te]))
        ft = float(np.nanmean(np.clip(fwd[te][top], -0.9, 5))) * 100
        fa = float(np.nanmean(np.clip(fwd[te], -0.9, 5))) * 100
        per_year.append({"year": int(y), "n": int(test.sum()), "top_rate": round(st, 3), "base_rate": round(sa, 3),
                         "excess": round(ft - fa, 2)})
        n_top += top.sum(); n_all += len(te)
        s_top += st * top.sum(); s_all += sa * len(te)
        f_top += ft * top.sum(); f_all += fa * len(te)
        buckets.append(np.column_stack([np.minimum((pct * 5).astype(int), 4), succ[te]]))
    if not per_year:
        return {"n_years": 0}
    top_rate, base_rate = s_top / max(n_top, 1), s_all / max(n_all, 1)
    excess = f_top / max(n_top, 1) - f_all / max(n_all, 1)
    b = np.vstack(buckets)
    quint = [round(float(b[b[:, 0] == k, 1].mean()), 3) if (b[:, 0] == k).any() else None for k in range(5)]
    lift = top_rate / base_rate if base_rate > 0 else None
    return {"n_years": len(per_year), "n_events": int(n_all), "top_rate": round(top_rate, 3),
            "base_rate": round(base_rate, 3), "lift": round(lift, 2) if lift else None,
            "excess": round(excess, 2), "quintiles": quint, "per_year": per_year,
            "valid": bool(len(per_year) >= 3 and (lift or 0) >= 1.3 and excess > 0)}


def _optimize(ctx: _Ctx, store) -> dict:
    """Best L per mode on the grid; cached per quarter in the market's data store."""
    q = quarter_end(ctx.fs.calendar[-1])
    cached = store.load_meta(OPT_META) if store is not None else None
    if (cached and os.environ.get("SURGE_OPTIMIZE") != "1" and cached.get("quarter") == q
            and cached.get("grid") == GRID and sorted(cached.get("modes", {})) == sorted(MODES)):
        log.info("breakout: using windows optimized for quarter %s", q)
        return cached
    out = {"quarter": q, "optimized_at": str(ctx.fs.calendar[-1].date()), "grid": GRID, "modes": {}}
    for mode in MODES:
        rows, per = [], {}
        for L in GRID:
            v = _validate(ctx, mode, L)
            if not v.get("n_years"):
                continue
            rows.append({"L": L, "lift": v["lift"], "excess": v["excess"], "top_rate": v["top_rate"],
                         "base_rate": v["base_rate"], "valid": v["valid"]})
            per[L] = {p["year"]: p for p in v["per_year"]}
            log.info("breakout %s L=%d: lift=%s excess=%s", mode, L, v["lift"], v["excess"])
        if not rows:
            continue
        best = max(rows, key=lambda x: (x["lift"] or 0, x["excess"]))
        # honest check: each year uses the L that was best on earlier years only
        hs = hb = hx = hn = 0.0
        picks = []
        for y in sorted({y for d in per.values() for y in d}):
            prior = {}
            for L, d in per.items():
                ys = [d[x] for x in d if x < y]
                if len(ys) >= 2:
                    prior[L] = sum(x["top_rate"] * x["n"] for x in ys) / sum(x["base_rate"] * x["n"] for x in ys)
            if not prior:
                continue
            L = max(prior, key=prior.get)
            if y not in per[L]:
                continue
            yr = per[L][y]
            picks.append({"year": int(y), "L": L})
            hs += yr["top_rate"] * yr["n"]; hb += yr["base_rate"] * yr["n"]; hx += yr["excess"] * yr["n"]; hn += yr["n"]
        honest = None
        if hn and hb:
            honest = {"lift": round(hs / hb, 2), "excess": round(hx / hn, 2), "years": len(picks), "picks": picks}
            honest["valid"] = bool(len(picks) >= 3 and honest["lift"] >= 1.3 and honest["excess"] > 0)
        out["modes"][mode] = {"best": best["L"], "grid": rows, "honest": honest}
    if store is not None and out["modes"]:
        try:
            store.save_meta(OPT_META, out)
        except Exception as e:
            log.warning("could not save breakout optimum: %s", e)
    return out


def _rate(succ, fwd, m):
    m = m & np.isfinite(succ)
    n = int(m.sum())
    if n == 0:
        return {"n": 0}
    return {"n": n, "rate": round(float(np.mean(succ[m])), 3),
            "ret": round(float(np.nanmean(np.clip(fwd[m], -0.9, 5))) * 100, 2)}


def patterns(ctx: _Ctx) -> dict:
    """Fixed price-pattern rules (no fitting): success rate of breakouts with / without each pattern."""
    r, c = ctx.ev_r, ctx.ev_c
    rr, vr, hl = ctx.range_ratio[r, c], ctx.vol_ratio[r, c], ctx.higher_low[r, c]
    tight = np.where(np.isfinite(rr), rr < RANGE_MAX, np.nan)
    out = {"defs": {"short": PAT_SHORT, "long": PAT_LONG, "range_max": RANGE_MAX}, "modes": {}}
    ok_all = np.isfinite(tight) & np.isfinite(hl) & np.isfinite(vr)
    vq = np.nanquantile(vr[ok_all], [0.2, 0.4, 0.6, 0.8]) if ok_all.any() else None
    for mode, (h, q) in MODES.items():
        known, succ, fwd = ctx.outcome(mode)
        base = known & ok_all
        if base.sum() < 200:
            continue
        t1, h1 = tight == 1, hl == 1
        m = {"all": _rate(succ, fwd, base),
             "tight": _rate(succ, fwd, base & t1), "not_tight": _rate(succ, fwd, base & ~t1),
             "higher_low": _rate(succ, fwd, base & h1), "not_higher_low": _rate(succ, fwd, base & ~h1),
             "both": _rate(succ, fwd, base & t1 & h1), "tight_only": _rate(succ, fwd, base & t1 & ~h1),
             "hl_only": _rate(succ, fwd, base & ~t1 & h1), "neither": _rate(succ, fwd, base & ~t1 & ~h1)}
        # volatility ratio (22-day / 123-day return volatility) by quintile, low = most contracted
        bins = np.digitize(vr, vq) if vq is not None else None
        m["vol_quintiles"] = [dict(_rate(succ, fwd, base & (bins == k)),
                                   upto=round(float(vq[k]), 2) if k < 4 else None) for k in range(5)]
        # stability: per-year rates for the main groups
        yrs = []
        for y in sorted(set(ctx.ev_year)):
            yb = base & (ctx.ev_year == y)
            if yb.sum() < 30:
                continue
            yrs.append({"year": int(y), "all": _rate(succ, fwd, yb)["rate"], "n": int(yb.sum()),
                        "tight": _rate(succ, fwd, yb & t1).get("rate"), "higher_low": _rate(succ, fwd, yb & h1).get("rate"),
                        "both": _rate(succ, fwd, yb & t1 & h1).get("rate"),
                        "vol_low": _rate(succ, fwd, yb & (bins == 0)).get("rate") if bins is not None else None})
        m["per_year"] = yrs
        m["base_q"] = q
        out["modes"][mode] = m
        log.info("patterns %s: all=%s tight=%s higher_low=%s both=%s vol_low=%s", mode, m["all"].get("rate"),
                 m["tight"].get("rate"), m["higher_low"].get("rate"), m["both"].get("rate"), m["vol_quintiles"][0].get("rate"))
    return out


def compute(fs: FactorSet, scores: dict, store=None) -> dict | None:
    if "score" not in scores:
        return None
    ctx = _Ctx(fs, scores)
    n_ev = len(ctx.ev_r)
    log.info("breakout: %d fresh 1-year-high breakouts with a full chip window", n_ev)
    if n_ev < 500:
        return {"status": "insufficient", "n_events": int(n_ev)}
    opt = _optimize(ctx, store)
    T = ctx.T
    t = T - 1
    # today's candidates: recent breakouts and stocks just below their 1-year high
    recent_rows = np.arange(max(0, T - RECENT_EVENT_DAYS), T)
    cand = {}
    for r in recent_rows[::-1]:
        for c in np.where(ctx.event[r])[0]:
            cand.setdefault(int(c), int(t - r))
    with np.errstate(invalid="ignore"):
        hi_now = np.fmax(ctx.prior_hi[t], ctx.pv[t])
        gap = 1 - ctx.pv[t] / hi_now
    has_t = scores["score"].notna().values[t]
    near = np.where(ctx.liq[t] & has_t & (gap > 0) & (gap <= NEAR_PCT))[0]
    for c in near:
        cand.setdefault(int(c), -1)                       # -1 = not yet broken out
    cols = np.array(sorted(cand), dtype=int)
    res = {"status": "ok", "n_events": int(n_ev), "params": {"high_lookback": HIGH_LOOKBACK, "fresh_days": FRESH_DAYS,
           "recent_days": RECENT_EVENT_DAYS, "near_pct": NEAR_PCT}, "optimization": opt, "modes": {}}
    today = {int(c): {"d": cand[int(c)], "g": round(float(gap[c]) * 100, 2)} for c in cols}
    for mode, (h, q) in MODES.items():
        mo = (opt.get("modes") or {}).get(mode)
        L = mo["best"] if mo else LATE
        v = _validate(ctx, mode, L)
        keys, X = ctx.feats(L)
        known, succ, _ = ctx.outcome(mode)
        kn = np.where(known)[0]
        if len(kn) < 300 or np.nansum(succ[kn]) < 30:
            res["modes"][mode] = {"name": MODE_NAMES[mode], "L": L, "status": "insufficient"}
            continue
        model = _fit(X[kn[succ[kn] == 1]], X[kn])
        s_hist, _ = _score(X[kn], model)
        w = model[2]
        order = np.argsort(-np.abs(w))[:12]
        traits = [{"name": _label(keys[k], fs), "d": round(float(w[k]), 2)} for k in order if w[k] != 0]
        mres = {"name": MODE_NAMES[mode], "h": h, "q": q, "L": L, "validation": v, "valid": v.get("valid", False),
                "traits": traits, "base_rate_all": round(float(np.nanmean(succ[kn])), 3), "status": "ok"}
        res["modes"][mode] = mres
        if len(cols):
            # breakouts: window ends the day before the breakout (as in history); near-highs: ends today
            rows_c = np.array([t - cand[int(c)] if cand[int(c)] >= 0 else t + 1 for c in cols])
            _, Xc = _features(ctx.wins, rows_c, cols, L)
            s_c, contrib = _score(Xc, model)
            pct = np.searchsorted(np.sort(s_hist), s_c) / len(s_hist)
            quint = v.get("quintiles") or [None] * 5
            for j, c in enumerate(cols):
                k5 = min(int(pct[j] * 5), 4)
                best = np.argsort(-contrib[j])[:3]
                today[int(c)][mode] = {"p": round(float(pct[j]) * 100, 1), "sr": quint[k5],
                                       "m": [_label(keys[k], fs) + ("偏高" if w[k] > 0 else "偏低")
                                             for k in best if contrib[j, k] > 0]}
        log.info("breakout %s: L=%d lift=%s excess=%s valid=%s", mode, L, v.get("lift"), v.get("excess"), v.get("valid"))
    # price patterns for today's candidates (breakouts: window ends the day before the breakout)
    for c in list(today):
        row = t - today[c]["d"] if today[c]["d"] >= 0 else t + 1
        if row < T:
            rr_, vr_, hl_ = ctx.range_ratio[row, c], ctx.vol_ratio[row, c], ctx.higher_low[row, c]
        else:   # near-highs: window ending today = what a breakout tomorrow would see
            with np.errstate(all="ignore"):
                p = ctx.pv[:, c]
                s22, s123 = p[T - PAT_SHORT:T], p[T - PAT_LONG:T]
                rr_ = (np.nanmax(s22) - np.nanmin(s22)) / (np.nanmax(s123) - np.nanmin(s123))
                lr = np.diff(np.log(p[T - PAT_LONG - 1:T]))
                vr_ = np.nanstd(lr[-PAT_SHORT:], ddof=1) / np.nanstd(lr, ddof=1)
                hl_ = float(np.nanmin(s22) > np.nanmin(p[T - PAT_LONG:T - PAT_SHORT]))
        today[c]["rr"] = round(float(rr_), 2) if np.isfinite(rr_) else None
        today[c]["vr"] = round(float(vr_), 2) if np.isfinite(vr_) else None
        today[c]["hl"] = bool(hl_ == 1) if np.isfinite(hl_) else None
    try:
        res["patterns"] = patterns(ctx)
    except Exception:
        log.exception("breakout pattern study failed")
    res["today"] = [{"c": str(fs.codes[c]), **today[c]} for c in today]
    return res
