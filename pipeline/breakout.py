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
from .factors import FDR_Q, FactorSet, bh_qvalues, exit_price, sign_test_p, xrank
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

    def __init__(self, fs: FactorSet, scores: dict, store=None):
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
        ev_ = exit_price(px).values.astype("float64")     # delisted stocks keep their last traded price
        for h in {h for h, _ in MODES.values()}:
            with np.errstate(all="ignore"):
                f = ev_[h:] / pv[:-h] - 1 if h < self.T else np.empty((0, pv.shape[1]))
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
        self._price_features(fs, store)

    TAIL = 6       # keep the last rows (recent breakouts + "tomorrow" for near-highs) of every price feature

    def _price_features(self, fs: FactorSet, store) -> None:
        """Price / volume features as of the day BEFORE each row, kept only at event points and the last rows.
        Row T (one past the data) = what a breakout tomorrow would see; used for stocks near their high."""
        cal, px = fs.calendar, fs.price
        T = self.T
        ext = cal.append(pd.DatetimeIndex([cal[-1] + pd.Timedelta(days=1)]))
        pe = px.reindex(ext)
        p1 = pe.shift(1)
        liq_e = np.vstack([self.liq, self.liq[-1:]])
        vol = None
        if store is not None:
            try:
                from .factors import wide
                vol = wide(store.load("price"), "volume", cal, fs.codes).reindex(ext)
            except Exception as e:
                log.warning("breakout: volume unavailable (%s)", e)
        self.pf_ev, self.pf_tail = {}, {}
        r, c = self.ev_r, self.ev_c

        def keep(name, frame):
            a = frame.values if hasattr(frame, "values") else frame
            a = np.asarray(a, dtype="float32")
            self.pf_ev[name] = a[r, c]
            self.pf_tail[name] = a[T + 1 - self.TAIL:T + 1].copy()

        with np.errstate(all="ignore"):
            mx = lambda n: p1.rolling(n, min_periods=n).max()
            mn = lambda n: p1.rolling(n, min_periods=n).min()
            keep("rr", (mx(PAT_SHORT) - mn(PAT_SHORT)) / (mx(PAT_LONG) - mn(PAT_LONG)))
            lr = np.log(p1 / p1.shift(1))
            keep("vr", lr.rolling(PAT_SHORT, min_periods=PAT_SHORT - 2).std() / lr.rolling(PAT_LONG, min_periods=PAT_LONG - 5).std())
            lo_r, lo_b = mn(PAT_SHORT), pe.shift(PAT_SHORT + 1).rolling(PAT_LONG - PAT_SHORT, min_periods=PAT_LONG - PAT_SHORT).min()
            keep("hl", np.where(np.isfinite(lo_r.values) & np.isfinite(lo_b.values), (lo_r.values > lo_b.values), np.nan))
            ma = {n: p1.rolling(n, min_periods=int(n * 0.9)).mean() for n in (50, 150, 200)}
            trend = (p1 > ma[50]) & (ma[50] > ma[150]) & (ma[150] > ma[200]) & (ma[200] > ma[200].shift(22))
            keep("trend", np.where(ma[200].notna().values & ma[200].shift(22).notna().values, trend.values, np.nan))
            keep("d50", p1 / ma[50] - 1)
            keep("d200", p1 / ma[200] - 1)
            keep("ret20", p1 / p1.shift(20) - 1)
            keep("ret60", p1 / p1.shift(60) - 1)
            r250 = (p1 / p1.shift(250) - 1).where(liq_e)
            keep("rs", r250.rank(axis=1, pct=True))
            # market: equal-weighted index of liquid stocks vs its 200-day average, and breadth
            dr = (pe / pe.shift(1) - 1).where(np.vstack([np.zeros((1, liq_e.shape[1]), bool), liq_e[:-1]]))
            idx = (1 + dr.clip(-0.5, 1).mean(axis=1).fillna(0)).cumprod()
            reg = (idx > idx.rolling(200, min_periods=150).mean()).astype("float32").shift(1)
            above = (p1 > ma[200]).where(liq_e & ma[200].notna().values)
            br = above.mean(axis=1).shift(0)
            keep("regime", np.repeat(reg.values[:, None], pe.shape[1], axis=1))
            keep("breadth", np.repeat(br.values[:, None], pe.shape[1], axis=1))
            if vol is not None and vol.notna().values.any():
                v = vol.where(vol > 0)
                keep("vs", v / v.shift(1).rolling(50, min_periods=40).mean())          # breakout-day volume
                chg = (pe - pe.shift(1)).shift(1)               # price change of the day before each row
                v1 = v.shift(1)
                okd = v1.notna() & chg.notna()
                upv = (v1 * (chg > 0)).where(okd).rolling(50, min_periods=40).sum()
                dnv = (v1 * (chg < 0)).where(okd).rolling(50, min_periods=40).sum()
                keep("ud", upv / dnv.where(dnv > 0))
        # TW: foreign net open interest in TAIEX futures, percentile within its past year (known the day before)
        fut = fs.market_series.get("外資台指期淨未平倉(口)") if getattr(fs, "market_series", None) else None
        if fut is not None and len(fut.dropna()) > 120:
            z = fut.reindex(ext).ffill().rolling(250, min_periods=120).rank(pct=True).shift(1)
            keep("fut", np.repeat(z.values[:, None], pe.shape[1], axis=1))
        # industry heat: other fresh breakouts in the same industry over the last 10 sessions
        ind = None
        try:
            ind = fs.meta.reindex(fs.codes)["industry"].fillna("").astype(str).values
        except Exception:
            pass
        if ind is not None and (ind != "").any():
            ev = pd.DataFrame(self.event.astype("float32")).rolling(10, min_periods=1).sum().values
            g = pd.Series(ind)
            peers = np.zeros_like(ev)
            for name, cols in g.groupby(g).groups.items():
                if not name:
                    continue
                cols = np.asarray(list(cols))
                peers[:, cols] = ev[:, cols].sum(axis=1, keepdims=True) - ev[:, cols]
            peers = np.vstack([peers, peers[-1:]])
            keep("peers", peers)

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


FLAGS = {   # fixed rules (no fitting); name -> (label, function of the feature dict)
    "tight": ("波動收斂（22 日區間 < 123 日區間的 40%）", lambda f: f["rr"] < RANGE_MAX),
    "higher_low": ("低點墊高", lambda f: f["hl"] == 1),
    "vol_surge": ("突破日量 ≥ 50 日均量 1.5 倍", lambda f: f["vs"] >= 1.5),
    "accum": ("前 50 日上漲量 > 下跌量", lambda f: f["ud"] > 1),
    "rs70": ("一年漲幅排名前 30%", lambda f: f["rs"] >= 0.7),
    "trend": ("趨勢模板（股價 > 50 > 150 > 200 日線且 200 日線上升）", lambda f: f["trend"] == 1),
    "regime": ("大盤在 200 日均線之上", lambda f: f["regime"] == 1),
    "breadth": ("過半股票在 200 日均線之上", lambda f: f["breadth"] >= 0.5),
    "peers": ("同產業 10 日內有 2 檔以上也突破", lambda f: f["peers"] >= 2),
    "fut_long": ("外資台指期淨部位位於一年內較偏多的一半", lambda f: f["fut"] >= 0.5),
}
NEEDS = {"tight": "rr", "higher_low": "hl", "vol_surge": "vs", "accum": "ud", "rs70": "rs", "trend": "trend",
         "regime": "regime", "breadth": "breadth", "peers": "peers", "fut_long": "fut"}


def _flag(ctx: _Ctx, key: str, src: dict | None = None):
    f = ctx.pf_ev if src is None else src
    col = NEEDS[key]
    if col not in f:
        return None
    x = f[col]
    with np.errstate(invalid="ignore"):
        return np.where(np.isfinite(x), FLAGS[key][1](f).astype("float32"), np.nan)


def patterns(ctx: _Ctx) -> dict:
    """Fixed price / volume / market rules: success rate of breakouts with vs without each rule."""
    out = {"defs": {k: v[0] for k, v in FLAGS.items()}, "modes": {}}
    flags = {k: _flag(ctx, k) for k in FLAGS}
    flags = {k: v for k, v in flags.items() if v is not None}
    vr = ctx.pf_ev["vr"]
    okv = np.isfinite(vr)
    vq = np.nanquantile(vr[okv], [0.2, 0.4, 0.6, 0.8]) if okv.sum() > 100 else None
    years = sorted(set(ctx.ev_year))
    for mode, (h, q) in MODES.items():
        known, succ, fwd = ctx.outcome(mode)
        if known.sum() < 200:
            continue
        m = {"all": _rate(succ, fwd, known), "base_q": q, "rules": {}}
        for k, fl in flags.items():
            has = known & np.isfinite(fl)
            yes, no = has & (fl == 1), has & (fl == 0)
            yr = []
            for y in years:
                yb = ctx.ev_year == y
                a, b = _rate(succ, fwd, yes & yb), _rate(succ, fwd, no & yb)
                if a.get("n", 0) >= 20 and b.get("n", 0) >= 20:
                    yr.append({"year": int(y), "yes": a["rate"], "no": b["rate"], "n_yes": a["n"]})
            wins = sum(1 for x in yr if x["yes"] > x["no"])
            m["rules"][k] = {"yes": _rate(succ, fwd, yes), "no": _rate(succ, fwd, no),
                             "share": round(float(yes.sum() / max(has.sum(), 1)), 3),
                             "years_better": wins, "years": len(yr), "per_year": yr,
                             # chance of being better in this many years by pure luck (sign test)
                             "p": round(sign_test_p(wins, len(yr)), 4) if yr else None}
        if "tight" in flags and "higher_low" in flags:
            t1, h1 = flags["tight"] == 1, flags["higher_low"] == 1
            m["combo"] = {"both": _rate(succ, fwd, known & t1 & h1), "tight_only": _rate(succ, fwd, known & t1 & ~h1),
                          "hl_only": _rate(succ, fwd, known & ~t1 & h1), "neither": _rate(succ, fwd, known & ~t1 & ~h1)}
        if vq is not None:
            bins = np.digitize(vr, vq)
            m["vol_quintiles"] = [dict(_rate(succ, fwd, known & okv & (bins == k)),
                                       upto=round(float(vq[k]), 2) if k < 4 else None) for k in range(5)]
        out["modes"][mode] = m
        log.info("patterns %s: all=%s %s", mode, m["all"].get("rate"),
                 {k: (v["yes"].get("rate"), v["no"].get("rate"), f'{v["years_better"]}/{v["years"]}') for k, v in m["rules"].items()})
    # multiple testing: every rule x success definition of this market is one test (Benjamini-Hochberg)
    pv = {(mode, k): r.get("p") for mode, mm in out["modes"].items() for k, r in mm["rules"].items()}
    qv = bh_qvalues(pv)
    for (mode, k), q in qv.items():
        r = out["modes"][mode]["rules"][k]
        r["q"] = q
        r["fdr_ok"] = bool(q is not None and q <= FDR_Q)
    out["fdr"] = {"q": FDR_Q, "tests": sum(1 for v in pv.values() if v is not None)}
    return out


# ---------------------------------------------------------------- LightGBM breakout model
ML_PRICE = ["rr", "vr", "hl", "vs", "ud", "rs", "trend", "d50", "d200", "ret20", "ret60", "regime", "breadth", "peers", "fut"]
ML_PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100, subsample=0.8,
                 subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, n_jobs=-1, verbose=-1)
PRICE_LABELS = {"rr": "22／123 日區間比", "vr": "22／123 日波動度比", "hl": "低點墊高", "vs": "突破日量倍數", "ud": "上漲量／下跌量",
                "rs": "一年相對強度", "trend": "趨勢模板", "d50": "距 50 日線", "d200": "距 200 日線", "ret20": "前 20 日漲幅",
                "ret60": "前 60 日漲幅", "regime": "大盤多頭", "breadth": "市場廣度", "peers": "同產業突破數",
                "fut": "外資台指期部位(一年百分位)"}


def _ml_matrix(ctx: _Ctx, L: int, rows=None, cols=None):
    """Feature matrix: chip window features (observation length L) + price/volume/market features."""
    if rows is None:
        keys, Xc = ctx.feats(L)
        pf = ctx.pf_ev
        Xp = np.column_stack([pf[k] for k in ML_PRICE if k in pf]) if pf else np.empty((len(Xc), 0))
    else:
        keys, Xc = _features(ctx.wins, rows, cols, L)
        base = ctx.T + 1 - ctx.TAIL
        Xp = np.column_stack([ctx.pf_tail[k][rows - base, cols] for k in ML_PRICE if k in ctx.pf_tail]) \
            if ctx.pf_tail else np.empty((len(rows), 0))
    names = keys + [f"px:{k}" for k in ML_PRICE if k in ctx.pf_ev]
    return names, np.column_stack([Xc, Xp]).astype("float32")


def ml_validate(ctx: _Ctx, mode: str, L: int) -> dict:
    """Walk-forward by year: train on breakouts whose outcome was known before the test year."""
    import lightgbm as lgb
    h, _ = MODES[mode]
    names, X = _ml_matrix(ctx, L)
    known, succ, fwd = ctx.outcome(mode)
    cal = ctx.fs.calendar
    per_year, scored, top_ev = [], [], []
    n_top = n_all = s_top = s_all = f_top = f_all = 0.0
    for y in sorted(set(ctx.ev_year)):
        test = (ctx.ev_year == y) & known
        y0 = np.searchsorted(cal.values, np.datetime64(f"{y}-01-01"))
        train = known & (ctx.ev_r + h < y0)
        if test.sum() < 30 or train.sum() < 1000 or np.nansum(succ[train]) < 100:
            continue
        mdl = lgb.LGBMClassifier(**ML_PARAMS).fit(X[train], succ[train].astype(int))
        te = np.where(test)[0]
        pr = mdl.predict_proba(X[te])[:, 1]
        pct = pd.Series(pr + np.random.default_rng(int(y)).random(len(pr)) * 1e-9).rank(pct=True, method="first").values
        top = pct > 1 - TOP_SHARE
        st, sa = float(np.mean(succ[te][top])), float(np.mean(succ[te]))
        ft = float(np.nanmean(np.clip(fwd[te][top], -0.9, 5))) * 100
        fa = float(np.nanmean(np.clip(fwd[te], -0.9, 5))) * 100
        per_year.append({"year": int(y), "n": int(len(te)), "top_rate": round(st, 3), "base_rate": round(sa, 3),
                         "excess": round(ft - fa, 2)})
        n_top += top.sum(); n_all += len(te); s_top += st * top.sum(); s_all += sa * len(te)
        f_top += ft * top.sum(); f_all += fa * len(te)
        scored.append(np.column_stack([np.minimum((pct * 5).astype(int), 4), succ[te]]))
        top_ev.append(te[top])
    if not per_year:
        return {"n_years": 0}
    top_rate, base_rate = s_top / max(n_top, 1), s_all / max(n_all, 1)
    b = np.vstack(scored)
    quint = [round(float(b[b[:, 0] == k, 1].mean()), 3) if (b[:, 0] == k).any() else None for k in range(5)]
    lift = top_rate / base_rate if base_rate > 0 else None
    excess = f_top / max(n_top, 1) - f_all / max(n_all, 1)
    years_better = sum(1 for p in per_year if p["top_rate"] > p["base_rate"])
    return {"n_years": len(per_year), "n_events": int(n_all), "top_rate": round(top_rate, 3),
            "base_rate": round(base_rate, 3), "lift": round(lift, 2) if lift else None, "excess": round(excess, 2),
            "quintiles": quint, "per_year": per_year, "years_better": years_better,
            "_top_events": np.concatenate(top_ev) if top_ev else np.array([], int),
            "valid": bool(len(per_year) >= 3 and (lift or 0) >= 1.3 and excess > 0)}


def ml_today(ctx: _Ctx, mode: str, L: int, rows: np.ndarray, cols: np.ndarray, quint: list):
    """Fit on every breakout with a known outcome, score today's candidates."""
    import lightgbm as lgb
    names, X = _ml_matrix(ctx, L)
    known, succ, _ = ctx.outcome(mode)
    if known.sum() < 1000 or np.nansum(succ[known]) < 100:
        return None, None
    mdl = lgb.LGBMClassifier(**ML_PARAMS).fit(X[known], succ[known].astype(int))
    hist = np.sort(mdl.predict_proba(X[known])[:, 1])
    imp = mdl.booster_.feature_importance("gain")
    order = np.argsort(-imp)[:12]

    def nice(n):
        if n.startswith("px:"):
            return PRICE_LABELS.get(n[3:], n[3:])
        return _label(n, ctx.fs)
    importance = [{"name": nice(names[k]), "gain": round(float(imp[k] / max(imp.sum(), 1e-9)) * 100, 1)} for k in order if imp[k] > 0]
    if not len(rows):
        return importance, {}
    _, Xc = _ml_matrix(ctx, L, rows, cols)
    pr = mdl.predict_proba(Xc)[:, 1]
    pct = np.searchsorted(hist, pr) / len(hist)
    out = {}
    for j, c in enumerate(cols):
        k5 = min(int(pct[j] * 5), 4)
        out[int(c)] = {"p": round(float(pct[j]) * 100, 1), "sr": (quint or [None] * 5)[k5]}
    return importance, out


def compute(fs: FactorSet, scores: dict, store=None) -> dict | None:
    if "score" not in scores:
        return None
    ctx = _Ctx(fs, scores, store)
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
    # price / volume features and fixed-rule flags for today's candidates
    rows_all = np.array([t - today[c]["d"] if today[c]["d"] >= 0 else t + 1 for c in cols], dtype=int)
    base = T + 1 - ctx.TAIL
    tail_ok = rows_all >= base
    for j, c in enumerate(cols):
        if not tail_ok[j]:
            continue
        src = {k: ctx.pf_tail[k][rows_all[j] - base, c:c + 1] for k in ctx.pf_tail}
        flags = {}
        for k in FLAGS:
            fl = _flag(ctx, k, src)
            if fl is not None and np.isfinite(fl[0]):
                flags[k] = bool(fl[0] == 1)
        today[int(c)]["f"] = flags
        for k in ("rr", "vr", "vs", "rs"):
            if k in src and np.isfinite(src[k][0]):
                today[int(c)][k] = round(float(src[k][0]), 2)
    res["_expect_picks"] = {"bo": pd.DataFrame({"date": fs.calendar[ctx.ev_r], "code": fs.codes[ctx.ev_c]})}
    try:
        res["patterns"] = patterns(ctx)
    except Exception:
        log.exception("breakout pattern study failed")
    # LightGBM breakout model (chips + price/volume + market), walk-forward validated per mode
    res["ml"] = {}
    for mode in MODES:
        try:
            L = (res["modes"].get(mode) or {}).get("L", LATE)
            v = ml_validate(ctx, mode, L)
            te = v.pop("_top_events", None)
            if mode == "h126q10" and te is not None and len(te):
                # out-of-sample top-fifth breakouts: what the live "bo_top" list should look like
                res.setdefault("_expect_picks", {})["bo_top"] = pd.DataFrame(
                    {"date": fs.calendar[ctx.ev_r[te]], "code": fs.codes[ctx.ev_c[te]]})
            ok = tail_ok
            imp, sc = ml_today(ctx, mode, L, rows_all[ok], cols[ok], v.get("quintiles"))
            res["ml"][mode] = {"L": L, "validation": v, "valid": v.get("valid", False), "importance": imp or []}
            for c, x in (sc or {}).items():
                today[c].setdefault("ml", {})[mode] = x
            log.info("breakout ML %s: lift=%s excess=%s years_better=%s/%s valid=%s", mode, v.get("lift"),
                     v.get("excess"), v.get("years_better"), v.get("n_years"), v.get("valid"))
        except Exception:
            log.exception("breakout ML %s failed", mode)
    res["today"] = [{"c": str(fs.codes[c]), **today[c]} for c in today]
    return res
