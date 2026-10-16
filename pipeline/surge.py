"""「飆股相似度」: how much does each stock's recent chip path look like past big winners just
after their pre-rally low?

1. Winners: every calendar year, the top TOP_PCT% liquid stocks by that year's return.
2. Anchor: the winner's pre-rally low (lowest close between ~3 months before the year and the
   year's peak). The profile window runs from PRE trading days before the low to POST days after it;
   all features are measured at the window's END (low + POST), using only data inside the window.
3. Features (all cross-sectional percentiles 0-100, so markets and years are comparable):
   for the composite score, each sub-score and each scoring factor:
     mean over the pre-low part, mean over the post-low part, and their difference;
   price itself is left out on purpose (winners are picked by price; it would swamp the chip traits).
4. What makes winners different: for each feature, (winner mean - all-stocks mean) / all-stocks std,
   measured on random liquid stock-dates. Features with |difference| >= 0.1 std get that weight.
5. Similarity today = weighted, standardized agreement with the winner direction, shown as a
   percentile (0-100) across today's liquid stocks, plus the traits contributing most.
6. Validation (walk-forward, no look-ahead): profile from winners of earlier years only, then on
   monthly dates of the next year take the 50 most similar stocks and check how often they become
   top-TOP_PCT% performers over the following 252 trading days, vs the TOP_PCT% base rate.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import log
from .factors import FactorSet, xrank

CONFIG = Path(__file__).resolve().parent.parent / "config" / "surge.json"
DEFAULTS = {"pre": 89, "post": 34, "top_pct": 10}
LIMITS = {"pre": (10, 250), "post": (0, 120), "top_pct": (1, 30)}
HOLD = 252           # forward horizon used in validation
TOPN = 50            # list size on the dashboard / in validation
MIN_D = 0.1          # minimum standardized difference for a feature to count
GROUPS = {"inst": "法人", "holders": "大戶散戶", "margin": "信用", "short": "放空借券", "insider": "內部人"}


def params() -> dict:
    p = dict(DEFAULTS)
    try:
        p.update({k: int(v) for k, v in json.loads(CONFIG.read_text()).items() if k in DEFAULTS})
    except Exception:
        pass
    for k, (lo, hi) in LIMITS.items():
        p[k] = int(min(hi, max(lo, p[k])))
    return p


def _series(fs: FactorSet, scores: dict) -> dict[str, pd.DataFrame]:
    out = {"score": scores["score"]}
    out.update({k: v for k, v in scores.items() if k.startswith("s_")})
    for g, items in fs.score_spec.items():
        for name, sign in items:
            if name in fs.factors:
                out[f"f_{name}"] = (xrank(fs.factors[name] * sign) * 100).astype("float32")
    return out


def _label(key: str, fs: FactorSet) -> str:
    seg, name = key.split(":", 1)
    if name == "score":
        base = "籌碼綜合分數"
    elif name.startswith("s_"):
        base = GROUPS.get(name[2:], name[2:]) + "子分數"
    elif name == "price":
        base = "股價漲跌"
    else:
        base = fs.labels.get(name[2:], name[2:])
    return {"pre": "低點前", "post": "低點後", "chg": "低點前後變化"}[seg] + "・" + base


class _Win:
    """Window means via cumulative sums, extracted only at the (row, col) points needed."""

    def __init__(self, df: pd.DataFrame):
        v = df.values
        ok = ~np.isnan(v)
        # float32 / int16 keep memory at ~6 bytes per stock-day (values are 0-100 percentiles)
        self.cs = np.vstack([np.zeros((1, v.shape[1]), "float32"),
                             np.cumsum(np.where(ok, v, 0), axis=0, dtype="float32")])
        self.cn = np.vstack([np.zeros((1, v.shape[1]), "int16"), np.cumsum(ok, axis=0, dtype="int16")])

    def mean(self, a: np.ndarray, b: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Mean over rows [a, b] inclusive, per point."""
        s = self.cs[b + 1, cols] - self.cs[a, cols]
        n = self.cn[b + 1, cols] - self.cn[a, cols]
        with np.errstate(all="ignore"):
            return np.where(n >= np.maximum(1, (b - a + 1) * 0.5), s / np.maximum(n, 1), np.nan)


def _features(wins: dict, pr: tuple, rows: np.ndarray, cols: np.ndarray, pre: int, post: int):
    """Feature matrix for window ends at (rows, cols): window = [end-post-pre, end]; low = end-post."""
    low = rows - post
    a = low - pre
    keys, mats = [], []
    for name, w in wins.items():
        m_pre = w.mean(a, low, cols)
        m_post = w.mean(low, rows, cols) if post > 0 else m_pre
        keys += [f"pre:{name}", f"post:{name}", f"chg:{name}"]
        mats += [m_pre, m_post, m_post - m_pre]
    # Price is deliberately NOT a feature: winners are chosen by price, so post-low price strength
    # would dominate and turn this into a momentum screen. The similarity is about chips only.
    return keys, np.column_stack(mats).astype("float32")


def _fit(Xw: np.ndarray, Xa: np.ndarray):
    mu, sd = np.nanmean(Xa, axis=0), np.nanstd(Xa, axis=0)
    sd = np.where(sd > 1e-6, sd, np.nan)
    d = (np.nanmean(Xw, axis=0) - mu) / sd
    w = np.where(np.abs(d) >= MIN_D, d, 0.0)
    w = np.nan_to_num(w)
    return mu, sd, w


def _score(X: np.ndarray, model) -> tuple[np.ndarray, np.ndarray]:
    mu, sd, w = model
    z = np.nan_to_num((X - mu) / sd)
    contrib = z * w
    tot = np.abs(w).sum() or 1.0
    return contrib.sum(axis=1) / tot, contrib


def compute(fs: FactorSet, scores: dict) -> dict | None:
    p = params()
    pre, post, top = p["pre"], p["post"], p["top_pct"] / 100
    if "score" not in scores:
        return None
    cal, codes, px = fs.calendar, fs.codes, fs.price
    T = len(cal)
    liq = (fs.liquid if fs.liquid is not None else pd.DataFrame(True, index=cal, columns=codes)).fillna(False).values
    has_score = scores["score"].notna().values
    first = int(np.argmax(has_score.any(axis=1)))          # chip history starts here
    wins = {k: _Win(v) for k, v in _series(fs, scores).items()}
    pv = px.values
    with np.errstate(all="ignore"):
        pr = ((xrank(px.shift(post) / px.shift(post + pre) - 1) * 100).values.astype("float32"),
              (xrank(px / px.shift(post) - 1) * 100).values.astype("float32") if post > 0
              else np.full(px.shape, 50, "float32"))

    # --- winners and their anchors (end of profile window = low + post) ---
    years = sorted(set(cal.year))
    win_rows, win_cols, win_year = [], [], []
    for y in years:
        ix = np.where(cal.year == y)[0]
        if len(ix) < 200:
            continue
        s, e = ix[0], ix[-1]
        live = liq[s:e + 1].mean(axis=0) >= 0.5
        with np.errstate(all="ignore"):
            ret = pv[e] / pv[s] - 1
        ok = live & np.isfinite(ret)
        if ok.sum() < 30:
            continue
        thr = np.nanquantile(ret[ok], 1 - top)
        for c in np.where(ok & (ret >= thr))[0]:
            seg = pv[s:e + 1, c]
            if np.isnan(seg).all():
                continue
            peak = s + int(np.nanargmax(seg))
            lo_s = max(0, s - 63)
            low = lo_s + int(np.nanargmin(pv[lo_s:peak + 1, c])) if peak >= lo_s else s
            end = low + post
            if end >= T or low - pre < first:
                continue
            win_rows.append(end); win_cols.append(c); win_year.append(y)
    win_rows, win_cols, win_year = map(np.array, (win_rows, win_cols, win_year))
    if len(win_rows) < 30:
        log.info("surge: only %d winner profiles (history too short) - skipped", len(win_rows))
        return {"params": p, "n_winners": int(len(win_rows)), "status": "insufficient"}

    # --- reference: liquid stock-dates every 10 days ---
    ref_r = np.arange(first + pre + post, T, 10)
    rr, cc = np.where(liq[ref_r] & has_score[ref_r])
    ref_rows, ref_cols = ref_r[rr], cc
    if len(ref_rows) > 300_000:
        pick = np.random.default_rng(0).choice(len(ref_rows), 300_000, replace=False)
        ref_rows, ref_cols = ref_rows[pick], ref_cols[pick]

    keys, Xw = _features(wins, pr, win_rows, win_cols, pre, post)
    _, Xa = _features(wins, pr, ref_rows, ref_cols, pre, post)

    # --- walk-forward validation ---
    fwd_ok = T - HOLD
    per_year, hits, n_dates, exc63, exc252 = [], [], 0, [], []
    for y in years:
        test = np.where((cal.year == y) & (np.arange(T) < fwd_ok) & (np.arange(T) >= first + pre + post))[0][::21]
        trw = (win_year < y) & (cal[win_rows].year < y)    # earlier winners whose window ended before the test year
        tra = cal[ref_rows] < pd.Timestamp(f"{y}-01-01")
        if len(test) == 0 or trw.sum() < 30 or tra.sum() < 1000:
            continue
        model = _fit(Xw[trw], Xa[tra])
        yh, yn = [], 0
        for t in test:
            cols = np.where(liq[t] & has_score[t])[0]
            if len(cols) < TOPN * 3:
                continue
            _, X = _features(wins, pr, np.full(len(cols), t), cols, pre, post)
            s_, _ = _score(X, model)
            with np.errstate(all="ignore"):
                f252 = pv[t + HOLD, cols] / pv[t, cols] - 1
                f63 = pv[min(T - 1, t + 63), cols] / pv[t, cols] - 1
            okf = np.isfinite(f252)
            if okf.sum() < TOPN * 3:
                continue
            thr = np.nanquantile(f252[okf], 1 - top)
            sel = np.argsort(-np.nan_to_num(s_, nan=-9))[:TOPN]
            sel = sel[okf[sel]]
            yh.append(float((f252[sel] >= thr).mean()))
            exc252.append(float(np.nanmean(np.clip(f252[sel], -0.9, 5)) - np.nanmean(np.clip(f252[okf], -0.9, 5))) * 100)
            exc63.append(float(np.nanmean(np.clip(f63[sel], -0.9, 3)) - np.nanmean(np.clip(f63[np.isfinite(f63)], -0.9, 3))) * 100)
            yn += 1
        if yn:
            per_year.append({"year": int(y), "dates": yn, "precision": round(float(np.mean(yh)), 3)})
            hits += yh
            n_dates += yn
    val = {"base": round(top, 3), "dates": n_dates}
    if n_dates:
        val.update({"precision": round(float(np.mean(hits)), 3), "lift": round(float(np.mean(hits)) / top, 2),
                    "excess_63": round(float(np.mean(exc63)), 2), "excess_252": round(float(np.mean(exc252)), 2),
                    "per_year": per_year})
    valid = bool(n_dates >= 12 and val.get("lift", 0) >= 1.3 and val.get("excess_252", -1) > 0)

    # --- today's similarity with all winners ---
    model = _fit(Xw, Xa)
    t = T - 1
    cols = np.where(liq[t] & has_score[t])[0]
    today = {}
    if len(cols):
        _, X = _features(wins, pr, np.full(len(cols), t), cols, pre, post)
        s_, contrib = _score(X, model)
        pct = pd.Series(s_).rank(pct=True).values * 100
        for j, c in enumerate(cols):
            best = np.argsort(-contrib[j])[:3]
            today[codes[c]] = {"sv": round(float(pct[j]), 1),
                               "sm": [_label(keys[k], fs) for k in best if contrib[j, k] > 0]}
    w = model[2]
    order = np.argsort(-np.abs(w))[:12]
    traits = [{"name": _label(keys[k], fs), "d": round(float(w[k]), 2)} for k in order if w[k] != 0]
    log.info("surge: %d winner profiles, validation precision=%s lift=%s valid=%s",
             len(win_rows), val.get("precision"), val.get("lift"), valid)
    return {"params": p, "n_winners": int(len(win_rows)), "years": [int(min(win_year)), int(max(win_year))],
            "validation": val, "valid": valid, "traits": traits, "today": today, "status": "ok"}
