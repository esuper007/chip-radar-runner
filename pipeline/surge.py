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
7. Window optimization (per market, once per quarter): every split of PRE + POST = 123 days in
   steps of 17 (123/0, 106/17, 89/34, ... 4/119) is validated as above; the split with the best
   lift becomes that market's default until the next quarter end. Because picking the best of 8 on
   the same history flatters it, an honest check is reported too: for each test year, the split
   that was best on EARLIER years only is used, and its result in that year is recorded.
   A manual override from the dashboard (config/surge.json "manual") applies to all markets.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .common import log
from .factors import FactorSet, xrank

CONFIG = Path(__file__).resolve().parent.parent / "config" / "surge.json"
DEFAULTS = {"pre": 89, "post": 34, "top_pct": 10}
LIMITS = {"pre": (0, 250), "post": (0, 120), "top_pct": (1, 30)}
GRID_TOTAL, GRID_STEP = 123, 17
GRID = [(GRID_TOTAL - post, post) for post in range(0, GRID_TOTAL + 1, GRID_STEP)]   # (pre, post)
OPT_META = "surge_opt"
HOLD = 252           # forward horizon used in validation
TOPN = 50            # list size on the dashboard / in validation
MIN_D = 0.1          # minimum standardized difference for a feature to count
GROUPS = {"inst": "法人", "holders": "大戶散戶", "margin": "信用", "short": "放空借券", "insider": "內部人"}


def _config() -> dict:
    try:
        return json.loads(CONFIG.read_text())
    except Exception:
        return {}


def _clip(p: dict) -> dict:
    for k, (lo, hi) in LIMITS.items():
        p[k] = int(min(hi, max(lo, int(p[k]))))
    return p


def params() -> dict:
    """Global settings: top_pct, plus pre/post only when set manually from the dashboard."""
    cfg = _config()
    p = dict(DEFAULTS, source="default")
    if "top_pct" in cfg:
        p["top_pct"] = int(cfg["top_pct"])
    man = cfg.get("manual")
    if not man and "pre" in cfg and "post" in cfg:        # legacy flat format = manual setting
        man = {"pre": cfg["pre"], "post": cfg["post"]}
    if man:
        p.update(pre=int(man["pre"]), post=int(man["post"]), source="manual")
    return _clip(p)


def quarter_end(day) -> str:
    """Last completed quarter end on or before `day` (a quarter's last day counts as completed)."""
    d = pd.Timestamp(day)
    q = pd.Timestamp(year=d.year, month=3 * ((d.month - 1) // 3) + 1, day=1) - pd.Timedelta(days=1)
    nxt = q + pd.offsets.QuarterEnd(1)
    return str((nxt if d >= nxt else q).date())


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
    # A feature only counts if most winners AND most reference stock-dates have it: data that
    # exists only for recent years (e.g. CN main-force flow) would otherwise compare recent
    # winners against a different era and produce a spurious difference.
    cov = np.minimum((~np.isnan(Xw)).mean(axis=0), (~np.isnan(Xa)).mean(axis=0))
    w = np.where((np.abs(d) >= MIN_D) & (cov >= 0.5), d, 0.0)
    w = np.nan_to_num(w)
    return mu, sd, w


def _score(X: np.ndarray, model) -> tuple[np.ndarray, np.ndarray]:
    mu, sd, w = model
    z = np.nan_to_num((X - mu) / sd)
    contrib = z * w
    tot = np.abs(w).sum() or 1.0
    return contrib.sum(axis=1) / tot, contrib


def compute(fs: FactorSet, scores: dict, store=None) -> dict | None:
    """Pick the window (manual, or this market's quarterly optimum) and compute the similarity."""
    if "score" not in scores:
        return None
    p = params()
    opt = None
    if p["source"] != "manual":
        opt = optimize(fs, scores, store, p["top_pct"])
        if opt and opt.get("best"):
            p.update(pre=opt["best"]["pre"], post=opt["best"]["post"], source="optimized")
    res = _run(fs, scores, p)
    if res is not None and opt:
        res["optimization"] = {k: v for k, v in opt.items() if k != "top_pct"}
    return res


def optimize(fs: FactorSet, scores: dict, store, top_pct: int) -> dict | None:
    """Best PRE/POST split on the 123-day grid; cached per quarter in the market's data store."""
    asof = fs.calendar[-1]
    q = quarter_end(asof)
    cached = store.load_meta(OPT_META) if store is not None else None
    force = os.environ.get("SURGE_OPTIMIZE") == "1"
    if (cached and not force and cached.get("quarter") == q and cached.get("top_pct") == top_pct
            and cached.get("grid_total") == GRID_TOTAL and cached.get("grid_step") == GRID_STEP):
        log.info("surge: using window optimized for quarter %s: %s", q, cached.get("best"))
        return cached
    log.info("surge: optimizing window for quarter %s over %d splits", q, len(GRID))
    rows, detail = [], {}
    for pre, post in GRID:
        r = _run(fs, scores, dict(pre=pre, post=post, top_pct=top_pct), today=False)
        v = (r or {}).get("validation") or {}
        if not v.get("dates"):
            continue
        rows.append({"pre": pre, "post": post, "precision": v["precision"], "lift": v["lift"],
                     "excess_252": v["excess_252"], "dates": v["dates"], "valid": r["valid"]})
        detail[(pre, post)] = {py["year"]: py for py in v["per_year"]}
        log.info("surge grid %d/%d: lift=%s excess252=%s", pre, post, v["lift"], v["excess_252"])
    if not rows:
        log.info("surge: not enough history to optimize; keeping the default window")
        return None
    best = max(rows, key=lambda x: (x["lift"], x["excess_252"]))
    # honest check: in each year use the split that was best on earlier years only
    years = sorted({y for d in detail.values() for y in d})
    picks, hit_sum, exc_sum, n_sum = [], 0.0, 0.0, 0
    for y in years:
        prior = {}
        for k, d in detail.items():
            ys = [d[x] for x in d if x < y]
            n = sum(x["dates"] for x in ys)
            if len(ys) >= 2 and n:
                prior[k] = sum(x["precision"] * x["dates"] for x in ys) / n
        if not prior or y not in detail[max(prior, key=prior.get)]:
            continue
        k = max(prior, key=prior.get)
        yr = detail[k][y]
        picks.append({"year": int(y), "pre": k[0], "post": k[1], "precision": yr["precision"]})
        hit_sum += yr["precision"] * yr["dates"]
        exc_sum += yr.get("excess_252", 0) * yr["dates"]
        n_sum += yr["dates"]
    honest = None
    if n_sum:
        prec = hit_sum / n_sum
        honest = {"precision": round(prec, 3), "lift": round(prec / (top_pct / 100), 2),
                  "excess_252": round(exc_sum / n_sum, 2), "dates": n_sum, "picks": picks}
        honest["valid"] = bool(n_sum >= 12 and honest["lift"] >= 1.3 and honest["excess_252"] > 0)
    out = {"quarter": q, "optimized_at": str(pd.Timestamp(asof).date()), "top_pct": top_pct,
           "grid_total": GRID_TOTAL, "grid_step": GRID_STEP,
           "best": {"pre": best["pre"], "post": best["post"]}, "grid": rows, "honest": honest}
    log.info("surge: best window pre=%s post=%s lift=%s; honest lift=%s", best["pre"], best["post"],
             best["lift"], honest and honest["lift"])
    if store is not None:
        try:
            store.save_meta(OPT_META, out)
        except Exception as e:
            log.warning("could not save surge optimum: %s", e)
    return out


def _run(fs: FactorSet, scores: dict, p: dict, today: bool = True) -> dict | None:
    pre, post, top = p["pre"], p["post"], p["top_pct"] / 100
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
        p = {k: p[k] for k in ("pre", "post", "top_pct", "source") if k in p}
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
    y_exc: list[float] = []
    for y in years:
        test = np.where((cal.year == y) & (np.arange(T) < fwd_ok) & (np.arange(T) >= first + pre + post))[0][::21]
        trw = (win_year < y) & (cal[win_rows].year < y)    # earlier winners whose window ended before the test year
        tra = cal[ref_rows] < pd.Timestamp(f"{y}-01-01")
        if len(test) == 0 or trw.sum() < 30 or tra.sum() < 1000:
            continue
        model = _fit(Xw[trw], Xa[tra])
        yh, yn = [], 0
        y_exc = []
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
            y_exc.append(exc252[-1])
            exc63.append(float(np.nanmean(np.clip(f63[sel], -0.9, 3)) - np.nanmean(np.clip(f63[np.isfinite(f63)], -0.9, 3))) * 100)
            yn += 1
        if yn:
            per_year.append({"year": int(y), "dates": yn, "precision": round(float(np.mean(yh)), 3),
                             "excess_252": round(float(np.mean(y_exc)), 2)})
            hits += yh
            n_dates += yn
    val = {"base": round(top, 3), "dates": n_dates}
    if n_dates:
        val.update({"precision": round(float(np.mean(hits)), 3), "lift": round(float(np.mean(hits)) / top, 2),
                    "excess_63": round(float(np.mean(exc63)), 2), "excess_252": round(float(np.mean(exc252)), 2),
                    "per_year": per_year})
    valid = bool(n_dates >= 12 and val.get("lift", 0) >= 1.3 and val.get("excess_252", -1) > 0)
    if not today:
        return {"params": p, "validation": val, "valid": valid}

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
    p = {k: p[k] for k in ("pre", "post", "top_pct", "source") if k in p}
    return {"params": p, "n_winners": int(len(win_rows)), "years": [int(min(win_year)), int(max(win_year))],
            "validation": val, "valid": valid, "traits": traits, "today": today, "status": "ok"}
