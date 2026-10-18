"""實戰紀錄: every build stores the lists the dashboard shows that day, and later measures what really
happened to them. Nothing here is backtested - only picks recorded on the day itself count.

Lists recorded each day (liquid stocks only):
  up_1q / up_1y   top 20 by blended up-probability (in/out-of-sample 70%, ML 15%, analog 15%)
  down_1q         bottom 20 by blended up-probability (expected to lag)
  gain_1q         top 20 by blended median forecast return
  sig:<key>       stocks whose rally signal fired that day
  bo              fresh 1-year-high breakouts that day (score = LightGBM half-year rank when available)
  bo_top          the best fifth of that day's breakouts by the LightGBM half-year model

Results (no transaction costs): bought at the close of the next trading day (the lists are made after the
close), return after 20 / 63 / 126 trading days, and excess return versus the
equal-weighted average of all liquid stocks over the same days. "hit" = share of picks that beat it.

Early warning: the same lists are rebuilt on the backtest (out of sample) and the live average excess
return is compared with the range history says to expect for a record of the same length (10th-90th
percentile of rolling averages). A 95% confidence interval is shown with an honest sample size: picks
made within the same holding period overlap, so the number of independent periods = pick days / horizon.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import log
from .factors import ENTRY_LAG, exit_price

DATASET = "picks"
TOPN = 20
HORIZONS = (20, 63, 126)
W = {"is": 0.70, "ml": 0.15, "an": 0.15}          # default; the backtest's quarterly blend weights replace it
W_PCT = {k: int(v * 100) for k, v in W.items()}
LIST_NAMES = {"up_1q": "1季跑贏大盤機率前 20", "up_1y": "1年跑贏大盤機率前 20", "down_1q": "1季落後大盤機率前 20（應落後）",
              "gain_1q": "1季預估超額報酬前 20", "bo": "當日突破一年高點（全部）", "bo_top": "當日突破・模型前 20%"}


# Shadow lists: hypotheses found in the factor check that are NOT used by the score. They are only recorded
# here, so the live record can confirm (or not) a direction that was noticed after looking at history.
# name -> (factor, expected direction of the top-20 by factor value: +1 beat the market, -1 lag it, label)
# The expected direction of a shadow list comes from its own backtest (the sign of its average 63-day excess
# return); the factor check ranks the whole market, and the 20 most extreme stocks need not behave the same way.
SHADOW = {
    "tw": {"short_dtc": (-1, "影子：融券回補天數最高 20"),
           "short_margin_ratio": (-1, "影子：券資比最高 20"),
           "sbl_dtc": (1, "影子：借券回補天數最高 20"),
           "sbl_sell_20": (1, "影子：借券賣出佔成交量最高 20")},
}
DOWN = {"down_1q"}       # lists expected to lag the market (a warning means they did NOT lag)


def list_dir(name: str, market: str) -> int:
    if name in DOWN:
        return -1
    if name.startswith("shadow:"):
        return SHADOW.get(market, {}).get(name[7:], (1, ""))[0]
    return 1


def shadow_picks(fs, step: int = 5) -> dict:
    """Weekly history of the shadow lists (top 20 liquid stocks by factor) for their expected ranges."""
    out = {}
    liq = fs.liquid.fillna(False) if fs.liquid is not None else None
    for f, (_, _) in SHADOW.get(fs.market, {}).items():
        x = fs.factors.get(f)
        if x is None:
            continue
        rows = []
        for d in fs.calendar[::step]:
            v = x.loc[d]
            if liq is not None:
                v = v[liq.loc[d]]
            v = v.dropna()
            if len(v) >= TOPN * 3:
                rows.append(pd.DataFrame({"date": d, "code": v.sort_values(ascending=False).head(TOPN).index}))
        if rows:
            out[f"shadow:{f}"] = pd.concat(rows, ignore_index=True)
    return out


def _blend(fc: pd.DataFrame, hk: str, stat: str, weights: dict | None = None) -> pd.Series:
    num = den = 0
    for m, w in (weights or W).items():
        w = w / 100 if weights else w
        col = f"{m}_{hk}_{stat}"
        if col in fc.columns:
            v = fc[col]
            num = num + v.fillna(0) * w
            den = den + v.notna() * w
    if isinstance(den, int):
        return pd.Series(dtype=float)
    return (num / den.where(den > 0)).dropna()


def record(st, fs, fc: pd.DataFrame, sig: dict | None, bo: dict | None, weights: dict | None = None) -> int:
    """weights: blend weights of the three models (same as the dashboard default), {is, ml, an} in %."""
    t = fs.calendar[-1]
    liq = fs.liquid.loc[t] if fs.liquid is not None else pd.Series(True, index=fs.codes)
    ok = set(liq[liq.fillna(False)].index)
    rows = []

    def add(name, codes_scores):
        for i, (c, s) in enumerate(codes_scores):
            rows.append({"date": t, "code": str(c), "list": name, "rank": i + 1,
                         "score": float(s) if s is not None and np.isfinite(s) else np.nan})

    if fc is not None and not fc.empty:
        f = fc[fc.index.isin(ok)]
        for hk, name in (("1q", "up_1q"), ("1y", "up_1y")):
            p = _blend(f, hk, "p_up", weights)
            if len(p) >= TOPN * 3:
                add(name, p.sort_values(ascending=False).head(TOPN).items())
                if hk == "1q":
                    add("down_1q", p.sort_values().head(TOPN).items())
        q = _blend(f, "1q", "q50", weights)
        if len(q) >= TOPN * 3:
            add("gain_1q", q.sort_values(ascending=False).head(TOPN).items())
    for c, ev in ((sig or {}).get("recent") or {}).items():
        if c in ok:
            for k, ago in ev.items():
                if ago == 0:
                    add(f"sig:{k}", [(c, None)])
    today = [x for x in (bo or {}).get("today", []) if x.get("d") == 0]
    if today:
        sc = lambda x: (x.get("ml") or {}).get("h126q10", {}).get("p")
        today.sort(key=lambda x: -(sc(x) if sc(x) is not None else -1))
        add("bo", [(x["c"], sc(x)) for x in today])
        scored = [x for x in today if sc(x) is not None]
        if scored:
            add("bo_top", [(x["c"], sc(x)) for x in scored if sc(x) >= 80])
    for f in SHADOW.get(fs.market, {}):
        x = fs.factors.get(f)
        if x is not None:
            v = x.loc[t].reindex(sorted(ok)).dropna()
            if len(v) >= TOPN * 3:
                add(f"shadow:{f}", v.sort_values(ascending=False).head(TOPN).items())
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    old = st.load(DATASET)
    if old is not None and not old.empty and (pd.to_datetime(old["date"]) == t).any():
        # a rebuild of the same day replaces that day's lists (they may differ, e.g. new blend weights)
        keep = old[pd.to_datetime(old["date"]) != t]
        st.rewrite(DATASET, pd.concat([keep, df], ignore_index=True))
    else:
        st.append(DATASET, df, keys=("date", "list", "code"))
    log.info("track: recorded %d picks in %d lists for %s", len(df), df["list"].nunique(), t.date())
    return len(df)


def expect_series(fs, picks: pd.DataFrame, horizons=HORIZONS) -> dict:
    """Per pick date: average excess return of that day's picks vs the equal-weighted liquid market."""
    cal, px = fs.calendar, fs.price
    T = len(cal)
    pv, ev = px.values, exit_price(px).values
    lv = (fs.liquid if fs.liquid is not None else pd.DataFrame(True, index=cal, columns=fs.codes)).fillna(False).values
    pos = pd.Series(np.arange(T), index=cal)
    col = pd.Series(np.arange(len(px.columns)), index=px.columns)
    p = picks.copy()
    p["i"] = pos.reindex(pd.to_datetime(p["date"])).values
    p["j"] = col.reindex(p["code"].astype(str)).values
    p = p.dropna(subset=["i", "j"])
    i, j = p["i"].astype(int).values, p["j"].astype(int).values
    out = {}
    for h in horizons:
        L = ENTRY_LAG
        ok = i + h + L < T
        if not ok.any():
            continue
        ii, jj = i[ok], j[ok]
        with np.errstate(all="ignore"):
            r = np.clip(ev[ii + L + h, jj] / pv[ii + L, jj] - 1, -0.9, 5)
            mk = {}
            for d in np.unique(ii):
                rr = ev[d + L + h] / pv[d + L] - 1
                rr = rr[lv[d] & np.isfinite(rr)]
                mk[d] = float(np.mean(np.clip(rr, -0.9, 5))) if len(rr) else np.nan
        e = pd.Series(r - np.array([mk[d] for d in ii]), index=ii).dropna()
        g = e.groupby(level=0).mean()
        if len(g):
            out[str(h)] = {"d": [str(cal[k].date()) for k in g.index], "v": [round(float(v), 4) for v in g.values]}
    return out


def _band(exp: dict, h: str, n_days: int, step: str) -> dict | None:
    """Range of the average excess return history gives for a record of n_days pick days."""
    e = (exp or {}).get(h)
    if not e or len(e["v"]) < 10:
        return None
    v = pd.Series(e["v"])
    k = max(1, round(n_days / 5)) if step == "week" else max(1, n_days)
    k = min(k, len(v))
    roll = v.rolling(k).mean().dropna()
    return {"mean": round(float(v.mean()) * 100, 2), "p10": round(float(roll.quantile(0.1)) * 100, 2),
            "p90": round(float(roll.quantile(0.9)) * 100, 2), "n_hist": int(len(v)), "k": int(k)}


def evaluate(st, fs, sig_names: dict | None = None, expect: dict | None = None) -> dict:
    """expect: {list: {"step": "week"|"event", "h": expect_series(...)}} from the backtest."""
    picks = st.load(DATASET)
    if picks is None or picks.empty:
        return {"since": None, "lists": {}}
    picks = picks.copy()
    picks["date"] = pd.to_datetime(picks["date"])
    cal, px = fs.calendar, fs.price
    T = len(cal)
    liq = (fs.liquid if fs.liquid is not None else pd.DataFrame(True, index=cal, columns=fs.codes)).fillna(False)
    pos = {d: i for i, d in enumerate(cal)}
    col = {c: j for j, c in enumerate(px.columns)}
    pv = px.values
    ev = exit_price(px).values          # a pick that gets delisted keeps its last traded price
    lv = liq.values
    out = {"since": str(picks["date"].min().date()), "days": int(picks["date"].nunique()), "lists": {}}
    mkt_cache = {}
    L = ENTRY_LAG      # picks are bought at the close of the next trading day

    def mkt(i, h):
        if (i, h) not in mkt_cache:
            with np.errstate(all="ignore"):
                r = ev[i + L + h] / pv[i + L] - 1
            r = r[lv[i] & np.isfinite(r)]
            mkt_cache[(i, h)] = float(np.mean(np.clip(r, -0.9, 5))) if len(r) else np.nan
        return mkt_cache[(i, h)]

    for name, g in picks.groupby("list"):
        nm = LIST_NAMES.get(name) or (f"起漲訊號：{(sig_names or {}).get(name[4:], name[4:])}" if name.startswith("sig:") else
                                      SHADOW.get(fs.market, {}).get(name[7:], (1, name))[1] if name.startswith("shadow:") else name)
        dirn = list_dir(name, fs.market)
        ex0 = (expect or {}).get(name) or {}
        if name.startswith("shadow:"):
            e = ex0.get("63") or ex0.get("20")
            if e and len(e.get("v", [])) >= 10:
                dirn = -1 if np.mean(e["v"]) < 0 else 1
            nm += "（回測平均" + ("落後" if dirn < 0 else "領先") + "大盤）"
        res = {"name": nm, "picks": int(len(g)), "days": int(g["date"].nunique()), "h": {}, "dir": dirn,
               "shadow": name.startswith("shadow:")}
        ex = (expect or {}).get(name) or {}
        warn = None
        for h in HORIZONS:
            ret, exc, dd = [], [], []
            for d, c in zip(g["date"], g["code"]):
                i, j = pos.get(d), col.get(c)
                if i is None or j is None or i + h + L >= T:
                    continue
                with np.errstate(all="ignore"):
                    r = ev[i + L + h, j] / pv[i + L, j] - 1
                m = mkt(i, h)
                if np.isfinite(r) and np.isfinite(m):
                    r = float(np.clip(r, -0.9, 5))
                    ret.append(r); exc.append(r - m); dd.append(i)
            if ret:
                x = {"n": len(ret), "ret": round(float(np.mean(ret)) * 100, 2),
                     "excess": round(float(np.mean(exc)) * 100, 2),
                     "hit": round(float(np.mean(np.array(exc) > 0)), 3)}
                daily = pd.Series(exc, index=dd).groupby(level=0).mean()
                nd = int(len(daily))
                n_eff = max(1, nd // h)          # picks within one holding period overlap
                x["days"], x["n_eff"] = nd, n_eff
                if nd >= 2 and n_eff >= 2:
                    se = float(daily.std(ddof=1)) * np.sqrt(h) / np.sqrt(nd) if nd > 1 else None
                    if se:
                        x["ci"] = [round((float(daily.mean()) - 1.96 * se) * 100, 2), round((float(daily.mean()) + 1.96 * se) * 100, 2)]
                b = _band(ex, str(h), nd, ex.get("step", "week"))
                if b:
                    x["expect"] = b
                    if nd >= 10:
                        x["status"] = "low" if x["excess"] < b["p10"] else "high" if x["excess"] > b["p90"] else "ok"
                        # a list meant to lag the market fails when it does better than history says
                        if x["status"] == ("low" if dirn > 0 else "high"):
                            warn = warn or h
                res["h"][str(h)] = x
            else:
                res["h"][str(h)] = {"n": 0}
                e = ex.get(str(h))
                if e and len(e.get("v", [])) >= 10:      # what history says to expect, shown while results accumulate
                    res["h"][str(h)]["expect_mean"] = round(float(np.mean(e["v"])) * 100, 2)
        res["warn"] = warn          # first horizon where the live record is below what history expects
        out["lists"][name] = res
    log.info("track: %d lists since %s", len(out["lists"]), out["since"])
    return out
