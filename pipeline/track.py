"""實戰紀錄: every build stores the lists the dashboard shows that day, and later measures what really
happened to them. Nothing here is backtested - only picks recorded on the day itself count.

Lists recorded each day (liquid stocks only):
  up_1q / up_1y   top 20 by blended up-probability (in/out-of-sample 70%, ML 15%, analog 15%)
  down_1q         bottom 20 by blended up-probability (expected to lag)
  gain_1q         top 20 by blended median forecast return
  sig:<key>       stocks whose rally signal fired that day
  bo              fresh 1-year-high breakouts that day (score = LightGBM half-year rank when available)
  bo_top          the best fifth of that day's breakouts by the LightGBM half-year model

Results (no transaction costs): return after 20 / 63 / 126 trading days, and excess return versus the
equal-weighted average of all liquid stocks over the same days. "hit" = share of picks that beat it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import log

DATASET = "picks"
TOPN = 20
HORIZONS = (20, 63, 126)
W = {"is": 0.70, "ml": 0.15, "an": 0.15}
LIST_NAMES = {"up_1q": "1季上漲機率前 20", "up_1y": "1年上漲機率前 20", "down_1q": "1季下跌機率前 20（應落後）",
              "gain_1q": "1季預估漲幅前 20", "bo": "當日突破一年高點（全部）", "bo_top": "當日突破・模型前 20%"}


def _blend(fc: pd.DataFrame, hk: str, stat: str) -> pd.Series:
    num = den = 0
    for m, w in W.items():
        col = f"{m}_{hk}_{stat}"
        if col in fc.columns:
            v = fc[col]
            num = num + v.fillna(0) * w
            den = den + v.notna() * w
    if isinstance(den, int):
        return pd.Series(dtype=float)
    return (num / den.where(den > 0)).dropna()


def record(st, fs, fc: pd.DataFrame, sig: dict | None, bo: dict | None) -> int:
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
            p = _blend(f, hk, "p_up")
            if len(p) >= TOPN * 3:
                add(name, p.sort_values(ascending=False).head(TOPN).items())
                if hk == "1q":
                    add("down_1q", p.sort_values().head(TOPN).items())
        q = _blend(f, "1q", "q50")
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
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    st.append(DATASET, df, keys=("date", "list", "code"))
    log.info("track: recorded %d picks in %d lists for %s", len(df), df["list"].nunique(), t.date())
    return len(df)


def evaluate(st, fs, sig_names: dict | None = None) -> dict:
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
    lv = liq.values
    out = {"since": str(picks["date"].min().date()), "days": int(picks["date"].nunique()), "lists": {}}
    mkt_cache = {}

    def mkt(i, h):
        if (i, h) not in mkt_cache:
            with np.errstate(all="ignore"):
                r = pv[i + h] / pv[i] - 1
            r = r[lv[i] & np.isfinite(r)]
            mkt_cache[(i, h)] = float(np.mean(np.clip(r, -0.9, 5))) if len(r) else np.nan
        return mkt_cache[(i, h)]

    for name, g in picks.groupby("list"):
        res = {"name": LIST_NAMES.get(name) or (f"起漲訊號：{(sig_names or {}).get(name[4:], name[4:])}" if name.startswith("sig:") else name),
               "picks": int(len(g)), "days": int(g["date"].nunique()), "h": {}}
        for h in HORIZONS:
            ret, exc = [], []
            for d, c in zip(g["date"], g["code"]):
                i, j = pos.get(d), col.get(c)
                if i is None or j is None or i + h >= T:
                    continue
                with np.errstate(all="ignore"):
                    r = pv[i + h, j] / pv[i, j] - 1
                m = mkt(i, h)
                if np.isfinite(r) and np.isfinite(m):
                    r = float(np.clip(r, -0.9, 5))
                    ret.append(r); exc.append(r - m)
            if ret:
                res["h"][str(h)] = {"n": len(ret), "ret": round(float(np.mean(ret)) * 100, 2),
                                    "excess": round(float(np.mean(exc)) * 100, 2),
                                    "hit": round(float(np.mean(np.array(exc) > 0)), 3)}
            else:
                res["h"][str(h)] = {"n": 0}
        out["lists"][name] = res
    log.info("track: %d lists since %s", len(out["lists"]), out["since"])
    return out
