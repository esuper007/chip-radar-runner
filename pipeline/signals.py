"""Bottom-up "chips turning positive" signals, each validated by an event study.

Every signal is an EVENT: the first day a condition becomes true (not every day it stays true).
For each signal we measure what happened after past events (20 and 63 trading days):
  - hit rate (return > 0) vs the base rate of all liquid stock-days,
  - excess return vs the same-day cross-sectional median of liquid stocks,
  - share of events that beat that median.
The history is split by date: the first 70% (in-sample) is where thresholds were chosen, the last
30% (out-of-sample, with a purge gap) is only used to judge the signal. A signal is marked valid
only if it worked out of sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import log
from .factors import FactorSet

HORIZONS = {"20": 20, "63": 63}
IS_SHARE = 0.7
RECENT_DAYS = 5          # events in the last N trading days are shown on the dashboard
MIN_OOS_EVENTS = 50
MIN_EXCESS_20 = 0.3       # % per event over 20 trading days
MIN_BEAT_20 = 0.51

DEFS = {
    "golden": ("籌碼分數黃金交叉", "籌碼分數 5 日均線由下往上穿過 20 日均線，且 20 日均線比 5 日前高"),
    "bottom": ("低檔翻揚", "近 20 日分數曾在 30 分以下（全市場後 30%），現在比低點回升 15 分以上且站上 40 分"),
    "diverge": ("價跌籌碼集中", "股價在 60 日低點 3% 以內，但籌碼分數 20 日內上升 15 分以上"),
    "resonance": ("主要面向共振", "權重最高的兩個籌碼面向（例：台股的法人與大戶散戶）子分數 5 日內同時上升 10 分以上"),
    "inst_turn": ("法人買賣轉折", "外資、投信（陸股為主力資金）由連續賣超 3 日以上，轉為連續買超 3 日"),
    "accumulate": ("籌碼沉澱", "融資減少，同時大戶增加、散戶與股東人數減少（各市場依可得資料）"),
    "squeeze": ("軋空潛力", "空單比例在全市場前 20%，但空單開始下降，且籌碼分數 10 日內上升 5 分以上"),
}


def _first(cond: pd.DataFrame) -> pd.DataFrame:
    """Event = first day a condition is true."""
    c = cond.fillna(False).astype(bool)
    return c & ~c.shift(1, fill_value=False)


def _f(fs: FactorSet, name: str):
    return fs.factors.get(name)


def compute(fs: FactorSet, scores: dict) -> dict[str, pd.DataFrame]:
    sc = scores.get("score")
    if sc is None:
        return {}
    ev: dict[str, pd.DataFrame] = {}
    ma5, ma20 = sc.rolling(5, min_periods=4).mean(), sc.rolling(20, min_periods=15).mean()
    ev["golden"] = (ma5 > ma20) & (ma5.shift(1) <= ma20.shift(1)) & (ma20 > ma20.shift(5))
    lo20 = sc.rolling(20, min_periods=15).min()
    ev["bottom"] = _first((lo20 <= 30) & (sc - lo20 >= 15) & (sc >= 40))
    px = fs.price
    ev["diverge"] = _first((px <= px.rolling(60, min_periods=40).min() * 1.03) & (sc - sc.shift(20) >= 15))

    groups = sorted(fs.group_weights, key=lambda g: -fs.group_weights[g])
    top2 = [g for g in groups if f"s_{g}" in scores][:2]
    if len(top2) == 2:
        a, b = (scores[f"s_{g}"] for g in top2)
        ev["resonance"] = _first((a - a.shift(5) >= 10) & (b - b.shift(5) >= 10))

    turns = []
    for name in ("foreign_streak", "trust_streak", "main_streak"):
        s = _f(fs, name)
        if s is not None:
            turns.append((s == 3) & (s.shift(3).rolling(10, min_periods=1).min() <= -3))
    if turns:
        t = turns[0]
        for x in turns[1:]:
            t = t | x
        ev["inst_turn"] = t

    acc = None
    if fs.market == "tw" and all(_f(fs, n) is not None for n in ("margin_chg_20", "big1000_chg_4w", "retail_chg_4w", "holders_chg_4w")):
        acc = (_f(fs, "margin_chg_20") < 0) & (_f(fs, "big1000_chg_4w") > 0) & (_f(fs, "retail_chg_4w") < 0) & (_f(fs, "holders_chg_4w") < 0)
    elif fs.market == "cn" and all(_f(fs, n) is not None for n in ("margin_chg_20", "holders_chg_q", "main_flow_20")):
        acc = (_f(fs, "margin_chg_20") < 0) & (_f(fs, "holders_chg_q") < 0) & (_f(fs, "main_flow_20") > 0)
    if acc is not None:
        ev["accumulate"] = _first(acc)

    lvl, chg = {"us": ("short_interest_pct", "short_interest_chg"), "jp": ("short_pos_pct", "short_pos_chg_20"),
                "hk": ("short_pos_value_ratio", "short_pos_chg_4w")}.get(fs.market, (None, None))
    if lvl and _f(fs, lvl) is not None and _f(fs, chg) is not None:
        hi = _f(fs, lvl).rank(axis=1, pct=True) >= 0.8
        ev["squeeze"] = _first(hi & (_f(fs, chg) < 0) & (sc - sc.shift(10) >= 5))

    liq = fs.liquid if fs.liquid is not None else pd.DataFrame(True, index=fs.calendar, columns=fs.codes)
    return {k: (v.fillna(False).astype(bool) & liq.fillna(False).astype(bool)) for k, v in ev.items()}


def _stats(mask: np.ndarray, fwd: np.ndarray, exc: np.ndarray) -> dict:
    sel = mask & ~np.isnan(fwd)
    n = int(sel.sum())
    if n == 0:
        return {"n": 0}
    f, e = fwd[sel], exc[sel]
    return {"n": n, "hit": round(float((f > 0).mean()), 3), "mean": round(float(f.mean()), 2),
            "median": round(float(np.median(f)), 2), "excess": round(float(e.mean()), 2),
            "beat": round(float((e > 0).mean()), 3)}


def evaluate(fs: FactorSet, events: dict[str, pd.DataFrame]) -> dict:
    """Event study per signal; returns {key: {name, desc, is:{h:stats}, oos:{h:stats}, base:{h:stats}, valid}}."""
    px = fs.price
    liq = (fs.liquid if fs.liquid is not None else pd.DataFrame(True, index=px.index, columns=px.columns)).fillna(False).values
    cal = fs.calendar
    out = {}
    fw = {}
    for hk, h in HORIZONS.items():
        fwd = (px.shift(-h) / px - 1).values * 100
        fwd = np.where(liq, fwd, np.nan)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(fwd, axis=1, keepdims=True)
        fw[hk] = (fwd, fwd - med)
    known = ~np.all(np.isnan(fw["63"][0]), axis=1)
    kd = np.where(known)[0]
    if len(kd) < 100:
        return {}
    split = kd[int(len(kd) * IS_SHARE)]
    is_rows = np.zeros(len(cal), bool); is_rows[:split] = True
    oos_rows = np.zeros(len(cal), bool); oos_rows[split + HORIZONS["63"]:] = True     # purge gap
    for k, ev in events.items():
        e = ev.values
        res = {"name": DEFS[k][0], "desc": DEFS[k][1], "is": {}, "oos": {}, "base": {}}
        for hk in HORIZONS:
            fwd, exc = fw[hk]
            res["is"][hk] = _stats(e & is_rows[:, None], fwd, exc)
            res["oos"][hk] = _stats(e & oos_rows[:, None], fwd, exc)
            res["base"][hk] = _stats(liq & oos_rows[:, None], fwd, exc)
        o, b = res["oos"]["20"], res["base"]["20"]
        o63 = res["oos"]["63"]
        # valid = worked OUT OF SAMPLE: beat the same-day market median on average and more often than not
        res["valid"] = bool(o.get("n", 0) >= MIN_OOS_EVENTS and o.get("excess", -1) >= MIN_EXCESS_20
                            and o.get("beat", 0) >= MIN_BEAT_20 and o63.get("excess", -1) > 0)
        res["split"] = {"is": f"{cal[kd[0]].date()} ~ {cal[split - 1].date()}",
                        "oos": f"{cal[min(len(cal) - 1, split + HORIZONS['63'])].date()} ~ {cal[kd[-1]].date()}"}
        out[k] = res
        log.info("signal %s: oos n=%s excess20=%s beat20=%s valid=%s", k, o.get("n"), o.get("excess"), o.get("beat"), res["valid"])
    return out


def recent(events: dict[str, pd.DataFrame], days: int = RECENT_DAYS) -> dict[str, dict[str, int]]:
    """{code: {signal: trading days ago}} for events in the last `days` sessions (0 = latest day)."""
    out: dict[str, dict[str, int]] = {}
    for k, ev in events.items():
        tail = ev.iloc[-days:]
        for back, (_, row) in enumerate(tail.iloc[::-1].iterrows()):
            for c in row.index[row.values]:
                out.setdefault(c, {}).setdefault(k, back)
    return out
