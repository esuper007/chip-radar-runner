"""多頭籌碼轉差預警: warnings for stocks in an uptrend whose chips turn worse, each validated by an event study.

A warning is an EVENT: the first day a trigger becomes true for a stock that was in an uptrend the day before.
Uptrend definitions (all use data known that day):
  ma      10/22/50-day moving averages in bullish order (10 > 22 > 50)
  bias    positive gap to the 22-day average at or above the 70th percentile of the stock's own positive gaps
          over the previous three years
  high52  close within 10% of the 252-day high
  rs      120-day return in the top 30% of liquid stocks
  any     any of the above
Triggers:
  chip    score_drop  composite chip score fell 25+ points from its 20-day high and is at most 45
          crowd       retail crowding (margin use, day-trading share, retail heat, ...) enters the market's top 10%
          inst_drop   institutional sub-score fell 30+ points from its 20-day high and is at most 35
          short_drop  short-selling / borrowing sub-score fell 30+ points from its 20-day high and is at most 35
  price   below22     first close below the 22-day average (a common trailing exit), for comparison
          trail10     close 10% below the 60-day high, for comparison
  combo   chip_price  first close below the 22-day average within 10 days after any chip trigger
Validation (no thresholds were fitted): for every (trigger, uptrend) pair, the excess return of warned stocks over the
next 20 / 60 trading days (measured from the next day's close) minus the average of all stocks in the same
uptrend on the same day; the drawdown over the next 60 days; the share that still rose 10%+ more than the market
(the cost of selling too early). A pair is valid when warned stocks lagged by at least 1% over 60 days, also over
20 days, in at least 65% of calendar years, in the latest 30% of history, and the quarterly p-value survives the
Benjamini-Hochberg check across all pairs of the market.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .common import log
from .factors import ENTRY_LAG, FDR_Q, FactorSet, bh_qvalues, exit_price, fwd_ret, t_test_p

BULL = {
    "ma": ("均線多頭排列", "10 日、22 日、50 日均線由上而下排列（10 > 22 > 50）"),
    "bias": ("正乖離高檔", "股價高於 22 日均線，且乖離幅度達這檔股票過去三年正乖離的 70 百分位以上"),
    "high52": ("接近一年高點", "收盤價在 252 日最高價的 90% 以上"),
    "rs": ("相對強勢", "120 日漲幅在流動性足夠股票的前 30%"),
    "any": ("任一多頭定義", "符合以上任一種"),
}
TRIG = {
    "score_drop": ("籌碼分數急降", "綜合分數比 20 日內高點下降 25 分以上，且降到 45 分以下", "chip"),
    "crowd": ("散戶擁擠升溫", "融資使用率、當沖比、散戶熱度等散戶追價指標的平均排名進入全市場前 10%", "chip"),
    "inst_drop": ("法人轉賣", "法人子分數比 20 日內高點下降 30 分以上，且降到 35 分以下", "chip"),
    "short_drop": ("空方加碼", "放空／借券子分數比 20 日內高點下降 30 分以上（空單或借券增加），且降到 35 分以下", "chip"),
    "below22": ("跌破月線（純價格）", "收盤價第一次跌破 22 日均線；不看籌碼，作為比較基準", "price"),
    "trail10": ("高點回落 10%（純價格）", "收盤價比 60 日最高價低 10% 以上；不看籌碼，作為比較基準", "price"),
    "chip_price": ("籌碼轉差＋跌破月線", "任一籌碼轉差訊號出現後 10 日內，收盤價第一次跌破 22 日均線", "combo"),
}
CROWD = {   # factors where a high value means retail crowding (sign -1 in the score)
    "tw": ["margin_use", "margin_ratio", "daytrade_ratio_20", "retail_heat"],
    "cn": ["margin_amt_ratio", "margin_chg_20"],
    "jp": ["margin_ratio_jp", "margin_buy_pct", "margin_buy_chg_20"],
}
H = {"20": 20, "60": 60}
RECENT_DAYS = 5
MIN_EVENTS = 100
MIN_LAG_60 = -1.0          # % (warned minus same-uptrend average) over 60 trading days
MIN_YEAR_SHARE = 0.65
OOS_SHARE = 0.3


def _first(c: np.ndarray) -> np.ndarray:
    prev = np.zeros_like(c)
    prev[1:] = c[:-1]
    return c & ~prev


def _bull(fs: FactorSet, liq: np.ndarray) -> dict[str, np.ndarray]:
    px = fs.price
    ma10, ma22, ma50 = (px.rolling(n, min_periods=int(n * 0.8)).mean() for n in (10, 22, 50))
    out = {"ma": ((ma10 > ma22) & (ma22 > ma50)).values}
    bias = px / ma22 - 1
    pos = bias.where(bias > 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        thr = pos.rolling(756, min_periods=120).quantile(0.7).shift(1)
    out["bias"] = ((bias > 0) & (bias >= thr)).values
    out["high52"] = (px >= px.rolling(252, min_periods=200).max() * 0.9).values
    r120 = (px / px.shift(120) - 1).where(liq)
    out["rs"] = (r120.rank(axis=1, pct=True) >= 0.7).values
    out["any"] = out["ma"] | out["bias"] | out["high52"] | out["rs"]
    return out


def _drop(s: pd.DataFrame, fall: float, cap: float) -> np.ndarray:
    hi = s.rolling(20, min_periods=10).max()
    return ((s - hi <= -fall) & (s <= cap)).values


def _triggers(fs: FactorSet, scores: dict, liq: np.ndarray) -> dict[str, np.ndarray]:
    px = fs.price
    t = {}
    scores = {k: v.reindex(index=px.index, columns=px.columns) for k, v in scores.items()}
    sc = scores.get("score")
    if sc is not None:
        t["score_drop"] = _first(_drop(sc, 25, 45))
    names = [n for n in CROWD.get(fs.market, []) if n in fs.factors]
    if names:
        rk = None
        for n in names:
            r = fs.factors[n].where(liq).rank(axis=1, pct=True)
            rk = r if rk is None else rk.add(r, fill_value=np.nan)
        rk = rk / len(names)
        t["crowd"] = _first((rk.rank(axis=1, pct=True) >= 0.9).values)
    for g, k in (("inst", "inst_drop"), ("short", "short_drop")):
        s = scores.get(f"s_{g}")
        if s is not None and s.notna().values.any():
            t[k] = _first(_drop(s, 30, 35))
    ma22 = px.rolling(22, min_periods=18).mean()
    below = (px < ma22).values
    t["below22"] = _first(below)
    t["trail10"] = _first((px <= px.rolling(60, min_periods=40).max() * 0.9).values)
    chip = [t[k] for k in ("score_drop", "crowd", "inst_drop", "short_drop") if k in t]
    if chip:
        any_chip = np.logical_or.reduce(chip)
        recent_chip = pd.DataFrame(any_chip).rolling(10, min_periods=1).max().values.astype(bool)
        t["chip_price"] = t["below22"] & recent_chip
    return t


def compute(fs: FactorSet, scores: dict) -> dict:
    """Returns {"stats", "bull_today", "recent", "_events"}; `_events` (trigger -> DataFrame of events in an
    uptrend by any definition, out-of-sample part) feeds the live track record's expectations."""
    cal, px = fs.calendar, fs.price
    T = len(cal)
    liqdf = fs.liquid if fs.liquid is not None else pd.DataFrame(True, index=cal, columns=px.columns)
    liq = liqdf.reindex(columns=px.columns).fillna(False).values.astype(bool)
    bull = _bull(fs, liq)
    trig = _triggers(fs, scores, liq)
    prev = {k: np.vstack([np.zeros((1, v.shape[1]), bool), v[:-1]]) & liq for k, v in bull.items()}   # uptrend yesterday

    # outcomes, measured from the next day's close
    exc, big = {}, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for hk, h in H.items():
            f = np.where(liq, fwd_ret(px, h).values * 100, np.nan).astype("float32")
            lo, hi = np.nanpercentile(f, 1, axis=1, keepdims=True), np.nanpercentile(f, 99, axis=1, keepdims=True)
            f = np.clip(f, lo, hi)
            exc[hk] = f - np.nanmean(f, axis=1, keepdims=True)
        big = np.where(np.isfinite(exc["60"]), (exc["60"] > 10).astype("float32"), np.nan)
        ep = exit_price(px)
        L = ENTRY_LAG
        fmin = ep.iloc[::-1].rolling(60, min_periods=40).min().iloc[::-1].shift(-(L + 1))
        dd = np.where(liq, (fmin / px.shift(-L) - 1).values * 100, np.nan).astype("float32")
    known = np.isfinite(exc["60"]).any(axis=1)
    kd = np.where(known)[0]
    if len(kd) < 250:
        return {}
    oos_from = kd[int(len(kd) * (1 - OOS_SHARE))]
    years = cal.year.values
    quarters = cal.to_period("Q")

    stats, pvals = {}, {}
    for b, B in prev.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ctrl = {hk: np.nanmean(np.where(B, exc[hk], np.nan), axis=1) for hk in H}
            ctrl_dd = np.nanmean(np.where(B, dd, np.nan), axis=1)
            ctrl_big = np.nanmean(np.where(B, big, np.nan), axis=1)
        for k, E in trig.items():
            r, c = np.where(E & B)
            ok = np.isfinite(exc["60"][r, c]) & np.isfinite(ctrl["60"][r])
            r, c = r[ok], c[ok]
            res = {"n": int(len(r))}
            if len(r) < 30:
                stats.setdefault(k, {})[b] = res
                continue
            d = {hk: exc[hk][r, c] - ctrl[hk][r] for hk in H}
            res.update({f"diff{hk}": round(float(np.nanmean(d[hk])), 2) for hk in H})
            res["dd"] = round(float(np.nanmean(dd[r, c] - ctrl_dd[r])), 2)
            res["big"] = round(float(np.nanmean(big[r, c])), 3)
            res["big_ctrl"] = round(float(np.nanmean(ctrl_big[r])), 3)
            yr = pd.Series(d["60"], index=years[r]).groupby(level=0)
            ym = yr.mean()[yr.size() >= 20]
            res["years"] = int(len(ym))
            res["years_lag"] = int((ym < 0).sum())
            o = r >= oos_from
            res["oos_n"] = int(o.sum())
            res["oos_diff60"] = round(float(np.nanmean(d["60"][o])), 2) if o.sum() >= 20 else None
            qm = pd.Series(-d["60"], index=quarters[r]).groupby(level=0).mean()
            p = t_test_p(qm.values)          # one-sided: warned stocks lag the uptrend average
            res["p"] = round(p, 4) if p is not None else None
            res["n_quarters"] = int(len(qm))
            res["valid_raw"] = bool(res["n"] >= MIN_EVENTS and res["diff60"] <= MIN_LAG_60 and res["diff20"] < 0
                                    and res["years"] >= 4 and res["years_lag"] >= MIN_YEAR_SHARE * res["years"]
                                    and res["oos_diff60"] is not None and res["oos_diff60"] < 0)
            stats.setdefault(k, {})[b] = res
            pvals[(k, b)] = res["p"]
    qv = bh_qvalues(pvals)
    for (k, b), q in qv.items():
        s = stats[k][b]
        s["q"] = q
        s["fdr_ok"] = bool(q is not None and q <= FDR_Q)
        s["valid"] = bool(s["valid_raw"] and s["fdr_ok"])
    nv = sum(1 for k in stats for b in stats[k] if stats[k][b].get("valid"))
    log.info("exits: %d triggers x %d uptrend definitions, %d valid", len(trig), len(prev), nv)

    # today and the last few days
    t0 = T - 1
    bull_today = {}
    for j, c in enumerate(px.columns):
        bs = [b for b in ("ma", "bias", "high52", "rs") if bull[b][t0, j] and liq[t0, j]]
        if bs:
            bull_today[c] = bs
    recent = {}
    for k, E in trig.items():
        for back in range(RECENT_DAYS):
            i = t0 - back
            for j in np.where(E[i])[0]:
                bs = [b for b in ("ma", "bias", "high52", "rs") if prev[b][i, j]]
                if bs:
                    recent.setdefault(px.columns[j], []).append([k, back, bs])
    # events for the live record's expectation ranges (uptrend by any definition, latest 30% of history)
    events = {}
    for k, E in trig.items():
        r, c = np.where(E & prev["any"])
        m = r >= oos_from
        if m.sum():
            events[k] = pd.DataFrame({"date": cal[r[m]], "code": px.columns[c[m]]})
    return {"stats": stats, "bull_defs": {k: list(v) for k, v in BULL.items()},
            "triggers": {k: list(v) for k, v in TRIG.items() if k in trig},
            "n_bull_today": {b: int(sum(1 for v in bull_today.values() if b in v)) for b in ("ma", "bias", "high52", "rs")},
            "oos_from": str(cal[oos_from].date()), "bull_today": bull_today, "recent": recent, "_events": events}


def valid_pairs(res: dict) -> set:
    return {(k, b) for k, d in (res or {}).get("stats", {}).items() for b, s in d.items() if s.get("valid")}
