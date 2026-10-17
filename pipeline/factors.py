"""Factor engine: wide (date x code) matrices -> chip factors, sub-scores and composite score.

Every market module provides `build_factors(store) -> FactorSet`.  Everything downstream
(scoring, prediction, output) is market-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class FactorSet:
    market: str
    calendar: pd.DatetimeIndex                 # trading days
    codes: pd.Index
    factors: dict[str, pd.DataFrame]           # name -> wide frame (date x code)
    price: pd.DataFrame                        # adjusted close, wide
    meta: pd.DataFrame                         # code -> name, industry, mkt
    score_spec: dict[str, list[tuple[str, int]]]   # group -> [(factor, +1/-1)]
    group_weights: dict[str, float]
    labels: dict[str, str] = field(default_factory=dict)   # factor -> 中文說明
    series: list[str] = field(default_factory=list)        # factors to chart per stock
    liquid: pd.DataFrame | None = None          # bool wide mask of investable stocks
    market_series: dict[str, pd.Series] = field(default_factory=dict)  # market-level chip series
    quality: list = field(default_factory=list)                        # build-time data checks


# ---------------------------------------------------------------- wide helpers
def wide(df: pd.DataFrame, col: str, cal: pd.DatetimeIndex, codes: pd.Index,
         ffill_limit: int | None = 0) -> pd.DataFrame:
    if df is None or df.empty or col not in df.columns:
        return pd.DataFrame(np.nan, index=cal, columns=codes, dtype="float32")
    w = (df.dropna(subset=[col]).drop_duplicates(["date", "code"], keep="last")
         .pivot(index="date", columns="code", values=col))
    w.index = pd.to_datetime(w.index)
    if ffill_limit != 0:
        w = w.reindex(w.index.union(cal)).sort_index().ffill(limit=ffill_limit)
    w = w.reindex(index=cal, columns=codes)
    return w.astype("float32")


def exit_price(px: pd.DataFrame) -> pd.DataFrame:
    """Price used for the END of a forward-return window.

    A stock that stops trading (delisted, merged, long suspension) used to have no price at the end of
    the window, so that outcome silently dropped out of every backtest - and the stocks that drop out are
    mostly the losers (survivorship bias). Here the position is valued at its LAST traded price instead:
    the return up to the last trade counts, then it is held at that price. Entry prices still require
    a real trade on the day (use `px` for the start of the window).
    """
    return px.ffill()


# ---------------------------------------------------------------- multiple testing
def sign_test_p(wins: int, n: int) -> float | None:
    """One-sided P(X >= wins) for X ~ Binomial(n, 0.5): chance of winning this many years by luck."""
    if not n:
        return None
    from math import comb
    return float(sum(comb(n, k) for k in range(wins, n + 1)) / 2 ** n)


def t_test_p(values) -> float | None:
    """One-sided p-value that the mean of (roughly independent) values is > 0."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype="float64")
    if len(v) < 6 or v.std(ddof=1) == 0:
        return None
    from scipy.stats import t as tdist
    tt = v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))
    return float(tdist.sf(tt, len(v) - 1))


def bh_qvalues(pvals: dict) -> dict:
    """Benjamini-Hochberg q-values: the false discovery rate you accept by calling each test a discovery.
    None p-values stay None and do not count as tests."""
    items = [(k, p) for k, p in pvals.items() if p is not None]
    m = len(items)
    if not m:
        return {k: None for k in pvals}
    items.sort(key=lambda x: x[1])
    q, prev = {}, 1.0
    for i in range(m - 1, -1, -1):
        k, p = items[i]
        prev = min(prev, p * m / (i + 1))
        q[k] = round(prev, 4)
    return {k: q.get(k) for k in pvals}


FDR_Q = 0.10        # a rule / factor / signal counts as a discovery when its q-value is <= 10%


def since_start(w: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Blank out dates before a dataset's first observation.

    Event-style data (insider trades, short-position reports, LHB) is missing on days with no
    event, so we fill those with 0 — but only from the day the dataset starts. Filling zeros
    before that would invent years of fake history and poison the backtest.
    """
    if df is None or df.empty:
        return w * np.nan
    first = pd.to_datetime(df["date"]).min()
    out = w.copy()
    out.loc[out.index < first] = np.nan
    return out


def rsum(w: pd.DataFrame, n: int, minp: float = 0.6) -> pd.DataFrame:
    return w.rolling(n, min_periods=max(1, int(n * minp))).sum()


def streak(w: pd.DataFrame) -> pd.DataFrame:
    """Signed length of the current run of positive / negative values."""
    s = np.sign(w.fillna(0).values)
    out = np.zeros_like(s)
    for i in range(len(s)):
        if i == 0:
            out[i] = s[i]
            continue
        same = (s[i] == np.sign(out[i - 1])) & (s[i] != 0)
        out[i] = np.where(same, out[i - 1] + s[i], s[i])
    return pd.DataFrame(out, index=w.index, columns=w.columns).astype("float32")


def chg(w: pd.DataFrame, n: int) -> pd.DataFrame:
    return w - w.shift(n)


def pct_chg(w: pd.DataFrame, n: int) -> pd.DataFrame:
    prev = w.shift(n)
    return (w / prev.where(prev.abs() > 0) - 1) * 100


def price_context(px: pd.DataFrame, vol: pd.DataFrame | None, shares: pd.DataFrame | None) -> dict:
    r = px.pct_change(fill_method=None)
    f = {
        "ret_20": (px / px.shift(20) - 1) * 100,
        "ret_60": (px / px.shift(60) - 1) * 100,
        "ret_250": (px / px.shift(250) - 1) * 100,
        "vol_60": r.rolling(60, min_periods=40).std() * np.sqrt(252) * 100,
        "dist_52w_high": (px / px.rolling(250, min_periods=120).max() - 1) * 100,
    }
    if vol is not None and shares is not None:
        f["turnover_20"] = rsum(vol, 20) / 20 / shares.where(shares > 0) * 100
    return f


# ---------------------------------------------------------------- scoring
def xrank(w: pd.DataFrame, mask: pd.DataFrame | None = None) -> pd.DataFrame:
    """Cross-sectional percentile rank per date in [0, 1] (NaN stays NaN)."""
    if mask is not None:
        w = w.where(mask)
    return w.rank(axis=1, pct=True)


def alive(fs: FactorSet) -> pd.DataFrame:
    """Stocks that traded in the last 5 sessions. Delisted stocks stay in the data for the backtests, but
    must not be ranked (or shown) after they stopped trading."""
    return fs.price.notna().rolling(5, min_periods=1).max().astype(bool)


def compute_scores(fs: FactorSet) -> dict[str, pd.DataFrame]:
    """Sub-scores (0-100) per group and weighted composite `score`."""
    out = {}
    live = alive(fs)
    num = None
    den = None
    for g, items in fs.score_spec.items():
        parts = []
        for name, sign in items:
            if name not in fs.factors:
                continue
            rk = xrank(fs.factors[name] * sign, live)  # rank every trading stock; the page filters by liquidity
            parts.append(rk)
        if not parts:
            continue
        stack = np.stack([p.values for p in parts])
        with np.errstate(all="ignore"):
            g_score = np.nanmean(stack, axis=0)
        g_df = pd.DataFrame(g_score * 100, index=fs.calendar, columns=fs.codes)
        # require at least half the group's factors present
        cnt = np.sum(~np.isnan(stack), axis=0)
        g_df = g_df.where(cnt >= max(1, len(parts) // 2))
        out[f"s_{g}"] = g_df.astype("float32")
        wgt = fs.group_weights.get(g, 0.1)
        v = g_df.values
        num = (np.nan_to_num(v) * wgt) if num is None else num + np.nan_to_num(v) * wgt
        d = (~np.isnan(v)) * wgt
        den = d if den is None else den + d
    if num is not None:
        with np.errstate(all="ignore"):
            out["score"] = pd.DataFrame(np.where(den > 0, num / den, np.nan),
                                        index=fs.calendar, columns=fs.codes).astype("float32")
    return out


def signal_flags(fs: FactorSet, t: pd.Timestamp) -> dict[str, list[str]]:
    """Human-readable chip signals per code on date t (Chinese)."""
    F = {k: v.loc[t] for k, v in fs.factors.items() if t in v.index}
    flags: dict[str, list[str]] = {c: [] for c in fs.codes}

    def add(cond: pd.Series, text: str):
        for c in cond[cond.fillna(False)].index:
            flags[c].append(text)
    g = F.get
    if g("foreign_streak") is not None:
        add(g("foreign_streak") >= 5, "外資連買≥5日")
        add(g("foreign_streak") <= -5, "外資連賣≥5日")
    if g("trust_streak") is not None:
        add(g("trust_streak") >= 3, "投信連買≥3日")
        add(g("trust_streak") <= -3, "投信連賣≥3日")
    if g("big400_chg_4w") is not None:
        add(g("big400_chg_4w") >= 1.0, "大戶4週增持≥1pp")
        add(g("big400_chg_4w") <= -1.0, "大戶4週減持≥1pp")
    if g("margin_chg_20") is not None and g("ret_20") is not None:
        add((g("margin_chg_20") > 15) & (g("ret_20") < 0), "融資增價跌(散戶接刀)")
        add((g("margin_chg_20") < -10) & (g("ret_20") > 0), "融資減價漲(籌碼沉澱)")
    if g("short_margin_ratio") is not None:
        add(g("short_margin_ratio") > 30, "券資比>30%(軋空潛力)")
    if g("sbl_chg_20") is not None:
        add(g("sbl_chg_20") > 1.0, "借券賣出餘額20日增>1%股本")
    if g("insider_pledge_ratio") is not None:
        add(g("insider_pledge_ratio") > 50, "董監質押>50%")
    if g("insider_net_chg") is not None:
        add(g("insider_net_chg") > 0.5, "內部人淨買進")
        add(g("insider_net_chg") < -0.5, "內部人淨賣出")
    if g("short_interest_dtc") is not None:
        add(g("short_interest_dtc") > 7, "回補天數>7(軋空潛力)")
    if g("margin_ratio_jp") is not None:
        add(g("margin_ratio_jp") < 1, "信用倍率<1(賣殘多)")
    if g("north_chg_20") is not None:
        add(g("north_chg_20") > 0.5, "南北向資金20日增持>0.5pp")
    return flags
