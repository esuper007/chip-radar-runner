"""組合回測: the ranked lists rebuilt on the backtest (out of sample, weekly) held as real portfolios.

Each week the list's 20 stocks are bought in equal amounts at the next day's close and held until the next
week's list is bought (no transaction costs). The market is the equal-weighted average of all liquid stocks
over the same days. Reported per list:
  - relative value curve (portfolio / market), annualized excess return, max drawdown of the relative curve
  - weekly win rate (share of weeks that beat the market) and turnover (share of names replaced each week)
  - excess return per calendar year, and the annualized excess with the best 5% of weeks removed
    (if that is near zero, the result came from a few strong stretches)
  - neutral checks: excess versus the stock's own industry, own volatility fifth and own size fifth, so
    a list that only leans on one industry, on volatile stocks or on small caps shows it
  - concentration: average share of the largest industry, average volatility and size percentile
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import log
from .factors import ENTRY_LAG, FactorSet, exit_price

CLIP = (-0.9, 5)
MIN_GROUP = 5            # an industry / fifth needs this many liquid stocks to be its own benchmark
CURVE_POINTS = 160


def _industry_codes(fs: FactorSet) -> np.ndarray | None:
    try:
        ind = fs.meta.reindex(fs.codes)["industry"].fillna("").astype(str).values
    except Exception:
        return None
    if (ind != "").sum() <= len(ind) * 0.5:
        return None
    return np.where(ind == "", -1, pd.Series(ind).astype("category").cat.codes.values)


def _fifths(x: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """0-4 fifth of x among ok stocks (-1 elsewhere)."""
    out = np.full(len(x), -1)
    m = ok & np.isfinite(x)
    if m.sum() >= 25:
        r = pd.Series(x[m]).rank(pct=True).values
        out[m] = np.minimum((r * 5).astype(int), 4)
    return out


def _pct(x: np.ndarray, ok: np.ndarray, sel: np.ndarray) -> float | None:
    m = ok & np.isfinite(x)
    if m.sum() < 25:
        return None
    r = pd.Series(x).where(m).rank(pct=True).values
    v = r[sel]
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if len(v) else None


def _group_mean(r: np.ndarray, g: np.ndarray, ok: np.ndarray) -> dict:
    out = {}
    m = ok & np.isfinite(r) & (g >= 0)
    if m.any():
        s = pd.Series(r[m]).groupby(g[m])
        cnt, mean = s.size(), s.mean()
        out = {int(k): float(v) for k, v in mean[cnt >= MIN_GROUP].items()}
    return out


class _Ctx:
    def __init__(self, fs: FactorSet, size: pd.DataFrame | None):
        self.fs = fs
        self.cal = fs.calendar
        self.T = len(self.cal)
        self.pos = {d: i for i, d in enumerate(self.cal)}
        self.col = {c: j for j, c in enumerate(fs.price.columns)}
        self.pv = fs.price.values.astype("float64")
        self.ev = exit_price(fs.price).values.astype("float64")
        liq = fs.liquid if fs.liquid is not None else pd.DataFrame(True, index=self.cal, columns=fs.price.columns)
        self.lv = liq.reindex(columns=fs.price.columns).fillna(False).values.astype(bool)
        with np.errstate(all="ignore"):
            dr = fs.price.pct_change(fill_method=None)
        self.vol = dr.rolling(60, min_periods=40).std().values
        self.size = size.reindex(index=self.cal, columns=fs.price.columns).values if size is not None else None
        ic = _industry_codes(fs)
        self.ind = None
        if ic is not None:
            m = pd.Series(ic, index=fs.codes).reindex(fs.price.columns).fillna(-1).astype(int).values
            self.ind = m

    def describe(self, i: int, cols: np.ndarray) -> dict:
        """Concentration of one list on day i."""
        ok = self.lv[i]
        sel = np.zeros(len(ok), bool)
        sel[cols] = True
        res = {"n": int(len(cols))}
        v = _pct(self.vol[i], ok, sel)
        if v is not None:
            res["vol_pct"] = round(v * 100, 1)
        if self.size is not None:
            s = _pct(self.size[i], ok, sel)
            if s is not None:
                res["size_pct"] = round(s * 100, 1)
        if self.ind is not None:
            g = self.ind[cols]
            g = g[g >= 0]
            if len(g):
                res["top_ind_share"] = round(float(np.bincount(g).max() / len(cols)), 3)
                res["_ind"] = g
        return res


def simulate(fs: FactorSet, picks: dict, size: pd.DataFrame | None = None, ctx: _Ctx | None = None) -> dict:
    """picks: {list name: DataFrame(date, code)} with one list per (weekly) date."""
    ctx = ctx or _Ctx(fs, size)
    L = ENTRY_LAG
    out = {}
    for name, pk in picks.items():
        if pk is None or pk.empty:
            continue
        pk = pk.assign(date=pd.to_datetime(pk["date"]))
        dates = sorted(d for d in pk["date"].unique() if d in ctx.pos)
        groups = {d: g["code"].astype(str).tolist() for d, g in pk.groupby("date")}
        rows, prev, turn, conc = [], None, [], []
        for d0, d1 in zip(dates[:-1], dates[1:]):
            i0, i1 = ctx.pos[d0], ctx.pos[d1]
            s, e = i0 + L, i1 + L
            if e >= ctx.T or e - s < 1:
                continue
            cols = np.array([ctx.col[c] for c in groups[d0] if c in ctx.col], dtype=int)
            if len(cols) < 5:
                continue
            with np.errstate(all="ignore"):
                r = np.clip(ctx.ev[e] / ctx.pv[s] - 1, *CLIP)
            ok = ctx.lv[i0]
            rm = r[ok & np.isfinite(r)]
            if len(rm) < 30:
                continue
            mkt = float(rm.mean())
            rp = r[cols]
            fin = np.isfinite(rp)
            if fin.sum() < 3:
                continue
            port = float(rp[fin].mean())
            row = {"d": d0, "days": e - s, "port": port, "mkt": mkt}
            # neutral benchmarks: each pick against its own group
            for key, g in (("ind", ctx.ind),
                           ("vol", _fifths(ctx.vol[i0], ok)),
                           ("size", _fifths(ctx.size[i0], ok) if ctx.size is not None else None)):
                if g is None:
                    continue
                gm = _group_mean(r, g, ok)
                bm = np.array([gm.get(int(g[j]), mkt) for j in cols])
                row[f"vs_{key}"] = float(np.mean((rp - bm)[fin]))
            rows.append(row)
            cur = set(cols.tolist())
            if prev is not None:
                turn.append(1 - len(cur & prev) / max(len(cur), 1))
            prev = cur
            conc.append(ctx.describe(i0, cols))
        if len(rows) < 20:
            continue
        df = pd.DataFrame(rows)
        exc = (1 + df["port"]) / (1 + df["mkt"]) - 1          # weekly relative return
        rel = (1 + exc).cumprod()
        per_year = 252 / df["days"].mean()
        n_years = len(df) / per_year
        ann = lambda x: float((1 + x).prod() ** (1 / n_years) - 1) if n_years > 0 else None
        dd = float((rel / rel.cummax() - 1).min())
        best = exc.quantile(0.95)
        res = {"from": str(df["d"].min().date()), "to": str(df["d"].max().date()), "weeks": int(len(df)),
               "ann_excess": round(ann(exc) * 100, 2), "ann_port": round(ann(df["port"]) * 100, 2),
               "ann_mkt": round(ann(df["mkt"]) * 100, 2), "max_dd": round(dd * 100, 2),
               "win": round(float((exc > 0).mean()), 3),
               "turnover": round(float(np.mean(turn)), 3) if turn else None,
               "ann_ex_best5": round(ann(exc[exc < best]) * 100, 2)}
        yr = pd.DataFrame({"y": df["d"].dt.year, "x": exc}).groupby("y")["x"]
        res["years"] = [{"y": int(y), "x": round(float((1 + g).prod() - 1) * 100, 2), "w": int(len(g))}
                        for y, g in yr]
        neu = {}
        for key in ("ind", "vol", "size"):
            c = f"vs_{key}"
            if c in df and df[c].notna().sum() >= 20:
                neu[key] = round(ann(df[c].fillna(0)) * 100, 2)
        res["neutral"] = neu
        cdf = pd.DataFrame([{k: v for k, v in c.items() if not k.startswith("_")} for c in conc])
        res["conc"] = {k: round(float(cdf[k].mean()), 3 if k == "top_ind_share" else 1)
                       for k in ("top_ind_share", "vol_pct", "size_pct") if k in cdf and cdf[k].notna().any()}
        k = max(1, len(rel) // CURVE_POINTS)
        idx = list(range(0, len(rel), k))
        if idx[-1] != len(rel) - 1:
            idx.append(len(rel) - 1)
        res["curve"] = {"d": [str(df["d"].iloc[i].date()) for i in idx], "v": [round(float(rel.iloc[i]), 4) for i in idx]}
        out[name] = res
    log.info("portfolio: %d lists simulated", len(out))
    return out


def today(fs: FactorSet, picks: pd.DataFrame | None, size: pd.DataFrame | None = None) -> dict:
    """Concentration of the lists recorded on the last day (industry names, volatility / size percentile)."""
    if picks is None or picks.empty:
        return {}
    t = fs.calendar[-1]
    p = picks[pd.to_datetime(picks["date"]) == t]
    if p.empty:
        return {}
    ctx = _Ctx(fs, size)
    i = ctx.pos[t]
    names = None
    try:
        names = fs.meta.reindex(ctx.fs.price.columns)["industry"].fillna("").astype(str).values
    except Exception:
        pass
    out = {}
    for lst, g in p.groupby("list"):
        if lst.startswith(("sig:", "bo")):
            continue
        cols = np.array([ctx.col[c] for c in g["code"].astype(str) if c in ctx.col], dtype=int)
        if len(cols) < 5:
            continue
        d = ctx.describe(i, cols)
        d.pop("_ind", None)
        if ctx.ind is not None and names is not None:
            vc = pd.Series(names[cols]).replace("", np.nan).dropna().value_counts()
            d["top_ind"] = [{"i": k, "n": int(v)} for k, v in vc.head(3).items()]
        out[lst] = d
    return out
