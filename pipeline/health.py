"""因子體檢 + 自動權重.

For every scoring factor (with its intended sign):
1. Neutralize: cross-sectional rank, minus its industry average, then the part explained by size
   (log 60-day average traded value) is removed; re-ranked to 0-1. What is left is the chip signal
   that is not just "big caps" or "hot industry".
2. Health (weekly samples, liquid stocks, no transaction costs): rank IC with the forward 5 / 20 / 60
   trading-day excess return, IC information ratio, share of years with positive IC, top-minus-bottom
   quintile excess return over 20 days, turnover (1 - 20-day rank autocorrelation), IC in different
   market regimes, and a one-sided t-test of the IC with a Benjamini-Hochberg false-discovery check
   across all factors of the market (many factors are tested at once, so some pass by luck).
3. Automatic weights, refreshed every quarter WITHOUT look-ahead, aware of overlap between factors:
   at each quarter start, factors whose values move together (average cross-sectional correlation
   >= 0.5 over the previous 3 years) form one cluster. A cluster's weight = max(0, IC IR of the
   cluster's average IC over the previous 3 years, outcomes already known). Inside a cluster the weight
   is split by each member's own positive IR. So four ways of measuring the same "retail crowding"
   share one weight instead of getting four. A factor needs 2 years of known outcomes before it can
   get weight (data still accumulating = 0). Before 1 year of history the original equal weights apply.
4. Composite comparison: the same history scored four ways (equal weight / IR weights on raw ranks /
   IR weights on neutral ranks / cluster weights on neutral ranks), so the effect of neutralization and
   of the cluster weights can be seen, plus the effect of counting delisted stocks (survivorship).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .common import log
from .factors import FDR_Q, FactorSet, alive, bh_qvalues, exit_price, t_test_p, wide

warnings.filterwarnings("ignore", category=RuntimeWarning, module=__name__)   # empty-slice means on sparse rows

HORIZONS = (5, 20, 60)
SAMPLE_EVERY = 5
LOOKBACK_DAYS = 756          # 3 years of trading days for the weights
MIN_HISTORY = 252
MIN_WEEKS_FOR_WEIGHT = 104   # a factor needs ~2 years of known weekly outcomes before it gets weight
CLUSTER_CORR = 0.5           # factors correlated at least this much share one weight


def _size(fs: FactorSet, store) -> pd.DataFrame | None:
    if store is None:
        return None
    try:
        pr = store.load("price")
        close, vol = wide(pr, "close", fs.calendar, fs.codes), wide(pr, "volume", fs.calendar, fs.codes)
        tv = (close * vol).where(vol > 0)
        s = np.log(tv.rolling(60, min_periods=20).mean())
        return s if s.notna().values.any() else None
    except Exception as e:
        log.warning("health: size proxy unavailable (%s)", e)
        return None


def _industry(fs: FactorSet) -> np.ndarray | None:
    try:
        ind = fs.meta.reindex(fs.codes)["industry"].fillna("").astype(str).values
    except Exception:
        return None
    return ind if (ind != "").sum() > len(ind) * 0.5 else None


def neutralize(rk: pd.DataFrame, ind: np.ndarray | None, size_rk: np.ndarray | None) -> pd.DataFrame:
    """rk: 0-1 ranks. Remove industry means and the linear size effect per date, re-rank to 0-1."""
    v = rk.values.astype("float64")
    ok = np.isfinite(v)
    if ind is not None:
        codes = pd.Series(ind).astype("category").cat.codes.values
        k = codes.max() + 1
        out = v.copy()
        for g in range(k):
            cols = np.where((codes == g) & (np.asarray(ind) != ""))[0]
            if len(cols) < 5:
                continue
            with np.errstate(all="ignore"):
                m = np.nanmean(v[:, cols], axis=1, keepdims=True)
            out[:, cols] = v[:, cols] - np.nan_to_num(m) + 0.5
        v = np.where(ok, out, np.nan)
    if size_rk is not None:
        s = np.where(ok, size_rk, np.nan)
        with np.errstate(all="ignore"):
            sm, vm = np.nanmean(s, axis=1, keepdims=True), np.nanmean(v, axis=1, keepdims=True)
            cov = np.nanmean((s - sm) * (v - vm), axis=1, keepdims=True)
            var = np.nanvar(s, axis=1, keepdims=True)
            beta = np.where(var > 1e-9, cov / var, 0)
        v = v - np.nan_to_num(beta) * np.nan_to_num(s - sm)
    return pd.DataFrame(v, index=rk.index, columns=rk.columns).rank(axis=1, pct=True)


def _row_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = np.where(ok, a, np.nan), np.where(ok, b, np.nan)
    with np.errstate(all="ignore"):
        a = a - np.nanmean(a, axis=1, keepdims=True)
        b = b - np.nanmean(b, axis=1, keepdims=True)
        c = np.nansum(a * b, axis=1) / np.sqrt(np.nansum(a * a, axis=1) * np.nansum(b * b, axis=1))
    n = ok.sum(axis=1)
    return np.where(n >= 30, c, np.nan)


def _ir(v: np.ndarray) -> float | None:
    v = v[np.isfinite(v)][::4]           # 20-day outcomes sampled every 5 days overlap: keep every 4th
    return float(np.mean(v) / np.std(v)) if len(v) > 5 and np.std(v) > 0 else None


def _clusters(C: np.ndarray, names: list[str]) -> list[list[str]]:
    """Average-linkage clusters of factors whose values move together (corr >= CLUSTER_CORR)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    if len(names) == 1:
        return [names]
    D = np.clip(1 - np.nan_to_num(C, nan=0.0), 0, 2)
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)
    lab = fcluster(linkage(squareform(D, checks=False), "average"), t=1 - CLUSTER_CORR, criterion="distance")
    out: dict[int, list[str]] = {}
    for n, l in zip(names, lab):
        out.setdefault(int(l), []).append(n)
    return list(out.values())


def _composite(xs: dict, w: np.ndarray, names: list[str]) -> np.ndarray:
    """Weighted average of sampled ranks (rows = sample dates) re-ranked per row; NaN where no weight."""
    num = den = None
    for j, n in enumerate(names):
        wj = w[:, j][:, None].astype("float64")
        x = xs[n]
        ok = np.isfinite(x) & np.isfinite(wj) & (wj > 0)
        a, b = np.where(ok, x * wj, 0), np.where(ok, wj, 0)
        num = a if num is None else num + a
        den = b if den is None else den + b
    with np.errstate(all="ignore"):
        s = np.where(den > 0, num / den, np.nan)
    return pd.DataFrame(s).rank(axis=1, pct=True).values


def _ic_stats(x: np.ndarray, fwd_rank: np.ndarray, exc: np.ndarray | None, known: np.ndarray) -> dict:
    ic = _row_corr(x, fwd_rank)[known]
    v = ic[np.isfinite(ic)]
    if len(v) < 20:
        return {}
    ir = _ir(v)
    res = {"ic": round(float(np.mean(v)), 4), "ir": round(ir, 2) if ir is not None else None, "n": int(len(v)),
           "pos_share": round(float((v > 0).mean()), 3)}
    if exc is not None:
        q = pd.DataFrame(x).rank(axis=1, pct=True).values
        with np.errstate(all="ignore"):
            top = np.nanmean(np.where(q >= 0.8, exc, np.nan), axis=1)
            bot = np.nanmean(np.where(q <= 0.2, exc, np.nan), axis=1)
        spr = (top - bot)[known]
        res["q_spread"] = round(float(np.nanmean(spr)) * 100, 2) if np.isfinite(spr).sum() >= 20 else None
    return res


def compute(fs: FactorSet, store=None, base_score: pd.DataFrame | None = None) -> dict:
    """Returns {"factors": {name: health}, "signs": {name: sign}, "weights": DataFrame(date x factor), ...}."""
    cal, px = fs.calendar, fs.price
    T = len(cal)
    liq = (fs.liquid if fs.liquid is not None else pd.DataFrame(True, index=cal, columns=fs.codes)).fillna(False).values
    ind = _industry(fs)
    size = _size(fs, store)
    size_rk = size.where(liq).rank(axis=1, pct=True).values if size is not None else None
    samp = np.arange(0, T, SAMPLE_EVERY)
    S = len(samp)
    pv = px.values.astype("float64")
    ev = exit_price(px).values.astype("float64")
    fwd, fwd_old = {}, {}

    def _fwd(end, h):
        f = np.full_like(pv, np.nan)
        with np.errstate(all="ignore"):
            f[:-h] = end[h:] / pv[:-h] - 1
        return np.where(liq, f, np.nan)[samp]

    def _exc(f):
        with np.errstate(all="ignore"):
            fx = np.clip(f, -0.9, 5)
            return fx - np.nanmean(fx, axis=1, keepdims=True)

    n_recovered = 0
    for h in HORIZONS:
        f = _fwd(ev, h)          # delisted stocks valued at their last trade (no survivorship bias)
        fwd[h] = pd.DataFrame(f).rank(axis=1, pct=True).values
        if h == 20:
            fwd["exc20"] = _exc(f)
        if h in (20, 60):        # the old way (stocks that stop trading drop out), only to show the difference
            fo = _fwd(pv, h)
            fwd_old[h] = (pd.DataFrame(fo).rank(axis=1, pct=True).values, _exc(fo))
            if h == 60:
                n_recovered = int(np.isfinite(f[samp + 60 < T]).sum() - np.isfinite(fo[samp + 60 < T]).sum())
    known20 = samp + 20 < T
    years = cal[samp].year.values

    # market regimes at each sample date, from data known on that day
    with np.errstate(all="ignore"):
        r1 = np.where(liq[1:] & liq[:-1], pv[1:] / pv[:-1] - 1, np.nan)
        mr = np.nanmean(np.clip(r1, -0.5, 1), axis=1)
    mret = pd.Series(np.r_[np.nan, mr], index=cal)
    up60 = (np.log1p(mret.fillna(0)).rolling(60).sum()).values[samp] > 0
    v20 = mret.rolling(20, min_periods=15).std()
    hivol = (v20 > v20.rolling(252, min_periods=120).median()).values[samp]
    regimes = {"bull": up60, "bear": ~up60, "hivol": hivol, "lovol": ~hivol}
    regime_share = {k: round(float(v[known20].mean()), 3) for k, v in regimes.items()}

    factors, neutral, ic20, xs, xraw, nweeks = {}, {}, {}, {}, {}, {}
    for g, items in fs.score_spec.items():
        for name, sign in items:
            if name not in fs.factors or name in neutral:
                continue
            rk = (fs.factors[name] * sign).rank(axis=1, pct=True)
            nr = neutralize(rk, ind, size_rk)
            neutral[name] = sign
            x = np.where(liq[samp], nr.values[samp], np.nan).astype("float32")
            xr = np.where(liq[samp], rk.values[samp], np.nan).astype("float32")
            del nr, rk
            xs[name], xraw[name] = x, xr
            res = {"group": g, "sign": sign, "label": fs.labels.get(name, name)}
            for h in HORIZONS:
                ic = _row_corr(x, fwd[h])
                known = samp + h < T
                v = ic[known & np.isfinite(ic)]
                res[f"ic{h}"] = round(float(np.mean(v)), 4) if len(v) >= 20 else None
                if h == 20:
                    ic20[name] = np.where(known, ic, np.nan)
                    nweeks[name] = int(len(v))
                    res["weeks"] = int(len(v))
                    if len(v) >= 20:
                        ir = _ir(v)
                        res["ir20"] = round(ir, 2) if ir is not None else None
                        yr = pd.Series(ic[known & np.isfinite(ic)], index=years[known & np.isfinite(ic)]).groupby(level=0).mean()
                        res["pos_years"] = int((yr > 0).sum()); res["years"] = int(len(yr))
                        res["p"] = t_test_p(v[::4])
                        res["regime"] = {k: (round(float(np.nanmean(ic[known & msk])), 4)
                                             if np.isfinite(ic[known & msk]).sum() >= 10 else None) for k, msk in regimes.items()}
            # quintile spread of 20-day excess return and turnover
            q = pd.DataFrame(x).rank(axis=1, pct=True).values
            e = fwd["exc20"]
            with np.errstate(all="ignore"):
                top = np.nanmean(np.where(q >= 0.8, e, np.nan), axis=1)
                bot = np.nanmean(np.where(q <= 0.2, e, np.nan), axis=1)
            spr = (top - bot)[known20]
            res["q_spread20"] = round(float(np.nanmean(spr)) * 100, 2) if np.isfinite(spr).sum() >= 20 else None
            ac = _row_corr(x[:-4], x[4:]) if len(x) > 8 else np.array([np.nan])
            res["turnover"] = round(float(1 - np.nanmean(ac)), 2) if np.isfinite(ac).any() else None
            factors[name] = res
    names = list(neutral)
    F = len(names)

    # multiple testing: Benjamini-Hochberg across all factors of this market
    qv = bh_qvalues({n: factors[n].get("p") for n in names})
    for n in names:
        p = factors[n].pop("p", None)
        factors[n]["p"] = round(p, 4) if p is not None else None
        factors[n]["q"] = qv.get(n)
        factors[n]["fdr_ok"] = bool(qv.get(n) is not None and qv[n] <= FDR_Q)

    # cross-sectional correlation between (signed, neutral) factors at every sample date
    Cs = np.full((S, F, F), np.nan, dtype="float32")
    for i in range(S):
        Z = np.column_stack([xs[n][i] for n in names]).astype("float64")
        ok = np.isfinite(Z)
        cnt = ok.sum(axis=0)
        if (cnt >= 30).sum() < 2:
            continue
        Z = np.where(ok, Z - 0.5, 0.0)
        G = Z.T @ Z
        d = np.sqrt(np.diag(G))
        with np.errstate(all="ignore"):
            Ci = G / np.outer(d, d)
        bad = cnt < 30
        Ci[bad, :] = np.nan; Ci[:, bad] = np.nan
        Cs[i] = Ci.astype("float32")

    # quarterly weights without look-ahead: per-factor IR (previous method) and cluster-aware (used)
    qstarts = pd.DatetimeIndex(sorted({pd.Timestamp(d).to_period("Q").start_time for d in cal}))
    W = pd.DataFrame(np.nan, index=cal, columns=names, dtype="float32")
    WA = pd.DataFrame(np.nan, index=cal, columns=names, dtype="float32")
    latest_clusters, latest_C = None, None
    ICM = np.column_stack([ic20[n] for n in names]) if F else np.zeros((S, 0))
    for i, qs in enumerate(qstarts):
        rows = (cal >= qs) & (cal < (qstarts[i + 1] if i + 1 < len(qstarts) else cal[-1] + pd.Timedelta(days=1)))
        if not rows.any():
            continue
        q0 = int(np.argmax(rows))
        if q0 < MIN_HISTORY:
            continue
        m = (samp + 20 < q0) & (samp >= q0 - LOOKBACK_DAYS)        # outcomes known before the quarter starts
        if m.sum() < 40:
            continue
        irs = {n: (_ir(ic20[n][m]) or 0.0) for n in names}
        wa = {n: max(0.0, irs[n]) for n in names}
        if sum(wa.values()) <= 0:
            continue
        WA.loc[rows, :] = [wa[n] for n in names]
        # cluster-aware weights; factors still accumulating history get none
        ready = [j for j, n in enumerate(names) if np.isfinite(ic20[n][m]).sum() >= MIN_WEEKS_FOR_WEIGHT]
        if not ready:
            continue
        mc = (samp < q0) & (samp >= q0 - LOOKBACK_DAYS)
        with np.errstate(all="ignore"):
            C = np.nanmean(Cs[mc][:, ready][:, :, ready], axis=0)
        cl = _clusters(C, [names[j] for j in ready])
        w = {n: 0.0 for n in names}
        for members in cl:
            idx = [names.index(n) for n in members]
            with np.errstate(all="ignore"):
                cic = np.nanmean(ICM[m][:, idx], axis=1)
            cir = max(0.0, _ir(cic) or 0.0)
            if cir <= 0:
                continue
            pos = {n: irs[n] for n in members if irs[n] > 0}
            if pos:
                tot = sum(pos.values())
                for n, v in pos.items():
                    w[n] = cir * v / tot
            else:
                for n in members:
                    w[n] = cir / len(members)
        if sum(w.values()) <= 0:
            continue
        W.loc[rows, :] = [w[n] for n in names]
        latest_clusters = cl
        latest_C = (C, [names[j] for j in ready])
    latest, latest_a = W.iloc[-1], WA.iloc[-1]
    tot = float(np.nansum(latest.values)) or 1.0
    tot_a = float(np.nansum(latest_a.values)) or 1.0
    for n in names:
        factors[n]["weight"] = round(float(latest[n]) / tot * 100, 1) if np.isfinite(latest[n]) else None
        factors[n]["weight_ir"] = round(float(latest_a[n]) / tot_a * 100, 1) if np.isfinite(latest_a[n]) else None
        factors[n]["accumulating"] = nweeks.get(n, 0) < MIN_WEEKS_FOR_WEIGHT

    # composite comparison on the same sample dates (only where both weight sets exist)
    comp = {}
    Wv, WAv = W.values[samp], WA.values[samp]
    has_w = np.isfinite(Wv).any(axis=1) & np.isfinite(WAv).any(axis=1) & known20
    method = pd.Series("cluster_neutral", index=cal)        # which composite the score uses on each day
    if has_w.sum() >= 20 and F:
        exc = fwd["exc20"]
        variants = {"cluster_neutral": _composite(xs, Wv, names), "ir_neutral": _composite(xs, WAv, names),
                    "ir_raw": _composite(xraw, WAv, names)}
        # every quarter, use the method whose composite had the higher 20-day IC over the previous 3 years
        # (outcomes already known); until a year of composite history exists, the cluster method
        icv = {k: np.where(has_w, _row_corr(v, fwd[20]), np.nan) for k, v in variants.items()}
        chosen = []
        for i, qs in enumerate(qstarts):
            rows = (cal >= qs) & (cal < (qstarts[i + 1] if i + 1 < len(qstarts) else cal[-1] + pd.Timedelta(days=1)))
            if not rows.any():
                continue
            q0 = int(np.argmax(rows))
            m = (samp + 20 < q0) & (samp >= q0 - LOOKBACK_DAYS)
            best = "cluster_neutral"
            if np.isfinite(icv[best][m]).sum() >= 50:
                means = {k: float(np.nanmean(v[m])) for k, v in icv.items() if np.isfinite(v[m]).sum() >= 50}
                best = max(means, key=means.get)
            method[rows] = best
            chosen.append((str(qs.date()), best))
        ms = method.values[samp]
        variants["chosen"] = np.where((ms == "cluster_neutral")[:, None], variants["cluster_neutral"],
                                      np.where((ms == "ir_neutral")[:, None], variants["ir_neutral"], variants["ir_raw"]))
        comp["_method_now"] = method.iloc[-1]
        comp["_method_history"] = {k: sum(1 for _, b in chosen if b == k) for k in variants}
        if base_score is not None:
            b = np.where(liq[samp], base_score.reindex(index=cal, columns=fs.codes).values[samp], np.nan)
            variants["equal_raw"] = pd.DataFrame(b).rank(axis=1, pct=True).values
        for k, x in variants.items():
            comp[k] = _ic_stats(x, fwd[20], exc, has_w)
            kn60 = has_w & (samp + 60 < T)
            comp[k]["ic60"] = _ic_stats(x, fwd[60], None, kn60).get("ic")
        comp["_from"] = str(cal[samp[has_w]][0].date())
        # survivorship: the composite judged with delisted stocks dropped (old) vs kept (new)
        x = variants["cluster_neutral"]
        surv = {}
        for h in (20, 60):
            kn = has_w & (samp + h < T)
            surv[f"ic{h}_new"] = _ic_stats(x, fwd[h], None, kn).get("ic")
            surv[f"ic{h}_old"] = _ic_stats(x, fwd_old[h][0], None, kn).get("ic")
        surv["q_spread20_new"] = comp["cluster_neutral"].get("q_spread")
        surv["q_spread20_old"] = _ic_stats(x, fwd_old[20][0], fwd_old[20][1], has_w).get("q_spread")
        surv["samples_recovered_60"] = n_recovered
        comp["_survivorship"] = surv
    del xraw
    # the weights actually used per day follow the chosen method
    use_ir = method.isin(["ir_neutral", "ir_raw"]).values
    Wf = W.copy()
    Wf.loc[use_ir, :] = WA.loc[use_ir, :].values
    raw_days = method.eq("ir_raw").values
    now = method.iloc[-1]
    if now != "cluster_neutral":
        for n in names:
            factors[n]["weight"], factors[n]["weight_cluster"] = factors[n]["weight_ir"], factors[n]["weight"]

    # 借券／融券 split test (Taiwan): is the "wrong" direction coming from hedging-type short selling?
    split = {}
    if fs.market == "tw" and "dealer_hedge_20" in fs.factors:
        hedge = (fs.factors["dealer_hedge_20"].abs() > 0).values[samp] & liq[samp]
        big = (size_rk[samp] >= 2 / 3) if size_rk is not None else np.zeros_like(hedge)
        for n in ("sbl_ratio", "sbl_dtc", "sbl_sell_20", "sbl_chg_20", "short_dtc"):
            if n not in xs:
                continue
            res = {}
            for k, msk in (("hedged", hedge), ("unhedged", ~hedge & liq[samp]), ("large", big & liq[samp]),
                           ("small_mid", ~big & liq[samp])):
                xm = pd.DataFrame(np.where(msk, xs[n], np.nan)).rank(axis=1, pct=True).values
                ic = _row_corr(xm, np.where(msk, fwd[20], np.nan))
                v = ic[known20 & np.isfinite(ic)]
                p = t_test_p(v[::4]) if len(v) >= 30 else None
                res[k] = {"ic": round(float(np.mean(v)), 4) if len(v) >= 20 else None,
                          "p": round(p, 4) if p is not None else None}
            split[n] = {"sign": neutral[n], **res}

    corr_out = None
    if latest_C is not None:
        C, nm = latest_C
        corr_out = {"names": nm, "m": [[round(float(v), 2) if np.isfinite(v) else None for v in row] for row in C]}
    log.info("health: %d factors (%d pass FDR), neutralized by %s%s; %d clusters; weights active from %s", len(names),
             sum(1 for n in names if factors[n]["fdr_ok"]), "industry" if ind is not None else "-",
             "+size" if size_rk is not None else "", len(latest_clusters or []), W.dropna(how="all").index.min())
    return {"factors": factors, "signs": neutral, "weights": Wf, "raw_days": raw_days, "method": now, "ind": ind, "size_rk": size_rk,
            "neutralized_by": [x for x, ok in (("industry", ind is not None), ("size", size_rk is not None)) if ok],
            "clusters": latest_clusters or [], "corr": corr_out, "composite": comp, "sbl_split": split,
            "regime_share": regime_share, "fdr_q": FDR_Q, "min_weeks": MIN_WEEKS_FOR_WEIGHT}


def weighted_score(fs: FactorSet, h: dict, base_score: pd.DataFrame) -> pd.DataFrame:
    """Composite from neutralized ranks (raw ranks on days the raw method was chosen) with the quarterly
    weights; rows without weights keep the base score."""
    W = h["weights"]
    raw_days = h.get("raw_days")
    live = alive(fs)
    num = np.zeros((len(fs.calendar), len(fs.codes)))
    den = np.zeros_like(num)
    for n, sign in h["signs"].items():
        w = W[n].values[:, None].astype("float64")
        if not (np.isfinite(w) & (w > 0)).any():
            continue
        rk = (fs.factors[n] * sign).where(live).rank(axis=1, pct=True)
        v = neutralize(rk, h["ind"], h["size_rk"]).values.copy()
        if raw_days is not None and raw_days.any():
            v[raw_days] = rk.values[raw_days]
        ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
        num += np.where(ok, v * w, 0)
        den += np.where(ok, w, 0)
    with np.errstate(all="ignore"):
        s = np.where(den > 0, num / den, np.nan)
    s = pd.DataFrame(s, index=fs.calendar, columns=fs.codes).rank(axis=1, pct=True) * 100
    has_w = W.notna().any(axis=1).values[:, None]
    out = np.where(has_w & np.isfinite(s.values), s.values, base_score.values)
    return pd.DataFrame(out, index=fs.calendar, columns=fs.codes).astype("float32")
