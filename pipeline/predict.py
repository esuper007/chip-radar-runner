"""Forward-return forecasts from chip state: (A) historical analogs (k-NN) and (B) LightGBM.

Both give, for horizons 63/126/252 trading days (≈1 quarter / half year / 1 year):
  p_up  probability the (dividend-adjusted) return is > 0
  q10 / q50 / q90  return quantiles in %
Backtests are walk-forward with a purge gap equal to the horizon, so no future data leaks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import log
from .factors import FactorSet, exit_price

HORIZONS = {"1q": 63, "6m": 126, "1y": 252}
SAMPLE_EVERY = 5          # use one training row per stock per week
BT_MAX_TRAIN = 300_000    # backtest only: rows per training fold
BT_MAX_QUERY = 8_000      # backtest only: analog queries per test year
BT_ANALOG_POOL = 120_000  # backtest only: analog search pool
MIN_LABEL_WEEKS = 100     # forecasts are withheld until ~2 years of outcomes exist
EXCLUDE = {"holders"}     # absolute levels that are not comparable across stocks


# ---------------------------------------------------------------- dataset
def feature_panel(fs: FactorSet, scores: dict[str, pd.DataFrame], dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Long frame (date, code) of rank-transformed features for the given dates."""
    cols = {}
    for name, w in list(fs.factors.items()) + list(scores.items()):
        if name in EXCLUDE:
            continue
        sub = w.reindex(dates)
        cols[name] = sub.rank(axis=1, pct=True) if not name.startswith("s_") and name != "score" else sub / 100
    # market regime features (same for all stocks on a date, time-series percentile)
    ret60 = fs.factors.get("ret_60")
    if ret60 is not None:
        mk = ret60.median(axis=1)
        cols["mkt_ret_60"] = pd.DataFrame(np.repeat(mk.rank(pct=True).reindex(dates).values[:, None],
                                                    len(fs.codes), axis=1), index=dates, columns=fs.codes)
    for i, (nm, s) in enumerate(sorted(fs.market_series.items())):
        z = s.rolling(20, min_periods=5).sum() if "買賣超" in nm else s if "未平倉" in nm else s.pct_change(20, fill_method=None)
        z = z.reindex(fs.calendar).ffill().rank(pct=True).reindex(dates)
        cols[f"mkt_s{i}"] = pd.DataFrame(np.repeat(z.values[:, None], len(fs.codes), axis=1),
                                                             index=dates, columns=fs.codes)
    long = pd.concat({k: v.stack(future_stack=True) for k, v in cols.items()}, axis=1)
    long.index.names = ["date", "code"]
    return long.astype("float32")


def labels(fs: FactorSet, dates: pd.DatetimeIndex) -> pd.DataFrame:
    px = fs.price
    ex = exit_price(px)            # a delisted stock keeps its last traded price (no survivorship bias)
    out = {}
    for hk, h in HORIZONS.items():
        fwd = (ex.shift(-h) / px - 1) * 100
        out[f"y_{hk}"] = fwd.reindex(dates).stack(future_stack=True)
    y = pd.concat(out, axis=1)
    y.index.names = ["date", "code"]
    return y


def training_frame(fs: FactorSet, scores) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = fs.calendar[::-1][::SAMPLE_EVERY][::-1]
    X = feature_panel(fs, scores, dates)
    Y = labels(fs, dates)
    liq = fs.liquid.reindex(dates).stack(future_stack=True) if fs.liquid is not None else None
    df = X.join(Y)
    if liq is not None:
        df = df[liq.reindex(df.index).fillna(False).values]
    # need at least the composite score to be present
    if "score" in df:
        df = df[df["score"].notna()]
    return df[X.columns], df[Y.columns]


# ---------------------------------------------------------------- (A) analogs
ANALOG_FEATS_PREFIX = ("s_", "ret_", "vol_60", "dist_52w_high", "mkt_")


def _analog_cols(cols):
    return [c for c in cols if c.startswith(ANALOG_FEATS_PREFIX)]


def analog_predict(Xtr: pd.DataFrame, ytr: pd.Series, Xq: pd.DataFrame, k: int = 300,
                   max_pool: int = 400_000, seed: int = 0) -> pd.DataFrame:
    from sklearn.neighbors import NearestNeighbors
    cols = _analog_cols(Xtr.columns)
    m = ytr.notna().values
    P, y = Xtr.loc[m, cols], ytr.values[m]
    if len(P) > max_pool:
        idx = np.random.default_rng(seed).choice(len(P), max_pool, replace=False)
        P, y = P.iloc[idx], y[idx]
    if len(P) < k * 2:
        return pd.DataFrame(index=Xq.index)
    nn = NearestNeighbors(n_neighbors=k).fit(P.fillna(0.5).values)
    _, ind = nn.kneighbors(Xq[cols].fillna(0.5).values)
    Ry = y[ind]
    q = np.percentile(Ry, [10, 25, 50, 75, 90], axis=1)
    return pd.DataFrame({"p_up": (Ry > 0).mean(axis=1), "q10": q[0], "q25": q[1], "q50": q[2],
                         "q75": q[3], "q90": q[4], "mean": Ry.mean(axis=1)}, index=Xq.index)


# ---------------------------------------------------------------- (B) LightGBM
LGB_PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=200,
                  subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
                  n_jobs=-1, verbose=-1)


def ml_fit(X: pd.DataFrame, y: pd.Series) -> dict:
    import lightgbm as lgb
    m = y.notna()
    X, y = X[m], y[m]
    models = {"cls": lgb.LGBMClassifier(**LGB_PARAMS).fit(X, (y > 0).astype(int))}
    for a in (0.1, 0.5, 0.9):
        models[f"q{int(a * 100)}"] = lgb.LGBMRegressor(objective="quantile", alpha=a, **LGB_PARAMS).fit(X, y)
    return models


def ml_predict(models: dict, X: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=X.index)
    out["p_up"] = models["cls"].predict_proba(X)[:, 1]
    q = np.sort(np.vstack([models[k].predict(X) for k in ("q10", "q50", "q90")]), axis=0)  # no crossing
    out["q10"], out["q50"], out["q90"] = q
    return out


def feature_importance(models: dict, cols, top=15) -> list[dict]:
    imp = pd.Series(models["cls"].booster_.feature_importance("gain"), index=cols)
    imp = (imp / imp.sum() * 100).sort_values(ascending=False).head(top)
    return [{"feature": k, "gain_pct": round(float(v), 2)} for k, v in imp.items()]


# ---------------------------------------------------------------- backtest
def _metrics(pred: pd.DataFrame, y: pd.Series) -> dict:
    from sklearn.metrics import roc_auc_score
    d = pred.join(y.rename("y")).dropna(subset=["y", "p_up"])
    if len(d) < 500:
        return {}
    up = d["y"] > 0
    res = {"n": int(len(d)), "base_rate_up": round(float(up.mean()), 3),
           "hit_rate": round(float(((d["p_up"] > 0.5) == up).mean()), 3),
           "auc": round(float(roc_auc_score(up, d["p_up"])), 3),
           "brier": round(float(((d["p_up"] - up) ** 2).mean()), 4)}
    if "q10" in d and "q90" in d:
        res["coverage_80"] = round(float(((d["y"] >= d["q10"]) & (d["y"] <= d["q90"])).mean()), 3)
    key = "q50" if "q50" in d else "p_up"
    ics, spreads = [], []
    for _, g in d.groupby(level="date"):
        if len(g) < 30:
            continue
        ics.append(g[key].rank().corr(g["y"].rank()))
        qn = pd.qcut(g[key].rank(method="first"), 5, labels=False)
        spreads.append(g["y"][qn == 4].mean() - g["y"][qn == 0].mean())
    if ics:
        res["rank_ic"] = round(float(np.nanmean(ics)), 3)
        res["ic_ir"] = round(float(np.nanmean(ics) / (np.nanstd(ics) + 1e-9)), 2)
        res["top_minus_bottom_pct"] = round(float(np.nanmean(spreads)), 2)
    return res


def backtest(X: pd.DataFrame, Y: pd.DataFrame, cal: pd.DatetimeIndex, first_test_year: int | None = None) -> dict:
    """Yearly walk-forward. Train on rows whose label window ends before the test year starts."""
    dates = X.index.get_level_values("date")
    years = sorted(set(dates.year))
    if first_test_year is None:
        first_test_year = years[0] + 4 if len(years) > 6 else years[min(2, len(years) - 1)]
    out = {}
    bt_preds, calib = {}, {}
    for hk, h in HORIZONS.items():
        y = Y[f"y_{hk}"]
        preds_ml, preds_an, score_rows = [], [], []
        for yr in [y_ for y_ in years if y_ >= first_test_year]:
            test_start = pd.Timestamp(f"{yr}-01-01")
            pos = cal.searchsorted(test_start)
            cutoff = cal[max(0, pos - h - 1)]          # purge: label must end before test year
            tr = dates <= cutoff
            te = (dates >= test_start) & (dates < pd.Timestamp(f"{yr + 1}-01-01"))
            if tr.sum() < 20_000 or te.sum() == 0:
                continue
            # Backtests are for measuring quality, not for the live model: cap the work per fold so
            # large markets (US ~7k stocks x 12 years) finish in minutes instead of hours.
            rng = np.random.default_rng(yr)
            tr_idx = np.where(tr)[0]
            if len(tr_idx) > BT_MAX_TRAIN:
                tr_idx = np.sort(rng.choice(tr_idx, BT_MAX_TRAIN, replace=False))
            m = ml_fit(X.iloc[tr_idx], y.iloc[tr_idx])
            preds_ml.append(ml_predict(m, X[te]))
            # analogs on monthly test dates, at most BT_MAX_QUERY stocks-dates per year
            te_dates = sorted(set(dates[te]))[::4]
            q_idx = np.where(dates.isin(te_dates))[0]
            if len(q_idx) > BT_MAX_QUERY:
                q_idx = np.sort(rng.choice(q_idx, BT_MAX_QUERY, replace=False))
            preds_an.append(analog_predict(X.iloc[tr_idx], y.iloc[tr_idx], X.iloc[q_idx], max_pool=BT_ANALOG_POOL))
            log.info("backtest %s %d: train=%d test=%d", hk, yr, tr.sum(), te.sum())
        res = {}
        P = {}
        if preds_ml:
            P["ml"] = pd.concat(preds_ml)
            res["ml"] = _metrics(P["ml"], y)
        if preds_an:
            P["an"] = pd.concat(preds_an)
            res["analog"] = _metrics(P["an"], y)
        iso = is_oos_fit(X, y, h, cal)
        if iso is not None:
            P["is"] = iso["oos_pred"]
        bt_preds[hk] = P
        for mk, pr in P.items():
            c = _calibrate(pr, y, cal, h)
            if c:
                calib.setdefault(mk, {})[hk] = c
        if "score" in X:
            sc = X[["score"]].rename(columns={"score": "p_up"})
            res["score_only"] = {k: v for k, v in _metrics(sc[dates >= pd.Timestamp(f"{first_test_year}-01-01")], y).items()
                                 if k in ("n", "rank_ic", "ic_ir", "top_minus_bottom_pct")}
        out[hk] = res
    out["calib"] = calib
    try:
        out["blend"] = _blend_weights(bt_preds, Y, calib)
    except Exception:
        log.exception("blend weights failed")
    try:
        out["_picks"] = _list_picks(bt_preds, calib, out.get("blend"))
    except Exception:
        log.exception("backtest list picks failed")
    # which chip groups actually predicted returns (rank IC of each sub-score)
    out["group_ic"] = {}
    for c in [c for c in X.columns if c.startswith("s_")]:
        ics = []
        y = Y["y_1q"]
        for _, g in X[[c]].join(y).dropna().groupby(level="date"):
            if len(g) > 30:
                ics.append(g[c].rank().corr(g["y_1q"].rank()))
        if ics:
            out["group_ic"][c] = round(float(np.nanmean(ics)), 3)
    return out


# ---------------------------------------------------------------- calibration and blend weights
CAL_BINS = np.linspace(0, 1, 11)


def _bins(p: np.ndarray, up: np.ndarray) -> list[dict]:
    """Reliability table: predicted probability in 10 bins vs how often the stock really went up."""
    out = []
    for lo, hi in zip(CAL_BINS[:-1], CAL_BINS[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.sum() >= 30:
            out.append({"lo": round(float(lo), 1), "p": round(float(p[m].mean()), 3),
                        "up": round(float(up[m].mean()), 3), "n": int(m.sum())})
    return out


def _calibrate(pred: pd.DataFrame, y: pd.Series, cal: pd.DatetimeIndex, h: int) -> dict | None:
    """Walk-forward isotonic calibration of p_up: for each test year, fit on out-of-sample predictions whose
    outcome was known before that year starts, apply to the year. Kept only if it lowers the Brier score."""
    from sklearn.isotonic import IsotonicRegression
    d = pred[["p_up"]].join(y.rename("y")).dropna()
    if len(d) < 3000:
        return None
    dates = d.index.get_level_values("date")
    p, up = d["p_up"].values.astype("float64"), (d["y"].values > 0).astype("float64")
    pc = np.full(len(d), np.nan)
    for yr in sorted(set(dates.year)):
        start = pd.Timestamp(f"{yr}-01-01")
        cut = cal[max(0, cal.searchsorted(start) - h - 1)]
        fit, te = dates <= cut, dates.year == yr
        if fit.sum() < 3000 or te.sum() == 0 or up[fit].min() == up[fit].max():
            continue
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.05, y_max=0.95).fit(p[fit], up[fit])
        pc[te] = iso.predict(p[te])
    ev = np.isfinite(pc)
    res = {"n": int(len(d)), "bins_raw": _bins(p, up), "brier_all": round(float(np.mean((p - up) ** 2)), 4)}
    if ev.sum() >= 1000:
        res["n_eval"] = int(ev.sum())
        res["brier_raw"] = round(float(np.mean((p[ev] - up[ev]) ** 2)), 4)
        res["brier_cal"] = round(float(np.mean((pc[ev] - up[ev]) ** 2)), 4)
        res["bins_cal"] = _bins(pc[ev], up[ev])
        res["apply"] = bool(res["brier_cal"] < res["brier_raw"] * 0.995)
    else:
        res["apply"] = False
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.05, y_max=0.95).fit(p, up)
    xs, ys = iso.X_thresholds_, iso.y_thresholds_
    if len(xs) > 80:
        k = np.linspace(0, len(xs) - 1, 80).astype(int)
        xs, ys = xs[k], ys[k]
    res["map"] = {"x": [round(float(v), 4) for v in xs], "y": [round(float(v), 4) for v in ys]}
    return res


def apply_calibration(p: np.ndarray, c: dict | None) -> np.ndarray:
    if not c or not c.get("apply") or not c.get("map"):
        return p
    return np.interp(p, c["map"]["x"], c["map"]["y"])


def _cal_p(P: dict, calib: dict, hk: str, mk: str) -> pd.Series:
    return pd.Series(apply_calibration(P[mk]["p_up"].values, (calib.get(mk) or {}).get(hk)), index=P[mk].index)


def _daily_ic(score: pd.Series, y: pd.Series) -> float | None:
    d = pd.DataFrame({"s": score, "y": y}).dropna()
    ics = [g["s"].rank().corr(g["y"].rank()) for _, g in d.groupby(level="date") if len(g) >= 30]
    return float(np.nanmean(ics)) if ics else None


BLEND_DEFAULT = {"is": 70, "ml": 15, "an": 15}


def _blend_weights(bt_preds: dict, Y: pd.DataFrame, calib: dict) -> dict | None:
    """Weights of the three models for the ranking blend, from out-of-sample rows where all three exist:
    the grid weights (5% steps) with the highest average daily rank IC over the three horizons, shrunk
    halfway toward the default 70/15/15 so a lucky grid point cannot dominate."""
    sets = {}
    for hk, P in bt_preds.items():
        if not all(k in P for k in ("is", "ml", "an")):
            continue
        J = pd.concat({k: _cal_p(P, calib, hk, k) for k in ("is", "ml", "an")}, axis=1).join(Y[f"y_{hk}"].rename("y")).dropna()
        if J.index.get_level_values("date").nunique() >= 12:
            sets[hk] = J
    if not sets:
        return None
    grid = [(a, b, 100 - a - b) for a in range(0, 101, 5) for b in range(0, 101 - a, 5)]

    def score(w):
        vals = []
        for J in sets.values():
            s = (J["is"] * w[0] + J["ml"] * w[1] + J["an"] * w[2]) / 100
            v = _daily_ic(s, J["y"])
            if v is not None:
                vals.append(v)
        return float(np.mean(vals)) if vals else -1

    res = {w: score(w) for w in grid}
    best = max(res, key=res.get)
    d = BLEND_DEFAULT
    final = [round((best[i] + d[k]) / 2) for i, k in enumerate(("is", "ml", "an"))]
    final[0] += 100 - sum(final)
    d = BLEND_DEFAULT
    out = {"weights": dict(zip(("is", "ml", "an"), final)), "best_grid": dict(zip(("is", "ml", "an"), best)),
           "ic": {"best_grid": round(res[best], 4), "final": round(score(tuple(final)), 4),
                  "default": round(score((d["is"], d["ml"], d["an"])), 4), "equal": round(score((34, 33, 33)), 4),
                  "is_only": round(res[(100, 0, 0)], 4), "ml_only": round(res[(0, 100, 0)], 4), "an_only": round(res[(0, 0, 100)], 4)},
           "n_dates": {hk: int(J.index.get_level_values("date").nunique()) for hk, J in sets.items()},
           "period": {hk: f"{J.index.get_level_values('date').min().date()} ~ {J.index.get_level_values('date').max().date()}" for hk, J in sets.items()},
           "quarter": str(pd.Timestamp.today().to_period("Q"))}
    log.info("blend weights: %s (avg IC final=%s default=%s)", out["weights"], out["ic"]["final"], out["ic"]["default"])
    return out


def _list_picks(bt_preds: dict, calib: dict, blend: dict | None, topn: int = 20) -> dict:
    """The daily lists of the live track record, rebuilt on the backtest (weekly, out of sample), so the live
    results can be compared with what history says to expect. Uses the in/out-of-sample model and LightGBM
    (the analog model only scores a sample of stocks in the backtest), renormalized to their blend weights."""
    w = (blend or {}).get("weights") or BLEND_DEFAULT
    out = {}
    for hk, lists in (("1q", (("up_1q", "p", -1), ("down_1q", "p", 1), ("gain_1q", "q", -1))), ("1y", (("up_1y", "p", -1),))):
        P = bt_preds.get(hk) or {}
        if not ("is" in P and "ml" in P):
            continue
        wi, wm = w["is"] / (w["is"] + w["ml"]), w["ml"] / (w["is"] + w["ml"])
        J = pd.DataFrame({"pi": _cal_p(P, calib, hk, "is"), "pm": _cal_p(P, calib, hk, "ml"),
                          "qi": P["is"]["q50"], "qm": P["ml"]["q50"]}).dropna()
        J["p"] = J["pi"] * wi + J["pm"] * wm
        J["q"] = J["qi"] * wi + J["qm"] * wm
        for name, col, direction in lists:
            rows = []
            for d, g in J.groupby(level="date"):
                if len(g) < topn * 3:
                    continue
                g = g.sort_values(col, ascending=direction > 0).head(topn)
                rows.append(pd.DataFrame({"date": d, "code": g.index.get_level_values("code")}))
            if rows:
                out[name] = pd.concat(rows, ignore_index=True)
    return out


# ---------------------------------------------------------------- (C) in-sample model, out-of-sample check
IS_SHARE = 0.7            # first 70% of dated samples are in-sample; the rest is held out


def _is_features(X: pd.DataFrame) -> list[str]:
    """Columns usable by a linear model: present in at least half of the rows."""
    return [c for c in X.columns if X[c].notna().mean() >= 0.5]


def _basic_metrics(p_up, pred, y, dates) -> dict:
    from sklearn.metrics import r2_score, roc_auc_score
    up = y > 0
    res = {"n": int(len(y)), "base_rate_up": round(float(up.mean()), 3),
           "hit_rate": round(float(((p_up > 0.5) == up).mean()), 3),
           "r2": round(float(r2_score(y, pred)), 4)}
    try:
        res["auc"] = round(float(roc_auc_score(up, p_up)), 3)
    except ValueError:
        pass
    d = pd.DataFrame({"pred": pred, "y": y, "date": dates})
    ics, spreads = [], []
    for _, g in d.groupby("date"):
        if len(g) < 30:
            continue
        ics.append(g["pred"].rank().corr(g["y"].rank()))
        qn = pd.qcut(g["pred"].rank(method="first"), 5, labels=False)
        spreads.append(g["y"][qn == 4].mean() - g["y"][qn == 0].mean())
    if ics:
        res["rank_ic"] = round(float(np.nanmean(ics)), 3)
        res["top_minus_bottom_pct"] = round(float(np.nanmean(spreads)), 2)
    return res


def is_oos_fit(X: pd.DataFrame, y: pd.Series, h: int, cal: pd.DatetimeIndex) -> dict | None:
    """Build on the in-sample period, test on the held-out period, keep the in-sample model.

    The model that makes today's forecast is exactly the one that was tested out of sample,
    and the forecast interval comes from out-of-sample errors, so it reflects real accuracy.
    """
    from sklearn.linear_model import LogisticRegression, Ridge
    m = y.notna()
    X, y = X[m], y[m]
    dates = X.index.get_level_values("date")
    ud = np.sort(dates.unique())
    if len(ud) < MIN_LABEL_WEEKS:
        return None
    split = ud[int(len(ud) * IS_SHARE)]
    pos = cal.searchsorted(split)
    oos_start = cal[min(len(cal) - 1, pos + h)]          # purge: in-sample outcomes end before OOS starts
    ins, oos = dates < split, dates >= oos_start
    if ins.sum() < 5_000 or oos.sum() < 2_000:
        return None
    cols = _is_features(X[ins])
    Xi, Xo = X.loc[ins, cols].fillna(0.5).values, X.loc[oos, cols].fillna(0.5).values
    lo, hi = np.nanpercentile(y[ins], [1, 99])
    yi = y[ins].clip(lo, hi).values
    reg = Ridge(alpha=50.0).fit(Xi, yi)
    clf = LogisticRegression(C=0.1, max_iter=500).fit(Xi, (y[ins] > 0).astype(int).values)
    pi, po = reg.predict(Xi), reg.predict(Xo)
    qi, qo = clf.predict_proba(Xi)[:, 1], clf.predict_proba(Xo)[:, 1]
    resid = y[oos].values - po
    rq = np.percentile(resid, [10, 50, 90])
    oos_m = _basic_metrics(qo, po, y[oos].values, dates[oos])
    oos_m["coverage_80"] = round(float(((resid >= rq[0]) & (resid <= rq[2])).mean()), 3)
    oos_pred = pd.DataFrame({"p_up": qo, "q50": po + rq[1]}, index=X.index[oos])
    coef = pd.Series(reg.coef_, index=cols)
    top = coef.reindex(coef.abs().sort_values(ascending=False).index).head(12)
    return {
        "reg": reg, "clf": clf, "cols": cols, "resid_q": rq.tolist(), "oos_pred": oos_pred,
        "metrics": {
            "in_sample": _basic_metrics(qi, pi, y[ins].values, dates[ins]),
            "out_of_sample": oos_m,
            "split": {"in_sample": f"{pd.Timestamp(ud[0]).date()} ~ {pd.Timestamp(split).date()}",
                      "out_of_sample": f"{pd.Timestamp(oos_start).date()} ~ {pd.Timestamp(ud[-1]).date()}"},
            # coefficient = change in expected % return when the factor moves from the bottom to the top of the market
            "coef": [{"feature": k, "coef": round(float(v), 2)} for k, v in top.items()],
        },
    }


def is_predict(model: dict, Xq: pd.DataFrame) -> pd.DataFrame:
    Z = Xq.reindex(columns=model["cols"]).fillna(0.5).values
    pred = model["reg"].predict(Z)
    q10, q50, q90 = model["resid_q"]
    return pd.DataFrame({"p_up": model["clf"].predict_proba(Z)[:, 1], "q10": pred + q10,
                         "q50": pred + q50, "q90": pred + q90}, index=Xq.index)


# ---------------------------------------------------------------- training (weekly) and live forecast (daily)
def train_models(fs: FactorSet, scores) -> dict:
    """Fit every method for every horizon. Saved and reused on weekdays (see run.py)."""
    X, Y = training_frame(fs, scores)
    dates = X.index.get_level_values("date")
    bundle = {"cols": list(X.columns), "horizons": {}, "info": {"importance": {}, "train_rows": {}, "train_weeks": {},
              "is_oos": {}, "trained_at": str(pd.Timestamp.today().date()),
              "history_start": str(dates.min().date()) if len(dates) else None}}
    for hk, h in HORIZONS.items():
        y = Y[f"y_{hk}"]
        n_lab = int(y.notna().sum())
        n_dates = int(dates[y.notna().values].nunique()) if n_lab else 0
        bundle["info"]["train_rows"][hk] = n_lab
        bundle["info"]["train_weeks"][hk] = n_dates
        # need ~2 years of weekly samples with known outcomes before a forecast means anything
        if n_lab < 5_000 or n_dates < MIN_LABEL_WEEKS:
            log.warning("too little history for %s (%d rows, %d weeks) - skipping", hk, n_lab, n_dates)
            continue
        m = y.notna()
        cols = _analog_cols(X.columns)
        pool, py = X.loc[m, cols], y[m].values
        if len(pool) > 400_000:
            idx = np.random.default_rng(0).choice(len(pool), 400_000, replace=False)
            pool, py = pool.iloc[idx], py[idx]
        hz = {"analog_pool": (pool.astype("float32"), py.astype("float32")), "ml": ml_fit(X, y)}
        bundle["info"]["importance"][hk] = feature_importance(hz["ml"], X.columns)
        iso = is_oos_fit(X, y, h, fs.calendar)
        if iso:
            iso.pop("oos_pred", None)
            hz["is"] = iso
            bundle["info"]["is_oos"][hk] = iso["metrics"]
        bundle["horizons"][hk] = hz
        log.info("trained %s: %d rows, %d weeks", hk, n_lab, n_dates)
    return bundle


def forecast(fs: FactorSet, scores, bundle: dict, asof: pd.Timestamp | None = None,
             calib: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """calib: {model: {horizon: calibration}} from the backtest; applied to p_up where it helped out of sample."""
    asof = asof or fs.calendar[-1]
    Xq = feature_panel(fs, scores, pd.DatetimeIndex([asof]))
    Xq = Xq[Xq["score"].notna()] if "score" in Xq else Xq
    Xq = Xq.reindex(columns=bundle["cols"])
    frames = []
    for hk, hz in bundle["horizons"].items():
        pool, py = hz["analog_pool"]
        frames.append(analog_predict(pool, pd.Series(py, index=pool.index), Xq).add_prefix(f"an_{hk}_"))
        frames.append(ml_predict(hz["ml"], Xq).add_prefix(f"ml_{hk}_"))
        if "is" in hz:
            frames.append(is_predict(hz["is"], Xq).add_prefix(f"is_{hk}_"))
    res = pd.concat(frames, axis=1) if frames else pd.DataFrame(index=Xq.index)
    for col in [c for c in res.columns if c.endswith("_p_up")]:
        mk, hk = col.split("_")[:2]
        c = ((calib or {}).get("an" if mk == "an" else mk) or {}).get(hk)
        if c and c.get("apply"):
            res[col] = apply_calibration(res[col].values.astype("float64"), c)
    res = res.reset_index(level="date", drop=True)
    return res, bundle["info"]
