"""Adjusted daily prices for any market via Yahoo Finance (yfinance), in batches.

Stored as dataset `price`: date, code, adj_close, close, volume.
"""
from __future__ import annotations

import datetime as dt
import time

import numpy as np
import pandas as pd

from .common import log
from .store import Store

import logging as _logging
_logging.getLogger("yfinance").setLevel(_logging.CRITICAL)   # delisted-ticker noise is expected, not an error


def download(symbols: dict[str, str], start: dt.date, end: dt.date | None = None,
             batch: int = 150, splits: list | None = None) -> pd.DataFrame:
    """symbols: {code: yahoo_symbol}. Returns long frame. When `splits` is a list, stock splits in the
    period are appended to it as {date, code, ratio} (Yahoo's ratio: 10 = 10-for-1, 0.1 = 1-for-10)."""
    import yfinance as yf

    inv = {v: k for k, v in symbols.items()}
    tickers = list(symbols.values())
    out = []
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        for attempt in range(3):
            try:
                df = yf.download(chunk, start=start.isoformat(),
                                 end=(end + dt.timedelta(days=1)).isoformat() if end else None,
                                 auto_adjust=False, actions=splits is not None, group_by="ticker",
                                 threads=True, progress=False)
                break
            except Exception as e:  # rate limit / network
                log.warning("yfinance batch %d failed (%s), retrying", i, e)
                time.sleep(30 * (attempt + 1))
        else:
            continue
        if df is None or df.empty:
            continue
        if not isinstance(df.columns, pd.MultiIndex):  # single ticker
            df.columns = pd.MultiIndex.from_product([[chunk[0]], df.columns])
        for t in chunk:
            if t not in df.columns.get_level_values(0):
                continue
            s = df[t].dropna(subset=["Close"])
            if s.empty:
                continue
            if splits is not None and "Stock Splits" in s.columns:
                sp = s["Stock Splits"].fillna(0)
                sp = sp[(sp > 0) & (sp != 1)]
                for d, v in sp.items():
                    splits.append({"date": pd.Timestamp(d).tz_localize(None) if pd.Timestamp(d).tz is not None else pd.Timestamp(d),
                                   "code": inv[t], "ratio": float(v)})
            out.append(pd.DataFrame({
                "date": s.index.tz_localize(None) if s.index.tz is not None else s.index,
                "code": inv[t],
                "adj_close": s["Adj Close"].astype("float64").values,
                "close": s["Close"].astype("float64").values,
                "volume": s["Volume"].astype("float64").values,
            }))
        time.sleep(2)
    if not out:
        return pd.DataFrame(columns=["date", "code", "adj_close", "close", "volume"])
    res = pd.concat(out, ignore_index=True)
    log.info("prices: %d rows for %d symbols", len(res), res["code"].nunique())
    return res


def update(store: Store, symbols: dict[str, str], years_back: int = 12, last_day: dt.date | None = None,
           track_splits: bool = False) -> None:
    """Incremental: refetch last 10 days (adjustments) for known codes, full history for new ones.
    Bars after `last_day` (a session that has not closed yet) are intraday snapshots and are dropped.
    track_splits: also keep the dataset `splits` (needed to recover the price a stock really traded at,
    because Yahoo's closes are split-adjusted)."""
    def done(df):
        return df if last_day is None or df.empty else df[pd.to_datetime(df["date"]) <= pd.Timestamp(last_day)]
    have = store.load("price")
    today = dt.date.today()
    known = set(have["code"].unique()) if not have.empty else set()
    new = {c: s for c, s in symbols.items() if c not in known}
    old = {c: s for c, s in symbols.items() if c in known}
    sp = [] if track_splits else None
    start_full = today.replace(year=today.year - years_back)
    if track_splits and not store.load_meta("splits_backfilled") and old:
        # one-time: split history of every known code (prices themselves are kept)
        hist = []
        download(old, start_full, splits=hist)
        _save_splits(store, hist)
        store.save_meta("splits_backfilled", {"date": str(today), "codes": len(old), "splits": len(hist)})
        log.info("splits: history for %d codes, %d splits", len(old), len(hist))
    if new:
        store.append("price", done(download(new, start_full, splits=sp)))
    if old:
        recent = done(download(old, today - dt.timedelta(days=10), splits=sp))
        store.append("price", recent)
        # Dividends/splits change the whole adjusted history: rescale stored prices
        # when the overlapping day moved (cheap fix, avoids full re-download).
        if not have.empty and not recent.empty:
            _rescale(store, have, recent)
    if sp:
        _save_splits(store, sp)


def _save_splits(store: Store, rows: list) -> None:
    if rows:
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        store.append("splits", df, keys=("date", "code"))


def _rescale(store: Store, have: pd.DataFrame, recent: pd.DataFrame) -> None:
    """On the first refetched day of each code, compare with the stored row. A dividend changes only
    adj_close; a split changes close, adj_close and volume (Yahoo adjusts all three). Older rows are
    rescaled by the same factors so the stored history stays continuous."""
    first = recent.sort_values("date").groupby("code").first().reset_index()
    m = first.merge(have[["date", "code", "adj_close", "close"]], on=["date", "code"], suffixes=("", "_old"))
    m["k_adj"] = m["adj_close"] / m["adj_close_old"]
    m["k_close"] = m["close"] / m["close_old"]
    bad = m[((m["k_adj"] - 1).abs() > 1e-4) | ((m["k_close"] - 1).abs() > 1e-4)]
    bad = bad[bad["k_adj"].between(1e-4, 1e4) & bad["k_close"].between(1e-4, 1e4)]
    if bad.empty:
        return
    log.info("rescaling adjusted history for %d codes (dividends/splits)", len(bad))
    k_adj, k_close = dict(zip(bad["code"], bad["k_adj"])), dict(zip(bad["code"], bad["k_close"]))
    full = store.load("price").copy()
    cut = dict(zip(bad["code"], bad["date"]))
    mask = full["code"].isin(k_adj) & (full["date"] < full["code"].map(cut))
    codes = full.loc[mask, "code"]
    full.loc[mask, "adj_close"] *= codes.map(k_adj)
    full.loc[mask, "close"] *= codes.map(k_close)
    full.loc[mask, "volume"] /= codes.map(k_close)
    for y, part in full.groupby(full["date"].dt.year):
        p = store._path("price", int(y))
        part.sort_values(["date", "code"]).to_parquet(p, index=False, compression="zstd")
        store.dirty.add(p)
        store.replaced.add(p)
    store._cache.pop("price", None)


def traded_close(close: pd.DataFrame, splits: pd.DataFrame | None) -> pd.DataFrame:
    """The price a stock actually traded at on each day: Yahoo's split-adjusted close times every split
    that happened later (a 1-for-10 reverse split has ratio 0.1, so earlier prices come back down).
    Without this, a stock that later did reverse splits looks expensive in the past - a look-ahead."""
    if splits is None or splits.empty:
        return close
    s = splits.copy()
    s["date"] = pd.to_datetime(s["date"])
    s = s[s["code"].isin(close.columns) & (s["ratio"] > 0)]
    if s.empty:
        return close
    lr = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    pos = close.index.searchsorted(s["date"].values)          # first trading day on/after the split date
    for p, c, r in zip(pos, s["code"], s["ratio"]):
        if p < len(close.index):
            lr.iat[p, lr.columns.get_loc(c)] += np.log(r)
    later = lr[::-1].cumsum()[::-1] - lr                       # splits strictly after each day
    return close * np.exp(later)
