"""Adjusted daily prices for any market via Yahoo Finance (yfinance), in batches.

Stored as dataset `price`: date, code, adj_close, close, volume.
"""
from __future__ import annotations

import datetime as dt
import time

import pandas as pd

from .common import log
from .store import Store

import logging as _logging
_logging.getLogger("yfinance").setLevel(_logging.CRITICAL)   # delisted-ticker noise is expected, not an error


def download(symbols: dict[str, str], start: dt.date, end: dt.date | None = None,
             batch: int = 150) -> pd.DataFrame:
    """symbols: {code: yahoo_symbol}. Returns long frame."""
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
                                 auto_adjust=False, actions=False, group_by="ticker",
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


def update(store: Store, symbols: dict[str, str], years_back: int = 12) -> None:
    """Incremental: refetch last 10 days (adjustments) for known codes, full history for new ones."""
    have = store.load("price")
    today = dt.date.today()
    known = set(have["code"].unique()) if not have.empty else set()
    new = {c: s for c, s in symbols.items() if c not in known}
    old = {c: s for c, s in symbols.items() if c in known}
    if new:
        store.append("price", download(new, today.replace(year=today.year - years_back)))
    if old:
        recent = download(old, today - dt.timedelta(days=10))
        store.append("price", recent)
        # Dividends/splits change the whole adjusted history: rescale stored adj_close
        # when the latest overlapping ratio moved (cheap fix, avoids full re-download).
        if not have.empty and not recent.empty:
            _rescale(store, have, recent)


def _rescale(store: Store, have: pd.DataFrame, recent: pd.DataFrame) -> None:
    first = recent.sort_values("date").groupby("code").first().reset_index()
    m = first.merge(have[["date", "code", "adj_close", "close"]], on=["date", "code"], suffixes=("", "_old"))
    m["f_new"] = m["adj_close"] / m["close"]
    m["f_old"] = m["adj_close_old"] / m["close_old"]
    m["k"] = m["f_new"] / m["f_old"]
    bad = m[(m["k"] - 1).abs() > 1e-4]
    if bad.empty:
        return
    log.info("rescaling adjusted history for %d codes (dividends/splits)", len(bad))
    k = dict(zip(bad["code"], bad["k"]))
    full = store.load("price").copy()
    cut = dict(zip(bad["code"], bad["date"]))
    mask = full["code"].isin(k) & (full["date"] < full["code"].map(cut))
    full.loc[mask, "adj_close"] *= full.loc[mask, "code"].map(k)
    for y, part in full.groupby(full["date"].dt.year):
        p = store._path("price", int(y))
        part.sort_values(["date", "code"]).to_parquet(p, index=False, compression="zstd")
        store.dirty.add(p)
        store.replaced.add(p)
    store._cache.pop("price", None)
