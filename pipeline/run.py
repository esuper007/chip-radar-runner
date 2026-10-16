"""CLI entry point.

  python -m pipeline.run daily    <market> [--date YYYY-MM-DD]
  python -m pipeline.run backfill <market> --start YYYY-MM-DD [--end ...] [--max-minutes 330]
  python -m pipeline.run prices   <market> [--years 12]
  python -m pipeline.run build    <market> [--backtest]
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import time
from pathlib import Path

import pandas as pd

from .common import drain_checks, log, parse_date, today_in, weekdays
from .store import Store

MARKETS = ["tw", "us", "jp", "hk", "cn"]


# Local closing time (+30 min buffer) per market; a bar dated today before this is an intraday snapshot.
CLOSE_LOCAL = {"tw": 14.0, "jp": 16.0, "hk": 16.75, "cn": 15.5, "us": 17.0}


def last_complete_day(market: str, tz: int) -> dt.date:
    """The latest date whose session has closed in the market's local time."""
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) + dt.timedelta(hours=tz)
    closed = now.hour + now.minute / 60 >= CLOSE_LOCAL.get(market, 17.0)
    return now.date() if closed else now.date() - dt.timedelta(days=1)


def mod(market: str):
    return importlib.import_module(f"pipeline.markets.{market}")


def cmd_daily(a):
    m, st = mod(a.market), Store(a.market)
    st.pull()
    d = parse_date(a.date) if a.date else today_in(m.TZ)
    try:
        res = m.daily(st, d)
    finally:
        st.append("_checks", drain_checks())
        st.push()                          # keep whatever was fetched even if a later step fails
    log.info("daily done: %s", json.dumps(res, default=str)[:2000])


def cmd_backfill(a):
    m, st = mod(a.market), Store(a.market)
    st.pull()
    start = parse_date(a.start)
    end = parse_date(a.end) if a.end else today_in(m.TZ)
    t0 = time.time()
    days = list(weekdays(start, end))
    days.reverse()                         # newest first: most useful data arrives first
    for i, d in enumerate(days):
        if (time.time() - t0) / 60 > a.max_minutes:
            log.info("time budget reached at %s; re-run to continue", d)
            break
        m.backfill_day(st, d)
        if i % 40 == 39:
            st.append("_checks", drain_checks())
            st.push()                      # checkpoint
    st.append("_checks", drain_checks())
    st.push()


def cmd_prices(a):
    from . import prices
    m, st = mod(a.market), Store(a.market)
    st.pull()
    prices.update(st, m.symbols(st), years_back=a.years, last_day=last_complete_day(a.market, m.TZ))
    st.push()


def cmd_build(a):
    from . import output, predict
    from .factors import compute_scores
    m, st = mod(a.market), Store(a.market)
    st.pull()
    cut = last_complete_day(a.market, m.TZ)
    px = st.load("price")
    if not px.empty and (pd.to_datetime(px["date"]) > pd.Timestamp(cut)).any():
        log.warning("ignoring intraday price rows after %s (session not closed yet)", cut)
        st._cache["price"] = px[pd.to_datetime(px["date"]) <= pd.Timestamp(cut)]
    fs = m.build_factors(st)
    log.info("factors: %d days x %d codes, %d factors", len(fs.calendar), len(fs.codes), len(fs.factors))
    scores = compute_scores(fs)

    out_dir = Path("dist") / a.market
    metrics_path = out_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    if a.backtest:
        X, Y = predict.training_frame(fs, scores)
        metrics = predict.backtest(X, Y, fs.calendar)
        metrics["computed_at"] = str(dt.date.today())
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=1))
    bundle = None if a.train else st.load_models()
    age = (dt.date.today() - dt.date.fromisoformat(bundle["info"]["trained_at"])).days if bundle else None
    if bundle is None or age > 8:
        log.info("training models (%s)", "requested" if a.train else "no saved models" if bundle is None else f"{age} days old")
        bundle = predict.train_models(fs, scores)
        st.save_models(bundle)
    else:
        log.info("using models trained %s", bundle["info"]["trained_at"])
    fc, info = predict.forecast(fs, scores, bundle)
    if info.get("is_oos"):
        for hk, m_ in info["is_oos"].items():
            metrics.setdefault(hk, {})["is"] = m_

    freshness = {}
    for name in getattr(m, "FRESHNESS_DATASETS", ["price"]):
        df = st.load(name)
        freshness[name] = str(pd.to_datetime(df["date"]).max().date()) if not df.empty else None
    from . import quality
    daily_sets = [d for d in getattr(m, "FRESHNESS_DATASETS", ["price"])]
    qrep = quality.report(st, fs, daily_sets, getattr(m, "SOURCES", {}))
    log.info("data quality: %d of %d checks need attention", qrep["n_fail"], qrep["n"])
    from . import signals
    try:
        ev = signals.compute(fs, scores)
        sig = {"stats": signals.evaluate(fs, ev), "recent": signals.recent(ev)}
    except Exception:
        log.exception("signals failed")
        sig = None
    from . import surge
    try:
        sg = surge.compute(fs, scores, st)
    except Exception:
        log.exception("surge similarity failed")
        sg = None
    from . import breakout
    try:
        bo = breakout.compute(fs, scores, st)
    except Exception:
        log.exception("breakout analysis failed")
        bo = None
    res = output.write(fs, scores, fc, info, metrics, freshness, m.KEY_FACTORS, qrep, sig, sg, bo)
    log.info("build done: %s", res)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("daily", "backfill", "prices", "build"):
        s = sub.add_parser(name)
        s.add_argument("market", choices=MARKETS)
        if name == "daily":
            s.add_argument("--date")
        if name == "backfill":
            s.add_argument("--start", required=True)
            s.add_argument("--end")
            s.add_argument("--max-minutes", type=float, default=330)
        if name == "prices":
            s.add_argument("--years", type=int, default=12)
        if name == "build":
            s.add_argument("--backtest", action="store_true")
            s.add_argument("--train", action="store_true", help="retrain models instead of reusing the saved ones")
    a = p.parse_args()
    try:
        globals()[f"cmd_{a.cmd}"](a)
    except Exception:
        log.exception("%s %s failed", a.cmd, a.market)
        raise


if __name__ == "__main__":
    main()
