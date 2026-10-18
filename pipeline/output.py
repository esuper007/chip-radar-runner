"""Write dashboard-ready files to dist/<market>/.

Files use the .js extension but contain plain JSON, so the dashboard can fetch() them and
up to 25 of them can be uploaded to the artifact asset store in one call.
  summary.js     market overview, top/bottom lists, model quality, data freshness
  screen.js      one row per stock: score, sub-scores, key factors, signals, forecasts
  series_<n>.js  per-stock daily history (last SERIES_DAYS days), sharded by code
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .factors import FactorSet, signal_flags

DIST = Path("dist")
SHARDS = 8
SERIES_DAYS = 130


def _r(x, nd=2):
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return x
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, nd)


def _s(x) -> str:
    """Text field: NaN/None -> '' (NaN is not valid JSON)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return str(x)


def shard_of(code: str) -> int:
    return sum(map(ord, code)) % SHARDS


def _plain(o):
    """numpy scalars -> plain Python for JSON."""
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=_plain)
    if "NaN" in txt or "Infinity" in txt:      # never ship invalid JSON to the dashboard
        txt = json.dumps(json.loads(txt, parse_constant=lambda _: None), ensure_ascii=False, separators=(",", ":"))
    path.write_text(txt)


def _model_labels(fs: FactorSet) -> dict:
    """Names for model-only features so coefficient lists read in plain words."""
    lab = {"score": "籌碼綜合分數", "mkt_ret_60": "大盤60日漲跌(歷史位置)"}
    groups = {"inst": "法人", "holders": "大戶散戶", "margin": "信用", "short": "放空借券", "insider": "內部人"}
    lab.update({f"s_{g}": f"{n}子分數" for g, n in groups.items()})
    for i, nm in enumerate(sorted(fs.market_series)):
        lab[f"mkt_s{i}"] = f"大盤：{nm}(歷史位置)"
    return lab


def write(fs: FactorSet, scores: dict, fc: pd.DataFrame, fc_info: dict, metrics: dict,
          freshness: dict, key_factors: list[str], quality: dict | None = None, sig: dict | None = None,
          surge: dict | None = None, breakout: dict | None = None,
          track: dict | None = None, health: dict | None = None, contrib: dict | None = None) -> dict:
    out_dir = DIST / fs.market
    t = fs.calendar[-1]
    sc = {k: v.loc[t] for k, v in scores.items()}
    F = {k: fs.factors[k].loc[t] for k in key_factors if k in fs.factors}
    flags = signal_flags(fs, t)
    px_now = fs.price.loc[t]
    liquid = fs.liquid.loc[t] if fs.liquid is not None else pd.Series(True, index=fs.codes)

    rows = []
    for c in fs.codes:
        s = sc.get("score", pd.Series()).get(c)
        if s is None or pd.isna(s):
            continue
        m = fs.meta.loc[c] if c in fs.meta.index else {}
        row = {"c": c, "n": _s(m.get("name") if hasattr(m, "get") else None) or c,
               "i": _s(m.get("industry") if hasattr(m, "get") else None),
               "x": _s(m.get("mkt") if hasattr(m, "get") else None),
               "liq": bool(liquid.get(c, False)), "px": _r(px_now.get(c), 3),
               "sc": _r(s, 1)}
        row["sub"] = {k[2:]: _r(v.get(c), 1) for k, v in sc.items() if k.startswith("s_")}
        row["f"] = {k: _r(v.get(c), 3) for k, v in F.items()}
        row["sig"] = flags.get(c, [])
        if contrib and c in contrib:
            row["ct"] = contrib[c]                # [[factor, points, rank 0-100], ...] largest first
        if sig and c in sig.get("recent", {}):
            row["bs"] = sig["recent"][c]          # bottom-up signals: {key: trading days ago}
        if surge and c in surge.get("today", {}):
            row["sv"] = surge["today"][c]["sv"]    # similarity to past winners (percentile 0-100)
            row["sm"] = surge["today"][c]["sm"]    # traits contributing most
        if c in fc.index:
            fr = fc.loc[c]
            row["fc"] = {}
            for col, v in fr.items():
                meth, hk, stat = col.split("_", 2)
                row["fc"].setdefault(meth, {}).setdefault(hk, {})[stat] = _r(v, 3)
        rows.append(row)
    _dump(out_dir / "screen.js", {"asof": str(t.date()), "rows": rows})

    # per-stock series
    win = fs.calendar[-SERIES_DAYS:]
    wwin = win[::-1][::5][::-1]                     # chip series: weekly points keep files small
    shards: dict[int, dict] = {i: {} for i in range(SHARDS)}
    ser_names = [n for n in fs.series if n in fs.factors]
    px_w, sc_w = fs.price.loc[win], scores["score"].loc[win]
    fac_w = {n: fs.factors[n].loc[wwin] for n in ser_names}
    for r in rows:
        if not r["liq"]:
            continue
        c = r["c"]
        d = {"px": [_r(v, 3) for v in px_w[c].values], "score": [_r(v, 1) for v in sc_w[c].values]}
        for n in ser_names:
            d[n] = [_r(v, 3) for v in fac_w[n][c].values]
        shards[shard_of(c)][c] = d
    for i, obj in shards.items():
        _dump(out_dir / f"series_{i}.js", {"dates": [str(x.date()) for x in win],
                                           "wdates": [str(x.date()) for x in wwin], "data": obj})

    # summary
    liq_rows = [r for r in rows if r["liq"]]
    top = sorted(liq_rows, key=lambda r: -r["sc"])[:30]
    bot = sorted(liq_rows, key=lambda r: r["sc"])[:30]
    slim = lambda r: {k: r[k] for k in ("c", "n", "i", "px", "sc", "sub", "sig")} | {
        "fc": {m: {h: v.get("p_up") for h, v in hz.items()} for m, hz in r.get("fc", {}).items()}}
    mwin = fs.calendar[-250:]
    mkt = {}
    for nm, ser in fs.market_series.items():
        ser = ser.sort_index()
        if ("買賣超" in nm or "淨" in nm or "净" in nm or "流入" in nm) and "未平倉" not in nm:   # daily flows: 20-day total
            ser = ser.rolling(20, min_periods=5).sum()
            nm = nm.replace(")", "，20日累計)") if ")" in nm else nm + "(20日累計)"
        ser = ser.reindex(mwin).dropna()
        mkt[nm] = {"d": [str(x.date()) for x in ser.index], "v": [_r(v, 3) for v in ser.values]}
    dist = np.histogram([r["sc"] for r in liq_rows], bins=10, range=(0, 100))[0].tolist() if liq_rows else []
    summary = {
        "market": fs.market, "asof": str(t.date()), "n_stocks": len(rows), "n_liquid": len(liq_rows),
        "score_hist": dist, "top": [slim(r) for r in top], "bottom": [slim(r) for r in bot],
        "market_series": mkt, "metrics": metrics, "importance": fc_info.get("importance", {}),
        "train_rows": fc_info.get("train_rows", {}), "train_weeks": fc_info.get("train_weeks", {}), "history_start": fc_info.get("history_start"),
        "freshness": freshness, "labels": {**fs.labels, **_model_labels(fs)}, "groups": list(fs.score_spec.keys()),
        "group_weights": fs.group_weights, "key_factors": key_factors, "series_names": ser_names,
        "shards": SHARDS, "quality": quality or {},
        "signals": (sig or {}).get("stats", {}),
        "surge": {k: v for k, v in (surge or {}).items() if k != "today"},
        "breakout": breakout or {},
        "track": track or {},
        "health": health or {},
    }
    summary["alerts"] = _alerts(fs, rows, sig, breakout, track, quality, health)
    _dump(out_dir / "summary.js", summary)
    return {"rows": len(rows), "asof": str(t.date())}


def _alerts(fs: FactorSet, rows: list, sig: dict | None, bo: dict | None, track: dict | None,
            quality: dict | None = None, health: dict | None = None) -> dict:
    """What a daily notification should mention (the notifier adds the viewer's watchlist):
    live lists below their backtest range, stocks whose validated signal fired on the last day, and fresh
    breakouts in the top fifth of a breakout model that passed validation. `data` lists the data checks that
    currently fail (the notifier reports only those that are new since its last message, and the ones that
    recovered); `tdcc` is the progress of the TDCC history fill, with `stalled` when the last run added nothing."""
    names = {r["c"]: r["n"] for r in rows}
    out = {"asof": str(fs.calendar[-1].date()), "track_warn": [], "signals": [], "breakouts": []}
    for k, v in ((track or {}).get("lists") or {}).items():
        if v.get("warn"):
            out["track_warn"].append({"list": v.get("name", k), "h": v["warn"]})
    stats = (sig or {}).get("stats") or {}
    for c, ev in ((sig or {}).get("recent") or {}).items():
        ks = [stats[k]["name"] for k, ago in ev.items() if ago == 0 and (stats.get(k) or {}).get("valid")]
        if ks and c in names:
            out["signals"].append({"c": c, "n": names[c], "s": ks})
    for mode, m in ((bo or {}).get("ml") or {}).items():
        if not m.get("valid"):
            continue
        for x in (bo or {}).get("today", []):
            p = ((x.get("ml") or {}).get(mode) or {}).get("p")
            if x.get("d") == 0 and p is not None and p >= 80 and x["c"] in names:
                out["breakouts"].append({"c": x["c"], "n": names[x["c"]], "mode": mode, "p": p,
                                         "sr": ((x.get("ml") or {}).get(mode) or {}).get("sr")})
    F = (health or {}).get("factors") or {}
    out["decay"] = [{"name": F.get(n, {}).get("label", n), "ic": (F.get(n, {}).get("recent") or {}).get("ic"),
                     "long": (F.get(n, {}).get("recent") or {}).get("long")} for n in ((health or {}).get("decay") or [])]
    out["gain_new"] = [{"c": c, "n": names[c]} for c in ((track or {}).get("gain_new") or []) if c in names]
    out["data"] = [{"name": c["name"], "detail": c.get("detail", "")} for c in (quality or {}).get("checks", [])
                   if not c.get("ok")]
    tp = (quality or {}).get("tdcc_progress")
    if tp:
        out["tdcc"] = {"done_pct": tp.get("done_pct"), "left": tp.get("left"), "date": tp.get("date"),
                       "stalled": bool(tp.get("left") and not tp.get("added"))}
    return out
