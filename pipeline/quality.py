"""Data-quality report shown on the dashboard.

Three kinds of checks:
  1. fetch-time cross-checks against official totals (dataset `_checks`, written while downloading)
  2. build-time cross-source checks (FactorSet.quality, e.g. official vs Yahoo closes)
  3. generic completeness checks on every dataset (missing days, thin days, staleness)
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from .store import Store


DATASET_NAMES = {
    "price": "股價", "quote": "收盤行情", "inst": "三大法人", "margin": "融資融券", "sbl": "借券賣出",
    "qfii": "外資持股", "daytrade": "當沖", "tdcc": "集保分級", "insider": "董監持股", "exright": "除權息",
    "tpex_ref": "櫃買參考價", "shortvol": "放空量", "si": "空單餘額", "activist": "13D/13G", "ftd": "交割失敗",
    "inst13f": "13F 機構持股", "short_pos": "空賣餘額報告", "mtall": "信用殘高", "invtype": "投資部門別",
    "southbound": "港股通持股", "shortsell": "沽空成交", "shortpos": "淡倉申報", "north": "陸股通持股",
    "flow": "主力資金", "lhb": "龍虎榜", "gdhs": "股東戶數", "ggcg": "增減持",
}


# Event-type data: the number of rows per day varies naturally, so "thin days" are not errors.
EVENT_DATASETS = {"lhb", "ggcg", "gdhs", "activist", "exright", "ftd", "inst13f", "insider13"}
# Data published with a known delay (calendar days); judged by staleness instead of missing days.
EXPECTED_LAG = {"ftd": 35, "inst13f": 150, "si": 20, "tdcc": 10, "shortpos": 16, "short_pos": 10,
                "gdhs": 100, "invtype": 14, "insider": 45, "mtall": 4, "margin_cn": 3}


def completeness(store: Store, datasets: list[str], calendar: pd.DatetimeIndex, lookback: int = 60) -> list[dict]:
    out = []
    recent = calendar[-lookback:]
    for name in datasets:
        df = store.load(name)
        if df.empty:
            out.append(dict(name=f"{DATASET_NAMES.get(name, name)}：資料完整度", ok=False, level="warn", detail="尚無資料"))
            continue
        d = pd.to_datetime(df["date"])
        last = d.max().date()
        lag = (calendar[-1].date() - last).days
        max_lag = EXPECTED_LAG.get(name)
        counts = d[d >= recent[0]].value_counts().sort_index()
        if counts.empty:
            ok = max_lag is not None and lag <= max_lag
            out.append(dict(name=f"{DATASET_NAMES.get(name, name)}：資料完整度", ok=ok, lag_days=lag,
                            detail=f"最新 {last}" + (f"（此資料約每 {max_lag} 天內更新一次，屬正常）" if ok
                                                    else f"；近 {lookback} 個交易日沒有資料")))
            continue
        med = counts.median()
        event = name in EVENT_DATASETS
        thin = counts[counts < med * 0.7] if not event else counts.iloc[:0]
        dense = len(counts) >= len(recent) * 0.5 and max_lag is None and not event   # daily dataset published without delay
        missing = [x for x in recent if x not in counts.index and x.date() <= last] if dense else []
        ok = len(thin) == 0 and len(missing) <= 2 and (max_lag is None or lag <= max_lag) and \
            (max_lag is not None or event or lag <= 4)
        parts = [f"最新 {last}", f"每日筆數中位數 {int(med)}" if not event else f"近 {lookback} 日 {int(counts.sum())} 筆事件"]
        if max_lag is not None:
            parts.append(f"官方公布本來就有延遲（容許 {max_lag} 天）")
        if len(thin):
            parts.append(f"筆數偏少的日期 {', '.join(str(x.date()) for x in thin.index[-3:])}")
        if missing:
            parts.append(f"缺 {len(missing)} 個交易日（如 {missing[-1].date()}）")
        out.append(dict(name=f"{DATASET_NAMES.get(name, name)}：資料完整度", ok=ok, detail="；".join(parts), lag_days=lag))
    return out


def fetch_checks(store: Store, days: int = 30) -> list[dict]:
    ch = store.load("_checks")
    if ch.empty:
        return []
    ch = ch[pd.to_datetime(ch["date"]) >= pd.Timestamp(dt.date.today() - dt.timedelta(days=days))]
    out = []
    for name, g in ch.groupby("code"):
        g = g.sort_values("date")
        fails = g[~g["ok"].astype(bool)]
        last = g.iloc[-1]
        detail = f"近 {days} 天檢查 {len(g)} 次，失敗 {len(fails)} 次。最近：{last['detail']}"
        if len(fails):
            detail += f"（最近失敗 {pd.Timestamp(fails.iloc[-1]['date']).date()}）"
        out.append(dict(name=CHECK_NAMES.get(name, name), ok=len(fails) == 0, detail=detail))
    return out


CHECK_NAMES = {
    "tw_margin_bal_total": "融資餘額：證交所市場合計 = 個股加總",
    "tw_short_bal_total": "融券餘額：證交所市場合計 = 個股加總",
    "tw_inst_components": "上市三大法人：合計 = 各法人加總",
    "tpex_inst_components": "上櫃三大法人：合計 = 各法人加總",
    "tdcc_pct_sum": "集保股權分散：各級距比例加總 = 100%",
}


def report(store: Store, fs, datasets: list[str], sources: dict) -> dict:
    checks = fetch_checks(store) + list(getattr(fs, "quality", []) or []) + completeness(store, datasets, fs.calendar)
    n_fail = sum(1 for c in checks if not c["ok"])
    return {"checks": checks, "n_fail": n_fail, "n": len(checks),
            "sources": [{"dataset": k, "source": v[0], "official": v[1]} for k, v in sources.items()]}
