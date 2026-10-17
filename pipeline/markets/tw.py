"""Taiwan (TWSE listed + TPEx OTC) chip data: full-market, by date.

Datasets written to the store (all keyed by date+code):
  inst      三大法人買賣超 (shares)       daily   TWSE T86 / TPEx dailyTrade
  margin    融資融券餘額 (lots)           daily   TWSE MI_MARGN / TPEx margin balance
  sbl       借券賣出餘額 (shares)         daily   TWSE TWT93U / TPEx sbl
  qfii      外資持股比率 & 發行股數       daily   TWSE MI_QFIIS / TPEx qfii
  daytrade  現股當沖成交股數              daily   TWSE TWTB4U / TPEx
  quote     收盤價/成交量 (unadjusted)     daily   TWSE MI_INDEX / TPEx dailyQuotes
  tdcc      集保股權分散 (derived ratios) weekly  TDCC opendata 1-5
  insider   董監持股/質押 (monthly)        monthly TWSE/TPEx openapi t187ap11
Adjusted prices for returns come from pipeline.prices (Yahoo) as `price`.
"""
from __future__ import annotations

import datetime as dt
import os
import re

import pandas as pd

from ..common import http_get, http_json, http_bytes, log, log_fields_once, num, read_csv_bytes, record_check, roc_to_date, ymd
from ..store import Store

MARKET = "tw"
TZ = 8
STOCK_RE = re.compile(r"^[1-9]\d{3}$")  # common stocks only (excludes ETFs/warrants)

TWSE = "https://www.twse.com.tw/rwd/zh"
TPEX = "https://www.tpex.org.tw/www/zh-tw"


# ---------------------------------------------------------------- helpers
def _ok_twse(j) -> bool:
    return bool(j) and str(j.get("stat", "")).upper() == "OK"


def _tpex_table(j) -> tuple[list[str], list[list]]:
    """TPEx new-site JSON: {'tables':[{'fields':[...],'data':[...]}], 'stat':'ok'}"""
    if not j:
        return [], []
    tables = j.get("tables") or []
    best = max(tables, key=lambda t: len(t.get("data") or []), default=None)
    if not best:
        return [], []
    log_fields_once(str(j.get("reportTitle") or best.get("title") or "tpex"), best.get("fields") or [])
    if best.get("data"):
        log_fields_once("row:" + str(best.get("title") or "tpex"), best["data"][0])
    return best.get("fields") or [], best.get("data") or []


def _col(fields: list[str], *must: str, exclude: tuple[str, ...] = (), nth: int = 0) -> int | None:
    hits = [i for i, f in enumerate(fields)
            if all(m in f for m in must) and not any(e in f for e in exclude)]
    return hits[nth] if len(hits) > nth else None


def _frame(rows: list[dict], d: dt.date) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["code"] = df["code"].astype(str).str.strip()
    df = df[df["code"].str.match(STOCK_RE)]
    df.insert(0, "date", pd.Timestamp(d))
    return df


def _tpex_date(d: dt.date) -> str:
    return d.strftime("%Y/%m/%d")


# ---------------------------------------------------------------- 三大法人
def fetch_inst(d: dt.date) -> pd.DataFrame:
    rows = []
    j = http_json(f"{TWSE}/fund/T86", {"date": ymd(d), "selectType": "ALLBUT0999", "response": "json"})
    if _ok_twse(j):
        for r in j["data"]:
            rows.append(dict(code=r[0], foreign=num(r[4]), foreign_dealer=num(r[7]), trust=num(r[10]),
                             dealer_self=num(r[14]), dealer_hedge=num(r[17]), inst_total=num(r[18]), mkt="TWSE"))
        bad = sum(1 for x in rows if None not in (x["foreign"], x["foreign_dealer"], x["trust"], x["dealer_self"],
                                                   x["dealer_hedge"], x["inst_total"])
                  and abs(x["foreign"] + x["foreign_dealer"] + x["trust"] + x["dealer_self"] + x["dealer_hedge"]
                          - x["inst_total"]) > 1)
        record_check("tw_inst_components", d, bad == 0, f"三大法人合計 = 各法人加總（不符 {bad} 檔）", 0, bad)
    try:
        j = http_json(f"{TPEX}/insti/dailyTrade",
                      {"type": "Daily", "sect": "EW", "date": _tpex_date(d), "response": "json"})
        f, data = _tpex_table(j)
        if data:
            ix = dict(
                foreign=_col(f, "外資", "買賣超", exclude=("外資自營商-",)) ,
                foreign_dealer=_col(f, "外資自營商", "買賣超"),
                trust=_col(f, "投信", "買賣超"),
                dealer_self=_col(f, "自行買賣", "買賣超"),
                dealer_hedge=_col(f, "避險", "買賣超"),
                inst_total=_col(f, "三大法人", "買賣超"),
            )
            if len(f) >= 24 and (ix["foreign"] is None or ix["trust"] is None or len(set(f)) < len(f)):
                # generic repeated headers (買進股數/賣出股數/買賣超股數 per group): use positions
                ix = dict(foreign=4, foreign_dealer=7, trust=13, dealer_self=16, dealer_hedge=19, inst_total=len(f) - 1)
            elif ix["foreign"] is None or ix["trust"] is None:
                log.warning("TPEx inst fields not recognised: %s", f)
            n0 = len(rows)
            for r in data:
                rows.append(dict(code=r[0], mkt="TPEx",
                                 **{k: (num(r[i]) if i is not None else None) for k, i in ix.items()}))
            tp = rows[n0:]
            bad = sum(1 for x in tp if None not in (x["foreign"], x["foreign_dealer"], x["trust"], x["dealer_self"],
                                                     x["dealer_hedge"], x["inst_total"])
                      and abs(x["foreign"] + x["foreign_dealer"] + x["trust"] + x["dealer_self"] + x["dealer_hedge"]
                              - x["inst_total"]) > 1)
            miss = sum(1 for x in tp if x["foreign"] is None)
            record_check("tpex_inst_components", d, bad == 0 and miss == 0,
                         f"櫃買三大法人合計 = 各法人加總（不符 {bad} 檔，欄位缺漏 {miss} 檔）", 0, bad + miss)
    except Exception as e:  # TPEx outage must not kill TWSE data
        log.warning("TPEx inst %s failed: %s", d, e)
    return _frame(rows, d)


# ---------------------------------------------------------------- 融資融券
def fetch_margin(d: dt.date) -> pd.DataFrame:
    rows = []
    j = http_json(f"{TWSE}/marginTrading/MI_MARGN", {"date": ymd(d), "selectType": "ALL", "response": "json"})
    if _ok_twse(j):
        tbl = next((t for t in j.get("tables", []) if t.get("fields") and t["fields"][0] == "代號"), None)
        for r in (tbl or {}).get("data", []):
            rows.append(dict(code=r[0], margin_buy=num(r[2]), margin_sell=num(r[3]), margin_bal=num(r[6]),
                             margin_limit=num(r[7]), short_sell=num(r[9]), short_bal=num(r[12]),
                             offset=num(r[14]), mkt="TWSE"))
        # official market total (交易單位) must equal the sum of every listed row (stocks + ETFs)
        tot = next((t for t in j.get("tables", []) if t.get("fields") and t["fields"][0] == "項目"), None)
        if tot and rows:
            for item, col in (("融資(交易單位)", "margin_bal"), ("融券(交易單位)", "short_bal")):
                row = next((r for r in tot["data"] if str(r[0]).startswith(item)), None)
                if row:
                    exp, act = num(row[5]), sum(x[col] or 0 for x in rows)
                    record_check(f"tw_{col}_total", d, exp is not None and abs(act - exp) <= max(1, exp * 1e-6),
                                 f"證交所{item}今日餘額 vs 個股加總", exp, act)
    try:
        j = http_json(f"{TPEX}/margin/balance", {"date": _tpex_date(d), "response": "json"})
        f, data = _tpex_table(j)
        if data:
            bal = [i for i, x in enumerate(f) if "餘額" in x and "前" not in x]
            lim = [i for i, x in enumerate(f) if "限額" in x]
            if len(bal) >= 2:
                for r in data:
                    rows.append(dict(code=r[0], margin_bal=num(r[bal[0]]), short_bal=num(r[bal[1]]),
                                     margin_limit=num(r[lim[0]]) if lim else None, mkt="TPEx"))
            else:
                log.warning("TPEx margin fields not recognised: %s", f)
    except Exception as e:
        log.warning("TPEx margin %s failed: %s", d, e)
    return _frame(rows, d)


# ---------------------------------------------------------------- 借券賣出
def fetch_sbl(d: dt.date) -> pd.DataFrame:
    rows = []
    j = http_json(f"{TWSE}/marginTrading/TWT93U", {"date": ymd(d), "response": "json"})
    if _ok_twse(j):
        for r in j["data"]:
            rows.append(dict(code=r[0], sbl_sell=num(r[9]), sbl_return=num(r[10]), sbl_bal=num(r[12]),
                             mkt="TWSE"))
    try:
        j = http_json(f"{TPEX}/margin/sbl", {"date": _tpex_date(d), "response": "json"})
        f, data = _tpex_table(j)
        if data and len(f) >= 13:
            for r in data:
                rows.append(dict(code=r[0], sbl_sell=num(r[9]), sbl_return=num(r[10]), sbl_bal=num(r[12]),
                                 mkt="TPEx"))
    except Exception as e:
        log.warning("TPEx sbl %s failed: %s", d, e)
    return _frame(rows, d)


# ---------------------------------------------------------------- 外資持股
def fetch_qfii(d: dt.date) -> pd.DataFrame:
    rows = []
    j = http_json(f"{TWSE}/fund/MI_QFIIS", {"date": ymd(d), "selectType": "ALLBUT0999", "response": "json"})
    if _ok_twse(j):
        for r in j["data"]:
            rows.append(dict(code=r[0], shares=num(r[3]), foreign_shares=num(r[5]),
                             foreign_ratio=num(r[7]), foreign_cap=num(r[8]), mkt="TWSE"))
    try:
        j = http_json(f"{TPEX}/insti/qfii", {"type": "Daily", "date": _tpex_date(d), "response": "json"})
        f, data = _tpex_table(j)
        if data:
            ix = dict(shares=_col(f, "發行股數"), foreign_shares=_col(f, "持有股數"),
                      foreign_ratio=_col(f, "持股比率", exclude=("尚可", "上限")),
                      foreign_cap=_col(f, "上限"))
            for r in data:
                rows.append(dict(code=r[1] if not STOCK_RE.match(str(r[0]).strip()) else r[0], mkt="TPEx",
                                 **{k: (num(r[i]) if i is not None else None) for k, i in ix.items()}))
    except Exception as e:
        log.warning("TPEx qfii %s failed: %s", d, e)
    return _frame(rows, d)


# ---------------------------------------------------------------- 當沖
def fetch_daytrade(d: dt.date) -> pd.DataFrame:
    rows = []
    j = http_json(f"{TWSE}/dayTrading/TWTB4U", {"date": ymd(d), "selectType": "All", "response": "json"})
    if _ok_twse(j):
        for t in j.get("tables", []):
            f = t.get("fields") or []
            if f and f[0] == "證券代號":
                log_fields_once("TWTB4U", f)
                iv = _col(f, "成交股數")
                if iv is None:
                    iv = _col(f, "股數")
                if iv is None:
                    continue
                for r in t.get("data", []):
                    rows.append(dict(code=r[0], daytrade_shares=num(r[iv]), mkt="TWSE"))
    return _frame(rows, d)


# ---------------------------------------------------------------- 收盤行情 (unadjusted; also gives turnover)
def fetch_quote(d: dt.date) -> pd.DataFrame:
    rows = []
    j = http_json(f"{TWSE}/afterTrading/MI_INDEX", {"date": ymd(d), "type": "ALLBUT0999", "response": "json"})
    if _ok_twse(j):
        for t in j.get("tables", []):
            f = t.get("fields") or []
            if f and f[0] == "證券代號":
                for r in t.get("data", []):
                    rows.append(dict(code=r[0], name=str(r[1]).strip(), volume=num(r[2]), value=num(r[4]),
                                     close=num(r[8]), mkt="TWSE"))
    try:
        j = http_json(f"{TPEX}/afterTrading/dailyQuotes", {"date": _tpex_date(d), "type": "EW", "response": "json"})
        f, data = _tpex_table(j)
        if data:
            ic, iv, ival = _col(f, "收盤"), _col(f, "成交股數"), _col(f, "成交金額")
            for r in data:
                rows.append(dict(code=r[0], name=str(r[1]).strip(), close=num(r[ic]) if ic is not None else None,
                                 volume=num(r[iv]) if iv is not None else None,
                                 value=num(r[ival]) if ival is not None else None, mkt="TPEx"))
    except Exception as e:
        log.warning("TPEx quote %s failed: %s", d, e)
    return _frame(rows, d)


# ---------------------------------------------------------------- 櫃買 次日參考價 (official ex-rights/capital-change reference)
def fetch_tpex_ref(d: dt.date) -> pd.DataFrame:
    """TPEx daily quotes carry the NEXT day's official reference price; on an ex-date the
    previous day's 次日參考價 is the ex-rights reference, so adjustment = close / next_ref."""
    j = http_json(f"{TPEX}/afterTrading/dailyQuotes", {"date": _tpex_date(d), "type": "EW", "response": "json"})
    f, data = _tpex_table(j)
    if not data:
        return pd.DataFrame()
    ic, ir, ish = _col(f, "收盤"), _col(f, "參考價"), _col(f, "發行股數")
    rows = [dict(code=r[0], close=num(r[ic]), next_ref=num(r[ir]) if ir is not None else None,
                 shares=num(r[ish]) if ish is not None else None) for r in data]
    return _frame(rows, d)


# ---------------------------------------------------------------- 集保股權分散 (weekly, latest week only)
# Levels: 1:1-999 2:1-5張 3:5-10 4:10-15 5:15-20 6:20-30 7:30-40 8:40-50 9:50-100 10:100-200
#         11:200-400 12:400-600 13:600-800 14:800-1000 15:>1000 16:差異調整 17:合計
def fetch_tdcc() -> pd.DataFrame:
    raw = read_csv_bytes(http_bytes("https://opendata.tdcc.com.tw/getOD.ashx", {"id": "1-5"}), dtype=str)
    raw.columns = ["date", "code", "level", "holders", "shares", "pct"]
    raw["code"] = raw["code"].str.strip()
    raw = raw[raw["code"].str.match(STOCK_RE)]
    for c in ("level", "holders", "shares", "pct"):
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    return tdcc_derive(raw)


def tdcc_derive(raw: pd.DataFrame) -> pd.DataFrame:
    p = raw.pivot_table(index=["date", "code"], columns="level", values=["holders", "pct", "shares"], aggfunc="sum")
    tot = p["pct"][[l for l in range(1, 17) if l in p["pct"].columns]].sum(axis=1)
    off = int(((tot - 100).abs() > 0.5).sum())
    for dd in p.index.get_level_values("date").unique():
        record_check("tdcc_pct_sum", dd, off <= len(tot) * 0.01, f"集保各級距持股比例加總 ≈ 100%（偏離 {off} 檔）", 0, off)
    out = pd.DataFrame(index=p.index)

    def s(kind, levels):
        return p[kind][[l for l in levels if l in p[kind].columns]].sum(axis=1)
    out["holders"] = s("holders", range(1, 16))
    out["retail_pct"] = s("pct", range(1, 9))            # ≤50張
    out["mid_pct"] = s("pct", range(9, 12))              # 50-400張
    out["big400_pct"] = s("pct", range(12, 16))          # >400張
    out["big1000_pct"] = s("pct", [15])                  # >1000張
    out["big1000_holders"] = s("holders", [15])
    out["retail_holders"] = s("holders", range(1, 9))
    total_sh = s("shares", range(1, 16))
    out["avg_lots_per_holder"] = total_sh / out["holders"].replace(0, pd.NA) / 1000
    out = out.reset_index()
    out["date"] = pd.to_datetime(out["date"].astype(str), format="%Y%m%d")
    return out


# ---------------------------------------------------------------- 董監持股 / 質押 (monthly)
def fetch_insider() -> pd.DataFrame:
    rows = []
    for url, mkt in (("https://openapi.twse.com.tw/v1/opendata/t187ap11_L", "TWSE"),
                     ("https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap11_O", "TPEx")):
        try:
            data = http_json(url) or []
        except Exception as e:
            log.warning("insider %s failed: %s", mkt, e)
            continue
        if not data:
            continue
        keys = list(data[0].keys())
        log_fields_once(f"insider-{mkt}", data[0].items())
        k_code = next((k for k in keys if "代號" in k or "Code" in k), None)
        k_date = next((k for k in keys if "資料年月" in k), None) or next((k for k in keys if "年月" in k or "日期" in k), None)
        k_hold = (next((k for k in keys if "目前持股" in k and "關係人" not in k), None)
                  or next((k for k in keys if "持股" in k and "質" not in k and "比" not in k and "選任" not in k), None))
        k_pledge = next((k for k in keys if "質" in k and "股" in k and "比" not in k and "關係人" not in k), None)
        k_title = next((k for k in keys if "職稱" in k), None)
        if not (k_code and k_hold):
            log.warning("insider %s keys not recognised: %s", mkt, keys)
            continue
        for r in data:
            rows.append(dict(code=str(r[k_code]).strip(), asof=r.get(k_date), title=r.get(k_title),
                             hold=num(r.get(k_hold)), pledge=num(r.get(k_pledge)) if k_pledge else None, mkt=mkt))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df[df["code"].str.match(STOCK_RE)]
    agg = df.groupby(["code", "mkt"]).agg(insider_shares=("hold", "sum"), insider_pledge=("pledge", "sum"),
                                          asof=("asof", "first")).reset_index()

    def to_date(x):
        s = re.sub(r"\D", "", str(x or ""))
        try:
            if len(s) == 5:   # ROC yyy mm
                return dt.date(int(s[:3]) + 1911, int(s[3:5]), 1)
            if len(s) == 7:   # ROC yyy mm dd
                return roc_to_date(f"{s[:3]}/{s[3:5]}/{s[5:]}")
            if len(s) == 8:
                return dt.date(int(s[:4]), int(s[4:6]), int(s[6:]))
        except ValueError:
            pass
        return dt.date.today().replace(day=1)
    agg["date"] = pd.to_datetime(agg["asof"].map(to_date))
    return agg.drop(columns=["asof"])


# ---------------------------------------------------------------- 除權除息 (official adjustment events)
def _roc_or_iso(x) -> dt.date | None:
    s_ = re.sub(r"[^0-9/]", "/", str(x)).strip("/")
    parts = [p for p in s_.split("/") if p]
    try:
        if len(parts) == 3:
            y = int(parts[0])
            return dt.date(y + 1911 if y < 1911 else y, int(parts[1]), int(parts[2]))
        digits = re.sub(r"\D", "", str(x))
        if len(digits) == 7:
            return dt.date(int(digits[:3]) + 1911, int(digits[3:5]), int(digits[5:]))
        if len(digits) == 8:
            return dt.date(int(digits[:4]), int(digits[4:6]), int(digits[6:]))
    except ValueError:
        return None
    return None


def fetch_exright(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Ex-rights / ex-dividend reference prices: TWSE TWT49U (history) + TPEx openapi (latest)."""
    rows = []
    j = http_json(f"{TWSE}/exRight/TWT49U", {"startDate": ymd(start), "endDate": ymd(end), "response": "json"})
    if _ok_twse(j):
        f = j.get("fields") or []
        log_fields_once("TWT49U", f)
        i_d, i_c = _col(f, "日期"), _col(f, "代號")
        i_pre, i_ref = _col(f, "前收盤"), _col(f, "參考價", exclude=("減除",))
        for r in j.get("data", []):
            d_ = _roc_or_iso(r[i_d])
            if d_:
                rows.append(dict(date=pd.Timestamp(d_), code=str(r[i_c]).strip(), pre_close=num(r[i_pre]),
                                 ref_price=num(r[i_ref]), mkt="TWSE"))
    try:
        data = http_json("https://www.tpex.org.tw/openapi/v1/tpex_exright_daily") or []
        if data:
            log_fields_once("tpex_exright_daily", data[0].items())
        for r in data:
            ks = list(r.keys())
            k_d = next((k for k in ks if "日期" in k or "Date" in k), None)
            k_c = next((k for k in ks if "代號" in k or "Code" in k), None)
            k_pre = next((k for k in ks if "前收盤" in k or k.startswith("ClosePriceBefore")), None)
            k_ref = next((k for k in ks if k in ("ExRightsDiviendQuote", "除權息參考價")), None) or \
                next((k for k in ks if "參考價" in k and "開盤" not in k), None)
            d_ = _roc_or_iso(r.get(k_d)) if k_d else None
            if d_ and k_c and k_pre and k_ref:
                rows.append(dict(date=pd.Timestamp(d_), code=str(r[k_c]).strip(), pre_close=num(r[k_pre]),
                                 ref_price=num(r[k_ref]), mkt="TPEx"))
    except Exception as e:
        log.warning("TPEx exright: %s", e)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[df["code"].str.match(STOCK_RE) & (df["pre_close"] > 0) & (df["ref_price"] > 0)]
    return df


# ---------------------------------------------------------------- universe
def fetch_universe() -> pd.DataFrame:
    rows = []
    for url, mkt in (("https://openapi.twse.com.tw/v1/opendata/t187ap03_L", "TWSE"),
                     ("https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O", "TPEx")):
        try:
            data = http_json(url) or []
        except Exception as e:
            log.warning("universe %s failed: %s", mkt, e)
            continue
        if data:
            log_fields_once(f"universe-{mkt}", data[0].items())
        for r in data:
            ks = list(r.keys())
            code = r.get(next((k for k in ks if "代號" in k or k == "SecuritiesCompanyCode"), ""), "")
            name = (r.get(next((k for k in ks if "簡稱" in k or k == "CompanyAbbreviation"), ""), "")
                    or r.get(next((k for k in ks if "名稱" in k), ""), ""))
            ind = r.get(next((k for k in ks if "產業" in k or k == "SecuritiesIndustryCode"), ""), "")
            ind = str(ind).strip()
            ind = INDUSTRY.get(ind.zfill(2), ind) if ind.isdigit() else ind
            rows.append(dict(code=str(code).strip(), name=str(name).strip(), industry=ind, mkt=mkt))
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[df["code"].str.match(STOCK_RE)].drop_duplicates("code")
    return df


INDUSTRY = {"01": "水泥", "02": "食品", "03": "塑膠", "04": "紡織纖維", "05": "電機機械", "06": "電器電纜",
            "08": "玻璃陶瓷", "09": "造紙", "10": "鋼鐵", "11": "橡膠", "12": "汽車", "14": "建材營造",
            "15": "航運", "16": "觀光餐旅", "17": "金融保險", "18": "貿易百貨", "19": "綜合", "20": "其他",
            "21": "化學", "22": "生技醫療", "23": "油電燃氣", "24": "半導體", "25": "電腦週邊", "26": "光電",
            "27": "通信網路", "28": "電子零組件", "29": "電子通路", "30": "資訊服務", "31": "其他電子",
            "32": "文化創意", "33": "農業科技", "34": "電子商務", "35": "綠能環保", "36": "數位雲端",
            "37": "運動休閒", "38": "居家生活", "80": "管理股票", "91": "存託憑證"}


def yahoo_symbol(code: str, mkt: str) -> str:
    return f"{code}.TW" if mkt == "TWSE" else f"{code}.TWO"


# ---------------------------------------------------------------- orchestration
DAILY = {
    "inst": fetch_inst,
    "margin": fetch_margin,
    "sbl": fetch_sbl,
    "qfii": fetch_qfii,
    "daytrade": fetch_daytrade,
    "quote": fetch_quote,
    "tpex_ref": fetch_tpex_ref,
}


def update_day(store: Store, d: dt.date, datasets=None, skip_existing=True) -> dict:
    """Fetch every daily dataset for one date. Returns {dataset: rows}."""
    fetched = store.load("_fetchlog")
    seen = set()
    if not fetched.empty:
        fl = fetched.assign(date=pd.to_datetime(fetched["date"]).dt.date)
        # recent days logged with too few rows (partial publication) are fetched again
        med = fl[fl["rows"] > 0].groupby("code")["rows"].median()
        recent = fl["date"] >= dt.date.today() - dt.timedelta(days=7)
        thin = recent & (fl["rows"] > 0) & (fl["rows"] < 0.8 * fl["code"].map(med))
        seen = set(zip(fl.loc[~thin, "date"], fl.loc[~thin, "code"]))
    res, logrows = {}, []
    for name, fn in DAILY.items():
        if datasets and name not in datasets:
            continue
        force = name in os.environ.get("FORCE_DATASETS", "").split(",")
        if skip_existing and not force and (d, name) in seen:
            res[name] = "cached"
            continue
        try:
            df = fn(d)
        except Exception as e:
            log.error("%s %s failed: %s", name, d, e)
            res[name] = f"error: {e}"
            continue
        store.append(name, df)
        res[name] = len(df)
        # A partial day (e.g. TPEx published, TWSE not yet) must be re-fetched on the next run,
        # so recent days with far fewer rows than usual are not marked as done.
        partial = False
        if len(df) > 0 and not fetched.empty and d >= dt.date.today() - dt.timedelta(days=7):
            prev = fetched[(fetched["code"] == name) & (fetched["rows"] > 0)].sort_values("date")["rows"].tail(10)
            if len(prev) >= 3 and len(df) < 0.8 * prev.median():
                partial = True
                res[name] = f"{len(df)} (partial, will retry)"
                log.warning("%s %s: only %d rows vs usual %d - will retry", name, d, len(df), int(prev.median()))
        # 0 rows today may just mean "not published yet": only remember empty days once they are old
        if not partial and (len(df) > 0 or d < dt.date.today() - dt.timedelta(days=3)):
            logrows.append(dict(date=pd.Timestamp(d), code=name, rows=len(df)))
    # remember fetched dates (holidays give 0 rows) so backfills never re-request them
    store.append("_fetchlog", pd.DataFrame(logrows))
    log.info("TW %s -> %s", d, res)
    return res


TAIFEX = "https://www.taifex.com.tw/cht/3/"


def _taifex_rows(path: str, start: dt.date, end: dt.date, product: str) -> list[list[str]]:
    r = http_get(TAIFEX + path, params={"queryStartDate": start.strftime("%Y/%m/%d"),
                                        "queryEndDate": end.strftime("%Y/%m/%d"), "commodityId": product}, retries=3)
    txt = r.content.decode("cp950", errors="replace")
    rows = [ln.split(",") for ln in txt.strip().splitlines()[1:] if ln.count(",") >= 10]
    return rows


def fetch_taifex(start: dt.date, end: dt.date) -> pd.DataFrame:
    """期交所三大法人: foreign net open interest in TAIEX futures (TXF, contracts) and TAIEX options
    (TXO calls / puts, contracts and NT$ thousand). One row per date, code 'TAIEX'."""
    out: dict[str, dict] = {}
    for ln in _taifex_rows("futContractsDateDown", start, end, "TXF"):
        if len(ln) >= 15 and "外資" in ln[2]:
            out.setdefault(ln[0], {})["fut_foreign_oi"] = float(ln[13])
    for ln in _taifex_rows("callsAndPutsDateDown", start, end, "TXO"):
        if len(ln) >= 16 and "外資" in ln[3]:
            side = "call" if ("CALL" in ln[2].upper() or "買權" in ln[2]) else "put"
            out.setdefault(ln[0], {})[f"opt_foreign_{side}_oi"] = float(ln[14])
            out[ln[0]][f"opt_foreign_{side}_amt"] = float(ln[15])
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame([{"date": pd.Timestamp(k.replace("/", "-")), "code": "TAIEX", **v} for k, v in out.items()])
    return df.sort_values("date")


def update_taifex(store: Store, max_months: int = 6) -> None:
    """Fill any month in the last ~3 years that has no data yet (the site rate-limits, so at most
    `max_months` per run, oldest gaps first, with a pause between requests); the current month is
    always refreshed."""
    import time
    have: dict[tuple[int, int], int] = {}
    for d in store.dates("taifex"):
        have[(d.year, d.month)] = have.get((d.year, d.month), 0) + 1
    today = dt.date.today()
    first = (today - dt.timedelta(days=3 * 365)).replace(day=1)
    months, m = [], first
    while m <= today:
        months.append(m)
        m = (m + dt.timedelta(days=32)).replace(day=1)
    cur = (today.year, today.month)
    todo = [m for m in months if (m.year, m.month) != cur and have.get((m.year, m.month), 0) < 12][:max_months]
    todo.append(today.replace(day=1))
    n = 0
    for k, m in enumerate(todo):
        end = min(today, (m + dt.timedelta(days=32)).replace(day=1) - dt.timedelta(days=1))
        if k:
            time.sleep(8)
        try:
            df = fetch_taifex(m, end)
            if not df.empty:
                store.append("taifex", df)
                n += len(df)
        except Exception as e:
            log.warning("taifex %s: %s", m.strftime("%Y-%m"), e)
            if "429" in str(e):
                break
    missing = sum(1 for m in months if (m.year, m.month) != cur and have.get((m.year, m.month), 0) < 12) - len(todo) + 1
    log.info("taifex: %d new days; %d older months still to fill", n, max(0, missing))


def update_weekly_monthly(store: Store) -> None:
    try:
        update_taifex(store)
    except Exception as e:
        log.error("taifex failed: %s", e)
    try:
        today = dt.date.today()
        store.append("exright", fetch_exright(today - dt.timedelta(days=20), today))
    except Exception as e:
        log.error("exright failed: %s", e)
    try:
        store.append("tdcc", fetch_tdcc())
    except Exception as e:
        log.error("tdcc failed: %s", e)
    try:
        store.append("insider", fetch_insider(), keys=("date", "code"))
    except Exception as e:
        log.error("insider failed: %s", e)
    uni = fetch_universe()
    if not uni.empty:
        uni["date"] = pd.Timestamp(dt.date.today())
        store.append("universe", uni)


# ================================================================ factors
def build_factors(store: Store, since: str = "2012-01-01"):
    from ..factors import FactorSet, chg, pct_chg, price_context, rsum, streak, wide

    s = pd.Timestamp(since)
    price = store.load("price", s)
    quote = store.load("quote", s)
    inst = store.load("inst", s)
    margin = store.load("margin", s)
    sbl = store.load("sbl", s)
    qfii = store.load("qfii", s)
    dtr = store.load("daytrade", s)
    tdcc = store.load("tdcc", s)
    ins = store.load("insider", s)
    uni = store.load("universe")

    days = set()
    for d in (price, quote, inst):
        if not d.empty:
            days |= set(pd.to_datetime(d["date"].unique()))
    cal = pd.DatetimeIndex(sorted(days))
    codes = set()
    for d in (price, inst, quote):
        if not d.empty:
            codes |= set(d["code"].unique())
    if not uni.empty:
        codes &= set(uni["code"]) | set(inst["code"].unique() if not inst.empty else [])
    codes = pd.Index(sorted(codes), name="code")

    W = lambda df, col, lim=0: wide(df, col, cal, codes, lim)
    exr = store.load("exright", s)
    tref = store.load("tpex_ref", s)
    if not tref.empty:
        # turn TPEx 次日參考價 into ex-rights events: on day t, factor = close(t-1) / next_ref(t-1)
        tr = tref.dropna(subset=["close", "next_ref"]).sort_values(["code", "date"])
        tr = tr[(tr["next_ref"] > 0) & ((tr["close"] / tr["next_ref"] - 1).abs() > 0.001)]
        pos = cal.searchsorted(pd.to_datetime(tr["date"]), side="right")
        tr = tr[pos < len(cal)]
        pos = pos[pos < len(cal)]
        ev = pd.DataFrame({"date": cal[pos], "code": tr["code"].values, "pre_close": tr["close"].values,
                           "ref_price": tr["next_ref"].values, "mkt": "TPEx"})
        exr = pd.concat([exr, ev], ignore_index=True) if not exr.empty else ev
    px, qual = official_adjusted(W(quote, "close"), W(price, "close"), W(price, "adj_close"), exr, cal, codes)
    vol = W(quote, "volume")
    vol = vol.where(vol.notna(), W(price, "volume"))
    shares = W(qfii, "shares", 260)
    shares = shares.ffill().bfill()

    f: dict[str, pd.DataFrame] = {}
    sh = shares.where(shares > 0)
    bps = lambda x: x / sh * 1e4  # basis points of shares outstanding

    # --- 法人
    fo = W(inst, "foreign").fillna(0) + W(inst, "foreign_dealer").fillna(0)
    tr = W(inst, "trust")
    ds = W(inst, "dealer_self")
    dh = W(inst, "dealer_hedge")
    has_inst = W(inst, "inst_total").notna()
    fo = fo.where(has_inst)
    for n in (5, 20, 60):
        f[f"foreign_net_{n}"] = bps(rsum(fo, n))
        f[f"trust_net_{n}"] = bps(rsum(tr, n))
    f["dealer_net_20"] = bps(rsum(ds, 20))
    f["dealer_hedge_20"] = bps(rsum(dh, 20))
    f["inst_vol_share_20"] = rsum(fo.fillna(0) + tr.fillna(0) + ds.fillna(0), 20) / rsum(vol, 20).where(lambda x: x > 0) * 100
    f["foreign_streak"] = streak(fo)
    f["foreign_persist_20"] = rsum((fo > 0).astype("float32").where(has_inst), 20) / 20 * 100   # 近20日外資買超天數%
    f["foreign_accel"] = f["foreign_net_5"] * 4 - f["foreign_net_20"]                        # 近5日買超速度 vs 20日
    f["trust_streak"] = streak(tr)
    f["trust_hold_est"] = rsum(tr, 250, 0.3) / sh * 100          # 投信近一年累積持股(估) %
    fr = W(qfii, "foreign_ratio", 5)
    f["foreign_ratio"] = fr
    f["foreign_ratio_chg_20"] = chg(fr, 20)
    f["foreign_ratio_chg_60"] = chg(fr, 60)
    f["foreign_room"] = W(qfii, "foreign_cap", 5) - fr

    # --- 信用 (margin balances are in lots of 1000 shares)
    mb = W(margin, "margin_bal", 3)      # margin is published at night: carry last value
    sb = W(margin, "short_bal", 3)
    f["margin_ratio"] = mb * 1000 / sh * 100
    f["margin_chg_20"] = pct_chg(mb, 20)
    f["margin_chg_5"] = pct_chg(mb, 5)
    f["margin_use"] = mb / W(margin, "margin_limit", 3).where(lambda x: x > 0) * 100
    f["short_margin_ratio"] = sb / mb.where(mb > 0) * 100
    f["short_chg_20"] = pct_chg(sb, 20)

    # --- 借券
    sbl_bal = W(sbl, "sbl_bal", 3)
    f["sbl_ratio"] = sbl_bal / sh * 100
    f["sbl_chg_20"] = chg(f["sbl_ratio"], 20)
    f["sbl_sell_20"] = rsum(W(sbl, "sbl_sell"), 20) / rsum(vol, 20).where(lambda x: x > 0) * 100
    adv = (rsum(vol, 20) / 20).where(lambda x: x > 0)
    f["sbl_dtc"] = sbl_bal / adv                         # 借券賣出餘額 / 日均量 = 回補天數
    f["short_dtc"] = sb * 1000 / adv                     # 融券餘額 / 日均量

    # --- 當沖
    f["daytrade_ratio_20"] = rsum(W(dtr, "daytrade_shares"), 20) / rsum(vol, 20).where(lambda x: x > 0) * 100

    # --- 集保 (weekly -> forward fill up to 8 trading days)
    if not tdcc.empty:
        tw_ = {c: W(tdcc, c, 8) for c in ("big400_pct", "big1000_pct", "retail_pct", "holders",
                                          "avg_lots_per_holder", "mid_pct")}
        f["big400_pct"] = tw_["big400_pct"]
        f["big1000_pct"] = tw_["big1000_pct"]
        f["retail_pct"] = tw_["retail_pct"]
        f["holders"] = tw_["holders"]
        # changes vs N weeks ago, computed on the weekly series then spread to days
        wk = tdcc.sort_values("date")
        for col, nm in (("big400_pct", "big400"), ("big1000_pct", "big1000"), ("retail_pct", "retail")):
            for n in (1, 4, 12):
                wk[f"{nm}_chg_{n}w"] = wk.groupby("code")[col].diff(n)
        wk["holders_chg_4w"] = wk.groupby("code")["holders"].pct_change(4, fill_method=None) * 100
        wk["avg_lots_chg_4w"] = wk.groupby("code")["avg_lots_per_holder"].pct_change(4, fill_method=None) * 100
        for c in [c for c in wk.columns if "_chg_" in c]:
            f[c] = W(wk, c, 8)

    # --- 董監 (monthly)
    if not ins.empty:
        ins = ins[pd.to_datetime(ins["date"]).dt.day == 1]      # valid snapshots are monthly (資料年月)
        ins = ins.sort_values("date")
        ins["insider_net_chg_raw"] = ins.groupby("code")["insider_shares"].diff()
        f["insider_ratio"] = W(ins, "insider_shares", 70) / sh * 100
        f["insider_pledge_ratio"] = W(ins, "insider_pledge", 70) / W(ins, "insider_shares", 70).where(lambda x: x > 0) * 100
        f["insider_net_chg"] = W(ins, "insider_net_chg_raw", 70) / sh * 100

    # --- 價格背景
    f.update(price_context(px, vol, shares))

    # 主力(近似): 法人買超 + 大戶增持 同向；籌碼背離：價跌但法人買
    f["chip_price_divergence"] = (f["foreign_net_20"].rank(axis=1, pct=True) +
                                  f["trust_net_20"].rank(axis=1, pct=True)) / 2 - f["ret_20"].rank(axis=1, pct=True)

    # 散戶熱度: day trading, margin growth / usage and retail holders rising together (Barber-Lee-Liu-Odean:
    # Taiwan day traders lose money in aggregate) - used as a contrarian factor
    heat = [f[k].rank(axis=1, pct=True) for k in ("daytrade_ratio_20", "margin_chg_20", "margin_use", "retail_chg_4w")
            if k in f]
    if heat:
        f["retail_heat"] = sum(h.fillna(0.5) for h in heat) / len(heat) * 100
        f["retail_heat"] = f["retail_heat"].where(px.notna())

    # liquidity: ≥ 200 lots/day (20d avg) and price ≥ 10
    liquid = (rsum(vol, 20) / 20 >= 200_000) & (W(quote, "close", 5).fillna(px) >= 10)

    meta = pd.DataFrame(index=codes)
    if not uni.empty:
        u = uni.sort_values("date").drop_duplicates("code", keep="last").set_index("code")
        meta = meta.join(u[["name", "industry", "mkt"]])
    if not quote.empty:
        qn = quote.sort_values("date").drop_duplicates("code", keep="last").set_index("code")
        meta["name"] = meta.get("name", pd.Series(dtype=str)).fillna(qn["name"]) if "name" in meta else qn["name"]
        meta["mkt"] = meta["mkt"].fillna(qn["mkt"]) if "mkt" in meta else qn["mkt"]

    # market-level series
    ms = {}
    tx = store.load("taifex", s)
    if not tx.empty:
        tx = tx.assign(date=pd.to_datetime(tx["date"])).set_index("date").sort_index()
        if "fut_foreign_oi" in tx:
            ms["外資台指期淨未平倉(口)"] = tx["fut_foreign_oi"]
        if {"opt_foreign_call_amt", "opt_foreign_put_amt"} <= set(tx.columns):
            ms["外資選擇權淨未平倉(億元)"] = (tx["opt_foreign_call_amt"] - tx["opt_foreign_put_amt"]) / 1e5
    tot_inst = store.load("inst", s)
    if not tot_inst.empty:
        g = tot_inst.groupby("date")
        ms["外資買賣超(億股)"] = (g["foreign"].sum() + g["foreign_dealer"].sum()) / 1e8
        ms["投信買賣超(億股)"] = g["trust"].sum() / 1e8
    if not margin.empty:
        ms["融資餘額(萬張)"] = margin.groupby("date")["margin_bal"].sum() / 1e4
        ms["融券餘額(萬張)"] = margin.groupby("date")["short_bal"].sum() / 1e4
    if not sbl.empty:
        ms["借券賣出餘額(億股)"] = sbl.groupby("date")["sbl_bal"].sum() / 1e8

    spec = {
        "inst": [("foreign_net_20", 1), ("foreign_net_60", 1), ("trust_net_20", 1), ("trust_net_60", 1),
                 ("foreign_ratio_chg_20", 1), ("foreign_streak", 1), ("trust_streak", 1), ("dealer_net_20", 1),
                 ("foreign_persist_20", 1), ("foreign_accel", 1)],
        "holders": [("big400_chg_4w", 1), ("big400_chg_12w", 1), ("big1000_chg_4w", 1),
                    ("retail_chg_4w", -1), ("holders_chg_4w", -1), ("avg_lots_chg_4w", 1)],
        "margin": [("margin_chg_20", -1), ("margin_ratio", -1), ("margin_use", -1),
                   ("short_margin_ratio", 1), ("daytrade_ratio_20", -1), ("retail_heat", -1), ("short_dtc", 1)],
        "short": [("sbl_chg_20", -1), ("sbl_ratio", -1), ("sbl_sell_20", -1), ("sbl_dtc", -1)],
        "insider": [("insider_net_chg", 1), ("insider_pledge_ratio", -1)],
    }
    weights = {"inst": 0.35, "holders": 0.25, "margin": 0.15, "short": 0.15, "insider": 0.10}
    labels = LABELS
    series = ["foreign_net_20", "trust_net_20", "foreign_ratio", "big400_pct", "big1000_pct", "retail_pct",
              "holders", "margin_ratio", "short_margin_ratio", "sbl_ratio", "daytrade_ratio_20", "insider_pledge_ratio"]
    qual += foreign_flow_check(W(qfii, "foreign_shares", 0), fo, sh)
    fs_ = FactorSet(MARKET, cal, codes, f, px, meta, spec, weights, labels, series, liquid, ms)
    fs_.quality = qual
    return fs_


def official_adjusted(C, yc, ya, exr, cal, codes):
    """Dividend/split-adjusted prices built from OFFICIAL exchange closes.

    Daily return = official close / previous official close, times the official ex-rights factor
    (除權息前收盤價 / 參考價) on ex-dates. Where no official event exists but Yahoo's adjustment
    changes (e.g. capital reductions, TPEx history), Yahoo's factor is used. Before official quotes
    start, Yahoo's adjusted returns fill the gap. Prices are scaled to today's official close.
    Returns (adjusted prices, quality checks).
    """
    import numpy as np
    checks = []
    F = pd.DataFrame(1.0, index=cal, columns=codes)
    n_off = 0
    if exr is not None and not exr.empty:
        e = exr.drop_duplicates(["date", "code"], keep="last")
        e = e[e["code"].isin(codes) & e["date"].isin(cal)]
        for r in e.itertuples():
            F.at[r.date, r.code] = r.pre_close / r.ref_price
        n_off = len(e)
    yk = (ya / yc)                                   # Yahoo cumulative adjustment ratio
    yf = (yk / yk.shift(1))                          # Yahoo per-day adjustment factor
    use_y = (F == 1.0) & ((yf - 1).abs() > 0.005)    # events Yahoo knows that the official table lacks
    F = F.where(~use_y, yf)
    # official vs Yahoo factor on official ex-dates
    both = (F != 1.0) & ~use_y & yf.notna()
    if both.values.any():
        diff = ((((F / yf) - 1).abs() > 0.01) & both).values.sum()
        dm = ((((F / yf) - 1).abs() > 0.01) & both).stack()
        ex = ", ".join(f"{c} {d.date()}" for d, c in dm[dm].index[-5:])
        checks.append(dict(name="除權息調整：官方參考價 vs Yahoo", ok=True,
                           detail=f"官方除權息事件 {int(both.values.sum())} 筆，與 Yahoo 調整差異超過 1% 的有 {int(diff)} 筆"
                                  + (f"（如 {ex}）；一律採用官方參考價" if diff else "")))
    ro = (C / C.shift(1)) * F
    ry = ya / ya.shift(1)
    r = ro.where(ro.notna(), ry)
    have = C.notna() | ya.notna()
    idx = r.fillna(1.0).where(have.cumsum() > 0).cumprod()
    idx = idx.where(have)
    last_px = C.ffill().iloc[-1].fillna(yc.ffill().iloc[-1])
    px = (idx / idx.ffill().iloc[-1] * last_px).astype("float32")
    # jumps beyond the ±10% daily limit (after adjustment) usually mean bad data or a missing event
    # New listings (first 5 trading days) and resumptions after capital reduction have no price limit.
    traded = C.notna().astype(int)
    age = traded.cumsum()
    gap = traded.rolling(6, min_periods=1).sum() < 6          # a halt in the last 5 days (e.g. 減資停止買賣)
    nolimit = (age <= 5) | gap
    jump = (((ro > 1.105) | (ro < 0.895)) & ~nolimit).iloc[-60:]
    nj = int(jump.values.sum())
    js = jump.stack()
    ex = ", ".join(f"{c} {d.date()} {ro.at[d, c] - 1:+.0%}" for d, c in js[js].index[-6:])
    checks.append(dict(name="漲跌幅合理性（官方收盤，近 60 日）", ok=nj <= 5,
                       detail=f"排除新上市前 5 日與停止買賣後恢復交易，還原後單日漲跌超過 ±10.5% 的有 {nj} 筆"
                              + (f"（如 {ex}）" if nj else "")))
    # official vs Yahoo closes, last 20 days
    rec = (C.iloc[-20:], yc.iloc[-20:])
    both_px = rec[0].notna() & rec[1].notna()
    if both_px.values.any():
        mism = ((((rec[0] - rec[1]).abs() / rec[0]) > 0.01) & both_px)
        rate = float(mism.values.sum()) / both_px.values.sum()
        checks.append(dict(name="收盤價：證交所/櫃買 vs Yahoo（近 20 日）", ok=rate < 0.01,
                           detail=f"比對 {int(both_px.values.sum())} 筆，差異超過 1% 的比例 {rate:.2%}；模型一律使用官方收盤價"))
    checks.append(dict(name="除權息資料來源", ok=True,
                       detail=f"官方除權息事件 {n_off} 筆；官方表未列、改用 Yahoo 調整的事件 {int(use_y.values.sum())} 筆"))
    return px, checks


def foreign_flow_check(fshares, fo, sh):
    """外資持股變化應與外資(含外資自營商)買賣超大致相同。"""
    d_hold = (fshares - fshares.shift(1)).iloc[-20:]
    flow = fo.iloc[-20:]
    ok = d_hold.notna() & flow.notna() & (flow != 0)
    if not ok.values.any():
        return []
    tol = (sh.iloc[-20:] * 0.001).clip(lower=2000)       # 0.1% of shares or 2 lots
    match = (((d_hold - flow).abs() <= tol) & ok)
    rate = float(match.values.sum()) / ok.values.sum()
    return [dict(name="外資持股變化 vs 外資買賣超（近 20 日）", ok=rate >= 0.85,
                 detail=f"比對 {int(ok.values.sum())} 筆，一致比例 {rate:.1%}（容許股本 0.1%；除權、轉換公司債等會造成少量差異）")]


LABELS = {
    "foreign_net_5": "外資5日買賣超(股本bp)", "foreign_net_20": "外資20日買賣超(股本bp)",
    "foreign_net_60": "外資60日買賣超(股本bp)", "trust_net_5": "投信5日買賣超(股本bp)",
    "trust_net_20": "投信20日買賣超(股本bp)", "trust_net_60": "投信60日買賣超(股本bp)",
    "dealer_net_20": "自營商(自行)20日買賣超(股本bp)", "dealer_hedge_20": "自營商(避險)20日買賣超(股本bp)",
    "inst_vol_share_20": "法人買賣超佔成交量%(20日)", "foreign_streak": "外資連續買(+)/賣(-)天數",
    "trust_streak": "投信連續買(+)/賣(-)天數", "trust_hold_est": "投信近一年累積持股(估)%",
    "foreign_ratio": "外資持股比%", "foreign_ratio_chg_20": "外資持股比20日變化(pp)",
    "foreign_ratio_chg_60": "外資持股比60日變化(pp)", "foreign_room": "外資可投資空間(pp)",
    "margin_ratio": "融資餘額佔股本%", "margin_chg_20": "融資餘額20日增減%", "margin_chg_5": "融資餘額5日增減%",
    "margin_use": "融資使用率%", "short_margin_ratio": "券資比%", "short_chg_20": "融券餘額20日增減%",
    "sbl_ratio": "借券賣出餘額佔股本%", "sbl_chg_20": "借券賣出餘額20日變化(pp)",
    "sbl_sell_20": "借券賣出佔成交量%(20日)", "daytrade_ratio_20": "當沖比%(20日)",
    "big400_pct": "400張以上大戶持股%", "big1000_pct": "1000張以上大戶持股%", "retail_pct": "50張以下散戶持股%",
    "holders": "集保股東人數", "big400_chg_1w": "400張大戶週變化(pp)", "big400_chg_4w": "400張大戶4週變化(pp)",
    "big400_chg_12w": "400張大戶12週變化(pp)", "big1000_chg_1w": "1000張大戶週變化(pp)",
    "big1000_chg_4w": "1000張大戶4週變化(pp)", "big1000_chg_12w": "1000張大戶12週變化(pp)",
    "sbl_dtc": "借券回補天數", "short_dtc": "融券回補天數", "retail_heat": "散戶熱度(0-100)",
    "foreign_persist_20": "近20日外資買超天數%", "foreign_accel": "外資買超加速度",
    "retail_chg_1w": "散戶週變化(pp)", "retail_chg_4w": "散戶4週變化(pp)", "retail_chg_12w": "散戶12週變化(pp)",
    "holders_chg_4w": "股東人數4週變化%", "avg_lots_chg_4w": "人均持股4週變化%",
    "insider_ratio": "董監持股%", "insider_pledge_ratio": "董監質押比%", "insider_net_chg": "董監持股月變化(股本%)",
    "ret_20": "20日漲跌%", "ret_60": "60日漲跌%", "ret_250": "一年漲跌%", "vol_60": "年化波動%",
    "dist_52w_high": "距52週高點%", "turnover_20": "日均週轉率%", "chip_price_divergence": "籌碼價格背離度",
}


# ================================================================ market interface
KEY_FACTORS = ["foreign_net_5", "foreign_net_20", "trust_net_5", "trust_net_20", "foreign_streak", "trust_streak",
               "foreign_ratio", "foreign_ratio_chg_20", "trust_hold_est", "big400_pct", "big400_chg_4w",
               "big1000_chg_4w", "retail_chg_4w", "holders_chg_4w", "margin_chg_20", "margin_use",
               "short_margin_ratio", "sbl_ratio", "sbl_chg_20", "daytrade_ratio_20", "insider_pledge_ratio",
               "insider_net_chg", "ret_20", "ret_60", "ret_250"]


def symbols(store: Store) -> dict[str, str]:
    uni = store.load("universe")
    if uni.empty:
        q = store.load("quote")
        uni = q.drop_duplicates("code", keep="last")[["code", "mkt"]] if not q.empty else uni
    return {r.code: yahoo_symbol(r.code, r.mkt) for r in uni.drop_duplicates("code", keep="last").itertuples()}


def daily(store: Store, d: dt.date) -> dict:
    """Run after the evening publication (margin ~21:30 TPE). Also heals the last few days."""
    res = {}
    for back in range(6, -1, -1):
        dd = d - dt.timedelta(days=back)
        if dd.weekday() < 5:
            res[str(dd)] = update_day(store, dd)
    update_weekly_monthly(store)
    return res


def backfill_day(store: Store, d: dt.date) -> dict:
    res = update_day(store, d)
    # official ex-rights history, one request per calendar month
    first = d.replace(day=1)
    if d.weekday() < 5 and d - first < dt.timedelta(days=7) and \
            not any(x.year == d.year and x.month == d.month for x in store.dates("_exright_months")):
        nxt = (first + dt.timedelta(days=32)).replace(day=1) - dt.timedelta(days=1)
        try:
            store.append("exright", fetch_exright(first, nxt))
            store.append("_exright_months", pd.DataFrame([dict(date=pd.Timestamp(first), code="done")]))
        except Exception as e:
            log.warning("exright %s: %s", first, e)
    return res
FRESHNESS_DATASETS = ["price", "quote", "inst", "margin", "sbl", "qfii", "daytrade", "tdcc", "insider", "exright", "tpex_ref"]


SOURCES = {
    "price": ("證交所／櫃買中心官方收盤價＋官方除權息參考價（Yahoo 只用於補缺與交叉比對）", True),
    "quote": ("證交所 MI_INDEX／櫃買中心每日收盤行情", True),
    "inst": ("證交所 T86／櫃買中心三大法人買賣明細", True),
    "margin": ("證交所 MI_MARGN／櫃買中心融資融券餘額", True),
    "sbl": ("證交所 TWT93U／櫃買中心借券賣出餘額", True),
    "qfii": ("證交所 MI_QFIIS／櫃買中心外資持股", True),
    "daytrade": ("證交所 TWTB4U 當日沖銷", True),
    "tdcc": ("集保結算所 股權分散表（開放資料 1-5）", True),
    "insider": ("證交所／櫃買中心 OpenAPI 董監持股與質押（公開資訊觀測站）", True),
    "exright": ("證交所 TWT49U 除權除息計算結果", True),
    "tpex_ref": ("櫃買中心每日收盤行情的「次日參考價」（上櫃除權息、減資調整）", True),
    "taifex": ("期交所 三大法人 台指期／台指選擇權未平倉（大盤環境）", True),
}
