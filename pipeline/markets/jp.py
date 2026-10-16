"""Japan (TSE): JPX public statistics.

  short_pos  空売り残高報告 (≥0.5% positions) per stock, daily        JPX short-selling page (Excel)
  margin     信用取引残高 per stock (買残/売残), daily for 日々公表 + weekly for all stocks
  invtype    投資部門別売買状況 (market level, weekly): 海外投資家/個人/信託銀行… net buy
Prices: Yahoo `<code>.T`.
Files are discovered by crawling JPX pages for Excel links, so URL renames do not break it.
"""
from __future__ import annotations

import datetime as dt
import io
import re

import pandas as pd

from ..common import http_bytes, http_get, log, log_fields_once
from ..store import Store

MARKET = "jp"
TZ = 9
JPX = "https://www.jpx.co.jp"
CODE_RE = re.compile(r"^\d{4}[0-9A-Z]?$")

PAGES = {
    "short_pos": ["/markets/public/short-selling/index.html"],
    "margin": ["/markets/statistics-equities/margin/01.html"],
    "invtype": ["/markets/statistics-equities/investor-type/index.html"],
}


def _links(page: str) -> list[str]:
    try:
        html = http_get(JPX + page, retries=2).text
    except Exception as e:
        log.warning("JPX page %s: %s", page, e)
        return []
    hrefs = re.findall(r'href="([^"]+\.(?:xlsx?|csv|pdf))"', html, re.I)
    return [h if h.startswith("http") else JPX + h for h in hrefs]


def _date_in(url: str) -> dt.date | None:
    m = re.search(r"(20\d{6})", url)
    if not m:
        m6 = re.search(r"(?<!\d)(\d{2})(\d{2})(\d{2})(?!\d)", url.rsplit("/", 1)[-1])
        if not m6:
            return None
        try:
            return dt.date(2000 + int(m6.group(1)), int(m6.group(2)), int(m6.group(3)))
        except ValueError:
            return None
    s = m.group(1)
    try:
        return dt.date(int(s[:4]), int(s[4:6]), int(s[6:]))
    except ValueError:
        return None


def _read_excel(b: bytes) -> list[pd.DataFrame]:
    try:
        x = pd.read_excel(io.BytesIO(b), sheet_name=None, header=None)
    except Exception as e:
        log.warning("excel read failed: %s", e)
        return []
    return list(x.values())


def _header_frame(raw: pd.DataFrame, must: str) -> pd.DataFrame | None:
    """Find the header row containing `must` (e.g. 'コード'), return frame with that header."""
    for i in range(min(40, len(raw))):
        row = [str(v) for v in raw.iloc[i].tolist()]
        if any(must in v for v in row):
            df = raw.iloc[i + 1:].copy()
            df.columns = [re.sub(r"\s+", "", v) for v in row]
            return df
    return None


# ---------------------------------------------------------------- short positions (aggregate per stock)
def parse_short_positions(b: bytes, d: dt.date) -> pd.DataFrame:
    for raw in _read_excel(b):
        df = _header_frame(raw, "コード") if raw is not None else None
        if df is None:
            continue
        log_fields_once("jp-short", df.columns)
        c_code = next((c for c in df.columns if "コード" in c), None)
        c_ratio = next((c for c in df.columns if "割合" in c and "前" not in c), None)
        c_qty = next((c for c in df.columns if "残高数量" in c and "前" not in c), None)
        if not (c_code and c_ratio):
            continue
        df["code"] = df[c_code].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
        df = df[df["code"].str.match(CODE_RE)]
        df["ratio"] = pd.to_numeric(df[c_ratio], errors="coerce")
        if df["ratio"].dropna().median() < 0.05:      # stored as fraction
            df["ratio"] *= 100
        g = df.groupby("code").agg(short_pos_pct=("ratio", "sum"), short_reporters=("ratio", "size")).reset_index()
        if c_qty:
            g = g.merge(df.assign(q=pd.to_numeric(df[c_qty], errors="coerce")).groupby("code")["q"].sum()
                        .rename("short_pos_qty").reset_index(), on="code")
        g["date"] = pd.Timestamp(d)
        return g
    return pd.DataFrame()


# ---------------------------------------------------------------- margin balances (per stock)
def parse_margin(b: bytes, d: dt.date) -> pd.DataFrame:
    for raw in _read_excel(b):
        df = _header_frame(raw, "コード")
        if df is None:
            continue
        log_fields_once(f"jp-margin-{len(df.columns)}", df.columns)
        cols = list(df.columns)
        c_code = next((c for c in cols if "コード" in c), None)
        buy = [c for c in cols if ("買残" in c or "買い残" in c) and "前" not in c and "比" not in c]
        sell = [c for c in cols if ("売残" in c or "売り残" in c) and "前" not in c and "比" not in c]
        if not (c_code and buy and sell):
            continue
        out = pd.DataFrame({"code": df[c_code].astype(str).str.replace(r"\.0$", "", regex=True).str.strip(),
                            "margin_buy_bal": pd.to_numeric(df[buy[0]], errors="coerce"),
                            "margin_sell_bal": pd.to_numeric(df[sell[0]], errors="coerce")})
        out = out[out["code"].str.match(CODE_RE)].groupby("code").sum().reset_index()
        out["date"] = pd.Timestamp(d)
        return out
    return pd.DataFrame()


MTALL_RE = re.compile(r"\s([0-9]{3}[0-9A-Z])[0-9A-Z]\s+JP\w{10}\s*株数\s*Shs\.\s+([\d,]+)\s+(▲\s*)?[\d,]+\s+([\d.]+)%\s+"
                      r"([\d,]+)\s+(▲\s*)?[\d,]+\s+([\d.]+)%")


def parse_margin_pdf(b: bytes, d: dt.date) -> pd.DataFrame:
    """銘柄別信用取引残高 (daily, all issues, PDF): 売残高/買残高 shares and % of listed shares."""
    import pdfplumber
    rows = []
    with pdfplumber.open(io.BytesIO(b)) as pdf:
        for page in pdf.pages:
            for m in MTALL_RE.finditer(" " + (page.extract_text() or "").replace("\n", " \n ")):
                rows.append(dict(code=m.group(1), margin_sell_bal=float(m.group(2).replace(",", "")),
                                 margin_sell_pct=float(m.group(4)), margin_buy_bal=float(m.group(5).replace(",", "")),
                                 margin_buy_pct=float(m.group(7))))
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.groupby("code", as_index=False).sum()
        df["date"] = pd.Timestamp(d)
    log.info("JP margin pdf %s: %d issues", d, len(df))
    return df


# ---------------------------------------------------------------- investor type (market level)
def parse_invtype(b: bytes, d: dt.date) -> pd.DataFrame:
    rows = []
    for raw in _read_excel(b):
        txt = raw.astype(str)
        for label, key in (("海外投資家", "foreign"), ("個人", "individual"), ("信託銀行", "trust_bank"),
                           ("投資信託", "inv_trust"), ("事業法人", "corporate"), ("証券自己計", "proprietary")):
            hit = txt.apply(lambda col: col.str.contains(label, na=False)).values.nonzero()
            if len(hit[0]):
                r = raw.iloc[hit[0][0]]
                nums = pd.to_numeric(r, errors="coerce").dropna()
                if len(nums) >= 3:
                    rows.append(dict(code=key, sell=nums.iloc[0], buy=nums.iloc[1] if len(nums) > 1 else None,
                                     net=nums.iloc[-1]))
        if rows:
            break
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.Timestamp(d)
    return df


# ---------------------------------------------------------------- orchestration
def _collect(store: Store, kind: str, parser, max_files: int = 40) -> int:
    dataset = "mtall" if kind == "margin" else kind      # daily all-issue margin balances (PDF)
    have = store.dates(dataset)
    n = 0
    links = []
    if kind == "margin":
        parser = parse_margin_pdf
    for p in PAGES[kind]:
        links += _links(p)
    for url in dict.fromkeys(links):
        d = _date_in(url)
        if d is None or d in have:
            continue
        if kind == "short_pos" and "Short_Positions" not in url:
            continue
        if kind == "invtype" and not re.search(r"\.xlsx?$", url, re.I):
            continue
        if kind == "margin" and not re.search(r"mtall\.pdf$", url, re.I):
            continue
        try:
            df = parser(http_bytes(url, retries=2), d)
        except Exception as e:
            log.warning("JPX %s %s: %s", kind, url, e)
            continue
        if not df.empty:
            store.append(dataset, df)
            n += 1
        if n >= max_files:
            break
    log.info("JP %s: %d new files", kind, n)
    return n


def fetch_universe() -> pd.DataFrame:
    """TSE listed issues (Excel on JPX: 東証上場銘柄一覧)."""
    for url in _links("/markets/statistics-equities/misc/01.html"):
        if "data_j" in url or url.endswith(".xls"):
            for raw in _read_excel(http_bytes(url)):
                df = _header_frame(raw, "コード")
                if df is None:
                    continue
                c_name = next((c for c in df.columns if "銘柄名" in c), None)
                c_mkt = next((c for c in df.columns if "市場" in c), None)
                c_ind = next((c for c in df.columns if "33業種区分" in c), None)
                out = pd.DataFrame({"code": df[next(c for c in df.columns if "コード" in c)].astype(str)
                                    .str.replace(r"\.0$", "", regex=True),
                                    "name": df[c_name], "mkt": df[c_mkt] if c_mkt else "",
                                    "industry": df[c_ind] if c_ind else ""})
                out = out[out["code"].str.match(CODE_RE) & out["mkt"].astype(str).str.contains("プライム|スタンダード|グロース")]
                return out
    return pd.DataFrame()


def daily(store: Store, d: dt.date) -> dict:
    res = {k: _collect(store, k, fn) for k, fn in
           (("short_pos", parse_short_positions), ("margin", parse_margin), ("invtype", parse_invtype))}
    uni = fetch_universe()
    if not uni.empty:
        uni["date"] = pd.Timestamp(d)
        store.append("universe", uni)
    return res


def backfill_day(store: Store, d: dt.date) -> dict:
    # JPX keeps only recent files online; history accumulates from the daily job.
    return {}


def symbols(store: Store) -> dict[str, str]:
    u = store.load("universe")
    return {} if u.empty else {c: f"{c}.T" for c in u.drop_duplicates("code", keep="last")["code"]}


# ================================================================ factors
def build_factors(store: Store, since: str = "2014-01-01"):
    from ..factors import FactorSet, chg, pct_chg, price_context, rsum, since_start, wide

    s = pd.Timestamp(since)
    price, sp, mg, inv = (store.load(x, s) for x in ("price", "short_pos", "mtall", "invtype"))
    uni = store.load("universe")
    cal = pd.DatetimeIndex(sorted(pd.to_datetime(price["date"].unique())))
    u = uni.sort_values("date").drop_duplicates("code", keep="last").set_index("code")
    codes = pd.Index(sorted(set(price["code"].unique())), name="code")
    W = lambda df, col, lim=0: wide(df, col, cal, codes, lim)
    px, vol = W(price, "adj_close"), W(price, "volume")
    f = {}
    if not sp.empty:
        f["short_pos_pct"] = since_start(W(sp, "short_pos_pct", 5).fillna(0), sp).where(px.notna())
        f["short_pos_chg_20"] = chg(f["short_pos_pct"], 20)
        f["short_reporters"] = since_start(W(sp, "short_reporters", 5).fillna(0), sp).where(px.notna())
    if not mg.empty:
        b, sl = W(mg, "margin_buy_bal", 3), W(mg, "margin_sell_bal", 3)
        f["margin_buy_pct"] = W(mg, "margin_buy_pct", 3)          # 買殘佔上市股數%
        f["margin_sell_pct"] = W(mg, "margin_sell_pct", 3)
        f["margin_ratio_jp"] = b / sl.where(sl > 0)                      # 信用倍率
        f["margin_buy_vol"] = b / (rsum(vol, 20) / 20).where(lambda x: x > 0)   # 買残 / 平均出来高 (日数)
        f["margin_buy_chg_20"] = pct_chg(b, 20)
        f["margin_sell_chg_20"] = pct_chg(sl, 20)
    f.update(price_context(px, vol, None))
    liquid = (rsum(W(price, "close") * vol, 20) / 20 >= 100_000_000) & px.notna()
    meta = u.reindex(codes)[["name", "industry", "mkt"]] if not u.empty else pd.DataFrame(index=codes)
    ms = {}
    if not inv.empty:
        for k, nm in (("foreign", "海外投資家淨買(億円)"), ("individual", "個人淨買(億円)"), ("trust_bank", "信託銀行淨買(億円)")):
            x = inv[inv["code"] == k].set_index("date")["net"]
            if len(x):
                ms[nm] = x / 1e5
    spec = {"short": [("short_pos_pct", -1), ("short_pos_chg_20", -1), ("short_reporters", -1)],
            "margin": [("margin_ratio_jp", -1), ("margin_buy_vol", -1), ("margin_buy_chg_20", -1),
                       ("margin_sell_chg_20", 1), ("margin_buy_pct", -1)]}
    weights = {"short": 0.5, "margin": 0.5}
    series = ["short_pos_pct", "margin_ratio_jp", "margin_buy_vol"]
    return FactorSet(MARKET, cal, codes, f, px, meta, spec, weights, LABELS, series, liquid, ms)


LABELS = {"short_pos_pct": "大口空賣餘額報告合計%", "short_pos_chg_20": "大口空賣20日變化(pp)",
          "short_reporters": "空賣申報機構數", "margin_ratio_jp": "信用倍率(買殘/賣殘)",
          "margin_buy_vol": "融資買殘/日均量(天)", "margin_buy_chg_20": "買殘20日變化%",
          "margin_sell_chg_20": "賣殘20日變化%", "margin_buy_pct": "融資買殘佔上市股數%",
          "margin_sell_pct": "融券賣殘佔上市股數%", "ret_20": "20日漲跌%", "ret_60": "60日漲跌%",
          "ret_250": "一年漲跌%", "vol_60": "年化波動%", "dist_52w_high": "距52週高點%"}
KEY_FACTORS = ["short_pos_pct", "short_pos_chg_20", "short_reporters", "margin_ratio_jp", "margin_buy_pct",
               "margin_sell_pct", "margin_buy_vol",
               "margin_buy_chg_20", "margin_sell_chg_20", "ret_20", "ret_60", "ret_250"]
FRESHNESS_DATASETS = ["price", "short_pos", "mtall", "invtype"]


SOURCES = {
    "price": ("Yahoo Finance（JPX 官方股價需付費 J-Quants）", False),
    "short_pos": ("JPX 空売り残高報告", True),
    "mtall": ("JPX 銘柄別信用取引残高（PDF）", True),
    "invtype": ("JPX 投資部門別売買状況", True),
}
