"""Hong Kong (HKEX main board).

  southbound  港股通(南向) holdings per stock, daily       HKEXnews mutual-market page (sh+sz connect)
  shortsell   short-selling turnover per stock, daily      HKEX ASHTMAIN text report
  shortpos    SFC aggregated reportable short positions    weekly CSV
Prices: Yahoo `<code>.HK`.
"""
from __future__ import annotations

import datetime as dt
import io
import re

import pandas as pd

from ..common import http_get, log, log_fields_once, num
from ..store import Store

MARKET = "hk"
TZ = 8


def _code(x) -> str | None:
    m = re.search(r"(\d{1,5})", str(x))
    return m.group(1).zfill(5) if m else None


def fetch_southbound(d: dt.date | None = None) -> pd.DataFrame:
    """Combined Shanghai+Shenzhen connect southbound holdings."""
    frames = []
    for t in ("hk",):               # t=hk: HK shares held via Shanghai & Shenzhen Connect (southbound)
        url = f"https://www3.hkexnews.hk/sdw/search/mutualmarket_c.aspx?t={t}"
        try:
            html = http_get(url, retries=2).text
            tables = pd.read_html(io.StringIO(html))
        except Exception as e:
            log.warning("southbound %s: %s", t, e)
            continue
        tb = max(tables, key=len) if tables else None
        if tb is None or tb.empty:
            continue
        tb.columns = [re.sub(r"\s+", "", str(c)) for c in tb.columns]
        tb = tb.apply(lambda col: col.astype(str).str.split(":").str[-1].str.strip())
        log_fields_once(f"hk-south-{t}", tb.columns)
        c_code = next((c for c in tb.columns if "代號" in c or "Code" in c), tb.columns[0])
        c_sh = next((c for c in tb.columns if "持股量" in c or "Shareholding" in c), None)
        c_pct = next((c for c in tb.columns if "%" in c or "百分比" in c), None)
        m = re.search(r"(\d{4}/\d{2}/\d{2})", html)
        asof = pd.Timestamp(m.group(1)) if m else pd.Timestamp(d or dt.date.today())
        frames.append(pd.DataFrame({"code": tb[c_code].map(_code),
                                    "sb_shares": tb[c_sh].map(num) if c_sh else None,
                                    "sb_pct": tb[c_pct].map(num) if c_pct else None, "date": asof}))
    if not frames:
        return pd.DataFrame()
    return clean_southbound(pd.concat(frames))


def clean_southbound(df: pd.DataFrame) -> pd.DataFrame:
    """Keep real HK stock codes (00001-09999) with a holding; HKEXnews sometimes serves the
    Shenzhen/Shanghai (northbound) table or an empty pre-publication page instead."""
    if df.empty:
        return df
    df = df.dropna(subset=["code", "sb_shares"])
    df = df[df["code"].astype(int) < 10000]
    if len(df) < 300:                      # a real day lists ~1000 southbound-eligible stocks
        log.warning("southbound: only %d rows - ignored", len(df))
        return pd.DataFrame()
    return df.groupby(["date", "code"], as_index=False)[["sb_shares", "sb_pct"]].sum(min_count=1)


def fetch_shortsell(d: dt.date) -> pd.DataFrame:
    try:
        txt = http_get("https://www.hkex.com.hk/eng/stat/smstat/ssturnover/ncms/ASHTMAIN.HTM", retries=2).text
    except Exception as e:
        log.warning("HK short selling: %s", e)
        return pd.DataFrame()
    m = re.search(r"(\d{1,2})\s+([A-Z]{3})\s+(\d{4})", txt)
    asof = pd.Timestamp(f"{m.group(1)} {m.group(2)} {m.group(3)}") if m else pd.Timestamp(d)
    rows = {}
    for line in re.sub(r"<[^>]+>", "", txt).splitlines():
        mm = re.match(r"^\s*\*?\s*(\d{1,5})\s+(.+?)\s+([\d,]+)\s+([\d,]+)\s*$", line)
        if mm:
            c = mm.group(1).zfill(5)
            sh, val = num(mm.group(3)), num(mm.group(4))
            a = rows.setdefault(c, [0.0, 0.0])           # morning + afternoon sessions add up
            a[0] += sh or 0
            a[1] += val or 0
    return pd.DataFrame([{"date": asof, "code": c, "ss_shares": v[0], "ss_value": v[1]} for c, v in rows.items()])


SFC_LINE = re.compile(r"^\s*(\d{1,5})\s+(.+?)\s+([\d,]+)\s+\$\s*([\d,]+)\s*$")


def fetch_shortpos(d: dt.date, exact: bool = False) -> pd.DataFrame:
    """SFC weekly aggregated reportable short positions (PDF, as of each Friday; published ~1 week later)."""
    import pdfplumber
    fri = d - dt.timedelta(days=(d.weekday() - 4) % 7)
    for back in range(0, 1 if exact else 3):
        x = fri - dt.timedelta(days=7 * back)
        url = (f"https://www.sfc.hk/-/media/EN/pdf/spr/{x:%Y}/{x:%m}/{x:%d}/"
               f"Short_Position_Reporting_Aggregated_Data_Eng_{x:%Y%m%d}.pdf")
        try:
            b = http_get(url, retries=1).content
        except Exception:
            continue
        if not b.lstrip().startswith(b"%PDF"):
            continue
        rows = []
        with pdfplumber.open(io.BytesIO(b)) as pdf:
            for page in pdf.pages:
                for line in (page.extract_text() or "").splitlines():
                    m = SFC_LINE.match(line)
                    if m:
                        rows.append({"code": m.group(1).zfill(5), "sp_shares": num(m.group(3)),
                                     "sp_value": num(m.group(4))})
        if rows:
            df = pd.DataFrame(rows)
            df["date"] = pd.Timestamp(x)
            log.info("SFC short positions %s: %d stocks", x, len(df))
            return df
    return pd.DataFrame()


def fetch_southbound_hist(d: dt.date) -> pd.DataFrame:
    """Southbound holdings on a past date (HKEXnews keeps about one year) via the page's search form."""
    url = "https://www3.hkexnews.hk/sdw/search/mutualmarket_c.aspx?t=hk"
    try:
        html = http_get(url, retries=2).text
        form = dict(re.findall(r'<input[^>]+name="([^"]+)"[^>]*value="([^"]*)"', html))
        form.update({"txtShareholdingDate": d.strftime("%Y/%m/%d"), "btnSearch": "搜尋"})
        import requests
        r = requests.post(url, data=form, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
        tb = max(pd.read_html(io.StringIO(r.text)), key=len)
    except Exception as e:
        log.warning("southbound history %s: %s", d, e)
        return pd.DataFrame()
    tb.columns = [re.sub(r"\s+", "", str(c)) for c in tb.columns]
    tb = tb.apply(lambda col: col.astype(str).str.split(":").str[-1].str.strip())
    c_code = next((c for c in tb.columns if "代號" in c), tb.columns[0])
    c_sh = next((c for c in tb.columns if "持股量" in c), None)
    c_pct = next((c for c in tb.columns if "%" in c or "百分比" in c), None)
    m = re.search(r"(\d{4}/\d{2}/\d{2})", r.text)
    asof = pd.Timestamp(m.group(1)) if m else pd.Timestamp(d)
    if asof.date() != d:            # holiday: page falls back to another date
        return pd.DataFrame()
    return clean_southbound(pd.DataFrame({"code": tb[c_code].map(_code), "sb_shares": tb[c_sh].map(num) if c_sh else None,
                                          "sb_pct": tb[c_pct].map(num) if c_pct else None, "date": asof}))


def fetch_universe() -> pd.DataFrame:
    """HKEX List of Securities (Excel): ordinary shares on the Main Board and GEM."""
    try:
        b = http_get("https://www.hkex.com.hk/eng/services/trading/securities/securitieslists/ListOfSecurities.xlsx",
                     retries=2).content
        raw = pd.read_excel(io.BytesIO(b), header=None)
    except Exception as e:
        log.warning("hk universe: %s", e)
        return pd.DataFrame()
    hdr = next(i for i in range(10) if any("Stock Code" in str(v) for v in raw.iloc[i]))
    df = raw.iloc[hdr + 1:].copy()
    df.columns = [str(v).strip() for v in raw.iloc[hdr]]
    log_fields_once("hk-universe", df.columns)
    cat = next((c for c in df.columns if c == "Category"), None)
    sub = next((c for c in df.columns if "Sub-Category" in c), None)
    if cat:
        df = df[df[cat].astype(str).str.contains("Equity")]
    if sub:
        df = df[df[sub].astype(str).str.contains("Equity Securities")]
    out = pd.DataFrame({"code": df["Stock Code"].map(_code), "name": df[next(c for c in df.columns if "Name" in c)],
                        "mkt": df[sub].astype(str).str.contains("GEM").map({True: "GEM", False: "Main"}) if sub else "HKEX",
                        "industry": ""})
    return out.dropna(subset=["code"])


def purge_bad_southbound(store: Store) -> None:
    sb = store.load("southbound")
    if sb.empty:
        return
    bad = sb["code"].astype(int).ge(10000) | sb["sb_shares"].isna()
    if bad.any():
        log.warning("southbound: purging %d bad rows (non-HK codes or empty)", int(bad.sum()))
        store.rewrite("southbound", sb[~bad])


def daily(store: Store, d: dt.date) -> dict:
    res = {}
    purge_bad_southbound(store)
    for name, fn in (("southbound", lambda: fetch_southbound(d)), ("shortsell", lambda: fetch_shortsell(d)),
                     ("shortpos", lambda: fetch_shortpos(d))):
        try:
            df = fn()
            store.append(name, df)
            res[name] = len(df)
        except Exception as e:
            log.error("hk %s: %s", name, e)
    uni = fetch_universe()
    if not uni.empty:
        uni["date"] = pd.Timestamp(d)
        store.append("universe", uni)
    return res


def backfill_day(store: Store, d: dt.date) -> dict:
    res = {}
    if d.weekday() == 4 and d not in store.dates("shortpos"):
        df = fetch_shortpos(d, exact=True)
        store.append("shortpos", df)
        res["shortpos"] = len(df)
    sb = store.load("southbound")
    good = set(pd.to_datetime(sb.loc[sb["sb_pct"].notna(), "date"]).dt.date) if not sb.empty else set()
    if d >= dt.date.today() - dt.timedelta(days=360) and d not in good:
        df = fetch_southbound_hist(d)
        store.append("southbound", df)
        res["southbound"] = len(df)
    return res


def symbols(store: Store) -> dict[str, str]:
    u = store.load("universe")
    if u.empty:
        return {}
    # Yahoo uses 4-digit codes for HK (0700.HK)
    return {c: f"{int(c):04d}.HK" for c in u.drop_duplicates("code", keep="last")["code"]}


# ================================================================ factors
def build_factors(store: Store, since: str = "2014-01-01"):
    from ..factors import FactorSet, chg, pct_chg, price_context, rsum, wide

    s = pd.Timestamp(since)
    price, sb, ss, sp = (store.load(x, s) for x in ("price", "southbound", "shortsell", "shortpos"))
    uni = store.load("universe")
    cal = pd.DatetimeIndex(sorted(pd.to_datetime(price["date"].unique())))
    codes = pd.Index(sorted(set(price["code"].unique())), name="code")
    W = lambda df, col, lim=0: wide(df, col, cal, codes, lim)
    px, vol, close = W(price, "adj_close"), W(price, "volume"), W(price, "close")
    f = {}
    if not sb.empty:
        f["south_pct"] = W(sb, "sb_pct", 5)
        f["south_chg_5"] = chg(f["south_pct"], 5)
        f["south_chg_20"] = chg(f["south_pct"], 20)
        f["north_chg_20"] = f["south_chg_20"]          # reuse flag "陸股通20日增持" in signal flags
    if not ss.empty:
        f["short_sell_ratio_20"] = rsum(W(ss, "ss_shares"), 20, 0.3) / rsum(vol, 20).where(lambda x: x > 0) * 100
    if not sp.empty:
        shares_est = None
        f["short_pos_value_ratio"] = W(sp, "sp_value", 10) / (close * vol).rolling(20, min_periods=5).mean().where(lambda x: x > 0)
        f["short_pos_chg_4w"] = pct_chg(W(sp, "sp_shares", 10), 20)
    f.update(price_context(px, vol, None))
    liquid = (rsum(close * vol, 20) / 20 >= 10_000_000) & px.notna()
    u = uni.sort_values("date").drop_duplicates("code", keep="last").set_index("code") if not uni.empty else pd.DataFrame()
    meta = u.reindex(codes)[["name", "industry", "mkt"]] if not u.empty else pd.DataFrame(index=codes)
    ms = {}
    if not ss.empty:
        ms["沽空成交(億港元)"] = ss.groupby("date")["ss_value"].sum() / 1e8
    spec = {"inst": [("south_chg_5", 1), ("south_chg_20", 1)],
            "short": [("short_sell_ratio_20", -1), ("short_pos_value_ratio", -1), ("short_pos_chg_4w", -1)]}
    weights = {"inst": 0.5, "short": 0.5}
    series = ["south_pct", "short_sell_ratio_20", "short_pos_value_ratio"]
    return FactorSet(MARKET, cal, codes, f, px, meta, spec, weights, LABELS, series, liquid, ms)


LABELS = {"south_pct": "港股通(南向)持股%", "south_chg_5": "南向持股5日變化(pp)", "south_chg_20": "南向持股20日變化(pp)",
          "north_chg_20": "南向持股20日變化(pp)", "short_sell_ratio_20": "沽空佔成交量%(20日)",
          "short_pos_value_ratio": "淡倉市值/日均成交額(天)", "short_pos_chg_4w": "淡倉4週變化%",
          "ret_20": "20日漲跌%", "ret_60": "60日漲跌%", "ret_250": "一年漲跌%", "vol_60": "年化波動%",
          "dist_52w_high": "距52週高點%"}
KEY_FACTORS = ["south_pct", "south_chg_5", "south_chg_20", "short_sell_ratio_20", "short_pos_value_ratio",
               "short_pos_chg_4w", "ret_20", "ret_60", "ret_250"]
FRESHNESS_DATASETS = ["price", "southbound", "shortsell", "shortpos"]


SOURCES = {
    "price": ("Yahoo Finance", False),
    "southbound": ("港交所 HKEXnews 港股通持股", True),
    "shortsell": ("港交所 每日沽空成交報告", True),
    "shortpos": ("證監會 SFC 淡倉申報合計", True),
}
