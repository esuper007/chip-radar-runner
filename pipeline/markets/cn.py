"""China A-shares (Shanghai + Shenzhen) via AKShare (East Money / exchange sites).

  margin   融资融券明细 per stock, daily (SSE + SZSE)            backfillable by date
  flow     主力/超大单/大单/中单/小单 净流入 (today snapshot)        daily, accumulates
  lhb      龙虎榜 机构买卖每日统计                                   backfillable by range
  gdhs     股东户数 (latest disclosure per stock)                    periodic
  ggcg     高管/大股东 增减持                                        periodic (history included)
"""
from __future__ import annotations

import datetime as dt
import re
import unicodedata

import pandas as pd

from ..common import log, log_fields_once, ymd
from ..store import Store

MARKET = "cn"
TZ = 8
CODE_RE = re.compile(r"^(60|68|00|30)\d{4}$")


def _ak():
    import akshare as ak
    return ak


def _pick(df: pd.DataFrame, *keys: str, exclude: tuple = ()) -> str | None:
    return next((c for c in df.columns if all(k in c for k in keys) and not any(e in c for e in exclude)), None)


def _norm_code(s: pd.Series) -> pd.Series:
    return s.astype(str).str.extract(r"(\d{6})")[0]


def fetch_margin(d: dt.date) -> pd.DataFrame:
    ak, out = _ak(), []
    for fn, ex in (("stock_margin_detail_sse", "SSE"), ("stock_margin_detail_szse", "SZSE")):
        try:
            df = getattr(ak, fn)(date=ymd(d))
        except Exception as e:
            log.debug("%s %s: %s", fn, d, e)
            continue
        if df is None or df.empty:
            continue
        log_fields_once(fn, df.columns)
        out.append(pd.DataFrame({
            "code": _norm_code(df[_pick(df, "代码")]),
            "margin_bal": pd.to_numeric(df[_pick(df, "融资余额")], errors="coerce"),
            "margin_buy": pd.to_numeric(df[_pick(df, "融资买入额")], errors="coerce"),
            "short_qty": pd.to_numeric(df[_pick(df, "融券余量")], errors="coerce"),
        }))
    if not out:
        return pd.DataFrame()
    df = pd.concat(out)
    df = df[df["code"].fillna("").str.match(CODE_RE)]
    df["date"] = pd.Timestamp(d)
    return df


def fetch_flow(d: dt.date) -> pd.DataFrame:
    try:
        df = _ak().stock_individual_fund_flow_rank(indicator="今日")
    except Exception as e:                       # East Money push2 often refuses overseas IPs
        log.warning("em fund flow failed (%s); falling back to 10jqka", e)
        t = _ak().stock_fund_flow_individual(symbol="即时")
        log_fields_once("ths_flow", t.columns)
        out = pd.DataFrame({"code": _norm_code(t[_pick(t, "代码")])})

        def money(x):
            x = str(x).replace(",", "")
            mult = 1e8 if "亿" in x else 1e4 if "万" in x else 1.0
            try:
                return float(re.sub(r"[^\d.\-]", "", x)) * mult
            except ValueError:
                return None
        out["flow_main"] = t[_pick(t, "净额")].map(money)
        out = out[out["code"].fillna("").str.match(CODE_RE)]
        out["date"] = pd.Timestamp(d)
        return out
    log_fields_once("fund_flow", df.columns)
    out = pd.DataFrame({"code": _norm_code(df[_pick(df, "代码")])})
    for key, nm in (("主力净流入-净额", "main"), ("超大单净流入-净额", "xl"), ("大单净流入-净额", "l"),
                    ("中单净流入-净额", "m"), ("小单净流入-净额", "s"), ("主力净流入-净占比", "main_pct")):
        c = _pick(df, key)
        out[f"flow_{nm}"] = pd.to_numeric(df[c], errors="coerce") if c else None
    out = out[out["code"].fillna("").str.match(CODE_RE)]
    out["date"] = pd.Timestamp(d)
    return out


def fetch_lhb(start: dt.date, end: dt.date) -> pd.DataFrame:
    try:
        df = _ak().stock_lhb_jgmmtj_em(start_date=ymd(start), end_date=ymd(end))
    except Exception as e:
        log.warning("lhb %s-%s: %s", start, end, e)
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    log_fields_once("lhb", df.columns)
    out = pd.DataFrame({"code": _norm_code(df[_pick(df, "代码")]),
                        "date": pd.to_datetime(df[_pick(df, "日期")], errors="coerce"),
                        "lhb_inst_net": pd.to_numeric(df[_pick(df, "机构", "净买", exclude=("占",))] if _pick(df, "机构", "净买", exclude=("占",)) else df[_pick(df, "净额")], errors="coerce")})
    out = out.dropna(subset=["date"])
    return out.groupby(["date", "code"], as_index=False)["lhb_inst_net"].sum()


def fetch_gdhs() -> pd.DataFrame:
    df = _ak().stock_zh_a_gdhs(symbol="最新")
    log_fields_once("gdhs", df.columns)
    c_date = _pick(df, "统计截止日") or _pick(df, "截止日") or _pick(df, "日期")
    out = pd.DataFrame({"code": _norm_code(df[_pick(df, "代码")]),
                        "holders": pd.to_numeric(df[_pick(df, "股东户数", "本次")], errors="coerce"),
                        "holders_chg_pct": pd.to_numeric(df[_pick(df, "增减比例")], errors="coerce"),
                        "date": pd.to_datetime(df[c_date], errors="coerce")})
    return out.dropna(subset=["date"])


def fetch_ggcg() -> pd.DataFrame:
    df = _ak().stock_ggcg_em(symbol="全部")
    log_fields_once("ggcg", df.columns)
    c_dir = _pick(df, "增减")
    c_qty = _pick(df, "变动数量") or _pick(df, "数量")
    c_date = _pick(df, "公告日") or _pick(df, "变动截止日") or _pick(df, "日期")
    c_pct = _pick(df, "占总股本比例") or _pick(df, "占流通股比例")
    out = pd.DataFrame({"code": _norm_code(df[_pick(df, "代码")]),
                        "date": pd.to_datetime(df[c_date], errors="coerce"),
                        "sign": df[c_dir].astype(str).map(lambda x: 1 if "增" in x else -1),
                        "pct": pd.to_numeric(df[c_pct], errors="coerce") if c_pct else 0.0})
    out["holder_net_pct"] = out["sign"] * out["pct"].fillna(0)
    out = out.dropna(subset=["date"])
    return out.groupby(["date", "code"], as_index=False)["holder_net_pct"].sum()


def fetch_north(d: dt.date | None = None) -> pd.DataFrame:
    """Northbound (陆股通) holdings per A-share, from HKEXnews (t=sh Shanghai, t=sz Shenzhen).
    With `d`, asks the page's search form for that past date (about one year is kept)."""
    import io as _io
    import requests
    from ..common import http_get, num
    frames = []
    for t in ("sh", "sz"):
        url = f"https://www3.hkexnews.hk/sdw/search/mutualmarket_c.aspx?t={t}"
        try:
            html = http_get(url, retries=2).text
            if d is not None:
                form = dict(re.findall(r'<input[^>]+name="([^"]+)"[^>]*value="([^"]*)"', html))
                form.update({"txtShareholdingDate": d.strftime("%Y/%m/%d"), "btnSearch": "搜尋"})
                html = requests.post(url, data=form, headers={"User-Agent": "Mozilla/5.0"}, timeout=60).text
            tb = max(pd.read_html(_io.StringIO(html)), key=len)
        except Exception as e:
            log.warning("north %s: %s", t, e)
            continue
        tb.columns = [re.sub(r"\s+", "", str(c)) for c in tb.columns]
        tb = tb.apply(lambda col: col.astype(str).str.split(":").str[-1].str.strip())
        m = re.search(r"(\d{4}/\d{2}/\d{2})", html)
        if d is not None and (not m or pd.Timestamp(m.group(1)).date() != d):
            continue                                   # holiday: the page shows another date
        c_pct = next((c for c in tb.columns if "%" in c or "百分比" in c), None)
        c_sh = next((c for c in tb.columns if "持股量" in c), None)
        code = tb[tb.columns[0]].astype(str).str.extract(r"(\d{5})")[0]
        # HKEX codes: SSE 9xxxx (90000→600000, 91318→601318, 98xxx→688xxx);
        #             SZSE 7xxxx (70001→000001, 72xxx→002xxx, 77xxx→300xxx, 78xxx→301xxx)
        def to_a(c):
            if not isinstance(c, str) or len(c) != 5:
                return None
            r = c[1:]
            if t == "sh" and c[0] == "9":
                return ("68" if r[0] == "8" else "60") + r
            if t == "sz" and c[0] == "7":
                return {"7": "300" + r[1:], "8": "301" + r[1:]}.get(r[0], "00" + r)
            return None
        frames.append(pd.DataFrame({"code": code.map(to_a), "north_shares": tb[c_sh].map(num) if c_sh else None,
                                    "north_pct": tb[c_pct].map(num) if c_pct else None,
                                    "date": pd.Timestamp(m.group(1)) if m else pd.Timestamp.today().normalize()}))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames)
    log.info("north sample codes: %s", df["code"].head(5).tolist())
    return df[df["code"].fillna("").str.match(CODE_RE)]


def fetch_universe() -> pd.DataFrame:
    try:
        df = _ak().stock_info_a_code_name()          # exchange lists (SSE/SZSE/BSE)
    except Exception as e:
        log.warning("stock_info_a_code_name failed (%s); trying East Money", e)
        df = _ak().stock_zh_a_spot_em()
    log_fields_once("cn-universe", df.columns)
    out = pd.DataFrame({"code": _norm_code(df[_pick(df, "code") or _pick(df, "代码")]),
                        "name": df[_pick(df, "name") or _pick(df, "名称")].astype(str)
                                .map(lambda x: re.sub(r"\s+", "", unicodedata.normalize("NFKC", x)))})   # "万  科Ａ" -> "万科A"
    out = out[out["code"].fillna("").str.match(CODE_RE)]
    out["mkt"] = out["code"].map(lambda c: "SSE" if c.startswith("6") else "SZSE")
    out["industry"] = ""
    return out


def margin_done(store: Store) -> set:
    """Dates where BOTH exchanges' margin detail is stored (a day with only SSE is re-fetched)."""
    m = store.load("margin")
    if m.empty:
        return set()
    ex = m["code"].str[0].map(lambda c: "SSE" if c == "6" else "SZSE")
    g = m.assign(ex=ex).groupby("date")["ex"].nunique()
    return set(pd.to_datetime(g[g == 2].index).date)


def daily(store: Store, d: dt.date) -> dict:
    res = {}
    done = margin_done(store)
    for back in range(6, -1, -1):
        dd = d - dt.timedelta(days=back)
        if dd.weekday() < 5 and dd not in done:
            m = fetch_margin(dd)
            store.append("margin", m)
            res[f"margin {dd}"] = len(m)
    for name, fn in (("north", fetch_north), ("flow", lambda: fetch_flow(d)), ("lhb", lambda: fetch_lhb(d - dt.timedelta(days=7), d)),
                     ("gdhs", fetch_gdhs), ("ggcg", fetch_ggcg)):
        try:
            df = fn()
            store.append(name, df)
            res[name] = len(df)
        except Exception as e:
            log.error("cn %s failed: %s", name, e)
    try:
        uni = fetch_universe()
        uni["date"] = pd.Timestamp(d)
        store.append("universe", uni)
    except Exception as e:
        log.error("cn universe: %s", e)
    return res


def backfill_day(store: Store, d: dt.date) -> dict:
    res = {}
    nb_ = store.load("north")
    good = set(pd.to_datetime(nb_.loc[nb_["north_pct"].notna(), "date"]).dt.date) if not nb_.empty else set()
    if d >= dt.date.today() - dt.timedelta(days=360) and d not in good:
        nb = fetch_north(d)
        store.append("north", nb)
        res["north"] = len(nb)
    if d not in margin_done(store):
        m = fetch_margin(d)
        store.append("margin", m)
        res["margin"] = len(m)
    if d.weekday() == 4:   # one LHB call per week
        store.append("lhb", fetch_lhb(d - dt.timedelta(days=6), d))
    if d.month in (1, 4, 7, 10) and d.day <= 7 and d.weekday() == 0:
        q = (pd.Timestamp(d) - pd.DateOffset(days=10)) + pd.offsets.QuarterEnd(0)
        try:
            df = _ak().stock_zh_a_gdhs(symbol=q.strftime("%Y%m%d"))
            c_date = _pick(df, "统计截止日") or _pick(df, "截止日")
            store.append("gdhs", pd.DataFrame({
                "code": _norm_code(df[_pick(df, "代码")]),
                "holders": pd.to_numeric(df[_pick(df, "股东户数", "本次")], errors="coerce"),
                "holders_chg_pct": pd.to_numeric(df[_pick(df, "增减比例")], errors="coerce"),
                "date": pd.to_datetime(df[c_date], errors="coerce")}).dropna(subset=["date"]))
        except Exception as e:
            log.warning("gdhs %s: %s", q, e)
    return res


# ---------------------------------------------------------------- delisted stocks (survivorship bias)
DELISTED_META = "delisted_tried"
DELISTED_PER_RUN = 400


def fetch_delisted() -> pd.DataFrame:
    """Official lists of terminated listings from SSE and SZSE: code, name, delisting date."""
    ak, out = _ak(), []
    for fn, kw, c_code, c_name, c_date in (("stock_info_sh_delist", {"symbol": "全部"}, "公司代码", "公司简称", "暂停上市日期"),
                                           ("stock_info_sz_delist", {"symbol": "终止上市公司"}, "证券代码", "证券简称", "终止上市日期")):
        try:
            df = getattr(ak, fn)(**kw)
            log_fields_once(f"cn-{fn}", df.columns)
            out.append(pd.DataFrame({"code": _norm_code(df[c_code]), "name": df[c_name].astype(str),
                                     "date": pd.to_datetime(df[c_date], errors="coerce")}))
        except Exception as e:
            log.warning("cn %s failed: %s", fn, e)
    if not out:
        return pd.DataFrame(columns=["code", "name", "date"])
    df = pd.concat(out, ignore_index=True).dropna(subset=["code", "date"])
    return df[df["code"].str.match(CODE_RE)].drop_duplicates("code", keep="last")


class _Baostock:
    """Baostock (free, keeps delisted A-shares). East Money and Sina refuse requests from GitHub's servers."""
    def __init__(self):
        self.bs = None
        try:
            import baostock as bs
            if bs.login().error_code == "0":
                self.bs = bs
        except Exception as e:
            log.warning("baostock unavailable: %s", type(e).__name__)

    def hist(self, code: str, adjust: str) -> pd.DataFrame:
        if self.bs is None:
            return pd.DataFrame()
        sym = ("sh." if code.startswith("6") else "sz.") + code
        try:
            rs = self.bs.query_history_k_data_plus(sym, "date,close,volume", start_date="2013-01-01",
                                                   end_date=dt.date.today().isoformat(), frequency="d",
                                                   adjustflag="1" if adjust == "hfq" else "3")
            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())
        except Exception as e:
            log.warning("baostock history failed: %s", type(e).__name__)
            return pd.DataFrame()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["date", "close", "volume"])
        return pd.DataFrame({"date": pd.to_datetime(df["date"]), "close": pd.to_numeric(df["close"], errors="coerce"),
                             "volume": pd.to_numeric(df["volume"], errors="coerce")})        # shares

    def close(self):
        if self.bs is not None:
            try:
                self.bs.logout()
            except Exception:
                pass


def _hist(code: str, adjust: str, bsq: "_Baostock | None" = None) -> pd.DataFrame:
    if bsq is not None:
        df = bsq.hist(code, adjust)
        if not df.empty and df["close"].notna().sum() > 20:
            return df
    ak = _ak()
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date="20130101", end_date="20991231", adjust=adjust)
    except Exception as e:
        log.warning("cn delisted history (east money) failed: %s", type(e).__name__)
        df = pd.DataFrame()
    if df is None or df.empty:
        try:     # Sina keeps delisted symbols too
            sd = ak.stock_zh_a_daily(symbol=("sh" if code.startswith("6") else "sz") + code, adjust=adjust)
            return pd.DataFrame({"date": pd.to_datetime(sd["date"]), "close": pd.to_numeric(sd["close"], errors="coerce"),
                                 "volume": pd.to_numeric(sd["volume"], errors="coerce")})
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame({"date": pd.to_datetime(df["日期"]), "close": pd.to_numeric(df["收盘"], errors="coerce"),
                         "volume": pd.to_numeric(df["成交量"], errors="coerce") * 100})     # 手 -> shares


MAX_ATTEMPTS = 3


def delisted_prices(store: Store) -> None:
    """Yahoo only serves stocks that still trade, so delisted A-shares had no price at all and dropped out
    of every backtest. Fill them in a few at a time from the exchanges' official delisting lists."""
    import time
    dl = fetch_delisted()
    if dl.empty:
        return
    dl = dl[dl["date"] >= pd.Timestamp("2014-01-01")]
    store.append("delisted", dl, keys=("date", "code"))
    have = set(store.load("price")["code"].unique())
    meta = store.load_meta(DELISTED_META) or {}
    tries = {c: 1 for c in meta} if isinstance(meta, list) else dict(meta)     # old format: plain list
    todo = [c for c in dl["code"] if c not in have and tries.get(c, 0) < MAX_ATTEMPTS][:DELISTED_PER_RUN]
    log.info("cn delisted: %d listed since 2014, %d without prices, fetching %d now", len(dl),
             sum(1 for c in dl["code"] if c not in have), len(todo))
    if not todo:
        return
    bsq = _Baostock()
    rows = []
    try:
        for c in todo:
            raw, adj = _hist(c, "", bsq), _hist(c, "hfq", bsq)
            tries[c] = tries.get(c, 0) + 1
            if raw.empty or adj.empty:
                continue
            m = raw.merge(adj[["date", "close"]].rename(columns={"close": "adj_close"}), on="date", how="inner")
            m["code"] = c
            m = m[(m["volume"].fillna(0) > 0)]           # suspended days carry the old close with no trades
            rows.append(m[["date", "code", "adj_close", "close", "volume"]].dropna(subset=["close", "adj_close"]))
            time.sleep(0.3)
    finally:
        bsq.close()
    if rows:
        df = pd.concat(rows, ignore_index=True)
        store.append("price", df)
        log.info("cn delisted: added prices for %d stocks (%d rows)", df["code"].nunique(), len(df))
    store.save_meta(DELISTED_META, tries)


def symbols(store: Store) -> dict[str, str]:
    u = store.load("universe")
    if u.empty:
        return {}
    return {c: f"{c}.SS" if c.startswith("6") else f"{c}.SZ" for c in u.drop_duplicates("code", keep="last")["code"]}


# ================================================================ factors
def build_factors(store: Store, since: str = "2014-01-01"):
    from ..factors import FactorSet, chg, pct_chg, price_context, rsum, since_start, streak, wide

    s = pd.Timestamp(since)
    price, mg, fl, lhb, gd, gg = (store.load(x, s) for x in ("price", "margin", "flow", "lhb", "gdhs", "ggcg"))
    nb = store.load("north", s)
    uni = store.load("universe")
    cal = pd.DatetimeIndex(sorted(pd.to_datetime(price["date"].unique())))
    codes = pd.Index(sorted(set(price["code"].unique())), name="code")
    W = lambda df, col, lim=0: wide(df, col, cal, codes, lim)
    px, vol, close = W(price, "adj_close"), W(price, "volume"), W(price, "close")
    amount = close * vol
    f = {}
    if not mg.empty:
        mb = W(mg, "margin_bal", 3)        # exchanges publish margin the next morning
        f["margin_amt_ratio"] = mb / (rsum(amount, 20) / 20).where(lambda x: x > 0)    # 融资余额/日均成交额
        f["margin_chg_20"] = pct_chg(mb, 20)
        f["margin_chg_5"] = pct_chg(mb, 5)
        f["short_qty_chg_20"] = pct_chg(W(mg, "short_qty", 3), 20)
        f["short_dtc"] = W(mg, "short_qty", 3) / (rsum(vol, 20) / 20).where(lambda x: x > 0)   # 融券余量/日均量
    if not fl.empty:
        a20 = rsum(amount, 20, 0.3).where(lambda x: x > 0)
        f["main_flow_20"] = rsum(W(fl, "flow_main"), 20, 0.3) / a20 * 100
        f["main_flow_5"] = rsum(W(fl, "flow_main"), 5, 0.4) / (rsum(amount, 5, 0.4)).where(lambda x: x > 0) * 100
        f["small_flow_20"] = rsum(W(fl, "flow_s"), 20, 0.3) / a20 * 100
        f["main_streak"] = streak(W(fl, "flow_main"))
    if not nb.empty:
        f["north_pct"] = W(nb, "north_pct", 5)
        f["north_chg_5"] = chg(f["north_pct"], 5)
        f["north_chg_20"] = chg(f["north_pct"], 20)
    if not lhb.empty:
        f["lhb_inst_60"] = rsum(since_start(W(lhb, "lhb_inst_net").fillna(0), lhb), 60, 0) / (rsum(amount, 60) / 60).where(lambda x: x > 0)
    if not gd.empty:
        f["holders_chg_q"] = W(gd, "holders_chg_pct", 130)
    if not gg.empty:
        f["insider_net_chg"] = rsum(since_start(W(gg, "holder_net_pct").fillna(0), gg), 126, 0)
    f.update(price_context(px, vol, None))
    liquid = (rsum(amount, 20) / 20 >= 30_000_000) & px.notna()
    u = uni.sort_values("date").drop_duplicates("code", keep="last").set_index("code") if not uni.empty else pd.DataFrame()
    meta = u.reindex(codes)[["name", "industry", "mkt"]] if not u.empty else pd.DataFrame(index=codes)
    dl = store.load("delisted")
    if not dl.empty and "name" in meta:                 # delisted stocks are not in today's list: name them
        nm = dl.drop_duplicates("code", keep="last").set_index("code")["name"]
        miss = meta["name"].isna() & meta.index.isin(nm.index)
        meta.loc[miss, "name"] = nm.reindex(meta.index[miss]).values + "(已下市)"
        meta.loc[miss, "mkt"] = [("SSE" if c.startswith("6") else "SZSE") for c in meta.index[miss]]
        meta["industry"] = meta["industry"].fillna("")
    ms = {}
    if not mg.empty:
        ms["两融融资余额(亿元)"] = mg.groupby("date")["margin_bal"].sum() / 1e8
    if not fl.empty:
        ms["主力净流入(亿元)"] = fl.groupby("date")["flow_main"].sum() / 1e8
    spec = {"inst": [("main_flow_20", 1), ("main_flow_5", 1), ("main_streak", 1), ("lhb_inst_60", 1),
                     ("north_chg_5", 1), ("north_chg_20", 1)],
            "holders": [("holders_chg_q", -1), ("small_flow_20", -1)],
            "margin": [("margin_chg_20", -1), ("margin_amt_ratio", -1), ("short_qty_chg_20", -1), ("short_dtc", -1)],
            "insider": [("insider_net_chg", 1)]}
    weights = {"inst": 0.4, "holders": 0.25, "margin": 0.2, "insider": 0.15}
    series = ["north_pct", "main_flow_20", "small_flow_20", "margin_amt_ratio", "holders_chg_q"]
    return FactorSet(MARKET, cal, codes, f, px, meta, spec, weights, LABELS, series, liquid, ms)


LABELS = {"north_pct": "陆股通(北向)持股%", "north_chg_5": "北向持股5日变化(pp)", "north_chg_20": "北向持股20日变化(pp)",
          "margin_amt_ratio": "融资余额/日均成交额(天)", "margin_chg_20": "融资余额20日变化%",
          "margin_chg_5": "融资余额5日变化%", "short_qty_chg_20": "融券余量20日变化%", "short_dtc": "融券回补天数",
          "main_flow_20": "主力20日净流入(成交额%)", "main_flow_5": "主力5日净流入(成交额%)",
          "small_flow_20": "小单(散户)20日净流入(成交额%)", "main_streak": "主力连续流入(+)/流出(-)天数",
          "lhb_inst_60": "龙虎榜机构60日净买(日均额倍数)", "holders_chg_q": "股东户数最近一期变化%",
          "insider_net_chg": "大股东/高管半年净增减持(股本%)", "ret_20": "20日涨跌%", "ret_60": "60日涨跌%",
          "ret_250": "一年涨跌%", "vol_60": "年化波动%", "dist_52w_high": "距52周高点%"}
KEY_FACTORS = ["north_pct", "north_chg_20", "main_flow_5", "main_flow_20", "main_streak", "small_flow_20", "lhb_inst_60", "margin_chg_20",
               "margin_amt_ratio", "short_qty_chg_20", "holders_chg_q", "insider_net_chg", "ret_20", "ret_60", "ret_250"]
FRESHNESS_DATASETS = ["price", "north", "margin", "flow", "lhb", "gdhs", "ggcg"]


SOURCES = {
    "price": ("Yahoo Finance", False),
    "margin": ("上交所／深交所 融資融券明細（經 AKShare 擷取交易所網站）", True),
    "north": ("港交所 HKEXnews 陸股通持股", True),
    "flow": ("東方財富／同花順 主力資金（業者自行定義大單，非官方）", False),
    "lhb": ("東方財富整理之交易所龍虎榜機構席位", False),
    "gdhs": ("東方財富整理之上市公司股東戶數公告", False),
    "ggcg": ("東方財富整理之大股東／高管增減持公告", False),
}
