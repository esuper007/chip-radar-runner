"""United States: SEC + FINRA public data, full market.

Datasets:
  universe   SEC company_tickers_exchange.json (ticker, cik, name, exchange)
  shares     SEC XBRL frames dei:EntityCommonStockSharesOutstanding (latest per company)
  shortvol   FINRA Reg SHO daily short-sale volume (CNMS consolidated)           daily
  si         FINRA consolidated short interest (twice a month)                    semi-monthly
  insider    SEC Form 4 open-market buys (P) / sells (S), aggregated per ticker/day daily
  activist   SEC SC 13D / 13G filings per subject company                          daily
  ftd        SEC fails-to-deliver (also gives CUSIP->ticker map)                  semi-monthly
  inst13f    SEC Form 13F holdings aggregated per CUSIP (shares, #managers)        quarterly
  funds      tickers that are funds, excluded from the stock universe:              monthly
             CIKs filing N-CSR / N-CSRS / N-CEN / NPORT-P (closed-end funds, ETFs, other
             registered investment companies) + Nasdaq Trader symbol directory ETF flag
"""
from __future__ import annotations

import datetime as dt
import io
import re
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd

from ..common import http_bytes, http_get, http_json, log, num, weekdays, ymd
from ..store import Store

MARKET = "us"
TZ = -5
SEC = "https://www.sec.gov"
TICKER_RE = re.compile(r"^[A-Z]{1,5}$")


# ---------------------------------------------------------------- universe & shares
def fetch_universe() -> pd.DataFrame:
    j = http_json(f"{SEC}/files/company_tickers_exchange.json")
    df = pd.DataFrame(j["data"], columns=j["fields"])
    df = df.rename(columns={"ticker": "code", "exchange": "mkt"})
    df = df[df["mkt"].isin(["NYSE", "Nasdaq", "NYSE American", "CBOE"])]
    df = df[df["code"].astype(str).str.match(TICKER_RE)]
    return df[["code", "cik", "name", "mkt"]].drop_duplicates("code")


def fetch_shares() -> pd.DataFrame:
    today = dt.date.today()
    frames = []
    for back in range(1, 6):
        y, q = today.year, (today.month - 1) // 3 + 1 - back
        while q <= 0:
            y, q = y - 1, q + 4
        try:
            j = http_json(f"https://data.sec.gov/api/xbrl/frames/dei/EntityCommonStockSharesOutstanding/shares/CY{y}Q{q}I.json")
            frames.append(pd.DataFrame(j["data"]))
        except Exception as e:
            log.warning("shares frame CY%dQ%dI: %s", y, q, e)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames)
    df = df.sort_values("end").drop_duplicates("cik", keep="last")
    return df[["cik", "val", "end"]].rename(columns={"val": "shares"})


# N-2 is left out on purpose: business development companies (e.g. ARCC) file it but report like
# operating companies (10-K/10-Q) and trade as stocks.
FUND_FORMS = {"N-CSR", "N-CSRS", "N-CSR/A", "N-CSRS/A", "N-CEN", "N-CEN/A", "NPORT-P", "NPORT-P/A", "N-30D"}


def fetch_funds(today: dt.date | None = None) -> pd.DataFrame:
    """Codes to exclude: registered investment companies (closed-end funds, ETFs, interval funds).

    Source 1 (SEC, official): every CIK that filed a fund report (N-CSR, N-CEN, NPORT-P, ...) in the
    last four quarters of the EDGAR master index. Source 2 (Nasdaq Trader symbol directory): ETF flag,
    which also catches grantor-trust ETFs (e.g. gold trusts) that file 10-K instead."""
    import gzip
    today = today or dt.date.today()
    fund_ciks: set[int] = set()
    y, q = today.year, (today.month - 1) // 3 + 1
    for _ in range(5):
        base = f"{SEC}/Archives/edgar/full-index/{y}/QTR{q}/master"
        txt = None
        try:
            txt = gzip.decompress(http_bytes(base + ".gz", retries=2)).decode("latin-1")
        except Exception:
            try:
                txt = http_bytes(base + ".idx", retries=2).decode("latin-1")
            except Exception as e:
                log.warning("EDGAR master index %d Q%d: %s", y, q, e)
        if txt:
            for line in txt.splitlines():
                parts = line.split("|")
                if len(parts) == 5 and parts[2] in FUND_FORMS and parts[0].isdigit():
                    fund_ciks.add(int(parts[0]))
        q -= 1
        if q == 0:
            y, q = y - 1, 4
    uni = fetch_universe()
    rows = []
    if fund_ciks and not uni.empty:
        hit = uni[uni["cik"].astype(int).isin(fund_ciks)]
        rows += [dict(code=c, reason="SEC 基金申報") for c in hit["code"]]
    try:
        raw = http_bytes("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt", retries=2).decode("latin-1")
        df = pd.read_csv(io.StringIO(raw), sep="|", dtype=str)
        log_fields = list(df.columns)
        log.info("FIELDS nasdaqtraded: %s", log_fields)
        etf = df[(df.get("ETF") == "Y")]["Symbol"].dropna().str.replace(".", "-", regex=False)
        rows += [dict(code=c, reason="ETF") for c in etf]
    except Exception as e:
        log.warning("Nasdaq Trader symbol directory: %s", e)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).drop_duplicates("code")
    out["date"] = pd.Timestamp(today)
    log.info("funds: %d SEC fund CIKs -> %d excluded tickers", len(fund_ciks), len(out))
    return out


def fund_codes(store: Store) -> set:
    f = store.load("funds")
    if f.empty:
        return set()
    last = f["date"].max()
    return set(f.loc[f["date"] == last, "code"])


# ---------------------------------------------------------------- FINRA daily short volume
def fetch_shortvol(d: dt.date) -> pd.DataFrame:
    try:
        b = http_bytes(f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{ymd(d)}.txt", retries=2)
    except Exception:
        return pd.DataFrame()
    df = pd.read_csv(io.BytesIO(b), sep="|", dtype={"Symbol": str})
    df = df[df["Date"].astype(str).str.match(r"^\d{8}$")]
    df = df.rename(columns={"Symbol": "code", "ShortVolume": "short_vol", "TotalVolume": "total_vol",
                            "ShortExemptVolume": "short_exempt"})
    df["date"] = pd.Timestamp(d)
    return df[["date", "code", "short_vol", "short_exempt", "total_vol"]]


# ---------------------------------------------------------------- FINRA short interest
def fetch_si(settle: dt.date) -> pd.DataFrame:
    url = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
    rows, offset = [], 0
    while True:
        body = {"limit": 5000, "offset": offset,
                "compareFilters": [{"compareType": "EQUAL", "fieldName": "settlementDate",
                                    "fieldValue": settle.isoformat()}]}
        import requests
        r = requests.post(url, json=body, headers={"Accept": "application/json"}, timeout=90)
        if r.status_code != 200:
            log.warning("FINRA SI %s -> HTTP %s %s", settle, r.status_code, r.text[:200])
            break
        chunk = r.json() if r.text.strip() else []
        rows += chunk
        if len(chunk) < 5000:
            break
        offset += 5000
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    pick = lambda *keys: next((c for c in df.columns if all(k.lower() in c.lower() for k in keys)), None)
    out = pd.DataFrame({
        "code": df[pick("symbol")],
        "si": pd.to_numeric(df[pick("current", "short")], errors="coerce"),
        "si_prev": pd.to_numeric(df[pick("previous", "short")], errors="coerce"),
        "adv": pd.to_numeric(df[pick("average", "volume")], errors="coerce") if pick("average", "volume") else None,
        "dtc": pd.to_numeric(df[pick("days", "cover")], errors="coerce") if pick("days", "cover") else None,
    })
    out["date"] = pd.Timestamp(settle)
    return out[out["code"].astype(str).str.match(TICKER_RE)]


def si_settlement_dates(start: dt.date, end: dt.date) -> list[dt.date]:
    """FINRA settlement dates: 15th and last business day of each month (approx; empty ones skipped)."""
    out, d = [], dt.date(start.year, start.month, 1)
    while d <= end:
        mid = d.replace(day=15)
        while mid.weekday() >= 5:
            mid -= dt.timedelta(days=1)
        nxt = (d.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        last = nxt - dt.timedelta(days=1)
        while last.weekday() >= 5:
            last -= dt.timedelta(days=1)
        out += [x for x in (mid, last) if start <= x <= end]
        d = nxt
    return out


# ---------------------------------------------------------------- SEC daily index (Form 4, 13D/G)
def _daily_index(d: dt.date) -> pd.DataFrame:
    q = (d.month - 1) // 3 + 1
    try:
        txt = http_get(f"{SEC}/Archives/edgar/daily-index/{d.year}/QTR{q}/form.{ymd(d)}.idx", retries=2).text
    except Exception as e:
        log.warning("SEC daily index %s: %s", d, e)
        return pd.DataFrame()
    lines = txt.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith("----")), None)
    if start is None:
        return pd.DataFrame()
    rows = []
    for l in lines[start + 1:]:
        m = re.match(r"^(\S+(?: \S+)*?)\s{2,}(.+?)\s{2,}(\d+)\s+(\d{8})\s+(\S+)\s*$", l)
        if m:
            rows.append(dict(form=m.group(1), company=m.group(2), cik=int(m.group(3)), path=m.group(5)))
    return pd.DataFrame(rows)


def _form4_rows(path: str) -> list[dict]:
    # path like edgar/data/320193/0000320193-26-000101.txt -> full submission text containing the XML
    txt = http_get(f"{SEC}/Archives/{path}", retries=2).text
    m = re.search(r"<ownershipDocument>.*?</ownershipDocument>", txt, re.S)
    if not m:
        return []
    root = ET.fromstring(m.group(0))
    sym = (root.findtext("issuer/issuerTradingSymbol") or "").strip().upper()
    if not TICKER_RE.match(sym):
        return []
    owner = root.findtext("reportingOwner/reportingOwnerId/rptOwnerCik") or ""
    rel = root.find("reportingOwner/reportingOwnerRelationship")
    is_officer = rel is not None and (rel.findtext("isOfficer") in ("1", "true") or rel.findtext("isDirector") in ("1", "true"))
    out = []
    for t in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
        code = t.findtext("transactionCoding/transactionCode")
        if code not in ("P", "S"):
            continue
        sh = num(t.findtext("transactionAmounts/transactionShares/value")) or 0
        px = num(t.findtext("transactionAmounts/transactionPricePerShare/value")) or 0
        plan = (t.findtext("transactionCoding/equitySwapInvolved") or "")  # placeholder field
        out.append(dict(code=sym, owner=owner, buy=code == "P", shares=sh, value=sh * px, officer=is_officer))
    return out


def fetch_filings(d: dt.date, max_form4: int = 4000) -> dict[str, pd.DataFrame]:
    idx = _daily_index(d)
    res = {"insider": pd.DataFrame(), "activist": pd.DataFrame()}
    if idx.empty:
        return res
    f4 = idx[idx["form"] == "4"].drop_duplicates("path").head(max_form4)
    tx = []
    for p in f4["path"]:
        try:
            tx += _form4_rows(p)
        except Exception as e:
            log.debug("form4 %s: %s", p, e)
    if tx:
        t = pd.DataFrame(tx)
        t["buy_value"] = t["value"].where(t["buy"], 0)
        t["sell_value"] = t["value"].where(~t["buy"], 0)
        t["buyer"] = t["owner"].where(t["buy"])
        g = t.groupby("code").agg(ins_buy_value=("buy_value", "sum"), ins_sell_value=("sell_value", "sum"),
                                  ins_buyers=("buyer", "nunique"), ins_tx=("value", "size")).reset_index()
        g["date"] = pd.Timestamp(d)
        res["insider"] = g
    act = idx[idx["form"].str.contains(r"13D|13G", regex=True)]
    if not act.empty:
        a = act.assign(d13=act["form"].str.contains("13D")).groupby("cik").agg(
            n_13d=("d13", "sum"), n_13g=("d13", lambda s: (~s).sum())).reset_index()
        a["date"] = pd.Timestamp(d)
        res["activist"] = a      # keyed by filer/subject CIK; mapped to tickers in factors
    return res


# ---------------------------------------------------------------- FTD & 13F
def fetch_ftd(year: int, month: int, half: str) -> pd.DataFrame:
    try:
        b = http_bytes(f"{SEC}/files/data/fails-deliver-data/cnsfails{year}{month:02d}{half}.zip", retries=2)
    except Exception:
        return pd.DataFrame()
    z = zipfile.ZipFile(io.BytesIO(b))
    df = pd.read_csv(z.open(z.namelist()[0]), sep="|", dtype=str, encoding="latin-1", on_bad_lines="skip")
    df.columns = [c.strip().upper() for c in df.columns]
    df = df.rename(columns={"SETTLEMENT DATE": "date", "SYMBOL": "code", "QUANTITY (FAILS)": "fails", "CUSIP": "cusip"})
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df["fails"] = pd.to_numeric(df["fails"], errors="coerce")
    return df.dropna(subset=["date"])[["date", "code", "cusip", "fails"]]


def fetch_13f(year: int, quarter: int) -> pd.DataFrame:
    """SEC Form 13F data set -> shares held per CUSIP + number of managers, for one report quarter.

    SEC packages 13F filings by filing window: Q1 reports are filed Mar-May, Q2 Jun-Aug,
    Q3 Sep-Nov, Q4 Dec-Feb of the next year.
    """
    start = {1: f"01mar{year}", 2: f"01jun{year}", 3: f"01sep{year}", 4: f"01dec{year}"}[quarter]
    page = http_get(f"{SEC}/data-research/sec-markets-data/form-13f-data-sets").text
    links = re.findall(r'href="([^"]+_form13f\.zip)"', page, re.I)
    want = [l for l in links if start in l.lower()]
    if not want:
        log.warning("13F data set for %dQ%d (window %s) not published yet", year, quarter, start)
        return pd.DataFrame()
    url = want[0] if want[0].startswith("http") else SEC + want[0]
    z = zipfile.ZipFile(io.BytesIO(http_bytes(url, timeout=900)))
    pick = lambda key: next(n for n in z.namelist() if key in n.upper())
    qend = (pd.Timestamp(year, 3 * quarter, 1) + pd.offsets.MonthEnd(0))
    sub = pd.read_csv(z.open(pick("SUBMISSION")), sep="\t", dtype=str,
                      usecols=lambda c: c.upper() in ("ACCESSION_NUMBER", "PERIODOFREPORT"))
    sub.columns = [c.upper() for c in sub.columns]
    sub = sub[pd.to_datetime(sub["PERIODOFREPORT"], format="%d-%b-%Y", errors="coerce") == qend]
    it = pd.read_csv(z.open(pick("INFOTABLE")), sep="\t", dtype=str,
                     usecols=lambda c: c.upper() in ("ACCESSION_NUMBER", "CUSIP", "SSHPRNAMT", "SSHPRNAMTTYPE", "PUTCALL"))
    it.columns = [c.upper() for c in it.columns]
    it = it[it["ACCESSION_NUMBER"].isin(sub["ACCESSION_NUMBER"])]
    it = it[(it["SSHPRNAMTTYPE"] == "SH") & it["PUTCALL"].isna()]
    it["SSHPRNAMT"] = pd.to_numeric(it["SSHPRNAMT"], errors="coerce")
    g = it.groupby("CUSIP").agg(inst_shares=("SSHPRNAMT", "sum"), n_mgr=("ACCESSION_NUMBER", "nunique")).reset_index()
    g["date"] = qend
    log.info("13F %dQ%d: %d filings, %d securities", year, quarter, sub["ACCESSION_NUMBER"].nunique(), len(g))
    return g.rename(columns={"CUSIP": "cusip"})


# ---------------------------------------------------------------- orchestration
def update_day(store: Store, d: dt.date) -> dict:
    fl = store.load("_fetchlog")
    seen = set(zip(pd.to_datetime(fl["date"]).dt.date, fl["code"])) if not fl.empty else set()
    res, logrows = {}, []
    if (d, "shortvol") not in seen:
        sv = fetch_shortvol(d)
        store.append("shortvol", sv)
        res["shortvol"] = len(sv)
        if len(sv) or d < dt.date.today() - dt.timedelta(days=4):
            logrows.append(dict(date=pd.Timestamp(d), code="shortvol", rows=len(sv)))
    if (d, "filings") not in seen:
        fl_ = fetch_filings(d)
        store.append("insider", fl_["insider"])
        store.append("activist", fl_["activist"], keys=("date", "cik"))
        res["insider"] = len(fl_["insider"])
        if len(fl_["insider"]) or d < dt.date.today() - dt.timedelta(days=4):
            logrows.append(dict(date=pd.Timestamp(d), code="filings", rows=len(fl_["insider"])))
    store.append("_fetchlog", pd.DataFrame(logrows))
    log.info("US %s -> %s", d, res)
    return res


def update_periodic(store: Store, d: dt.date, lookback_days: int = 45) -> None:
    for name, fn in (("universe/shares", _periodic_universe), ("short interest", _periodic_si),
                     ("fails-to-deliver", _periodic_ftd), ("13F", _periodic_13f)):
        try:
            fn(store, d, lookback_days)
        except Exception:
            log.exception("US periodic %s failed", name)


def _periodic_universe(store: Store, d: dt.date, lookback_days: int) -> None:
    uni = fetch_universe()
    if not uni.empty:
        uni["date"] = pd.Timestamp(d)
        store.append("universe", uni)
    sh = fetch_shares()
    if not sh.empty:
        sh["date"] = pd.Timestamp(d)
        store.append("shares", sh, keys=("date", "cik"))
    have = store.dates("funds")
    if not have or (d - max(have)).days >= 28:          # refresh the fund list monthly
        try:
            store.append("funds", fetch_funds(d))
        except Exception:
            log.exception("fund list failed")


def _periodic_si(store: Store, d: dt.date, lookback_days: int) -> None:
    have_si = store.dates("si")
    for s in si_settlement_dates(d - dt.timedelta(days=lookback_days), d):
        if s not in have_si:
            store.append("si", fetch_si(s))


def _periodic_ftd(store: Store, d: dt.date, lookback_days: int) -> None:
    have_ftd = {(x.year, x.month) for x in store.dates("ftd")}
    for k in range(0, 3):
        m = (pd.Timestamp(d) - pd.DateOffset(months=k))
        if (m.year, m.month) in have_ftd and k > 1:
            continue
        for half in ("a", "b"):
            store.append("ftd", fetch_ftd(m.year, m.month, half))


def _periodic_13f(store: Store, d: dt.date, lookback_days: int) -> None:
    have13 = store.dates("inst13f")
    for k in range(1, 3):
        qd = pd.Timestamp(d) - pd.DateOffset(months=3 * k)
        qend = (qd + pd.offsets.QuarterEnd(0)).date()
        if qend not in have13:
            try:
                store.append("inst13f", fetch_13f(qend.year, (qend.month - 1) // 3 + 1), keys=("date", "cusip"))
            except Exception as e:
                log.warning("13F %s: %s", qend, e)


def daily(store: Store, d: dt.date) -> dict:
    res = {}
    for back in range(6, -1, -1):
        dd = d - dt.timedelta(days=back)
        if dd.weekday() < 5:
            res[str(dd)] = update_day(store, dd)
    update_periodic(store, d)
    return res


def fetch_insider_bulk(year: int, q: int) -> pd.DataFrame:
    """SEC quarterly Form 3/4/5 data set -> same shape as daily `insider` rows (open-market P/S only)."""
    url = f"{SEC}/files/structureddata/data/insider-transactions-data-sets/{year}q{q}_form345.zip"
    z = zipfile.ZipFile(io.BytesIO(http_bytes(url, timeout=600)))
    rd = lambda n, cols: pd.read_csv(z.open(next(x for x in z.namelist() if x.upper().endswith(n))), sep="\t",
                                     dtype=str, usecols=lambda c: c.upper() in cols, on_bad_lines="skip")
    sub = rd("SUBMISSION.TSV", {"ACCESSION_NUMBER", "ISSUERTRADINGSYMBOL", "DOCUMENT_TYPE"})
    tr = rd("NONDERIV_TRANS.TSV", {"ACCESSION_NUMBER", "TRANS_CODE", "TRANS_SHARES", "TRANS_PRICEPERSHARE", "TRANS_DATE"})
    own = rd("REPORTINGOWNER.TSV", {"ACCESSION_NUMBER", "RPTOWNERCIK"})
    for df in (sub, tr, own):
        df.columns = [c.upper() for c in df.columns]
    t = tr[tr["TRANS_CODE"].isin(["P", "S"])].merge(sub, on="ACCESSION_NUMBER").merge(
        own.drop_duplicates("ACCESSION_NUMBER"), on="ACCESSION_NUMBER", how="left")
    t["code"] = t["ISSUERTRADINGSYMBOL"].str.strip().str.upper()
    t = t[t["code"].fillna("").str.match(TICKER_RE)]
    t["value"] = pd.to_numeric(t["TRANS_SHARES"], errors="coerce") * pd.to_numeric(t["TRANS_PRICEPERSHARE"], errors="coerce")
    t["date"] = pd.to_datetime(t["TRANS_DATE"], format="%d-%b-%Y", errors="coerce")
    t = t.dropna(subset=["date", "value"])
    t["buy"] = t["TRANS_CODE"] == "P"
    t["buy_value"] = t["value"].where(t["buy"], 0)
    t["sell_value"] = t["value"].where(~t["buy"], 0)
    t["buyer"] = t["RPTOWNERCIK"].where(t["buy"])
    g = t.groupby(["date", "code"]).agg(ins_buy_value=("buy_value", "sum"), ins_sell_value=("sell_value", "sum"),
                                        ins_buyers=("buyer", "nunique"), ins_tx=("value", "size")).reset_index()
    return g


def backfill_day(store: Store, d: dt.date) -> dict:
    """History: daily short volume per day; insiders from quarterly bulk sets (Form 4 one-by-one is too slow)."""
    res = {}
    if d not in store.dates("shortvol"):
        sv = fetch_shortvol(d)
        store.append("shortvol", sv)
        res["shortvol"] = len(sv)
    if d.month in (3, 6, 9, 12) and d.day <= 7 and d.weekday() == 0:
        # filing window that closed ~3 months ago holds the previous quarter's 13F reports
        qend = (pd.Timestamp(d) - pd.DateOffset(months=3) + pd.offsets.QuarterEnd(0)).date()
        if qend not in store.dates("inst13f"):
            try:
                store.append("inst13f", fetch_13f(qend.year, (qend.month - 1) // 3 + 1), keys=("date", "cusip"))
            except Exception as e:
                log.warning("13F %s: %s", qend, e)
    if d.month in (1, 4, 7, 10) and d.day <= 7 and d.weekday() == 0:
        pq = pd.Timestamp(d) - pd.DateOffset(months=3)
        try:
            store.append("insider", fetch_insider_bulk(pq.year, (pq.month - 1) // 3 + 1))
        except Exception as e:
            log.warning("insider bulk %s: %s", pq, e)
    if d.day in (15, 16, 17) or d.day >= 26:          # pick up SI / FTD around their release dates
        for s in si_settlement_dates(d - dt.timedelta(days=20), d):
            if s not in store.dates("si"):
                store.append("si", fetch_si(s))
    if d.day <= 7 and d.weekday() == 0:                 # previous month's fails-to-deliver files
        m = pd.Timestamp(d) - pd.DateOffset(months=1)
        if (m.year, m.month) not in {(x.year, x.month) for x in store.dates("ftd")}:
            for half in ("a", "b"):
                store.append("ftd", fetch_ftd(m.year, m.month, half))
    return res


def symbols(store: Store) -> dict[str, str]:
    uni = store.load("universe")
    if uni.empty:
        return {}
    u = uni.sort_values("date").drop_duplicates("code", keep="last")
    funds = fund_codes(store)
    return {c: c.replace(".", "-") for c in u["code"] if c not in funds}


# ================================================================ factors
def build_factors(store: Store, since: str = "2014-01-01"):
    from ..factors import FactorSet, chg, pct_chg, price_context, rsum, since_start, wide

    s = pd.Timestamp(since)
    price = store.load("price", s)
    sv = store.load("shortvol", s)
    si = store.load("si", s)
    ins = store.load("insider", s)
    act = store.load("activist", s)
    ftd = store.load("ftd", s)
    f13 = store.load("inst13f", s)
    uni = store.load("universe")
    shr = store.load("shares")

    cal = pd.DatetimeIndex(sorted(pd.to_datetime(price["date"].unique())))
    u = uni.sort_values("date").drop_duplicates("code", keep="last").set_index("code")
    funds = fund_codes(store)          # closed-end funds / ETFs are not stocks: keep them out entirely
    codes = pd.Index(sorted((set(price["code"].unique()) & set(u.index)) - funds), name="code")
    if funds:
        log.info("excluding %d fund/ETF tickers", len(funds & set(u.index)))
    W = lambda df, col, lim=0: wide(df, col, cal, codes, lim)

    px = W(price, "adj_close")
    vol = W(price, "volume")
    dollar_vol = W(price, "close") * vol

    # shares outstanding: SEC cover-page figure per CIK -> ticker
    shares = pd.DataFrame(index=cal, columns=codes, dtype="float32")
    if not shr.empty:
        last = shr.sort_values("end").drop_duplicates("cik", keep="last").set_index("cik")["shares"]
        sh_map = u["cik"].map(last)
        shares = pd.DataFrame(sh_map.reindex(codes).values[None, :].repeat(len(cal), 0), index=cal,
                              columns=codes).astype("float32")
    sh = shares.where(shares > 0)

    f: dict[str, pd.DataFrame] = {}
    # FINRA short volume ratio (off-exchange short selling share)
    shv, tot = W(sv, "short_vol"), W(sv, "total_vol")
    f["short_vol_ratio_20"] = rsum(shv, 20) / rsum(tot, 20).where(lambda x: x > 0) * 100
    f["short_vol_ratio_chg"] = f["short_vol_ratio_20"] - f["short_vol_ratio_20"].shift(20)
    # short interest
    if not si.empty:
        si_w = W(si, "si", 15)
        f["short_interest_pct"] = si_w / sh * 100
        f["short_interest_chg"] = pct_chg(W(si, "si", 15), 15)
        f["short_interest_dtc"] = si_w / (rsum(vol, 20) / 20).where(lambda x: x > 0)
    # fails to deliver
    if not ftd.empty:
        f["ftd_pct_20"] = rsum(W(ftd, "fails"), 20) / sh * 100
    # insiders (90-day window)
    if not ins.empty:
        bv = since_start(W(ins, "ins_buy_value").fillna(0), ins)
        sv_ = since_start(W(ins, "ins_sell_value").fillna(0), ins)
        mcap = W(price, "close") * sh
        f["insider_net_buy_90"] = (rsum(bv, 63, 0) - rsum(sv_, 63, 0)) / mcap.where(mcap > 0) * 1e4
        f["insider_buyers_90"] = rsum(since_start(W(ins, "ins_buyers").fillna(0), ins), 63, 0)
        f["insider_net_chg"] = f["insider_net_buy_90"] / 100  # in % of mcap, for signal flags
    # activist 13D (subject company CIK -> ticker)
    if not act.empty:
        cik2t = u.reset_index().drop_duplicates("cik").set_index("cik")["code"]
        a = act.assign(code=act["cik"].map(cik2t)).dropna(subset=["code"])
        f["activist_13d_90"] = rsum(since_start(W(a, "n_13d").fillna(0), act), 63, 0)
    # 13F institutional ownership (quarterly, CUSIP -> ticker via FTD map)
    if not f13.empty and not ftd.empty:
        cmap = ftd.dropna(subset=["cusip", "code"]).drop_duplicates("cusip", keep="last").set_index("cusip")["code"]
        q = f13.assign(code=f13["cusip"].map(cmap)).dropna(subset=["code"])
        q = q.groupby(["date", "code"]).agg(inst_shares=("inst_shares", "sum"), n_mgr=("n_mgr", "sum")).reset_index()
        q = q.sort_values("date")
        q["n_mgr_chg"] = q.groupby("code")["n_mgr"].pct_change(fill_method=None) * 100
        q["inst_sh_chg"] = q.groupby("code")["inst_shares"].pct_change(fill_method=None) * 100
        lag = 70  # 13F appears ~45 days after quarter end; ffill a full quarter
        f["inst_own_pct"] = W(q, "inst_shares", lag).shift(32) / sh * 100
        f["inst_own_chg_q"] = W(q, "inst_sh_chg", lag).shift(32)
        f["inst_holders_chg_q"] = W(q, "n_mgr_chg", lag).shift(32)
    f.update(price_context(px, vol, shares))

    liquid = (rsum(dollar_vol, 20) / 20 >= 2_000_000) & (W(price, "close") >= 3)
    meta = u.reindex(codes)[["name", "mkt"]].assign(industry="")
    ms = {}
    if not sv.empty:
        g = sv.groupby("date")
        ms["全市場放空量比%"] = g["short_vol"].sum() / g["total_vol"].sum() * 100
    if not ins.empty:
        g = ins.groupby("date")
        ms["內部人淨買(百萬美元)"] = (g["ins_buy_value"].sum() - g["ins_sell_value"].sum()) / 1e6

    spec = {
        "inst": [("inst_own_chg_q", 1), ("inst_holders_chg_q", 1)],
        "insider": [("insider_net_buy_90", 1), ("insider_buyers_90", 1)],
        "short": [("short_interest_pct", -1), ("short_interest_chg", -1), ("short_vol_ratio_20", -1),
                  ("short_vol_ratio_chg", -1), ("ftd_pct_20", -1)],
        "holders": [("activist_13d_90", 1)],
    }
    weights = {"inst": 0.35, "insider": 0.25, "short": 0.3, "holders": 0.10}
    series = ["short_interest_pct", "short_interest_dtc", "short_vol_ratio_20", "insider_net_buy_90",
              "inst_own_pct", "ftd_pct_20"]
    return FactorSet(MARKET, cal, codes, f, px, meta, spec, weights, LABELS, series, liquid, ms)


LABELS = {
    "short_vol_ratio_20": "場外放空量比%(20日)", "short_vol_ratio_chg": "放空量比20日變化(pp)",
    "short_interest_pct": "空單佔流通股%", "short_interest_chg": "空單半月變化%", "short_interest_dtc": "回補天數",
    "ftd_pct_20": "交割失敗佔股本%(20日)", "insider_net_buy_90": "內部人90日淨買(市值bp)",
    "insider_buyers_90": "內部人90日買進人次", "insider_net_chg": "內部人淨買(市值%)",
    "activist_13d_90": "90日內13D申報數", "inst_own_pct": "13F機構持股%", "inst_own_chg_q": "13F機構持股季變化%",
    "inst_holders_chg_q": "13F持有機構數季變化%", "ret_20": "20日漲跌%", "ret_60": "60日漲跌%",
    "ret_250": "一年漲跌%", "vol_60": "年化波動%", "dist_52w_high": "距52週高點%", "turnover_20": "日均週轉率%",
}
KEY_FACTORS = ["inst_own_pct", "inst_own_chg_q", "inst_holders_chg_q", "insider_net_buy_90", "insider_buyers_90",
               "short_interest_pct", "short_interest_chg", "short_interest_dtc", "short_vol_ratio_20",
               "ftd_pct_20", "activist_13d_90", "ret_20", "ret_60", "ret_250"]
FRESHNESS_DATASETS = ["price", "shortvol", "si", "insider", "activist", "ftd", "inst13f"]


SOURCES = {
    "price": ("Yahoo Finance（美股無免費的官方全市場收盤資料）", False),
    "shortvol": ("FINRA Reg SHO 每日放空量", True),
    "si": ("FINRA 空單餘額（半月）", True),
    "insider": ("SEC EDGAR Form 4／內部人交易資料集", True),
    "activist": ("SEC EDGAR 13D/13G 申報索引", True),
    "ftd": ("SEC 交割失敗資料", True),
    "inst13f": ("SEC Form 13F 資料集", True),
    "shares": ("SEC XBRL 流通股數", True),
    "funds": ("SEC EDGAR 基金申報索引（排除封閉式基金、ETF）＋Nasdaq Trader ETF 標記", True),
}
