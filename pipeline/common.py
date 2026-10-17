"""Shared helpers: polite HTTP, number parsing, date utilities, logging."""
from __future__ import annotations

import datetime as dt
import io
import json
import logging
import os
import random
import time
from typing import Any

import requests

log = logging.getLogger("chip-radar")
_handlers = [logging.StreamHandler()]
if os.environ.get("LOG_FILE"):  # CI writes logs into dist/_logs so they can be inspected later
    os.makedirs(os.path.dirname(os.environ["LOG_FILE"]) or ".", exist_ok=True)
    _handlers.append(logging.FileHandler(os.environ["LOG_FILE"], encoding="utf-8"))
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(message)s", handlers=_handlers)
_seen_fields: set = set()


def log_fields_once(tag: str, fields) -> None:
    """Record an endpoint's column layout once per run (helps when a site changes its format)."""
    if tag not in _seen_fields:
        _seen_fields.add(tag)
        # names only: the code repo and its Actions logs are public, so never log data values
        names = [f[0] if isinstance(f, tuple) else f for f in fields]
        log.info("FIELDS %s: %s", tag, [str(n) for n in names])

UA = os.environ.get(
    "HTTP_USER_AGENT",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130 Safari/537.36",
)
# SEC requires a contact in the UA; set SEC_CONTACT secret in GitHub Actions.
SEC_CONTACT = os.environ.get("SEC_CONTACT", "").strip()
SEC_UA = f"chip-radar research {SEC_CONTACT}"

_session = requests.Session()
_last_call: dict[str, float] = {}

# Minimum seconds between calls per host (TWSE bans IPs that exceed ~3 req / 5 s).
HOST_GAP = {"www.taifex.com.tw": 5.0, 
    "www.twse.com.tw": 3.2,
    "www.tpex.org.tw": 2.0,
    "openapi.twse.com.tw": 1.0,
    "opendata.tdcc.com.tw": 1.0,
    "www.sec.gov": 0.15,
    "data.sec.gov": 0.15,
    "www.jpx.co.jp": 1.0,
}


def _throttle(url: str) -> None:
    host = url.split("/")[2]
    gap = HOST_GAP.get(host, 0.5)
    wait = _last_call.get(host, 0) + gap - time.time()
    if wait > 0:
        time.sleep(wait)
    _last_call[host] = time.time()


def http_get(url: str, params: dict | None = None, *, headers: dict | None = None,
             retries: int = 4, timeout: int = 60) -> requests.Response:
    if "sec.gov" in url and not SEC_CONTACT:
        raise RuntimeError("SEC_CONTACT secret is not set; SEC data skipped")
    h = {"User-Agent": SEC_UA if "sec.gov" in url else UA}
    if headers:
        h.update(headers)
    last_err: Exception | None = None
    for attempt in range(retries):
        _throttle(url)
        try:
            r = _session.get(url, params=params, headers=h, timeout=timeout)
            if r.status_code == 200:
                return r
            last_err = RuntimeError(f"HTTP {r.status_code} for {r.url}")
            if r.status_code in (403, 429):
                time.sleep(20 * (attempt + 1))
        except requests.RequestException as e:  # network hiccup
            last_err = e
        time.sleep(2 ** attempt + random.random())
    raise RuntimeError(f"GET failed after {retries} tries: {url} {params} -> {last_err}")


def http_json(url: str, params: dict | None = None, **kw) -> Any:
    r = http_get(url, params, **kw)
    txt = r.text.strip()
    if not txt:
        return None
    return json.loads(txt)


def http_bytes(url: str, params: dict | None = None, **kw) -> bytes:
    return http_get(url, params, **kw).content


def num(x: Any) -> float | None:
    """Parse '1,234', '-5.6', '--', '', 'X' (TWSE no-trade marks) -> float or None."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).replace(",", "").replace("%", "").strip()
    if s in ("", "--", "---", "-", "X", "N/A", "除權息", "除權", "除息"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def ymd(d: dt.date) -> str:
    return d.strftime("%Y%m%d")


def parse_date(s: str) -> dt.date:
    s = s.replace("/", "-")
    if len(s) == 8 and s.isdigit():
        return dt.date(int(s[:4]), int(s[4:6]), int(s[6:]))
    return dt.date.fromisoformat(s)


def roc_to_date(s: str) -> dt.date:
    """'115/10/14' (ROC calendar) -> date."""
    y, m, d = s.strip().split("/")
    return dt.date(int(y) + 1911, int(m), int(d))


def weekdays(start: dt.date, end: dt.date):
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += dt.timedelta(days=1)


def today_in(tz_hours: int) -> dt.date:
    return (dt.datetime.utcnow() + dt.timedelta(hours=tz_hours)).date()


def read_csv_bytes(b: bytes, **kw):
    import pandas as pd
    for enc in ("utf-8-sig", "cp950", "big5", "shift_jis", "utf-8"):
        try:
            return pd.read_csv(io.BytesIO(b), encoding=enc, **kw)
        except (UnicodeDecodeError, LookupError):
            continue
    raise ValueError("could not decode CSV")


# ---------------------------------------------------------------- data-quality checks recorded while fetching
CHECKS: list[dict] = []


def record_check(name: str, d, ok: bool, detail: str = "", expected=None, actual=None) -> None:
    """Cross-source / consistency check made at fetch time; stored in dataset `_checks`."""
    CHECKS.append(dict(date=str(d), code=name, ok=bool(ok), detail=str(detail)[:300],
                       expected=None if expected is None else float(expected),
                       actual=None if actual is None else float(actual)))
    if not ok:
        log.warning("CHECK FAILED %s %s: %s (expected %s, got %s)", name, d, detail, expected, actual)


def drain_checks():
    import pandas as pd
    if not CHECKS:
        return pd.DataFrame()
    df = pd.DataFrame(CHECKS)
    CHECKS.clear()
    # checks come from different sources: "2026-10-16", "20261008", Timestamps ...
    df["date"] = pd.to_datetime(df["date"].astype(str).str.replace("-", "", regex=False).str[:8],
                                format="%Y%m%d", errors="coerce")
    return df.dropna(subset=["date"])
