"""History store: one parquet per (dataset, year), kept as GitHub Release assets.

Keeping history in a Release (tag `store-<market>`) instead of git commits keeps the
repository small: re-uploading a year file replaces the asset without growing history.
Locally (no GITHUB_ACTIONS env) it simply works on ./data/<market>/.
"""
from __future__ import annotations

import datetime as dt
import glob
import os
import subprocess
from pathlib import Path

import pandas as pd

from .common import log

ROOT = Path(os.environ.get("CHIP_DATA_DIR", "data"))
IN_CI = os.environ.get("GITHUB_ACTIONS") == "true"


def _gh(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["gh", *args], capture_output=True, text=True, check=check)


class Store:
    def __init__(self, market: str):
        self.market = market
        self.tag = f"store-{market}"
        self.dir = ROOT / market
        self.dir.mkdir(parents=True, exist_ok=True)
        self.dirty: set[Path] = set()
        self.new_rows: dict[Path, list[pd.DataFrame]] = {}   # rows written by THIS job, per file
        self.replaced: set[Path] = set()                     # files rewritten wholesale (e.g. price rescale)
        self._cache: dict[str, pd.DataFrame] = {}

    # ---------- sync with GitHub Release ----------
    def pull(self) -> None:
        if not IN_CI:
            return
        r = _gh("release", "view", self.tag, check=False)
        if r.returncode != 0:
            log.info("release %s not found yet; starting empty", self.tag)
            return
        _gh("release", "download", self.tag, "-D", str(self.dir), "--clobber", "-p", "*.parquet")
        log.info("pulled %d files for %s", len(list(self.dir.glob("*.parquet"))), self.market)

    def push(self) -> None:
        if not IN_CI or not self.dirty:
            return
        if _gh("release", "view", self.tag, check=False).returncode != 0:
            _gh("release", "create", self.tag, "--title", f"Data store: {self.market}",
                "--notes", "Auto-managed history store. Do not edit by hand.", "--latest=false")
        # Another job (daily vs backfill) may have uploaded newer rows meanwhile. Start from the
        # remote copy and lay only the rows THIS job wrote on top, so stale rows pulled at the
        # start of a long job never overwrite fresher ones.
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            for p in sorted(self.dirty):
                if p in self.replaced:
                    continue
                r = _gh("release", "download", self.tag, "-p", p.name, "-D", tmp, "--clobber", check=False)
                remote = Path(tmp) / p.name
                mine = self.new_rows.get(p)
                if r.returncode == 0 and remote.exists() and mine:
                    try:
                        both = pd.concat([pd.read_parquet(remote), *mine], ignore_index=True)
                        keys = [k for k in ("date", "code", "cik", "cusip") if k in both.columns][:2]
                        both = both.drop_duplicates(keys, keep="last").sort_values(keys)
                        both.to_parquet(p, index=False, compression="zstd")
                    except Exception as e:
                        log.warning("merge with remote %s failed: %s", p.name, e)
        files = [str(p) for p in sorted(self.dirty)]
        for i in range(0, len(files), 20):
            _gh("release", "upload", self.tag, *files[i:i + 20], "--clobber")
        log.info("pushed %d files to %s", len(files), self.tag)
        self.dirty.clear()
        self.new_rows.clear()
        self.replaced.clear()
        self._cache.clear()

    # ---------- read / write ----------
    def _path(self, dataset: str, year: int) -> Path:
        return self.dir / f"{dataset}_{year}.parquet"

    def append(self, dataset: str, df: pd.DataFrame, keys=("date", "code")) -> None:
        if df is None or df.empty:
            return
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        for year, part in df.groupby(df["date"].dt.year):
            p = self._path(dataset, int(year))
            self.new_rows.setdefault(p, []).append(part.copy())
            if p.exists():
                old = pd.read_parquet(p)
                part = pd.concat([old, part], ignore_index=True)
            part = part.drop_duplicates(list(keys), keep="last").sort_values(list(keys))
            part.to_parquet(p, index=False, compression="zstd")
            self.dirty.add(p)
        self._cache.pop(dataset, None)

    def rewrite(self, dataset: str, df: pd.DataFrame) -> None:
        """Replace a dataset wholesale (used to purge bad rows); every year file is re-uploaded as-is."""
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        years = {int(Path(f).stem.rsplit("_", 1)[1]) for f in glob.glob(str(self.dir / f"{dataset}_*.parquet"))}
        for y in years | set(df["date"].dt.year.unique().tolist()):
            p = self._path(dataset, int(y))
            part = df[df["date"].dt.year == y]
            part.sort_values(["date", "code"]).to_parquet(p, index=False, compression="zstd")
            self.dirty.add(p)
            self.replaced.add(p)
        self._cache.pop(dataset, None)

    def load(self, dataset: str, since: dt.date | None = None) -> pd.DataFrame:
        if dataset not in self._cache:
            files = sorted(glob.glob(str(self.dir / f"{dataset}_*.parquet")))
            if not files:
                return pd.DataFrame()
            self._cache[dataset] = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df = self._cache[dataset]
        if since is not None and not df.empty:
            df = df[df["date"] >= pd.Timestamp(since)]
        return df

    def dates(self, dataset: str) -> set[dt.date]:
        df = self.load(dataset)
        if df.empty:
            return set()
        return set(pd.to_datetime(df["date"].unique()).date)

    # ---------- trained models (one pickle per market, replaced weekly) ----------
    def save_models(self, bundle) -> None:
        import pickle
        p = self.dir / "models.pkl"
        with open(p, "wb") as f:
            pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("saved models (%.1f MB)", p.stat().st_size / 1e6)
        if IN_CI:
            if _gh("release", "view", self.tag, check=False).returncode != 0:
                _gh("release", "create", self.tag, "--title", f"Data store: {self.market}",
                    "--notes", "Auto-managed history store. Do not edit by hand.", "--latest=false")
            _gh("release", "upload", self.tag, str(p), "--clobber")

    def load_models(self):
        import pickle
        p = self.dir / "models.pkl"
        if IN_CI and not p.exists():
            _gh("release", "download", self.tag, "-D", str(self.dir), "--clobber", "-p", "models.pkl", check=False)
        if not p.exists():
            return None
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            log.warning("could not load saved models: %s", e)
            return None

    def save_meta(self, name: str, obj) -> None:
        import json
        p = self.dir / f"{name}.json"
        p.write_text(json.dumps(obj, ensure_ascii=False, default=str, indent=1))
