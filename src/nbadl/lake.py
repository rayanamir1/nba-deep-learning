"""Raw lake: append-only storage of untouched API responses.

Layout:  <RAW_DIR>/<source>/<key>.json.gz   plus   <RAW_DIR>/<source>/_manifest.jsonl

Every write records `fetched_at` (UTC). Raw files are never edited; if a source
changes, we fetch again under a new key (e.g. odds snapshots are keyed by time).
Everything downstream can be rebuilt from here.
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from . import config


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class RawRecord:
    source: str
    key: str
    fetched_at: datetime
    payload: Any


class RawLake:
    def __init__(self, root: Path | None = None):
        self.root = Path(root or config.RAW_DIR)

    def _path(self, source: str, key: str) -> Path:
        safe = key.replace("/", "_")
        return self.root / source / f"{safe}.json.gz"

    def exists(self, source: str, key: str) -> bool:
        return self._path(source, key).exists()

    def write(self, source: str, key: str, payload: Any, fetched_at: datetime | None = None) -> Path:
        """Write once. Raises if the key already exists, because raw data is immutable."""
        path = self._path(source, key)
        if path.exists():
            raise FileExistsError(f"raw record already exists: {source}/{key}")
        fetched_at = fetched_at or utcnow()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump({"fetched_at": fetched_at.isoformat(), "payload": payload}, f)
        tmp.rename(path)  # atomic on POSIX: a crash never leaves half a file
        with open(path.parent / "_manifest.jsonl", "a", encoding="utf-8") as m:
            m.write(json.dumps({"key": key, "fetched_at": fetched_at.isoformat()}) + "\n")
        return path

    def read(self, source: str, key: str) -> RawRecord:
        with gzip.open(self._path(source, key), "rt", encoding="utf-8") as f:
            blob = json.load(f)
        return RawRecord(source, key, datetime.fromisoformat(blob["fetched_at"]), blob["payload"])

    def keys(self, source: str) -> list[str]:
        d = self.root / source
        if not d.exists():
            return []
        return sorted(p.name[: -len(".json.gz")] for p in d.glob("*.json.gz"))

    def iter(self, source: str) -> Iterator[RawRecord]:
        for k in self.keys(source):
            yield self.read(source, k)
