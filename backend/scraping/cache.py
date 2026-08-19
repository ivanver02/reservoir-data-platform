"""JSON cache for service lookups"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class LookupCache:
    def __init__(self, root: Path, provider: str):
        """ Creates a file cache scoped to one external provider """
        self.root = root / provider
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, query: str) -> Path:
        """ Returns the cache path for a query """
        key = hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()
        return self.root / f"{key}.json"

    def get(self, query: str) -> Any | None:
        """ Reads a cached response or returns None when it is absent """
        path = self._path(query)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, query: str, value: Any) -> None:
        """ Stores a response in a JSON file """
        self._path(query).write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
