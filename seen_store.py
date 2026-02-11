"""Persistence helpers for cross-run seen eBay item IDs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Set


class SeenItemStore:
    """Stores and loads item IDs that were already processed in prior runs."""

    def __init__(self, path: str = "data/seen_item_ids.json") -> None:
        self.path = Path(path)
        self._seen: Set[str] = set()

    def load(self) -> Set[str]:
        """Load seen IDs from disk. Returns an in-memory set."""
        if not self.path.exists():
            self._seen = set()
            return self._seen

        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._seen = set()
            return self._seen

        if isinstance(raw, list):
            self._seen = {str(x) for x in raw if x}
            return self._seen

        if isinstance(raw, dict):
            ids = raw.get("seen_item_ids", [])
            if isinstance(ids, list):
                self._seen = {str(x) for x in ids if x}
            else:
                self._seen = set()
            return self._seen

        self._seen = set()
        return self._seen

    @property
    def seen(self) -> Set[str]:
        return self._seen

    def contains(self, item_id: str) -> bool:
        return item_id in self._seen

    def add(self, item_id: str) -> None:
        if item_id:
            self._seen.add(item_id)

    def add_many(self, item_ids: Iterable[str]) -> None:
        for item_id in item_ids:
            self.add(item_id)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"seen_item_ids": sorted(self._seen)}
        self.path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
