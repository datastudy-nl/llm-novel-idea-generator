"""Structured result returned to the calling agent."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import List


@dataclass
class NovelIdeaResult:
    """Structured, JSON-serialisable result returned to the calling agent."""

    term: str
    whitelist: List[str]
    blacklist: List[str]
    concepts: List[str]
    raw_generation: str = field(default="", repr=False)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kwargs)
