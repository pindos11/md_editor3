from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Node:
    id: int
    parent_id: int | None
    title: str
    markdown_content: str
    sort_order: int
    created_at: datetime
    updated_at: datetime
