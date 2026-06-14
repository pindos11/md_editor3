from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .models import Node

SCHEMA_VERSION = 1


class _Unset:
    pass


_UNSET = _Unset()


class RepositoryError(RuntimeError):
    pass


class CycleError(RepositoryError):
    pass


class NodeNotFoundError(RepositoryError):
    pass


class Repository:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.migrate()

    def close(self) -> None:
        self.conn.close()

    def migrate(self) -> None:
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS app_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS nodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parent_id INTEGER REFERENCES nodes(id) ON DELETE CASCADE,
                    title TEXT NOT NULL CHECK(length(trim(title)) > 0),
                    markdown_content TEXT NOT NULL DEFAULT '',
                    sort_order INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_nodes_parent_sort
                ON nodes(parent_id, sort_order, title)
                """
            )
            self.conn.execute(
                """
                INSERT INTO app_meta(key, value)
                VALUES ('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(SCHEMA_VERSION),),
            )

    def create_node(
        self,
        parent_id: int | None,
        title: str,
        content: str = "",
        sort_order: int | None = None,
    ) -> Node:
        clean_title = self._clean_title(title)
        if parent_id is not None:
            self._require_node(parent_id)
        if sort_order is None:
            sort_order = self._next_sort_order(parent_id)
        now = _now()
        with self.conn:
            cur = self.conn.execute(
                """
                INSERT INTO nodes(parent_id, title, markdown_content, sort_order, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (parent_id, clean_title, content, sort_order, now, now),
            )
        return self.get_node(int(cur.lastrowid))

    def update_node(
        self,
        id: int,
        title: str | None = None,
        content: str | None = None,
        parent_id: int | None | object = _UNSET,
        sort_order: int | None = None,
    ) -> Node:
        self._require_node(id)
        assignments: list[str] = []
        values: list[object] = []
        if title is not None:
            assignments.append("title = ?")
            values.append(self._clean_title(title))
        if content is not None:
            assignments.append("markdown_content = ?")
            values.append(content)
        if parent_id is not _UNSET:
            new_parent = parent_id if isinstance(parent_id, int) else None
            self._validate_parent(id, new_parent)
            assignments.append("parent_id = ?")
            values.append(new_parent)
        if sort_order is not None:
            assignments.append("sort_order = ?")
            values.append(sort_order)
        if not assignments:
            return self.get_node(id)
        assignments.append("updated_at = ?")
        values.append(_now())
        values.append(id)
        with self.conn:
            self.conn.execute(
                f"UPDATE nodes SET {', '.join(assignments)} WHERE id = ?",
                values,
            )
        return self.get_node(id)

    def delete_node(self, id: int, cascade: bool = True) -> None:
        self._require_node(id)
        if not cascade and self.list_children(id):
            raise RepositoryError("Cannot delete node with children when cascade is false.")
        with self.conn:
            self.conn.execute("DELETE FROM nodes WHERE id = ?", (id,))

    def move_node(self, id: int, new_parent_id: int | None, new_sort_order: int | None = None) -> Node:
        self._validate_parent(id, new_parent_id)
        if new_sort_order is None:
            new_sort_order = self._next_sort_order(new_parent_id)
        with self.conn:
            self.conn.execute(
                """
                UPDATE nodes
                SET parent_id = ?, sort_order = ?, updated_at = ?
                WHERE id = ?
                """,
                (new_parent_id, new_sort_order, _now(), id),
            )
        return self.get_node(id)

    def get_node(self, id: int) -> Node:
        row = self.conn.execute("SELECT * FROM nodes WHERE id = ?", (id,)).fetchone()
        if row is None:
            raise NodeNotFoundError(f"Node {id} does not exist.")
        return _row_to_node(row)

    def list_children(self, parent_id: int | None) -> list[Node]:
        if parent_id is not None:
            self._require_node(parent_id)
        if parent_id is None:
            rows = self.conn.execute(
                """
                SELECT * FROM nodes
                WHERE parent_id IS NULL
                ORDER BY sort_order, title COLLATE NOCASE, id
                """
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT * FROM nodes
                WHERE parent_id = ?
                ORDER BY sort_order, title COLLATE NOCASE, id
                """,
                (parent_id,),
            ).fetchall()
        return [_row_to_node(row) for row in rows]

    def iter_subtree(self, root_id: int | None = None) -> Iterable[Node]:
        for node in self.list_children(root_id):
            yield node
            yield from self.iter_subtree(node.id)

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM app_meta WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        return str(row["value"])

    def set_meta(self, key: str, value: str) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO app_meta(key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, value),
            )

    def _next_sort_order(self, parent_id: int | None) -> int:
        if parent_id is None:
            row = self.conn.execute(
                "SELECT COALESCE(MAX(sort_order), -1) + 1 AS next_order FROM nodes WHERE parent_id IS NULL"
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT COALESCE(MAX(sort_order), -1) + 1 AS next_order FROM nodes WHERE parent_id = ?",
                (parent_id,),
            ).fetchone()
        return int(row["next_order"])

    def _require_node(self, id: int) -> None:
        exists = self.conn.execute("SELECT 1 FROM nodes WHERE id = ?", (id,)).fetchone()
        if exists is None:
            raise NodeNotFoundError(f"Node {id} does not exist.")

    def _validate_parent(self, id: int, new_parent_id: int | None) -> None:
        self._require_node(id)
        if new_parent_id is None:
            return
        self._require_node(new_parent_id)
        if id == new_parent_id:
            raise CycleError("A node cannot be its own parent.")
        current = new_parent_id
        while current is not None:
            parent = self.get_node(current).parent_id
            if parent == id:
                raise CycleError("Moving this node would create a cycle.")
            current = parent

    @staticmethod
    def _clean_title(title: str) -> str:
        clean = title.strip()
        if not clean:
            raise ValueError("Node title cannot be blank.")
        return clean


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _row_to_node(row: sqlite3.Row) -> Node:
    return Node(
        id=int(row["id"]),
        parent_id=row["parent_id"],
        title=str(row["title"]),
        markdown_content=str(row["markdown_content"]),
        sort_order=int(row["sort_order"]),
        created_at=datetime.fromisoformat(str(row["created_at"])),
        updated_at=datetime.fromisoformat(str(row["updated_at"])),
    )
