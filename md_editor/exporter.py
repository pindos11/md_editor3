from __future__ import annotations

import re
from pathlib import Path

from .db import Repository
from .models import Node


def export_tree(repo: Repository, output_folder: str | Path, root_id: int | None = None) -> list[Path]:
    output = Path(output_folder)
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    roots = [repo.get_node(root_id)] if root_id is not None else repo.list_children(None)
    for node in roots:
        written.extend(_export_node(repo, node, output))
    return written


def _export_node(repo: Repository, node: Node, folder: Path) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    file_path = _unique_path(folder / f"{_safe_name(node.title)}.md")
    file_path.write_text(node.markdown_content, encoding="utf-8")
    written = [file_path]

    children = repo.list_children(node.id)
    if children:
        child_folder = folder / _safe_name(node.title)
        for child in children:
            written.extend(_export_node(repo, child, child_folder))
    return written


def _safe_name(value: str) -> str:
    safe = re.sub('[<>:"/\\\\|?*\x00-\x1f]+', "_", value.strip())
    safe = safe.strip(" .")
    return safe or "Untitled"


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    index = 2
    while True:
        candidate = path.with_name(f"{stem} ({index}){suffix}")
        if not candidate.exists():
            return candidate
        index += 1
