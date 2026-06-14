from __future__ import annotations

from pathlib import Path

from .db import Repository

MARKDOWN_EXTENSIONS = {".md", ".markdown"}


def import_folder(repo: Repository, folder: str | Path) -> list[int]:
    root = Path(folder).resolve()
    if not root.is_dir():
        raise ValueError(f"Import path is not a folder: {root}")

    files = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in MARKDOWN_EXTENSIONS),
        key=lambda path: (len(path.relative_to(root).parts), path.relative_to(root).as_posix().lower()),
    )
    directories = _directories_with_markdown(root, files)
    imported_dirs: dict[Path, int] = {}
    created_ids: list[int] = []

    root_node = repo.create_node(None, root.name or str(root), "")
    imported_dirs[root] = root_node.id
    created_ids.append(root_node.id)

    for directory in directories:
        if directory == root:
            continue
        parent_id = imported_dirs[directory.parent]
        node = repo.create_node(parent_id, directory.name, "")
        imported_dirs[directory] = node.id
        created_ids.append(node.id)

    for path in files:
        parent_id = imported_dirs[path.parent]
        content = path.read_text(encoding="utf-8-sig")
        node = repo.create_node(parent_id, _title_for(path, content), content)
        created_ids.append(node.id)

    return created_ids


def _directories_with_markdown(root: Path, files: list[Path]) -> list[Path]:
    directories = {root}
    for path in files:
        directories.add(path.parent)
        directories.update(parent for parent in path.parent.parents if parent == root or root in parent.parents)
    return sorted(directories, key=lambda path: (len(path.relative_to(root).parts), path.as_posix().lower()))


def _title_for(path: Path, content: str) -> str:
    title = path.stem.strip()
    if title:
        return title
    for line in content.splitlines():
        clean = line.strip()
        if clean.startswith("#"):
            heading = clean.lstrip("#").strip()
            if heading:
                return heading
    return "Untitled"
