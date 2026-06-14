from __future__ import annotations

import pytest

from md_editor.db import CycleError, Repository, RepositoryError


def test_migration_creates_schema(tmp_path):
    repo = Repository(tmp_path / "library.sqlite3")
    row = repo.conn.execute("SELECT value FROM app_meta WHERE key = 'schema_version'").fetchone()
    assert row["value"] == "1"


def test_node_crud_preserves_child_order(tmp_path):
    repo = Repository(tmp_path / "library.sqlite3")
    root = repo.create_node(None, "Root")
    second = repo.create_node(root.id, "Second", sort_order=2)
    first = repo.create_node(root.id, "First", sort_order=1)

    assert [node.id for node in repo.list_children(root.id)] == [first.id, second.id]

    updated = repo.update_node(first.id, title="First changed", content="# Hi")
    assert updated.title == "First changed"
    assert updated.markdown_content == "# Hi"
    assert updated.updated_at >= updated.created_at


def test_move_rejects_cycles(tmp_path):
    repo = Repository(tmp_path / "library.sqlite3")
    root = repo.create_node(None, "Root")
    child = repo.create_node(root.id, "Child")
    grandchild = repo.create_node(child.id, "Grandchild")

    with pytest.raises(CycleError):
        repo.move_node(root.id, grandchild.id)


def test_delete_requires_cascade_when_children_exist(tmp_path):
    repo = Repository(tmp_path / "library.sqlite3")
    root = repo.create_node(None, "Root")
    repo.create_node(root.id, "Child")

    with pytest.raises(RepositoryError):
        repo.delete_node(root.id, cascade=False)

    repo.delete_node(root.id, cascade=True)
    assert repo.list_children(None) == []


def test_meta_values_are_persisted(tmp_path):
    path = tmp_path / "library.sqlite3"
    repo = Repository(path)
    assert repo.get_meta("ui.selected_node_id") is None
    repo.set_meta("ui.selected_node_id", "42")
    repo.close()

    reopened = Repository(path)
    assert reopened.get_meta("ui.selected_node_id") == "42"
