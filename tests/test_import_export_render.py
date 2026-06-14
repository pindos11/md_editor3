from __future__ import annotations

from md_editor.db import Repository
from md_editor.exporter import export_tree
from md_editor.importer import import_folder
from md_editor.markdown_render import (
    normalize_markdown_for_render,
    normalize_ordered_list_markers,
    render_markdown,
    strip_frontmatter,
)


def test_folder_import_preserves_folder_tree_as_empty_nodes(tmp_path):
    source = tmp_path / "source"
    child_dir = source / "chapter"
    section_dir = child_dir / "section"
    section_dir.mkdir(parents=True)
    (source / "intro.md").write_text("# Intro\n\nHello", encoding="utf-8")
    (source / "notes.txt").write_text("ignore me", encoding="utf-8")
    (child_dir / "details.markdown").write_text("Details", encoding="utf-8")
    (section_dir / "deep.md").write_text("Deep", encoding="utf-8")

    repo = Repository(tmp_path / "library.sqlite3")
    imported = import_folder(repo, source)

    assert len(imported) == 6
    root = repo.list_children(None)[0]
    assert root.title == "source"
    assert root.markdown_content == ""

    root_children = repo.list_children(root.id)
    assert [node.title for node in root_children] == ["chapter", "intro"]
    chapter = root_children[0]
    intro = root_children[1]
    assert intro.markdown_content == "# Intro\n\nHello"

    chapter_children = repo.list_children(chapter.id)
    assert [node.title for node in chapter_children] == ["section", "details"]
    section = chapter_children[0]
    assert repo.list_children(section.id)[0].title == "deep"


def test_export_tree_writes_markdown_files_using_tree_shape(tmp_path):
    repo = Repository(tmp_path / "library.sqlite3")
    root = repo.create_node(None, "Root Node", "# Root")
    repo.create_node(root.id, "Child Node", "Child")

    written = export_tree(repo, tmp_path / "export")

    relative = sorted(path.relative_to(tmp_path / "export").as_posix() for path in written)
    assert relative == ["Root Node.md", "Root Node/Child Node.md"]
    assert (tmp_path / "export" / "Root Node.md").read_text(encoding="utf-8") == "# Root"


def test_markdown_rendering_outputs_expected_html():
    html = render_markdown("# Heading\n\n- one\n- two\n\n`code`")

    assert "<h1" in html
    assert "Heading" in html
    assert "<li>one</li>" in html
    assert "<code>code</code>" in html


def test_markdown_rendering_hides_frontmatter():
    html = render_markdown("---\ntitle: Hidden\n---\n# Visible\n")

    assert "title: Hidden" not in html
    assert "<h1" in html
    assert "Visible" in html


def test_frontmatter_stripping_only_applies_at_document_start():
    markdown = "# Heading\n\n---\n\nBody"

    assert strip_frontmatter(markdown) == markdown


def test_markdown_rendering_accepts_parenthesized_ordered_lists():
    html = render_markdown("1) first\n2) second")

    assert "<ol>" in html
    assert "<li>first</li>" in html
    assert "<li>second</li>" in html


def test_markdown_rendering_accepts_lists_without_leading_blank_line():
    html = render_markdown("Intro text\n- first\n- second")

    assert "<p>Intro text</p>" in html
    assert "<ul>" in html
    assert "<li>first</li>" in html
    assert "<li>second</li>" in html


def test_markdown_rendering_accepts_parenthesized_lists_without_leading_blank_line():
    html = render_markdown("Intro text\n1) first\n2) second")

    assert "<p>Intro text</p>" in html
    assert "<ol>" in html
    assert "<li>first</li>" in html
    assert "<li>second</li>" in html


def test_list_spacing_normalization_ignores_fenced_code():
    normalized = normalize_markdown_for_render("Intro\n\n```\nText\n- not a list\n```\n")

    assert normalized == "Intro\n\n```\nText\n- not a list\n```\n"


def test_parenthesized_list_normalization_ignores_fenced_code():
    normalized = normalize_ordered_list_markers("1) first\n\n```\n1) not a list\n```\n")

    assert normalized == "1. first\n\n```\n1) not a list\n```\n"


def test_markdown_rendering_accepts_lightly_indented_closing_fence():
    html = render_markdown("```mermaid\nflowchart LR\n  A --> B\n ```\n")

    assert "<pre>" in html
    assert "<code" in html
    assert "flowchart LR" in html
    assert "<p><code>mermaid" not in html


def test_markdown_rendering_supports_dark_theme():
    html = render_markdown("# Heading", theme="dark")

    assert "#1f2328" in html
    assert "#e6edf3" in html
