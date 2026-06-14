from __future__ import annotations

import html
import re

try:
    import markdown
except ImportError:  # pragma: no cover - exercised only when dependency is absent
    markdown = None

LIST_MARKER = re.compile(r"^\s{0,3}(?:[-+*]|\d+[.)])\s+")
ORDERED_PAREN_MARKER = re.compile(r"^(\s{0,3})(\d+)\)(\s+)")
FENCE_MARKER = re.compile(r"^\s{0,3}(```|~~~)")
FENCE_ONLY_MARKER = re.compile(r"^\s{1,3}(```|~~~)\s*$")
FRONTMATTER_DELIMITER = re.compile(r"^---\s*$")


def render_markdown(markdown_text: str, theme: str = "light") -> str:
    if markdown is None:
        body = f"<pre>{html.escape(markdown_text)}</pre>"
    else:
        body = markdown.markdown(
            normalize_markdown_for_render(markdown_text),
            extensions=["extra", "sane_lists", "toc"],
            output_format="html5",
        )
    colors = _theme_colors(theme)
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
html {{
  background: {colors["background"]};
}}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.55;
  margin: 24px;
  color: {colors["foreground"]};
  background: {colors["background"]};
}}
pre, code {{
  font-family: Consolas, "Courier New", monospace;
}}
pre {{
  background: {colors["code_background"]};
  padding: 12px;
  overflow: auto;
}}
blockquote {{
  border-left: 4px solid {colors["border"]};
  margin-left: 0;
  padding-left: 12px;
  color: {colors["muted"]};
}}
table {{
  border-collapse: collapse;
}}
td, th {{
  border: 1px solid {colors["border"]};
  padding: 4px 8px;
}}
a {{
  color: {colors["link"]};
}}
</style>
</head>
<body>
{body}
</body>
</html>"""


def _theme_colors(theme: str) -> dict[str, str]:
    if theme == "dark":
        return {
            "background": "#1f2328",
            "foreground": "#e6edf3",
            "muted": "#aab6c4",
            "code_background": "#2d333b",
            "border": "#56616f",
            "link": "#7bb7ff",
        }
    return {
        "background": "#ffffff",
        "foreground": "#202124",
        "muted": "#4f5b67",
        "code_background": "#f4f6f8",
        "border": "#b8c0cc",
        "link": "#1558d6",
    }


def normalize_markdown_for_render(markdown_text: str) -> str:
    markdown_text = strip_frontmatter(markdown_text)
    normalized: list[str] = []
    in_fence = False

    for line in markdown_text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        line_ending = line[len(content) :]
        if FENCE_MARKER.match(content):
            in_fence = not in_fence
            normalized.append(FENCE_ONLY_MARKER.sub(r"\1", content) + line_ending)
            continue
        if in_fence:
            normalized.append(line)
            continue

        content = ORDERED_PAREN_MARKER.sub(r"\1\2.\3", content)
        if _needs_blank_before_list(normalized, content):
            normalized.append(_line_break_for(line_ending))
        normalized.append(content + line_ending)

    return "".join(normalized)


def strip_frontmatter(markdown_text: str) -> str:
    lines = markdown_text.splitlines(keepends=True)
    if not lines:
        return markdown_text

    first_content = lines[0].rstrip("\r\n")
    if not FRONTMATTER_DELIMITER.match(first_content):
        return markdown_text

    for index, line in enumerate(lines[1:], start=1):
        content = line.rstrip("\r\n")
        if FRONTMATTER_DELIMITER.match(content):
            return "".join(lines[index + 1 :])

    return markdown_text


def normalize_ordered_list_markers(markdown_text: str) -> str:
    normalized: list[str] = []
    in_fence = False

    for line in markdown_text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        line_ending = line[len(content) :]
        if FENCE_MARKER.match(content):
            in_fence = not in_fence
            normalized.append(FENCE_ONLY_MARKER.sub(r"\1", content) + line_ending)
            continue
        if in_fence:
            normalized.append(line)
            continue
        normalized.append(ORDERED_PAREN_MARKER.sub(r"\1\2.\3", content) + line_ending)

    return "".join(normalized)


def _needs_blank_before_list(normalized: list[str], content: str) -> bool:
    if not LIST_MARKER.match(content) or not normalized:
        return False
    previous = normalized[-1].rstrip("\r\n")
    return bool(previous.strip()) and not LIST_MARKER.match(previous)


def _line_break_for(line_ending: str) -> str:
    return "\r\n" if line_ending == "\r\n" else "\n"
