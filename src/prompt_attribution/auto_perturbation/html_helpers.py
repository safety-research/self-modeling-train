"""
Module: prompt_attribution/auto_perturbation/html_helpers.py

Shared HTML helper functions for self-contained trace visualizations.
Used by both `pipeline_tracer.py` (pipeline stages 1-6) and
`discovery_tracer.py` (pre-pipeline discovery stages).

Structure:
- _esc: HTML-escape text
- _pre: Render text in a <pre> block
- _details: Render a collapsible <details> section
- _table: Render a 2-column key-value table
- _data_table: Render a multi-column data table with headers
- _badge: Render a colored inline badge/tag
- _render_diff: Render inline diff between two texts
- _render_word_diff: Render word-level diff for long lines
- HTML_TEMPLATE: Self-contained HTML page template with CSS
"""

import difflib
import html


# =============================================================================
# HTML Helpers
# =============================================================================

CATEGORY_COLORS = {
    "flip_inducing": "#dc3545",
    "non_flip": "#28a745",
    "boundary": "#fd7e14",
}


def _esc(text: str) -> str:
    """HTML-escape text."""
    return html.escape(str(text))


def _pre(text: str, max_len: int = 50000) -> str:
    """Render text in a <pre> block, truncated if needed."""
    if len(text) > max_len:
        text = text[:max_len] + f"\n\n... [truncated, {len(text)} chars total]"
    return f'<pre class="code-block">{_esc(text)}</pre>'


def _details(
    summary: str,
    content: str,
    css_class: str = "",
    open_default: bool = False,
) -> str:
    """Render a collapsible <details> section."""
    open_attr = " open" if open_default else ""
    cls = f' class="{css_class}"' if css_class else ""
    return f"<details{open_attr}{cls}><summary>{summary}</summary>{content}</details>"


def _table(rows: list[tuple[str, str]]) -> str:
    """Render a simple 2-column key-value table."""
    body = "".join(
        f"<tr><td><strong>{_esc(k)}</strong></td><td>{_esc(v)}</td></tr>"
        for k, v in rows
    )
    return f'<table class="kv-table">{body}</table>'


def _data_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a multi-column data table with headers.

    Args:
        headers: Column header strings
        rows: List of rows, each row is a list of cell strings
    """
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = ""
    for row in rows:
        cells = "".join(f"<td>{_esc(str(c))}</td>" for c in row)
        body += f"<tr>{cells}</tr>"
    return f'<table class="data-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def _badge(text: str, color: str = "#6c757d", bg: str = "") -> str:
    """Render an inline badge/tag.

    Args:
        text: Badge text
        color: Text color (CSS)
        bg: Background color (CSS). If empty, uses a light version of color.
    """
    if not bg:
        bg = color + "20"  # 20 = ~12% opacity hex
    return (
        f'<span style="display:inline-block; padding:2px 8px; border-radius:4px; '
        f'font-size:12px; font-weight:700; color:{color}; background:{bg}; '
        f'margin-left:6px;">{_esc(text)}</span>'
    )


def _render_diff(baseline: str, lever: str) -> str:
    """Render inline diff between baseline and lever prompts.

    Uses word-level diff within lines for readable problem edits.
    Long lines with small edits show the changed words highlighted
    instead of duplicating the entire line in red/green.
    """
    baseline_lines = baseline.splitlines()
    lever_lines = lever.splitlines()

    matcher = difflib.SequenceMatcher(None, baseline_lines, lever_lines)
    parts = []
    parts.append('<div class="diff-header">--- baseline</div>')
    parts.append('<div class="diff-header">+++ lever</div>')

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            # Show max 2 context lines around changes
            ctx_lines = baseline_lines[i1:i2]
            if len(ctx_lines) > 4:
                for line in ctx_lines[:2]:
                    parts.append(f'<div class="diff-ctx"> {_esc(line)}</div>')
                parts.append(f'<div class="diff-hunk">  ... ({len(ctx_lines) - 4} unchanged lines) ...</div>')
                for line in ctx_lines[-2:]:
                    parts.append(f'<div class="diff-ctx"> {_esc(line)}</div>')
            else:
                for line in ctx_lines:
                    parts.append(f'<div class="diff-ctx"> {_esc(line)}</div>')
        elif tag == "replace":
            # Word-level diff within replaced lines
            for bi, li in zip(
                range(i1, i2), range(j1, j2),
            ):
                b_line = baseline_lines[bi] if bi < len(baseline_lines) else ""
                l_line = lever_lines[li] if li < len(lever_lines) else ""
                # If lines are long, do word-level highlighting
                if len(b_line) > 100 or len(l_line) > 100:
                    parts.append(_render_word_diff(b_line, l_line))
                else:
                    parts.append(f'<div class="diff-del">-{_esc(b_line)}</div>')
                    parts.append(f'<div class="diff-add">+{_esc(l_line)}</div>')
            # Handle unequal line counts
            for bi in range(i1 + (j2 - j1), i2):
                parts.append(f'<div class="diff-del">-{_esc(baseline_lines[bi])}</div>')
            for li in range(j1 + (i2 - i1), j2):
                parts.append(f'<div class="diff-add">+{_esc(lever_lines[li])}</div>')
        elif tag == "delete":
            for line in baseline_lines[i1:i2]:
                parts.append(f'<div class="diff-del">-{_esc(line)}</div>')
        elif tag == "insert":
            for line in lever_lines[j1:j2]:
                parts.append(f'<div class="diff-add">+{_esc(line)}</div>')

    if len(parts) <= 2:  # Only headers
        return '<p class="meta">No differences found.</p>'

    return f'<div class="diff-block">{"".join(parts)}</div>'


def _render_word_diff(baseline_line: str, lever_line: str) -> str:
    """Render word-level diff for long lines, highlighting only changed words."""
    b_words = baseline_line.split()
    l_words = lever_line.split()
    matcher = difflib.SequenceMatcher(None, b_words, l_words)

    b_parts = []
    l_parts = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            text = " ".join(b_words[i1:i2])
            b_parts.append(_esc(text))
            l_parts.append(_esc(text))
        elif tag == "replace":
            b_text = " ".join(b_words[i1:i2])
            l_text = " ".join(l_words[j1:j2])
            b_parts.append(f'<span class="word-del">{_esc(b_text)}</span>')
            l_parts.append(f'<span class="word-add">{_esc(l_text)}</span>')
        elif tag == "delete":
            b_text = " ".join(b_words[i1:i2])
            b_parts.append(f'<span class="word-del">{_esc(b_text)}</span>')
        elif tag == "insert":
            l_text = " ".join(l_words[j1:j2])
            l_parts.append(f'<span class="word-add">{_esc(l_text)}</span>')

    return (
        f'<div class="diff-del">-{" ".join(b_parts)}</div>'
        f'<div class="diff-add">+{" ".join(l_parts)}</div>'
    )


# =============================================================================
# HTML Template
# =============================================================================

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    background: #f8f9fa;
    color: #212529;
    line-height: 1.5;
  }}
  h1 {{
    border-bottom: 2px solid #dee2e6;
    padding-bottom: 10px;
    color: #343a40;
  }}
  h2 {{
    color: #495057;
    margin-top: 24px;
  }}
  details {{
    margin: 8px 0;
    border: 1px solid #dee2e6;
    border-radius: 6px;
    background: white;
    overflow: hidden;
  }}
  summary {{
    padding: 10px 14px;
    cursor: pointer;
    font-weight: 600;
    background: #f1f3f5;
    border-bottom: 1px solid #dee2e6;
  }}
  summary:hover {{
    background: #e9ecef;
  }}
  details > :not(summary) {{
    padding: 0 14px;
  }}
  a {{
    color: #2563eb;
    text-decoration: none;
  }}
  a:hover {{
    text-decoration: underline;
  }}
  .example-section {{
    border: 2px solid #495057;
    margin: 16px 0;
  }}
  .example-section > summary {{
    background: #343a40;
    color: white;
    font-size: 1.1em;
  }}
  .example-section > summary a {{
    color: #93c5fd;
  }}
  .stage-section {{
    border-color: #6c757d;
    margin: 10px 0;
  }}
  .stage-section > summary {{
    background: #e9ecef;
    font-size: 1.05em;
  }}
  .category-section > summary {{
    background: #f8f9fa;
  }}
  .prompt-section > summary {{
    color: #0d6efd;
  }}
  .response-section > summary {{
    color: #6f42c1;
  }}
  .diff-section > summary {{
    color: #d63384;
    font-weight: 700;
  }}
  .code-block {{
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 4px;
    padding: 12px;
    overflow-x: auto;
    white-space: pre-wrap;
    word-wrap: break-word;
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 13px;
    line-height: 1.4;
    max-height: 600px;
    overflow-y: auto;
  }}
  .config-table, .kv-table, .data-table {{
    border-collapse: collapse;
    margin: 8px 0;
    width: 100%;
  }}
  .config-table td, .kv-table td, .data-table td, .data-table th {{
    border: 1px solid #dee2e6;
    padding: 6px 10px;
    font-size: 14px;
  }}
  .data-table th {{
    background: #e9ecef;
    text-align: left;
  }}
  .kv-table td:first-child {{
    width: 200px;
    white-space: nowrap;
  }}
  .meta {{
    color: #6c757d;
    font-size: 13px;
    margin: 4px 0;
  }}
  .flip-result {{
    font-size: 16px;
    margin: 8px 0;
    padding: 6px 12px;
    border-radius: 4px;
    background: #f8f9fa;
    display: inline-block;
  }}
  .candidate-list {{
    list-style-type: disc;
    padding-left: 20px;
  }}
  .candidate-list li {{
    margin: 4px 0;
    font-size: 14px;
  }}
  /* Diff styling */
  .diff-block {{
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 13px;
    line-height: 1.4;
    border: 1px solid #dee2e6;
    border-radius: 4px;
    overflow-x: auto;
    margin: 8px 0;
  }}
  .diff-header {{
    background: #e9ecef;
    padding: 2px 8px;
    color: #495057;
    font-weight: bold;
  }}
  .diff-hunk {{
    background: #dbeafe;
    padding: 2px 8px;
    color: #1e40af;
  }}
  .diff-del {{
    background: #fecaca;
    padding: 2px 8px;
    color: #991b1b;
  }}
  .diff-add {{
    background: #bbf7d0;
    padding: 2px 8px;
    color: #166534;
  }}
  .diff-ctx {{
    padding: 2px 8px;
    color: #6b7280;
  }}
  .word-del {{
    background: #fca5a5;
    padding: 1px 3px;
    border-radius: 2px;
    font-weight: bold;
  }}
  .word-add {{
    background: #86efac;
    padding: 1px 3px;
    border-radius: 2px;
    font-weight: bold;
  }}
  /* Discovery-specific styles */
  .check-pass {{
    color: #166534;
  }}
  .check-fail {{
    color: #991b1b;
  }}
  .summary-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
    gap: 12px;
    margin: 12px 0;
  }}
  .summary-card {{
    border: 1px solid #dee2e6;
    border-radius: 6px;
    padding: 12px;
    background: white;
  }}
  .summary-card h3 {{
    margin: 0 0 8px 0;
    font-size: 14px;
    color: #495057;
  }}
  .summary-card .big-number {{
    font-size: 28px;
    font-weight: 700;
    color: #212529;
  }}
  .toolbar {{
    position: sticky;
    top: 0;
    z-index: 100;
    background: #fff;
    border-bottom: 1px solid #dee2e6;
    padding: 8px 0;
    margin-bottom: 12px;
    display: flex;
    gap: 8px;
    align-items: center;
  }}
  .toolbar button {{
    padding: 5px 14px;
    border: 1px solid #dee2e6;
    border-radius: 4px;
    background: #f8f9fa;
    cursor: pointer;
    font-size: 13px;
    font-weight: 600;
    color: #495057;
  }}
  .toolbar button:hover {{
    background: #e9ecef;
  }}
  {extra_css}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="toolbar">
  <button onclick="document.querySelectorAll('details').forEach(d => d.open = true)">Expand All</button>
  <button onclick="document.querySelectorAll('details').forEach(d => d.open = false)">Collapse All</button>
  <button onclick="document.querySelectorAll('details.example-section').forEach(d => d.open = !d.open)">Toggle Top-Level</button>
</div>
{config_section}
{body}
</body>
</html>"""
