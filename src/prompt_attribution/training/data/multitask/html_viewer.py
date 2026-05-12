"""
Module: prompt_attribution/training/data/multitask/html_viewer.py

Interactive HTML debug viewer for multi-task training data.
Generates a single HTML file with tabs per task, collapsible cards
per example, filtering, and summary stats.

Structure:
- generate_html: Main entry point
- _render_stats: Per-task statistics
- _render_card: Per-example card
"""

import html
from collections import Counter
from pathlib import Path

from prompt_attribution.training.data.multitask.schema import MultitaskRecord


def generate_html(
    records_by_task: dict[str, list[MultitaskRecord]],
    output_path: Path,
    corpus_dir: str = "",
    max_per_task: int = 200,
) -> None:
    """Generate interactive HTML viewer for multi-task training data.

    Args:
        records_by_task: Dict of task_type -> list of records
        output_path: Where to write the HTML file
        corpus_dir: Source corpus path (for display)
        max_per_task: Max examples to show per task (for file size)
    """
    tabs_html = []
    panels_html = []

    for task_type in sorted(records_by_task.keys()):
        records = records_by_task[task_type]
        tab_label = task_type.replace("_", " ").upper()
        tab_id = task_type.replace(" ", "_")

        tabs_html.append(
            f'<button class="tab-btn" onclick="showTab(\'{tab_id}\')" '
            f'id="tab-{tab_id}">{tab_label} ({len(records)})</button>'
        )

        stats = _render_stats(records, task_type)
        cards = []
        display_records = _sample_across_datasets(records, max_per_task)
        for i, rec in enumerate(display_records):
            cards.append(_render_card(rec, i))

        truncation_note = ""
        if len(records) > max_per_task:
            truncation_note = (
                f'<p class="truncation">Showing {max_per_task} of '
                f'{len(records)} examples.</p>'
            )

        panel = f"""
        <div class="tab-panel" id="panel-{tab_id}" style="display:none">
            {stats}
            <div class="controls">
                <input type="text" id="search-{tab_id}" placeholder="Filter by text..."
                       oninput="filterCards('{tab_id}')">
                <select id="filter-gt-{tab_id}" onchange="filterCards('{tab_id}')">
                    <option value="">All GT labels</option>
                    {_gt_filter_options(records)}
                </select>
                <select id="filter-dataset-{tab_id}" onchange="filterCards('{tab_id}')">
                    <option value="">All datasets</option>
                    {_dataset_filter_options(records)}
                </select>
                <button onclick="toggleAll('{tab_id}', true)">Expand All</button>
                <button onclick="toggleAll('{tab_id}', false)">Collapse All</button>
            </div>
            {truncation_note}
            <div class="cards" id="cards-{tab_id}">
                {''.join(cards)}
            </div>
        </div>
        """
        panels_html.append(panel)

    full_html = _HTML_TEMPLATE.format(
        corpus_dir=html.escape(corpus_dir),
        total_records=sum(len(r) for r in records_by_task.values()),
        n_tasks=len(records_by_task),
        tabs=''.join(tabs_html),
        panels=''.join(panels_html),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(full_html)


def _sample_across_datasets(
    records: list[MultitaskRecord], max_total: int
) -> list[MultitaskRecord]:
    """Sample records evenly across datasets so the viewer shows diversity.

    Round-robins through datasets, picking one record from each in turn,
    until we hit max_total.
    """
    if len(records) <= max_total:
        return records

    # Group by dataset
    by_dataset: dict[str, list[MultitaskRecord]] = {}
    for rec in records:
        ds = rec.dataset_id or "unknown"
        by_dataset.setdefault(ds, []).append(rec)

    # Round-robin across datasets
    sampled: list[MultitaskRecord] = []
    iterators = {ds: iter(recs) for ds, recs in sorted(by_dataset.items())}
    while len(sampled) < max_total and iterators:
        exhausted = []
        for ds, it in iterators.items():
            if len(sampled) >= max_total:
                break
            try:
                sampled.append(next(it))
            except StopIteration:
                exhausted.append(ds)
        for ds in exhausted:
            del iterators[ds]

    return sampled


def _render_stats(records: list[MultitaskRecord], _task_type: str) -> str:
    """Render stats header for a task panel."""
    gt_type = records[0].gt_type if records else "?"

    if gt_type == "binary":
        dist = Counter(r.gt_label for r in records)
        dist_html = " | ".join(
            f'<span class="stat-label">{k}:</span> {v} ({v / len(records) * 100:.1f}%)'
            for k, v in dist.most_common()
        )
    elif gt_type == "mcq":
        dist = Counter(r.gt_label for r in records)
        dist_html = " | ".join(
            f'<span class="stat-label">{k}:</span> {v} ({v / len(records) * 100:.1f}%)'
            for k, v in sorted(dist.items())
        )
    elif gt_type == "continuous":
        values = [r.gt_value for r in records if r.gt_value is not None]
        if values:
            mean_v = sum(values) / len(values)
            dist_html = (
                f'Mean: {mean_v:.3f} | '
                f'0.0: {sum(1 for v in values if v == 0.0)} | '
                f'1.0: {sum(1 for v in values if v == 1.0)} | '
                f'Other: {sum(1 for v in values if 0.0 < v < 1.0)}'
            )
        else:
            dist_html = "No values"
    else:
        dist_html = f"Type: {gt_type}"

    dataset_dist = Counter(r.dataset_id for r in records)
    top_datasets = dataset_dist.most_common(5)
    ds_html = ", ".join(f"{ds}: {c}" for ds, c in top_datasets)
    if len(dataset_dist) > 5:
        ds_html += f", ... (+{len(dataset_dist) - 5} more)"

    variant_dist = Counter(r.template_variant for r in records)
    var_html = " | ".join(f"{v}: {c}" for v, c in variant_dist.most_common())

    return f"""
    <div class="stats">
        <div><strong>Total:</strong> {len(records)} examples</div>
        <div><strong>GT distribution:</strong> {dist_html}</div>
        <div><strong>Templates:</strong> {var_html}</div>
        <div><strong>Datasets:</strong> {ds_html}</div>
    </div>
    """


def _render_card(rec: MultitaskRecord, idx: int) -> str:
    """Render a single collapsible example card."""
    # GT badge color
    if rec.gt_type == "binary":
        badge_color = "#2e7d32" if rec.gt_label == "Yes" else "#c62828"
        badge_text = rec.gt_label
    elif rec.gt_type == "mcq":
        badge_color = "#1565c0"
        badge_text = rec.gt_label
    elif rec.gt_type == "continuous":
        v = rec.gt_value if rec.gt_value is not None else 0.0
        badge_color = f"hsl({120 * v}, 70%, 35%)"
        badge_text = f"{v:.2f}"
    elif rec.gt_type == "text":
        badge_color = "#6a1b9a"
        badge_text = "text"
    else:
        badge_color = "#555"
        badge_text = "?"

    # Flip fraction badge
    ff = rec.empirical_flip_fraction
    ff_text = f"{ff:.2f}" if ff is not None else "N/A"

    # E6-specific: show options with flip rates
    e6_detail = ""
    if rec.e6_options:
        rows = []
        for opt in rec.e6_options:
            is_gt = opt.get("letter") == rec.gt_label
            hl = ' class="gt-highlight"' if is_gt else ""
            rows.append(
                f'<tr{hl}><td>{opt["letter"]}</td>'
                f'<td>{html.escape(opt.get("lever_text", "")[:120])}</td>'
                f'<td>{opt.get("flip_fraction", 0.0):.2f}</td>'
                f'<td>{html.escape(opt.get("mechanism_name", ""))}</td>'
                f'<td>{html.escape(opt.get("category", ""))}</td></tr>'
            )
        e6_detail = f"""
        <div class="detail-section">
            <strong>E6 Options:</strong>
            <table class="e6-table">
                <tr><th>Letter</th><th>Lever text</th><th>Flip rate</th><th>Mechanism</th><th>Category</th></tr>
                {''.join(rows)}
            </table>
        </div>
        """

    # E2-specific: show GT text
    e2_detail = ""
    if rec.gt_text:
        e2_detail = f"""
        <div class="detail-section">
            <strong>GT Response:</strong>
            <pre class="gt-response">{html.escape(rec.gt_text[:500])}</pre>
        </div>
        """

    # E9-specific: show feature info
    e9_detail = ""
    if rec.e9_feature_name:
        e9_detail = f"""
        <div class="detail-section">
            <strong>Feature:</strong> {html.escape(rec.e9_feature_name)}
            — {html.escape(rec.e9_feature_description)}
        </div>
        """

    prompt_escaped = html.escape(rec.task_prompt)
    question_escaped = html.escape(rec.question[:300]) if rec.question else ""

    return f"""
    <div class="card" data-gt="{html.escape(badge_text)}"
         data-dataset="{html.escape(rec.dataset_id)}"
         data-text="{html.escape((rec.question + rec.lever_text)[:200].lower())}">
        <div class="card-header" onclick="this.parentElement.classList.toggle('expanded')">
            <span class="card-idx">#{idx}</span>
            <span class="badge" style="background:{badge_color}">{badge_text}</span>
            <span class="flip-badge">flip={ff_text}</span>
            <span class="dataset-tag">{html.escape(rec.dataset_id)}</span>
            <span class="variant-tag">{html.escape(rec.template_variant)}</span>
            <span class="question-preview">{question_escaped[:80]}...</span>
        </div>
        <div class="card-body">
            <div class="detail-section">
                <strong>Task Prompt (what model sees):</strong>
                <pre class="prompt-text">{prompt_escaped}</pre>
            </div>
            <div class="detail-section">
                <strong>Source:</strong> {html.escape(rec.dataset_id)} / example {rec.example_idx}
                | <strong>Lever:</strong> {html.escape(rec.lever_text[:150])}
                | <strong>Category:</strong> {html.escape(rec.category)}
            </div>
            {e6_detail}
            {e2_detail}
            {e9_detail}
        </div>
    </div>
    """


def _gt_filter_options(records: list[MultitaskRecord]) -> str:
    """Build GT label filter <option> tags."""
    if not records:
        return ""
    gt_type = records[0].gt_type
    if gt_type in ("binary", "mcq"):
        labels = sorted(set(r.gt_label for r in records))
        return "".join(f'<option value="{html.escape(l)}">{html.escape(l)}</option>' for l in labels)
    elif gt_type == "continuous":
        return (
            '<option value="0.00">0.0</option>'
            '<option value="1.00">1.0</option>'
        )
    return ""


def _dataset_filter_options(records: list[MultitaskRecord]) -> str:
    """Build dataset filter <option> tags."""
    datasets = sorted(set(r.dataset_id for r in records))
    return "".join(
        f'<option value="{html.escape(d)}">{html.escape(d)}</option>'
        for d in datasets[:30]  # Cap at 30 to keep dropdown reasonable
    )


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Multi-Task Training Data Viewer</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
h1 {{ margin: 0 0 5px; font-size: 1.4em; }}
.meta {{ color: #666; font-size: 0.85em; margin-bottom: 15px; }}
.tabs {{ display: flex; gap: 4px; margin-bottom: 15px; flex-wrap: wrap; }}
.tab-btn {{ padding: 6px 14px; border: 1px solid #ccc; background: #fff; cursor: pointer;
            border-radius: 4px 4px 0 0; font-size: 0.85em; }}
.tab-btn.active {{ background: #1565c0; color: white; border-color: #1565c0; }}
.stats {{ background: #fff; padding: 10px 14px; border-radius: 4px; margin-bottom: 10px;
          font-size: 0.85em; line-height: 1.6; border: 1px solid #ddd; }}
.stat-label {{ font-weight: 600; }}
.controls {{ display: flex; gap: 8px; margin-bottom: 10px; align-items: center; flex-wrap: wrap; }}
.controls input {{ padding: 5px 10px; border: 1px solid #ccc; border-radius: 4px; width: 200px; }}
.controls select {{ padding: 5px; border: 1px solid #ccc; border-radius: 4px; }}
.controls button {{ padding: 5px 10px; border: 1px solid #ccc; border-radius: 4px;
                    background: #fff; cursor: pointer; }}
.truncation {{ color: #e65100; font-size: 0.85em; }}
.card {{ background: #fff; border: 1px solid #ddd; border-radius: 4px; margin-bottom: 4px; }}
.card-header {{ display: flex; align-items: center; gap: 8px; padding: 6px 10px;
                cursor: pointer; font-size: 0.85em; }}
.card-header:hover {{ background: #f0f0f0; }}
.card-body {{ display: none; padding: 10px 14px; border-top: 1px solid #eee; }}
.card.expanded .card-body {{ display: block; }}
.card-idx {{ color: #999; min-width: 30px; }}
.badge {{ color: #fff; padding: 2px 8px; border-radius: 3px; font-size: 0.8em; font-weight: 600; }}
.flip-badge {{ color: #555; font-size: 0.8em; background: #f0f0f0; padding: 2px 6px; border-radius: 3px; }}
.dataset-tag {{ color: #1565c0; font-size: 0.8em; }}
.variant-tag {{ color: #6a1b9a; font-size: 0.75em; background: #f3e5f5; padding: 1px 5px; border-radius: 3px; }}
.question-preview {{ color: #666; font-size: 0.8em; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.detail-section {{ margin-bottom: 8px; }}
.detail-section strong {{ font-size: 0.85em; }}
pre.prompt-text {{ background: #f8f8f8; padding: 10px; border-radius: 4px; overflow-x: auto;
                   font-size: 0.8em; white-space: pre-wrap; word-break: break-word;
                   max-height: 400px; overflow-y: auto; border: 1px solid #e0e0e0; }}
pre.gt-response {{ background: #e8f5e9; padding: 8px; border-radius: 4px; font-size: 0.8em;
                   white-space: pre-wrap; max-height: 200px; overflow-y: auto; }}
.e6-table {{ width: 100%; border-collapse: collapse; font-size: 0.8em; margin-top: 4px; }}
.e6-table th, .e6-table td {{ border: 1px solid #ddd; padding: 4px 8px; text-align: left; }}
.e6-table th {{ background: #f5f5f5; }}
.gt-highlight {{ background: #e8f5e9; font-weight: 600; }}
.hidden {{ display: none !important; }}
</style>
</head>
<body>
<h1>Multi-Task Training Data Viewer</h1>
<div class="meta">Corpus: {corpus_dir} | {total_records} total records | {n_tasks} tasks</div>

<div class="tabs">{tabs}</div>
{panels}

<script>
function showTab(id) {{
    document.querySelectorAll('.tab-panel').forEach(p => p.style.display = 'none');
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('panel-' + id).style.display = 'block';
    document.getElementById('tab-' + id).classList.add('active');
}}
function toggleAll(tabId, expand) {{
    document.querySelectorAll('#cards-' + tabId + ' .card').forEach(c => {{
        if (expand) c.classList.add('expanded'); else c.classList.remove('expanded');
    }});
}}
function filterCards(tabId) {{
    const search = (document.getElementById('search-' + tabId).value || '').toLowerCase();
    const gt = document.getElementById('filter-gt-' + tabId).value;
    const ds = document.getElementById('filter-dataset-' + tabId).value;
    document.querySelectorAll('#cards-' + tabId + ' .card').forEach(c => {{
        const matchText = !search || c.dataset.text.includes(search);
        const matchGt = !gt || c.dataset.gt === gt;
        const matchDs = !ds || c.dataset.dataset === ds;
        c.classList.toggle('hidden', !(matchText && matchGt && matchDs));
    }});
}}
// Show first tab
document.querySelector('.tab-btn')?.click();
</script>
</body>
</html>
"""
