"""
Module: prompt_attribution/auto_perturbation/discovery_tracer.py

HTML tracer for pre-pipeline discovery stages: research agent, dataset profiling,
adapter schema detection, and label ideation.

Renders self-contained HTML files with collapsible sections showing:
- Research agent: LLM planning, web searches, content fetches, benchmark extraction,
  HuggingFace mapping attempts
- Profiling: suitability checks, detection results, capability tags, complexity,
  label ideation, adapter schema

Usage:
    tracer = DiscoveryTracer()
    tracer.record_research_plan(...)
    tracer.render_research_html(Path("research_trace.html"))

    tracer.record_profiling_start(...)
    tracer.record_profiling_dataset(...)
    tracer.render_profiling_html(Path("profiling_trace.html"))

Structure:
- DiscoveryLLMCall: A single LLM prompt/response pair
- WebAction: A web search or page fetch
- ModelResearchTrace: One model family's research journey
- BenchmarkMappingTrace: One benchmark's HF mapping attempts
- DatasetProfilingTrace: One dataset's suitability checks + detection + ideation
- DiscoveryTracer: Main tracer with recording methods + HTML rendering
"""

import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .html_helpers import (
    HTML_TEMPLATE,
    _badge,
    _data_table,
    _details,
    _esc,
    _pre,
    _table,
)

logger = logging.getLogger(__name__)


def _hf_link(hf_id: str) -> str:
    """Render a HuggingFace dataset ID as a clickable link.

    Handles config syntax like "org/dataset:config" by linking to
    the base dataset page.
    """
    base_id = hf_id.split(":")[0] if ":" in hf_id else hf_id
    url = f"https://huggingface.co/datasets/{base_id}"
    return f'<a href="{_esc(url)}" target="_blank"><code>{_esc(hf_id)}</code></a>'


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class DiscoveryLLMCall:
    """A single LLM prompt/response pair in the discovery pipeline."""

    stage: str  # "plan_research", "decide_url", "extract_benchmarks", etc.
    prompt: str
    response: str
    model: str = ""
    temperature: float = 0.0
    label: str = ""  # optional context (model name, benchmark name, etc.)


@dataclass
class WebAction:
    """A web search or page fetch."""

    action_type: str  # "search" or "fetch"
    query_or_url: str
    results: list[dict] = field(default_factory=list)  # [{title, url, snippet}]
    content_length: int = 0
    content_format: str = ""  # "html", "pdf", "text"
    error: str = ""


@dataclass
class BenchmarkMappingTrace:
    """One benchmark's HuggingFace mapping attempts."""

    name: str
    category: str = ""
    difficulty_tier: str = ""
    method: str = ""  # "known_map", "hub_search", "web_search", "llm_guided"
    failed_attempts: list[str] = field(default_factory=list)
    llm_calls: list[DiscoveryLLMCall] = field(default_factory=list)
    web_actions: list[WebAction] = field(default_factory=list)
    result_hf_id: str = ""  # empty if unmapped
    content_match: Optional[bool] = None  # None = not checked


@dataclass
class ModelResearchTrace:
    """One model family's research journey."""

    model_family: str
    lab: str = ""
    search_query: str = ""

    # URL finding
    known_urls_checked: list[str] = field(default_factory=list)
    web_actions: list[WebAction] = field(default_factory=list)
    url_decision_llm: Optional[DiscoveryLLMCall] = None
    result_url: str = ""
    url_confidence: str = ""

    # Content fetch
    content_length: int = 0
    content_format: str = ""

    # Benchmark extraction
    extraction_llm: Optional[DiscoveryLLMCall] = None
    benchmarks_extracted: list[dict] = field(default_factory=list)

    # HF mapping per benchmark
    benchmark_mappings: list[BenchmarkMappingTrace] = field(default_factory=list)

    # Validation
    validation_llm: Optional[DiscoveryLLMCall] = None
    validated: bool = True


@dataclass
class DatasetProfilingTrace:
    """One dataset's suitability checks, detection, and ideation."""

    dataset_id: str
    seed_source: str = ""  # "curated", "research_agent", "hub_search"
    seed_detail: str = ""  # extra info: "task=question-answering" or "tag=benchmark"
    suitable: bool = False
    rejection_reasons: list[str] = field(default_factory=list)

    # Suitability checks (list of (check_name, passed, detail))
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    # Detection context (columns available, how fields were matched)
    available_columns: list[str] = field(default_factory=list)
    detection_notes: list[str] = field(default_factory=list)  # human-readable reasoning

    # Detection results
    task_type: str = ""
    question_field: str = ""
    answer_field: str = ""
    choices_field: str = ""
    context_field: str = ""
    n_total: int = 0
    prompt_template: str = ""

    # Capability and complexity
    capability_tags: list[str] = field(default_factory=list)
    capability_method: str = ""  # "keyword", "known_benchmark", "llm", "research_agent"
    complexity: dict[str, str] = field(default_factory=dict)

    # Label ideation
    ideation_llm: Optional[DiscoveryLLMCall] = None
    answer_labels: list[dict] = field(default_factory=list)
    instruction_placement: str = ""

    # Additional LLM calls (capability classification, etc.)
    llm_calls: list[DiscoveryLLMCall] = field(default_factory=list)

    # Adapter schema (LLM detection fallback)
    adapter_heuristic_results: dict[str, str] = field(default_factory=dict)
    adapter_llm: Optional[DiscoveryLLMCall] = None
    adapter_fields_changed: list[str] = field(default_factory=list)
    final_prompt_template: str = ""  # Post-LLM template (if different from heuristic)


# =============================================================================
# Discovery Tracer
# =============================================================================


class DiscoveryTracer:
    """Records and renders HTML traces for pre-pipeline discovery stages.

    Thread-safe for sequential use (not concurrent). All record_* methods
    are designed to be no-ops when called on None (use `if tracer:` guard
    at call sites).
    """

    def __init__(self) -> None:
        # Research agent traces
        self._research_config: dict[str, Any] = {}
        self._research_plan_llm: Optional[DiscoveryLLMCall] = None
        self._research_plan_web_actions: list[WebAction] = []
        self._research_plan_table: list[dict] = []  # [{model_family, lab, search_query}]
        self._model_traces: list[ModelResearchTrace] = []

        # Profiling traces
        self._profiling_config: dict[str, Any] = {}
        self._dataset_traces: list[DatasetProfilingTrace] = []

        self._start_time = time.time()

    # -----------------------------------------------------------------
    # Research Agent Recording
    # -----------------------------------------------------------------

    def record_research_config(self, **kwargs: Any) -> None:
        """Record research agent configuration."""
        self._research_config.update(kwargs)

    def record_research_plan_web(self, action: WebAction) -> None:
        """Record a web search/fetch during planning."""
        self._research_plan_web_actions.append(action)

    def record_research_plan(
        self,
        model_queries: list[dict],
        llm_call: DiscoveryLLMCall,
    ) -> None:
        """Record the planning LLM call and resulting model list."""
        self._research_plan_llm = llm_call
        self._research_plan_table = model_queries

    def start_model_research(self, model_family: str, lab: str, search_query: str) -> ModelResearchTrace:
        """Start tracing a model family's research. Returns the trace object for in-place updates."""
        trace = ModelResearchTrace(
            model_family=model_family,
            lab=lab,
            search_query=search_query,
        )
        self._model_traces.append(trace)
        return trace

    # -----------------------------------------------------------------
    # Profiling Recording
    # -----------------------------------------------------------------

    def record_profiling_config(self, **kwargs: Any) -> None:
        """Record profiling configuration."""
        self._profiling_config.update(kwargs)

    def start_dataset_profiling(self, dataset_id: str) -> DatasetProfilingTrace:
        """Start tracing a dataset's profiling. Returns the trace object for in-place updates."""
        trace = DatasetProfilingTrace(dataset_id=dataset_id)
        self._dataset_traces.append(trace)
        return trace

    # -----------------------------------------------------------------
    # Research HTML Rendering
    # -----------------------------------------------------------------

    def render_research_html(self, output_path: Path) -> None:
        """Render research agent trace as self-contained HTML."""
        sections = []

        # Config section
        config_rows = [(k, str(v)) for k, v in self._research_config.items()]
        config_rows.append(("elapsed", f"{time.time() - self._start_time:.1f}s"))
        config_html = _table(config_rows) if config_rows else ""

        # Planning section
        sections.append(self._render_research_planning())

        # Per-model sections
        for trace in self._model_traces:
            sections.append(self._render_model_trace(trace))

        # Summary
        sections.append(self._render_research_summary())

        body = "\n".join(sections)
        html_content = HTML_TEMPLATE.format(
            title="Research Agent Trace",
            config_section=config_html,
            body=body,
            extra_css="",
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(html_content)
        logger.info(f"DiscoveryTracer: wrote research trace to {output_path}")

    def _render_research_planning(self) -> str:
        """Render the planning phase (web searches + LLM synthesis)."""
        parts = []

        # Web searches during planning
        if self._research_plan_web_actions:
            web_parts = []
            for wa in self._research_plan_web_actions:
                if wa.action_type == "search":
                    result_items = "".join(
                        f"<li>{_esc(r.get('title', ''))} — "
                        f"<a href=\"{_esc(r.get('url', ''))}\">{_esc(r.get('url', '')[:80])}</a></li>"
                        for r in wa.results
                    )
                    web_parts.append(
                        f'<p class="meta">Query: <strong>{_esc(wa.query_or_url)}</strong></p>'
                        f"<ul>{result_items}</ul>" if result_items else
                        f'<p class="meta">Query: <strong>{_esc(wa.query_or_url)}</strong> — no results</p>'
                    )
            if web_parts:
                parts.append(_details(
                    "Web Searches for Latest Models",
                    "\n".join(web_parts),
                ))

        # LLM synthesis
        if self._research_plan_llm:
            llm = self._research_plan_llm
            parts.append(_details(
                "LLM Planning Call",
                _details("Prompt", _pre(llm.prompt), css_class="prompt-section")
                + _details("Response", _pre(llm.response), css_class="response-section"),
            ))

        # Model list table
        if self._research_plan_table:
            rows = [
                [m.get("model_family", ""), m.get("lab", ""), m.get("search_query", "")]
                for m in self._research_plan_table
            ]
            parts.append(_data_table(
                ["Model Family", "Lab", "Search Query"],
                rows,
            ))

        return _details(
            f"Planning ({len(self._research_plan_table)} models)",
            "\n".join(parts),
            css_class="stage-section",
            open_default=True,
        )

    def _render_model_trace(self, trace: ModelResearchTrace) -> str:
        """Render one model family's research journey."""
        parts = []

        # Validation status
        status = "confirmed" if trace.validated else "not confirmed"
        header = f"Model: {trace.model_family} ({trace.lab})"
        if not trace.validated:
            header += _badge("UNCONFIRMED", "#dc3545")

        # URL Finding + Content Fetch
        url_parts = []
        if trace.known_urls_checked:
            known_links = ", ".join(
                f'<a href="{_esc(u)}" target="_blank">{_esc(u[:80])}</a>'
                for u in trace.known_urls_checked
            )
            url_parts.append(f'<p class="meta">Known URLs checked: {known_links}</p>')

        for wa in trace.web_actions:
            if wa.action_type == "search":
                items = "".join(
                    f"<li>{_esc(r.get('title', '')[:80])} \u2014 "
                    f"<a href=\"{_esc(r.get('url', ''))}\" target=\"_blank\">{_esc(r.get('url', '')[:80])}</a></li>"
                    for r in wa.results
                )
                url_parts.append(
                    f'<p class="meta">Search: <strong>{_esc(wa.query_or_url)}</strong></p>'
                    + (f"<ul>{items}</ul>" if items else '<p class="meta">No results</p>')
                )
            elif wa.action_type == "fetch":
                status_str = f"{wa.content_length:,} chars ({wa.content_format})" if wa.content_length else wa.error or "empty"
                url_parts.append(
                    f'<p class="meta">Fetch: '
                    f'<a href="{_esc(wa.query_or_url)}" target="_blank">{_esc(wa.query_or_url[:100])}</a>'
                    f' \u2014 {_esc(status_str)}</p>'
                )

        if trace.url_decision_llm:
            llm = trace.url_decision_llm
            url_parts.append(_details(
                "LLM URL Decision",
                _details("Prompt", _pre(llm.prompt), css_class="prompt-section")
                + _details("Response", _pre(llm.response), css_class="response-section"),
            ))

        if trace.result_url:
            conf = f" (confidence: {trace.url_confidence})" if trace.url_confidence else ""
            content_info = f" \u2014 {trace.content_length:,} chars ({trace.content_format})" if trace.content_length else ""
            url_parts.append(
                f'<p class="check-pass"><strong>Result:</strong> '
                f'<a href="{_esc(trace.result_url)}" target="_blank">{_esc(trace.result_url[:120])}</a>'
                f'{_esc(conf)}{content_info}</p>'
            )

        if url_parts:
            parts.append(_details("URL Finding + Content Fetch", "\n".join(url_parts)))

        # Benchmark Extraction
        if trace.extraction_llm:
            llm = trace.extraction_llm
            extract_parts = [
                _details("Prompt", _pre(llm.prompt), css_class="prompt-section"),
                _details("Response", _pre(llm.response), css_class="response-section"),
            ]
            if trace.benchmarks_extracted:
                rows = [
                    [
                        b.get("name", ""),
                        b.get("category", ""),
                        b.get("difficulty_tier", ""),
                        b.get("reported_score", ""),
                    ]
                    for b in trace.benchmarks_extracted
                ]
                extract_parts.append(_data_table(
                    ["Name", "Category", "Tier", "Score"],
                    rows,
                ))
            parts.append(_details(
                f"Benchmark Extraction ({len(trace.benchmarks_extracted)} found)",
                "\n".join(extract_parts),
            ))

        # HF Mapping per benchmark
        if trace.benchmark_mappings:
            # Flow explanation
            mapping_parts = [
                '<p class="meta">Mapping flow per benchmark: '
                '(1) known_map \u2192 dictionary lookup, no LLM &nbsp;'
                '(2) hub_search \u2192 HfApi search + LLM content validation &nbsp;'
                '(3) web_search \u2192 DuckDuckGo + LLM content validation &nbsp;'
                '(4) llm_guided \u2192 LLM suggests IDs + validates each. '
                'Stops at first successful match.</p>'
            ]
            for bm in trace.benchmark_mappings:
                icon = "&#x2713;" if bm.result_hf_id else "&#x2717;"
                color_cls = "check-pass" if bm.result_hf_id else "check-fail"
                result_html = _hf_link(bm.result_hf_id) if bm.result_hf_id else "<code>unmapped</code>"

                # Header line
                bm_header = (
                    f'<span class="{color_cls}">{icon}</span> '
                    f'<strong>{_esc(bm.name)}</strong> '
                    f'({_esc(bm.category)})'
                )

                # Build step-by-step detail
                steps = []
                steps.append(
                    f'<p class="meta"><strong>Resolved via:</strong> {_esc(bm.method)} '
                    f'\u2192 {result_html}</p>'
                )
                if bm.failed_attempts:
                    failed_links = ", ".join(_hf_link(a) for a in bm.failed_attempts)
                    steps.append(f'<p class="meta"><strong>Failed attempts:</strong> {failed_links}</p>')

                # Searches (web + HF hub)
                for wa in bm.web_actions:
                    search_type = "HF Hub" if wa.action_type == "hf_hub_search" else "Web"
                    items = "".join(
                        f"<li>{_esc(r.get('title', '')[:60])} \u2014 "
                        f"<code>{_esc(r.get('url', '')[:80])}</code>"
                        + (f' <span class="meta">({_esc(r.get("snippet", "")[:60])})</span>' if r.get("snippet") else "")
                        + "</li>"
                        for r in wa.results
                    )
                    steps.append(
                        f'<p class="meta"><strong>{search_type} search:</strong> {_esc(wa.query_or_url)}</p>'
                        + (f"<ul>{items}</ul>" if items else '<p class="meta">(no results)</p>')
                    )

                # LLM calls
                for llm in bm.llm_calls:
                    steps.append(_details(
                        f"LLM: {_esc(llm.stage)}"
                        + (f" ({_esc(llm.label)})" if llm.label else ""),
                        _details("Prompt", _pre(llm.prompt), css_class="prompt-section")
                        + _details("Response", _pre(llm.response), css_class="response-section"),
                    ))

                has_interactions = bm.web_actions or bm.llm_calls or bm.failed_attempts
                if has_interactions:
                    mapping_parts.append(_details(
                        bm_header,
                        "\n".join(steps),
                    ))
                else:
                    # known_map: just show the one-liner
                    mapping_parts.append(f"<p>{bm_header} \u2192 {result_html}</p>")

            mapped_count = sum(1 for bm in trace.benchmark_mappings if bm.result_hf_id)
            parts.append(_details(
                f"HF Mapping ({mapped_count}/{len(trace.benchmark_mappings)} mapped)",
                "\n".join(mapping_parts),
            ))

        return _details(header, "\n".join(parts), css_class="example-section")

    def _render_research_summary(self) -> str:
        """Render research summary statistics."""
        all_benchmarks = []
        for mt in self._model_traces:
            all_benchmarks.extend(mt.benchmark_mappings)

        total = len(all_benchmarks)
        mapped = sum(1 for b in all_benchmarks if b.result_hf_id)
        unmapped = total - mapped

        category_counts = Counter(b.category for b in all_benchmarks if b.category)
        method_counts = Counter(b.method for b in all_benchmarks if b.method)

        parts = []
        parts.append(
            f'<div class="summary-grid">'
            f'<div class="summary-card"><h3>Total Benchmarks</h3>'
            f'<div class="big-number">{total}</div></div>'
            f'<div class="summary-card"><h3>Mapped to HF</h3>'
            f'<div class="big-number" style="color:#166534">{mapped}</div></div>'
            f'<div class="summary-card"><h3>Unmapped</h3>'
            f'<div class="big-number" style="color:#991b1b">{unmapped}</div></div>'
            f'<div class="summary-card"><h3>Models Researched</h3>'
            f'<div class="big-number">{len(self._model_traces)}</div></div>'
            f'</div>'
        )

        if category_counts:
            rows = [[cat, str(cnt)] for cat, cnt in category_counts.most_common()]
            parts.append(_details(
                "By Category",
                _data_table(["Category", "Count"], rows),
            ))

        if method_counts:
            rows = [[method, str(cnt)] for method, cnt in method_counts.most_common()]
            parts.append(_details(
                "By Mapping Method",
                _data_table(["Method", "Count"], rows),
            ))

        return _details(
            "Summary",
            "\n".join(parts),
            css_class="stage-section",
            open_default=True,
        )

    # -----------------------------------------------------------------
    # Profiling HTML Rendering
    # -----------------------------------------------------------------

    def render_profiling_html(self, output_path: Path) -> None:
        """Render profiling trace as self-contained HTML."""
        sections = []

        # Config section
        config_rows = [(k, str(v)) for k, v in self._profiling_config.items()]
        config_rows.append(("elapsed", f"{time.time() - self._start_time:.1f}s"))
        config_html = _table(config_rows) if config_rows else ""

        # Per-dataset sections
        for trace in self._dataset_traces:
            sections.append(self._render_dataset_trace(trace))

        # Summary
        sections.append(self._render_profiling_summary())

        body = "\n".join(sections)
        html_content = HTML_TEMPLATE.format(
            title="Dataset Profiling Trace",
            config_section=config_html,
            body=body,
            extra_css="",
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(html_content)
        logger.info(f"DiscoveryTracer: wrote profiling trace to {output_path}")

    def _render_dataset_trace(self, trace: DatasetProfilingTrace) -> str:
        """Render one dataset's profiling trace."""
        parts = []

        # Status badge
        if trace.suitable:
            badge = _badge("SUITABLE", "#166534", "#bbf7d0")
        else:
            badge = _badge("REJECTED", "#991b1b", "#fecaca")

        source_badge = ""
        if trace.seed_source:
            source_colors = {
                "curated": ("#495057", "#e9ecef"),
                "research_agent": ("#0d6efd", "#dbeafe"),
                "hub_search": ("#6f42c1", "#e8daef"),
            }
            sc, sb = source_colors.get(trace.seed_source, ("#6c757d", "#f8f9fa"))
            label = trace.seed_source
            if trace.seed_detail:
                label += f": {trace.seed_detail}"
            source_badge = _badge(label, sc, sb)
        header = f"{_hf_link(trace.dataset_id)}{badge}{source_badge}"

        # Suitability checks
        if trace.checks:
            check_parts = []
            for check_name, passed, detail in trace.checks:
                icon = "&#x2713;" if passed else "&#x2717;"
                cls = "check-pass" if passed else "check-fail"
                check_parts.append(
                    f'<p class="{cls}">{icon} {_esc(check_name)}'
                    + (f': {_esc(detail)}' if detail else "")
                    + "</p>"
                )
            parts.append(_details(
                "Suitability Checks",
                "\n".join(check_parts),
                open_default=not trace.suitable,
            ))

        # Rejection reason (if rejected)
        if not trace.suitable and trace.rejection_reasons:
            parts.append(
                f'<p class="check-fail"><strong>Rejection:</strong> '
                + "; ".join(_esc(r) for r in trace.rejection_reasons)
                + "</p>"
            )

        # Schema Detection — unified two-phase section
        if trace.suitable and (trace.task_type or trace.adapter_heuristic_results or trace.adapter_llm):
            schema_parts = []

            # Phase explanation
            if trace.adapter_llm:
                schema_parts.append(
                    '<p class="meta"><strong>Phase 1 (profiling):</strong> '
                    'Heuristic field matching by column name + HF schema. '
                    '<strong>Phase 2 (pipeline):</strong> LLM reads full schema + '
                    'sample rows to verify and correct.</p>'
                )
            else:
                schema_parts.append(
                    '<p class="meta"><strong>Heuristic only</strong> — '
                    'LLM verification runs later during pipeline execution.</p>'
                )

            # Phase 1: Heuristic detection with reasoning
            heuristic_parts = []
            if trace.available_columns:
                heuristic_parts.append(
                    f'<p class="meta">Available columns: '
                    + ", ".join(f"<code>{_esc(c)}</code>" for c in trace.available_columns)
                    + "</p>"
                )
            if trace.detection_notes:
                for note in trace.detection_notes:
                    heuristic_parts.append(f'<p class="meta">{_esc(note)}</p>')
            det_rows = [
                ("task_type", trace.task_type),
                ("question_field", trace.question_field),
                ("answer_field", trace.answer_field or "(none)"),
                ("choices_field", trace.choices_field or "(none)"),
                ("context_field", trace.context_field or "(none)"),
                ("n_total", str(trace.n_total)),
            ]
            heuristic_parts.append(_table(det_rows))
            if trace.prompt_template:
                heuristic_parts.append(_details("Prompt Template", _pre(trace.prompt_template)))
            schema_parts.append(_details(
                "Phase 1: Heuristic Detection (profiling)",
                "\n".join(heuristic_parts),
                open_default=not trace.adapter_llm,  # open if no LLM phase yet
            ))

            # Phase 2: LLM verification
            if trace.adapter_llm:
                llm = trace.adapter_llm
                correction_note = ""
                if trace.adapter_fields_changed:
                    correction_note = (
                        '<p class="meta" style="color:#dc3545"><strong>LLM corrected: </strong>'
                        + ", ".join(f"<code>{_esc(f)}</code>" for f in trace.adapter_fields_changed)
                        + "</p>"
                    )
                else:
                    correction_note = '<p class="meta" style="color:#28a745">LLM confirmed heuristic results (no changes)</p>'
                phase2_content = (
                    _details("Prompt", _pre(llm.prompt), css_class="prompt-section")
                    + _details("Response", _pre(llm.response), css_class="response-section")
                    + correction_note
                )
                if trace.final_prompt_template:
                    phase2_content += _details(
                        "Final Prompt Template (after LLM)",
                        _pre(trace.final_prompt_template),
                    )
                schema_parts.append(_details(
                    "Phase 2: LLM Verification (pipeline)",
                    phase2_content,
                ))

            parts.append(_details("Schema Detection", "\n".join(schema_parts)))

        # Complexity (affects pipeline filtering)
        if trace.complexity:
            im = trace.complexity.get("interaction_mode", "static")
            cs = trace.complexity.get("context_source", "single")
            cl = trace.complexity.get("context_length", "short")
            will_skip = (
                cs == "multimodal_context"
                or im in ("tool_use", "multi_turn")
                or cl == "long"
            )
            skip_reasons = []
            if cs == "multimodal_context":
                skip_reasons.append("multimodal context")
            if im in ("tool_use", "multi_turn"):
                skip_reasons.append(f"interaction_mode={im}")
            if cl == "long":
                skip_reasons.append("long context")
            filter_note = (
                f'<p class="check-fail">Pipeline will skip this dataset '
                f'({", ".join(skip_reasons)})</p>'
                if will_skip else ""
            )
            parts.append(_details(
                "Complexity",
                _table([(k, v) for k, v in trace.complexity.items()])
                + filter_note,
            ))

        # Label ideation
        if trace.ideation_llm or trace.answer_labels:
            ideation_parts = []
            if trace.ideation_llm:
                llm = trace.ideation_llm
                ideation_parts.append(_details(
                    "LLM Prompt",
                    _pre(llm.prompt),
                    css_class="prompt-section",
                ))
                ideation_parts.append(_details(
                    "LLM Response",
                    _pre(llm.response),
                    css_class="response-section",
                ))
            if trace.answer_labels:
                rows = [
                    [
                        al.get("name", ""),
                        al.get("value_type", ""),
                        al.get("verification_method", ""),
                        al.get("extraction_pattern", "") or al.get("judge_prompt", "")[:60],
                    ]
                    for al in trace.answer_labels
                ]
                ideation_parts.append(_data_table(
                    ["Name", "Type", "Method", "Pattern / Judge"],
                    rows,
                ))
            parts.append(_details(
                f"Label Ideation ({len(trace.answer_labels)} labels)",
                "\n".join(ideation_parts),
            ))

        # Capability Classification (LLM calls + resulting tags)
        if trace.llm_calls or trace.capability_tags:
            cap_parts = []
            for llm in trace.llm_calls:
                cap_parts.append(_details(
                    f"LLM: {_esc(llm.stage)}"
                    + (f" ({_esc(llm.label)})" if llm.label else ""),
                    _details("Prompt", _pre(llm.prompt), css_class="prompt-section")
                    + _details("Response", _pre(llm.response), css_class="response-section"),
                ))
            if trace.capability_tags:
                tag_html = " ".join(
                    _badge(tag, "#0d6efd", "#dbeafe") for tag in trace.capability_tags
                )
                cap_parts.append(f'<p>Result: {tag_html}</p>')
            parts.append(_details(
                f"Capability Classification ({', '.join(trace.capability_tags) if trace.capability_tags else 'pending'})",
                "\n".join(cap_parts),
            ))

        return _details(
            header,
            "\n".join(parts),
            css_class="example-section",
            open_default=not trace.suitable,  # open rejected ones for debugging
        )

    def _render_profiling_summary(self) -> str:
        """Render profiling summary statistics."""
        total = len(self._dataset_traces)
        suitable = sum(1 for t in self._dataset_traces if t.suitable)
        rejected = total - suitable

        # Rejection reason counts
        rejection_reasons: Counter = Counter()
        for t in self._dataset_traces:
            if not t.suitable:
                for reason in t.rejection_reasons:
                    # Normalize to first ~30 chars for grouping
                    key = reason[:50].split(":")[0].strip()
                    rejection_reasons[key] += 1

        # Capability distribution
        cap_counts: Counter = Counter()
        for t in self._dataset_traces:
            if t.suitable:
                for tag in t.capability_tags:
                    cap_counts[tag] += 1

        # Task type distribution
        task_counts: Counter = Counter()
        for t in self._dataset_traces:
            if t.suitable and t.task_type:
                task_counts[t.task_type] += 1

        parts = []
        parts.append(
            f'<div class="summary-grid">'
            f'<div class="summary-card"><h3>Total Profiled</h3>'
            f'<div class="big-number">{total}</div></div>'
            f'<div class="summary-card"><h3>Suitable</h3>'
            f'<div class="big-number" style="color:#166534">{suitable}</div></div>'
            f'<div class="summary-card"><h3>Rejected</h3>'
            f'<div class="big-number" style="color:#991b1b">{rejected}</div></div>'
            f'</div>'
        )

        if rejection_reasons:
            rows = [[reason, str(cnt)] for reason, cnt in rejection_reasons.most_common()]
            parts.append(_details(
                "Rejection Reasons",
                _data_table(["Reason", "Count"], rows),
                open_default=True,
            ))

        if cap_counts:
            rows = [[cap, str(cnt)] for cap, cnt in cap_counts.most_common()]
            parts.append(_details(
                "Capability Distribution",
                _data_table(["Capability", "Count"], rows),
            ))

        if task_counts:
            rows = [[tt, str(cnt)] for tt, cnt in task_counts.most_common()]
            parts.append(_details(
                "Task Type Distribution",
                _data_table(["Task Type", "Count"], rows),
            ))

        return _details(
            "Summary",
            "\n".join(parts),
            css_class="stage-section",
            open_default=True,
        )

