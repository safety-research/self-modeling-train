"""
Module: prompt_attribution/auto_perturbation/pipeline_tracer.py

Per-example debug tracer for the training data pipeline.

Follows N examples through every pipeline stage, recording the exact prompts
sent to LLMs, responses received, and decisions made. Renders a self-contained
HTML file with collapsible sections and baseline-vs-lever diff visualization.

Usage:
    # Enable via CLI flag:
    --trace_examples 2

    # Or via config:
    PipelineConfig(trace_examples=2)

    # Output: {run_dir}/trace.html

Structure:
- LLMCall: A single LLM prompt/response pair
- StageTrace: Trace data for one pipeline stage for one example
- FeedbackRoundTrace: Trace data for one feedback loop round
- ExampleTrace: Complete trace for one example through all stages
- PipelineTracer: Main tracer class with per-stage recording + HTML rendering
"""

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .html_helpers import (
    CATEGORY_COLORS as _CATEGORY_COLORS,
    HTML_TEMPLATE as _HTML_TEMPLATE,
    _details,
    _esc,
    _pre,
    _render_diff,
    _table,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class LLMCall:
    """A single LLM prompt/response pair."""

    component: str  # "decomposer", "generator", "critic", "mini_verifier", "verifier"
    label: str  # "flip_inducing", "candidate_abc123", etc.
    system_prompt: str  # "" if single-message prompt
    user_prompt: str
    response: str
    model: str
    temperature: float


@dataclass
class StageTrace:
    """Trace data for one pipeline stage for one example."""

    stage_name: str
    llm_calls: list[LLMCall] = field(default_factory=list)
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    decisions: list[str] = field(default_factory=list)


@dataclass
class FeedbackRoundTrace:
    """Trace data for one feedback loop round."""

    round_num: int
    category: str
    mismatches: list[dict] = field(default_factory=list)
    feedback_message: str = ""
    new_candidates: list[dict] = field(default_factory=list)
    verification_results: list[dict] = field(default_factory=list)


@dataclass
class CandidateTrace:
    """Trace for a single candidate through critic/verify/export stages."""

    candidate_id: str
    category: str
    mechanism_name: str
    perturbation_type: str
    lever_text: str
    target_label_axis: str
    baseline_prompt: str = ""
    lever_prompt: str = ""
    # Critic
    critic_prompt: str = ""
    critic_response: str = ""
    predicted_flip: Optional[float] = None
    critic_notes: str = ""
    # Mini-verify
    mini_verify_baseline_resp: str = ""
    mini_verify_lever_resp: str = ""
    mini_verify_flip_result: dict = field(default_factory=dict)
    # Full verify
    verify_baseline_resps: list[str] = field(default_factory=list)
    verify_lever_resps: list[str] = field(default_factory=list)
    verify_flip_fraction: Optional[float] = None
    verify_flipped: Optional[bool] = None
    # Export
    export_fields: dict = field(default_factory=dict)


@dataclass
class ExampleTrace:
    """Complete trace for one example through all stages."""

    example_idx: int
    question: str = ""
    ground_truth_answer: str = ""
    context: str = ""
    choices: Optional[list[str]] = None
    dataset_id: str = ""
    stages: dict[str, StageTrace] = field(default_factory=dict)
    candidates: dict[str, CandidateTrace] = field(default_factory=dict)
    feedback_rounds: list[FeedbackRoundTrace] = field(default_factory=list)
    # Adapt-stage profile info
    profile_info: dict = field(default_factory=dict)


# =============================================================================
# PipelineTracer
# =============================================================================


class PipelineTracer:
    """Traces N examples through every pipeline stage.

    Selected once at adapt time, the same examples are tracked through all
    subsequent stages. Thread-safe for asyncio (single-threaded cooperative).

    Each recording method is a no-op if the example is not traced.
    """

    def __init__(self, n_trace: int, random_seed: int):
        self._n_trace = n_trace
        self._random_seed = random_seed
        self._traced_indices: set[int] = set()
        self._traces: dict[int, ExampleTrace] = {}
        self._config_info: dict = {}
        # All export fields (not just traced) for dataset summary
        self._all_exports: list[dict] = []

    def set_config_info(self, config_info: dict) -> None:
        """Store pipeline config for the HTML header."""
        self._config_info = config_info

    def select_examples(self, example_indices: list[int]) -> None:
        """Pick N examples to trace. Called ONCE after adapt stage.

        Args:
            example_indices: All available example indices.
        """
        rng = random.Random(self._random_seed)
        n = min(self._n_trace, len(example_indices))
        selected = rng.sample(example_indices, n)
        self._traced_indices = set(selected)
        for idx in selected:
            self._traces[idx] = ExampleTrace(example_idx=idx)
        logger.info(f"PipelineTracer: tracing examples {sorted(selected)}")

    def is_traced(self, example_idx: int) -> bool:
        """Fast O(1) check — used at every hook point."""
        return example_idx in self._traced_indices

    # -----------------------------------------------------------------
    # Stage recording methods
    # -----------------------------------------------------------------

    def record_adapt(
        self,
        example_idx: int,
        question: str,
        ground_truth_answer: str,
        context: str,
        choices: Optional[list[str]],
        dataset_id: str,
        profile_info: dict,
    ) -> None:
        """Record adapt stage data."""
        if not self.is_traced(example_idx):
            return
        trace = self._traces[example_idx]
        trace.question = question
        trace.ground_truth_answer = ground_truth_answer
        trace.context = context or ""
        trace.choices = choices
        trace.dataset_id = dataset_id
        trace.profile_info = profile_info

        trace.stages["adapt"] = StageTrace(
            stage_name="1. Adapt",
            inputs={
                "dataset_id": dataset_id,
                "question": question[:200] + ("..." if len(question) > 200 else ""),
                "ground_truth_answer": ground_truth_answer,
            },
            outputs={"profile": profile_info},
        )

    def record_decompose(
        self,
        example_idx: int,
        prompt: str,
        response: str,
        parsed_analysis: dict,
        model: str,
        temperature: float,
    ) -> None:
        """Record decompose stage: exact prompt, response, parsed analysis."""
        if not self.is_traced(example_idx):
            return
        trace = self._traces[example_idx]
        stage = StageTrace(
            stage_name="2. Decompose",
            llm_calls=[
                LLMCall(
                    component="decomposer",
                    label="decompose",
                    system_prompt="",
                    user_prompt=prompt,
                    response=response,
                    model=model,
                    temperature=temperature,
                )
            ],
            outputs=parsed_analysis,
        )
        trace.stages["decompose"] = stage

    def record_generate(
        self,
        example_idx: int,
        category: str,
        system_prompt: str,
        user_prompt: str,
        response: str,
        candidates: list[dict],
        model: str,
        temperature: float,
    ) -> None:
        """Record generate stage: per-category prompt/response/parsed candidates."""
        if not self.is_traced(example_idx):
            return
        trace = self._traces[example_idx]
        if "generate" not in trace.stages:
            trace.stages["generate"] = StageTrace(stage_name="3. Generate")
        stage = trace.stages["generate"]
        stage.llm_calls.append(
            LLMCall(
                component="generator",
                label=category,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response=response,
                model=model,
                temperature=temperature,
            )
        )
        # Store parsed candidates in outputs keyed by category
        stage.outputs[category] = candidates

    def record_critic(
        self,
        example_idx: int,
        candidate_id: str,
        category: str,
        mechanism_name: str,
        perturbation_type: str,
        lever_text: str,
        target_label_axis: str,
        prompt: str,
        response: str,
        predicted_flip: float,
        notes: str,
        model: str,
    ) -> None:
        """Record critic stage: per-candidate prompt/response/score."""
        if not self.is_traced(example_idx):
            return
        trace = self._traces[example_idx]
        ct = trace.candidates.setdefault(
            candidate_id,
            CandidateTrace(
                candidate_id=candidate_id,
                category=category,
                mechanism_name=mechanism_name,
                perturbation_type=perturbation_type,
                lever_text=lever_text,
                target_label_axis=target_label_axis,
            ),
        )
        ct.critic_prompt = prompt
        ct.critic_response = response
        ct.predicted_flip = predicted_flip
        ct.critic_notes = notes

    def record_mini_verify(
        self,
        example_idx: int,
        candidate_id: str,
        category: str,
        mechanism_name: str,
        perturbation_type: str,
        lever_text: str,
        target_label_axis: str,
        baseline_prompt: str,
        lever_prompt: str,
        baseline_response: str,
        lever_response: str,
        flip_result: dict,
    ) -> None:
        """Record mini-verification: prompts, model responses, flip result."""
        if not self.is_traced(example_idx):
            return
        trace = self._traces[example_idx]
        ct = trace.candidates.setdefault(
            candidate_id,
            CandidateTrace(
                candidate_id=candidate_id,
                category=category,
                mechanism_name=mechanism_name,
                perturbation_type=perturbation_type,
                lever_text=lever_text,
                target_label_axis=target_label_axis,
            ),
        )
        ct.baseline_prompt = baseline_prompt
        ct.lever_prompt = lever_prompt
        ct.mini_verify_baseline_resp = baseline_response
        ct.mini_verify_lever_resp = lever_response
        ct.mini_verify_flip_result = flip_result

    def record_feedback(
        self,
        example_idx: int,
        round_num: int,
        category: str,
        mismatches: list[dict],
        feedback_message: str,
        new_candidates: list[dict],
    ) -> None:
        """Record feedback loop round: mismatches, feedback text, new candidates."""
        if not self.is_traced(example_idx):
            return
        trace = self._traces[example_idx]
        trace.feedback_rounds.append(
            FeedbackRoundTrace(
                round_num=round_num,
                category=category,
                mismatches=mismatches,
                feedback_message=feedback_message,
                new_candidates=new_candidates,
            )
        )

    def record_verify(
        self,
        example_idx: int,
        candidate_id: str,
        category: str,
        mechanism_name: str,
        perturbation_type: str,
        lever_text: str,
        target_label_axis: str,
        baseline_prompt: str,
        lever_prompt: str,
        baseline_responses: list[str],
        lever_responses: list[str],
        flip_fraction: float,
        flipped: bool,
    ) -> None:
        """Record full verification: all runs' responses, flip stats."""
        if not self.is_traced(example_idx):
            return
        trace = self._traces[example_idx]
        ct = trace.candidates.setdefault(
            candidate_id,
            CandidateTrace(
                candidate_id=candidate_id,
                category=category,
                mechanism_name=mechanism_name,
                perturbation_type=perturbation_type,
                lever_text=lever_text,
                target_label_axis=target_label_axis,
            ),
        )
        ct.baseline_prompt = baseline_prompt
        ct.lever_prompt = lever_prompt
        ct.verify_baseline_resps = baseline_responses
        ct.verify_lever_resps = lever_responses
        ct.verify_flip_fraction = flip_fraction
        ct.verify_flipped = flipped

    def record_export(
        self,
        example_idx: int,
        candidate_id: str,
        export_fields: dict,
    ) -> None:
        """Record export stage: final TrainingExample fields."""
        # Always accumulate for dataset summary (all examples)
        self._all_exports.append(export_fields)
        # Detailed trace only for traced examples
        if not self.is_traced(example_idx):
            return
        trace = self._traces[example_idx]
        if candidate_id in trace.candidates:
            trace.candidates[candidate_id].export_fields = export_fields

    # -----------------------------------------------------------------
    # HTML Rendering
    # -----------------------------------------------------------------

    def render_html(self, output_path: Path) -> None:
        """Render complete HTML trace report and write to file."""
        if not self._traces:
            logger.warning("PipelineTracer: no traces to render")
            return

        sections = []
        for idx in sorted(self._traces.keys()):
            trace = self._traces[idx]
            sections.append(self._render_example(trace))

        # Dataset summary at the end (all examples, not just traced)
        sections.append(self._render_dataset_summary())

        config_html = self._render_config()
        body = "\n".join(sections)

        html_content = _HTML_TEMPLATE.format(
            title=f"Pipeline Trace — {self._config_info.get('dataset_id', 'unknown')}",
            config_section=config_html,
            body=body,
            extra_css="",
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(html_content)
        logger.info(f"PipelineTracer: wrote trace to {output_path}")

    def _render_config(self) -> str:
        """Render config summary as HTML table."""
        if not self._config_info:
            return "<p>No config info available.</p>"
        rows = []
        for key, val in self._config_info.items():
            rows.append(
                f"<tr><td><strong>{_esc(key)}</strong></td>"
                f"<td>{_esc(str(val))}</td></tr>"
            )
        return f'<table class="config-table">{"".join(rows)}</table>'

    def _render_example(self, trace: ExampleTrace) -> str:
        """Render one example's full trace."""
        question_preview = trace.question[:80] + ("..." if len(trace.question) > 80 else "")
        parts = []

        # Stage 1: Adapt
        if "adapt" in trace.stages:
            parts.append(self._render_adapt(trace))

        # Stage 2: Decompose
        if "decompose" in trace.stages:
            parts.append(self._render_decompose(trace.stages["decompose"]))

        # Stage 3: Generate
        if "generate" in trace.stages:
            parts.append(self._render_generate(trace.stages["generate"]))

        # Stage 4: Critic / Mini-Verify (per-candidate)
        if trace.candidates:
            parts.append(self._render_candidates(trace))

        # Stage 5: Feedback Loop
        if trace.feedback_rounds:
            parts.append(self._render_feedback(trace.feedback_rounds))

        # Stage 6: Verify (inside candidates section above)
        # Stage 7: Export — show diff visualization per candidate
        export_candidates = [
            ct for ct in trace.candidates.values() if ct.export_fields
        ]
        if export_candidates:
            parts.append(self._render_export(export_candidates))

        content = "\n".join(parts)
        return _details(
            f'Example idx={trace.example_idx}: "{_esc(question_preview)}"',
            content,
            css_class="example-section",
            open_default=True,
        )

    def _render_adapt(self, trace: ExampleTrace) -> str:
        """Render adapt stage."""
        stage = trace.stages["adapt"]
        rows = [
            ("Dataset", trace.dataset_id),
            ("Question", trace.question),
            ("Ground Truth", trace.ground_truth_answer),
        ]
        if trace.context:
            rows.append(("Context", trace.context[:500] + ("..." if len(trace.context) > 500 else "")))
        if trace.choices:
            rows.append(("Choices", ", ".join(trace.choices)))

        # Profile info
        profile = trace.profile_info
        if profile:
            rows.append(("Task Type", profile.get("task_type", "")))
            rows.append(("Capability Tags", str(profile.get("capability_tags", []))))
            rows.append(("Instruction Placement", profile.get("instruction_placement", "")))
            answer_labels = profile.get("answer_labels", [])
            if answer_labels:
                label_names = [l.get("name", "") for l in answer_labels]
                rows.append(("Answer Labels", ", ".join(label_names)))

        table = _table(rows)

        # Show prompt template if available
        template = profile.get("prompt_template", "")
        template_section = ""
        if template:
            template_section = _details(
                "Prompt Template",
                _pre(template),
                css_class="prompt-section",
            )

        return _details(
            "Stage 1: Adapt",
            table + template_section,
            css_class="stage-section",
        )

    def _render_decompose(self, stage: StageTrace) -> str:
        """Render decompose stage."""
        parts = []
        for call in stage.llm_calls:
            parts.append(_details("Prompt Sent", _pre(call.user_prompt), css_class="prompt-section"))
            parts.append(_details("Raw LLM Response", _pre(call.response), css_class="response-section"))
            parts.append(f'<p class="meta">Model: {_esc(call.model)} | Temp: {call.temperature}</p>')

        # Parsed output
        outputs = stage.outputs
        if outputs:
            sketch = outputs.get("solution_sketch", "")
            if sketch:
                parts.append(f"<p><strong>Solution Sketch:</strong> {_esc(sketch)}</p>")
            elements = outputs.get("elements", [])
            if elements:
                elem_rows = []
                for e in elements:
                    if isinstance(e, dict):
                        elem_rows.append((
                            e.get("element_type", ""),
                            e.get("description", ""),
                            (e.get("text_span") or "")[:100],
                        ))
                if elem_rows:
                    header = "<tr><th>Type</th><th>Description</th><th>Text Span</th></tr>"
                    body_rows = "".join(
                        f"<tr><td>{_esc(r[0])}</td><td>{_esc(r[1])}</td><td>{_esc(r[2])}</td></tr>"
                        for r in elem_rows
                    )
                    parts.append(f'<table class="data-table">{header}{body_rows}</table>')

        return _details("Stage 2: Decompose", "\n".join(parts), css_class="stage-section")

    def _render_generate(self, stage: StageTrace) -> str:
        """Render generate stage, grouped by category."""
        category_order = ["flip_inducing", "non_flip", "boundary"]
        parts = []

        for call in stage.llm_calls:
            category = call.label
            cat_parts = []
            cat_parts.append(_details("System Prompt", _pre(call.system_prompt), css_class="prompt-section"))
            cat_parts.append(_details("User Prompt", _pre(call.user_prompt), css_class="prompt-section"))
            cat_parts.append(_details("Raw Response", _pre(call.response), css_class="response-section"))
            cat_parts.append(f'<p class="meta">Model: {_esc(call.model)} | Temp: {call.temperature}</p>')

            # Parsed candidates
            candidates = stage.outputs.get(category, [])
            if candidates:
                cand_parts = []
                for i, c in enumerate(candidates):
                    if not isinstance(c, dict):
                        continue
                    ptype = c.get('perturbation_type', '?')
                    mechanism = c.get('mechanism_name', '?')
                    axis = c.get('target_label_axis', '?')

                    if ptype == "problem_edit":
                        # Show edits for problem_edit candidates
                        edits = c.get('problem_edits', [])
                        edit_strs = []
                        for e in edits:
                            if isinstance(e, dict):
                                orig = e.get('original', '')[:80]
                                repl = e.get('replacement', '')[:80]
                                field = e.get('field', '?')
                                edit_strs.append(
                                    f'<br>&nbsp;&nbsp;{_esc(field)}: '
                                    f'<span class="word-del">{_esc(orig)}</span> → '
                                    f'<span class="word-add">{_esc(repl)}</span>'
                                )
                        edits_html = "".join(edit_strs) if edit_strs else "(no edits)"
                        cand_parts.append(
                            f"<li><strong>{_esc(mechanism)}</strong> "
                            f"[{_esc(ptype)}] "
                            f"target: {_esc(axis)}"
                            f"{edits_html}</li>"
                        )
                    else:
                        # Show lever text for instruction_add candidates
                        lever = c.get('lever', '')[:200]
                        cand_parts.append(
                            f"<li><strong>{_esc(mechanism)}</strong> "
                            f"[{_esc(ptype)}] "
                            f"target: {_esc(axis)} — "
                            f"{_esc(lever)}</li>"
                        )
                cat_parts.append(f'<ul class="candidate-list">{"".join(cand_parts)}</ul>')

            color = _CATEGORY_COLORS.get(category, "#666")
            parts.append(_details(
                f'<span style="color:{color}">{_esc(category)}</span> '
                f'({len(candidates)} candidates)',
                "\n".join(cat_parts),
                css_class="category-section",
            ))

        return _details("Stage 3: Generate", "\n".join(parts), css_class="stage-section")

    def _render_candidates(self, trace: ExampleTrace) -> str:
        """Render critic/mini-verify data per candidate."""
        parts = []
        for cid, ct in trace.candidates.items():
            cand_parts = []
            color = _CATEGORY_COLORS.get(ct.category, "#666")

            # Candidate metadata
            cand_parts.append(
                f'<p><strong>Category:</strong> <span style="color:{color}">{_esc(ct.category)}</span> | '
                f'<strong>Mechanism:</strong> {_esc(ct.mechanism_name)} | '
                f'<strong>Type:</strong> {_esc(ct.perturbation_type)} | '
                f'<strong>Target:</strong> {_esc(ct.target_label_axis)}</p>'
            )
            cand_parts.append(f'<p><strong>Lever:</strong> {_esc(ct.lever_text[:300])}</p>')

            # Critic
            if ct.critic_prompt:
                cand_parts.append(_details("Critic Prompt", _pre(ct.critic_prompt), css_class="prompt-section"))
                cand_parts.append(_details("Critic Response", _pre(ct.critic_response), css_class="response-section"))
                cand_parts.append(
                    f'<p><strong>Predicted Flip:</strong> {ct.predicted_flip} | '
                    f'<strong>Notes:</strong> {_esc(ct.critic_notes[:200])}</p>'
                )

            # Mini-verify
            if ct.mini_verify_flip_result:
                cand_parts.append(_details(
                    "Baseline Prompt (Phase 1)", _pre(ct.baseline_prompt), css_class="prompt-section",
                ))
                cand_parts.append(_details(
                    "Lever Prompt (Phase 1)", _pre(ct.lever_prompt), css_class="prompt-section",
                ))
                cand_parts.append(_details(
                    "Baseline vs Lever Diff",
                    _render_diff(ct.baseline_prompt, ct.lever_prompt),
                    css_class="diff-section",
                ))
                cand_parts.append(_details(
                    "Mini-Verify: Baseline Response", _pre(ct.mini_verify_baseline_resp), css_class="response-section",
                ))
                cand_parts.append(_details(
                    "Mini-Verify: Lever Response", _pre(ct.mini_verify_lever_resp), css_class="response-section",
                ))
                flip_res = ct.mini_verify_flip_result
                flip_icon = "FLIP" if flip_res.get("flipped") else "NO FLIP"
                flip_color = "#dc3545" if flip_res.get("flipped") else "#28a745"
                cand_parts.append(
                    f'<p class="flip-result" style="color:{flip_color}">'
                    f'<strong>{flip_icon}</strong> '
                    f'(fraction: {flip_res.get("flip_fraction", "?")})</p>'
                )

            # Full verify
            if ct.verify_flip_fraction is not None:
                for i, (b_resp, l_resp) in enumerate(
                    zip(ct.verify_baseline_resps, ct.verify_lever_resps)
                ):
                    cand_parts.append(_details(
                        f"Verify Run {i+1}: Baseline",
                        _pre(b_resp),
                        css_class="response-section",
                    ))
                    cand_parts.append(_details(
                        f"Verify Run {i+1}: Lever",
                        _pre(l_resp),
                        css_class="response-section",
                    ))
                flip_icon = "FLIP" if ct.verify_flipped else "NO FLIP"
                flip_color = "#dc3545" if ct.verify_flipped else "#28a745"
                cand_parts.append(
                    f'<p class="flip-result" style="color:{flip_color}">'
                    f'<strong>{flip_icon}</strong> '
                    f'(fraction: {ct.verify_flip_fraction}, '
                    f'{len(ct.verify_baseline_resps)} runs)</p>'
                )

            parts.append(_details(
                f'{_esc(cid[:12])}... [{_esc(ct.category)}] {_esc(ct.mechanism_name)}',
                "\n".join(cand_parts),
                css_class="candidate-detail",
            ))

        return _details(
            f"Stage 4-6: Candidates ({len(trace.candidates)} total)",
            "\n".join(parts),
            css_class="stage-section",
        )

    def _render_feedback(self, rounds: list[FeedbackRoundTrace]) -> str:
        """Render feedback loop rounds."""
        parts = []
        for rnd in rounds:
            rnd_parts = []
            rnd_parts.append(
                f'<p><strong>Category:</strong> {_esc(rnd.category)} | '
                f'<strong>Mismatches:</strong> {len(rnd.mismatches)}</p>'
            )
            if rnd.mismatches:
                mismatch_items = []
                for m in rnd.mismatches:
                    mismatch_items.append(f"<li>{_esc(str(m))}</li>")
                rnd_parts.append(_details(
                    f"Mismatches ({len(rnd.mismatches)})",
                    f'<ul>{"".join(mismatch_items)}</ul>',
                    css_class="prompt-section",
                ))

            rnd_parts.append(_details(
                "Feedback Message Sent",
                _pre(rnd.feedback_message),
                css_class="prompt-section",
            ))

            if rnd.new_candidates:
                rnd_parts.append(
                    f"<p><strong>Regenerated:</strong> {len(rnd.new_candidates)} candidates</p>"
                )

            parts.append(_details(
                f"Round {rnd.round_num} ({_esc(rnd.category)})",
                "\n".join(rnd_parts),
                css_class="round-section",
            ))

        return _details(
            f"Feedback Loop ({len(rounds)} rounds)",
            "\n".join(parts),
            css_class="stage-section",
        )

    def _render_export(self, candidates: list[CandidateTrace]) -> str:
        """Render export stage with diff visualization."""
        parts = []
        for ct in candidates:
            exp_parts = []
            ef = ct.export_fields

            # Key fields table
            key_fields = [
                ("unique_id", ef.get("unique_id", "")),
                ("perturbation_id", ef.get("perturbation_id", ct.candidate_id)),
                ("category", ef.get("category", "")),
                ("mechanism_name", ef.get("mechanism_name", "")),
                ("perturbation_type", ef.get("perturbation_type", "")),
                ("target_label_axis", ef.get("target_label_axis", "")),
                ("empirical_flipped", str(ef.get("empirical_flipped", ""))),
                ("empirical_flip_fraction", str(ef.get("empirical_flip_fraction", ""))),
                ("predicted_flip_probability", str(ef.get("predicted_flip_probability", ""))),
                ("edit_distance", str(ef.get("edit_distance", ""))),
                ("edit_fraction", str(ef.get("edit_fraction", ""))),
            ]
            # Contrastive pair info (always show for cross-referencing)
            key_fields.extend([
                ("contrastive_pair_id", ef.get("contrastive_pair_id") or "(none)"),
                ("contrastive_role", ef.get("contrastive_role") or "(none)"),
                ("contrastive_source_id", ef.get("contrastive_source_id") or "(none)"),
            ])
            exp_parts.append(_table(key_fields))

            # Phase 1 lever prompt
            lever_prompt = ef.get("prompt_lever", ct.lever_prompt)
            baseline_prompt = ef.get("prompt_baseline", ct.baseline_prompt)
            if lever_prompt:
                exp_parts.append(_details(
                    "Phase 1 Lever Prompt (full)",
                    _pre(lever_prompt),
                    css_class="prompt-section",
                ))
            if baseline_prompt and lever_prompt:
                exp_parts.append(_details(
                    "Baseline vs Lever Diff",
                    _render_diff(baseline_prompt, lever_prompt),
                    css_class="diff-section",
                    open_default=True,
                ))

            color = _CATEGORY_COLORS.get(ct.category, "#666")
            # Contrastive badge
            contrastive_badge = ""
            if ef.get("contrastive_pair_id"):
                role = ef.get("contrastive_role", "?")
                pid = ef.get("contrastive_pair_id", "")[:8]
                contrastive_badge = (
                    f' <span style="background:#6f42c1;color:white;'
                    f'padding:2px 6px;border-radius:3px;font-size:12px">'
                    f'CONTRASTIVE: {_esc(role)} (pair {_esc(pid)})</span>'
                )
            parts.append(_details(
                f'<span style="color:{color}">{_esc(ct.category)}</span> — '
                f'{_esc(ct.mechanism_name)} [{_esc(ct.candidate_id[:12])}...]'
                f'{contrastive_badge}',
                "\n".join(exp_parts),
                css_class="export-candidate",
            ))

        # Separate count for contrastive
        n_contrastive = sum(1 for ct in candidates if ct.export_fields.get("contrastive_pair_id"))
        label = f"Stage 7: Export ({len(candidates)} candidates"
        if n_contrastive:
            label += f", {n_contrastive} contrastive"
        label += ")"

        return _details(
            label,
            "\n".join(parts),
            css_class="stage-section",
            open_default=True,
        )

    def _render_dataset_summary(self) -> str:
        """Render dataset-level summary of empirical flip distribution.

        Shows stats across ALL exported examples (not just traced ones).
        """
        exports = self._all_exports
        if not exports:
            return ""

        total = len(exports)

        # By category
        by_cat: dict[str, list[dict]] = {}
        for e in exports:
            cat = e.get("category", "unknown")
            by_cat.setdefault(cat, []).append(e)

        # By target axis
        by_axis: dict[str, list[dict]] = {}
        for e in exports:
            axis = e.get("target_label_axis", "") or "(empty)"
            by_axis.setdefault(axis, []).append(e)

        # Contrastive pairs
        n_contrastive = sum(1 for e in exports if e.get("contrastive_pair_id"))
        n_pairs = len(set(e.get("contrastive_pair_id") for e in exports
                         if e.get("contrastive_pair_id")))

        # Build tables
        parts = []

        # Overall stats
        n_flipped = sum(1 for e in exports if e.get("empirical_flipped") is True)
        n_not_flipped = sum(1 for e in exports if e.get("empirical_flipped") is False)
        n_unknown = total - n_flipped - n_not_flipped
        parts.append(_table([
            ("Total examples", str(total)),
            ("Empirical flipped", f"{n_flipped} ({n_flipped/total*100:.0f}%)" if total else "0"),
            ("Empirical not flipped", f"{n_not_flipped} ({n_not_flipped/total*100:.0f}%)" if total else "0"),
            ("Unknown/unverified", str(n_unknown)),
            ("Contrastive pairs", f"{n_pairs} pairs ({n_contrastive} entries)"),
        ]))

        # By category table
        cat_header = "<tr><th>Category</th><th>Count</th><th>Flipped</th><th>Not Flipped</th><th>Flip Rate</th></tr>"
        cat_rows = []
        for cat in ["flip_inducing", "non_flip", "boundary"]:
            items = by_cat.get(cat, [])
            n = len(items)
            flipped = sum(1 for e in items if e.get("empirical_flipped") is True)
            not_flipped = sum(1 for e in items if e.get("empirical_flipped") is False)
            rate = f"{flipped/(flipped+not_flipped)*100:.0f}%" if (flipped + not_flipped) > 0 else "N/A"
            color = _CATEGORY_COLORS.get(cat, "#666")
            cat_rows.append(
                f'<tr><td style="color:{color}"><strong>{_esc(cat)}</strong></td>'
                f'<td>{n}</td><td>{flipped}</td><td>{not_flipped}</td><td>{rate}</td></tr>'
            )
        parts.append(f'<table class="data-table">{cat_header}{"".join(cat_rows)}</table>')

        # By target axis table
        axis_header = "<tr><th>Target Axis</th><th>Count</th><th>Flipped</th><th>Not Flipped</th><th>Flip Rate</th></tr>"
        axis_rows = []
        for axis in sorted(by_axis.keys()):
            items = by_axis[axis]
            n = len(items)
            flipped = sum(1 for e in items if e.get("empirical_flipped") is True)
            not_flipped = sum(1 for e in items if e.get("empirical_flipped") is False)
            rate = f"{flipped/(flipped+not_flipped)*100:.0f}%" if (flipped + not_flipped) > 0 else "N/A"
            axis_rows.append(
                f'<tr><td><strong>{_esc(axis)}</strong></td>'
                f'<td>{n}</td><td>{flipped}</td><td>{not_flipped}</td><td>{rate}</td></tr>'
            )
        parts.append(f'<table class="data-table">{axis_header}{"".join(axis_rows)}</table>')

        # Mechanism distribution (top 10)
        mech_counts: dict[str, int] = {}
        for e in exports:
            m = e.get("mechanism_name", "unknown")
            mech_counts[m] = mech_counts.get(m, 0) + 1
        sorted_mechs = sorted(mech_counts.items(), key=lambda x: -x[1])[:10]
        if sorted_mechs:
            mech_header = "<tr><th>Mechanism</th><th>Count</th></tr>"
            mech_rows = "".join(
                f"<tr><td>{_esc(m)}</td><td>{c}</td></tr>" for m, c in sorted_mechs
            )
            parts.append(_details(
                f"Top mechanisms ({len(mech_counts)} total)",
                f'<table class="data-table">{mech_header}{mech_rows}</table>',
            ))

        # Contrastive pairs detail
        if n_pairs > 0:
            # Group by pair_id
            pairs_by_id: dict[str, list[dict]] = {}
            for e in exports:
                pid = e.get("contrastive_pair_id")
                if pid:
                    pairs_by_id.setdefault(pid, []).append(e)

            pair_parts = []
            pair_header = (
                "<tr><th>Pair ID</th><th>Role</th><th>Category</th>"
                "<th>Mechanism</th><th>Flipped</th><th>Lever / Edits</th></tr>"
            )
            pair_rows = []
            for pid, members in sorted(pairs_by_id.items()):
                for m in sorted(members, key=lambda x: x.get("contrastive_role", "")):
                    role = m.get("contrastive_role", "?")
                    cat = m.get("category", "?")
                    mech = m.get("mechanism_name", "?")
                    flipped = m.get("empirical_flipped")
                    flip_str = "FLIP" if flipped else ("NO FLIP" if flipped is False else "?")
                    flip_color = "#dc3545" if flipped else "#28a745"

                    # Show perturbation summary
                    if m.get("perturbation_type") == "problem_edit":
                        edits = m.get("problem_edits", [])
                        lever_str = "; ".join(
                            f'{e.get("original", "")[:30]}→{e.get("replacement", "")[:30]}'
                            for e in edits[:2]
                        ) if edits else "(edits)"
                    else:
                        lever_str = (m.get("lever_text") or "")[:80]

                    color = _CATEGORY_COLORS.get(cat, "#666")
                    pair_rows.append(
                        f'<tr><td>{_esc(pid[:8])}</td>'
                        f'<td><strong>{_esc(role)}</strong></td>'
                        f'<td style="color:{color}">{_esc(cat)}</td>'
                        f'<td>{_esc(mech)}</td>'
                        f'<td style="color:{flip_color}"><strong>{flip_str}</strong></td>'
                        f'<td>{_esc(lever_str)}</td></tr>'
                    )

            parts.append(_details(
                f"Contrastive Pairs ({n_pairs} pairs)",
                f'<table class="data-table">{pair_header}{"".join(pair_rows)}</table>',
                open_default=True,
            ))

        return _details(
            f"Dataset Summary ({total} examples)",
            "\n".join(parts),
            css_class="example-section",
            open_default=True,
        )

# HTML helpers (_esc, _pre, _details, _table, _render_diff, etc.) and
# HTML_TEMPLATE are imported from html_helpers.py at the top of this file.
