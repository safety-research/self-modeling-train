"""
Module: prompt_attribution/auto_perturbation/decomposer.py

Stage 2: Generic problem decomposition. Analyzes problems to identify
structural elements that perturbations could target.

Uses a single domain-agnostic prompt with 6 generic element types.

Structure:
- GENERIC_DECOMPOSER_PROMPT: The LLM prompt template
- ProblemDecomposer: Orchestrates decomposition for a batch of problems
"""

import asyncio
import json
import logging

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt

from ..config import PipelineConfig, ProblemAnalysis, StructuralElement
from ..dataset_adapter.dataset_adapter import DatasetAdapter, AdaptedExample

logger = logging.getLogger(__name__)


# =============================================================================
# Prompt
# =============================================================================


GENERIC_DECOMPOSER_PROMPT = """\
You are analyzing a problem to identify structural elements that a perturbation \
could target. Your job is to map out the problem's structure — what information \
it contains, what it assumes, and where the reasoning chain is fragile.

Do NOT suggest specific perturbations — that is a separate step. Just identify \
the elements and explain why each matters for the answer.

## Task Type: {task_type}
## Answer Format: {answer_format}

## Element Types (use exactly these types)
- **content**: A key fact, value, entity, or piece of information in the problem
- **format**: How the answer is requested or formatted (MCQ letter, number, text)
- **context**: Background, framing, or scenario information
- **assumption**: Something the problem takes for granted
- **constraint**: An explicit condition, requirement, or rule
- **implicit_premise**: Something unstated but assumed

## Guidelines
- Identify 5-8 structural elements
- For each element, explain WHY it matters for the answer (how the reasoning \
chain depends on it)
- Focus on elements where the effect on the answer is UNCERTAIN — not \
elements where any change would trivially change the answer
- Include at least one "implicit_premise" — something the model assumes \
that isn't explicitly stated

## Problem
{problem_text}

## Correct Answer
{ground_truth_answer}

## Output Format
Output ONLY a valid JSON object:
{{
  "solution_sketch": "Step-by-step reasoning chain. For each step, note what it depends on and where it could go wrong. Example: '(1) Parse nested comparison to get base amounts [fragile: could misparse scope of clause]. (2) Add delta to base [fragile: could treat delta as standalone value]. (3) Sum all amounts [fragile: could miss one person].'",
  "elements": [
    {{
      "element_type": "content|format|context|assumption|constraint|implicit_premise",
      "description": "What this element is and why it matters for the answer",
      "text_span": "Relevant text from the problem (exact quote if possible)"
    }}
  ]
}}"""


# =============================================================================
# Decomposer
# =============================================================================


class ProblemDecomposer:
    """Stage 2: Generic problem decomposition.

    Analyzes each problem to identify structural elements and their
    perturbation lever axes. Uses a single domain-agnostic prompt.
    """

    def __init__(
        self,
        api: InferenceAPI,
        config: PipelineConfig,
        adapter: DatasetAdapter,
        prompt_logger=None,
        tracer=None,
    ):
        self.api = api
        self.config = config
        self.adapter = adapter
        self._prompt_logger = prompt_logger
        self._tracer = tracer

    async def decompose(self, example: AdaptedExample) -> ProblemAnalysis:
        """Decompose a single problem into structural elements.

        Args:
            example: The adapted example to analyze

        Returns:
            ProblemAnalysis with identified elements
        """
        # Get task type info from adapter
        task_type = self.adapter.profile.task_type
        answer_format = self.adapter.profile.answer_extraction
        answer_labels = getattr(self.adapter.profile, 'answer_labels', None) or []

        # Build problem content — show with {instruction} placeholder
        # since the actual task framing varies per axis.
        problem_text = self._build_problem_content(example)

        # Escape curly braces in user content to prevent KeyError
        safe_problem = problem_text.replace("{", "{{").replace("}", "}}")
        safe_answer = (example.ground_truth_answer or "(not available)").replace("{", "{{").replace("}", "}}")

        # Build axes section — shows each axis's task framing + response format
        axes_section = self._build_axes_section(answer_labels, safe_answer)

        if axes_section:
            # Replace "## Correct Answer" with the axes section directly
            ground_truth_block = axes_section
        else:
            ground_truth_block = safe_answer

        prompt_text = GENERIC_DECOMPOSER_PROMPT.format(
            task_type=task_type,
            answer_format=answer_format,
            problem_text=safe_problem,
            ground_truth_answer=ground_truth_block,
        )

        # Fallback: append old label_descriptions if no axes
        if not answer_labels:
            label_desc = getattr(self.adapter.profile, 'label_descriptions', '')
            if label_desc:
                prompt_text += f"\n\n## {label_desc}"

        # Call LLM with retry on transient errors (cache race conditions, etc.)
        from ..utils.retry import retry_async

        async def _do_decompose():
            responses = await self.api(
                model_id=self.config.generator_model,
                prompt=Prompt(messages=[
                    ChatMessage(role=MessageRole.user, content=prompt_text),
                ]),
                n=1,
                temperature=0.3,
                max_tokens=4096,
            )
            return responses[0].completion if responses else ""

        try:
            response_text = await retry_async(
                _do_decompose,
                stage_name="decomposer",
                item_id=f"example_{example.idx}",
                api=self.api,
            )

            # Log prompt + response
            if self._prompt_logger:
                self._prompt_logger.log(
                    component="decomposer",
                    label=f"problem_{example.idx}",
                    user_prompt=prompt_text,
                    response=response_text,
                    extra={"model": self.config.generator_model, "temperature": 0.3},
                )

            # Parse JSON response
            analysis = self._parse_response(
                response_text, example, problem_text, task_type,
            )

            # Record in tracer
            if self._tracer and self._tracer.is_traced(example.idx):
                self._tracer.record_decompose(
                    example_idx=example.idx,
                    prompt=prompt_text,
                    response=response_text,
                    parsed_analysis={
                        "solution_sketch": analysis.solution_sketch,
                        "elements": [e.to_dict() for e in analysis.elements],
                    },
                    model=self.config.generator_model,
                    temperature=0.3,
                )

            return analysis

        except Exception as e:
            logger.error(
                f"Decomposition FAILED for example {example.idx} after retries: {e}. "
                f"Using fallback."
            )
            return self._fallback_analysis(example, problem_text, task_type)

    def _build_problem_content(self, example: AdaptedExample) -> str:
        """Build problem text with {instruction} placeholder instead of
        hardcoded preamble/response format.

        The task framing varies per axis, so the decomposer sees the raw
        content structure with a placeholder showing where instructions go.
        """
        import re as _re

        # Get the full prompt (with default preamble + response format)
        full = self.adapter.get_problem_for_attribution(example)

        # Strip the hardcoded response format (axis-specific formats replace it)
        content = _re.sub(
            r'\n*Respond with JSON:.*$', '', full, flags=_re.MULTILINE,
        )
        # Strip boxed format instructions too
        content = _re.sub(
            r'\n*Put your final.*\\boxed\{\}.*$', '', content, flags=_re.MULTILINE,
        )
        content = content.strip()

        # Replace the preamble with {instruction} placeholder
        # The preamble is the first line(s) before the content fields
        profile = self.adapter.profile
        prompt_template = profile.prompt_template
        if prompt_template and "{instruction}" in prompt_template:
            # Find where {instruction} appears in the template to show structure
            content = f"{{instruction}}\n\n{content}"
            # Strip the default preamble (first line) since it's axis-specific
            lines = content.split("\n")
            # Remove the old preamble line if it's a generic "Classify..." or "Answer..."
            if len(lines) > 2:
                first_content = lines[2] if lines[1] == "" else lines[1]
                # Check if line 2+ starts with content markers
                for i, line in enumerate(lines[2:], 2):
                    if line.startswith(("Text:", "Question:", "Problem:", "Context:",
                                        "Review:", "{context}", "{question}")):
                        # Keep from {instruction} and this content line onward
                        content = "{instruction}\n\n" + "\n".join(lines[i:])
                        break

        return content

    def _build_axes_section(
        self, answer_labels: list[dict], default_answer: str,
    ) -> str:
        """Build the axes section showing per-axis task framing + format.

        Replaces the simple "Correct Answer" with a structured view
        showing what each axis measures and how.
        """
        if not answer_labels:
            return ""

        lines = []
        lines.append("## Answer label axes (each axis has its own task framing):")
        lines.append("The {{instruction}} placeholder is replaced by each axis's "
                      "task framing. Each axis asks the model a different question "
                      "about the same problem content.\n")

        first = True
        for label in answer_labels:
            if not isinstance(label, dict):
                continue
            name = label.get("name", "")
            desc = label.get("description", "")
            preamble = label.get("axis_preamble", "")
            fmt = label.get("axis_response_format", "")

            lines.append(f"- **{name}**: {desc}")
            if preamble:
                lines.append(f"  Task framing: \"{preamble}\"")
            if fmt:
                lines.append(f"  Response format: \"{fmt}\"")
            # Show the ground truth answer only for the first (primary) axis
            if first and default_answer:
                lines.append(f"  Ground truth answer: {default_answer}")
                first = False

        lines.append(
            "\nIdentify elements whose modification could affect "
            "different axes in different ways."
        )
        return "\n".join(lines)

    async def decompose_batch(
        self,
        examples: list[AdaptedExample],
    ) -> list[ProblemAnalysis]:
        """Decompose a batch of problems concurrently.

        Args:
            examples: List of adapted examples to analyze

        Returns:
            List of ProblemAnalysis (one per example, in order)
        """
        semaphore = asyncio.Semaphore(self.config.concurrency)

        async def bounded_decompose(ex: AdaptedExample) -> ProblemAnalysis:
            async with semaphore:
                return await self.decompose(ex)

        tasks = [bounded_decompose(ex) for ex in examples]
        results = await asyncio.gather(*tasks)
        return list(results)

    def _parse_response(
        self,
        response_text: str,
        example: AdaptedExample,
        problem_text: str,
        task_type: str,
    ) -> ProblemAnalysis:
        """Parse the LLM's JSON response into a ProblemAnalysis."""
        # Extract JSON from response
        try:
            # Try to find JSON block
            json_str = response_text.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]

            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Try to extract JSON object from response
            import re
            match = re.search(r'\{[\s\S]*\}', response_text)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.warning(
                        f"Could not parse JSON for example {example.idx}"
                    )
                    return self._fallback_analysis(example, problem_text, task_type)
            else:
                return self._fallback_analysis(example, problem_text, task_type)

        # Build ProblemAnalysis from parsed data
        elements = []
        for elem_data in data.get("elements", []):
            elements.append(StructuralElement(
                element_type=elem_data.get("element_type", "content"),
                description=elem_data.get("description", ""),
                text_span=elem_data.get("text_span", ""),
                lever_axes=elem_data.get("lever_axes", []),
            ))

        return ProblemAnalysis(
            example_idx=example.idx,
            question=example.question,
            ground_truth_answer=example.ground_truth_answer,
            elements=elements,
            solution_sketch=data.get("solution_sketch", ""),
            prompt_template=problem_text,
            task_type=task_type,
            dataset_id=self.adapter.profile.dataset_id,
        )

    def _fallback_analysis(
        self,
        example: AdaptedExample,
        problem_text: str,
        task_type: str,
    ) -> ProblemAnalysis:
        """Create a minimal fallback analysis when LLM parsing fails."""
        # Create a basic content element from the question
        elements = [
            StructuralElement(
                element_type="content",
                description="The main question/problem text",
                text_span=example.question[:200],
                lever_axes=["Reframe the question", "Add a constraint"],
            ),
            StructuralElement(
                element_type="format",
                description="The expected answer format",
                text_span=task_type,
                lever_axes=["Change output format requirements"],
            ),
        ]

        return ProblemAnalysis(
            example_idx=example.idx,
            question=example.question,
            ground_truth_answer=example.ground_truth_answer,
            elements=elements,
            solution_sketch="(fallback: decomposition failed)",
            prompt_template=problem_text,
            task_type=task_type,
            dataset_id=self.adapter.profile.dataset_id,
        )
