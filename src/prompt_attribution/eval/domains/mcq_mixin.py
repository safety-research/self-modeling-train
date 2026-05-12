"""
Module: prompt_attribution/domains/mcq_mixin.py

Structure:
- MCQMixin: Mixin providing 5-option MCQ (A, B, C, D, E) answer parsing
"""

import re
from typing import Optional


class MCQMixin:
    """Mixin providing 5-option MCQ (A, B, C, D, E) answer parsing.

    This mixin can be added to any verifier that needs to support
    multiple-choice question format (e.g., MMLU benchmarks).
    """

    def parse_mcq_answer(self, raw_output: str) -> Optional[str]:
        """Extract selected option (A, B, C, D, or E) from model output.

        Tries patterns in order:
        1. JSON format: {"answer": "X"}
        2. Direct letter at start of response
        3. Letter with punctuation (A., A), (A), etc.)
        4. "Answer: X" pattern
        5. "option X" or "choice X" pattern
        6. "I would choose X" or "I select X" pattern
        7. Letter in parentheses or brackets
        8. First standalone A/B/C/D/E found

        Args:
            raw_output: Full model response text

        Returns:
            Selected option letter (A, B, C, D, or E) or None
        """
        if not raw_output:
            return None

        text = raw_output.strip()
        text_upper = text.upper()

        # 1. JSON format: {"answer": "X"} - most reliable
        match = re.search(r'\{\s*"answer"\s*:\s*"([ABCDE])"\s*\}', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # 2. Direct letter at start (most common for well-behaved models)
        if text_upper and text_upper[0] in "ABCDE":
            # Make sure it's not just part of a word
            if len(text_upper) == 1 or not text_upper[1].isalpha():
                return text_upper[0]

        # 3. Letter with punctuation: "A.", "A)", "(A)", "[A]", "A:"
        match = re.match(r"^\s*[\(\[\s]*([ABCDE])[\)\]\.\:\s]", text_upper)
        if match:
            return match.group(1)

        # 4. "Answer: X" or "answer is X" pattern (use LAST match for final answer)
        matches = re.findall(r"answer[:\s]+(?:is\s+)?([ABCDE])\b", text_upper)
        if matches:
            return matches[-1]

        # 5. "option X" or "choice X" pattern (use LAST match)
        matches = re.findall(r"(?:option|choice)[:\s]+([ABCDE])\b", text_upper)
        if matches:
            return matches[-1]

        # 6. "I would choose X" or "I select X" pattern (use LAST match)
        matches = re.findall(r"(?:choose|select|pick)[:\s]+([ABCDE])\b", text_upper)
        if matches:
            return matches[-1]

        # 7. Look for standalone letter in parentheses or brackets (use LAST match)
        matches = re.findall(r"[\(\[]([ABCDE])[\)\]]", text_upper)
        if matches:
            return matches[-1]

        # 8. Fallback: find LAST standalone A, B, C, D, or E (word boundary)
        matches = re.findall(r"\b([ABCDE])\b", text_upper)
        if matches:
            return matches[-1]

        return None

    def mcq_answers_match(
        self, answer1: Optional[str], answer2: Optional[str]
    ) -> bool:
        """Compare MCQ answers by checking if same option was selected.

        Args:
            answer1: First parsed answer (A, B, C, D, or E)
            answer2: Second parsed answer (A, B, C, D, or E)

        Returns:
            True if same option was selected
        """
        if answer1 is None or answer2 is None:
            return answer1 == answer2

        return answer1.upper() == answer2.upper()

    def format_mcq_answer(
        self, answer: Optional[str], choices: Optional[list[str]] = None
    ) -> str:
        """Format MCQ answer for attribution display.

        Args:
            answer: The parsed answer (A, B, C, D, or E)
            choices: Optional list of choice texts for the selected option

        Returns:
            Formatted answer string like "A" or 'A ("choice text")'
        """
        if answer is None:
            return "(no answer)"

        letter = answer.upper()

        # If we have the choices, include the text
        if choices and letter in "ABCDE":
            idx = ord(letter) - ord("A")
            if idx < len(choices):
                text = choices[idx]
                # Truncate long choice text
                truncated = text[:40] + "..." if len(text) > 40 else text
                return f'{letter} ("{truncated}")'

        return letter
