"""
Module: prompt_attribution/domains/math/verifier.py

Structure:
- MathVerifier: Verifier for math problems using math_verify library
"""

import re
from typing import Any, Optional

from math_verify import parse as math_parse, verify as math_verify

from ..base import BaseVerifier


class MathVerifier(BaseVerifier):
    """Verifier for math problems using math_verify library.

    """

    def parse_answer(
        self, raw_output: str
    ) -> Optional[str]:
        """Extract answer from model output.

        For numeric answers, tries in order:
        1. \\boxed{} extraction with nested brace handling (most reliable)
        2. math_verify library parsing
        3. Last number found in text


        Args:
            raw_output: Full model response text

        Returns:
            Extracted answer as string, or None
        """
        if not raw_output:
            return None


        # 1. Try \boxed{} extraction with nested brace handling
        boxed_answer = self._extract_boxed_answer(raw_output)
        if boxed_answer:
            return self._normalize_number(boxed_answer)

        # 2. Try math_verify parsing
        try:
            parsed = math_parse(raw_output)
            if parsed:
                result = parsed[0] if isinstance(parsed, (list, tuple)) else parsed
                return str(result)
        except Exception:
            pass

        # 3. Fallback: last number in text
        numbers = re.findall(r'[-]?\d+(?:,\d{3})*(?:\.\d+)?', raw_output)
        if numbers:
            return self._normalize_number(numbers[-1])

        return None

    def _extract_boxed_answer(self, text: str) -> Optional[str]:
        """Extract answer from \\boxed{} handling nested braces.

        Args:
            text: Full text containing \\boxed{}

        Returns:
            Content inside \\boxed{} or None if not found
        """
        # Find the last \boxed{ in the text
        matches = list(re.finditer(r'\\boxed\{', text))
        if not matches:
            return None

        # Get the last match (final answer)
        last_match = matches[-1]
        start = last_match.end()

        # Count braces to find matching close brace
        depth = 1
        pos = start
        while pos < len(text) and depth > 0:
            if text[pos] == '{':
                depth += 1
            elif text[pos] == '}':
                depth -= 1
            pos += 1

        if depth == 0:
            return text[start:pos-1]
        return None

    def _normalize_number(self, num_str: str) -> str:
        r"""Normalize number string by stripping LaTeX formatting and units.

        Handles:
        - Currency: \$8 -> 8
        - Units: 8 \text{ m} -> 8, 8 \text{ dollars} -> 8
        - Commas: 1,234 -> 1234
        - Whitespace and common LaTeX commands
        """
        if not num_str:
            return num_str

        result = num_str

        # Remove LaTeX currency symbol
        result = re.sub(r'\\?\$', '', result)

        # Remove \text{...} and its contents (units like "m", "dollars", etc.)
        result = re.sub(r'\\text\s*\{[^}]*\}', '', result)

        # Remove \mathrm{...} and its contents
        result = re.sub(r'\\mathrm\s*\{[^}]*\}', '', result)

        # Remove common unit suffixes (standalone words after number)
        result = re.sub(r'\s+(meters?|m|cm|km|feet|ft|inches?|in|dollars?|cents?|years?|days?|hours?|minutes?|seconds?|kg|g|lbs?|oz)\b', '', result, flags=re.IGNORECASE)

        # Remove commas from numbers like 1,234
        result = result.replace(',', '')

        # Clean up whitespace
        result = result.strip()

        return result

    def _extract_numeric_value(self, text: str) -> Optional[str]:
        """Extract pure numeric value from text, stripping all formatting.

        Args:
            text: Text that may contain a number with formatting

        Returns:
            Extracted numeric string or None
        """
        if not text:
            return None

        # First normalize (remove LaTeX, units, etc.)
        normalized = self._normalize_number(text)

        # Try to extract a number (integer, decimal, negative)
        match = re.search(r'[-]?\d+(?:\.\d+)?', normalized)
        if match:
            return match.group()

        return normalized if normalized else None

    def answers_match(
        self,
        answer1: Optional[str],
        answer2: Optional[str],
    ) -> bool:
        """Compare answers for equivalence.

        For numeric answers, uses math_verify for symbolic equivalence.

        Comparison strategy for numeric:
        1. Try math_verify library for symbolic equivalence
        2. Compare normalized strings (strip LaTeX, units)
        3. Compare extracted numeric values (exact)

        Args:
            answer1: First parsed answer
            answer2: Second parsed answer

        Returns:
            True if answers are equivalent
        """
        if answer1 is None or answer2 is None:
            return answer1 == answer2


        # 1. Try math_verify for mathematical equivalence
        try:
            parsed1 = math_parse(answer1)
            parsed2 = math_parse(answer2)
            if parsed1 and parsed2:
                if math_verify(parsed1, parsed2):
                    return True
        except Exception:
            pass

        # 2. Compare normalized strings
        norm1 = self._normalize_number(answer1)
        norm2 = self._normalize_number(answer2)
        if norm1 and norm2 and norm1 == norm2:
            return True

        # 3. Compare extracted numeric values (exact)
        num1 = self._extract_numeric_value(answer1)
        num2 = self._extract_numeric_value(answer2)
        if num1 and num2:
            try:
                return float(num1) == float(num2)
            except ValueError:
                pass

        return False

    def check_feature(
        self,
        answer: str,
        parsed_answer: Any = None,
        perturbation: Any = None,
        example: Any = None,
        **kwargs: Any,
    ) -> bool:
        """Check if a math-domain feature is present in the answer.

        Supports:
        - answer_equals_target: Check if parsed answer equals perturbation.feature_target_value
        - answer_has_unit: Check if answer text contains the target unit string
        - answer_differs_from_gt: Check if answer differs from example ground truth

        Args:
            answer: Raw response text
            parsed_answer: Parsed answer string (from parse_answer)
            perturbation: PerturbationConfig with target_features and feature_target_value
            example: Math Example with ground_truth_answer

        Returns:
            True if the target feature is present
        """
        if not perturbation or not perturbation.target_features:
            return False

        target = perturbation.target_features[0]
        parsed = parsed_answer if parsed_answer else self.parse_answer(answer)

        if target == "response_exceeds_200_words":
            # Count words in the full raw response
            return len((answer or "").split()) > 200

        elif target == "answer_equals_target":
            target_value = perturbation.feature_target_value
            if not target_value or not parsed:
                return False
            try:
                return float(self._normalize_number(str(parsed))) == float(target_value)
            except (ValueError, TypeError):
                return str(parsed).strip() == str(target_value).strip()

        elif target == "answer_has_unit":
            unit = perturbation.feature_target_value
            if not unit or not answer:
                return False
            return unit.lower() in answer.lower()

        elif target == "answer_differs_from_gt":
            gt_answer = getattr(example, "ground_truth_answer", None) if example else None
            if gt_answer is None or parsed is None:
                return False
            return not self.answers_match(str(parsed), str(gt_answer))

        return False

