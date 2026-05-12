"""
Module: prompt_attribution/eval/domains/base.py

Structure:
- BaseVerifier: Abstract base class for parse-answer / answers-match verifiers.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseVerifier(ABC):
    """Abstract base class for answer verification."""

    @abstractmethod
    def parse_answer(self, raw_output: str) -> Any:
        """Parse answer from model output."""
        pass

    @abstractmethod
    def answers_match(self, answer1: Any, answer2: Any) -> bool:
        """Compare two answers for equivalence."""
        pass
