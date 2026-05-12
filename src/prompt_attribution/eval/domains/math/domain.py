"""
Module: prompt_attribution/domains/math/domain.py

Structure:
- MathDomain: Domain implementation for math problems
"""


from ..base import BaseDomain
from prompt_attribution.eval.benchmarks.base import Example
from .verifier import MathVerifier


class MathDomain(BaseDomain):
    """Domain implementation for math problems.

    Uses MathVerifier for numerical answer comparison.
    """

    @property
    def name(self) -> str:
        return "math"

    def create_verifier(self) -> MathVerifier:
        """Create MathVerifier instance."""
        return MathVerifier()

    def get_answers_match_kwargs(self, example: Example) -> dict:
        """Get extra kwargs for verifier.answers_match() call.

        """
        kwargs = {}
        return kwargs

