"""Domain primitives used by the dataset adapter.

Only `math` is kept — the adapter routes `task_type == "open_numeric"`
records directly to MathVerifier.
"""

from .base import BaseVerifier
from .math import MathVerifier

__all__ = ["BaseVerifier", "MathVerifier"]
