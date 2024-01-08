from typing import Protocol

import numpy as np


class Filter(Protocol):
    """Filter protocol."""

    def process(self, _input: np.ndarray) -> np.ndarray:
        """Process input signal."""
