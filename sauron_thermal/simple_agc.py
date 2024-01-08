import logging
from typing import NamedTuple

import numpy as np
from jaxtyping import Float32

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
logger.addHandler(handler)

class SimpleAgcConfig(NamedTuple):
    """AGC (Automatic Gain Control) configuration."""
    sample_rate: float = 9.0
    ewma_coeff: float = 0.9
    eps: float = 1e-5


class SimpleAgc:
    class _State(NamedTuple):
        running_min: float
        running_max: float

    def __init__(self, config: SimpleAgcConfig):
        """Automatic Gain Control (AGC) module."""
        self.config = config
        self._state = SimpleAgc._State(
            running_min=0.0,
            running_max=1.0,
        )

    def ewma(self, x, y):
        coeff = self.config.ewma_coeff / self.config.sample_rate
        return x * (1.0 - coeff) + y * coeff

    def process(self, x: Float32[np.ndarray, '...']) -> Float32[np.ndarray, '...']:
        """Process input signal."""
        self._state = SimpleAgc._State(
            running_min=self.ewma(x.min(), self._state.running_min),
            running_max=self.ewma(x.max(), self._state.running_max),
        )
        logger.info(f"running_min={self._state.running_min}, running_max={self._state.running_max}")

        interval = np.maximum(self._state.running_max - self._state.running_min, self.config.eps)

        x = (x + self._state.running_min) / interval
        return x
