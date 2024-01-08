"""AGC (Automatic Gain Control) module."""
from typing import NamedTuple

import numpy as np
from scipy.signal import lfilter
from jaxtyping import Float32

from sauron_thermal.filter import Filter


class AgcConfig(NamedTuple):
    """AGC (Automatic Gain Control) configuration."""
    sample_rate: float
    attack_time: float
    release_time: float
    reference: float
    gain_floor: float
    gain_ceiling: float


class AGC(Filter):
    class _State(NamedTuple):
        gain: float
        envelope: float

    def __init__(
            self,
            config: AgcConfig,
    ):
        """Automatic Gain Control (AGC) module."""
        self.config = config

        self._state = AGC._State(
            gain=1.0,
            envelope=0.0,
        )

        self._attack_coeff = np.exp(-1.0 / (self.config.attack_time * self.config.sample_rate))
        self._release_coeff = np.exp(-1.0 / (self.config.release_time * self.config.sample_rate))

    def process(self, _input: Float32[np.ndarray, '...']) -> Float32[np.ndarray, '...']:
        """Process input signal."""
        envelope = np.abs(_input)
        gain = np.where(
            envelope > self.config.reference,
            self._attack_coeff,
            self._release_coeff,
        )
        gain = lfilter([gain], [1.0, -gain], envelope)
        gain = np.clip(gain, self.config.gain_floor, self.config.gain_ceiling)
        output = _input * gain
        self._state = AGC._State(
            gain=gain[-1],
            envelope=envelope[-1],
        )
        return output
