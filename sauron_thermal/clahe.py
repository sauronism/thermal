"""
CLAHE (Contrast Limited Adaptive Histogram Equalization) is a technique for
"""

from typing import NamedTuple

import cv2
import numpy as np
from jaxtyping import Float32

from sauron_thermal.filter import Filter
from sauron_thermal.to_float import ToFloat32, ToU16


class ClaheConfig(NamedTuple):
    """CLAHE (Contrast Limited Adaptive Histogram Equalization) configuration."""
    clip_limit: float
    tile_grid_size: tuple[int, int]


class CLAHE(Filter):
    def __init__(
            self,
            config: ClaheConfig,
    ):
        """CLAHE (Contrast Limited Adaptive Histogram Equalization) module."""
        self.config = config
        self.to_u16 = ToU16()
        self.to_float = ToFloat32()
        self.clahe = cv2.createCLAHE(
            clipLimit=self.config.clip_limit,
            tileGridSize=self.config.tile_grid_size,
        )

    def process(self, x: Float32[np.ndarray, 'h w c']) -> Float32[np.ndarray, 'h w c']:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        x = self.to_u16.process(x)
        x = self.clahe.apply(x)
        x = self.to_float.process(x)
        return x
