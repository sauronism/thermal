"""16bit -> 32bit float conversion"""
import functools

import numpy as np
from jaxtyping import Float32, UInt16

from sauron_thermal.filter import Filter


@functools.cache
def _u16_max():
    from ctypes import c_uint16
    u16_max = np.float32(c_uint16(-1).value)
    assert u16_max == 65535
    return u16_max


@functools.cache
def _u16_to_32f_normalization():
    return 1.0 / _u16_max()


class ToFloat32(Filter):
    def process(self, _input: UInt16[np.ndarray, '...']) -> Float32[np.ndarray, '...']:
        return _input.astype(np.float32) * _u16_to_32f_normalization()


class ToU16(Filter):
    def process(self, x: Float32[np.ndarray, '...']) -> UInt16[np.ndarray, '...']:
        x = x * _u16_max()
        x = np.clip(x, 0, _u16_max())
        return x.astype(np.uint16)
