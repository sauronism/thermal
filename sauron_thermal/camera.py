"""
Camera module for Sauron Thermal.

Thin wrapper around OpenCV's VideoCapture class.
With some additional functionality.

"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from flirpy.camera.boson import Boson
from jaxtyping import UInt8

from sauron_thermal.filter import Filter


class Camera:
    """Camera module."""

    def __init__(
            self,
            filters: list[Filter] = None,
    ):
        """Camera module."""
        if filters is None:
            filters = []
        self.filters = filters
        self._camera = Boson()
        self._capture_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video_capture")
        self._process_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video_process")


    def _process(self, _input: np.ndarray) -> np.ndarray:
        for f in self.filters:
            _input = f.process(_input)
        return _input

    def _grab(self, camera) -> UInt8[np.ndarray, 'h w c']:
        """Grab frame."""
        x = camera.grab().astype(np.float32)
        x = np.clip(x, 0.0, 1.0)
        x = (x * 255.0).astype(np.uint8)
        return x



    def _frame_generator(self, camera: Boson) -> Iterator[np.ndarray]:
        """Frame generator."""
        while True:
            ret, frame = self._capture_executor.submit(self._grab, camera).result()
            if not ret:
                break
            result = self._process_executor.submit(self._process, frame).result()
            yield result

    async def _aframe_generator(self, camera: Boson) -> AsyncIterator[np.ndarray]:
        """Async frame generator."""
        while True:
            ret, frame = await asyncio.get_running_loop().run_in_executor(
                self._capture_executor,
                self._grab,
                camera,
            )
            if not ret:
                break
            result = await asyncio.get_running_loop().run_in_executor(
                self._process_executor,
                self._process,
                frame,
            )
            yield result

    def __iter__(self):
        with self._camera as camera:
            yield from self._frame_generator(camera)

    async def __aiter__(self):
        with self._camera as camera:
            async for frame in self._aframe_generator(camera):
                yield frame
