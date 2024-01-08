"""
Display module for Sauron Thermal
"""
from __future__ import annotations

import asyncio
import threading
from typing import NamedTuple, Self
from concurrent.futures import ThreadPoolExecutor

import cv2 as cv
import numpy as np

from sauron_thermal.to_float import _u16_max


class DisplayQuit(Exception):
    """Display quit exception."""


class DisplayConfig(NamedTuple):
    """Display configuration."""
    width: int
    height: int
    title: str
    key_sample_duration_ms: int = 5


class Display:
    def __init__(
            self,
            config: DisplayConfig,
    ):
        """Display module."""
        self.config = config
        self._render_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video_render")
        self._render_thread_id = self._render_executor.submit(lambda: threading.get_ident()).result()
        self._key_loop = threading.Thread(target=self._key_loop, name="key_loop", daemon=True)
        self._quit = threading.Event()

    def _sample_key(self):
        """Sample key."""
        if threading.get_ident() != self._render_thread_id:
            raise RuntimeError("Must be called from render thread.")
        return cv.waitKey(self.config.key_sample_duration_ms)

    def _key_loop(self):
        """Key loop."""
        while self._quit.is_set():
            fut = self._render_executor.submit(self._sample_key)
            key = fut.result()
            if key == ord("q"):
                self._quit.set()

    def __enter__(self):
        """Enter context."""
        return self._render_executor.submit(self._open).result()

    def _open(self: Self) -> Display:
        """Open display."""
        if threading.get_ident() != self._render_thread_id:
            raise RuntimeError("Must be called from render thread.")
        cv.namedWindow(self.config.title, cv.WINDOW_NORMAL)
        cv.resizeWindow(self.config.title, self.config.width, self.config.height)
        self._key_loop.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit context."""
        self._render_executor.submit(self._close).result()

    def _close(self: Self):
        """Close display."""
        if threading.get_ident() != self._render_thread_id:
            raise RuntimeError("Must be called from render thread.")
        cv.destroyAllWindows()
        self._key_loop.join()

    def _draw(self, x: np.ndarray):
        """Draw input."""
        if threading.get_ident() != self._render_thread_id:
            raise RuntimeError("Must be called from render thread.")
        if self._quit.is_set():
            raise DisplayQuit()
        x = self._normalize_for_display(x)
        cv.imshow(self.config.title, x)

    def _normalize_for_display(self, x: np.ndarray):
        """Normalize input for display."""
        if threading.get_ident() != self._render_thread_id:
            raise RuntimeError("Must be called from render thread.")
        match x.dtype:
            case np.float32:
                x = x * 255.0
                x = np.clip(x, 0, 255)
                x = x.astype(np.uint8)
            case np.uint16:
                x = x.astype(np.float32) / _u16_max()
                x = x * 255.0
                x = np.clip(x, 0, 255)
                x = x.astype(np.uint8)
            case _:
                raise TypeError(f"Unsupported dtype: {x.dtype}")
        return x

    def render(self, _input: np.ndarray):
        """Render input."""
        fut = self._render_executor.submit(self._draw, _input)
        fut.result()

    async def arender(self, x: np.ndarray):
        """Async render input."""
        await asyncio.get_running_loop().run_in_executor(
            self._render_executor,
            self._draw,
            x,
        )
