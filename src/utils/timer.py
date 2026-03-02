"""CUDA-aware timing utilities."""

import time
from typing import Optional


class CUDATimer:
    """Context manager for CUDA-synchronized timing.

    Usage:
        with CUDATimer("enhancement") as t:
            result = model(input)
        print(f"Took {t.elapsed_ms:.2f} ms")
    """

    def __init__(self, label: str = "", use_cuda: bool = True):
        self.label = label
        self.use_cuda = use_cuda
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self):
        if self.use_cuda:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except ImportError:
                pass
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.use_cuda:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except ImportError:
                pass
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        return False


class Timer:
    """Simple wall-clock timer (no CUDA sync)."""

    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        return False


def cuda_sync():
    """Synchronize CUDA if available."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass
