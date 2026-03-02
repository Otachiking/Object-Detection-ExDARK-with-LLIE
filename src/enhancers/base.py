"""
Abstract base class for LLIE enhancers.

All enhancers must implement:
- load_model(device) → None
- enhance(img_bgr: np.ndarray) → np.ndarray

Contract:
- Input: BGR uint8 numpy array (H, W, 3)
- Output: BGR uint8 numpy array (H, W, 3) — SAME dimensions as input
- The enhancer must NOT change image dimensions (critical for label alignment)
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional


class BaseEnhancer(ABC):
    """Abstract base class for Low-Light Image Enhancement models."""

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.device = "cpu"
        self._loaded = False

    @abstractmethod
    def load_model(self, device: str = "cuda") -> None:
        """Load pretrained weights and initialize model.

        Args:
            device: 'cuda' or 'cpu'
        """
        pass

    @abstractmethod
    def enhance(self, img_bgr: np.ndarray) -> np.ndarray:
        """Enhance a single image.

        Args:
            img_bgr: BGR uint8 image (H, W, 3)

        Returns:
            Enhanced BGR uint8 image (H, W, 3) — SAME shape as input

        Raises:
            AssertionError if output shape != input shape
        """
        pass

    def enhance_safe(self, img_bgr: np.ndarray) -> np.ndarray:
        """Enhance with safety checks (dimension, dtype, range).

        Use this instead of enhance() for production pipeline.
        """
        assert img_bgr is not None, "Input image is None"
        assert len(img_bgr.shape) == 3, f"Expected 3D array, got shape {img_bgr.shape}"
        assert img_bgr.shape[2] == 3, f"Expected 3 channels, got {img_bgr.shape[2]}"

        input_shape = img_bgr.shape
        result = self.enhance(img_bgr)

        # Critical: output must have same dimensions as input
        assert result.shape == input_shape, (
            f"[{self.name}] Dimension mismatch! "
            f"Input: {input_shape}, Output: {result.shape}. "
            f"Enhancement must NOT change image dimensions."
        )

        # Ensure proper dtype and range
        if result.dtype != np.uint8:
            result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    def enhance_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Enhance a batch of images. Default: sequential loop.

        Override in subclass for batch-optimized inference.
        """
        return [self.enhance_safe(img) for img in images]

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, loaded={self._loaded})"
