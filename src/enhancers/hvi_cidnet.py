"""
HVI-CIDNet enhancer wrapper.

Uses the "Generalization" pretrained weights (cross-dataset, random gamma augment).
Repo: https://github.com/Fediory/HVI-CIDNet
Weights: https://huggingface.co/Fediory/HVI-CIDNet-Generalization

Model is fully convolutional — works at any resolution.
"""

import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Optional

from src.enhancers.base import BaseEnhancer


class HVICIDNetEnhancer(BaseEnhancer):
    """HVI-CIDNet Low-Light Image Enhancement wrapper."""

    def __init__(self, cache_dir: str = "cache/HVI-CIDNet"):
        super().__init__(name="HVI-CIDNet")
        self.cache_dir = cache_dir
        self.repo_dir = os.path.join(cache_dir, "repo")
        self.weight_path = None

    def _clone_repo(self) -> None:
        """Clone HVI-CIDNet repo if not exists."""
        if os.path.exists(os.path.join(self.repo_dir, "net")):
            print(f"[HVI-CIDNet] Repo already exists: {self.repo_dir}")
            return

        os.makedirs(self.cache_dir, exist_ok=True)
        print("[HVI-CIDNet] Cloning repository...")
        ret = os.system(f"git clone https://github.com/Fediory/HVI-CIDNet.git {self.repo_dir}")
        if ret != 0:
            raise RuntimeError("Failed to clone HVI-CIDNet repository")
        print("[HVI-CIDNet] Repository cloned successfully")

    def _download_weights(self) -> str:
        """Download Generalization weights from HuggingFace."""
        weights_dir = os.path.join(self.cache_dir, "weights")
        weight_file = os.path.join(weights_dir, "net_rgb.pth")

        if os.path.exists(weight_file):
            print(f"[HVI-CIDNet] Weights already exist: {weight_file}")
            return weight_file

        os.makedirs(weights_dir, exist_ok=True)
        print("[HVI-CIDNet] Downloading Generalization weights from HuggingFace...")

        try:
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(
                repo_id="Fediory/HVI-CIDNet-Generalization",
                filename="net_rgb.pth",
                local_dir=weights_dir,
            )
            print(f"[HVI-CIDNet] Weights downloaded: {downloaded}")
            return weight_file
        except ImportError:
            # Fallback: direct URL download
            url = "https://huggingface.co/Fediory/HVI-CIDNet-Generalization/resolve/main/net_rgb.pth"
            import urllib.request
            print(f"[HVI-CIDNet] Downloading from: {url}")
            urllib.request.urlretrieve(url, weight_file)
            print(f"[HVI-CIDNet] Weights saved: {weight_file}")
            return weight_file

    def load_model(self, device: str = "cuda") -> None:
        """Load HVI-CIDNet model with Generalization weights.

        Args:
            device: 'cuda' or 'cpu'
        """
        self.device = device

        # Clone repo and download weights
        self._clone_repo()
        self.weight_path = self._download_weights()

        # Add repo to Python path so we can import its modules
        if self.repo_dir not in sys.path:
            sys.path.insert(0, self.repo_dir)

        # Import model architecture from the repo
        try:
            from net.CIDNet import CIDNet
        except ImportError as e:
            raise ImportError(
                f"Failed to import CIDNet from {self.repo_dir}/net/CIDNet.py. "
                f"Make sure the repo is cloned correctly. Error: {e}"
            )

        # Initialize model
        self.model = CIDNet()
        checkpoint = torch.load(self.weight_path, map_location=device, weights_only=False)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(device)
        self.model.eval()
        self._loaded = True
        print(f"[HVI-CIDNet] Model loaded on {device}")

    @torch.no_grad()
    def enhance(self, img_bgr: np.ndarray) -> np.ndarray:
        """Enhance a single low-light image.

        Args:
            img_bgr: BGR uint8 (H, W, 3)

        Returns:
            Enhanced BGR uint8 (H, W, 3) — same size
        """
        assert self._loaded, "Model not loaded. Call load_model() first."

        h, w = img_bgr.shape[:2]

        # BGR → RGB, normalize to [0, 1], to tensor (1, 3, H, W)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(self.device)

        # Pad to multiple of 4 (some conv architectures require this)
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        if pad_h > 0 or pad_w > 0:
            img_tensor = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')

        # Inference
        output = self.model(img_tensor)

        # Handle different output formats
        if isinstance(output, (tuple, list)):
            output = output[0]

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            output = output[:, :, :h, :w]

        # Tensor → numpy BGR uint8
        output = output.squeeze(0).clamp(0, 1).cpu().numpy()
        output = (output * 255).astype(np.uint8)
        output = np.transpose(output, (1, 2, 0))  # (3, H, W) → (H, W, 3)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        return output
