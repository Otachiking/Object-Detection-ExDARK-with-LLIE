"""
HVI-CIDNet enhancer wrapper.

Uses the LOL-v1 pretrained weights (with perceptual loss).
Repo: https://github.com/Fediory/HVI-CIDNet
Weights: https://huggingface.co/Fediory/HVI-CIDNet-LOLv1-wperc

Weights are loaded from the local `llie-weights/` folder in the GitHub repo.
Falls back to model_cache if available, then to HuggingFace download as last resort.
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

    # Weight file name in llie-weights/ folder
    WEIGHT_FILENAME = "hvi_cidnet_LOL_v1.pth"
    CACHE_WEIGHT_FILENAME = "pytorch_model.bin"

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

    def _resolve_weights(self) -> str:
        """Resolve weight file path with priority:
        1. model_cache (already staged, e.g. from Kaggle weight staging)
        2. llie-weights/ in the GitHub repo (committed to repo)
        3. HuggingFace download (last resort)
        """
        weights_dir = os.path.join(self.cache_dir, "weights")
        cache_weight = os.path.join(weights_dir, self.CACHE_WEIGHT_FILENAME)

        # Priority 1: model_cache (staged weights)
        if os.path.exists(cache_weight):
            print(f"[HVI-CIDNet] Weights found in cache: {cache_weight}")
            return cache_weight

        # Priority 2: llie-weights/ in repo root
        repo_weight = self._find_repo_weight()
        if repo_weight:
            print(f"[HVI-CIDNet] Weights found in repo: {repo_weight}")
            # Copy to cache with expected filename
            os.makedirs(weights_dir, exist_ok=True)
            import shutil
            shutil.copy2(repo_weight, cache_weight)
            print(f"[HVI-CIDNet] Copied to cache as: {cache_weight}")
            return cache_weight

        # Priority 3: HuggingFace download (last resort)
        print("[HVI-CIDNet] Weights not found locally. Attempting HuggingFace download...")
        os.makedirs(weights_dir, exist_ok=True)

        try:
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(
                repo_id="Fediory/HVI-CIDNet-LOLv1-wperc",
                filename="pytorch_model.bin",
                local_dir=weights_dir,
            )
            if not os.path.exists(cache_weight) and os.path.exists(downloaded):
                import shutil
                shutil.move(downloaded, cache_weight)
            print(f"[HVI-CIDNet] Weights downloaded: {cache_weight}")
            return cache_weight
        except ImportError:
            # Fallback: direct URL download
            try:
                url = "https://huggingface.co/Fediory/HVI-CIDNet-LOLv1-wperc/resolve/main/pytorch_model.bin"
                import urllib.request
                print(f"[HVI-CIDNet] Downloading from: {url}")
                urllib.request.urlretrieve(url, cache_weight)
                print(f"[HVI-CIDNet] Weights saved: {cache_weight}")
                return cache_weight
            except Exception as e:
                print(f"[HVI-CIDNet] URL download failed: {e}")
        except Exception as e:
            print(f"[HVI-CIDNet] HuggingFace download failed: {e}")

        raise FileNotFoundError(
            f"HVI-CIDNet weights not found.\n"
            f"  Checked:\n"
            f"    1. Cache: {cache_weight}\n"
            f"    2. Repo llie-weights/: {self.WEIGHT_FILENAME}\n"
            f"    3. HuggingFace download: failed\n"
            f"  Solution: place '{self.WEIGHT_FILENAME}' in llie-weights/ folder."
        )

    def _find_repo_weight(self) -> Optional[str]:
        """Search for weight file in llie-weights/ directory of the repo."""
        search_bases = [
            os.getcwd(),
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        ]
        for base in search_bases:
            candidate = os.path.join(base, "llie-weights", self.WEIGHT_FILENAME)
            if os.path.isfile(candidate):
                return candidate
        return None

    def load_model(self, device: str = None) -> None:
        """Load HVI-CIDNet model with LOL-v1 (wperc) weights.

        Args:
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        from src.enhancers.base import _auto_device
        self.device = _auto_device(device)

        # Clone repo and resolve weights
        self._clone_repo()
        self.weight_path = self._resolve_weights()

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
        checkpoint = torch.load(self.weight_path, map_location=self.device, weights_only=False)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        print(f"[HVI-CIDNet] Model loaded on {self.device}")

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

        # Pad to multiple of 8 (CIDNet has 3 downsampling stages → needs 2^3=8)
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
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
