"""
RetinexFormer enhancer wrapper.

Uses LOL_v1.pth pretrained weights (validated on ExDark by the original authors).
Repo: https://github.com/caiyuanhao1998/Retinexformer

Weights are loaded from the local `llie-weights/` folder in the GitHub repo.
Falls back to model_cache if available, then to gdown download as last resort.
"""

import os
import cv2
import torch
import numpy as np
from typing import Optional

from src.enhancers.base import BaseEnhancer


class RetinexFormerEnhancer(BaseEnhancer):
    """RetinexFormer Low-Light Image Enhancement wrapper."""

    # Weight file name expected by this enhancer
    WEIGHT_FILENAME = "retinexformer_LOL_v1.pth"
    CACHE_WEIGHT_FILENAME = "LOL_v1.pth"

    def __init__(self, cache_dir: str = "cache/Retinexformer"):
        super().__init__(name="RetinexFormer")
        self.cache_dir = cache_dir
        self.repo_dir = os.path.join(cache_dir, "repo")
        self.weight_path = None

    def _clone_repo(self) -> None:
        """Clone RetinexFormer repo if not exists."""
        if os.path.exists(os.path.join(self.repo_dir, "basicsr")):
            print(f"[RetinexFormer] Repo already exists: {self.repo_dir}")
            return

        os.makedirs(self.cache_dir, exist_ok=True)
        print("[RetinexFormer] Cloning repository...")
        ret = os.system(f"git clone https://github.com/caiyuanhao1998/Retinexformer.git {self.repo_dir}")
        if ret != 0:
            raise RuntimeError("Failed to clone RetinexFormer repository")
        print("[RetinexFormer] Repository cloned successfully")

    def _resolve_weights(self) -> str:
        """Resolve weight file path with priority:
        1. model_cache (already staged, e.g. from Kaggle weight staging)
        2. Kaggle input dataset (manual upload)
        3. llie-weights/ in the GitHub repo (committed to repo)
        4. gdown download from Google Drive (last resort)
        """
        weights_dir = os.path.join(self.cache_dir, "weights")
        cache_weight = os.path.join(weights_dir, self.CACHE_WEIGHT_FILENAME)

        # Priority 1: model_cache (staged weights)
        if os.path.exists(cache_weight):
            print(f"[RetinexFormer] Weights found in cache: {cache_weight}")
            return cache_weight

        # Priority 2: Kaggle input dataset (manual upload)
        kaggle_weight = self._find_kaggle_weight()
        if kaggle_weight:
            print(f"[RetinexFormer] Weights found in Kaggle input: {kaggle_weight}")
            os.makedirs(weights_dir, exist_ok=True)
            import shutil
            shutil.copy2(kaggle_weight, cache_weight)
            print(f"[RetinexFormer] Copied to cache: {cache_weight}")
            return cache_weight

        # Priority 3: llie-weights/ in repo root
        repo_weight = self._find_repo_weight()
        if repo_weight:
            print(f"[RetinexFormer] Weights found in repo: {repo_weight}")
            # Copy to cache for consistent path usage
            os.makedirs(weights_dir, exist_ok=True)
            import shutil
            shutil.copy2(repo_weight, cache_weight)
            print(f"[RetinexFormer] Copied to cache: {cache_weight}")
            return cache_weight

        # Priority 4: gdown download (last resort)
        print("[RetinexFormer] Weights not found locally. Attempting gdown download...")
        print("  Manual download: https://drive.google.com/drive/folders/1ynK5hfQachzc8y96ZumhkPPDXzHJwaQV")
        print(f"  Save to: {cache_weight}")
        os.makedirs(weights_dir, exist_ok=True)

        try:
            import gdown
            url = "https://drive.google.com/uc?id=1sft9MU-fwtVH0ubNg-GPmGi1BVvIsFSi"
            gdown.download(url, cache_weight, quiet=False)
            if os.path.exists(cache_weight):
                print(f"[RetinexFormer] Weights downloaded: {cache_weight}")
                return cache_weight
        except Exception as e:
            print(f"[RetinexFormer] gdown download failed: {e}")

        raise FileNotFoundError(
            f"RetinexFormer weights not found.\n"
            f"  Checked:\n"
            f"    1. Cache: {cache_weight}\n"
            f"    2. Kaggle input: {self.WEIGHT_FILENAME} or {self.CACHE_WEIGHT_FILENAME}\n"
            f"    3. Repo llie-weights/: {self.WEIGHT_FILENAME}\n"
            f"    4. gdown download: failed\n"
            f"  Solution: place '{self.WEIGHT_FILENAME}' in llie-weights/ folder."
        )

    def _find_kaggle_weight(self) -> Optional[str]:
        """Search for weight file in Kaggle input dataset root."""
        kaggle_root = "/kaggle/input"
        if not os.path.isdir(kaggle_root):
            return None

        import glob

        patterns = [
            os.path.join(kaggle_root, "**", self.WEIGHT_FILENAME),
            os.path.join(kaggle_root, "**", self.CACHE_WEIGHT_FILENAME),
        ]
        for pattern in patterns:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                return matches[0]
        return None

    def _find_repo_weight(self) -> Optional[str]:
        """Search for weight file in llie-weights/ directory of the repo."""
        # Search upward from CWD and from this file's location
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
        """Load RetinexFormer model with LOL_v1 weights."""
        from src.enhancers.base import _auto_device
        self.device = _auto_device(device)

        self._clone_repo()
        self.weight_path = self._resolve_weights()

        # Load architecture directly from the cloned repo file.
        # We CANNOT rely on sys.path + `import basicsr.models.archs...` because
        # the pip-installed `basicsr` (dependency of pyiqa) shadows the repo's
        # local `basicsr/` package.  importlib bypasses this entirely.
        import importlib.util

        arch_file = None
        candidates = [
            os.path.join(self.repo_dir, "basicsr", "models", "archs", "RetinexFormer_arch.py"),
            os.path.join(self.repo_dir, "basicsr", "models", "archs", "Retinexformer_arch.py"),
            os.path.join(self.repo_dir, "basicsr", "archs", "RetinexFormer_arch.py"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                arch_file = c
                break

        if arch_file is None:
            raise FileNotFoundError(
                f"RetinexFormer_arch.py not found in repo.\n"
                f"  Searched: {candidates}\n"
                f"  Repo dir: {self.repo_dir}"
            )

        spec = importlib.util.spec_from_file_location("RetinexFormer_arch", arch_file)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        RetinexFormer = getattr(mod, "RetinexFormer", None) or getattr(mod, "Retinexformer", None)
        if RetinexFormer is None:
            raise ImportError(f"Class 'RetinexFormer' not found in {arch_file}")

        # Initialize model with default config for LOL
        self.model = RetinexFormer(
            in_channels=3,
            out_channels=3,
            n_feat=40,
            stage=1,
            num_blocks=[1, 2, 2],
        )

        # Load weights
        checkpoint = torch.load(self.weight_path, map_location=self.device, weights_only=False)
        if "params" in checkpoint:
            self.model.load_state_dict(checkpoint["params"])
        elif "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        print(f"[RetinexFormer] Model loaded on {self.device}")

    @torch.no_grad()
    def enhance(self, img_bgr: np.ndarray) -> np.ndarray:
        """Enhance a single low-light image."""
        assert self._loaded, "Model not loaded. Call load_model() first."

        h, w = img_bgr.shape[:2]

        # BGR → RGB, normalize, to tensor
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(self.device)

        # Pad to multiple of 8 (RetinexFormer requirement)
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            img_tensor = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')

        # Inference
        output = self.model(img_tensor)

        if isinstance(output, (tuple, list)):
            output = output[0]

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            output = output[:, :, :h, :w]

        # To numpy BGR
        output = output.squeeze(0).clamp(0, 1).cpu().numpy()
        output = (output * 255).astype(np.uint8)
        output = np.transpose(output, (1, 2, 0))
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        return output
