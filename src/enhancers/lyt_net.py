"""
LYT-Net enhancer wrapper (PyTorch port).

Uses the PyTorch pretrained weights (LOL-v1 trained).
Repo: https://github.com/albrateanu/LYT-Net
Weights: PyTorch .pth from Google Drive

Weights are loaded from the local `llie-weights/` folder in the GitHub repo.
Falls back to model_cache if available, then to gdown download as last resort.

LYT-Net is extremely lightweight: ~45K params, 3.49 GFLOPs.
"""

import os
import sys
import shutil
import subprocess
import zipfile
import tempfile
import cv2
import torch
import numpy as np
import importlib
import importlib.util
from typing import Optional

from src.enhancers.base import BaseEnhancer


class LYTNetEnhancer(BaseEnhancer):
    """LYT-Net Low-Light Image Enhancement wrapper (PyTorch)."""

    # Weight file name in llie-weights/ folder
    WEIGHT_FILENAME = "lyt_net_lol.pth"
    CACHE_WEIGHT_FILENAME = "lyt_net_lol.pth"

    def __init__(self, cache_dir: str = "cache/LYT-Net"):
        super().__init__(name="LYT-Net")
        self.cache_dir = cache_dir
        self.repo_dir = os.path.join(cache_dir, "repo")
        self.weight_path = None

    def _clone_repo(self) -> None:
        """Clone LYT-Net repo if not exists."""
        model_marker = os.path.join(self.repo_dir, "PyTorch", "model.py")

        # Upstream repo stores PyTorch code under repo/PyTorch/model.py
        if (
            os.path.isdir(self.repo_dir)
            and (
                os.path.isfile(model_marker)
                or os.path.isfile(os.path.join(self.repo_dir, "model.py"))
                or os.path.exists(os.path.join(self.repo_dir, "models"))
            )
        ):
            print(f"[LYT-Net] Repo already exists: {self.repo_dir}")
            return

        # If a partial/corrupted repo directory exists, remove it before cloning.
        if os.path.isdir(self.repo_dir):
            print(f"[LYT-Net] Found incomplete repo cache, removing: {self.repo_dir}")
            shutil.rmtree(self.repo_dir, ignore_errors=True)

        os.makedirs(self.cache_dir, exist_ok=True)
        print("[LYT-Net] Cloning repository...")

        proc = subprocess.run(
            ["git", "clone", "https://github.com/albrateanu/LYT-Net.git", self.repo_dir],
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            raise RuntimeError(
                "Failed to clone LYT-Net repository. "
                f"Return code: {proc.returncode}. "
                f"STDERR: {stderr or '<empty>'}. "
                f"STDOUT: {stdout or '<empty>'}"
            )
        print("[LYT-Net] Repository cloned successfully")

    def _resolve_weights(self) -> str:
        """Resolve weight file path with priority:
        1. model_cache (already staged, e.g. from Kaggle weight staging)
        2. llie-weights/ in the GitHub repo (committed to repo)
        3. gdown download from Google Drive (last resort)
        """
        weights_dir = os.path.join(self.cache_dir, "weights")
        cache_weight = os.path.join(weights_dir, self.CACHE_WEIGHT_FILENAME)

        # Priority 1: model_cache (staged weights)
        if os.path.exists(cache_weight):
            print(f"[LYT-Net] Weights found in cache: {cache_weight}")
            return cache_weight

        # Priority 2: llie-weights/ in repo root
        repo_weight = self._find_repo_weight()
        if repo_weight:
            print(f"[LYT-Net] Weights found in repo: {repo_weight}")
            # Copy to cache for consistent path usage
            os.makedirs(weights_dir, exist_ok=True)
            shutil.copy2(repo_weight, cache_weight)
            print(f"[LYT-Net] Copied to cache: {cache_weight}")
            return cache_weight

        # Priority 3: gdown download (last resort)
        print("[LYT-Net] Weights not found locally. Attempting gdown download...")
        os.makedirs(weights_dir, exist_ok=True)

        try:
            import gdown
            url = "https://drive.google.com/uc?id=1GeEkasO2ubFi847pzrxfQ1fB3Y9NuhZ1"
            gdown.download(url, cache_weight, quiet=False)
            if os.path.exists(cache_weight):
                print(f"[LYT-Net] Weights downloaded: {cache_weight}")
                return cache_weight
        except Exception as e:
            print(f"[LYT-Net] gdown download failed: {e}")

        raise FileNotFoundError(
            f"LYT-Net weights not found.\n"
            f"  Checked:\n"
            f"    1. Cache: {cache_weight}\n"
            f"    2. Repo llie-weights/: {self.WEIGHT_FILENAME}\n"
            f"    3. gdown download: failed\n"
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

    def _load_checkpoint_robust(self, weight_path: str):
        """Load checkpoint with fallback for malformed/packaged archives.

        Some shared LYT-Net weight files are distributed as zip containers that include
        an inner *.pth file (e.g., best_model_LOLv1.pth). torch.load() on the outer file
        can fail with 'file in archive is not in a subdirectory'.
        """
        try:
            return torch.load(weight_path, map_location=self.device, weights_only=False)
        except RuntimeError as e:
            msg = str(e)
            if "file in archive is not in a subdirectory" not in msg:
                raise

            if not zipfile.is_zipfile(weight_path):
                raise

            with zipfile.ZipFile(weight_path, "r") as zf:
                members = zf.namelist()
                candidates = [
                    m for m in members
                    if m.lower().endswith((".pth", ".pt", ".ckpt")) and not m.endswith("/")
                ]
                if not candidates:
                    raise RuntimeError(
                        f"Weights archive detected but no .pth/.pt/.ckpt inside: {weight_path}"
                    )

                # Prefer best_model* when available, otherwise first candidate.
                candidates.sort(key=lambda x: (0 if "best_model" in x.lower() else 1, len(x)))
                inner_member = candidates[0]
                print(f"[LYT-Net] Loading inner checkpoint from archive: {inner_member}")

                with zf.open(inner_member, "r") as fin, tempfile.NamedTemporaryFile(
                    suffix=".pth", delete=False
                ) as tmp:
                    tmp.write(fin.read())
                    tmp_path = tmp.name

            try:
                return torch.load(tmp_path, map_location=self.device, weights_only=False)
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def load_model(self, device: str = None) -> None:
        """Load LYT-Net PyTorch model."""
        from src.enhancers.base import _auto_device
        self.device = _auto_device(device)

        self._clone_repo()
        self.weight_path = self._resolve_weights()

        # Add repo and common subfolders to path
        candidate_paths = [
            self.repo_dir,
            os.path.join(self.repo_dir, "PyTorch"),
            os.path.join(self.repo_dir, "pytorch"),
        ]
        for p in candidate_paths:
            if os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)

        # Import model architecture
        # LYT-Net PyTorch port structure may vary — try multiple import paths
        model = None
        import_errors = []

        # Try different import patterns based on repo structure
        for module_name in [
            "model",
            "PyTorch.model",
            "pytorch.model",
            "models",
            "models.LYTNet",
        ]:
            try:
                mod = importlib.import_module(module_name)
                for class_name in ("LYT", "LYTNet", "LYT_Net"):
                    if hasattr(mod, class_name):
                        model = getattr(mod, class_name)()
                        break
                if model is not None:
                    break
            except Exception as e:
                import_errors.append(f"{module_name}: {e}")
                continue

        if model is None:
            # Fallback: load directly from model.py path
            direct_model_paths = [
                os.path.join(self.repo_dir, "PyTorch", "model.py"),
                os.path.join(self.repo_dir, "pytorch", "model.py"),
                os.path.join(self.repo_dir, "model.py"),
            ]
            for model_py in direct_model_paths:
                if not os.path.isfile(model_py):
                    continue
                try:
                    spec = importlib.util.spec_from_file_location("lyt_model_module", model_py)
                    if spec is None or spec.loader is None:
                        continue
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    for class_name in ("LYT", "LYTNet", "LYT_Net"):
                        if hasattr(mod, class_name):
                            model = getattr(mod, class_name)()
                            break
                    if model is not None:
                        break
                except Exception as e:
                    import_errors.append(f"direct:{model_py}: {e}")

        if model is None:
            import glob
            py_files = glob.glob(os.path.join(self.repo_dir, "**", "*.py"), recursive=True)
            model_files = [f for f in py_files if "model" in os.path.basename(f).lower()]
            raise ImportError(
                f"Failed to import LYT-Net model. "
                f"Tried multiple import paths. Errors: {import_errors}. "
                f"Found Python files with 'model': {model_files}"
            )

        # Load weights
        checkpoint = self._load_checkpoint_robust(self.weight_path)
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            elif "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)
        model.eval()
        self.model = model
        self._loaded = True
        print(f"[LYT-Net] Model loaded on {self.device}")

    @torch.no_grad()
    def enhance(self, img_bgr: np.ndarray) -> np.ndarray:
        """Enhance a single low-light image."""
        assert self._loaded, "Model not loaded. Call load_model() first."

        h, w = img_bgr.shape[:2]

        # BGR → RGB, normalize, to tensor
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(self.device)

        # Pad to multiple of 32 to fix maxpool tensor matching length
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
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
