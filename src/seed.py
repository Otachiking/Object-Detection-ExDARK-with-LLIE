"""
Global seed management for reproducibility.

Sets seeds for: random, numpy, torch (CPU+CUDA), and cudnn deterministic mode.
"""

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility.

    Args:
        seed: Integer seed value. Default 42.
    """
    # Python built-in
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (all GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # CuDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[SEED] Global seed set to {seed} (deterministic mode enabled)")
