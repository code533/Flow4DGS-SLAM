"""Reproducibility helpers for controlled SLAM ablations.

Seeding controls Python/NumPy/PyTorch RNG streams. It does not claim bitwise
CUDA determinism: custom CUDA kernels and multiprocessing can still introduce
small nondeterministic differences.
"""

import random

import numpy as np
import torch


def seed_everything(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
