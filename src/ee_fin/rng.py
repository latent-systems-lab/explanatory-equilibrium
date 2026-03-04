from __future__ import annotations

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def derive_seed(master_seed: int, *tokens: int) -> int:
    seq = np.random.SeedSequence(master_seed, spawn_key=tokens)
    return int(seq.generate_state(1)[0])
