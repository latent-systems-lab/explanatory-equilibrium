from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any

from .config import ExperimentConfig


class SenderBase(ABC):
    @abstractmethod
    def generate(
        self,
        true_state: Dict[str, Any],
        config: ExperimentConfig,
        episode_idx: int,
        max_words: int,
    ) -> Dict[str, Any]:
        raise NotImplementedError
