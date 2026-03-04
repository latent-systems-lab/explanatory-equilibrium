from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .config import ExperimentConfig


@dataclass(frozen=True)
class TrueState:
    intent: str
    risk_limit: float
    true_risk: float
    true_net_delta: float
    true_hedge_size: float
    delta_limit: float
    episode_type: str

    @property
    def compliant(self) -> bool:
        return (self.true_risk <= self.risk_limit) and (
            abs(self.true_net_delta) <= self.delta_limit
        )


def _sample_clipped_normal(
    rng: np.random.Generator,
    mean: float,
    sd: float,
    low: float,
    high: float,
) -> float:
    value = rng.normal(mean, sd)
    return float(np.clip(value, low, high))


def _sample_ambiguous_hedge(rng: np.random.Generator, config: ExperimentConfig) -> TrueState:
    risk_limit = float(rng.uniform(config.risk_limit_low, config.risk_limit_high))
    eps_r = float(rng.uniform(0.0, 0.08 * risk_limit))
    eps_d = float(rng.uniform(0.0, 0.08 * config.delta_limit))
    true_risk = risk_limit - eps_r
    true_net_delta = (config.delta_limit - eps_d) * (1.0 if rng.random() < 0.5 else -1.0)
    true_hedge_size = float(rng.uniform(0.4, 1.0))

    return TrueState(
        intent="HEDGE",
        risk_limit=risk_limit,
        true_risk=true_risk,
        true_net_delta=true_net_delta,
        true_hedge_size=true_hedge_size,
        delta_limit=config.delta_limit,
        episode_type="ambiguous_hedge",
    )


def _sample_normal_state(rng: np.random.Generator, config: ExperimentConfig) -> TrueState:
    intent = rng.choice(["HEDGE", "SPECULATE"])
    risk_limit = float(rng.uniform(config.risk_limit_low, config.risk_limit_high))

    if intent == "HEDGE":
        true_risk = float(rng.uniform(0.2, 1.0))
        true_net_delta = _sample_clipped_normal(rng, 0.0, 0.1, -0.3, 0.3)
        true_hedge_size = float(rng.uniform(0.5, 1.0))
    else:
        true_risk = float(rng.uniform(0.9, 2.0))
        true_net_delta = _sample_clipped_normal(rng, 0.8, 0.2, 0.3, 1.5)
        true_hedge_size = float(rng.uniform(0.0, 0.3))

    return TrueState(
        intent=intent,
        risk_limit=risk_limit,
        true_risk=true_risk,
        true_net_delta=true_net_delta,
        true_hedge_size=true_hedge_size,
        delta_limit=config.delta_limit,
        episode_type="normal",
    )


def sample_state(rng: np.random.Generator, config: ExperimentConfig) -> TrueState:
    if rng.random() < config.ambiguous_rate:
        return _sample_ambiguous_hedge(rng, config)
    return _sample_normal_state(rng, config)


def state_to_dict(state: TrueState) -> Dict[str, float | str]:
    margin_risk = state.risk_limit - state.true_risk
    margin_delta = state.delta_limit - abs(state.true_net_delta)
    return {
        "true_intent": state.intent,
        "risk_limit_L": state.risk_limit,
        "true_risk": state.true_risk,
        "true_net_delta": state.true_net_delta,
        "true_hedge_size": state.true_hedge_size,
        "delta_limit": state.delta_limit,
        "true_compliant": state.compliant,
        "episode_type": state.episode_type,
        "margin_risk": margin_risk,
        "margin_delta": margin_delta,
    }
