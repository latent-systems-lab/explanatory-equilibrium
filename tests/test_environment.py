import numpy as np

from ee_fin.config import ExperimentConfig
from ee_fin.environment import sample_state
from ee_fin.rng import make_rng


def test_compliance_bounds():
    config = ExperimentConfig()
    rng = make_rng(1)
    for _ in range(50):
        state = sample_state(rng, config)
        assert config.risk_limit_low <= state.risk_limit <= config.risk_limit_high
        assert -1.5 <= state.true_net_delta <= 1.5
        assert 0.0 <= state.true_hedge_size <= 1.0
        assert isinstance(state.compliant, bool)


def test_ambiguous_sampling():
    config = ExperimentConfig(ambiguous_rate=1.0)
    rng = make_rng(2)
    for _ in range(10):
        state = sample_state(rng, config)
        assert state.episode_type == "ambiguous_hedge"
        assert state.intent == "HEDGE"
        margin_risk = state.risk_limit - state.true_risk
        margin_delta = state.delta_limit - abs(state.true_net_delta)
        assert margin_risk >= 0.0
        assert margin_delta >= 0.0
