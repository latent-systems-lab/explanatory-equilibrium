import pandas as pd

from ee_fin.config import ExperimentConfig
from ee_fin.sender_mock import MockTraderSender
from ee_fin.simulation import run_experiment


class NoExplanationSender:
    def __init__(self, base_sender):
        self.base_sender = base_sender

    def generate(self, true_state, config, episode_idx, max_words):
        payload = self.base_sender.generate(true_state, config, episode_idx, max_words)
        payload = dict(payload)
        payload["claims"] = []
        payload["free_text"] = ""
        return payload


def test_ambiguous_approval_differs():
    config = ExperimentConfig(episodes=40, ambiguous_rate=1.0)
    qs = [0.3]
    max_words_list = [30]
    seeds = [0]

    base_sender = MockTraderSender(seed=0)
    senders = {
        "llm": base_sender,
        "llm_no_expl": NoExplanationSender(base_sender),
    }

    logs = run_experiment(senders, config, qs, max_words_list, seeds, run_id="test")
    ambiguous = logs[logs["episode_type"] == "ambiguous_hedge"]
    rates = (
        ambiguous.groupby("sender")["rm_decision"]
        .apply(lambda x: (x == "APPROVE").mean())
        .to_dict()
    )
    assert rates["llm"] > rates["llm_no_expl"]
