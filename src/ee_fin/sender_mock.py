from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .config import ExperimentConfig
from .sender_base import SenderBase


class MockTraderSender(SenderBase):
    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def _build_claims(
        self,
        reported_intent: str,
        risk_within: bool,
        delta_bounded: bool,
        reported_risk: float,
        confidence: float,
        max_claims: int,
    ) -> List[Dict[str, Any]]:
        claims = [
            {"type": "intent", "value": reported_intent, "name": None},
            {"type": "risk_within_limit", "value": risk_within, "name": None},
            {"type": "net_delta_bounded", "value": delta_bounded, "name": None},
            {"type": "risk_metric", "name": "VaR_proxy", "value": reported_risk},
            {"type": "confidence", "value": confidence, "name": None},
        ]
        return claims[:max_claims]

    def generate(
        self,
        true_state: Dict[str, Any],
        config: ExperimentConfig,
        episode_idx: int,
        max_words: int,
    ) -> Dict[str, Any]:
        compliant = bool(true_state["true_risk"] <= true_state["risk_limit"] and abs(true_state["true_net_delta"]) <= true_state["delta_limit"])
        lie = self.rng.random() < config.lie_prob

        if compliant:
            action = "REQUEST_APPROVAL"
            reported_intent = true_state["intent"]
            reported_risk = true_state["true_risk"]
            reported_net_delta = true_state["true_net_delta"]
            reported_hedge_size = true_state["true_hedge_size"]
            risk_within = True
            delta_bounded = True
        else:
            if lie:
                action = "REQUEST_APPROVAL"
                reported_intent = "HEDGE"
                reported_risk = min(true_state["risk_limit"] - 0.01, 0.99)
                reported_net_delta = float(np.clip(true_state["true_net_delta"], -0.2, 0.2))
                reported_hedge_size = max(true_state["true_hedge_size"], 0.4)
                risk_within = True
                delta_bounded = True
            else:
                action = "NO_TRADE"
                reported_intent = true_state["intent"]
                reported_risk = true_state["true_risk"]
                reported_net_delta = true_state["true_net_delta"]
                reported_hedge_size = true_state["true_hedge_size"]
                risk_within = bool(true_state["true_risk"] <= true_state["risk_limit"])
                delta_bounded = bool(abs(true_state["true_net_delta"]) <= true_state["delta_limit"])

        confidence = float(np.clip(self.rng.normal(0.7, 0.1), 0.0, 1.0))
        claims = self._build_claims(
            reported_intent,
            risk_within,
            delta_bounded,
            reported_risk,
            confidence,
            config.max_claims,
        )

        rationale = (
            "Risk profile aligns with limits; hedging offsets exposure. "
            "Position sizing reflects current volatility."
        )
        words = rationale.split()
        free_text = " ".join(words[:max_words])

        return {
            "action": action,
            "proposal": {
                "reported_intent": reported_intent,
                "reported_risk": float(reported_risk),
                "reported_net_delta": float(reported_net_delta),
                "reported_hedge_size": float(reported_hedge_size),
            },
            "claims": claims,
            "free_text": free_text,
        }
