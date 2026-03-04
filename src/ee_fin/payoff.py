from __future__ import annotations

from dataclasses import dataclass

from .config import ExperimentConfig


@dataclass(frozen=True)
class PayoffResult:
    trader_payoff: float
    rm_payoff: float
    total_welfare: float


def compute_payoff(
    action: str,
    decision: str,
    true_compliant: bool,
    reasoning_words: int,
    audited: bool,
    audit_fail: bool | None,
    config: ExperimentConfig,
    approval_reward: float | None = None,  # V: overrides default if provided
    lie_penalty: float | None = None,  # L: overrides default if provided
) -> PayoffResult:
    # Use provided values or fall back to config defaults
    V = approval_reward if approval_reward is not None else config.approval_reward
    L = lie_penalty if lie_penalty is not None else config.lie_penalty
    
    trader = 0.0
    rm = 0.0

    if action == "NO_TRADE":
        trader += 1.0
        rm += 1.0
    elif action == "REQUEST_APPROVAL":
        if decision == "APPROVE":
            trader += V
            rm += V if true_compliant else -10.0
        else:
            trader += 0.0
            rm += 0.0

    trader -= config.alpha_words * reasoning_words

    if audited:
        rm -= config.audit_cost
        if audit_fail:
            trader -= L

    total = trader + rm
    return PayoffResult(trader_payoff=trader, rm_payoff=rm, total_welfare=total)
