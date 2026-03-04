from ee_fin.config import ExperimentConfig
from ee_fin.payoff import compute_payoff


def test_payoff_approval_compliant():
    config = ExperimentConfig()
    result = compute_payoff(
        action="REQUEST_APPROVAL",
        decision="APPROVE",
        true_compliant=True,
        reasoning_words=10,
        audited=False,
        audit_fail=None,
        config=config,
    )
    assert result.trader_payoff > 0
    assert result.rm_payoff > 0


def test_payoff_audit_penalty():
    config = ExperimentConfig()
    result = compute_payoff(
        action="REQUEST_APPROVAL",
        decision="BLOCK",
        true_compliant=False,
        reasoning_words=0,
        audited=True,
        audit_fail=True,
        config=config,
    )
    assert result.trader_payoff < 0
    assert result.rm_payoff < 0
