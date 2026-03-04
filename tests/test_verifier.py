import numpy as np

from ee_fin.config import ExperimentConfig
from ee_fin.schemas import SenderMessage
from ee_fin.verifier import audit_message, decide


def test_audit_and_decision():
    config = ExperimentConfig()
    true_state = {
        "true_intent": "HEDGE",
        "risk_limit_L": 1.0,
        "true_risk": 0.9,
        "true_net_delta": 0.1,
        "true_hedge_size": 0.7,
        "delta_limit": 0.3,
        "true_compliant": True,
        "episode_type": "normal",
    }
    message = SenderMessage.model_validate(
        {
            "action": "REQUEST_APPROVAL",
            "proposal": {
                "reported_intent": "HEDGE",
                "reported_risk": 0.9,
                "reported_net_delta": 0.1,
                "reported_hedge_size": 0.7,
            },
            "claims": [
                {"type": "intent", "value": "HEDGE", "name": None},
                {"type": "risk_within_limit", "value": True, "name": None},
                {"type": "net_delta_bounded", "value": True, "name": None},
                {"type": "risk_metric", "name": "VaR_proxy", "value": 0.9},
                {"type": "confidence", "value": 0.8, "name": None},
            ],
            "free_text": "ok",
        }
    )

    audit = audit_message(message, true_state, config, do_audit=True)
    assert audit.audit_pass is True
    assert decide(message, true_state, config, audit) == "APPROVE"

    tampered = message.model_copy(deep=True)
    # Audits check claims only; tampering with an auditable claim should be detected.
    tampered.claims[1].value = False
    audit_bad = audit_message(tampered, true_state, config, do_audit=True)
    assert audit_bad.audit_fail is True


def test_audit_inconclusive_when_missing_claims():
    config = ExperimentConfig()
    true_state = {
        "true_intent": "HEDGE",
        "risk_limit_L": 1.0,
        "true_risk": 0.9,
        "true_net_delta": 0.1,
        "true_hedge_size": 0.7,
        "delta_limit": 0.3,
        "true_compliant": True,
        "episode_type": "normal",
    }
    message = SenderMessage.model_validate(
        {
            "action": "REQUEST_APPROVAL",
            "proposal": {
                "reported_intent": "HEDGE",
                "reported_risk": 0.9,
                "reported_net_delta": 0.1,
                "reported_hedge_size": 0.7,
            },
            "claims": [],
            "free_text": "",
        }
    )

    audit = audit_message(message, true_state, config, do_audit=True)
    assert audit.audited is True
    assert audit.audit_pass is False
    assert audit.audit_fail is False
    assert audit.audit_inconclusive is True
    assert audit.audit_status == "inconclusive"
    assert decide(message, true_state, config, audit) == "BLOCK"


def test_audit_budget_uses_random_sampling_with_rng():
    config = ExperimentConfig()
    true_state = {
        "true_intent": "HEDGE",
        "risk_limit_L": 1.0,
        "true_risk": 0.9,
        "true_net_delta": 0.1,
        "true_hedge_size": 0.7,
        "delta_limit": 0.3,
        "true_compliant": True,
        "episode_type": "normal",
    }
    message = SenderMessage.model_validate(
        {
            "action": "REQUEST_APPROVAL",
            "proposal": {
                "reported_intent": "HEDGE",
                "reported_risk": 0.9,
                "reported_net_delta": 0.1,
                "reported_hedge_size": 0.7,
            },
            "claims": [
                {"type": "intent", "value": "HEDGE", "name": None},
                {"type": "risk_within_limit", "value": False, "name": None},
                {"type": "net_delta_bounded", "value": True, "name": None},
                {"type": "confidence", "value": 0.8, "name": None},
            ],
            "free_text": "ok",
        }
    )

    seed = 8
    claim_types = ["intent", "risk_within_limit", "net_delta_bounded", "confidence"]
    sampled_idx = int(np.random.default_rng(seed).choice(len(claim_types), size=1, replace=False)[0])
    sampled_type = claim_types[sampled_idx]

    audit = audit_message(
        message,
        true_state,
        config,
        do_audit=True,
        audit_budget=1,
        rng=np.random.default_rng(seed),
    )
    if sampled_type == "risk_within_limit":
        assert audit.audit_status == "fail"
        assert audit.audit_fail is True
    else:
        assert audit.audit_status == "pass"
        assert audit.audit_fail is False


def test_audit_zero_budget_is_inconclusive():
    config = ExperimentConfig()
    true_state = {
        "true_intent": "HEDGE",
        "risk_limit_L": 1.0,
        "true_risk": 0.9,
        "true_net_delta": 0.1,
        "true_hedge_size": 0.7,
        "delta_limit": 0.3,
        "true_compliant": True,
        "episode_type": "normal",
    }
    message = SenderMessage.model_validate(
        {
            "action": "REQUEST_APPROVAL",
            "proposal": {
                "reported_intent": "HEDGE",
                "reported_risk": 0.9,
                "reported_net_delta": 0.1,
                "reported_hedge_size": 0.7,
            },
            "claims": [
                {"type": "intent", "value": "HEDGE", "name": None},
                {"type": "risk_within_limit", "value": True, "name": None},
                {"type": "net_delta_bounded", "value": True, "name": None},
            ],
            "free_text": "ok",
        }
    )

    audit = audit_message(message, true_state, config, do_audit=True, audit_budget=0)
    assert audit.audited is True
    assert audit.audit_pass is False
    assert audit.audit_fail is False
    assert audit.audit_inconclusive is True
    assert audit.audit_status == "inconclusive"
    assert audit.reason == "inconclusive_zero_budget"


def test_ambiguous_requires_explanation():
    config = ExperimentConfig()
    true_state = {
        "true_intent": "HEDGE",
        "risk_limit_L": 1.0,
        "true_risk": 0.98,
        "true_net_delta": 0.29,
        "true_hedge_size": 0.7,
        "delta_limit": 0.3,
        "true_compliant": True,
        "episode_type": "ambiguous_hedge",
    }
    message = SenderMessage.model_validate(
        {
            "action": "REQUEST_APPROVAL",
            "proposal": {
                "reported_intent": "HEDGE",
                "reported_risk": 0.98,
                "reported_net_delta": 0.29,
                "reported_hedge_size": 0.7,
            },
            "claims": [],
            "free_text": "",
        }
    )
    audit = audit_message(message, true_state, config, do_audit=False)
    assert decide(message, true_state, config, audit) == "BLOCK"
