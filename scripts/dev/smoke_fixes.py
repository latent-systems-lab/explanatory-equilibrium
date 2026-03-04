#!/usr/bin/env python3
"""Quick smoke test for the three critical fixes."""

from ee_fin.config import ExperimentConfig
from ee_fin.schemas import SenderMessage
from ee_fin.verifier import audit_message, decide

def test_ambiguous_without_explanation_but_low_margin():
    """Test that ambiguous WITHOUT explanation can still approve if margins are comfortable."""
    config = ExperimentConfig()
    
    # Ambiguous episode with comfortable margins
    true_state = {
        "true_intent": "HEDGE",
        "risk_limit_L": 1.0,
        "true_risk": 0.92,          # margin = 0.08 > 0.05 ✓
        "true_net_delta": 0.25,     # margin = 0.05 = 0.05 (not >)
        "true_hedge_size": 0.7,
        "delta_limit": 0.3,
        "true_compliant": True,
        "episode_type": "ambiguous_hedge",
    }
    
    # Message: NO claims, NO explanation, but reported within limits with comfortable margins
    message = SenderMessage.model_validate({
        "action": "REQUEST_APPROVAL",
        "proposal": {
            "reported_intent": "HEDGE",
            "reported_risk": 0.92,          # Within limit, margin = 0.08
            "reported_net_delta": 0.24,     # Within limit, margin = 0.06
            "reported_hedge_size": 0.7,
        },
        "claims": [],
        "free_text": "",
    })
    
    audit = audit_message(message, true_state, config, do_audit=False)
    decision = decide(message, true_state, config, audit)
    
    print(f"Test 1 - No explanation + comfortable margins: {decision}")
    assert decision == "APPROVE", f"Expected APPROVE but got {decision}"
    print("✓ PASS")

def test_ambiguous_without_explanation_tight_margin():
    """Test that ambiguous WITHOUT explanation blocks if margins are tight."""
    config = ExperimentConfig()
    
    # Ambiguous episode with tight margins
    true_state = {
        "true_intent": "HEDGE",
        "risk_limit_L": 1.0,
        "true_risk": 0.99,          # margin = 0.01 < 0.05 ✗
        "true_net_delta": 0.29,
        "true_hedge_size": 0.7,
        "delta_limit": 0.3,
        "true_compliant": True,
        "episode_type": "ambiguous_hedge",
    }
    
    # Message: NO claims, NO explanation, reported at tight margins
    message = SenderMessage.model_validate({
        "action": "REQUEST_APPROVAL",
        "proposal": {
            "reported_intent": "HEDGE",
            "reported_risk": 0.99,          # Within limit, but margin = 0.01
            "reported_net_delta": 0.29,     # Within limit, but margin = 0.01
            "reported_hedge_size": 0.7,
        },
        "claims": [],
        "free_text": "",
    })
    
    audit = audit_message(message, true_state, config, do_audit=False)
    decision = decide(message, true_state, config, audit)
    
    print(f"Test 2 - No explanation + tight margins: {decision}")
    assert decision == "BLOCK", f"Expected BLOCK but got {decision}"
    print("✓ PASS")

def test_ambiguous_with_explanation_tight_margin():
    """Test that ambiguous WITH explanation approves even with tight margins."""
    config = ExperimentConfig()
    
    # Ambiguous episode with tight margins
    true_state = {
        "true_intent": "HEDGE",
        "risk_limit_L": 1.0,
        "true_risk": 0.99,
        "true_net_delta": 0.29,
        "true_hedge_size": 0.7,
        "delta_limit": 0.3,
        "true_compliant": True,
        "episode_type": "ambiguous_hedge",
    }
    
    # Message: WITH claims and explanation
    message = SenderMessage.model_validate({
        "action": "REQUEST_APPROVAL",
        "proposal": {
            "reported_intent": "HEDGE",
            "reported_risk": 0.99,
            "reported_net_delta": 0.29,
            "reported_hedge_size": 0.7,
        },
        "claims": [
            {"type": "intent", "value": "HEDGE", "name": None},
            {"type": "risk_within_limit", "value": True, "name": None},
            {"type": "net_delta_bounded", "value": True, "name": None},
        ],
        "free_text": "Borderline hedging position.",
    })
    
    audit = audit_message(message, true_state, config, do_audit=False)
    decision = decide(message, true_state, config, audit)
    
    print(f"Test 3 - With explanation + tight margins: {decision}")
    assert decision == "APPROVE", f"Expected APPROVE but got {decision}"
    print("✓ PASS")

def test_normal_episode_unaffected():
    """Test that normal episodes still work the old way."""
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
    
    # Without explanation but with good reported values
    message = SenderMessage.model_validate({
        "action": "REQUEST_APPROVAL",
        "proposal": {
            "reported_intent": "HEDGE",
            "reported_risk": 0.9,
            "reported_net_delta": 0.1,
            "reported_hedge_size": 0.7,
        },
        "claims": [
            {"type": "risk_within_limit", "value": True, "name": None},
            {"type": "net_delta_bounded", "value": True, "name": None},
        ],
        "free_text": "",
    })
    
    audit = audit_message(message, true_state, config, do_audit=False)
    decision = decide(message, true_state, config, audit)
    
    print(f"Test 4 - Normal episode (should not trigger ambiguous logic): {decision}")
    assert decision == "APPROVE", f"Expected APPROVE but got {decision}"
    print("✓ PASS")

if __name__ == "__main__":
    print("=" * 60)
    print("SMOKE TEST: Critical Fixes")
    print("=" * 60)
    
    try:
        test_ambiguous_without_explanation_but_low_margin()
        test_ambiguous_without_explanation_tight_margin()
        test_ambiguous_with_explanation_tight_margin()
        test_normal_episode_unaffected()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        print("\nKey behavior verified:")
        print("1. Explanations are valuable (approves at tight margins)")
        print("2. Explanations are not required (approves comfortable margins without explanation)")
        print("3. Normal episodes unaffected")
        print("4. Ground truth no longer in prompt (fix #1 verified via code review)")
        print("5. Cache enabled (fix #3 verified via code review)")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
