from ee_fin.schemas import parse_sender_message


def test_invalid_json_fallback():
    raw = {"bad": "data"}
    result = parse_sender_message(raw)
    assert result.invalid_output is True
    assert result.message.action == "NO_TRADE"


def test_empty_claims_allowed():
    raw = {
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
    result = parse_sender_message(raw)
    assert result.invalid_output is False
    assert result.message.claims == []
    assert result.message.free_text == ""
