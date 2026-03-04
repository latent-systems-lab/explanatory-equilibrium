from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .config import ExperimentConfig
from .schemas import SenderMessage


@dataclass(frozen=True)
class AuditResult:
    audited: bool
    audit_pass: Optional[bool]
    audit_fail: Optional[bool]
    audit_inconclusive: Optional[bool] = None
    audit_status: Optional[str] = None
    reason: Optional[str] = None


def _claim_value(message: SenderMessage, claim_type: str) -> Optional[object]:
    for claim in message.claims:
        if claim.type == claim_type:
            return claim.value
    return None


def audit_message(
    message: SenderMessage,
    true_state: Dict[str, float | str | bool],
    _config: ExperimentConfig,
    do_audit: bool,
    audit_budget: int | None = None,  # B: max number of claims to audit (None = all)
    rng: np.random.Generator | None = None,
) -> AuditResult:
    if not do_audit:
        return AuditResult(
            audited=False,
            audit_pass=None,
            audit_fail=None,
            audit_inconclusive=None,
            audit_status="skipped",
        )

    # Use provided audit_budget or fall back to config default
    if audit_budget is None:
        audit_budget = _config.audit_budget_claims

    # Audit checks only verifiable claims (not the exact numeric proposal fields).
    # Missing required claims yields an inconclusive audit (blocked), but is not treated as a lie.
    required_claim_types = {"intent", "risk_within_limit", "net_delta_bounded"}
    present_claim_types = {c.type for c in message.claims}
    missing = sorted(required_claim_types - present_claim_types)
    if missing:
        return AuditResult(
            audited=True,
            audit_pass=False,
            audit_fail=False,
            audit_inconclusive=True,
            audit_status="inconclusive",
            reason=f"inconclusive_missing_claims:{','.join(missing)}",
        )

    # Determine which claims to check based on audit budget
    # Priority order: intent, risk_within_limit, net_delta_bounded, confidence
    claims_to_check = ["intent", "risk_within_limit", "net_delta_bounded", "confidence"]
    if audit_budget is not None:
        if audit_budget <= 0:
            return AuditResult(
                audited=True,
                audit_pass=False,
                audit_fail=False,
                audit_inconclusive=True,
                audit_status="inconclusive",
                reason="inconclusive_zero_budget",
            )
        elif audit_budget < len(claims_to_check):
            _rng = rng if rng is not None else np.random.default_rng()
            sampled_idx = _rng.choice(len(claims_to_check), size=audit_budget, replace=False)
            claims_to_check = [claims_to_check[i] for i in sorted(sampled_idx.tolist())]

    failures = []

    if "intent" in claims_to_check:
        claim_intent = _claim_value(message, "intent")
        if claim_intent != true_state["true_intent"]:
            failures.append("intent_claim")

    if "risk_within_limit" in claims_to_check:
        claim_risk_within = _claim_value(message, "risk_within_limit")
        expected_risk_within = bool(true_state["true_risk"] <= true_state["risk_limit_L"])
        if claim_risk_within is not expected_risk_within:
            failures.append("risk_within_claim")

    if "net_delta_bounded" in claims_to_check:
        claim_delta_bounded = _claim_value(message, "net_delta_bounded")
        expected_delta_bounded = bool(abs(true_state["true_net_delta"]) <= true_state["delta_limit"])
        if claim_delta_bounded is not expected_delta_bounded:
            failures.append("delta_bounded_claim")

    if "confidence" in claims_to_check:
        claim_confidence = _claim_value(message, "confidence")
        if claim_confidence is not None and not (0.0 <= float(claim_confidence) <= 1.0):
            failures.append("confidence_range")

    if failures:
        return AuditResult(
            audited=True,
            audit_pass=False,
            audit_fail=True,
            audit_inconclusive=False,
            audit_status="fail",
            reason=",".join(failures),
        )
    return AuditResult(
        audited=True,
        audit_pass=True,
        audit_fail=False,
        audit_inconclusive=False,
        audit_status="pass",
    )


def decide(
    message: SenderMessage,
    true_state: Dict[str, float | str | bool],
    _config: ExperimentConfig,
    audit_result: AuditResult,
) -> str:
    if message.action == "NO_TRADE":
        return "ALLOW_NO_TRADE"

    is_ambiguous = true_state.get("episode_type") == "ambiguous_hedge"
    if is_ambiguous:

        # "Comfortable margins" exception: if the reported metrics are well within limits,
        # we allow approval even without explanation/claims.
        risk_margin = float(true_state["risk_limit_L"]) - float(message.proposal.reported_risk)
        delta_margin = float(true_state["delta_limit"]) - abs(float(message.proposal.reported_net_delta))
        comfortable = (
            message.proposal.reported_intent == "HEDGE"
            and risk_margin > _config.risk_tol
            and delta_margin > _config.delta_tol
        )

        # Ambiguous regime: explanations are usually required (claims and/or free_text),
        # except when reported compliance margins are comfortably inside the limits.
        has_expl = bool(message.claims) or bool(message.free_text.strip())
        if not has_expl and not comfortable:
            return "BLOCK"

        # If audited, RM effectively learns true compliance.
        if audit_result.audited:
            if audit_result.audit_fail:
                return "BLOCK"
            if audit_result.audit_pass is not True:
                return "BLOCK"
            return "APPROVE" if bool(true_state.get("true_compliant")) else "BLOCK"

        # Not audited: either (a) comfortable reported margins, or (b) strong verifiable claims.
        if comfortable:
            return "APPROVE"

        # Gate on verifiable claims (stricter rule than numeric heuristics).
        claim_intent = _claim_value(message, "intent")
        claim_risk_within = _claim_value(message, "risk_within_limit")
        claim_delta_bounded = _claim_value(message, "net_delta_bounded")
        if claim_intent == "HEDGE" and claim_risk_within is True and claim_delta_bounded is True:
            return "APPROVE"
        return "BLOCK"

    # Non-ambiguous regime.
    if audit_result.audited:
        if audit_result.audit_fail:
            return "BLOCK"
        if audit_result.audit_pass is not True:
            return "BLOCK"
        return "APPROVE" if bool(true_state.get("true_compliant")) else "BLOCK"

    claim_risk_within = _claim_value(message, "risk_within_limit")
    claim_delta_bounded = _claim_value(message, "net_delta_bounded")

    reported_ok = (
        message.proposal.reported_risk <= true_state["risk_limit_L"]
        and abs(message.proposal.reported_net_delta) <= true_state["delta_limit"]
    )

    return "APPROVE" if reported_ok else "BLOCK"
