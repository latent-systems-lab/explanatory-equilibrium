from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError


class Proposal(BaseModel):
    reported_intent: Literal["HEDGE", "SPECULATE"]
    reported_risk: float
    reported_net_delta: float
    reported_hedge_size: float

    model_config = {"extra": "forbid"}


class Claim(BaseModel):
    type: Literal[
        "intent",
        "risk_within_limit",
        "net_delta_bounded",
        "risk_metric",
        "confidence",
    ] = Field(description="The claim type.")
    value: str | bool | int | float | None = Field(
        ...,
        description="Claim value: string for intent, bool for limits, number for metrics/confidence. Use null if unknown.",
    )
    name: str | None = Field(
        ...,
        max_length=30,
        description="Short label, e.g. 'VaR_proxy'. Only for risk_metric claims. Use null otherwise.",
    )

    model_config = {"extra": "forbid"}


class SenderMessage(BaseModel):
    action: Literal["REQUEST_APPROVAL", "NO_TRADE"] = Field(
        description="Trading action to take."
    )
    proposal: Proposal
    claims: List[Claim] = Field(
        max_length=5, description="List of verifiable claims (max 5)."
    )
    free_text: str = Field(
        ..., max_length=2000, description="Brief finance rationale, <=2000 characters (can be empty string)."
    )

    model_config = {"extra": "forbid"}


class SenderParseResult(BaseModel):
    message: SenderMessage
    invalid_output: bool
    error: Optional[str] = None


def parse_sender_message(raw: dict) -> SenderParseResult:
    try:
        msg = SenderMessage.model_validate(raw)
        return SenderParseResult(message=msg, invalid_output=False)
    except ValidationError as err:
        fallback = SenderMessage(
            action="NO_TRADE",
            proposal=Proposal(
                reported_intent="HEDGE",
                reported_risk=0.0,
                reported_net_delta=0.0,
                reported_hedge_size=0.0,
            ),
            claims=[],
            free_text="",
        )
        return SenderParseResult(message=fallback, invalid_output=True, error=str(err))
