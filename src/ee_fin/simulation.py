from __future__ import annotations

import inspect
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import ExperimentConfig
from .environment import sample_state, state_to_dict
from .payoff import compute_payoff
from .rng import derive_seed, make_rng
from .schemas import parse_sender_message
from .verifier import audit_message, decide


def _enforce_max_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def _extract_claims(message) -> Dict[str, object]:
    claim_map: Dict[str, object] = {
        "intent": None,
        "risk_within_limit": None,
        "net_delta_bounded": None,
        "risk_metric": None,
        "confidence": None,
    }
    for claim in message.claims:
        if claim.type == "risk_metric":
            claim_map["risk_metric"] = claim.value
        else:
            claim_map[claim.type] = claim.value
    return claim_map


def run_condition(
    sender,
    sender_name: str,
    config: ExperimentConfig,
    q: float,
    max_words: int,
    approval_reward: float,
    lie_penalty: float,
    audit_budget: int | None,
    seed: int,
    rng_seed: int,
    run_id: str,
    show_progress: bool = True,
) -> pd.DataFrame:
    rng = make_rng(rng_seed)
    rows: List[Dict[str, object]] = []

    episode_iter = range(config.episodes)
    if show_progress:
        episode_iter = tqdm(
            episode_iter, 
            desc=f"q={q}, max_words={max_words}, V={approval_reward}, L={lie_penalty}, B={audit_budget}", 
            leave=False
        )

    for episode_idx in episode_iter:
        # Create a config for this specific condition with current V, L, B values
        condition_config = replace(
            config,
            approval_reward=approval_reward,
            lie_penalty=lie_penalty,
            audit_budget_claims=audit_budget,
        )
        
        true_state = sample_state(rng, condition_config)

        # Provide the sender with noisy observations (not ground truth).
        # We still keep the underlying `true_*` fields in the dict for non-LLM baselines,
        # but the LLM prompt should only expose `obs_*` values.
        sender_state = asdict(true_state)
        sender_state.update(
            {
                "obs_risk": float(np.clip(true_state.true_risk + rng.normal(0.0, condition_config.obs_risk_sd), 0.0, np.inf)),
                "obs_net_delta": float(true_state.true_net_delta + rng.normal(0.0, condition_config.obs_delta_sd)),
                "obs_hedge_size": float(
                    np.clip(true_state.true_hedge_size + rng.normal(0.0, condition_config.obs_hedge_sd), 0.0, np.inf)
                ),
            }
        )
        raw_message = sender.generate(
            true_state=sender_state,
            config=condition_config,
            episode_idx=episode_idx,
            max_words=max_words,
        )
        parse_result = parse_sender_message(raw_message)
        message = parse_result.message
        invalid_output = parse_result.invalid_output
        message.free_text = _enforce_max_words(message.free_text, max_words)

        reasoning_words = len(message.free_text.split())
        reasoning_chars = len(message.free_text)

        do_audit = rng.random() < q
        true_state_dict = state_to_dict(true_state)

        audit_result = audit_message(
            message,
            true_state_dict,
            condition_config,
            do_audit,
            audit_budget,
            rng=rng,
        )
        decision = decide(message, true_state_dict, condition_config, audit_result)

        payoff = compute_payoff(
            action=message.action,
            decision=decision,
            true_compliant=true_state.compliant,
            reasoning_words=reasoning_words,
            audited=audit_result.audited,
            audit_fail=audit_result.audit_fail,
            config=condition_config,
            approval_reward=approval_reward,
            lie_penalty=lie_penalty,
        )

        claims = _extract_claims(message)

        rows.append(
            {
                "run_id": run_id,
                "sender": sender_name,
                "seed": seed,
                "q": q,
                "max_words": max_words,
                "approval_reward": approval_reward,
                "lie_penalty": lie_penalty,
                "audit_budget": audit_budget,
                "episode_idx": episode_idx,
                **true_state_dict,
                "sender_action": message.action,
                "reported_intent": message.proposal.reported_intent,
                "reported_risk": message.proposal.reported_risk,
                "reported_net_delta": message.proposal.reported_net_delta,
                "reported_hedge_size": message.proposal.reported_hedge_size,
                "claim_intent": claims["intent"],
                "claim_risk_within_limit": claims["risk_within_limit"],
                "claim_net_delta_bounded": claims["net_delta_bounded"],
                "claim_risk_metric": claims["risk_metric"],
                "claim_confidence": claims["confidence"],
                "reasoning_words": reasoning_words,
                "reasoning_chars": reasoning_chars,
                "invalid_output": invalid_output,
                "audited": audit_result.audited,
                "audit_pass": audit_result.audit_pass,
                "audit_fail": audit_result.audit_fail,
                "audit_inconclusive": audit_result.audit_inconclusive,
                "audit_status": audit_result.audit_status,
                "rm_decision": decision,
                "trader_payoff": payoff.trader_payoff,
                "rm_payoff": payoff.rm_payoff,
                "total_welfare": payoff.total_welfare,
                "bad_approval": decision == "APPROVE" and not true_state.compliant,
            }
        )

    return pd.DataFrame(rows)


def _build_sender(sender_spec: object, seed: int, q_idx: int, w_idx: int) -> object:
    """Build a sender instance for a specific condition.

    Supports either a concrete sender object (returned as-is) or a factory.
    Factory can be either `factory()` or `factory(seed, q_idx, w_idx)`.
    """
    if not callable(sender_spec):
        return sender_spec

    try:
        sig = inspect.signature(sender_spec)
        if len(sig.parameters) == 0:
            return sender_spec()  # type: ignore[misc]
        return sender_spec(seed, q_idx, w_idx)  # type: ignore[misc]
    except (TypeError, ValueError):
        # Fall back to no-arg factory if signature introspection fails.
        return sender_spec()  # type: ignore[misc]


def run_experiment(
    senders: Dict[str, object],
    config: ExperimentConfig,
    qs: Iterable[float],
    max_words_list: Iterable[int],
    approval_rewards: Iterable[float],
    lie_penalties: Iterable[float],
    audit_budgets: Iterable[int | None],
    seeds: Iterable[int],
    run_id: str,
    workers: int = 1,
) -> pd.DataFrame:
    workers = int(workers)
    if workers < 1:
        raise ValueError("workers must be >= 1")

    jobs: List[tuple[str, int, int, float, int, int, int, float, int, float, int, int | None]] = []
    for sender_name in senders.keys():
        for seed in seeds:
            for q_idx, q in enumerate(qs):
                for w_idx, max_words in enumerate(max_words_list):
                    for v_idx, approval_reward in enumerate(approval_rewards):
                        for l_idx, lie_penalty in enumerate(lie_penalties):
                            for b_idx, audit_budget in enumerate(audit_budgets):
                                jobs.append(
                                    (sender_name, seed, q_idx, q, w_idx, max_words, 
                                     v_idx, approval_reward, l_idx, lie_penalty, b_idx, audit_budget)
                                )

    def _run_job(job: tuple[str, int, int, float, int, int, int, float, int, float, int, int | None]) -> pd.DataFrame:
        sender_name, seed, q_idx, q, w_idx, max_words, v_idx, approval_reward, l_idx, lie_penalty, b_idx, audit_budget = job
        rng_seed = derive_seed(seed, q_idx, w_idx, v_idx, l_idx, b_idx)
        sender = _build_sender(senders[sender_name], seed, q_idx, w_idx)
        return run_condition(
            sender,
            sender_name,
            config,
            q,
            max_words,
            approval_reward,
            lie_penalty,
            audit_budget,
            seed,
            rng_seed,
            run_id,
            show_progress=(workers == 1),
        )

    if workers == 1:
        frames = [_run_job(job) for job in jobs]
        return pd.concat(frames, ignore_index=True)

    frames: List[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_job, job) for job in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="conditions"):
            frames.append(fut.result())
    return pd.concat(frames, ignore_index=True)
