from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from .analysis import build_summary, write_paper_snippet
from .config import ExperimentConfig
from .plots import make_plots
from .sender_llm import LLMNoExplanationSender, LLMTraderSender
from .sender_mock import MockTraderSender
from .simulation import run_experiment
from .rng import derive_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Suppress noisy loggers
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run explanatory equilibrium finance experiment")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run simulations")
    run.add_argument("--outdir", required=True)
    run.add_argument("--episodes", type=int, default=200)
    run.add_argument("--seed", type=int, default=123)
    run.add_argument("--seeds", type=str, default=None)
    run.add_argument("--qs", type=str, default="0,0.1,0.3,0.6,1.0")
    run.add_argument("--max_words", type=str, default="30,60,120")
    run.add_argument("--approval_rewards", type=str, default="1.0", help="Comma-separated V values (Experiment 1)")
    run.add_argument("--lie_penalties", type=str, default="2.0", help="Comma-separated L values (Experiment 1)")
    run.add_argument("--audit_budgets", type=str, default="None", help="Comma-separated B values or 'None' (Experiment 2)")
    run.add_argument("--sender", choices=["mock", "llm", "llm_no_expl"], default="mock")
    run.add_argument("--senders", type=str, default=None)
    run.add_argument("--ambiguous_rate", type=float, default=0.0)
    run.add_argument("--model", type=str, default=None)
    run.add_argument("--cache_path", type=str, default=None)
    run.add_argument("--llm_provider", type=str, default="openai", choices=["openai", "gemini"])
    run.add_argument("--workers", type=int, default=1, help="Number of parallel workers for condition runs")
    run.add_argument("--debug", action="store_true", help="Enable debug logging")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logger = logging.getLogger(__name__)
    logger.info("Starting experiment run")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    qs = [float(q) for q in args.qs.split(",") if q.strip()]
    max_words_list = [int(w) for w in args.max_words.split(",") if w.strip()]
    approval_rewards = [float(v) for v in args.approval_rewards.split(",") if v.strip()]
    lie_penalties = [float(l) for l in args.lie_penalties.split(",") if l.strip()]
    
    # Parse audit_budgets: support "None" or integer values
    audit_budgets = []
    for b in args.audit_budgets.split(","):
        b = b.strip()
        if b.lower() == "none":
            audit_budgets.append(None)
        elif b:
            audit_budgets.append(int(b))

    config = ExperimentConfig(
        episodes=args.episodes,
        qs=tuple(qs),
        max_words_list=tuple(max_words_list),
        approval_rewards=tuple(approval_rewards),
        lie_penalties=tuple(lie_penalties),
        audit_budgets=tuple(audit_budgets),
        random_seed=args.seed,
        ambiguous_rate=args.ambiguous_rate,
    )

    sender_names = [args.sender]
    if args.senders:
        sender_names = [name.strip() for name in args.senders.split(",") if name.strip()]

    seeds = [args.seed]
    if args.seeds:
        seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]

    senders: dict[str, object] = {}
    llm_params: dict[str, object] = {}
    if any(name in {"llm", "llm_no_expl"} for name in sender_names):
        provider = args.llm_provider or config.llm_provider
        if provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY not set")
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set")
        if provider == "gemini":
            model = args.model or "gemini-2.5-flash"
        else:
            model = args.model or config.sender_model
        cache_path = args.cache_path or config.cache_path
        llm_params = {"api_key": api_key, "model": model, "cache_path": cache_path, "provider": provider}

    for name in sender_names:
        if name == "mock":
            # Seed sender RNG per condition for reproducibility, independent of parallel scheduling.
            senders[name] = lambda seed, q_idx, w_idx: MockTraderSender(seed=derive_seed(seed, 999, q_idx, w_idx))
        elif name == "llm":
            senders[name] = lambda *_: LLMTraderSender(**llm_params)
        elif name == "llm_no_expl":
            senders[name] = lambda *_: LLMNoExplanationSender(LLMTraderSender(**llm_params))
        else:
            raise ValueError(f"Unknown sender: {name}")

    run_id = outdir.name
    logger.info(
        "Running experiment with %d q values, %d word limits, %d V values, %d L values, %d B values, %d episodes, %d seeds, %d sender modes",
        len(qs),
        len(max_words_list),
        len(approval_rewards),
        len(lie_penalties),
        len(audit_budgets),
        args.episodes,
        len(seeds),
        len(senders),
    )
    logs = run_experiment(
        senders,
        config,
        qs,
        max_words_list,
        approval_rewards,
        lie_penalties,
        audit_budgets,
        seeds,
        run_id,
        args.workers,
    )

    logs_path = outdir / "logs.csv"
    logs.to_csv(logs_path, index=False)
    logger.info("Saved logs to %s", logs_path)

    summary = build_summary(logs, config)
    summary_path = outdir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("Saved summary to %s", summary_path)

    make_plots(summary, outdir)
    logger.info("Saved plots to %s", outdir)

    write_paper_snippet(summary, outdir)
    logger.info("Saved paper snippet to %s", outdir)

    print("\n" + "="*60)
    print("RUN COMPLETE")
    print("="*60)
    print(summary)
    print("="*60)


if __name__ == "__main__":
    main()
