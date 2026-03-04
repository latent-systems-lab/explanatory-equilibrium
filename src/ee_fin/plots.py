from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def _save_plot(fig: plt.Figure, outpath: Path) -> None:
    fig.savefig(outpath.with_suffix(".png"))
    fig.savefig(outpath.with_suffix(".pdf"))


def _plot_two_lines(
    subset: pd.DataFrame,
    metric_llm: str,
    metric_no_expl: str,
    ylabel: str,
    title: str,
    outpath: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    subset = subset.sort_values("q")
    ax.plot(subset["q"], subset[metric_llm], marker="o", label="llm")
    ax.plot(subset["q"], subset[metric_no_expl], marker="o", label="llm_no_expl")
    ax.set_xlabel("Audit probability q")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _save_plot(fig, outpath)
    plt.close(fig)


def _plot_for_max_words(
    summary: pd.DataFrame,
    max_words: int,
    plots_dir: Path,
    suffix: str,
) -> None:
    subset = summary[summary["max_words"] == max_words]

    # Check if llm/llm_no_expl columns exist
    if "ambiguous_approval_rate_mean_llm" in summary.columns and "ambiguous_approval_rate_mean_llm_no_expl" in summary.columns:
        _plot_two_lines(
            subset,
            "ambiguous_approval_rate_mean_llm",
            "ambiguous_approval_rate_mean_llm_no_expl",
            "Approval rate on ambiguous episodes",
            "Ambiguous approval rate vs q",
            plots_dir / f"ambiguous_approval_vs_q{suffix}",
        )

        _plot_two_lines(
            subset,
            "ambiguous_welfare_mean_llm",
            "ambiguous_welfare_mean_llm_no_expl",
            "Average welfare on ambiguous episodes",
            "Ambiguous welfare vs q",
            plots_dir / f"ambiguous_welfare_vs_q{suffix}",
        )


def make_plots(summary: pd.DataFrame, outdir: Path) -> None:
    plots_dir = outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    max_words_values = sorted(summary["max_words"].unique().tolist())
    if len(max_words_values) == 1:
        _plot_for_max_words(summary, max_words_values[0], plots_dir, suffix="")
        return

    for max_words in max_words_values:
        suffix = f"_mw{max_words}"
        _plot_for_max_words(summary, max_words, plots_dir, suffix=suffix)
