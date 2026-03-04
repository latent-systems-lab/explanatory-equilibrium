from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import ExperimentConfig


def _safe_mean(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(series.mean())


def _format_mean_std(mean_val: float, std_val: float, precision: int = 3) -> str:
    if np.isnan(mean_val):
        return "NA"
    if np.isnan(std_val):
        return f"{mean_val:.{precision}f}"
    return f"{mean_val:.{precision}f} +/- {std_val:.{precision}f}"


def build_summary(logs: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    logs = logs.copy()
    logs["intent_truth"] = logs["claim_intent"] == logs["true_intent"]
    logs["risk_truth"] = (logs["reported_risk"] - logs["true_risk"]).abs() <= config.risk_tol
    logs["delta_truth"] = (
        logs["reported_net_delta"] - logs["true_net_delta"]
    ).abs() <= config.delta_tol
    logs["compliance_claim_truth"] = logs["claim_risk_within_limit"] == (
        logs["true_risk"] <= logs["risk_limit_L"]
    )

    def _group_metrics(group: pd.DataFrame) -> pd.Series:
        ambiguous = group[group["episode_type"] == "ambiguous_hedge"]
        return pd.Series(
            {
                "episodes": len(group),
                "ambiguous_episodes": len(ambiguous),
                "intent_truth": group["intent_truth"].mean(),
                "risk_truth": group["risk_truth"].mean(),
                "delta_truth": group["delta_truth"].mean(),
                "compliance_claim_truth": group["compliance_claim_truth"].mean(),
                "audit_fail_rate": group["audit_fail"].mean(),
                "approval_rate": (group["rm_decision"] == "APPROVE").mean(),
                "bad_approval_rate": group["bad_approval"].mean(),
                "avg_welfare": group["total_welfare"].mean(),
                "avg_reasoning_words": group["reasoning_words"].mean(),
                "ambiguous_approval_rate": _safe_mean(
                    (ambiguous["rm_decision"] == "APPROVE").astype(float)
                ),
                "ambiguous_welfare": _safe_mean(ambiguous["total_welfare"]),
            }
        )

    # Apply metrics calculation
    seed_results = []
    groupby_cols = ["sender", "seed", "q", "max_words", "approval_reward", "lie_penalty", "audit_budget"]
    for group_keys, group in logs.groupby(groupby_cols, dropna=False):
        metrics = _group_metrics(group)
        row_dict = metrics.to_dict()
        row_dict["sender"] = group_keys[0]
        row_dict["seed"] = group_keys[1]
        row_dict["q"] = group_keys[2]
        row_dict["max_words"] = group_keys[3]
        row_dict["approval_reward"] = group_keys[4]
        row_dict["lie_penalty"] = group_keys[5]
        row_dict["audit_budget"] = group_keys[6]
        seed_results.append(row_dict)
    
    seed_summary = pd.DataFrame(seed_results)

    metric_cols = [
        "episodes",
        "ambiguous_episodes",
        "intent_truth",
        "risk_truth",
        "delta_truth",
        "compliance_claim_truth",
        "audit_fail_rate",
        "approval_rate",
        "bad_approval_rate",
        "avg_welfare",
        "avg_reasoning_words",
        "ambiguous_approval_rate",
        "ambiguous_welfare",
    ]

    # Aggregate across seeds for each (sender, q, max_words, approval_reward, lie_penalty, audit_budget)
    aggregated_data = []
    agg_groupby_cols = ["sender", "q", "max_words", "approval_reward", "lie_penalty", "audit_budget"]
    for group_keys, group in seed_summary.groupby(agg_groupby_cols, dropna=False):
        row = {
            "sender": group_keys[0],
            "q": group_keys[1],
            "max_words": group_keys[2],
            "approval_reward": group_keys[3],
            "lie_penalty": group_keys[4],
            "audit_budget": group_keys[5],
        }
        for col in metric_cols:
            row[f"{col}_mean"] = group[col].mean()
            row[f"{col}_std"] = group[col].std(ddof=0)
        aggregated_data.append(row)
    
    aggregated = pd.DataFrame(aggregated_data)

    sender_values = aggregated["sender"].unique().tolist()
    index_cols = ["q", "max_words", "approval_reward", "lie_penalty", "audit_budget"]
    wide = aggregated.set_index(index_cols)
    if len(sender_values) > 1:
        wide = wide.pivot_table(index=index_cols, columns="sender")
        wide.columns = [f"{col}_{sender}" for col, sender in wide.columns]
        wide = wide.reset_index()
    else:
        wide = wide.reset_index()

    if {"llm", "llm_no_expl"}.issubset(sender_values):
        pivot_index = ["seed", "q", "max_words", "approval_reward", "lie_penalty", "audit_budget"]
        pivot = seed_summary.pivot_table(
            index=pivot_index,
            columns="sender",
            values=["ambiguous_approval_rate", "ambiguous_welfare"],
            dropna=False,
        )
        # If there are zero ambiguous episodes everywhere (e.g. ambiguous_rate=0 with few episodes),
        # these metrics can be all-NaN; pandas may drop them unless dropna=False. Still, be defensive.
        have_metrics = isinstance(pivot.columns, pd.MultiIndex) and {
            "ambiguous_approval_rate",
            "ambiguous_welfare",
        }.issubset(set(pivot.columns.get_level_values(0)))
        have_senders = (
            isinstance(pivot.columns, pd.MultiIndex)
            and "llm" in set(pivot.columns.get_level_values(1))
            and "llm_no_expl" in set(pivot.columns.get_level_values(1))
        )

        if have_metrics and have_senders:
            effect = (
                pivot["ambiguous_approval_rate"]["llm"]
                - pivot["ambiguous_approval_rate"]["llm_no_expl"]
            ).reset_index(name="explanation_effect")
            gain = (
                pivot["ambiguous_welfare"]["llm"]
                - pivot["ambiguous_welfare"]["llm_no_expl"]
            ).reset_index(name="explanation_gain_welfare")
            diffs = effect.merge(gain, on=pivot_index, how="outer")
        else:
            diffs_cols = pivot_index.copy()
            diffs = seed_summary[diffs_cols].drop_duplicates().copy()
            diffs["explanation_effect"] = np.nan
            diffs["explanation_gain_welfare"] = np.nan

        groupby_cols_diff = ["q", "max_words", "approval_reward", "lie_penalty", "audit_budget"]
        diff_summary = diffs.groupby(groupby_cols_diff, as_index=False, dropna=False).agg(
            {
                "explanation_effect": ["mean", "std"],
                "explanation_gain_welfare": ["mean", "std"],
            }
        )
        # `groupby(..., as_index=False).agg(...)` produces MultiIndex columns like
        # ('q','') and ('explanation_effect','mean'). Keep group keys as 'q'/'max_words'
        # so the merge below works.
        if isinstance(diff_summary.columns, pd.MultiIndex):
            new_cols = []
            for col, stat in diff_summary.columns:
                if stat is None or str(stat).strip() == "":
                    new_cols.append(str(col))
                else:
                    new_cols.append(f"{col}_{stat}")
            diff_summary.columns = new_cols
        wide = wide.merge(diff_summary, on=groupby_cols_diff, how="left")
    std_cols = [col for col in wide.columns if col.endswith("_std")]
    if std_cols:
        wide[std_cols] = wide[std_cols].fillna(0.0)

    return wide


def _collect_table_rows(summary: pd.DataFrame) -> List[Dict[str, object]]:
    rows = []
    for _, row in summary.sort_values(["q", "max_words"]).iterrows():
        record = {"q": row["q"], "max_words": row["max_words"]}
        
        # Dynamically detect available senders and metrics.
        # Wide format uses `<metric>_mean_<sender>` / `<metric>_std_<sender>`.
        for col in summary.columns:
            if "_mean_" in col:
                base, sender = col.split("_mean_", 1)
                if not sender:
                    continue
                std_col = f"{base}_std_{sender}"
                mean_val = row.get(col, float("nan"))
                std_val = row.get(std_col, float("nan"))
                record[f"{base}_{sender}"] = _format_mean_std(mean_val, std_val)
        
        # Add explanation effect and welfare gain if available
        if "explanation_effect_mean" in summary.columns:
            record["explanation_effect"] = _format_mean_std(
                row.get("explanation_effect_mean", float("nan")),
                row.get("explanation_effect_std", float("nan")),
            )
        if "explanation_gain_welfare_mean" in summary.columns:
            record["explanation_gain_welfare"] = _format_mean_std(
                row.get("explanation_gain_welfare_mean", float("nan")),
                row.get("explanation_gain_welfare_std", float("nan")),
            )
        
        rows.append(record)
    return rows


def write_paper_snippet(summary: pd.DataFrame, outdir: Path) -> None:
    rows = _collect_table_rows(summary)
    if not rows:
        return

    target_row = rows[0]
    for row in rows:
        if abs(row["q"] - 0.3) < 1e-6:
            target_row = row
            break

    # Get values from available columns
    expl_rate = target_row.get("ambiguous_approval_rate_llm", "NA")
    no_expl_rate = target_row.get("ambiguous_approval_rate_llm_no_expl", "NA")
    expl_welfare = target_row.get("ambiguous_welfare_llm", "NA")
    no_expl_welfare = target_row.get("ambiguous_welfare_llm_no_expl", "NA")
    effect = target_row.get("explanation_effect", "NA")
    gain = target_row.get("explanation_gain_welfare", "NA")

    paragraph = (
        "We study a borderline hedging regime where risk is within limits but near the boundary. "
        f"At q={target_row['q']:.1f}, the ambiguous approval rate is {expl_rate} with explanations and "
        f"{no_expl_rate} without explanations. "
        f"Ambiguous welfare is {expl_welfare} with explanations versus {no_expl_welfare} without. "
        f"The explanation effect is {effect} and the welfare gain is {gain}, "
        "indicating that explanations shift decisions under ambiguity rather than merely reflecting compliance."
    )

    table_lines = [
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "q & Ambig. approval (expl) & Ambig. approval (no-expl) & Ambig. welfare (expl) & "
        "Ambig. welfare (no-expl) & Expl. effect & Expl. welfare gain \\",
        "\\midrule",
    ]
    for row in rows:
        row_vals = [f"{row['q']:.1f}"]
        for col_key in ["ambiguous_approval_rate_llm", "ambiguous_approval_rate_llm_no_expl",
                        "ambiguous_welfare_llm", "ambiguous_welfare_llm_no_expl",
                        "explanation_effect", "explanation_gain_welfare"]:
            row_vals.append(str(row.get(col_key, "NA")))
        table_lines.append(" & ".join(row_vals) + " \\")
    
    table_lines.extend(["\\bottomrule", "\\end{tabular}"])

    content = "\n".join([
        "## Results",
        "",
        paragraph,
        "",
        "```latex",
        "\n".join(table_lines),
        "```",
        "",
    ])

    outpath = outdir / "paper_snippet.md"
    outpath.write_text(content)
