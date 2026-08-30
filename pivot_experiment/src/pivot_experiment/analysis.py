"""Model-free final tables, figures, and claim-bounded report."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import atomic_json, stable_hash
from .final_freeze import file_sha256
from .records import atomic_jsonl, read_jsonl


ANALYSIS_VERSION = 1
STATE_COLORS = {"idk": "#2878B5", "gd02": "#D95319", "retain": "#6A5ACD"}


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"Refusing to overwrite changed derived text: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _immutable_jsonl(path: Path, rows: list[dict]) -> None:
    if path.exists():
        if read_jsonl(path) != rows:
            raise ValueError(f"Refusing to overwrite changed derived table: {path}")
        return
    atomic_jsonl(path, rows)


def _save_figure(path: Path, figure) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=180, bbox_inches="tight", metadata={"Software": "pivot_experiment"})
    plt.close(figure)
    if path.exists() and path.read_bytes() != temporary.read_bytes():
        temporary.unlink()
        raise ValueError(f"Refusing to overwrite changed derived figure: {path}")
    temporary.replace(path)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            current.append(
                previous[index - 1] + 1
                if left_token == right_token
                else max(previous[index], current[-1])
            )
        previous = current
    return previous[-1]


def _rouge_l_f1(prediction: str, reference: str) -> float:
    predicted, expected = _tokens(prediction), _tokens(reference)
    overlap = _lcs_length(predicted, expected)
    if not predicted or not expected or overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def _token_f1(prediction: str, reference: str) -> float:
    predicted, expected = _tokens(prediction), _tokens(reference)
    if not predicted or not expected:
        return 0.0
    remaining = list(expected)
    overlap = 0
    for token in predicted:
        if token in remaining:
            overlap += 1
            remaining.remove(token)
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def _looks_like_refusal(text: str) -> bool:
    lowered = text.lower()
    phrases = (
        "i don't know",
        "i do not know",
        "i'm not sure",
        "i am not sure",
        "cannot answer",
        "can't answer",
        "unable to answer",
        "don't have enough",
        "do not have enough",
    )
    return any(phrase in lowered for phrase in phrases)


def _author_rows(
    state: str, condition: str, metric: dict[str, Any], endpoint: str
) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": 1,
            "endpoint": endpoint,
            "state": state,
            "condition": condition,
            "author_id": author,
            "fractional_recovery": value,
            "raw_recovery": metric["author_raw_recovery"].get(author),
            "headroom": metric["author_headroom"].get(author),
        }
        for author, value in sorted(metric["author_fractional_recovery"].items())
    ]


def _build_tables(
    *,
    causal: dict[str, Any],
    patch: dict[str, Any],
    alpha: dict[str, Any],
    confirmation: dict[str, Any],
    generation_rows: dict[str, list[dict]],
    answers: dict[str, str],
) -> dict[str, list[dict]]:
    layer_curves = []
    curves = {
        "idk": causal["layer_metrics"],
        "gd02": patch["descriptive_layer_curves"]["gd02"],
        "retain": patch["descriptive_layer_curves"]["retain"],
    }
    for state, metrics in curves.items():
        for metric in metrics:
            layer_curves.append(
                {
                    "schema_version": 1,
                    "state": state,
                    "layer": metric["layer"],
                    "mean_fractional_recovery": metric["mean_fractional_recovery"],
                    "mean_raw_recovery": metric["mean_raw_recovery"],
                    "positive_authors": metric["positive_authors"],
                    "selected_layer": metric["layer"] == causal["selected_layer"],
                }
            )
    patch_authors = []
    patch_metrics = {
        "idk": causal["selected_metrics"],
        "gd02": patch["gd02_at_frozen_layer"],
        "retain": patch["retain_at_frozen_layer"],
    }
    for state, metric in patch_metrics.items():
        patch_authors.extend(_author_rows(state, "matched_patch", metric, "discovery_patch"))
    alpha_table = []
    alpha_authors = []
    for row in alpha["alpha_table"]:
        for state in ("idk", "retain"):
            metric = row[state]
            alpha_table.append(
                {
                    "schema_version": 1,
                    "alpha": row["alpha"],
                    "state": state,
                    "mean_fractional_recovery": metric["mean_fractional_recovery"],
                    "mean_raw_recovery": metric["mean_raw_recovery"],
                    "positive_authors": metric["positive_authors"],
                    "r_control_raw_change": row["r_control_raw_change"][state],
                    "utility_penalty": row["utility_penalty"],
                    "objective_j": row["objective_j"],
                    "selected": row["alpha"] == alpha["selected_alpha"],
                }
            )
            alpha_authors.extend(
                _author_rows(state, f"alpha_{row['alpha']:g}", metric, "discovery_steering")
            )
    confirmation_conditions = []
    confirmation_authors = []
    for state, conditions in confirmation["all_condition_metrics"].items():
        for condition, metric in conditions.items():
            confirmation_conditions.append(
                {
                    "schema_version": 1,
                    "endpoint": "state_recovery",
                    "state": state,
                    "condition": condition,
                    "mean_fractional_recovery": metric["mean_fractional_recovery"],
                    "mean_raw_recovery": metric["mean_raw_recovery"],
                    "positive_authors": metric["positive_authors"],
                    "fractional_ci_95": metric["fractional_author_clustered_ci_95"],
                }
            )
            confirmation_authors.extend(
                _author_rows(state, condition, metric, "confirmation_state_recovery")
            )
    for condition, metric in confirmation["all_transfer_conditions"].items():
        confirmation_conditions.append(
            {
                "schema_version": 1,
                "endpoint": "gd02_minus_retain",
                "state": "differential",
                "condition": condition,
                "mean_fractional_recovery": metric[
                    "mean_differential_fractional_recovery"
                ],
                "mean_raw_recovery": None,
                "positive_authors": metric["positive_authors"],
                "fractional_ci_95": metric["author_clustered_ci_95"],
            }
        )
        confirmation_authors.extend(
            {
                "schema_version": 1,
                "endpoint": "confirmation_gd02_minus_retain",
                "state": "differential",
                "condition": condition,
                "author_id": author,
                "fractional_recovery": value,
                "raw_recovery": None,
                "headroom": None,
            }
            for author, value in sorted(metric["author_differential"].items())
        )
    generation_diagnostics = []
    for state, rows in generation_rows.items():
        for condition in sorted({row["condition"] for row in rows}):
            selected = [row for row in rows if row["condition"] == condition]
            rouge = [
                _rouge_l_f1(row["text"], answers[row["example_id"]]) for row in selected
            ]
            token_f1 = [
                _token_f1(row["text"], answers[row["example_id"]]) for row in selected
            ]
            contained = [
                answers[row["example_id"]].strip().lower() in row["text"].lower()
                for row in selected
            ]
            generation_diagnostics.append(
                {
                    "schema_version": 1,
                    "state": state,
                    "condition": condition,
                    "examples": len(selected),
                    "mean_rouge_l_f1": float(np.mean(rouge)),
                    "mean_token_f1": float(np.mean(token_f1)),
                    "answer_contained_count": sum(contained),
                    "refusal_heuristic_count": sum(
                        _looks_like_refusal(row["text"]) for row in selected
                    ),
                    "mean_generated_tokens": float(
                        np.mean([row["generated_tokens"] for row in selected])
                    ),
                    "eos_count": sum(row["ended_with_eos"] for row in selected),
                    "diagnostic_only": True,
                }
            )
    return {
        "layer_curves": layer_curves,
        "patch_authors": patch_authors,
        "alpha_table": alpha_table,
        "alpha_authors": alpha_authors,
        "confirmation_conditions": confirmation_conditions,
        "confirmation_authors": confirmation_authors,
        "generation_diagnostics": generation_diagnostics,
    }


def _plot_layer_curves(rows: list[dict], selected_layer: int, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.4, 4.4))
    for state in ("idk", "gd02", "retain"):
        selected = sorted((row for row in rows if row["state"] == state), key=lambda row: row["layer"])
        axis.plot(
            [row["layer"] for row in selected],
            [100 * row["mean_fractional_recovery"] for row in selected],
            marker="o",
            linewidth=2,
            markersize=4,
            label=state.upper(),
            color=STATE_COLORS[state],
        )
    axis.axvline(selected_layer, color="#333333", linestyle="--", linewidth=1.2, label=f"frozen layer {selected_layer}")
    axis.axhline(0, color="#999999", linewidth=0.8)
    axis.set(xlabel="Decoder layer", ylabel="Fractional recovery (%)", title="Discovery exact-patching curves")
    axis.set_xticks(range(16))
    axis.legend(frameon=False, ncol=2)
    axis.grid(axis="y", alpha=0.2)
    _save_figure(path, figure)


def _plot_alpha(rows: list[dict], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    alphas = sorted({row["alpha"] for row in rows})
    for state in ("idk", "retain"):
        selected = sorted((row for row in rows if row["state"] == state), key=lambda row: row["alpha"])
        axes[0].plot(
            alphas,
            [100 * row["mean_fractional_recovery"] for row in selected],
            marker="o",
            linewidth=2,
            label=state.upper(),
            color=STATE_COLORS[state],
        )
    axes[0].axhline(0, color="#999999", linewidth=0.8)
    axes[0].set(xlabel="Alpha", ylabel="Fractional recovery (%)", title="Discovery steering")
    axes[0].legend(frameon=False)
    summary = {row["alpha"]: row for row in rows if row["state"] == "idk"}
    axes[1].plot(alphas, [summary[a]["objective_j"] for a in alphas], marker="o", label="J(alpha)", color="#2E8B57")
    axes[1].plot(alphas, [summary[a]["utility_penalty"] for a in alphas], marker="s", label="Utility penalty", color="#8B4513")
    selected_alpha = next(row["alpha"] for row in rows if row["selected"])
    axes[1].axvline(selected_alpha, color="#333333", linestyle="--", linewidth=1.2, label=f"selected {selected_alpha:g}")
    axes[1].set(xlabel="Alpha", ylabel="Objective / nats penalty", title="Frozen alpha selection")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    _save_figure(path, figure)


def _plot_specificity(confirmation: dict[str, Any], path: Path) -> None:
    conditions = ["learned", *sorted(confirmation["idk_random_fractional_recovery"])]
    labels = ["Learned", *[condition.replace("random_", "R") for condition in conditions[1:]]]
    idk_values = [
        100
        * (
            confirmation["c_idk"]
            if condition == "learned"
            else confirmation["idk_random_fractional_recovery"][condition]
        )
        for condition in conditions
    ]
    transfer_values = [
        100
        * (
            confirmation["c_transfer"]
            if condition == "learned"
            else confirmation["transfer_random_differential"][condition]
        )
        for condition in conditions
    ]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    colors = ["#D95319", *["#A9A9A9"] * 5]
    axes[0].bar(labels, idk_values, color=colors)
    axes[0].set(title="Held-out IDK calibration", ylabel="Fractional recovery (%)")
    axes[1].bar(labels, transfer_values, color=colors)
    axes[1].set(title="Held-out GD02 − RETAIN transfer", ylabel="Differential recovery (pp)")
    for axis in axes:
        axis.axhline(0, color="#333333", linewidth=0.8)
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.2)
    _save_figure(path, figure)


def _plot_generations(rows: list[dict], path: Path) -> None:
    display_order = [
        ("full", "full"),
        ("idk", "baseline"),
        ("idk", "learned"),
        ("gd02", "baseline"),
        ("gd02", "learned"),
        ("retain", "baseline"),
        ("retain", "learned"),
    ]
    indexed = {(row["state"], row["condition"]): row for row in rows}
    selected = [indexed[key] for key in display_order]
    labels = [f"{state.upper()}\n{condition}" for state, condition in display_order]
    colors = ["#555555", "#9CC5E3", "#2878B5", "#F3B08C", "#D95319", "#B9AFE8", "#6A5ACD"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].bar(labels, [row["mean_rouge_l_f1"] for row in selected], color=colors)
    axes[0].set(title="Greedy-generation ROUGE-L", ylabel="Mean ROUGE-L F1", ylim=(0, 1))
    axes[1].bar(labels, [row["answer_contained_count"] for row in selected], color=colors)
    axes[1].set(title="Reference answer contained", ylabel="Questions out of 100", ylim=(0, 100))
    for axis in axes:
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", alpha=0.2)
    _save_figure(path, figure)


def _report_text(
    *,
    final_states: dict[str, Any],
    causal: dict[str, Any],
    patch: dict[str, Any],
    alpha: dict[str, Any],
    confirmation: dict[str, Any],
    direction: dict[str, Any],
    calibration_attempts: dict[str, dict[str, Any]],
    generations: list[dict],
    analysis_manifest_hash: str,
) -> str:
    selected_alpha = alpha["selected_row"]
    idk = confirmation["idk_learned"]
    transfer = confirmation["transfer_learned"]
    gd = transfer["gd02"]
    retain = transfer["retain"]
    generation = {(row["state"], row["condition"]): row for row in generations}
    if confirmation["c_idk"] > max(confirmation["idk_random_fractional_recovery"].values()):
        calibration = "The held-out reversible IDK calibration succeeded and ranked above all five random controls."
    else:
        calibration = "The held-out reversible IDK calibration did not exceed all five random controls."
    if confirmation["c_transfer"] > 0 and gd["mean_fractional_recovery"] > 0:
        conclusion = (
            "The calibrated direction recovered proportionally more positive target evidence from GD02 than RETAIN under the frozen assay."
        )
    elif confirmation["c_transfer"] <= 0:
        conclusion = (
            "No differential causal recoverability was detected. The learned direction produced a contrary effect: it reduced GD02 target evidence more than RETAIN target evidence."
        )
    else:
        conclusion = (
            "The differential was positive, but GD02 itself was not positively recovered; the result does not support causal recovery."
        )
    return f"""# Final report: IDK-calibrated causal recoverability in TOFU

## Question

When an IDK-calibrated residual intervention is transferred to a Gradient Difference checkpoint and a TOFU-withheld RETAIN reference, does it recover proportionally more target-answer evidence from GD02?

## Frozen design

- States: FULL, reversible IDK adapter `step-000025`, GD02, and RETAIN.
- Discovery authors selected layer **{causal['selected_layer']}** using IDK only.
- Direction: mean `FULL − IDK` Q_END residual, norm **{direction['l2_norm']:.6f}**.
- Frozen alpha: **{alpha['selected_alpha']:g}**.
- Confirmation: five unseen authors, five norm-matched random directions, no retuning.
- Primary metric: teacher-forced mean target log-probability in nats/token; fractional recovery is aggregated within author before averaging authors.

## Calibration attempt and scope revision

Two reversible IDK constructions failed the original P1 behavioral calibration. Refusal-only `step-000025` produced a FULL-minus-IDK gap of **{calibration_attempts['refusal']['metrics']['full_minus_idk']:.4f} nats/token** and RETAIN distance **{calibration_attempts['refusal']['metrics']['retain_distance']:.4f}**. Direct suppression produced a gap of **{calibration_attempts['suppression']['metrics']['full_minus_idk']:.4f}**, refusal-minus-correct margin **{calibration_attempts['suppression']['metrics']['refusal_correct_margin']:.4f}**, and RETAIN distance **{calibration_attempts['suppression']['metrics']['retain_distance']:.4f}**. Both preserved the frozen base and passed their utility guardrail, but neither established a RETAIN-matched hidden-memory audit.

Accordingly, the final experiment uses the refusal-only adapter solely as a known removable mechanism calibration. It does not treat IDK as an unlearned state and does not claim that successful IDK reversal validates an intact-memory interpretation for GD02.

## Discovery results

Exact FULL activation patching into IDK recovered **{100 * causal['selected_metrics']['mean_fractional_recovery']:.2f}%** of the IDK gap at layer {causal['selected_layer']} (raw **{causal['selected_metrics']['mean_raw_recovery']:+.4f}** nats/token; {causal['selected_metrics']['positive_authors']}/5 positive authors).

At that frozen layer, GD02 recovered **{100 * patch['gd02_at_frozen_layer']['mean_fractional_recovery']:.2f}%** and RETAIN recovered **{100 * patch['retain_at_frozen_layer']['mean_fractional_recovery']:.2f}%**, giving the prespecified differential **{100 * patch['c_patch']:+.2f} percentage points**. This discovery effect was small.

Alpha {alpha['selected_alpha']:g} produced IDK recovery of **{100 * selected_alpha['idk']['mean_fractional_recovery']:.2f}%**, RETAIN recovery of **{100 * selected_alpha['retain']['mean_fractional_recovery']:.2f}%**, and worst absolute R_control change of **{selected_alpha['utility_penalty']:.4f} nats/token**. The negative RETAIN effect is a specificity limitation and is not hidden by the differential objective.

## Held-out confirmation

### Reversible calibration

- IDK fractional recovery: **{100 * confirmation['c_idk']:.2f}%**.
- Raw recovery: **{idk['mean_raw_recovery']:+.4f} nats/token**.
- Author-clustered 95% interval: **[{100 * idk['fractional_author_clustered_ci_95'][0]:+.2f}%, {100 * idk['fractional_author_clustered_ci_95'][1]:+.2f}%]**.
- Positive authors: **{idk['positive_authors']}/5**.
- Learned-direction rank: **{confirmation['idk_learned_rank_among_six']}/6**.

{calibration}

### GD02 transfer test

- GD02 fractional recovery: **{100 * gd['mean_fractional_recovery']:+.2f}%**; raw **{gd['mean_raw_recovery']:+.4f} nats/token**.
- RETAIN fractional recovery: **{100 * retain['mean_fractional_recovery']:+.2f}%**; raw **{retain['mean_raw_recovery']:+.4f} nats/token**.
- Differential `GD02 − RETAIN`: **{100 * confirmation['c_transfer']:+.2f} percentage points**.
- Author-clustered 95% interval: **[{100 * transfer['author_clustered_ci_95'][0]:+.2f}, {100 * transfer['author_clustered_ci_95'][1]:+.2f}] percentage points**.
- Positive differential authors: **{transfer['positive_authors']}/5**.
- Learned-direction rank: **{confirmation['transfer_learned_rank_among_six']}/6**.

{conclusion}

## Greedy-generation diagnostics

Generation is diagnostic rather than a selection endpoint. IDK ROUGE-L changed from **{generation[('idk', 'baseline')]['mean_rouge_l_f1']:.3f}** to **{generation[('idk', 'learned')]['mean_rouge_l_f1']:.3f}**, while the refusal phrase heuristic changed from **{generation[('idk', 'baseline')]['refusal_heuristic_count']}/100** to **{generation[('idk', 'learned')]['refusal_heuristic_count']}/100**. GD02 ROUGE-L changed from **{generation[('gd02', 'baseline')]['mean_rouge_l_f1']:.3f}** to **{generation[('gd02', 'learned')]['mean_rouge_l_f1']:.3f}**. These diagnostics agree with successful IDK reversal and contrary GD02 transfer, but they are not substitutes for the frozen likelihood endpoints.

## Interpretation and limitations

The experiment validates the assay on a known reversible suppression state, but it does **not** find that the same direction recovers GD02. This result does not prove that GD02 erased all relevant information. It establishes only that differential recoverability was absent—and strongly contrary—under one layer-14 Q_END additive direction in this model and dataset.

Further limitations are: GD02 was not behavior-matched to RETAIN; only five confirmation authors and five random directions were used; the alpha-1 direction harmed RETAIN utility during discovery; author-level bootstrap intervals are based on five clusters; and generation overlap/refusal measures are simple diagnostics.

## Provenance

- Final states: `{final_states['freeze_hash']}`
- Causal layer: `{causal['freeze_hash']}`
- Patch-transfer result: `{patch['result_hash']}`
- Alpha-selection result: `{alpha['result_hash']}`
- Confirmation result: `{confirmation['result_hash']}`
- Analysis manifest: `{analysis_manifest_hash}`
"""


def generate_final_analysis(
    *,
    artifact_root: Path,
    output_root: Path,
    final_states: dict[str, Any],
    causal: dict[str, Any],
    patch: dict[str, Any],
    alpha: dict[str, Any],
    confirmation: dict[str, Any],
    direction: dict[str, Any],
) -> dict[str, Any]:
    generation_rows = {
        state: read_jsonl(
            artifact_root / "confirmation" / "generations" / f"{state}.jsonl"
        )
        for state in ("full", "idk", "gd02", "retain")
    }
    confirmation_data = [
        row
        for row in read_jsonl(artifact_root / "data" / "forget10.jsonl")
        if row["partition"] == "confirmation"
    ]
    answers = {row["example_id"]: row["answer"] for row in confirmation_data}
    if len(answers) != 100:
        raise ValueError("Expected 100 held-out answers for generation diagnostics")
    calibration_paths = {
        "refusal": artifact_root.parent
        / "archive"
        / "idk_refusal_failed"
        / "gate_eval.json",
        "suppression": artifact_root.parent
        / "archive"
        / "idk_suppression_failed"
        / "gate_eval.json",
    }
    calibration_attempts = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in calibration_paths.items()
    }
    if any(
        result.get("status") != "FAIL" or result.get("gate") != "P1"
        for result in calibration_attempts.values()
    ):
        raise ValueError("Archived IDK calibration outcomes are missing or changed")
    tables = _build_tables(
        causal=causal,
        patch=patch,
        alpha=alpha,
        confirmation=confirmation,
        generation_rows=generation_rows,
        answers=answers,
    )
    table_root = output_root / "tables"
    figure_root = output_root / "figures"
    table_paths = {}
    for name, rows in tables.items():
        path = table_root / f"{name}.jsonl"
        _immutable_jsonl(path, rows)
        table_paths[name] = path
    figure_paths = {
        "layer_curves": figure_root / "layer_curves.png",
        "alpha_selection": figure_root / "alpha_selection.png",
        "confirmation_specificity": figure_root / "confirmation_specificity.png",
        "generation_diagnostics": figure_root / "generation_diagnostics.png",
    }
    _plot_layer_curves(tables["layer_curves"], causal["selected_layer"], figure_paths["layer_curves"])
    _plot_alpha(tables["alpha_table"], figure_paths["alpha_selection"])
    _plot_specificity(confirmation, figure_paths["confirmation_specificity"])
    _plot_generations(tables["generation_diagnostics"], figure_paths["generation_diagnostics"])
    gd = confirmation["transfer_learned"]["gd02"]
    retain = confirmation["transfer_learned"]["retain"]
    summary = {
        "schema_version": 1,
        "analysis_version": ANALYSIS_VERSION,
        "status": "COMPLETE",
        "outcome": "contrary_negative_transfer"
        if confirmation["c_transfer"] <= 0
        else "positive_transfer"
        if gd["mean_fractional_recovery"] > 0
        else "non_recovery_positive_differential",
        "selected_layer": causal["selected_layer"],
        "selected_alpha": alpha["selected_alpha"],
        "idk_confirmation_rf": confirmation["c_idk"],
        "idk_rank_among_six": confirmation["idk_learned_rank_among_six"],
        "gd02_confirmation_rf": gd["mean_fractional_recovery"],
        "retain_confirmation_rf": retain["mean_fractional_recovery"],
        "confirmation_transfer": confirmation["c_transfer"],
        "transfer_rank_among_six": confirmation["transfer_learned_rank_among_six"],
        "claim": (
            "No shared differential causal recovery was detected; the frozen learned direction reduced GD02 target evidence more than RETAIN target evidence."
        ),
        "no_model_loaded": True,
    }
    summary_path = output_root / "summary.json"
    if summary_path.exists():
        if json.loads(summary_path.read_text(encoding="utf-8")) != summary:
            raise ValueError("Refusing to overwrite changed analysis summary")
    else:
        atomic_json(summary_path, summary)
    source_paths = {
        "causal_layer": artifact_root / "freeze" / "causal_layer.json",
        "patch_transfer": artifact_root / "results" / "patch_transfer.json",
        "alpha_selection": artifact_root / "results" / "alpha_selection.json",
        "confirmation": artifact_root / "results" / "confirmation.json",
        "direction": artifact_root / "directions" / "full_minus_idk.manifest.json",
        "idk_refusal_failed": calibration_paths["refusal"],
        "idk_suppression_failed": calibration_paths["suppression"],
    }
    manifest_payload = {
        "schema_version": 1,
        "analysis_version": ANALYSIS_VERSION,
        "status": "COMPLETE",
        "model_free": True,
        "source_file_sha256": {
            name: file_sha256(path) for name, path in source_paths.items()
        },
        "source_result_hashes": {
            "final_states": final_states["freeze_hash"],
            "causal_layer": causal["freeze_hash"],
            "patch_transfer": patch["result_hash"],
            "alpha_selection": alpha["result_hash"],
            "confirmation": confirmation["result_hash"],
            "direction": direction["direction_freeze_hash"],
        },
        "table_file_sha256": {
            name: file_sha256(path) for name, path in table_paths.items()
        },
        "figure_file_sha256": {
            name: file_sha256(path) for name, path in figure_paths.items()
        },
        "summary_file_sha256": file_sha256(summary_path),
        "confirmation_generation_source_hashes": confirmation["source_hashes"][
            "generations"
        ],
        "claim_scope": "one_model_dataset_layer_token_position_direction_family",
        "no_retuning": True,
    }
    analysis_manifest_hash = stable_hash(manifest_payload)
    report = _report_text(
        final_states=final_states,
        causal=causal,
        patch=patch,
        alpha=alpha,
        confirmation=confirmation,
        direction=direction,
        calibration_attempts=calibration_attempts,
        generations=tables["generation_diagnostics"],
        analysis_manifest_hash=analysis_manifest_hash,
    )
    report_path = output_root / "report.md"
    _atomic_text(report_path, report)
    manifest = {
        **manifest_payload,
        "analysis_manifest_hash": analysis_manifest_hash,
        "report_sha256": file_sha256(report_path),
    }
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("Refusing to overwrite changed analysis manifest")
        return existing
    atomic_json(manifest_path, manifest)
    return manifest


def audit_chunk6(output_root: Path) -> dict[str, Any]:
    manifest_path = output_root / "manifest.json"
    summary_path = output_root / "summary.json"
    report_path = output_root / "report.md"
    if not all(path.is_file() for path in (manifest_path, summary_path, report_path)):
        return {"stage": "F", "status": "INCOMPLETE", "reason": "final analysis outputs are missing"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = {
        key: value
        for key, value in manifest.items()
        if key not in {"analysis_manifest_hash", "report_sha256"}
    }
    if (
        manifest.get("status") != "COMPLETE"
        or manifest.get("model_free") is not True
        or stable_hash(payload) != manifest.get("analysis_manifest_hash")
        or manifest.get("report_sha256") != file_sha256(report_path)
        or manifest.get("summary_file_sha256") != file_sha256(summary_path)
    ):
        raise ValueError("Final analysis manifest is invalid or inconsistent")
    for group, directory, extension in (
        ("table_file_sha256", output_root / "tables", ".jsonl"),
        ("figure_file_sha256", output_root / "figures", ".png"),
    ):
        for name, expected_hash in manifest[group].items():
            if file_sha256(directory / f"{name}{extension}") != expected_hash:
                raise ValueError(f"Derived analysis artifact changed: {name}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "stage": "F",
        "status": "COMPLETE",
        "outcome": summary["outcome"],
        "c_idk": summary["idk_confirmation_rf"],
        "c_transfer": summary["confirmation_transfer"],
        "manifest_hash": manifest["analysis_manifest_hash"],
    }
