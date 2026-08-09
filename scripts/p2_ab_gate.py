#!/usr/bin/env python3
"""Apply the preregistered P2 SIGReg/action-conditioning decision gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

from p2_artifacts import resolve_artifact_path, verify_sha256_file


NONFINITE_TRAINING_PATTERN = re.compile(
    r"(?:\bnan\b|\b(?:non[- ]finite|not finite)\b|\binfinite (?:loss|gradient)\b)",
    re.IGNORECASE,
)


def finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return True


def nested(data: dict[str, Any], *keys: str) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def load_run(
    root: Path, seed: int, arm: str, update: int, treatment_arm: str
) -> dict[str, Any]:
    arm_dir = root / f"seed-{seed}" / arm
    report_path = arm_dir / f"eval-update-{update:04d}" / "eval_report.json"
    sha_path = arm_dir / f"eval-update-{update:04d}" / "sha256.txt"
    manifest_path = arm_dir / "manifest.json"
    config_path = arm_dir / "config.json"
    phases_path = arm_dir / "phases.jsonl"
    errors: list[str] = []
    try:
        report = json.loads(report_path.read_text())
    except Exception as exc:  # artifact corruption is itself a failed gate
        report = {}
        errors.append(f"cannot read {report_path}: {exc}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as exc:
        manifest = {}
        errors.append(f"cannot read {manifest_path}: {exc}")
    try:
        config = json.loads(config_path.read_text())
    except Exception as exc:
        config = {}
        errors.append(f"cannot read {config_path}: {exc}")
    expected_schema = (
        "p2.sigreg_geometry_arm.v1"
        if treatment_arm == "pre-rms-spatial"
        else "p2.sigreg_action_arm.v1"
    )
    expected_geometry = {
        "control": "post_rms_pooled_spatial_cells",
        "projector": "pre_rms_global_pool_linear_projector",
        "pre-rms-spatial": "pre_rms_unpooled_spatial_cells",
    }[arm]
    for key, expected in (
        ("schema", expected_schema),
        ("arm", arm),
        ("seed", seed),
        ("physical_batch", 1024),
        ("grad_accum", 1),
        ("sigreg_geometry", expected_geometry),
    ):
        if manifest.get(key) != expected:
            errors.append(f"manifest {key}={manifest.get(key)!r}, expected {expected!r}")
    if not isinstance(manifest.get("complete_through_update"), int) or manifest[
        "complete_through_update"
    ] < update:
        errors.append(
            f"manifest complete_through_update={manifest.get('complete_through_update')!r} "
            f"does not cover update {update}"
        )
    for provenance_key in ("git_sha", "candle_git_sha", "binary_sha256"):
        if not manifest.get(provenance_key):
            errors.append(f"manifest missing {provenance_key}")
    if config_path.is_file():
        actual_config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
        if manifest.get("config_sha256") != actual_config_sha:
            errors.append("manifest config_sha256 does not match config.json")
    shared_contract = {
        "physical_batch": 1024,
        "grad_accum": 1,
        "sigreg_max_rows": 32768,
        "lessons": ["dynamics"],
    }
    for key, expected in shared_contract.items():
        if config.get(key) != expected:
            errors.append(f"config {key}={config.get(key)!r}, expected {expected!r}")
    expected_flags = {
        "control": (True, True, False, False),
        "projector": (True, True, False, True),
        "pre-rms-spatial": (True, False, True, False),
    }[arm]
    actual_flags = tuple(
        bool(config.get(key, False))
        for key in (
            "sigreg_spatial",
            "sigreg_spatial_pool",
            "sigreg_pre_rms_spatial",
            "sigreg_projector",
        )
    )
    if actual_flags != expected_flags:
        errors.append(f"config SIGReg flags={actual_flags!r}, expected {expected_flags!r}")
    comparison_config = dict(config)
    for varying_key in (
        "output_dir",
        "sigreg_spatial",
        "sigreg_spatial_pool",
        "sigreg_pre_rms_spatial",
        "sigreg_projector",
        "sigreg_projector_dim",
    ):
        comparison_config.pop(varying_key, None)
    comparison_config_sha256 = hashlib.sha256(
        json.dumps(comparison_config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if report.get("schema") != "p2.eval_report.v9":
        errors.append(f"unexpected eval schema {report.get('schema')!r}")
    if not finite_tree(report):
        errors.append("report contains non-finite numeric values")
    errors.extend(verify_sha256_file(sha_path, root))
    expected_stages = ["train-1000", "eval-1000"]
    if update >= 2000:
        expected_stages.extend(("train-2000", "eval-2000"))
    if update >= 4000:
        expected_stages.extend(("train-4000", "eval-4000"))
    phases_by_stage: dict[str, list[dict[str, Any]]] = {}
    if phases_path.is_file():
        for line_number, line in enumerate(phases_path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                phase = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"invalid phases.jsonl line {line_number}: {exc}")
                continue
            stage = phase.get("stage")
            if isinstance(stage, str):
                phases_by_stage.setdefault(stage, []).append(phase)
            for provenance_key in ("git_sha", "candle_git_sha", "binary_sha256"):
                if phase.get(provenance_key) != manifest.get(provenance_key):
                    errors.append(
                        f"phase line {line_number} {provenance_key} does not match manifest"
                    )
    else:
        errors.append(f"missing {phases_path}")
    for stage in expected_stages:
        attempts = phases_by_stage.get(stage, [])
        if not attempts:
            errors.append(f"missing phase {stage}")
        elif str(attempts[-1].get("status")) != "0":
            errors.append(f"latest phase {stage} exited {attempts[-1].get('status')}")
    for stage, attempts in phases_by_stage.items():
        if not stage.startswith("train-"):
            continue
        for attempt_index, phase in enumerate(attempts, 1):
            log_text = phase.get("log_path")
            log_path = (
                resolve_artifact_path(Path(log_text), root)
                if isinstance(log_text, str)
                else arm_dir / f"{stage}.log"
            )
            if log_path is None:
                errors.append(f"training log path is outside experiment root: {log_text}")
                continue
            try:
                log_contents = log_path.read_text(errors="replace")
            except OSError as exc:
                errors.append(f"cannot read retained training log {log_path}: {exc}")
                continue
            if NONFINITE_TRAINING_PATTERN.search(log_contents):
                errors.append(
                    f"training phase {stage} attempt {attempt_index} recorded a non-finite value"
                )

    dynamics = report.get("synthetic_dynamics", {})
    representation = dynamics.get("representation") or {}
    aggregate = nested(dynamics, "action_diagnostics", "aggregate", "shuffle") or {}
    random_one_step = (
        nested(dynamics, "action_diagnostics", "by_source", "random_one_step", "shuffle")
        or {}
    )
    changed = dynamics.get("changed_transitions") or {}

    sigreg_valid = (
        representation.get("sigreg_raw") is not None
        and representation.get("sigreg_bounded") is not None
        and representation.get("sigreg_near_bound") is False
    )
    aggregate_action = (
        aggregate.get("ratio") is not None
        and aggregate["ratio"] >= 1.10
        and aggregate.get("ratio_ci95_low") is not None
        and aggregate["ratio_ci95_low"] > 1.0
    )
    random_action = (
        random_one_step.get("ratio") is not None
        and random_one_step["ratio"] >= 1.10
        and random_one_step.get("ratio_ci95_low") is not None
        and random_one_step["ratio_ci95_low"] > 1.0
    )
    changed_copy = (
        changed.get("n", 0) > 0
        and changed.get("improvement_fraction") is not None
        and changed["improvement_fraction"] >= 0.10
    )
    noncollapse = representation.get("noncollapse_pass") is True
    artifact_valid = not errors
    hard_pass = all(
        (artifact_valid, sigreg_valid, aggregate_action, random_action, changed_copy, noncollapse)
    )
    return {
        "seed": seed,
        "arm": arm,
        "update": update,
        "report_path": str(report_path),
        "git_sha": manifest.get("git_sha"),
        "candle_git_sha": manifest.get("candle_git_sha"),
        "binary_sha256": manifest.get("binary_sha256"),
        "comparison_config_sha256": comparison_config_sha256,
        "errors": errors,
        "artifact_valid": artifact_valid,
        "sigreg_valid": sigreg_valid,
        "aggregate_action_pass": aggregate_action,
        "random_one_step_action_pass": random_action,
        "changed_copy_forward_pass": changed_copy,
        "noncollapse_pass": noncollapse,
        "hard_pass": hard_pass,
        "sigreg_raw": representation.get("sigreg_raw"),
        "sigreg_bounded": representation.get("sigreg_bounded"),
        "encoder_variance": representation.get("mean_encoder_variance"),
        "effective_rank": representation.get("effective_rank"),
        "effective_rank_fraction": representation.get("effective_rank_fraction"),
        "aggregate_ratio": aggregate.get("ratio"),
        "aggregate_ci_low": aggregate.get("ratio_ci95_low"),
        "aggregate_ci_high": aggregate.get("ratio_ci95_high"),
        "random_one_step_ratio": random_one_step.get("ratio"),
        "random_one_step_ci_low": random_one_step.get("ratio_ci95_low"),
        "random_one_step_ci_high": random_one_step.get("ratio_ci95_high"),
        "changed_n": changed.get("n"),
        "changed_improvement": changed.get("improvement_fraction"),
        "changed_improvement_ci_low": changed.get("improvement_ci95_low"),
        "changed_improvement_ci_high": changed.get("improvement_ci95_high"),
        "rollout_mse_8": nested(dynamics, "rollout", "mse_8"),
    }


def at_least(value: Any, threshold: float) -> bool:
    return isinstance(value, (int, float)) and value >= threshold


def improved(early: dict[str, Any], late: dict[str, Any], key: str) -> bool:
    a, b = early.get(key), late.get(key)
    return isinstance(a, (int, float)) and isinstance(b, (int, float)) and b > a


def credible_approach(early: dict[str, Any], late: dict[str, Any]) -> bool:
    return (
        late["artifact_valid"]
        and late["sigreg_valid"]
        and late["noncollapse_pass"]
        and at_least(late["aggregate_ratio"], 1.05)
        and at_least(late["random_one_step_ratio"], 1.05)
        and at_least(late["changed_improvement"], 0.05)
        and improved(early, late, "aggregate_ratio")
        and improved(early, late, "random_one_step_ratio")
        and improved(early, late, "changed_improvement")
    )


def treatment_materially_better(control: dict[str, Any], treatment: dict[str, Any]) -> bool:
    keys = ("aggregate_ratio", "random_one_step_ratio", "changed_improvement")
    return all(
        isinstance(control.get(key), (int, float))
        and isinstance(treatment.get(key), (int, float))
        and treatment[key] - control[key] >= 0.05
        for key in keys
    )


def pilot_decision(runs: list[dict[str, Any]], treatment_arm: str) -> dict[str, Any]:
    indexed = {(run["arm"], run["update"]): run for run in runs}
    control = indexed[("control", 2000)]
    treatment = indexed[(treatment_arm, 2000)]
    control_credible = credible_approach(indexed[("control", 1000)], control)
    treatment_credible = credible_approach(indexed[(treatment_arm, 1000)], treatment)
    material = treatment_materially_better(control, treatment)
    if control["hard_pass"] and treatment["hard_pass"]:
        branch = "C"
        action = "replicate_both_arms_seeds_2_and_3"
    elif control["hard_pass"] != treatment["hard_pass"]:
        branch = "B"
        action = "replicate_both_arms_seeds_2_and_3"
    elif control_credible or treatment_credible:
        branch = "B"
        action = "replicate_both_arms_seeds_2_and_3"
    else:
        branch = "A"
        action = "stop_after_pilot"
    return {
        "stage": "pilot",
        "terminal_branch": branch,
        "action": action,
        "control_credible_approach": control_credible,
        "treatment_arm": treatment_arm,
        "treatment_credible_approach": treatment_credible,
        "treatment_material_improvement": material,
        "note": (
            "Credible approach was preregistered as all 2,000-step point metrics reaching "
            "1.05/1.05/0.05, improving from 1,000, with valid SIGReg and noncollapse. "
            "Any arm meeting that monotonic criterion advances both arms; material "
            "treatment improvement requires +0.05 in both action ratios and "
            "changed-transition improvement."
        ),
    }


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def arm_three_seed_pass(arm_runs: list[dict[str, Any]]) -> bool:
    if len(arm_runs) != 3 or any(not run["artifact_valid"] for run in arm_runs):
        return False
    aggregate = [run["aggregate_ratio"] for run in arm_runs]
    aggregate_low = [run["aggregate_ci_low"] for run in arm_runs]
    random_one = [run["random_one_step_ratio"] for run in arm_runs]
    random_one_low = [run["random_one_step_ci_low"] for run in arm_runs]
    changed = [run["changed_improvement"] for run in arm_runs]
    if not all(
        isinstance(value, (int, float))
        for value in aggregate + aggregate_low + random_one + random_one_low + changed
    ):
        return False
    return (
        median(aggregate) >= 1.10
        and median(aggregate_low) > 1.0
        and median(random_one) >= 1.10
        and median(random_one_low) > 1.0
        and min(aggregate) > 1.05
        and min(random_one) > 1.05
        and min(changed) >= 0.10
        and all(run["sigreg_valid"] and run["noncollapse_pass"] for run in arm_runs)
    )


def severe_relative_regressions(
    candidate_runs: list[dict[str, Any]], reference_runs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    regressions = []
    for seed in (1, 2, 3):
        candidate = next(run for run in candidate_runs if run["seed"] == seed)
        reference = next(run for run in reference_runs if run["seed"] == seed)
        for key, direction in (
            ("rollout_mse_8", "higher"),
            ("encoder_variance", "lower"),
            ("effective_rank_fraction", "lower"),
        ):
            candidate_value, reference_value = candidate[key], reference[key]
            if not (
                isinstance(candidate_value, (int, float))
                and isinstance(reference_value, (int, float))
                and reference_value > 0
            ):
                continue
            ratio = candidate_value / reference_value
            severe = ratio > 1.25 if direction == "higher" else ratio < 0.75
            if severe:
                regressions.append({"seed": seed, "metric": key, "ratio": ratio})
    return regressions


def three_seed_decision(
    runs: list[dict[str, Any]], decision_update: int, allow_extension: bool, treatment_arm: str
) -> dict[str, Any]:
    arms = ("control", treatment_arm)
    late = [run for run in runs if run["update"] == decision_update]
    by_arm = {arm: [run for run in late if run["arm"] == arm] for arm in arms}
    qualifies = {arm: arm_three_seed_pass(by_arm[arm]) for arm in arms}
    severe_regressions = {
        "control": severe_relative_regressions(by_arm["control"], by_arm[treatment_arm]),
        treatment_arm: severe_relative_regressions(by_arm[treatment_arm], by_arm["control"]),
    }
    for arm in arms:
        qualifies[arm] &= not severe_regressions[arm]

    paired = {}
    for key in ("aggregate_ratio", "random_one_step_ratio", "changed_improvement"):
        paired[key] = [
            next(run for run in by_arm[treatment_arm] if run["seed"] == seed)[key]
            - next(run for run in by_arm["control"] if run["seed"] == seed)[key]
            for seed in (1, 2, 3)
        ]
    treatment_material = all(median(values) >= 0.05 for values in paired.values())
    if qualifies["control"] and (not qualifies[treatment_arm] or not treatment_material):
        promoted = "control"
    elif qualifies[treatment_arm]:
        promoted = treatment_arm
    else:
        promoted = None

    early_update = 1000 if decision_update == 2000 else 2000
    early = [run for run in runs if run["update"] == early_update]
    improving_arms = []
    for arm in arms:
        if all(
            credible_approach(
                next(run for run in early if run["arm"] == arm and run["seed"] == seed),
                next(run for run in late if run["arm"] == arm and run["seed"] == seed),
            )
            for seed in (1, 2, 3)
        ):
            improving_arms.append(arm)
    extend = allow_extension and promoted is None and bool(improving_arms)
    return {
        "stage": "three_seed" if decision_update == 2000 else "post_extension",
        "decision_update": decision_update,
        "terminal_branch": "E" if decision_update == 4000 or extend else "D",
        "action": (
            "extend_all_six_to_4000"
            if extend
            else "stop_after_4000"
            if decision_update == 4000
            else "stop_at_2000"
        ),
        "qualifies": qualifies,
        "promoted_arm": promoted,
        "treatment_arm": treatment_arm,
        "paired_treatment_minus_control": paired,
        "treatment_material_across_seeds": treatment_material,
        "severe_relative_regressions": severe_regressions,
        "improving_arms": improving_arms,
        "regression_threshold_note": (
            "A candidate is not promoted if any paired seed has >25% worse horizon-8 "
            "rollout MSE or >25% lower encoder variance/effective-rank fraction."
        ),
    }


def markdown(result: dict[str, Any]) -> str:
    lines = [
        f"# P2 {result['experiment']} gate result",
        "",
        "| seed | arm | update | hard pass | aggregate ratio [95% CI] | random_one_step ratio [95% CI] | changed improvement [95% CI] | variance | effective rank | raw / bounded SIGReg |",
        "|---:|---|---:|---|---|---|---|---:|---:|---|",
    ]
    for run in result["runs"]:
        def ci(value: Any, low: Any, high: Any) -> str:
            if value is None:
                return "n/a"
            return f"{value:.4f} [{low:.4f}, {high:.4f}]" if low is not None and high is not None else f"{value:.4f}"

        lines.append(
            "| {seed} | {arm} | {update} | {hard} | {aggregate} | {random} | {changed} | {variance} | {rank} | {sigreg} |".format(
                seed=run["seed"],
                arm=run["arm"],
                update=run["update"],
                hard="yes" if run["hard_pass"] else "no",
                aggregate=ci(run["aggregate_ratio"], run["aggregate_ci_low"], run["aggregate_ci_high"]),
                random=ci(run["random_one_step_ratio"], run["random_one_step_ci_low"], run["random_one_step_ci_high"]),
                changed=ci(run["changed_improvement"], run["changed_improvement_ci_low"], run["changed_improvement_ci_high"]),
                variance="n/a" if run["encoder_variance"] is None else f"{run['encoder_variance']:.6g}",
                rank="n/a" if run["effective_rank"] is None else f"{run['effective_rank']:.3f}",
                sigreg=(
                    "n/a"
                    if run["sigreg_raw"] is None or run["sigreg_bounded"] is None
                    else f"{run['sigreg_raw']:.3f} / {run['sigreg_bounded']:.3f}"
                ),
            )
        )
    decision = result["decision"]
    lines.extend(
        [
            "",
            f"Decision: branch {decision['terminal_branch']} — `{decision['action']}`.",
            "",
            "This is synthetic held-out experimental evidence, not a public ARC result or research claim.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--final-update", type=int, choices=(2000, 4000), default=2000)
    parser.add_argument(
        "--treatment-arm",
        choices=("projector", "pre-rms-spatial"),
        default="projector",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    seeds = sorted(set(args.seeds))
    if seeds not in ([1], [1, 2, 3]):
        raise SystemExit("--seeds must be exactly 1 or 1 2 3")
    if args.final_update == 4000 and seeds != [1, 2, 3]:
        raise SystemExit("--final-update 4000 requires --seeds 1 2 3")
    arms = ("control", args.treatment_arm)
    updates = (1000, 2000) if args.final_update == 2000 else (1000, 2000, 4000)
    runs = [
        load_run(args.root, seed, arm, update, args.treatment_arm)
        for seed in seeds
        for arm in arms
        for update in updates
    ]
    invalid_runs = [run for run in runs if not run["artifact_valid"]]
    if invalid_runs:
        failure = {
            "schema": "p2.ab_gate_validation_failure.v1",
            "root": str(args.root),
            "treatment_arm": args.treatment_arm,
            "runs": runs,
            "errors": [
                {"seed": run["seed"], "arm": run["arm"], "update": run["update"], "errors": run["errors"]}
                for run in invalid_runs
            ],
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        lines = ["# P2 A/B artifact validation failed", ""]
        for run in invalid_runs:
            lines.append(f"## seed {run['seed']} / {run['arm']} / update {run['update']}")
            lines.extend(f"- {error}" for error in run["errors"])
            lines.append("")
        args.output_md.write_text("\n".join(lines))
        raise SystemExit("artifact validation failed; decision gate was not evaluated")
    provenance = {
        (run["git_sha"], run["candle_git_sha"], run["binary_sha256"])
        for run in runs
    }
    if len(provenance) != 1:
        raise SystemExit(
            "runs do not share one immutable provenance tuple: "
            f"{sorted(provenance, key=repr)!r}"
        )
    for seed in seeds:
        config_contracts = {
            run["comparison_config_sha256"] for run in runs if run["seed"] == seed
        }
        if len(config_contracts) != 1:
            raise SystemExit(
                f"seed {seed} arms differ outside preregistered SIGReg geometry fields"
            )
    decision = (
        pilot_decision(runs, args.treatment_arm)
        if seeds == [1]
        else three_seed_decision(
            runs,
            decision_update=args.final_update,
            allow_extension=args.final_update == 2000,
            treatment_arm=args.treatment_arm,
        )
    )
    experiment = (
        "SIGReg/action-conditioning"
        if args.treatment_arm == "projector"
        else "SIGReg geometry-isolation"
    )
    schema = (
        "p2.sigreg_action_gate.v2"
        if args.treatment_arm == "projector"
        else "p2.sigreg_geometry_gate.v1"
    )
    result = {
        "schema": schema,
        "experiment": experiment,
        "treatment_arm": args.treatment_arm,
        "root": str(args.root),
        "runs": runs,
        "decision": decision,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.output_md.write_text(markdown(result))
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
