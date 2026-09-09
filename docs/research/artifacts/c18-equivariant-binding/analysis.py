#!/usr/bin/env python3
"""C18 finite scoring and action-renaming checks; no model or fitting.

Runtime checks belong to the pinned external receipt. Dataset semantics,
post-ablation input hashes, output identities, metrics and gates are checked here.
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import re
import signal
import sys
import time
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
DATA_SHA = "36c975babfbd783db93360ea2946cec32fb7e0fee66c2d03abdd873a33f5a67e"
assert hashlib.sha256((HERE / "data.py").read_bytes()).hexdigest() == DATA_SHA
_spec = importlib.util.spec_from_file_location("c18_pinned_data", HERE / "data.py")
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)
require, exact, file_sha, decode = D.require, D.exact, D.file_sha, D.decode
SCHEMA = "looped-action-binding-analysis-config-v2"
EQ_ATOL, EQ_RTOL, TRUE_MARGIN = 1e-4, 1e-5, 1e-3
DEPENDENCY = "1ea5cc5c5998a1c2bd388eb158653dd0d1b7452a"
CHECKS = set("data source build initialization completed_training gradients device profiles checkpoints cleanup".split())
EXTRA = {"logits", "stage", "loops", "cleared", "query_cleared", "model_input_sha256", "model_kind"}


def stream_specs():
    specs = {}
    for cohort in ("fit", "heldout"):
        specs[f"initial-{cohort}-l4"] = dict(cohort=cohort, stage="initial", loops=4, cleared=False, query_cleared=False)
        for loops in (1, 2, 4, 8):
            specs[f"final-{cohort}-l{loops}"] = dict(cohort=cohort, stage="final", loops=loops, cleared=False, query_cleared=False)
        for control in ("effects-zero", "query-zero"):
            specs[f"final-{cohort}-{control}"] = dict(cohort=cohort, stage="final", loops=4,
                cleared=control == "effects-zero", query_cleared=control == "query-zero")
    specs["final-cached-visual-l4"] = dict(cohort="cached_visual", stage="final", loops=4, cleared=False, query_cleared=False)
    return specs


SPECS = stream_specs()


def keys(value, expected, where):
    require(type(value) is dict and set(value) == set(expected), f"{where}: exact keys required")


def digest(value):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value), "invalid SHA256")
    return value


def verify(path, expected):
    require(file_sha(path) == digest(expected), f"file hash mismatch: {path}")


def read(path):
    require(Path(path).stat().st_size <= 128 * 1024 * 1024, "oversized JSON")
    return decode(Path(path).read_bytes())


def verify_files(files):
    require(type(files) is dict and files, "nonempty frozen files required")
    for path, expected in files.items():
        verify(path, expected)


def configuration(config):
    keys(config, {"schema", "dataset", "dataset_sha256", "registration", "frozen_files", "streams",
                  "integrity_receipt", "integrity_receipt_sha256", "model_kind", "effective_batch"}, "config")
    require(config["schema"] == SCHEMA, "config schema")
    require(config["model_kind"] in ("legacy", "equivariant"), "model kind")
    batch = D.batch_size(config["effective_batch"])
    keys(config["registration"], {"path", "sha256"}, "registration")
    verify_files(config["frozen_files"])
    selected = {config["dataset"]: config["dataset_sha256"],
                config["registration"]["path"]: config["registration"]["sha256"]}
    for path in (HERE / "data.py", HERE / "analysis.py", HERE / "analysis_tests.py"):
        require(config["frozen_files"].get(str(path)) == file_sha(path), "unfrozen data/scorer/tests")
    keys(config["streams"], SPECS, "selected stream names")
    for name, expected in SPECS.items():
        record = config["streams"][name]
        keys(record, {"rows", "sha256"} | set(expected), "stream record")
        require(exact({key: record[key] for key in expected}, expected), "stream semantic selector")
        require(record["rows"] not in selected, "reused dataset/stream path")
        selected[record["rows"]] = record["sha256"]
    for path, expected in selected.items():
        verify(path, expected)
    verify(config["integrity_receipt"], config["integrity_receipt_sha256"])
    receipt = read(config["integrity_receipt"])
    require(receipt.get("schema") == "looped-action-binding-integrity-v2" and receipt.get("accepted") is True,
            "runtime integrity not accepted")
    require(receipt.get("model_kind") == config["model_kind"] and exact(receipt.get("effective_batch"), batch),
            "receipt model/batch mismatch")
    require(exact(receipt.get("completed_schedule"), completed_schedule(batch)), "receipt completed schedule mismatch")
    keys(receipt.get("checks"), CHECKS, "runtime checks")
    require(all(v is True for v in receipt["checks"].values()), "runtime check failed/untyped")
    require(type(receipt.get("source_revision")) is str and re.fullmatch(r"[0-9a-f]{40}", receipt["source_revision"]), "source revision")
    digest(receipt.get("binary_sha256"))
    require(receipt.get("dependency_revision") == DEPENDENCY, "dependency drift")
    require(receipt.get("dataset_sha256") == config["dataset_sha256"] and
            receipt.get("registration_sha256") == config["registration"]["sha256"], "receipt dataset/registration")
    checkpoints = receipt.get("checkpoints")
    keys(checkpoints, {"initial", "final"}, "checkpoint stages")
    for checkpoint in checkpoints.values():
        keys(checkpoint, {"sha256", "parameter_sha256"}, "checkpoint identity")
        for value in checkpoint.values():
            digest(value)
    require(checkpoints["initial"]["parameter_sha256"] != checkpoints["final"]["parameter_sha256"], "unchanged trained parameters")
    keys(receipt.get("streams"), SPECS, "receipt streams")
    for name, record in config["streams"].items():
        expected = dict(record, checkpoint_sha256=checkpoints[record["stage"]]["sha256"],
                        parameter_sha256=checkpoints[record["stage"]]["parameter_sha256"])
        require(exact(receipt["streams"][name], expected), "receipt stream/checkpoint binding")
    verify_files(receipt.get("frozen_files"))
    require(config["integrity_receipt"] not in receipt["frozen_files"] and exact(config["frozen_files"],
            dict(receipt["frozen_files"], **{config["integrity_receipt"]: config["integrity_receipt_sha256"]})),
            "config/receipt frozen files differ")
    for path, expected in config["frozen_files"].items():
        if path != config["integrity_receipt"]:  # No circular receipt self-hash.
            require(path not in selected or selected[path] == expected, "conflicting selected binding")
            selected[path] = expected
    require(all(receipt["frozen_files"].get(path) == expected for path, expected in selected.items()), "receipt does not close selected/source files")
    return receipt


def validate_abstract(rows, map_ids):
    require(type(rows) is list and len(rows) == len(map_ids) * 96, "abstract population size")
    permutations = tuple(itertools.permutations(range(4)))
    triples = tuple(itertools.permutations(range(4), 3))
    directions = ((0, -1), (0, 1), (-1, 0), (1, 0))
    for index, row in enumerate(rows):
        keys(row, D.ROW_FIELDS, "abstract row")
        group, slot = divmod(index, len(map_ids))
        actions, desired, map_id = triples[group // 4], group % 4, map_ids[slot]
        mapping = permutations[map_id]
        label = next(a for a in range(4) if mapping[a] == desired)
        omitted = next(a for a in range(4) if a not in actions)
        expected = dict(index=index, group_index=group, map_id=map_id, observed_actions=list(actions),
            desired_direction=desired, correct_action=label, demonstrated=label in actions,
            orbit_id=slot * 16 + omitted * 4 + desired, canonical=list(actions) == sorted(actions))
        require(exact({key: row[key] for key in expected}, expected), "abstract membership/labels/order")
        x = D.finite(row["features"], (4, 7))
        expected_x = np.zeros((4, 7))
        for j, action in enumerate(actions):
            expected_x[j, :2] = directions[mapping[action]]
            expected_x[j, 2 + action] = 1
        expected_x[3, :2] = directions[desired]; expected_x[3, 6] = 1
        require(np.array_equal(x, expected_x), "abstract feature semantics")
        require(row["input_sha256"] == D.sha(x.astype("<f4").tobytes()) and
                row["label_sha256"] == D.sha(np.asarray([label], dtype="<u4").tobytes()), "abstract input/label hashes")
        # Independent explicit completion of the missing action; audit only.
        table = {a: np.asarray(directions[mapping[a]]) for a in actions}
        table[omitted] = -sum(table.values())
        scores = [int(table[a] @ np.asarray(directions[desired])) for a in range(4)]
        require(scores[label] == 1 and sorted(scores) == [-1, 0, 0, 1], "analytic gap-one control")


def completed_schedule(batch):
    D.batch_size(batch)
    return dict(updates=(588800 + batch - 1) // batch, effective_batch=batch, presentations=588800,
                indices_sha256=D.C17_FLAT_SHA, tail_rows=588800 % batch)


def validate_dataset(value, effective_batch=None):
    keys(value, {"schema", "fit", "heldout", "updates", "cached_visual", "provenance", "audit", "schedule"}, "dataset")
    require(value["schema"] == D.SCHEMA, "dataset schema")
    spec = value["schedule"]
    keys(spec, {"kind", "effective_batch", "presentations", "indices_sha256"}, "dataset schedule")
    batch = D.batch_size(spec["effective_batch"])
    require(effective_batch is None or exact(batch, effective_batch), "analysis/dataset batch mismatch")
    require(exact(spec, dict(kind="training", effective_batch=batch, presentations=588800,
                             indices_sha256=D.C17_FLAT_SHA)), "completed training schedule required")
    for name, maps in (("fit", D.FIT), ("heldout", D.HELD)):
        validate_abstract(value[name], maps)
    rng = np.random.Generator(np.random.PCG64(20260922))
    sequence = np.concatenate([rng.permutation(1536) for _ in range(384)])[:588800]
    expected_updates = [sequence[start:start + batch].tolist() for start in range(0, 588800, batch)]
    require(exact(value["updates"], expected_updates), "registered schedule differs")
    visual, audit = D.load_visual()  # Frozen C15 cache preparation, not a learned-model operation.
    require(len(visual) == 1024 and exact(value["cached_visual"], visual), "cached visual features/identity differ")
    reconstructed = D.dataset(batch, visual=visual, visual_audit=audit)
    require(exact(value["audit"], reconstructed["audit"]) and exact(value["provenance"], reconstructed["provenance"]),
            "dataset analytic audit/provenance differs")
    return {name: value[name] for name in ("fit", "heldout", "cached_visual")}


def actual_input(row, spec):
    x = D.finite(row["features"], (4, 7)).astype("<f4")
    require(not (spec["cleared"] and spec["query_cleared"]), "combined ablations forbidden")
    if spec["cleared"]:
        x[:3, :2] = 0
    if spec["query_cleared"]:
        x[3, :2] = 0
    return x


def row_logits(row, expected, spec, model_kind="legacy"):
    keys(row, set(expected) | EXTRA, "evaluation row")
    require(model_kind in ("legacy", "equivariant") and row["model_kind"] == model_kind, "evaluation model kind")
    require(exact({key: row[key] for key in expected}, expected), "evaluation row identity/order")
    require(exact({key: row[key] for key in ("stage", "loops", "cleared", "query_cleared")},
                  {key: spec[key] for key in ("stage", "loops", "cleared", "query_cleared")}), "evaluation stage/depth/ablation")
    require(row["model_input_sha256"] == D.sha(actual_input(expected, spec).tobytes()), "actual post-ablation model input hash")
    logits = D.finite(row["logits"], (4,))
    require((np.abs(logits) <= np.finfo(np.float32).max).all(), "logits outsideF32 range")
    return logits


def read_stream(path, population, spec, model_kind="legacy"):
    result = []
    with Path(path).open("rb") as handle:
        for index, line in enumerate(handle):
            require(index < len(population) and 0 < len(line) <= 512 * 1024 and line.strip(), "extra/blank/oversized output row")
            result.append(row_logits(decode(line), population[index], spec, model_kind))
    require(len(result) == len(population), "missing output rows")
    return np.asarray(result)


def vectors(logits, labels):
    require(logits.shape == (len(labels), 4) and np.isfinite(logits).all(), "metric array shape/finiteness")
    predicted = logits.argmax(axis=1)
    shifted = logits - logits.max(axis=1, keepdims=True)
    ce = np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(labels)), labels]
    others = logits.copy(); others[np.arange(len(labels)), labels] = -np.inf
    margin = logits[np.arange(len(labels)), labels] - others.max(axis=1)
    top = np.sort(logits, axis=1)[:, -2:]
    require(np.isfinite(ce).all() and np.isfinite(margin).all(), "nonfinite endpoint")
    return {"predictions": predicted, "correct": predicted == labels, "ce": ce,
            "margin": margin, "winner_margin": top[:, 1] - top[:, 0]}


def basic(v, labels, mask):
    n = int(mask.sum())
    result = {"rows": n, "correct": int(v["correct"][mask].sum()),
              "accuracy": float(v["correct"][mask].mean()) if n else None,
              "ce": float(v["ce"][mask].mean()) if n else None,
              "label_histogram": np.bincount(labels[mask], minlength=4).tolist(),
              "prediction_histogram": np.bincount(v["predictions"][mask], minlength=4).tolist()}
    for key in ("margin", "winner_margin"):
        result[key] = {"mean": float(v[key][mask].mean()) if n else None,
                       "min": float(v[key][mask].min()) if n else None}
    return result


def summarize(v, labels, demonstrated, map_ids, mask):
    result = basic(v, labels, mask)
    result["subsets"] = {name: basic(v, labels, mask & subset)
        for name, subset in (("demonstrated", demonstrated), ("omitted", ~demonstrated))}
    result["per_map"] = {str(map_id): basic(v, labels, mask & (map_ids == map_id)) for map_id in sorted(set(map_ids.tolist()))}
    return result


def ordering(population, logits, predicted, cached=False):
    if cached:
        return {"applicable": False, "orbits": None, "inconsistent_orbits": None,
                "winner_invariant": None, "max_logit_deviation": None}
    groups = {}
    for index, row in enumerate(population):
        groups.setdefault(row["orbit_id"], []).append(index)
    inconsistent, deviation = 0, 0.
    for members in groups.values():
        require(len(members) == 6 and sum(population[i]["canonical"] for i in members) == 1, "incomplete semantic orbit")
        canonical = next(i for i in members if population[i]["canonical"])
        canonical_x = actual_input(population[canonical], dict(cleared=False, query_cleared=False))
        for index in members:
            x = actual_input(population[index], dict(cleared=False, query_cleared=False))
            require(np.array_equal(x[np.argsort(population[index]["observed_actions"])], canonical_x[:3]) and
                    np.array_equal(x[3], canonical_x[3]), "wrong support-order orbit membership")
        inconsistent += int(not (predicted[members] == predicted[canonical]).all())
        # Maximum across any pair of orderings, not just each against the canonical.
        deviation = max(deviation, float(np.ptp(logits[members], axis=0).max()))
    return {"applicable": True, "orbits": len(groups), "inconsistent_orbits": inconsistent,
            "winner_invariant": inconsistent == 0, "max_logit_deviation": deviation}


def controls(population, spec, predicted):
    if not (spec["cleared"] or spec["query_cleared"]):
        return None
    groups = {}
    for index, row in enumerate(population):
        groups.setdefault(actual_input(row, spec).tobytes(), []).append(index)
    inconsistent = 0
    for members in groups.values():
        counts = np.bincount([population[i]["correct_action"] for i in members], minlength=4)
        require(np.array_equal(counts, np.full(4, len(members) // 4)), "ablated identical-input labels not balanced")
        inconsistent += int(not (predicted[members] == predicted[members[0]]).all())
    correct = sum(int(predicted[i] == row["correct_action"]) for i, row in enumerate(population))
    return {"identical_input_groups": len(groups), "inconsistent_groups": inconsistent,
            "winner_invariant": inconsistent == 0, "balanced_labels": True,
            "correct": correct, "rows": len(population), "exact_quarter": 4 * correct == len(population),
            "interpretation": "25% follows from balanced labels on identical ablated inputs, not learned collapse."}


def score(population, logits, spec):
    labels = np.asarray([r["correct_action"] for r in population])
    demonstrated = np.asarray([r["demonstrated"] for r in population], dtype=bool)
    cached = spec["cohort"] == "cached_visual"
    maps = np.asarray([r["audit"]["permutation_id"] if cached else r["map_id"] for r in population])
    canonical = np.ones(len(labels), dtype=bool) if cached else np.asarray([r["canonical"] for r in population])
    v = vectors(logits, labels)
    result = {"raw": summarize(v, labels, demonstrated, maps, np.ones(len(labels), dtype=bool)),
              "canonical": summarize(v, labels, demonstrated, maps, canonical),
              "canonical_basis": "all_retained_cached_visual_rows" if cached else "ascending_observed_action_ids",
              "ordering": ordering(population, logits, v["predictions"], cached),
              "control": controls(population, spec, v["predictions"])}
    return result, v, canonical, demonstrated


def paired(first, second, mask):
    n = int(mask.sum())
    return {"rows": n, "accuracy_delta": float((first["correct"][mask].astype(float) - second["correct"][mask]).mean()) if n else None,
            "ce_delta": float((first["ce"][mask] - second["ce"][mask]).mean()) if n else None,
            "margin_delta": float((first["margin"][mask] - second["margin"][mask]).mean()) if n else None,
            "both_correct": int((first["correct"][mask] & second["correct"][mask]).sum()),
            "first_only_correct": int((first["correct"][mask] & ~second["correct"][mask]).sum()),
            "second_only_correct": int((~first["correct"][mask] & second["correct"][mask]).sum()),
            "changed_predictions": int((first["predictions"][mask] != second["predictions"][mask]).sum())}


def contrast(first, second, canonical, demonstrated):
    result = {}
    for label, mask in (("raw", np.ones(len(canonical), dtype=bool)), ("canonical", canonical)):
        result[label] = paired(first, second, mask)
        result[label]["subsets"] = {name: paired(first, second, mask & subset)
            for name, subset in (("demonstrated", demonstrated), ("omitted", ~demonstrated))}
    return result


def external_controls(populations):
    """Audit policies only; their scores are not learned logits/probabilities."""
    result = {}
    for cohort, rows in populations.items():
        labels = np.asarray([r["correct_action"] for r in rows])
        demonstrated = np.asarray([r["demonstrated"] for r in rows], dtype=bool)
        cached = cohort == "cached_visual"
        canonical = np.ones(len(rows), bool) if cached else np.asarray([r["canonical"] for r in rows])
        missing, analytic, margins = [], [], []
        for row in rows:
            x = np.asarray(row["features"])
            actions = row["audit"]["observed_support_action_ids"] if cached else row["observed_actions"]
            omitted = next(a for a in range(4) if a not in actions)
            table = np.zeros((4, 2))
            for index, action in enumerate(actions):
                table[action] = x[index, :2]
            table[omitted] = -sum(table[a] for a in actions)
            scores = table @ x[3, :2]
            label = row["correct_action"]
            missing.append(omitted); analytic.append(int(scores.argmax()))
            margins.append(float(scores[label] - max(scores[a] for a in range(4) if a != label)))
        result[cohort] = {}
        for name, predictions in (("constant_action0", np.zeros(len(rows), int)),
                                  ("analytic", np.asarray(analytic)), ("always_missing_id", np.asarray(missing))):
            correct = predictions == labels
            def counts(mask):
                n = int(mask.sum())
                return {"rows": n, "correct": int(correct[mask].sum()), "accuracy": float(correct[mask].mean()) if n else None}
            control = {}
            for selection, mask in (("raw", np.ones(len(rows), bool)), ("canonical", canonical)):
                control[selection] = counts(mask)
                control[selection]["subsets"] = {label: counts(mask & subset)
                    for label, subset in (("demonstrated", demonstrated), ("omitted", ~demonstrated))}
                summary = control[selection]
                if name == "analytic":
                    require(summary["correct"] == summary["rows"], "analytic positive control failed")
                else:
                    require(4 * summary["correct"] == summary["rows"], "external quarter control failed")
                if name == "always_missing_id":
                    require(summary["subsets"]["omitted"]["correct"] == summary["subsets"]["omitted"]["rows"] and
                            summary["subsets"]["demonstrated"]["correct"] == 0, "missing-ID subset control failed")
            if name == "analytic":
                control["minimum_true_score_margin"] = min(margins)
                require(min(margins) > 0, "analytic true-action margin is not positive")
            result[cohort][name] = control
    return result


def legacy_decide(summaries):
    fit, held = (summaries[f"final-{c}-l4"]["canonical"] for c in ("fit", "heldout"))
    components = {"fit_254_of_256": fit["rows"] == 256 and fit["correct"] >= 254,
                  "heldout_116_of_128": held["rows"] == 128 and held["correct"] >= 116,
                  "heldout_demonstrated_87_of_96": held["subsets"]["demonstrated"]["rows"] == 96 and held["subsets"]["demonstrated"]["correct"] >= 87,
                  "heldout_omitted_29_of_32": held["subsets"]["omitted"]["rows"] == 32 and held["subsets"]["omitted"]["correct"] >= 29,
                  "terminal_l4_ordering": all(summaries[f"final-{c}-l4"]["ordering"]["winner_invariant"] for c in ("fit", "heldout")),
                  "ablation_controls": all(summaries[f"final-{c}-{a}"]["control"]["winner_invariant"] and
                       summaries[f"final-{c}-{a}"]["control"]["exact_quarter"] for c in ("fit", "heldout") for a in ("effects-zero", "query-zero"))}
    decision = "supported_single_seed_binding_prerequisite" if all(components.values()) else "registered_binding_screen_not_supported"
    return decision, components


def action_equivariance(populations, logits):
    """Audit maps align outputs only here; no truth or mapping enters forward."""
    groups = {}
    for cohort in ("fit", "heldout"):
        rows, values = populations[cohort], logits[cohort]
        require(values.shape == (len(rows), 4) and np.isfinite(values).all(), "equivariance logits")
        for row, value in zip(rows, values):
            if not row["canonical"]:
                continue
            mapping = D.MAPS[row["map_id"]]
            omitted = next(a for a in range(4) if a not in row["observed_actions"])
            key = (mapping[omitted], row["desired_direction"])
            groups.setdefault(key, []).append((row["map_id"], value[np.argsort(mapping)]))
    require(set(groups) == set(itertools.product(range(4), repeat=2)), "all16 renaming orbits required")
    orbits = {}
    for (missing, desired), members in sorted(groups.items()):
        members.sort(key=lambda x: x[0])
        require([m for m, _ in members] == list(range(24)), "each renaming orbit requires all24 maps exactly once")
        aligned = np.asarray([value for _, value in members])
        reference = aligned[0]  # Map0, fixed before outcomes.
        errors = np.abs(aligned - reference)
        tolerance = EQ_ATOL + EQ_RTOL * np.abs(reference)
        v = vectors(aligned, np.full(24, desired, dtype=int))
        error_max = float(errors.max())
        winner_min = float(v["winner_margin"].min())
        eligible = winner_min > 0 and winner_min > 2 * error_max
        orbits[f"{missing}/{desired}"] = {
            "omitted_effect": missing, "desired_direction": desired, "maps": 24, "reference_map_id": 0,
            "reference_logits_effect_order": reference.tolist(),
            "maximum_absolute_logit_error": error_max,
            "maximum_tolerance_ratio": float((errors / tolerance).max()),
            "numerically_equivariant": bool((errors <= tolerance).all()),
            "correct": int(v["correct"].sum()), "minimum_true_margin": float(v["margin"].min()),
            "minimum_winner_margin": winner_min,
            "strict_true_margin": bool((v["margin"] >= TRUE_MARGIN).all()),
            "positive_margin_winner_comparison": eligible,
            "winner_invariant": bool((v["predictions"] == v["predictions"][0]).all()) if eligible else None}
    records = list(orbits.values())
    return {"orbits": 16, "canonical_rows": 384, "maps_per_orbit": 24,
            "reference_map_id": 0, "atol": EQ_ATOL, "rtol": EQ_RTOL, "required_true_margin": TRUE_MARGIN,
            "winner_comparison_criterion": "minimum_winner_margin > 2*maximum_absolute_logit_error and >0",
            "maximum_absolute_logit_error": max(r["maximum_absolute_logit_error"] for r in records),
            "maximum_tolerance_ratio": max(r["maximum_tolerance_ratio"] for r in records),
            "numerically_equivariant": all(r["numerically_equivariant"] for r in records),
            "positive_margin_orbits": sum(r["positive_margin_winner_comparison"] for r in records),
            "winner_inconsistent_orbits": sum(r["winner_invariant"] is False for r in records),
            "all_eligible_winners_invariant": all(r["winner_invariant"] is not False for r in records),
            "strict_true_margin_orbits": sum(r["strict_true_margin"] for r in records),
            "minimum_true_margin": min(r["minimum_true_margin"] for r in records), "by_orbit": orbits}


def decide(summaries, model_kind="legacy", equivariance=None):
    legacy_decision, legacy_gates = legacy_decide(summaries)
    if model_kind == "legacy":
        return legacy_decision, {"legacy_descriptive": legacy_gates, "selection": legacy_gates}
    require(model_kind == "equivariant" and equivariance is not None, "equivariant gate inputs")
    selected = {
        "initial_numerical_equivariance": equivariance["initial"]["numerically_equivariant"],
        "final_numerical_equivariance": equivariance["final"]["numerically_equivariant"],
        "positive_margin_winners": all(equivariance[s]["all_eligible_winners_invariant"] for s in ("initial", "final")),
        "all16_strict_true_margin": equivariance["final"]["strict_true_margin_orbits"] == 16,
        "terminal_l4_ordering": legacy_gates["terminal_l4_ordering"],
        "ablation_controls": legacy_gates["ablation_controls"]}
    decision = "supported_single_seed_action_equivariant_selection" if all(selected.values()) else "action_equivariant_selection_not_supported"
    return decision, {"legacy_descriptive": legacy_gates, "selection": selected}


def initial_check(dataset_path, fit_path, heldout_path, binding):
    """Initialization-only numerical check; source qualification is operator-owned."""
    started = time.monotonic()
    keys(binding, {"schema", "dataset_sha256", "fit_sha256", "heldout_sha256", "model_kind",
                   "effective_batch", "source_revision", "binary_sha256", "source_verified",
                   "initialization_verified", "frozen_files"}, "initial binding")
    require(binding["schema"] == "looped-action-binding-initial-check-v2", "initial check schema")
    require(binding["source_verified"] is True and binding["initialization_verified"] is True,
            "unqualified source/initialization")
    require(type(binding["source_revision"]) is str and re.fullmatch(r"[0-9a-f]{40}", binding["source_revision"]), "source revision")
    digest(binding["binary_sha256"])
    require(binding["model_kind"] in ("legacy", "equivariant"), "model kind")
    batch = D.batch_size(binding["effective_batch"])
    selected = {str(dataset_path): binding["dataset_sha256"], str(fit_path): binding["fit_sha256"],
                str(heldout_path): binding["heldout_sha256"]}
    require(len(selected) == 3, "initial selected paths must differ")
    for name in ("data.py", "analysis.py", "analysis_tests.py"):
        path = HERE / name
        selected[str(path)] = file_sha(path)
    require(type(binding["frozen_files"]) is dict and
            all(binding["frozen_files"].get(p) == v for p, v in selected.items()), "unfrozen initial files")
    require(binding["binary_sha256"] in binding["frozen_files"].values(), "initial binary not frozen")
    verify_files(binding["frozen_files"])
    populations = validate_dataset(read(dataset_path), batch)
    logits = {cohort: read_stream(path, populations[cohort], SPECS[f"initial-{cohort}-l4"], binding["model_kind"])
              for cohort, path in (("fit", fit_path), ("heldout", heldout_path))}
    result = action_equivariance(populations, logits)
    passed = bool(result["numerically_equivariant"] and result["all_eligible_winners_invariant"])
    verify_files(binding["frozen_files"])
    require(time.monotonic() - started <= 120, "initial check deadline")
    return {"schema": "looped-action-binding-initial-check-result-v2", "accepted": True,
            "initialization_only": True, "model_kind": binding["model_kind"], "effective_batch": batch,
            "passed": passed if binding["model_kind"] == "equivariant" else True,
            "source_revision": binding["source_revision"], "binary_sha256": binding["binary_sha256"],
            "action_equivariance": result,
            "limits": "No trained evidence or learning gate; source/device/profile/checkpoint qualification remains operator-owned."}


def analyze(path):
    started = time.monotonic()
    config_sha = file_sha(path)
    config = read(path)
    receipt = configuration(config)  # Provenance acceptance before numerical output access.
    populations = validate_dataset(read(config["dataset"]), config["effective_batch"])
    audit_controls = external_controls(populations)
    summaries, measured, factual = {}, {}, {}
    for name, spec in SPECS.items():
        logits = read_stream(config["streams"][name]["rows"], populations[spec["cohort"]], spec, config["model_kind"])
        summary, values, canonical, demonstrated = score(populations[spec["cohort"]], logits, spec)
        summaries[name] = summary
        measured[name] = (values, canonical, demonstrated)
        if spec["cohort"] in ("fit", "heldout") and spec["loops"] == 4 and not (spec["cleared"] or spec["query_cleared"]):
            factual.setdefault(spec["stage"], {})[spec["cohort"]] = logits
    require(all(s["control"] is None or (s["control"]["winner_invariant"] and s["control"]["exact_quarter"])
                for s in summaries.values()), "failed ablation control integrity; no scientific decision")
    contrasts = {}
    for cohort in ("fit", "heldout"):
        values, canonical, demonstrated = measured[f"final-{cohort}-l4"]
        others = {"final_minus_initial": f"initial-{cohort}-l4",
                  "factual_minus_effects_zero": f"final-{cohort}-effects-zero",
                  "factual_minus_query_zero": f"final-{cohort}-query-zero"}
        for label, other in others.items():
            contrasts[f"{label}/{cohort}"] = contrast(values, measured[other][0], canonical, demonstrated)
    equivariance = {stage: action_equivariance(populations, factual[stage]) for stage in ("initial", "final")}
    decision, gates = decide(summaries, config["model_kind"], equivariance)
    configuration(config)
    require(file_sha(path) == config_sha and time.monotonic() - started <= 120, "config drift/scoring deadline")
    return {"schema": "looped-action-binding-analysis-v2", "accepted": True,
            "classification": "single_seed_action_equivariance_screen", "decision": decision, "gates": gates,
            "model_kind": config["model_kind"], "effective_batch": config["effective_batch"],
            "completed_schedule": receipt["completed_schedule"], "action_equivariance": equivariance,
            "legacy_decision": legacy_decide(summaries)[0],
            "config_sha256": config_sha, "dataset_sha256": config["dataset_sha256"],
            "registration_sha256": config["registration"]["sha256"],
            "analysis_sha256": file_sha(HERE / "analysis.py"), "data_helper_sha256": DATA_SHA,
            "integrity_receipt_sha256": config["integrity_receipt_sha256"],
            "source_revision": receipt["source_revision"], "binary_sha256": receipt["binary_sha256"],
            "summaries": summaries, "contrasts": contrasts, "external_controls": audit_controls,
            "limits": ["Exact finite counts; no IID, binomial or bootstrap uncertainty interval.",
                       "Six support orderings represent one semantic input; action renaming leaves only16 abstract task orbits.",
                       "For the equivariant model, held mapping accuracy is redundant under symmetry and correct fitting, not independent semantic generalization.",
                       "A^T E supplies observed-action routing; no analytical missing-effect completion or matcher is executed by the model.",
                       "Numerical action equivariance uses all24 maps against fixed map0; winner comparison requires minimum margin greater than twice maximum logit error and greater than zero.",
                       "Source/build/initialization/training/gradient/device/profile/checkpoint/cleanup checks share the pinned runtime receipt.",
                       "Analytic reconstruction is an external control, absent from the learned model.",
                       "Depth1/2/8 are descriptive at fixed training depth4; no architecture/depth-benefit claim.",
                       "C15 cached visual rows reuse privileged initial selectors and old cases; no native visual or ARC claim.",
                       "A valid failure bounds this seed/recipe, not information presence or all binder architectures."],
            "elapsed_seconds": time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(args.output.is_absolute() and args.output.parent == args.output.parent.resolve(strict=True) and
            not args.output.exists() and not args.output.is_symlink(), "new absolute output required")
    def timeout(_signal, _frame):
        raise TimeoutError("analysis exceeded120seconds")
    signal.signal(signal.SIGALRM, timeout); signal.alarm(120)
    report = analyze(args.config)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False); handle.write("\n")
    signal.alarm(0)


if __name__ == "__main__":
    main()
