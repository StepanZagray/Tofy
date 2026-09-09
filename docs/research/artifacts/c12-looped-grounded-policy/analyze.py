#!/usr/bin/env python3
"""Frozen C12 scoring only: pinned streams, independent labels, paired group bootstrap.

No generator, fitting routine, model implementation, or GPU library is imported.
The externally reviewed integrity certificate owns runtime/provenance validation;
its digest and every selected stream are bound by the externally pinned config.
"""

import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import datetime
import functools
import hashlib
import itertools
import json
from pathlib import Path
import re
import signal
import time

import numpy as np


SCHEMA = "looped-grounded-policy-analysis-config-v1"
DATA_SCHEMA = "looped-grounded-policy-data-v1"
FIT = (0, 2, 4, 5, 7, 8, 9, 10, 13, 14, 15, 16, 18, 19, 21, 23)
HELDOUT = (1, 3, 6, 11, 12, 17, 20, 22)
MAPS = {"seen": FIT, "familiar": FIT, "heldout": HELDOUT}
STAGES = ("frozen_factual", "final_factual", "final_cleared")
PERMUTATIONS = tuple(itertools.permutations(range(4)))
TRAIN_SEED, EVAL_SEED = 20260920, 20260921
TRAIN_BASE, EVAL_BASE = 0x47524F554E445452, 0x47524F554E444556
GROUPS, RESAMPLES = 64, 10000
BOOTSTRAP_SEEDS = {"seen": 1930, "familiar": 1931, "heldout": 1932}
INITIAL_CORE = "4cd502f7a76fe3dd9729f6693e9d707c670972f7ebe58ad1cbdde37ea2bb2802"
INITIAL_POLICY = "a43ad78fe7a6b1594d23c205fb191c8ac81610deaaff25198cb421d423edc678"
DEPENDENCY = "1ea5cc5c5998a1c2bd388eb158653dd0d1b7452a"
CHECKS = {"data", "source", "gradient", "numerical", "device", "profiler", "unused_heads",
          "oracle", "initialization", "completed_training"}
AUDIT_FIELDS = {
    "schema", "cohort", "condition", "support_cleared", "row_index", "group_index",
    "training_group_index", "original_update", "data_seed", "episode_id", "permutation_id",
    "correct_action", "query_direction", "agent_patch", "goal_patch",
    "observed_support_action_ids", "omitted_action", "correct_action_demonstrated",
    "inferred_controls", "input_sha256", "factual_input_sha256", "cleared_input_sha256",
    "metadata_sha256", "query_sha256", "targets_sha256", "label_sha256", "query_cells",
}
OUTPUT_FIELDS = {"logits", "attention", "pooled", "checkpointstage"}
SHA_FIELDS = {x for x in AUDIT_FIELDS if x.endswith("_sha256")}


class Invalid(ValueError):
    """Fail-closed integrity/control rejection, never a negative learning result."""


def require(ok, message):
    if not ok:
        raise Invalid(message)


def keys(value, expected, where):
    require(isinstance(value, dict) and set(value) == set(expected), f"{where}: exact keys required")


def integer(value, low, high, where):
    require(type(value) is int and low <= value <= high, f"{where}: invalid integer")
    return value


def digest(value, where="sha256"):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            f"{where}: invalid SHA256")
    return value


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def decode(raw):
    def bad_constant(value):
        raise Invalid(f"nonfinite JSON constant: {value}")
    return json.loads(raw, object_pairs_hook=strict_object, parse_constant=bad_constant)


def exact(left, right):
    """JSON equality that does not equate true with 1 or 1 with 1.0."""
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def pinned(record, campaign=None):
    keys(record, {"path", "sha256"}, "file binding")
    path = Path(record["path"])
    require(path.is_absolute(), "bound path must be absolute")
    path = path.resolve(strict=True)
    if campaign is not None:
        require(path.is_relative_to(campaign), "stream escapes campaign root")
    require(path.is_file() and path.stat().st_size <= 128 * 1024 * 1024, "invalid/oversized file")
    data = path.read_bytes()
    require(sha_bytes(data) == digest(record["sha256"]), f"hash mismatch: {path}")
    return data


def rows(raw, expected):
    lines = raw.splitlines()
    require(len(lines) == expected and all(lines), "incorrect JSONL row count or blank row")
    return [decode(line) for line in lines]


def finite_array(value, shape, where):
    def check(v):
        if isinstance(v, list):
            return all(check(x) for x in v)
        return type(v) in (int, float)
    require(check(value), f"{where}: numeric arrays only")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError, OverflowError) as error:
        raise Invalid(f"{where}: malformed array") from error
    require(array.shape == shape and np.isfinite(array).all(), f"{where}: invalid shape/nonfinite")
    require(np.max(np.abs(array), initial=0) <= np.finfo(np.float32).max, f"{where}: outside F32 range")
    return array


@functools.lru_cache(maxsize=24)
def public_metadata(actions):
    frames = []
    for step, action in enumerate(actions):
        for role in (0, 1):
            data = np.zeros((64, 10), dtype="<f4")
            data[:, role] = 1
            data[:, 3 + action] = 1
            data[:, 7] = np.arange(64, dtype=np.float32) % 8 / np.float32(7)
            data[:, 8] = np.arange(64, dtype=np.float32) // 8 / np.float32(7)
            data[:, 9] = np.float32(step) / np.float32(3)
            frames.append(data)
    data = np.zeros((64, 10), dtype="<f4")
    data[:, 2] = 1
    data[:, 7] = np.arange(64, dtype=np.float32) % 8 / np.float32(7)
    data[:, 8] = np.arange(64, dtype=np.float32) // 8 / np.float32(7)
    data[:, 9] = 1
    frames.append(data)
    return np.concatenate(frames).astype("<f4").tobytes()


@functools.lru_cache(maxsize=256)
def query_geometry(cells):
    require(len(cells) == 64 and all(type(x) is int and 0 <= x <= 3 for x in cells), "query cells")
    require(cells.count(2) == cells.count(3) == 1, "unique visible current roles required")
    require(all(cells[i] == 1 for i in range(64) if i % 8 in (0, 7) or i // 8 in (0, 7)),
            "query boundary walls")
    agent, goal = cells.index(2), cells.index(3)
    displacement = (goal % 8 - agent % 8, goal // 8 - agent // 8)
    directions = ((0, -1), (0, 1), (-1, 0), (1, 0))
    require(displacement in directions, "query must be distance one")
    patches = np.repeat(np.asarray(cells, dtype="<u4"), 64).tobytes()
    return agent, goal, directions.index(displacement), patches


def validate_identity(row, cohort, condition, index):
    keys(row, AUDIT_FIELDS, "audit row")
    group, slot = divmod(index, len(MAPS[cohort]))
    map_id = MAPS[cohort][slot]
    training_group = group * 4599 // 63 if cohort == "seen" else None
    expected = {
        "schema": DATA_SCHEMA, "cohort": cohort, "condition": condition,
        "support_cleared": condition == "cleared", "row_index": index, "group_index": group,
        "training_group_index": training_group,
        "original_update": training_group // 4 + 1 if training_group is not None else None,
        "data_seed": TRAIN_SEED if cohort == "seen" else EVAL_SEED,
        "episode_id": TRAIN_BASE + training_group if cohort == "seen" else EVAL_BASE + group,
        "permutation_id": map_id,
    }
    require(all(exact(row[k], v) for k, v in expected.items()), "registered cohort/group/map/seed/id order")
    for name in SHA_FIELDS:
        digest(row[name], name)
    actions = row["observed_support_action_ids"]
    require(isinstance(actions, list) and len(actions) == 3, "support action count")
    for value in actions:
        integer(value, 0, 3, "support action")
    require(len(set(actions)) == 3, "support actions must identify distinct primitives")
    omitted = next(iter(set(range(4)) - set(actions)))
    require(type(row["query_cells"]) is list, "query_cells list")
    agent, goal, direction, patches = query_geometry(tuple(row["query_cells"]))
    label = PERMUTATIONS[map_id].index(direction)
    derived = {
        "agent_patch": agent, "goal_patch": goal, "query_direction": direction,
        "correct_action": label, "omitted_action": omitted,
        "correct_action_demonstrated": label in actions,
        "inferred_controls": list(PERMUTATIONS[map_id]),
        "label_sha256": sha_bytes(np.asarray([label], dtype="<u4").tobytes()),
        "query_sha256": sha_bytes(patches),
        "metadata_sha256": sha_bytes(public_metadata(tuple(actions))),
        "cleared_input_sha256": sha_bytes(bytes(6 * 4096 * 4) + patches + public_metadata(tuple(actions))),
    }
    require(all(exact(row[k], v) for k, v in derived.items()), "visible query/control/hash derivation")
    expected_input = row["cleared_input_sha256"] if condition == "cleared" else row["factual_input_sha256"]
    require(row["input_sha256"] == expected_input, "input condition hash mismatch")
    require(row["factual_input_sha256"] != row["cleared_input_sha256"], "factual equals cleared input")


def validate_audits(audits):
    for cohort, maps in MAPS.items():
        factual, cleared = audits[f"{cohort}/factual"], audits[f"{cohort}/cleared"]
        for condition, population in (("factual", factual), ("cleared", cleared)):
            for index, row in enumerate(population):
                validate_identity(row, cohort, condition, index)
        for original, clear in zip(factual, cleared):
            change = dict(original, condition="cleared", support_cleared=True,
                          input_sha256=original["cleared_input_sha256"])
            require(exact(change, clear), "clearing changes audit/scoring fields")
        for group in range(GROUPS):
            block = factual[group * len(maps):(group + 1) * len(maps)]
            for field in ("query_cells", "query_sha256", "metadata_sha256", "observed_support_action_ids",
                          "omitted_action", "cleared_input_sha256"):
                require(all(exact(row[field], block[0][field]) for row in block), "unpaired query/public inputs")
            require(len({row["input_sha256"] for row in block}) == len(maps), "duplicate factual map input")
            require(np.bincount([row["correct_action"] for row in block], minlength=4).tolist()
                    == [len(maps) // 4] * 4, "unbalanced action group")
    first = {c: audits[f"{c}/factual"][::len(MAPS[c])] for c in MAPS}
    for familiar, heldout in zip(first["familiar"], first["heldout"]):
        for field in ("query_cells", "query_sha256", "metadata_sha256", "observed_support_action_ids",
                      "omitted_action", "data_seed", "episode_id", "cleared_input_sha256"):
            require(exact(familiar[field], heldout[field]), "familiar/heldout query pairing mismatch")
    unseen = {row["query_sha256"] for row in first["familiar"]}
    require(len(unseen) == GROUPS, "unseen query duplicates")
    require(not unseen.intersection(row["query_sha256"] for row in first["seen"]), "seen/unseen overlap")


def validate_config(config, campaign):
    keys(config, {"schema", "campaign", "source", "registration", "checkpoints", "audits", "streams", "integrity"}, "config")
    require(config["schema"] == SCHEMA, "config schema")
    keys(config["source"], {"revision", "binary_sha256", "dependency_revision"}, "source")
    source = config["source"]
    require(type(source["revision"]) is str and re.fullmatch(r"[0-9a-f]{40}", source["revision"]), "source revision")
    digest(source["binary_sha256"])
    require(source["dependency_revision"] == DEPENDENCY, "dependency revision")
    keys(config["checkpoints"], {"frozen", "final"}, "checkpoints")
    for name, updates in (("frozen", 0), ("final", 1150)):
        checkpoint = config["checkpoints"][name]
        keys(checkpoint, {"core_sha256", "policy_sha256", "updates"}, "checkpoint")
        digest(checkpoint["core_sha256"]); digest(checkpoint["policy_sha256"])
        require(type(checkpoint["updates"]) is int and checkpoint["updates"] == updates, "checkpoint update")
    require(config["checkpoints"]["frozen"] == {"core_sha256": INITIAL_CORE, "policy_sha256": INITIAL_POLICY, "updates": 0}, "frozen initialization")
    require(all(config["checkpoints"]["final"][k] != config["checkpoints"]["frozen"][k]
                for k in ("core_sha256", "policy_sha256")), "unchanged active checkpoint")
    keys(config["audits"], {f"{c}/{q}" for c in MAPS for q in ("factual", "cleared")}, "audit selectors")
    keys(config["streams"], {f"{s}/{c}" for s in STAGES for c in MAPS}, "stream selectors")
    paths = []
    for selector, record in config["streams"].items():
        stage, cohort = selector.split("/")
        condition = "cleared" if stage == "final_cleared" else "factual"
        checkpointstage = "frozen" if stage == "frozen_factual" else "final"
        keys(record, {"path", "sha256", "cohort", "condition", "checkpointstage", "source", "checkpoint", "audit_sha256"}, "stream binding")
        for key, expected in (("cohort", cohort), ("condition", condition), ("checkpointstage", checkpointstage),
                              ("source", config["source"]), ("checkpoint", config["checkpoints"][checkpointstage]),
                              ("audit_sha256", config["audits"][f"{cohort}/{condition}"]["sha256"])):
            require(exact(record[key], expected), "stream selector/source/checkpoint mismatch")
        paths.append(record["path"])
    paths.extend(record["path"] for record in config["audits"].values())
    require(len({str(Path(p).resolve()) for p in paths}) == len(paths), "reused stream/audit path")
    require(campaign.is_dir(), "campaign root missing")


def certificate(config):
    raw = pinned(config["integrity"])
    report = decode(raw)
    keys(report, {"schema", "accepted", "campaign", "source", "registration_sha256", "checkpoints",
                  "audit_sha256", "stream_sha256", "checks", "parameter_changes", "gradient_norms"}, "integrity report")
    require(report["schema"] == "looped-grounded-policy-integrity-v1", "integrity schema")
    require(report["accepted"] is True, "external integrity rejected")
    for field in ("campaign", "source", "checkpoints"):
        require(exact(report[field], config[field]), "external integrity context mismatch")
    require(report["registration_sha256"] == config["registration"]["sha256"], "integrity registration")
    for field, selected in (("audit_sha256", "audits"), ("stream_sha256", "streams")):
        require(exact(report[field], {k: v["sha256"] for k, v in config[selected].items()}), "integrity stream closure")
    keys(report["checks"], CHECKS, "integrity checks")
    require(all(value is True for value in report["checks"].values()), "external check rejected/untyped")
    for field in ("parameter_changes", "gradient_norms"):
        require(isinstance(report[field], dict) and report[field], f"missing {field}")
    return report


def measure(output_rows, identity, checkpointstage):
    logits, attention, pooled = [], [], []
    for row, expected in zip(output_rows, identity):
        keys(row, AUDIT_FIELDS | OUTPUT_FIELDS, "output row")
        require(exact({k: row[k] for k in AUDIT_FIELDS}, expected), "output does not match pinned audit")
        require(row["checkpointstage"] == checkpointstage, "output checkpoint stage")
        logits.append(finite_array(row["logits"], (4,), "logits"))
        attention.append(finite_array(row["attention"], (2, 64), "attention"))
        pooled.append(finite_array(row["pooled"], (256,), "pooled"))
    logits, attention = np.asarray(logits), np.asarray(attention)
    require(np.logical_and(attention >= 0, attention <= 1).all(), "attention probability range")
    require(np.allclose(attention.sum(axis=2), 1, atol=1e-5, rtol=1e-5), "attention normalization")
    labels = np.asarray([row["correct_action"] for row in identity])
    omitted = np.asarray([row["omitted_action"] for row in identity])
    roles = np.asarray([[row["agent_patch"], row["goal_patch"]] for row in identity])
    predictions = logits.argmax(axis=1)
    shifted = logits - logits.max(axis=1, keepdims=True)
    ce = np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(labels)), labels]
    require(np.isfinite(ce).all(), "nonfinite CE")
    correct = predictions == labels
    role_correct = attention.argmax(axis=2) == roles
    role_mass = np.take_along_axis(attention, roles[:, :, None], axis=2).squeeze(2)
    metrics = {
        "accuracy": correct.astype(float), "ce": ce,
        "agent_accuracy": role_correct[:, 0].astype(float), "goal_accuracy": role_correct[:, 1].astype(float),
        "joint_role_accuracy": role_correct.all(axis=1).astype(float),
        "agent_mass": role_mass[:, 0], "goal_mass": role_mass[:, 1],
    }
    m = len(labels) // GROUPS
    group_metrics = {name: value.reshape(GROUPS, m).mean(axis=1) for name, value in metrics.items()}
    group_metrics["all_maps_correct"] = correct.reshape(GROUPS, m).all(axis=1).astype(float)
    subsets = {}
    for name, mask in (("omitted", labels == omitted), ("demonstrated", labels != omitted)):
        counts = mask.reshape(GROUPS, m).sum(axis=1)
        require((counts > 0).all(), "empty registered subset")
        group_metrics[f"{name}_accuracy"] = (correct & mask).reshape(GROUPS, m).sum(axis=1) / counts
        subsets[name] = {"rows": int(mask.sum()), "correct": int((correct & mask).sum()),
                         "accuracy": float(correct[mask].mean())}
    for action in range(4):
        group_metrics[f"predicted_action_{action}_rate"] = (predictions == action).reshape(GROUPS, m).mean(axis=1)
        group_metrics[f"action_{action}_accuracy"] = (correct & (labels == action)).reshape(GROUPS, m).sum(axis=1) / (m // 4)
    summary = {
        "rows": len(labels), "groups": GROUPS, "maps_per_group": m, "correct": int(correct.sum()),
        "accuracy": float(correct.mean()), "ce": float(ce.mean()),
        "label_histogram": np.bincount(labels, minlength=4).tolist(),
        "prediction_histogram": np.bincount(predictions, minlength=4).tolist(),
        "all_maps_correct_groups": int(group_metrics["all_maps_correct"].sum()),
        "subsets": subsets,
    }
    if identity[0]["condition"] == "cleared":
        for group in range(GROUPS):
            block = logits[group * m:(group + 1) * m]
            pred = predictions[group * m:(group + 1) * m]
            require(np.allclose(block, block[0], atol=1e-5, rtol=1e-5), "cleared logits differ across maps")
            require((pred == pred[0]).all(), "cleared predictions differ across maps")
        require(int(correct.sum()) * 4 == len(labels), "cleared accuracy is not exactly one quarter")
    return summary, group_metrics


def bootstrap(group_metrics, indices):
    result = {}
    for name, values in group_metrics.items():
        samples = values[indices].mean(axis=1)
        low, high = np.quantile(samples, [0.025, 0.975], method="linear")
        result[name] = {"estimate": float(values.mean()), "ci95": [float(low), float(high)]}
    return result


def decide(summaries, contrasts):
    gate = {"seen_accuracy": summaries["final_factual/seen"]["accuracy"] >= .90}
    for cohort in ("familiar", "heldout"):
        gate[f"{cohort}_accuracy"] = summaries[f"final_factual/{cohort}"]["accuracy"] >= .90
        gate[f"{cohort}_improves_frozen"] = contrasts[cohort]["final_minus_frozen"]["accuracy"]["ci95"][0] > 0
        gate[f"{cohort}_beats_cleared"] = contrasts[cohort]["final_minus_cleared"]["accuracy"]["ci95"][0] > .25
    if not gate["seen_accuracy"]:
        decision = "training_feasibility_not_supported"
    elif not all(gate.values()):
        decision = "transfer_not_supported"
    else:
        decision = "supported_single_seed_screen"
    return decision, gate


def analyze(config_path, config_sha256):
    started = time.monotonic()
    config = decode(pinned({"path": str(config_path), "sha256": config_sha256}))
    require(isinstance(config, dict) and type(config.get("campaign")) is str, "config campaign")
    require(Path(config["campaign"]).is_absolute(), "absolute campaign required")
    campaign = Path(config["campaign"]).resolve(strict=True)
    validate_config(config, campaign)
    pinned(config["registration"])
    integrity = certificate(config)  # Before accessing any scored population/model output.
    audits = {selector: rows(pinned(record, campaign), GROUPS * len(MAPS[selector.split("/")[0]]))
              for selector, record in config["audits"].items()}
    validate_audits(audits)
    summaries, grouped = {}, {}
    for selector, record in config["streams"].items():
        identity = audits[f"{record['cohort']}/{record['condition']}"]
        raw = pinned({k: record[k] for k in ("path", "sha256")}, campaign)
        summaries[selector], grouped[selector] = measure(rows(raw, len(identity)), identity, record["checkpointstage"])
    contrasts = {}
    draws_sha = {}
    for cohort, seed in BOOTSTRAP_SEEDS.items():
        draws = np.random.Generator(np.random.PCG64(seed)).integers(0, GROUPS, size=(RESAMPLES, GROUPS))
        draws_sha[cohort] = sha_bytes(draws.astype("<u8").tobytes())
        for stage in STAGES:
            selector = f"{stage}/{cohort}"
            summaries[selector]["endpoints"] = bootstrap(grouped[selector], draws)
        contrasts[cohort] = {}
        final = grouped[f"final_factual/{cohort}"]
        for label, stage in (("final_minus_frozen", "frozen_factual"), ("final_minus_cleared", "final_cleared")):
            other = grouped[f"{stage}/{cohort}"]
            contrasts[cohort][label] = bootstrap({name: value - other[name] for name, value in final.items()}, draws)
    decision, gates = decide(summaries, contrasts)
    require(time.monotonic() - started < 120, "analysis exceeded 120-second budget")
    return {
        "schema": "looped-grounded-policy-analysis-v1", "accepted": True,
        "supported": decision == "supported_single_seed_screen", "decision": decision,
        "campaign": config["campaign"], "source": config["source"], "config_sha256": config_sha256,
        "registration_sha256": config["registration"]["sha256"], "integrity_sha256": config["integrity"]["sha256"],
        "checkpoints": config["checkpoints"], "audit_sha256": integrity["audit_sha256"],
        "stream_sha256": integrity["stream_sha256"], "summaries": summaries, "contrasts": contrasts,
        "gates": gates, "integrity_checks": integrity["checks"],
        "parameter_changes": integrity["parameter_changes"], "gradient_norms": integrity["gradient_norms"],
        "controls": {"oracle_accuracy": 1.0, "support_blind_accuracy": .25,
                     "action_id_only_accuracy": .25, "constant_accuracy": .25,
                     "always_undemonstrated": {"accuracy": .25, "omitted_accuracy": 1.0, "demonstrated_accuracy": 0.0},
                     "action_id_only_demonstrated_upper_bound": 1 / 3},
        "bootstrap": {"generator": "PCG64", "seeds": BOOTSTRAP_SEEDS, "resamples": RESAMPLES,
                      "groups": GROUPS, "paired_across_endpoints": True, "quantile_method": "linear",
                      "interval_scope": "descriptive_pointwise", "draws_u64le_sha256": draws_sha},
        "scope": "single-seed screen with privileged role warm start; no recurrence/planning/ARC claim",
        "created_local": datetime.datetime.now().astimezone().isoformat(),
        "elapsed_seconds": time.monotonic() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    def timeout(_signal, _frame):
        raise Invalid("analysis exceeded 120-second deadline")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(120)
    try:
        report = analyze(args.config, args.config_sha256)
        code = 0
    except (Invalid, OSError, ValueError, TypeError, KeyError, OverflowError) as error:
        report = {"schema": "looped-grounded-policy-analysis-v1", "accepted": False, "supported": False,
                  "decision": "inconclusive_failed_control", "error": str(error),
                  "config_sha256": args.config_sha256,
                  "created_local": datetime.datetime.now().astimezone().isoformat()}
        code = 1
    finally:
        signal.alarm(0)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
