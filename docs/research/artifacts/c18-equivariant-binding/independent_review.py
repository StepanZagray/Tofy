#!/usr/bin/env python3
"""Independent C18 finite-data and numerical review. No model or fit is run.

Dataset construction, scalar metrics and gate calculations are implemented here;
the primary scorer/data module is never imported. External runtime provenance is
shared with a pinned lifecycle receipt, explicitly not independently rerun.
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import signal
import struct
import sys
import time

sys.dont_write_bytecode = True
import numpy as np

R = Path(__file__).resolve().parent
SCHEMA = "looped-action-binding-analysis-config-v2"
FIT = (0, 2, 4, 5, 7, 8, 9, 10, 13, 14, 15, 16, 18, 19, 21, 23)
HELD = (1, 3, 6, 11, 12, 17, 20, 22)
MAPS = tuple(itertools.permutations(range(4)))
TRIPLES = tuple(itertools.permutations(range(4), 3))
DIRECTIONS = ((0, -1), (0, 1), (-1, 0), (1, 0))
ROW_FIELDS = set("index group_index map_id observed_actions desired_direction correct_action demonstrated features input_sha256 label_sha256 orbit_id canonical".split())
VISUAL_FIELDS = set("index audit features correct_action demonstrated input_sha256 label_sha256".split())
ADDED_FIELDS = {"logits", "stage", "loops", "cleared", "query_cleared", "model_input_sha256", "model_kind"}
AUDIT_FIELDS = set("schema cohort condition support_cleared row_index group_index training_group_index original_update data_seed episode_id permutation_id correct_action query_direction agent_patch goal_patch observed_support_action_ids omitted_action correct_action_demonstrated inferred_controls input_sha256 factual_input_sha256 cleared_input_sha256 metadata_sha256 query_sha256 targets_sha256 label_sha256 query_cells".split())
CHECKS = set("data source build initialization completed_training gradients device profiles checkpoints cleanup".split())
DEPENDENCY = "1ea5cc5c5998a1c2bd388eb158653dd0d1b7452a"
C15_ROOT = Path("/home/stepan/Projects/code/.tofy-runs/looped-demonstration-grounding-20260909T191228-IST")
C15_SEAL = Path("/home/stepan/Research/_runs/2026-09-09T173927Z-tofy-looped-demonstration-grounding/completed-campaign.manifest.json")
C15_SEAL_SHA = "6007293c5f51d263b0e3f40dcd819124efa8ffdaf2809202eda0c60142ffd40d"
C15_ROWS = C15_ROOT / "frames-initial/evaluation-rows.jsonl"
C15_ROWS_SHA = "91a08ec737c5b8b784086e730f637ec08afe60f84281ad9fc05380f723a3b25a"
C15_SOURCE = "61e1335670239627772088a2fb6c81c2df499ed2"
SCHEDULE_SHA = "a46eac4347bfe16d55d220240452cb67643ccd007a97d2e56da16be45cb6f51e"
C17_DATA = Path("/home/stepan/Research/_runs/2026-09-09T182014Z-tofy-looped-action-binding/dataset-b512.json")
C17_DATA_SHA = "bc1413652b6bf38cadde28c8f1fd3d88d9cf781ab55413300bc155e341c47295"
PRESENTATIONS = 588800


class Invalid(ValueError):
    pass


def need(ok, reason):
    if not ok:
        raise Invalid(reason)


def keys(value, expected, where):
    need(type(value) is dict and set(value) == set(expected), f"{where}: exact keys required")


def exact(a, b):
    if type(a) is not type(b):
        return False
    if type(a) is dict:
        return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if type(a) is list:
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


def decode(raw):
    def pairs(items):
        out = {}
        for k, v in items:
            need(k not in out, "duplicate JSON key")
            out[k] = v
        return out
    def constant(value):
        raise Invalid("nonfinite JSON: " + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    need(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value), "malformed SHA256")
    return value


def regular(path):
    path = Path(path)
    need(path.is_absolute() and path == path.resolve(strict=True) and path.is_file(), "absolute regular nonsymlink file required")
    return path


def file_sha(path):
    path = regular(path)
    before = path.stat()
    with path.open("rb") as handle:
        result = hashlib.file_digest(handle, "sha256").hexdigest()
    after = path.stat()
    need((before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
         (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), "file changed during hash")
    return result


def bound(path, expected):
    need(file_sha(path) == digest(expected), "bound file hash mismatch")
    return regular(path)


def load(path):
    path = regular(path)
    need(path.stat().st_size <= 128 * 1024 * 1024, "JSON size bound")
    return decode(path.read_bytes())


def jsonl(path):
    need(regular(path).stat().st_size <= 256 * 1024 * 1024, "row stream size bound")
    with Path(path).open("rb") as handle:
        while line := handle.readline(512 * 1024 + 1):
            need(line.strip() and len(line) <= 512 * 1024, "empty or oversized row")
            yield decode(line)


def frozen(files):
    need(type(files) is dict and files, "nonempty frozen bindings required")
    for path, value in files.items():
        bound(path, value)


def finite(value, shape):
    def numeric(x):
        return all(numeric(v) for v in x) if type(x) is list else type(x) in (int, float) and math.isfinite(x)
    need(numeric(value), "numeric finite array required")
    out = np.asarray(value, dtype=np.float64)
    need(out.shape == shape and np.isfinite(out).all() and (np.abs(out) <= np.finfo(np.float32).max).all(), "array shape/F32 range")
    return out


def feature_bytes(features, cleared=False, query_cleared=False):
    need(type(cleared) is type(query_cleared) is bool and not (cleared and query_cleared), "invalid combined ablation")
    values = finite(features, (4, 7)).tolist()
    for row in range(4):
        if (cleared and row < 3) or (query_cleared and row == 3):
            values[row][0] = values[row][1] = 0.0
    return struct.pack("<28f", *(value for row in values for value in row))


def row_hashes(features, label):
    need(type(label) is int and 0 <= label < 4, "integer action label")
    return {"input_sha256": sha(feature_bytes(features)), "label_sha256": sha(struct.pack("<I", label))}


def abstract_rows(ids):
    result = []
    for triple_number, actions in enumerate(TRIPLES):
        omitted = next(a for a in range(4) if a not in actions)
        for desired in range(4):
            for slot, map_id in enumerate(ids):
                permutation = MAPS[map_id]
                features = []
                for action in actions:
                    dx, dy = DIRECTIONS[permutation[action]]
                    features.append([float(dx), float(dy)] + [float(a == action) for a in range(4)] + [0.0])
                features.append([float(v) for v in DIRECTIONS[desired]] + [0.0] * 4 + [1.0])
                label = permutation.index(desired)
                result.append({"index": len(result), "group_index": triple_number * 4 + desired,
                               "map_id": map_id, "observed_actions": list(actions), "desired_direction": desired,
                               "correct_action": label, "demonstrated": label in actions, "features": features,
                               "orbit_id": slot * 16 + omitted * 4 + desired, "canonical": list(actions) == sorted(actions),
                               **row_hashes(features, label)})
    return result


def batch_size(value):
    need(type(value) is int and 512 <= value <= 32768 and value & (value - 1) == 0,
         "power-of-two effective batch512..32768 required")
    return value


def flat_schedule():
    generator = np.random.Generator(np.random.PCG64(20260922))
    epochs = [generator.permutation(1536) for _ in range(384)]
    indices = np.concatenate(epochs)[:PRESENTATIONS].astype("<u4")
    need(sha(indices.tobytes()) == SCHEDULE_SHA, "registered PCG64 schedule identity")
    return indices


def schedule(effective_batch, kind="training", smoke_updates=None):
    batch_size(effective_batch)
    need(kind in ("training", "smoke"), "schedule kind")
    need((kind == "training" and smoke_updates is None) or
         (kind == "smoke" and type(smoke_updates) is int and smoke_updates in (2, 5)),
         "fixed training schedule or exactly2/5 smoke updates required")
    count = PRESENTATIONS if kind == "training" else effective_batch * smoke_updates
    indices = flat_schedule()[:count]
    return [indices[start:start + effective_batch].tolist() for start in range(0, count, effective_batch)]


def completed_schedule(effective_batch):
    batch_size(effective_batch)
    return {"updates": (PRESENTATIONS + effective_batch - 1) // effective_batch,
            "effective_batch": effective_batch, "presentations": PRESENTATIONS,
            "indices_sha256": SCHEDULE_SHA, "tail_rows": PRESENTATIONS % effective_batch}


def analytic(row):
    # A direct action table completion, separate from matrix polynomial producer.
    x = finite(row["features"], (4, 7)).tolist()
    table = {}
    for record in x[:3]:
        onehot = record[2:6]
        need(onehot.count(1.0) == 1 and onehot.count(0.0) == 3, "analytic action onehot")
        action = onehot.index(1.0)
        need(action not in table, "analytic distinct actions")
        table[action] = record[:2]
    missing = next(a for a in range(4) if a not in table)
    table[missing] = [-math.fsum(e[axis] for e in table.values()) for axis in (0, 1)]
    return [math.fsum(table[a][axis] * x[3][axis] for axis in (0, 1)) for a in range(4)]


def abstract_audit(rows, exposure=None):
    n = len(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row["orbit_id"]].append(row["index"])
    orbits = []
    for orbit, members in sorted(groups.items()):
        representatives = [i for i in members if rows[i]["canonical"]]
        need(len(members) == 6 and representatives == [min(members)], "six-member canonical orbit")
        fingerprints = []
        for i in members:
            supports = sorted(rows[i]["features"][:3], key=lambda r: r[2:6].index(1.0))
            fingerprints.append(feature_bytes(supports + [rows[i]["features"][3]]))
        need(len(set(fingerprints)) == 1, "semantic orbit features differ")
        orbits.append({"orbit_id": orbit, "canonical_index": representatives[0], "member_indices": members,
                       "training_presentations": sum(exposure[i] for i in members) if exposure else 0})
    need(len(orbits) == n // 6, "orbit count")
    for row in rows:
        scores = analytic(row)
        need(scores[row["correct_action"]] == 1 and sorted(scores) == [-1, 0, 0, 1], "analytic cardinal gap1")
    for erased in ("effects", "query"):
        classes = defaultdict(list)
        for row in rows:
            classes[feature_bytes(row["features"], erased == "effects", erased == "query")].append(row["correct_action"])
        need(all(len(set(Counter(labels).values())) == 1 and set(labels) == set(range(4)) for labels in classes.values()), "ablated identical-input class not balanced")
    return {"rows": n, "groups": 96, "maps_per_group": n // 96,
            "label_counts": [sum(r["correct_action"] == a for r in rows) for a in range(4)],
            "demonstrated_rows": sum(r["demonstrated"] for r in rows), "omitted_rows": sum(not r["demonstrated"] for r in rows),
            "constant_accuracy": .25, "cleared_effects_accuracy": .25, "query_zero_accuracy": .25,
            "query_zero_missing_id_fallback": {"overall_accuracy": .25, "omitted_accuracy": 1.0, "demonstrated_accuracy": 0.0},
            "analytic_correct": n, "analytic_minimum_gap": 1.0,
            "inputs_sha256": sha(b"".join(feature_bytes(r["features"]) for r in rows)),
            "labels_sha256": sha(struct.pack("<" + "I" * n, *(r["correct_action"] for r in rows))),
            "unique_semantic_orbits": len(orbits), "order_variants_per_orbit": 6,
            "canonical_row_indices": [o["canonical_index"] for o in orbits], "orbits": orbits}


def visual_row(source, index):
    keys(source, AUDIT_FIELDS | {"checkpointstage", "logits", "attention", "pooled", "frames", "public_metadata"}, "C15 row")
    audit = {k: source[k] for k in AUDIT_FIELDS}
    group, slot = divmod(index, 16)
    original, map_id = group * 4599 // 63, FIT[slot]
    identity = {"schema": "looped-grounded-policy-data-v1", "cohort": "seen", "condition": "factual", "support_cleared": False,
                "row_index": index, "group_index": group, "training_group_index": original, "original_update": original // 4 + 1,
                "data_seed": 20260920, "episode_id": 0x47524F554E445452 + original, "permutation_id": map_id}
    need(all(exact(audit[k], v) for k, v in identity.items()) and source["checkpointstage"] == "frozen", "C15 initial identity")
    frames = source["frames"]
    need(type(frames) is list and len(frames) == 7, "seven C15 frames")
    cells, attention, roles = [], [], []
    for frame in frames:
        keys(frame, {"cells", "attention", "pooled"}, "C15 frame")
        cell = frame["cells"]
        need(type(cell) is list and len(cell) == 64 and all(type(c) is int and 0 <= c <= 3 for c in cell)
             and cell.count(2) == cell.count(3) == 1, "C15 visible roles")
        cells.append(cell); roles.append([cell.index(2), cell.index(3)])
        attention.append(finite(frame["attention"], (2, 64)))
    p = np.asarray(attention)
    need(((p >= 0) & (p <= 1)).all() and (np.abs(p.sum(axis=2) - 1) <= 1e-5).all(), "C15 probabilities")
    # Keep the registered F64 operation order before the sole F32 rounding.
    p = p / p.sum(axis=2, keepdims=True)
    grid = np.asarray([(i % 8, i // 8) for i in range(64)], dtype=np.float64)
    coordinates = p @ grid
    actions = audit["observed_support_action_ids"]
    need(type(actions) is list and len(actions) == 3 and all(type(a) is int and 0 <= a < 4 for a in actions)
         and len(set(actions)) == 3, "cached observed actions")
    unrounded = []
    for pair, action in enumerate(actions):
        movement = coordinates[2*pair+1, 0] - coordinates[2*pair, 0]
        unrounded.append(movement.tolist() + [float(a == action) for a in range(4)] + [0.0])
        a, b = roles[2*pair][0], roles[2*pair+1][0]
        need((b % 8 - a % 8, b // 8 - a // 8) == DIRECTIONS[MAPS[map_id][action]], "cached public displacement")
    unrounded.append((coordinates[6, 1] - coordinates[6, 0]).tolist() + [0.0]*4 + [1.0])
    raw = np.asarray(unrounded)
    features = raw.astype("<f4").astype(np.float64)
    rounding = np.abs(features - raw)
    agent, goal = roles[6]
    query_delta = (goal % 8 - agent % 8, goal // 8 - agent // 8)
    need(query_delta in DIRECTIONS, "cached query distance1")
    direction = DIRECTIONS.index(query_delta)
    label = MAPS[map_id].index(direction)
    need(exact(cells[6], audit["query_cells"]) and exact(frames[6]["attention"], source["attention"]), "cached current alias")
    derived = {"correct_action": label, "inferred_controls": list(MAPS[map_id]), "query_direction": direction,
               "agent_patch": agent, "goal_patch": goal, "correct_action_demonstrated": label in actions}
    need(all(exact(audit[k], v) for k, v in derived.items()), "cached label/control identity")
    patches = b"".join(struct.pack("<I", c) * 64 for frame in cells for c in frame)
    metadata = struct.pack("<4480f", *finite(source["public_metadata"], (4480,)))
    need(sha(patches + metadata) == audit["input_sha256"] == audit["factual_input_sha256"] and
         sha(patches[-16384:]) == audit["query_sha256"] and sha(metadata) == audit["metadata_sha256"] and
         sha(struct.pack("<I", label)) == audit["label_sha256"], "cached source public hashes")
    mass = np.asarray([[p[f, r, roles[f][r]] for r in range(2)] for f in range(7)])
    endpoint = 7 * (1 - mass)
    error = np.abs(coordinates - grid[np.asarray(roles)])
    need((error <= endpoint[..., None] + 1e-12).all(), "cached coordinate bound")
    effect_bounds = [endpoint[2*i, 0] + endpoint[2*i+1, 0] + max(rounding[i, :2]) for i in range(3)]
    effect_error = float(sum(effect_bounds))
    query_error = float(sum(endpoint[6]) + max(rounding[3, :2]))
    gap_bound = 1 - 2 * (effect_error + query_error + 2*effect_error*query_error) - 1e-10
    diagnostic = {"minimum_normalized_true_mass": float(mass.min()), "maximum_coordinate_absolute_error": float(error.max()),
                  "maximum_endpoint_bound": float(endpoint.max()), "maximum_f32_rounding_error": float(rounding.max()),
                  "analytic_gap_lower_bound": float(gap_bound)}
    result = {"index": index, "audit": audit, "features": features.tolist(), "correct_action": label,
              "demonstrated": label in actions, **row_hashes(features.tolist(), label)}
    need(gap_bound > 0 and max(range(4), key=lambda i: analytic(result)[i]) == label, "cached analytic positive control")
    return result, diagnostic


def cached_rows():
    bound(C15_SEAL, C15_SEAL_SHA)
    seal = load(C15_SEAL)
    need(seal["source"] == C15_SOURCE and seal["campaign"] == str(C15_ROOT)
         and seal["classification"] == "completed_exploratory_grounding_evidence", "C15 seal identity")
    need(exact(seal["files"]["frames-initial/evaluation-rows.jsonl"], {"sha256": C15_ROWS_SHA, "bytes": 98604396}), "C15 selected artifact")
    bound(C15_ROWS, C15_ROWS_SHA)
    need(C15_ROWS.stat().st_size == 98604396, "C15 selected byte count")
    rows, diagnostics = [], []
    for i, source in enumerate(jsonl(C15_ROWS)):
        need(i < 1024, "excess cached rows")
        row, diagnostic = visual_row(source, i)
        rows.append(row); diagnostics.append(diagnostic)
    need(len(rows) == 1024, "cached row count")
    bound(C15_ROWS, C15_ROWS_SHA)
    return rows, {"rows": 1024, "classification": "reused_visual_seam_evaluation_only",
                  "minimum_normalized_true_mass": min(d["minimum_normalized_true_mass"] for d in diagnostics),
                  "minimum_analytic_gap_lower_bound": min(d["analytic_gap_lower_bound"] for d in diagnostics),
                  **{k: max(d[k] for d in diagnostics) for k in ("maximum_coordinate_absolute_error", "maximum_endpoint_bound", "maximum_f32_rounding_error")},
                  "analytic_correct": 1024}


def compare(actual, expected, where="report", stats=None):
    if stats is None:
        stats = {"compared_float_fields": 0, "compared_discrete_fields": 0, "max_absolute_float_difference": 0.0}
    if type(expected) is dict:
        keys(actual, expected, where)
        for k in expected:
            compare(actual[k], expected[k], f"{where}/{k}", stats)
    elif type(expected) is list:
        need(type(actual) is list and len(actual) == len(expected), f"{where}: list length/type")
        for i, item in enumerate(expected):
            compare(actual[i], item, f"{where}/{i}", stats)
    elif type(expected) is float:
        need(type(actual) in (int, float) and math.isfinite(actual), f"{where}: finite number")
        error = abs(actual - expected)
        need(error <= 1e-12 + 1e-10*abs(expected), f"{where}: numeric mismatch {actual} != {expected}")
        stats["compared_float_fields"] += 1
        stats["max_absolute_float_difference"] = max(stats["max_absolute_float_difference"], error)
    else:
        need(exact(actual, expected), f"{where}: exact discrete mismatch")
        stats["compared_discrete_fields"] += 1
    return stats


def dataset(value, visual=None):
    keys(value, {"schema", "fit", "heldout", "updates", "cached_visual", "provenance", "audit", "schedule"}, "dataset")
    spec = value["schedule"]
    keys(spec, {"kind", "effective_batch", "presentations", "indices_sha256"}, "dataset schedule")
    need(type(value["updates"]) is list, "schedule updates must be a list")
    updates = schedule(spec["effective_batch"], spec["kind"], len(value["updates"]) if spec["kind"] == "smoke" else None)
    indices = list(itertools.chain.from_iterable(updates))
    count, batch = len(indices), spec["effective_batch"]
    indices_sha = sha(struct.pack("<" + "I" * count, *indices))
    need(exact(spec, {"kind": spec["kind"], "effective_batch": batch, "presentations": count,
                     "indices_sha256": indices_sha}), "schedule count or identity differs")
    expected_fit, expected_held = abstract_rows(FIT), abstract_rows(HELD)
    need(value["schema"] == "looped-action-binding-data-v2" and exact(value["fit"], expected_fit)
         and exact(value["heldout"], expected_held) and exact(value["updates"], updates), "finite dataset or schedule differs")
    exposure = Counter(indices)
    exposure.update({i: 0 for i in range(1536)})
    visual, visual_audit = cached_rows() if visual is None else visual
    need(exact(value["cached_visual"], visual), "cached rounded features/identity differs")
    provenance = {"data_seed": 20260922, "numpy_version": np.__version__, "source": C15_SOURCE,
                  "c17_dataset": {"path": str(C17_DATA), "sha256": C17_DATA_SHA},
                  "c15_seal": {"path": str(C15_SEAL), "sha256": C15_SEAL_SHA}, "c15_rows": {"path": str(C15_ROWS), "sha256": C15_ROWS_SHA}}
    need(exact(value["provenance"], provenance), "dataset provenance")
    audit = {"abstract": {"fit": abstract_audit(expected_fit, exposure), "heldout": abstract_audit(expected_held)},
             "cached_visual": visual_audit,
             "schedule": {"updates": len(updates), "batch": batch, "presentations": count,
                          "full_updates": sum(len(update) == batch for update in updates), "tail_rows": count % batch,
                          "completed_epochs": count // 1536, "partial_epoch_rows": count % 1536,
                          "minimum_presentations_per_row": min(exposure.values()), "maximum_presentations_per_row": max(exposure.values()),
                          "rows_with_383_presentations": sum(v == 383 for v in exposure.values()),
                          "rows_with_384_presentations": sum(v == 384 for v in exposure.values()), "indices_sha256": indices_sha}}
    compare(value["audit"], audit, "dataset/audit")
    return audit


def selectors():
    result = {}
    for cohort in ("fit", "heldout"):
        result[f"initial-{cohort}-l4"] = dict(cohort=cohort, stage="initial", loops=4, cleared=False, query_cleared=False)
        for depth in (1, 2, 4, 8):
            result[f"final-{cohort}-l{depth}"] = dict(cohort=cohort, stage="final", loops=depth, cleared=False, query_cleared=False)
        for suffix, clear, query in (("effects-zero", True, False), ("query-zero", False, True)):
            result[f"final-{cohort}-{suffix}"] = dict(cohort=cohort, stage="final", loops=4, cleared=clear, query_cleared=query)
    result["final-cached-visual-l4"] = dict(cohort="cached_visual", stage="final", loops=4, cleared=False, query_cleared=False)
    return result


def authority(config):
    keys(config, {"schema", "dataset", "dataset_sha256", "registration", "frozen_files", "streams", "integrity_receipt", "integrity_receipt_sha256", "model_kind", "effective_batch"}, "config")
    need(config["schema"] == SCHEMA, "analysis config schema")
    need(config["model_kind"] in ("legacy", "equivariant"), "registered model kind")
    expected_schedule = completed_schedule(config["effective_batch"])
    keys(config["registration"], {"path", "sha256"}, "registration binding")
    bound(config["registration"]["path"], config["registration"]["sha256"])
    bound(config["dataset"], config["dataset_sha256"])
    frozen(config["frozen_files"])
    for name in ("data.py", "analysis.py", "independent_review.py", "independent_review_tests.py"):
        path = R/name
        need(config["frozen_files"].get(str(path)) == file_sha(path), "review/data/scorer source not frozen")
    receipt = load(bound(config["integrity_receipt"], config["integrity_receipt_sha256"]))
    need(receipt.get("schema") == "looped-action-binding-integrity-v2" and receipt.get("accepted") is True, "external integrity rejected")
    need(exact(receipt.get("model_kind"), config["model_kind"]) and
         exact(receipt.get("effective_batch"), config["effective_batch"]) and
         exact(receipt.get("completed_schedule"), expected_schedule), "receipt model/batch/completed schedule differs")
    keys(receipt.get("checks"), CHECKS, "integrity checks")
    need(all(v is True for v in receipt["checks"].values()), "external integrity component failed")
    need(type(receipt.get("source_revision")) is str and re.fullmatch(r"[0-9a-f]{40}", receipt["source_revision"]), "source revision")
    digest(receipt.get("binary_sha256"))
    need(receipt.get("dependency_revision") == DEPENDENCY and receipt.get("dataset_sha256") == config["dataset_sha256"]
         and receipt.get("registration_sha256") == config["registration"]["sha256"], "receipt identity differs")
    keys(receipt.get("checkpoints"), {"initial", "final"}, "checkpoints")
    for checkpoint in receipt["checkpoints"].values():
        keys(checkpoint, {"sha256", "parameter_sha256"}, "checkpoint digest")
        for value in checkpoint.values():
            digest(value)
    need(receipt["checkpoints"]["initial"]["parameter_sha256"] != receipt["checkpoints"]["final"]["parameter_sha256"], "terminal parameters unchanged")
    frozen(receipt.get("frozen_files"))
    closure = dict(receipt["frozen_files"])
    need(config["integrity_receipt"] not in closure, "receipt cannot bind itself")
    closure[config["integrity_receipt"]] = config["integrity_receipt_sha256"]
    need(exact(config["frozen_files"], closure), "receipt/config frozen closure differs")
    expected = selectors()
    keys(config["streams"], expected, "streams")
    keys(receipt.get("streams"), expected, "receipt streams")
    paths = []
    for name, selected in expected.items():
        record = config["streams"][name]
        keys(record, set(selected) | {"rows", "sha256"}, "stream record")
        need(all(exact(record[k], v) for k, v in selected.items()), "stream selector identity")
        path = bound(record["rows"], record["sha256"])
        paths.append(path)
        need(closure.get(str(path)) == record["sha256"], "stream not frozen")
        checkpoint = receipt["checkpoints"][record["stage"]]
        need(exact(receipt["streams"][name], {**record, "checkpoint_sha256": checkpoint["sha256"], "parameter_sha256": checkpoint["parameter_sha256"]}), "stream checkpoint binding")
    need(len(set(paths)) == 15 and closure.get(config["dataset"]) == config["dataset_sha256"]
         and closure.get(config["registration"]["path"]) == config["registration"]["sha256"], "selected input binding")
    return receipt


def stream(record, rows, model_kind):
    need(model_kind in ("legacy", "equivariant"), "output model kind")
    logits, hashes = [], []
    pairs = itertools.zip_longest(jsonl(record["rows"]), rows)
    for output, row in pairs:
        need(output is not None and row is not None, "stream row count mismatch")
        keys(output, set(row) | ADDED_FIELDS, "evaluation row")
        need(all(exact(output[k], v) for k, v in row.items()), "dataset/output row identity")
        for field in ("stage", "loops", "cleared", "query_cleared"):
            need(exact(output[field], record[field]), "output stage/intervention differs")
        need(exact(output["model_kind"], model_kind), "output model kind differs")
        actual = sha(feature_bytes(row["features"], record["cleared"], record["query_cleared"]))
        need(output["model_input_sha256"] == actual, "actual postclamp input hash differs")
        logits.append(finite(output["logits"], (4,)).tolist()); hashes.append(actual)
    return logits, hashes


def row_metrics(logits, label):
    need(type(label) is int and 0 <= label < 4, "integer metric label")
    z = finite(logits, (4,)).tolist()
    winner = max(range(4), key=lambda a: z[a])
    maximum = z[winner]
    ce = math.log(math.fsum(math.exp(v - maximum) for v in z)) + (maximum - z[label])
    margin = z[label] - max(v for a, v in enumerate(z) if a != label)
    ordered = sorted(z, reverse=True)
    return {"winner": winner, "correct": int(winner == label), "ce": ce, "margin": margin, "winner_margin": ordered[0] - ordered[1]}


def basic(rows, metrics, indices):
    chosen = list(indices)
    n = len(chosen)
    correct = sum(metrics[i]["correct"] for i in chosen)
    return {"rows": n, "correct": correct, "accuracy": correct/n if n else None,
            "ce": math.fsum(metrics[i]["ce"] for i in chosen)/n if n else None,
            "margin": {"mean": math.fsum(metrics[i]["margin"] for i in chosen)/n if n else None, "min": min(metrics[i]["margin"] for i in chosen) if n else None},
            "winner_margin": {"mean": math.fsum(metrics[i]["winner_margin"] for i in chosen)/n if n else None, "min": min(metrics[i]["winner_margin"] for i in chosen) if n else None},
            "label_histogram": [sum(rows[i]["correct_action"] == a for i in chosen) for a in range(4)],
            "prediction_histogram": [sum(metrics[i]["winner"] == a for i in chosen) for a in range(4)]}


def map_id(row):
    return row["map_id"] if "map_id" in row else row["audit"]["permutation_id"]


def detailed(rows, metrics, indices):
    indices = list(indices)
    result = basic(rows, metrics, indices)
    result["subsets"] = {name: basic(rows, metrics, [i for i in indices if rows[i]["demonstrated"] == demonstrated])
                         for name, demonstrated in (("demonstrated", True), ("omitted", False))}
    result["per_map"] = {str(mid): basic(rows, metrics, [i for i in indices if map_id(rows[i]) == mid]) for mid in sorted({map_id(rows[i]) for i in indices})}
    return result


def ordered_orbits(rows):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["orbit_id"]].append(i)
    return [(orbit, indices) for orbit, indices in sorted(groups.items())]


def summary(rows, logits, cached=False):
    metrics = [row_metrics(z, r["correct_action"]) for r, z in zip(rows, logits, strict=True)]
    if cached:
        canonical = list(range(len(rows)))
        ordering = {"applicable": False, "orbits": None, "inconsistent_orbits": None, "winner_invariant": None, "max_logit_deviation": None}
    else:
        orbits = ordered_orbits(rows)
        canonical, inconsistent, deviation = [], 0, 0.0
        for _, indices in orbits:
            reps = [i for i in indices if rows[i]["canonical"]]
            need(len(indices) == 6 and len(reps) == 1, "complete six-order orbit")
            reference = reps[0]
            canonical.append(reference)
            inconsistent += len({metrics[i]["winner"] for i in indices}) != 1
            deviation = max(deviation, *(max(logits[i][a] for i in indices) - min(logits[i][a] for i in indices) for a in range(4)))
        ordering = {"applicable": True, "orbits": len(orbits), "inconsistent_orbits": inconsistent,
                    "winner_invariant": inconsistent == 0, "max_logit_deviation": float(deviation)}
    result = {"raw": detailed(rows, metrics, range(len(rows))), "canonical": detailed(rows, metrics, canonical),
              "canonical_basis": "all_retained_cached_visual_rows" if cached else "ascending_observed_action_ids", "ordering": ordering,
              "control": None}
    return result, metrics, canonical


def paired(rows, first, second, indices):
    indices = list(indices)
    n = len(indices)
    return {"rows": n, "accuracy_delta": (sum(first[i]["correct"] for i in indices) - sum(second[i]["correct"] for i in indices))/n if n else None,
            "ce_delta": math.fsum(first[i]["ce"] - second[i]["ce"] for i in indices)/n if n else None,
            "margin_delta": math.fsum(first[i]["margin"] - second[i]["margin"] for i in indices)/n if n else None,
            "both_correct": sum(first[i]["correct"] and second[i]["correct"] for i in indices),
            "first_only_correct": sum(first[i]["correct"] and not second[i]["correct"] for i in indices),
            "second_only_correct": sum(not first[i]["correct"] and second[i]["correct"] for i in indices),
            "changed_predictions": sum(first[i]["winner"] != second[i]["winner"] for i in indices)}


def paired_detailed(rows, first, second, indices):
    indices = list(indices)
    result = paired(rows, first, second, indices)
    result["subsets"] = {name: paired(rows, first, second, [i for i in indices if rows[i]["demonstrated"] == demonstrated])
                         for name, demonstrated in (("demonstrated", True), ("omitted", False))}
    return result


def ablation_control(rows, logits, hashes, canonical):
    metrics = [row_metrics(z, row["correct_action"]) for row, z in zip(rows, logits, strict=True)]
    classes = defaultdict(list)
    for i, h in enumerate(hashes):
        classes[h].append(i)
    for indices in classes.values():
        labels = Counter(rows[i]["correct_action"] for i in indices)
        need(set(labels) == set(range(4)) and len(set(labels.values())) == 1, "ablation class labels not balanced")
        need(len({metrics[i]["winner"] for i in indices}) == 1, "identical ablation input has different winners")
    raw = sum(m["correct"] for m in metrics)
    canonical_correct = sum(metrics[i]["correct"] for i in canonical)
    need(raw*4 == len(rows) and canonical_correct*4 == len(canonical), "deterministic ablation not exactly25percent")
    return {"identical_input_groups": len(classes), "inconsistent_groups": 0, "winner_invariant": True,
            "balanced_labels": True, "correct": raw, "rows": len(rows), "exact_quarter": True,
            "interpretation": "25% follows from balanced labels on identical ablated inputs, not learned collapse."}


def external_controls(rows, cached=False):
    canonical = list(range(len(rows))) if cached else [i for i, r in enumerate(rows) if r["canonical"]]
    scores = [analytic(r) for r in rows]
    predictions = {"constant_action0": [0]*len(rows),
                   "analytic": [max(range(4), key=lambda a: z[a]) for z in scores],
                   "always_missing_id": [next(a for a in range(4) if a not in (r["audit"]["observed_support_action_ids"] if cached else r["observed_actions"])) for r in rows]}
    def counts(predicted, indices):
        indices = list(indices)
        correct = sum(predicted[i] == rows[i]["correct_action"] for i in indices)
        return {"rows": len(indices), "correct": correct, "accuracy": correct/len(indices) if indices else None}
    result = {}
    for name, predicted in predictions.items():
        result[name] = {}
        for basis, indices in (("raw", list(range(len(rows)))), ("canonical", canonical)):
            out = counts(predicted, indices)
            out["subsets"] = {label: counts(predicted, [i for i in indices if rows[i]["demonstrated"] == demonstrated])
                              for label, demonstrated in (("demonstrated", True), ("omitted", False))}
            result[name][basis] = out
            need(out["correct"] == out["rows"] if name == "analytic" else 4*out["correct"] == out["rows"], "external control correctness")
            if name == "always_missing_id":
                need(out["subsets"]["omitted"]["correct"] == out["subsets"]["omitted"]["rows"] and out["subsets"]["demonstrated"]["correct"] == 0,
                     "missing-ID subset control")
        if name == "analytic":
            result[name]["minimum_true_score_margin"] = min(z[r["correct_action"]] - max(z[a] for a in range(4) if a != r["correct_action"]) for z, r in zip(scores, rows, strict=True))
    return result


def legacy_gates(summaries):
    fit = summaries["final-fit-l4"]["canonical"]
    held = summaries["final-heldout-l4"]["canonical"]
    result = {
        "fit_254_of_256": fit["rows"] == 256 and fit["correct"] >= 254,
        "heldout_116_of_128": held["rows"] == 128 and held["correct"] >= 116,
        "heldout_demonstrated_87_of_96": held["subsets"]["demonstrated"]["rows"] == 96 and held["subsets"]["demonstrated"]["correct"] >= 87,
        "heldout_omitted_29_of_32": held["subsets"]["omitted"]["rows"] == 32 and held["subsets"]["omitted"]["correct"] >= 29,
        "terminal_l4_ordering": all(summaries[f"final-{c}-l4"]["ordering"]["winner_invariant"] is True for c in ("fit", "heldout")),
        "ablation_controls": all(summaries[f"final-{c}-{a}"]["control"]["winner_invariant"] is True and
                                 summaries[f"final-{c}-{a}"]["control"]["exact_quarter"] is True
                                 for c in ("fit", "heldout") for a in ("effects-zero", "query-zero")),
    }
    return ("supported_single_seed_binding_prerequisite" if all(result.values()) else "registered_binding_screen_not_supported"), result


def action_equivariance(populations, logits):
    """Align action scores to physical effects for audit only, using scalar indexing."""
    groups = defaultdict(dict)
    for cohort in ("fit", "heldout"):
        rows = populations[cohort]
        need(len(logits[cohort]) == len(rows), "equivariance row count")
        for row, raw in zip(rows, logits[cohort], strict=True):
            values = finite(raw, (4,)).tolist()
            if not row["canonical"]:
                continue
            mapping = MAPS[row["map_id"]]
            missing = next(a for a in range(4) if a not in row["observed_actions"])
            orbit = mapping[missing], row["desired_direction"]
            need(row["map_id"] not in groups[orbit], "duplicate map in action-renaming orbit")
            aligned = [0.0] * 4
            for action, effect in enumerate(mapping):
                aligned[effect] = values[action]
            groups[orbit][row["map_id"]] = aligned
    need(set(groups) == set(itertools.product(range(4), repeat=2)), "all16 action-renaming orbits required")
    records = {}
    for (missing, desired), members in sorted(groups.items()):
        need(set(members) == set(range(24)), "each action-renaming orbit requires all24 maps")
        reference = members[0]
        errors, ratios, metrics = [], [], []
        numerical = True
        for map_index in range(24):
            values = members[map_index]
            metrics.append(row_metrics(values, desired))
            for effect in range(4):
                error = abs(values[effect] - reference[effect])
                tolerance = 1e-4 + 1e-5 * abs(reference[effect])
                errors.append(error)
                ratios.append(error / tolerance)
                numerical = numerical and error <= tolerance
        minimum_winner = min(metric["winner_margin"] for metric in metrics)
        eligible = minimum_winner > 0 and minimum_winner > 2 * max(errors)
        minimum = min(metric["margin"] for metric in metrics)
        records[f"{missing}/{desired}"] = {
            "omitted_effect": missing, "desired_direction": desired, "maps": 24, "reference_map_id": 0,
            "reference_logits_effect_order": reference,
            "maximum_absolute_logit_error": max(errors), "maximum_tolerance_ratio": max(ratios),
            "numerically_equivariant": numerical, "correct": sum(metric["correct"] for metric in metrics),
            "minimum_true_margin": minimum, "minimum_winner_margin": minimum_winner,
            "strict_true_margin": minimum >= .001, "positive_margin_winner_comparison": eligible,
            "winner_invariant": len({metric["winner"] for metric in metrics}) == 1 if eligible else None}
    values = list(records.values())
    return {"orbits": 16, "canonical_rows": 384, "maps_per_orbit": 24, "reference_map_id": 0,
            "atol": 1e-4, "rtol": 1e-5, "required_true_margin": .001,
            "winner_comparison_criterion": "minimum_winner_margin > 2*maximum_absolute_logit_error and >0",
            "maximum_absolute_logit_error": max(value["maximum_absolute_logit_error"] for value in values),
            "maximum_tolerance_ratio": max(value["maximum_tolerance_ratio"] for value in values),
            "numerically_equivariant": all(value["numerically_equivariant"] for value in values),
            "positive_margin_orbits": sum(value["positive_margin_winner_comparison"] for value in values),
            "winner_inconsistent_orbits": sum(value["winner_invariant"] is False for value in values),
            "all_eligible_winners_invariant": all(value["winner_invariant"] is not False for value in values),
            "strict_true_margin_orbits": sum(value["strict_true_margin"] for value in values),
            "minimum_true_margin": min(value["minimum_true_margin"] for value in values), "by_orbit": records}


def gates(summaries, model_kind, equivariance):
    old_decision, old = legacy_gates(summaries)
    need(model_kind in ("legacy", "equivariant"), "selection model kind")
    if model_kind == "legacy":
        return old_decision, {"legacy_descriptive": old, "selection": old}
    selected = {
        "initial_numerical_equivariance": equivariance["initial"]["numerically_equivariant"],
        "final_numerical_equivariance": equivariance["final"]["numerically_equivariant"],
        "positive_margin_winners": all(equivariance[stage]["all_eligible_winners_invariant"] for stage in ("initial", "final")),
        "all16_strict_true_margin": equivariance["final"]["strict_true_margin_orbits"] == 16,
        "terminal_l4_ordering": old["terminal_l4_ordering"], "ablation_controls": old["ablation_controls"]}
    decision = "supported_single_seed_action_equivariant_selection" if all(selected.values()) else "action_equivariant_selection_not_supported"
    return decision, {"legacy_descriptive": old, "selection": selected}


def reconstruction(data, records, model_kind):
    summaries, metrics, canonical = {}, {}, {}
    factual_logits = {"initial": {}, "final": {}}
    for name, spec in selectors().items():
        rows = data[spec["cohort"]]
        logits, hashes = stream(records[name], rows, model_kind)
        summaries[name], metrics[name], canonical[name] = summary(rows, logits, spec["cohort"] == "cached_visual")
        if spec["cleared"] or spec["query_cleared"]:
            summaries[name]["control"] = ablation_control(rows, logits, hashes, canonical[name])
        elif spec["cohort"] in ("fit", "heldout") and spec["loops"] == 4:
            factual_logits[spec["stage"]][spec["cohort"]] = logits
    contrasts = {}
    for cohort in ("fit", "heldout"):
        factual = f"final-{cohort}-l4"
        for label, other in (("final_minus_initial", f"initial-{cohort}-l4"),
                             ("factual_minus_effects_zero", f"final-{cohort}-effects-zero"),
                             ("factual_minus_query_zero", f"final-{cohort}-query-zero")):
            contrasts[f"{label}/{cohort}"] = {basis: paired_detailed(data[cohort], metrics[factual], metrics[other], indices)
                for basis, indices in (("raw", range(len(data[cohort]))), ("canonical", canonical[factual]))}
    equivariance = {stage: action_equivariance(data, values) for stage, values in factual_logits.items()}
    decision, components = gates(summaries, model_kind, equivariance)
    return {"summaries": summaries, "contrasts": contrasts, "gates": components, "decision": decision,
            "legacy_decision": legacy_gates(summaries)[0], "action_equivariance": equivariance,
            "external_controls": {cohort: external_controls(data[cohort], cohort == "cached_visual") for cohort in ("fit", "heldout", "cached_visual")}}


def review(config_path, report_path):
    started = time.monotonic()
    config_sha, report_sha = file_sha(config_path), file_sha(report_path)
    config = load(config_path)
    receipt = authority(config)
    data = load(config["dataset"])
    need(type(data.get("schedule")) is dict and data["schedule"].get("kind") == "training" and
         exact(data["schedule"].get("effective_batch"), config["effective_batch"]), "analysis requires matched complete training dataset")
    audit = dataset(data)
    report = load(report_path)
    expected = reconstruction(data, config["streams"], config["model_kind"])
    header = {"schema": "looped-action-binding-analysis-v2", "accepted": True, "classification": "single_seed_action_equivariance_screen",
              "model_kind": config["model_kind"], "effective_batch": config["effective_batch"],
              "completed_schedule": completed_schedule(config["effective_batch"]),
              "config_sha256": config_sha, "dataset_sha256": config["dataset_sha256"],
              "registration_sha256": config["registration"]["sha256"], "analysis_sha256": file_sha(R/"analysis.py"),
              "data_helper_sha256": file_sha(R/"data.py"), "integrity_receipt_sha256": config["integrity_receipt_sha256"],
              "source_revision": receipt["source_revision"], "binary_sha256": receipt["binary_sha256"]}
    keys(report, set(header) | set(expected) | {"limits", "elapsed_seconds"}, "primary report")
    compare({k: report[k] for k in header}, header)
    need(type(report["limits"]) is list and report["limits"] and all(type(s) is str and s for s in report["limits"]), "report scope limits")
    need(type(report["elapsed_seconds"]) in (int, float) and math.isfinite(report["elapsed_seconds"]) and 0 <= report["elapsed_seconds"] <= 120,
         "primary analysis runtime")
    stats = compare({k: report[k] for k in expected}, expected)
    authority(config)
    need(file_sha(config_path) == config_sha and file_sha(report_path) == report_sha, "config/report drift")
    need(time.monotonic() - started <= 120, "review exceeded120seconds")
    return {"schema": "looped-action-binding-independent-review-v2", "accepted": True,
            "model_kind": config["model_kind"], "effective_batch": config["effective_batch"],
            "completed_schedule": completed_schedule(config["effective_batch"]),
            "report_sha256": report_sha, "config_sha256": config_sha, "dataset_sha256": config["dataset_sha256"],
            "reviewer_sha256": file_sha(Path(__file__).resolve()), **stats, "decision": expected["decision"], "gates": expected["gates"],
            "verified_streams": list(selectors()), "schedule_sha256": audit["schedule"]["indices_sha256"],
            "canonical_orbits": {"fit": 256, "heldout": 128}, "action_renaming_orbits": 16,
            "elapsed_seconds": time.monotonic()-started,
            "scope": "Independent finite rows/regrouped schedule/tail/orbit reconstruction, cached attention adapter, scalar CE/margins/counts, postclamp feature hashes, paired differences, analytic/missing/constant controls, action-renaming alignment and exact gates. Equivariant held-map correctness under correct fitting is a symmetry consequence, not independent semantic generalization. No fitting or model inference. External source/build/initialization/training/gradient/device/profile/checkpoint/cleanup acceptance shares the pinned lifecycle receipt; selected C15 cached producer provenance is inherited."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("config", "report", "output"):
        parser.add_argument("--"+argument, type=Path, required=True)
    args = parser.parse_args()
    need(args.output.is_absolute() and args.output.parent == args.output.parent.resolve(strict=True)
         and not args.output.exists() and not args.output.is_symlink(), "new absolute review output required")
    def timeout(_signal, _frame):
        raise Invalid("review exceeded120seconds")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(120)
    result = review(args.config, args.report)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    signal.alarm(0)


if __name__ == "__main__":
    main()
