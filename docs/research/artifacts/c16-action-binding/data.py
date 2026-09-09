#!/usr/bin/env python3
"""C16 finite abstract inputs plus a pinned, evaluation-only C15 visual cache.

No model, optimizer, simulator or fitting routine is imported. Labels and the
analytic solver are audit/scoring outputs; the learned model receives features.
"""
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import signal
import sys
import numpy as np

sys.dont_write_bytecode = True
SCHEMA = "looped-action-binding-data-v1"
FIT = (0, 2, 4, 5, 7, 8, 9, 10, 13, 14, 15, 16, 18, 19, 21, 23)
HELD = (1, 3, 6, 11, 12, 17, 20, 22)
MAPS = tuple(itertools.permutations(range(4)))
TRIPLES = tuple(itertools.permutations(range(4), 3))
DIRECTIONS = np.array(((0, -1), (0, 1), (-1, 0), (1, 0)), dtype=float)
GRID = np.array([(x, y) for y in range(8) for x in range(8)], dtype=float)
SEED, UPDATES, BATCH = 20260922, 1150, 512
C15_ROOT = Path("/home/stepan/Projects/code/.tofy-runs/looped-demonstration-grounding-20260909T191228-IST")
C15_SEAL = Path("/home/stepan/Research/_runs/2026-09-09T173927Z-tofy-looped-demonstration-grounding/completed-campaign.manifest.json")
C15_SEAL_SHA = "6007293c5f51d263b0e3f40dcd819124efa8ffdaf2809202eda0c60142ffd40d"
C15_SOURCE = "61e1335670239627772088a2fb6c81c2df499ed2"
C15_ROWS = C15_ROOT / "frames-initial/evaluation-rows.jsonl"
C15_ROWS_SHA = "91a08ec737c5b8b784086e730f637ec08afe60f84281ad9fc05380f723a3b25a"
AUDIT_FIELDS = set("schema cohort condition support_cleared row_index group_index training_group_index original_update data_seed episode_id permutation_id correct_action query_direction agent_patch goal_patch observed_support_action_ids omitted_action correct_action_demonstrated inferred_controls input_sha256 factual_input_sha256 cleared_input_sha256 metadata_sha256 query_sha256 targets_sha256 label_sha256 query_cells".split())
ROW_FIELDS = set("index group_index map_id observed_actions desired_direction correct_action demonstrated features input_sha256 label_sha256 orbit_id canonical".split())


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def exact(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def bad(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)


def canonical_bytes(value):
    """Deterministic nested key ordering, independent of Python's hash seed."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def file_sha(path):
    path = Path(path)
    require(path.is_absolute() and path == path.resolve(strict=True) and path.is_file(), "regular nonsymlink absolute file required")
    before = path.stat()
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns), "file changed while hashing")
    return digest


def finite(value, shape):
    def numeric(v):
        return all(numeric(x) for x in v) if isinstance(v, list) else type(v) in (int, float)
    require(numeric(value), "numeric arrays only")
    array = np.asarray(value, dtype=float)
    require(array.shape == shape and np.isfinite(array).all(), "array shape/nonfinite")
    return array


def identity(features, label):
    return {"input_sha256": sha(np.asarray(features, dtype="<f4").tobytes()),
            "label_sha256": sha(np.asarray([label], dtype="<u4").tobytes())}


def records(effects, actions, desired):
    require(len(actions) == 3 and all(type(x) is int and 0 <= x < 4 for x in actions)
            and len(set(actions)) == 3, "three distinct public actions required")
    result = np.zeros((4, 7), dtype=float)
    result[:3, :2], result[3, :2] = effects, desired
    result[np.arange(3), np.asarray(actions) + 2] = 1
    result[3, 6] = 1
    return result


def abstract_rows(map_ids):
    result = []
    for triple_index, actions in enumerate(TRIPLES):
        for direction in range(4):
            for slot, map_id in enumerate(map_ids):
                mapping = MAPS[map_id]
                label = mapping.index(direction)
                features = records(DIRECTIONS[[mapping[a] for a in actions]], list(actions), DIRECTIONS[direction]).tolist()
                result.append(dict(index=len(result), group_index=4 * triple_index + direction,
                                   map_id=map_id, observed_actions=list(actions), desired_direction=direction,
                                   orbit_id=slot * 16 + next(a for a in range(4) if a not in actions) * 4 + direction,
                                   canonical=list(actions) == sorted(actions),
                                   correct_action=label, demonstrated=label in actions, features=features,
                                   **identity(features, label)))
    return result


def schedule():
    rng = np.random.Generator(np.random.PCG64(SEED))
    batches = []
    while len(batches) < UPDATES:
        epoch = rng.permutation(1536).reshape(3, BATCH)
        batches.extend(epoch[:min(3, UPDATES - len(batches))].tolist())
    return batches


def analytic_scores(features):
    """External polynomial control, never a model-forward implementation."""
    x = np.asarray(features, dtype=float)
    require(x.shape == (4, 7) and np.isfinite(x).all(), "analytic input")
    a, e, q = x[:3, 2:6], x[:3, :2], x[3, :2]
    missing = np.ones(4) - a.T @ np.ones(3)
    full_effects = a.T @ e - missing[:, None] * e.sum(axis=0)
    return full_effects @ q


def soft_features(attention, actions):
    """Inference seam: attention/public action IDs only; no cells, roles or label."""
    p = finite(attention, (7, 2, 64))
    require(((0 <= p) & (p <= 1)).all(), "attention probability range")
    sums = p.sum(axis=-1, keepdims=True)
    require((np.abs(sums - 1) <= 1e-5).all(), "attention normalization")
    normalized = p / sums
    coordinates = normalized @ GRID
    x = records(coordinates[1:6:2, 0] - coordinates[0:6:2, 0], actions,
                coordinates[6, 1] - coordinates[6, 0])
    rounded = x.astype("<f4").astype(float)
    require(np.isfinite(rounded).all(), "F32 feature overflow")
    return rounded, normalized, coordinates, np.abs(rounded - x)


def coordinate_audit(normalized, coordinates, cells, rounding):
    """Scoring-only geometry and perturbation bounds; not inputs to the binder."""
    truth = []
    for frame in cells:
        require(type(frame) is list and len(frame) == 64 and all(type(x) is int and 0 <= x <= 3 for x in frame), "frame cells")
        require(frame.count(2) == frame.count(3) == 1, "unique public roles")
        truth.append([frame.index(2), frame.index(3)])
    truth = np.asarray(truth)
    p_true = np.take_along_axis(normalized, truth[..., None], axis=-1)[..., 0]
    endpoint_bound = 7 * (1 - p_true)
    actual_error = np.abs(coordinates - GRID[truth])
    require((actual_error <= endpoint_bound[..., None] + 1e-12).all(), "coordinate bound failure")
    support = endpoint_bound[1:6:2, 0] + endpoint_bound[0:6:2, 0] + rounding[:3, :2].max(axis=1)
    query = float(endpoint_bound[6].sum() + rounding[3, :2].max())
    worst_effect = float(support.sum())
    score_error = worst_effect + query + 2 * worst_effect * query
    return {"minimum_normalized_true_mass": float(p_true.min()),
            "maximum_coordinate_absolute_error": float(actual_error.max()),
            "maximum_endpoint_bound": float(endpoint_bound.max()),
            "maximum_f32_rounding_error": float(rounding.max()),
            "analytic_gap_lower_bound": float(1 - 2 * score_error - 1e-10)}


def visual_row(row, index):
    """Preserve sealed audit metadata, independently derive labels, then adapt."""
    require(set(row) == AUDIT_FIELDS | {"checkpointstage", "logits", "attention", "pooled", "frames", "public_metadata"}, "C15 row schema")
    audit = {key: row[key] for key in AUDIT_FIELDS}
    group, slot = divmod(index, 16)
    original, map_id = group * 4599 // 63, FIT[slot]
    expected = dict(schema="looped-grounded-policy-data-v1", cohort="seen", condition="factual",
                    support_cleared=False, row_index=index, group_index=group,
                    training_group_index=original, original_update=original // 4 + 1,
                    data_seed=20260920, episode_id=0x47524F554E445452 + original, permutation_id=map_id)
    require(all(exact(audit[k], v) for k, v in expected.items()) and row["checkpointstage"] == "frozen", "C15 registered identity/order")
    frames = row["frames"]
    require(type(frames) is list and len(frames) == 7 and all(set(f) == {"cells", "attention", "pooled"} for f in frames), "C15 seven frames")
    cells = [f["cells"] for f in frames]
    x, p, coordinates, rounding = soft_features([f["attention"] for f in frames], audit["observed_support_action_ids"])
    diagnostic = coordinate_audit(p, coordinates, cells, rounding)
    require(exact(cells[6], audit["query_cells"]) and exact(frames[6]["attention"], row["attention"]), "C15 current alias")
    query = cells[6]
    agent, goal = query.index(2), query.index(3)
    delta = GRID[goal] - GRID[agent]
    matches = np.all(DIRECTIONS == delta, axis=1)
    require(matches.sum() == 1, "cardinal distance-one query")
    direction = int(np.flatnonzero(matches)[0])
    label = MAPS[map_id].index(direction)
    actions = audit["observed_support_action_ids"]
    require(audit["correct_action"] == label and audit["inferred_controls"] == list(MAPS[map_id]) and
            audit["query_direction"] == direction and audit["agent_patch"] == agent and audit["goal_patch"] == goal and
            audit["correct_action_demonstrated"] is (label in actions), "C15 geometry/control label")
    for pair, action in enumerate(actions):
        before, after = cells[2 * pair], cells[2 * pair + 1]
        movement = GRID[after.index(2)] - GRID[before.index(2)]
        require(np.array_equal(movement, DIRECTIONS[MAPS[map_id][action]]), "C15 support movement/control")
    metadata = finite(row["public_metadata"], (4480,)).astype("<f4").tobytes()
    patches = np.repeat(np.asarray(cells, dtype="<u4"), 64, axis=1).tobytes()
    require(sha(patches + metadata) == audit["input_sha256"] == audit["factual_input_sha256"] and
            sha(patches[-16384:]) == audit["query_sha256"] and sha(metadata) == audit["metadata_sha256"] and
            sha(np.asarray([label], dtype="<u4").tobytes()) == audit["label_sha256"], "C15 public input/label hash")
    require(diagnostic["analytic_gap_lower_bound"] > 0, "cached soft-coordinate premise not justified")
    scores = analytic_scores(x)
    require(int(scores.argmax()) == label, "cached analytic positive control")
    return dict(index=index, audit=audit, features=x.tolist(), correct_action=label,
                demonstrated=label in actions, **identity(x, label)), diagnostic


def load_visual():
    require(file_sha(C15_SEAL) == C15_SEAL_SHA, "C15 outer seal hash")
    seal = decode(C15_SEAL.read_bytes())
    require(seal["source"] == C15_SOURCE and seal["campaign"] == str(C15_ROOT) and
            seal["classification"] == "completed_exploratory_grounding_evidence", "C15 source/evidence identity")
    entry = seal["files"]["frames-initial/evaluation-rows.jsonl"]
    require(entry == {"sha256": C15_ROWS_SHA, "bytes": 98604396}, "C15 selected rows binding")
    require(file_sha(C15_ROWS) == C15_ROWS_SHA and C15_ROWS.stat().st_size == entry["bytes"], "C15 row hash/size")
    rows, diagnostics = [], []
    with C15_ROWS.open("rb") as handle:
        for index, line in enumerate(handle):
            require(index < 1024 and 0 < len(line) <= 512 * 1024, "C15 row count/size")
            row, diagnostic = visual_row(decode(line), index)
            rows.append(row); diagnostics.append(diagnostic)
    require(len(rows) == 1024 and file_sha(C15_ROWS) == C15_ROWS_SHA, "C15 complete/stable rows")
    return rows, {"rows": 1024, "classification": "reused_visual_seam_evaluation_only",
                  "minimum_normalized_true_mass": min(d["minimum_normalized_true_mass"] for d in diagnostics),
                  "minimum_analytic_gap_lower_bound": min(d["analytic_gap_lower_bound"] for d in diagnostics),
                  "maximum_coordinate_absolute_error": max(d["maximum_coordinate_absolute_error"] for d in diagnostics),
                  "maximum_endpoint_bound": max(d["maximum_endpoint_bound"] for d in diagnostics),
                  "maximum_f32_rounding_error": max(d["maximum_f32_rounding_error"] for d in diagnostics),
                  "analytic_correct": 1024}


def dataset(visual=None, visual_audit=None):
    fit, held, updates = abstract_rows(FIT), abstract_rows(HELD), schedule()
    result = dict(schema=SCHEMA, fit=fit, heldout=held, updates=updates, cached_visual=visual or [],
                  provenance={"data_seed": SEED, "numpy_version": np.__version__, "source": C15_SOURCE,
                              "c15_seal": {"path": str(C15_SEAL), "sha256": C15_SEAL_SHA},
                              "c15_rows": {"path": str(C15_ROWS), "sha256": C15_ROWS_SHA}},
                  audit={"cached_visual": visual_audit, "abstract": {}})
    for name, rows in (("fit", fit), ("heldout", held)):
        margins = []
        for row in rows:
            scores = np.sort(analytic_scores(row["features"]))
            margins.append(float(scores[-1] - scores[-2]))
        result["audit"]["abstract"][name] = {
            "rows": len(rows), "groups": 96, "maps_per_group": len(rows) // 96,
            "label_counts": np.bincount([r["correct_action"] for r in rows], minlength=4).tolist(),
            "demonstrated_rows": sum(r["demonstrated"] for r in rows),
            "omitted_rows": sum(not r["demonstrated"] for r in rows),
            "constant_accuracy": .25, "cleared_effects_accuracy": .25,
            "query_zero_accuracy": .25,
            "query_zero_missing_id_fallback": {"overall_accuracy": .25, "omitted_accuracy": 1., "demonstrated_accuracy": 0.},
            "analytic_correct": len(rows), "analytic_minimum_gap": min(margins),
            "inputs_sha256": sha(b"".join(np.asarray(r["features"], dtype="<f4").tobytes() for r in rows)),
            "labels_sha256": sha(np.asarray([r["correct_action"] for r in rows], dtype="<u4").tobytes())}
        orbits = []
        exposure = np.bincount(np.asarray(updates).ravel(), minlength=1536) if name == "fit" else None
        for orbit in range(len(rows) // 6):
            members = [r["index"] for r in rows if r["orbit_id"] == orbit]
            canonical = [i for i in members if rows[i]["canonical"]]
            require(len(members) == 6 and len(canonical) == 1 and canonical[0] == min(members), "semantic orbit construction")
            orbits.append({"orbit_id": orbit, "canonical_index": canonical[0], "member_indices": members,
                           "training_presentations": int(exposure[members].sum()) if exposure is not None else 0})
        result["audit"]["abstract"][name].update(unique_semantic_orbits=len(orbits), order_variants_per_orbit=6,
            canonical_row_indices=[o["canonical_index"] for o in orbits], orbits=orbits)
    counts = np.bincount(np.asarray(updates).ravel(), minlength=1536)
    result["audit"]["schedule"] = {"updates": UPDATES, "batch": BATCH, "presentations": UPDATES * BATCH,
        "completed_epochs": 383, "partial_epoch_rows": 512, "minimum_presentations_per_row": int(counts.min()),
        "maximum_presentations_per_row": int(counts.max()), "rows_with_383_presentations": int((counts == 383).sum()),
        "rows_with_384_presentations": int((counts == 384).sum()),
        "indices_sha256": sha(np.asarray(updates, dtype="<u4").tobytes())}
    return result


def validate(value, visual=None, visual_audit=None):
    require(type(value) is dict and set(value) == {"schema", "fit", "heldout", "updates", "cached_visual", "provenance", "audit"}, "dataset schema")
    expected = dataset(visual, visual_audit)
    require(exact(value, expected), "dataset differs from deterministic registered reconstruction")
    for name, maps in (("fit", FIT), ("heldout", HELD)):
        rows = value[name]
        for start in range(0, len(rows), len(maps)):
            block = rows[start:start + len(maps)]
            cleared = []
            for row in block:
                require(set(row) == ROW_FIELDS, "abstract row schema")
                features = finite(row["features"], (4, 7))
                require(int(analytic_scores(features).argmax()) == row["correct_action"], "analytic label control")
                features[:3, :2] = 0
                cleared.append(features)
            require(all(np.array_equal(x, cleared[0]) for x in cleared), "cleared effects retain mapping")
            require(np.bincount([r["correct_action"] for r in block], minlength=4).tolist() == [len(maps) // 4] * 4,
                    "constant/clear group is not balanced")
        for triple in range(24):
            for slot in range(len(maps)):
                block = [rows[(triple * 4 + d) * len(maps) + slot] for d in range(4)]
                query_zero = np.asarray([r["features"] for r in block])
                query_zero[:, 3, :2] = 0
                require((query_zero == query_zero[0]).all() and sorted(r["correct_action"] for r in block) == list(range(4)),
                        "query-zero population is not balanced over identical inputs")
        for orbit in value["audit"]["abstract"][name]["orbits"]:
            ordered = []
            for i in orbit["member_indices"]:
                row = rows[i]
                features = np.asarray(row["features"])
                ordered.append(np.vstack((features[np.argsort(row["observed_actions"])], features[3])))
            require(all(np.array_equal(x, ordered[0]) for x in ordered), "orbit is not a support-order permutation")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(args.output.is_absolute() and args.output.parent == args.output.parent.resolve(strict=True) and
            not args.output.exists() and not args.output.is_symlink(), "new absolute output required")
    def timeout(_signum, _frame):
        raise TimeoutError("data preparation exceeded120seconds")
    signal.signal(signal.SIGALRM, timeout); signal.alarm(120)
    visual, visual_audit = load_visual()
    result = dataset(visual, visual_audit)
    validate(result, visual, visual_audit)
    with args.output.open("xb") as handle:
        handle.write(canonical_bytes(result))
    signal.alarm(0)


if __name__ == "__main__":
    main()
