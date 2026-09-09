#!/usr/bin/env python3
"""Independent C15 scalar scoring and weighted-bootstrap verification; no fitting.

No primary analyzer code is imported. Runtime/source/profile acceptance is shared
with the explicitly pinned external integrity receipt, not independently rerun.
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"
import argparse
from functools import lru_cache
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

SCHEMA = "looped-demonstration-grounding-analysis-v1"
FRAMES = ("before0", "after0", "before1", "after1", "before2", "after2", "current")
FIT = (0, 2, 4, 5, 7, 8, 9, 10, 13, 14, 15, 16, 18, 19, 21, 23)
PERMUTATIONS = tuple(itertools.permutations(range(4)))
DIRECTIONS = ((0, -1), (0, 1), (-1, 0), (1, 0))
AUDIT = set("schema cohort condition support_cleared row_index group_index training_group_index original_update data_seed episode_id permutation_id correct_action query_direction agent_patch goal_patch observed_support_action_ids omitted_action correct_action_demonstrated inferred_controls input_sha256 factual_input_sha256 cleared_input_sha256 metadata_sha256 query_sha256 targets_sha256 label_sha256 query_cells".split())
OUTPUT = {"logits", "attention", "pooled", "checkpointstage"}
CHECKPOINTS = {
    "initial": {"core_sha256": "4cd502f7a76fe3dd9729f6693e9d707c670972f7ebe58ad1cbdde37ea2bb2802", "head_sha256": "a43ad78fe7a6b1594d23c205fb191c8ac81610deaaff25198cb421d423edc678"},
    "final": {"core_sha256": "dc732f17a38f7bc6510dc6480a61dc4862fcb7c89ae1560f3a06515b8785d93f", "head_sha256": "37675528ef00055f16a17e826648078ff042492787087f3604d58cf0efc60fdf"},
}
DEPENDENCY = "1ea5cc5c5998a1c2bd388eb158653dd0d1b7452a"
HELPER = Path("/home/stepan/Research/_runs/2026-09-09T143629Z-tofy-looped-grounded-policy-learning/analyze.py")
HELPER_SHA = "1159f62d0cd260193907f45cb32eaf547388973ca5257f30318a59327a5b6a81"
CHECKS = {"source", "checkpoints", "zero_updates", "unchanged_parameters", "qualification", "profiles", "cleanup"}


class Invalid(ValueError):
    pass


def need(condition, message):
    if not condition:
        raise Invalid(message)


def same(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def keys(value, expected, where):
    need(type(value) is dict and set(value) == set(expected), f"{where}: exact keys required")


def decode(raw):
    def pairs(items):
        out = {}
        for k, v in items:
            need(k not in out, f"duplicate JSON key: {k}")
            out[k] = v
        return out
    def constant(value):
        raise Invalid(f"nonfinite JSON constant: {value}")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def regular(path):
    p = Path(path)
    need(p.is_absolute() and p == p.resolve(strict=True) and p.is_file(), "absolute regular nonsymlink file required")
    return p


def file_sha(path):
    p = regular(path)
    before = p.stat()
    with p.open("rb") as handle:
        result = hashlib.file_digest(handle, "sha256").hexdigest()
    after = p.stat()
    need((before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
         (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), "file changed while hashing")
    return result


def digest(value):
    need(type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value), "malformed SHA256")
    return value


def bound(record):
    keys(record, {"path", "sha256"}, "binding")
    need(file_sha(record["path"]) == digest(record["sha256"]), "bound file hash differs")
    return regular(record["path"])


def load(path):
    p = regular(path)
    need(p.stat().st_size <= 128 * 1024 * 1024, "JSON size limit")
    return decode(p.read_bytes())


def bindings(files):
    need(type(files) is dict and files, "empty frozen bindings")
    for path, value in files.items():
        bound({"path": path, "sha256": value})


def authority(config):
    keys(config, {"schema", "registration", "frozen_files", "integrity", "arms"}, "config")
    need(config["schema"] == SCHEMA, "config schema")
    bound(config["registration"])
    bindings(config["frozen_files"])
    scorer = Path(__file__).resolve().with_name("analysis.py")
    for path in (Path(__file__).resolve(), scorer, HELPER):
        need(config["frozen_files"].get(str(path)) == file_sha(path), "review/scorer/helper not frozen")
    need(file_sha(HELPER) == HELPER_SHA, "historical helper changed")
    receipt = load(bound(config["integrity"]))
    need(receipt.get("schema") == "looped-demonstration-grounding-integrity-v1" and receipt.get("accepted") is True,
         "external integrity not accepted")
    keys(receipt.get("checks"), CHECKS, "integrity checks")
    need(all(v is True for v in receipt["checks"].values()), "failed integrity check")
    need(type(receipt.get("source_revision")) is str and re.fullmatch(r"[a-f0-9]{40}", receipt["source_revision"]), "source revision")
    digest(receipt.get("binary_sha256"))
    need(receipt.get("dependency_revision") == DEPENDENCY and same(receipt.get("arms"), CHECKPOINTS), "source/checkpoint identity")
    bindings(receipt.get("frozen_files"))
    selected = dict(config["frozen_files"])
    records = [config["registration"]]
    keys(config["arms"], {"initial", "final"}, "arms")
    paths = []
    for arm in config["arms"].values():
        keys(arm, {"rows", "reference"}, "arm")
        for record in arm.values():
            paths.append(bound(record))
            records.append(record)
    need(len(set(paths)) == 4, "row/reference paths must be distinct")
    for record in records:
        p, h = record["path"], record["sha256"]
        need(p not in selected or selected[p] == h, "conflicting binding")
        selected[p] = h
    need(all(receipt["frozen_files"].get(p) == h for p, h in selected.items()), "receipt does not bind selected inputs")
    return receipt


def array(value, shape, probabilities=False):
    def numeric(v):
        return all(numeric(x) for x in v) if type(v) is list else type(v) in (int, float) and math.isfinite(v)
    need(numeric(value), "nonnumeric or nonfinite array")
    a = np.asarray(value, dtype=np.float64)
    need(a.shape == shape and np.isfinite(a).all() and (np.abs(a) <= np.finfo(np.float32).max).all(), "array shape/F32 range")
    if probabilities:
        need(((a >= 0) & (a <= 1)).all() and (np.abs(a.sum(axis=-1) - 1) <= 1e-5).all(), "invalid probabilities")
    return a


@lru_cache(maxsize=24)
def metadata(actions):
    floats = []
    for frame in range(7):
        kind = frame % 2 if frame < 6 else 2
        for cell in range(64):
            values = [0.0] * 10
            values[kind] = 1.0
            if frame < 6:
                values[3 + actions[frame // 2]] = 1.0
            values[7:10] = [cell % 8 / 7, cell // 8 / 7, frame // 2 / 3 if frame < 6 else 1.0]
            floats.extend(values)
    return struct.pack("<4480f", *floats)


def pixels(cells):
    return b"".join(struct.pack("<I", cell) * 64 for cell in cells)


def xy(cell):
    return cell % 8, cell // 8


def delta(before, after):
    bx, by = xy(before)
    ax, ay = xy(after)
    return ax - bx, ay - by


def public_truth(row, index):
    group, slot = divmod(index, 16)
    training = group * 4599 // 63
    expected = {"schema": "looped-grounded-policy-data-v1", "cohort": "seen", "condition": "factual", "support_cleared": False,
                "row_index": index, "group_index": group, "training_group_index": training, "original_update": training // 4 + 1,
                "data_seed": 20260920, "episode_id": 0x47524F554E445452 + training, "permutation_id": FIT[slot]}
    need(all(same(row[k], v) for k, v in expected.items()), "cohort/group/map/seed identity")
    for key in AUDIT:
        if key.endswith("_sha256"):
            digest(row[key])
    actions = row["observed_support_action_ids"]
    need(type(actions) is list and len(actions) == 3 and all(type(a) is int and 0 <= a < 4 for a in actions) and len(set(actions)) == 3,
         "three distinct public actions required")
    frames = row["frames"]
    need(type(frames) is list and len(frames) == 7, "frame count")
    truth = []
    for f, frame in enumerate(frames):
        keys(frame, {"cells", "attention", "pooled"}, "frame")
        cells = frame["cells"]
        need(type(cells) is list and len(cells) == 64 and all(type(c) is int and 0 <= c <= 3 for c in cells), "public cells")
        need(cells.count(2) == cells.count(3) == 1 and cells[0] not in (2, 3), "unique visible roles/nonrole cell0")
        if f == 6:
            need(all(cells[c] == 1 for c in range(64) if c % 8 in (0, 7) or c // 8 in (0, 7)), "query walls")
        else:
            need(1 not in cells and cells[63] == 3, "support walls/goal")
        truth.append((cells.index(2), cells.index(3)))
    need(same(frames[6]["cells"], row["query_cells"]), "query cells identity")
    agent, goal = truth[6]
    move = delta(agent, goal)
    need(move in DIRECTIONS, "query distance one")
    direction = DIRECTIONS.index(move)
    controls = PERMUTATIONS[FIT[slot]]
    label = controls.index(direction)
    meta = metadata(tuple(actions))
    query = pixels(frames[6]["cells"])
    derived = {"agent_patch": agent, "goal_patch": goal, "query_direction": direction, "correct_action": label,
               "omitted_action": next(a for a in range(4) if a not in actions), "correct_action_demonstrated": label in actions,
               "inferred_controls": list(controls), "label_sha256": sha(struct.pack("<I", label)),
               "query_sha256": sha(query), "metadata_sha256": sha(meta),
               "cleared_input_sha256": sha(bytes(6 * 4096 * 4) + query + meta)}
    need(all(same(row[k], v) for k, v in derived.items()), "visible query/label/metadata identity")
    x, y = xy(truth[0][0])
    need(2 <= x <= 5 and 2 <= y <= 5, "support start range")
    for pair, action in enumerate(actions):
        before, after = frames[pair * 2]["cells"], frames[pair * 2 + 1]["cells"]
        a, b = truth[pair * 2][0], truth[pair * 2 + 1][0]
        need(delta(a, b) == DIRECTIONS[controls[action]] and before[b] == 0, "support displacement/control")
        changed = list(before)
        changed[a], changed[b] = 0, 2
        need(same(after, changed), "support changed other cells")
        if pair:
            need(same(before, frames[pair * 2 - 1]["cells"]), "support continuity")
    observed = struct.pack("<4480f", *array(row["public_metadata"], (4480,)))
    need(observed == meta, "public metadata encoding")
    input_sha = sha(b"".join(pixels(f["cells"]) for f in frames) + meta)
    need(input_sha == row["input_sha256"] == row["factual_input_sha256"] != row["cleared_input_sha256"], "public input hash")
    return truth


def validate_row(row, reference, arm, index):
    keys(reference, AUDIT | OUTPUT, "reference row")
    keys(row, AUDIT | OUTPUT | {"frames", "public_metadata"}, "new row")
    need(all(same(row[k], reference[k]) for k in AUDIT), "C12 audit identity differs")
    need(row["checkpointstage"] == reference["checkpointstage"] == ("frozen" if arm == "initial" else "final"), "checkpoint stage")
    truth = public_truth(row, index)
    error = {}
    for key, shape in (("logits", (4,)), ("attention", (2, 64)), ("pooled", (256,))):
        a, b = array(row[key], shape, key == "attention"), array(reference[key], shape, key == "attention")
        need((np.abs(a - b) <= 1e-5 + 1e-5 * np.abs(b)).all(), "current numerical parity")
        if key != "pooled":
            need(np.array_equal(np.argmax(a, axis=-1), np.argmax(b, axis=-1)), "current winner parity")
        error[key] = float(np.max(np.abs(a - b)))
    attention = []
    for frame in row["frames"]:
        attention.append(array(frame["attention"], (2, 64), True))
        array(frame["pooled"], (256,))
    for key in ("attention", "pooled"):
        a, b = row["frames"][6][key], row[key]
        need(same(a, b) and np.asarray(a, dtype="<f4").tobytes() == np.asarray(b, dtype="<f4").tobytes(), "current frame alias")
    return truth, attention, int(max(range(4), key=lambda i: row["logits"][i]) == row["correct_action"]), error


def jsonl(path):
    need(regular(path).stat().st_size <= 512 * 1024 * 1024, "stream too large")
    with Path(path).open("rb") as handle:
        while line := handle.readline(512 * 1024 + 1):
            need(len(line) <= 512 * 1024 and line.strip(), "oversized/empty row")
            yield decode(line)


def collect(arm, records):
    truths, attentions, actions, audits = [], [], [], []
    parity = dict.fromkeys(("logits", "attention", "pooled"), 0.0)
    for i, (row, ref) in enumerate(itertools.zip_longest(jsonl(records["rows"]["path"]), jsonl(records["reference"]["path"]))):
        need(i < 1024 and row is not None and ref is not None, "row stream count")
        truth, attn, action, errors = validate_row(row, ref, arm, i)
        truths.append(truth); attentions.append(attn); actions.append(action)
        audits.append({k: row[k] for k in AUDIT})
        parity = {k: max(parity[k], errors[k]) for k in parity}
    need(len(audits) == 1024, "1024 rows required")
    for start in range(0, 1024, 16):
        block = audits[start:start + 16]
        for key in ("query_cells", "query_sha256", "metadata_sha256", "observed_support_action_ids", "cleared_input_sha256"):
            need(all(same(r[key], block[0][key]) for r in block), "unpaired query group")
        need(len({r["input_sha256"] for r in block}) == 16, "duplicate map input")
        need([sum(r["correct_action"] == a for r in block) for a in range(4)] == [4] * 4, "group label balance")
    return audits, np.asarray(truths), np.asarray(attentions), actions, parity


def score(truth, attention, actions):
    """Scalar argmax/coordinate implementation, separate from vectorized producer."""
    n = len(truth)
    need(np.shape(truth) == (n, 7, 2) and np.shape(attention) == (n, 7, 2, 64) and len(actions) == n, "metric input shape")
    values = {}
    for frame in FRAMES:
        for suffix in ("agent_accuracy", "agent_mass", "goal_accuracy", "goal_mass", "joint_accuracy"):
            values[f"{frame}/{suffix}"] = []
    for key in ("support/all_six_joint_accuracy", "support/pooled_displacement_accuracy", "support/all_three_displacements_accuracy", "current/action_accuracy"):
        values[key] = []
    for pair in range(3):
        for suffix in ("displacement_accuracy", "both_agent_accuracy"):
            values[f"pair{pair}/{suffix}"] = []
    for r in range(n):
        winners, correct = [], []
        for f, name in enumerate(FRAMES):
            picks, matches = [], []
            for role, label in enumerate(("agent", "goal")):
                probs = attention[r][f][role]
                selected = max(range(64), key=lambda c: probs[c])
                target = int(truth[r][f][role])
                picks.append(selected); matches.append(selected == target)
                values[f"{name}/{label}_accuracy"].append(int(selected == target))
                values[f"{name}/{label}_mass"].append(float(probs[target]))
            winners.append(picks); correct.append(matches)
            values[f"{name}/joint_accuracy"].append(int(all(matches)))
        values["support/all_six_joint_accuracy"].append(int(all(all(c) for c in correct[:6])))
        moves = []
        for p in range(3):
            a, b = 2 * p, 2 * p + 1
            ok = delta(winners[a][0], winners[b][0]) == delta(int(truth[r][a][0]), int(truth[r][b][0]))
            moves.append(ok)
            values[f"pair{p}/displacement_accuracy"].append(int(ok))
            values[f"pair{p}/both_agent_accuracy"].append(int(correct[a][0] and correct[b][0]))
        values["support/pooled_displacement_accuracy"].append(sum(moves) / 3)
        values["support/all_three_displacements_accuracy"].append(int(all(moves)))
        values["current/action_accuracy"].append(int(actions[r]))
    minima = {k: min(v) for k, v in values.items() if k.endswith("_mass")}
    counts = {k: sum(v) for k, v in values.items() if k.endswith("accuracy") and k != "support/pooled_displacement_accuracy"}
    counts["support/pooled_displacement_accuracy"] = sum(counts[f"pair{p}/displacement_accuracy"] for p in range(3))
    return {k: np.asarray(v, dtype=float) for k, v in values.items()}, minima, counts


def controls(truth):
    need(all(0 not in frame for row in truth for frame in row), "uniform control contains role at cell0")
    need(all(delta(int(row[2*p][0]), int(row[2*p+1][0])) in DIRECTIONS for row in truth for p in range(3)), "uniform displacement premise")
    return {"onehot": {"passed": True, "role_and_displacement_accuracy": 1, "role_mass": 1, "minimum_mass": 1.0},
            "uniform": {"passed": True, "role_and_displacement_accuracy": 0, "role_mass": 1/64, "minimum_mass": 1/64}}


def resampling():
    indices = np.random.Generator(np.random.PCG64(1944)).integers(0, 64, size=(10000, 64))
    weights = np.vstack([np.bincount(row, minlength=64) for row in indices]).astype(float) / 64
    metadata = {"seed": 1944, "draws": 10000, "groups": 64, "percentile": "linear", "draws_sha256": sha(indices.astype("<u8").tobytes())}
    return weights, metadata


def intervals(group_values, weights):
    names = list(group_values)
    matrix = np.column_stack([group_values[k] for k in names])
    samples = np.sort(weights @ matrix, axis=0)
    bounds = []
    for q in (.025, .975):
        position = (len(samples) - 1) * q
        lower = int(math.floor(position)); fraction = position - lower
        bounds.append(samples[lower] * (1 - fraction) + samples[min(lower + 1, len(samples) - 1)] * fraction)
    return {k: {"estimate": float(math.fsum(group_values[k]) / len(group_values[k])),
                "ci95": [float(bounds[0][i]), float(bounds[1][i])]} for i, k in enumerate(names)}


def reconstruct(truth, attention, actions, parity, weights):
    values, minima, counts = score(truth, attention, actions)
    need(len(truth) == 1024, "registered scoring population")
    groups = {k: np.asarray([math.fsum(v[g*16:(g+1)*16]) / 16 for g in range(64)]) for k, v in values.items()}
    endpoints = intervals(groups, weights)
    components = {}
    for frame in FRAMES[:6]:
        for role in ("agent", "goal"):
            key = f"{frame}/{role}_accuracy"
            components[key] = counts[key] / 1024 >= .99
        key = f"{frame}/joint_accuracy"
        components[key] = counts[key] / 1024 >= .98
    for p in range(3):
        key = f"pair{p}/displacement_accuracy"
        components[key] = counts[key] / 1024 >= .98
    summary = {"rows": 1024, "groups": 64, "maps_per_group": 16, "endpoints": endpoints,
               "minimum_mass": minima, "correct_counts": counts, "current_parity_max_absolute_error": parity, "controls": controls(truth)}
    return summary, groups, {"selector_reuse_supported": all(components.values()), "components": components}


def compare(actual, expected, path="report", statistics=None):
    if statistics is None:
        statistics = {"compared_float_fields": 0, "max_absolute_float_difference": 0.0, "compared_discrete_fields": 0}
    if type(expected) is dict:
        keys(actual, expected.keys(), path)
        for key in expected:
            compare(actual[key], expected[key], f"{path}/{key}", statistics)
    elif type(expected) is list:
        need(type(actual) is list and len(actual) == len(expected), f"{path}: list shape")
        for i, value in enumerate(expected):
            compare(actual[i], value, f"{path}/{i}", statistics)
    elif type(expected) is float:
        need(type(actual) in (int, float) and math.isfinite(actual), f"{path}: finite number required")
        error = abs(actual - expected)
        need(error <= 1e-12 + 1e-10 * abs(expected), f"{path}: numerical mismatch {actual} != {expected}")
        statistics["compared_float_fields"] += 1
        statistics["max_absolute_float_difference"] = max(statistics["max_absolute_float_difference"], error)
    else:
        need(same(actual, expected), f"{path}: exact discrete mismatch")
        statistics["compared_discrete_fields"] += 1
    return statistics


def review(config_path, report_path):
    started = time.monotonic()
    config_sha, report_sha = file_sha(config_path), file_sha(report_path)
    config = load(config_path)
    receipt = authority(config)  # Must complete before reading numerical streams.
    report = load(report_path)
    expected_header = {"schema": SCHEMA, "accepted": True, "classification": "exploratory_frozen_selector_reuse",
                       "config_sha256": config_sha, "registration_sha256": config["registration"]["sha256"],
                       "analysis_sha256": file_sha(Path(__file__).resolve().with_name("analysis.py")), "helper_sha256": HELPER_SHA,
                       "integrity_sha256": config["integrity"]["sha256"], "source_revision": receipt["source_revision"], "binary_sha256": receipt["binary_sha256"]}
    keys(report, set(expected_header) | {"summaries", "final_minus_initial", "gates", "bootstrap", "limits", "elapsed_seconds"}, "analysis report")
    compare({k: report[k] for k in expected_header}, expected_header)
    need(type(report["limits"]) is list and len(report["limits"]) == 6 and all(type(x) is str and x for x in report["limits"]), "report limits")
    need(type(report["elapsed_seconds"]) in (int, float) and math.isfinite(report["elapsed_seconds"]) and 0 <= report["elapsed_seconds"] < 120, "analysis duration")
    weights, boot = resampling()
    summaries, groups, gates = {}, {}, {}
    previous = None
    for arm in ("initial", "final"):
        audits, truth, attention, actions, parity = collect(arm, config["arms"][arm])
        if previous is not None:
            need(same(audits, previous[0]) and np.array_equal(truth, previous[1]), "unpaired initial/final inputs")
        previous = (audits, truth)
        summaries[arm], groups[arm], gates[arm] = reconstruct(truth, attention, actions, parity, weights)
    contrasts = intervals({k: groups["final"][k] - groups["initial"][k] for k in groups["initial"]}, weights)
    expected = {"summaries": summaries, "final_minus_initial": contrasts, "gates": gates, "bootstrap": boot}
    statistics = compare({k: report[k] for k in expected}, expected)
    authority(config)
    need(file_sha(config_path) == config_sha and file_sha(report_path) == report_sha, "config/report changed during review")
    need(time.monotonic() - started < 120, "review deadline exceeded")
    return {"schema": "looped-demonstration-grounding-independent-review-v1", "accepted": True,
            "report_sha256": report_sha, "config_sha256": config_sha, "reviewer_sha256": file_sha(Path(__file__).resolve()),
            **statistics, "verified_arms": ["initial", "final"], "rows_per_arm": 1024,
            "bootstrap": boot, "gates": gates, "elapsed_seconds": time.monotonic() - started,
            "scope": "Independent scalar role/location/displacement/input reconstruction, weighted paired bootstrap, counts and gates. No fitting or model inference. Runtime/source/checkpoint/profiling acceptance is shared with the pinned external receipt; target hashes inherit pinned C12 identities."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("config", "report", "output"):
        parser.add_argument("--" + flag, required=True, type=Path)
    args = parser.parse_args()
    need(args.output.is_absolute() and args.output.parent == args.output.parent.resolve(strict=True) and not args.output.exists() and not args.output.is_symlink(), "new absolute output required")
    def timeout(_signal, _frame):
        raise Invalid("review exceeded 120-second deadline")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(120)
    result = review(args.config, args.report)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    signal.alarm(0)


if __name__ == "__main__":
    main()
