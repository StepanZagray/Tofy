#!/usr/bin/env python3
"""C15 frozen frame scorer. Runtime provenance is certified externally.

CLI: analysis.py --config /absolute/config.json --output /absolute/NEW.json
Only strict JSON, C12 audit identity and metadata helpers are imported, by pinned
absolute file; no model, generator, fitting routine or C12 metric is invoked.
"""
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"
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

sys.dont_write_bytecode = True
import numpy as np

HELPER = Path("/home/stepan/Research/_runs/2026-09-09T143629Z-tofy-looped-grounded-policy-learning/analyze.py")
HELPER_SHA = "1159f62d0cd260193907f45cb32eaf547388973ca5257f30318a59327a5b6a81"
assert not HELPER.is_symlink() and hashlib.sha256(HELPER.read_bytes()).hexdigest() == HELPER_SHA
_spec = importlib.util.spec_from_file_location("c15_pinned_c12_identity", HELPER)
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)
require, keys, exact = H.require, H.keys, H.exact
SCHEMA = "looped-demonstration-grounding-analysis-v1"
DEPENDENCY = "1ea5cc5c5998a1c2bd388eb158653dd0d1b7452a"
CHECKPOINTS = {
    "initial": {"core_sha256": "4cd502f7a76fe3dd9729f6693e9d707c670972f7ebe58ad1cbdde37ea2bb2802",
                "head_sha256": "a43ad78fe7a6b1594d23c205fb191c8ac81610deaaff25198cb421d423edc678"},
    "final": {"core_sha256": "dc732f17a38f7bc6510dc6480a61dc4862fcb7c89ae1560f3a06515b8785d93f",
              "head_sha256": "37675528ef00055f16a17e826648078ff042492787087f3604d58cf0efc60fdf"},
}
CHECKS = {"source", "checkpoints", "zero_updates", "unchanged_parameters", "qualification", "profiles", "cleanup"}
FRAMES = ("before0", "after0", "before1", "after1", "before2", "after2", "current")
DIRECTIONS = np.asarray(((0, -1), (0, 1), (-1, 0), (1, 0)))
EXTRAS = {"frames", "public_metadata"}


def regular(path):
    path = Path(path)
    require(path.is_absolute() and path == path.resolve(strict=True) and path.is_file(),
            f"absolute regular nonsymlink file required: {path}")
    return path


def file_sha(path):
    path = regular(path)
    before = path.stat()
    with path.open("rb") as handle:
        result = hashlib.file_digest(handle, "sha256").hexdigest()
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), "file changed during hash")
    return result


def bound(record):
    keys(record, {"path", "sha256"}, "file binding")
    require(file_sha(record["path"]) == H.digest(record["sha256"]), f"hash mismatch: {record['path']}")
    return regular(record["path"])


def read_json(path):
    path = regular(path)
    require(path.stat().st_size <= 128 * 1024 * 1024, "oversized JSON")
    return H.decode(path.read_bytes())


def frozen(files):
    require(type(files) is dict and files, "nonempty frozen bindings required")
    for path, digest in files.items():
        bound({"path": path, "sha256": digest})


def validate_config(config):
    keys(config, {"schema", "registration", "frozen_files", "integrity", "arms"}, "config")
    require(config["schema"] == SCHEMA, "config schema")
    bound(config["registration"])
    frozen(config["frozen_files"])
    for path in (HELPER, Path(__file__).resolve()):
        require(config["frozen_files"].get(str(path)) == file_sha(path), "scorer/helper not frozen")
    receipt = read_json(bound(config["integrity"]))
    require(receipt.get("schema") == "looped-demonstration-grounding-integrity-v1"
            and receipt.get("accepted") is True, "integrity receipt not accepted")
    keys(receipt.get("checks"), CHECKS, "integrity checks")
    require(all(x is True for x in receipt["checks"].values()), "integrity check not passed")
    require(type(receipt.get("source_revision")) is str and
            re.fullmatch(r"[0-9a-f]{40}", receipt["source_revision"]), "source revision")
    H.digest(receipt.get("binary_sha256"))
    require(receipt.get("dependency_revision") == DEPENDENCY, "dependency drift")
    require(exact(receipt.get("arms"), CHECKPOINTS), "registered checkpoint/head mismatch")
    frozen(receipt.get("frozen_files"))
    bindings = dict(config["frozen_files"])
    keys(config["arms"], {"initial", "final"}, "arms")
    paths = []
    for arm in config["arms"].values():
        keys(arm, {"rows", "reference"}, "arm bindings")
        for record in arm.values():
            path = bound(record)
            paths.append(path)
            require(str(path) not in bindings or bindings[str(path)] == record["sha256"], "conflicting file binding")
            bindings[str(path)] = record["sha256"]
    bindings[config["registration"]["path"]] = config["registration"]["sha256"]
    require(len(set(paths)) == 4, "rows/reference paths must be distinct")
    require(all(receipt["frozen_files"].get(p) == h for p, h in bindings.items()),
            "receipt does not bind selected files/registration/scorer")
    return receipt


def jsonl(path):
    require(regular(path).stat().st_size <= 512 * 1024 * 1024, "oversized row stream")
    with Path(path).open("rb") as handle:
        while line := handle.readline(512 * 1024 + 1):
            require(len(line) <= 512 * 1024 and line.strip(), "oversized/blank JSONL row")
            yield H.decode(line)


def probabilities(value, shape):
    array = H.finite_array(value, shape, "attention")
    require(((array >= 0) & (array <= 1)).all(), "attention outside probability range")
    require((np.abs(array.sum(axis=-1) - 1) <= 1e-5).all(), "attention normalization")
    return array


def roles(cells, current):
    require(type(cells) is list and len(cells) == 64 and
            all(type(x) is int and 0 <= x <= 3 for x in cells), "invalid public cells")
    require(cells.count(2) == cells.count(3) == 1, "unique public roles required")
    if current:
        H.query_geometry(tuple(cells))
    else:
        require(1 not in cells and cells[63] == 3 and cells[0] == 0, "support topology/goal/non-role cell0")
    return (cells.index(2), cells.index(3))


def public_truth(row):
    frames = row["frames"]
    require(type(frames) is list and len(frames) == 7, "seven frames required")
    for frame in frames:
        keys(frame, {"cells", "attention", "pooled"}, "frame")
    truth = np.asarray([roles(f["cells"], i == 6) for i, f in enumerate(frames)])
    require(exact(frames[6]["cells"], row["query_cells"]), "current cells differ from audit")
    start = truth[0, 0]
    require(2 <= start // 8 <= 5 and 2 <= start % 8 <= 5, "initial calibration agent range")
    for pair, action in enumerate(row["observed_support_action_ids"]):
        before, after = frames[2 * pair]["cells"], frames[2 * pair + 1]["cells"]
        a, b = truth[2 * pair, 0], truth[2 * pair + 1, 0]
        delta = np.asarray((b % 8 - a % 8, b // 8 - a // 8))
        require(np.array_equal(delta, DIRECTIONS[row["inferred_controls"][action]]), "support control/displacement mismatch")
        expected = before.copy()
        require(expected[b] == 0, "support destination is not empty")
        expected[a], expected[b] = 0, 2
        require(exact(after, expected), "support transition changes non-agent pixels")
        if pair:
            require(exact(before, frames[2 * pair - 1]["cells"]), "support sequence discontinuity")
    metadata = H.finite_array(row["public_metadata"], (4480,), "public metadata").astype("<f4").tobytes()
    require(metadata == H.public_metadata(tuple(row["observed_support_action_ids"])), "public metadata bytes")
    patches = np.repeat(np.asarray([f["cells"] for f in frames], dtype="<u4"), 64, axis=1).tobytes()
    require(H.sha_bytes(patches + metadata) == row["input_sha256"] == row["factual_input_sha256"], "full public input hash")
    require(H.sha_bytes(patches[-4096 * 4:]) == row["query_sha256"] and
            H.sha_bytes(metadata) == row["metadata_sha256"], "public query/metadata hash")
    return truth


def validate_row(row, reference, arm, index):
    keys(reference, H.AUDIT_FIELDS | H.OUTPUT_FIELDS, "C12 reference row")
    keys(row, H.AUDIT_FIELDS | H.OUTPUT_FIELDS | EXTRAS, "C15 row")
    audit = {k: reference[k] for k in H.AUDIT_FIELDS}
    H.validate_identity(audit, "seen", "factual", index)
    require(exact({k: row[k] for k in H.AUDIT_FIELDS}, audit), "C12 audit identity/order differs")
    stage = "frozen" if arm == "initial" else "final"
    require(reference["checkpointstage"] == row["checkpointstage"] == stage, "checkpoint stage")
    errors = {}
    for key, shape in (("logits", (4,)), ("attention", (2, 64)), ("pooled", (256,))):
        parser = probabilities if key == "attention" else lambda v, s: H.finite_array(v, s, key)
        actual, old = parser(row[key], shape), parser(reference[key], shape)
        require((np.abs(actual - old) <= 1e-5 + 1e-5 * np.abs(old)).all(), f"current {key} parity")
        errors[key] = float(np.max(np.abs(actual - old)))
        if key != "pooled":
            require(np.array_equal(actual.argmax(axis=-1), old.argmax(axis=-1)), f"current {key} winner parity")
    truth = public_truth(row)
    attention = np.asarray([probabilities(f["attention"], (2, 64)) for f in row["frames"]])
    for frame in row["frames"]:
        H.finite_array(frame["pooled"], (256,), "frame pooled")
    for key in ("attention", "pooled"):
        a = np.asarray(row["frames"][6][key], dtype="<f4")
        b = np.asarray(row[key], dtype="<f4")
        require(a.tobytes() == b.tobytes() and exact(row["frames"][6][key], row[key]), "frame6 current alias differs")
    return audit, truth, attention, int(np.argmax(row["logits"]) == row["correct_action"]), errors


def collect(arm, records):
    audits, truths, attentions, actions = [], [], [], []
    maximum = {key: 0.0 for key in ("logits", "attention", "pooled")}
    pairs = itertools.zip_longest(jsonl(records["rows"]["path"]), jsonl(records["reference"]["path"]))
    for index, (row, reference) in enumerate(pairs):
        require(index < 1024 and row is not None and reference is not None, "stream row count differs")
        audit, truth, attention, action, errors = validate_row(row, reference, arm, index)
        audits.append(audit); truths.append(truth); attentions.append(attention); actions.append(action)
        maximum = {key: max(maximum[key], errors[key]) for key in maximum}
    require(len(audits) == 1024, "1024 rows required")
    for start in range(0, 1024, 16):
        block = audits[start:start + 16]
        for key in ("query_cells", "query_sha256", "metadata_sha256", "observed_support_action_ids", "cleared_input_sha256"):
            require(all(exact(row[key], block[0][key]) for row in block), "unpaired within-group public query")
        require(len({r["input_sha256"] for r in block}) == 16, "duplicate mapping input")
        require(np.bincount([r["correct_action"] for r in block], minlength=4).tolist() == [4] * 4, "group label balance")
    return audits, np.asarray(truths), np.asarray(attentions), np.asarray(actions), maximum


def measurements(attention, truth, action_correct):
    n = len(truth)
    require(attention.shape == (n, 7, 2, 64) and truth.shape == (n, 7, 2), "metric input shape")
    predicted = attention.argmax(axis=-1)  # NumPy selects the first index on ties.
    correct = predicted == truth
    mass = np.take_along_axis(attention, truth[..., None], axis=-1)[..., 0]
    xy = lambda x: np.stack((x % 8, x // 8), axis=-1)
    true_delta = xy(truth[:, 1:6:2, 0]) - xy(truth[:, 0:6:2, 0])
    pred_delta = xy(predicted[:, 1:6:2, 0]) - xy(predicted[:, 0:6:2, 0])
    displacement = (pred_delta == true_delta).all(axis=-1)
    values, minima = {}, {}
    for frame, name in enumerate(FRAMES):
        for role, label in enumerate(("agent", "goal")):
            values[f"{name}/{label}_accuracy"] = correct[:, frame, role].astype(float)
            values[f"{name}/{label}_mass"] = mass[:, frame, role]
            minima[f"{name}/{label}_mass"] = float(mass[:, frame, role].min())
        values[f"{name}/joint_accuracy"] = correct[:, frame].all(axis=1).astype(float)
    values["support/all_six_joint_accuracy"] = correct[:, :6].all(axis=(1, 2)).astype(float)
    for pair in range(3):
        values[f"pair{pair}/displacement_accuracy"] = displacement[:, pair].astype(float)
        values[f"pair{pair}/both_agent_accuracy"] = correct[:, 2 * pair:2 * pair + 2, 0].all(axis=1).astype(float)
    values["support/pooled_displacement_accuracy"] = displacement.mean(axis=1)
    values["support/all_three_displacements_accuracy"] = displacement.all(axis=1).astype(float)
    values["current/action_accuracy"] = np.asarray(action_correct, dtype=float)
    counts = {key: int(np.rint(value.sum() * (3 if key == "support/pooled_displacement_accuracy" else 1)))
              for key, value in values.items() if key.endswith("accuracy")}
    return values, minima, counts


def controls(truth):
    positive = np.zeros((*truth.shape, 64), dtype=float)
    np.put_along_axis(positive, truth[..., None], 1, axis=-1)
    result = {}
    for name, attention, expected_accuracy, expected_mass in (
            ("onehot", positive, 1, 1), ("uniform", np.full_like(positive, 1 / 64), 0, 1 / 64)):
        values, minima, _ = measurements(attention, truth, np.zeros(len(truth)))
        for key, values_row in values.items():
            if key != "current/action_accuracy":
                expected = expected_accuracy if key.endswith("accuracy") else expected_mass
                require((values_row == expected).all(), f"{name} analytic scorer control failed: {key}")
        result[name] = {"passed": True, "role_and_displacement_accuracy": expected_accuracy,
                        "role_mass": expected_mass, "minimum_mass": min(minima.values())}
    return result


def grouped(values):
    return {key: value.reshape(64, 16).mean(axis=1) for key, value in values.items()}


def bootstrap(values, draws):
    return {key: {"estimate": float(value.mean()),
                  "ci95": np.quantile(value[draws].mean(axis=1), [.025, .975], method="linear").tolist()}
            for key, value in values.items()}


def gate(endpoints):
    components = {}
    for frame in FRAMES[:6]:
        for role in ("agent", "goal"):
            key = f"{frame}/{role}_accuracy"
            components[key] = endpoints[key]["estimate"] >= .99
        key = f"{frame}/joint_accuracy"
        components[key] = endpoints[key]["estimate"] >= .98
    for pair in range(3):
        key = f"pair{pair}/displacement_accuracy"
        components[key] = endpoints[key]["estimate"] >= .98
    return {"selector_reuse_supported": all(components.values()), "components": components}


def analyze(config_path):
    start = time.monotonic()
    config_sha = file_sha(config_path)
    config = read_json(config_path)
    receipt = validate_config(config)  # Lifecycle acceptance precedes numerical reads.
    draws = np.random.Generator(np.random.PCG64(1944)).integers(0, 64, (10000, 64))
    summaries, groups, audit_reference, truths = {}, {}, None, None
    for arm in ("initial", "final"):
        audits, truth, attention, actions, parity = collect(arm, config["arms"][arm])
        if audit_reference is not None:
            require(exact(audits, audit_reference) and np.array_equal(truth, truths), "unpaired initial/final public inputs")
        audit_reference, truths = audits, truth
        values, minima, counts = measurements(attention, truth, actions)
        groups[arm] = grouped(values)
        summaries[arm] = {"rows": 1024, "groups": 64, "maps_per_group": 16,
                          "endpoints": bootstrap(groups[arm], draws), "minimum_mass": minima,
                          "correct_counts": counts, "current_parity_max_absolute_error": parity,
                          "controls": controls(truth)}
    differences = bootstrap({k: groups["final"][k] - groups["initial"][k] for k in groups["initial"]}, draws)
    gates = {arm: gate(summary["endpoints"]) for arm, summary in summaries.items()}
    # Rehash selected inputs and pinned authority after reading; no stale acceptance.
    validate_config(config)
    require(file_sha(config_path) == config_sha, "analysis config changed")
    require(time.monotonic() - start < 120, "scoring exceeded 120-second bound")
    return {"schema": SCHEMA, "accepted": True, "classification": "exploratory_frozen_selector_reuse",
            "config_sha256": config_sha, "registration_sha256": config["registration"]["sha256"],
            "analysis_sha256": file_sha(Path(__file__).resolve()), "helper_sha256": HELPER_SHA,
            "integrity_sha256": config["integrity"]["sha256"], "source_revision": receipt["source_revision"],
            "binary_sha256": receipt["binary_sha256"], "summaries": summaries,
            "final_minus_initial": differences, "gates": gates,
            "bootstrap": {"seed": 1944, "draws": 10000, "groups": 64, "percentile": "linear",
                          "draws_sha256": H.sha_bytes(draws.astype("<u8").tobytes())},
            "limits": ["Reused training queries and fixed checkpoints/maps; pointwise intervals, not population certainty.",
                       "Location and displacement diagnostics do not establish learned action binding or a policy.",
                       "Initial/final compare complete core/head pairs; initial privilege is not training credit.",
                       "Runtime source/checkpoint/qualification/profiler/cleanup provenance is shared with the pinned external receipt.",
                       "Targets hashes are inherited from exact pinned C12 identities, not a new simulator reconstruction.",
                       "No automatic training, method promotion, universal information-absence or ARC claim."],
            "elapsed_seconds": time.monotonic() - start}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(args.output.is_absolute() and args.output.parent.resolve() == args.output.parent
            and args.output.parent.is_dir() and not args.output.exists() and not args.output.is_symlink(), "new absolute output required")
    def timeout(_signal, _frame):
        raise H.Invalid("analysis deadline exceeded")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(120)
    result = analyze(args.config)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    signal.alarm(0)


if __name__ == "__main__":
    main()
