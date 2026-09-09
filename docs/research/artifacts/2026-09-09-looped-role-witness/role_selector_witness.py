#!/usr/bin/env python3
"""C10 selection-only CPU witness. Importing this module never opens real inputs."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
from datetime import datetime

# Set before NumPy import; runtime validation also checks the loaded BLAS library.
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_key] = "1"
import numpy as np

HERE = Path(__file__).resolve().parent
REG_SHA = "411afa3b56fb71f5cb78494b9b91911cec19313080d98ff7474b52b129568dfb"
C8 = Path("/home/stepan/Projects/code/.tofy-runs/looped-frozen-features-20260909T115455-IST")
C9 = Path("/home/stepan/Projects/code/.tofy-runs/looped-learned-readout-20260909T124221-IST")
R8 = Path("/home/stepan/Research/_runs/2026-09-09T104547Z-tofy-looped-frozen-features")
R9 = Path("/home/stepan/Research/_runs/2026-09-09T113612Z-tofy-looped-learned-readout")
C8_SEAL = "ebae6f94562879ed343895dc09e9243ab616900f5f8f5e64b54f7ac408cf9e60"
C9_SEAL = "ea0fb1e773400363e52e3b6966600fde5d8e0ffe37a14547e49772d2186fe477"
C8_SOURCE = "2f2aaa711eb8f39eb823cc4e354288c7b4fbcf42"
C9_SOURCE = "06a8d76fada203c5b1a11a45782ed777df27ff4c"
C8_BINARY = "83150acaf7275bab63204d897972194422b2eca01137c9b7752ade456105ea8e"
C9_BINARY = "fecbc1b91099ddc9b39ac9b6361ee8072c26563f6b710e06c29ec7b3adeef20e"
C9_EXTRACTOR = "220daa93fecf43fa66ed681577fef81efb40bc058c1de39d52ed4a7afa87b728"
CHECKPOINTS = {
    "initial": "4cd502f7a76fe3dd9729f6693e9d707c670972f7ebe58ad1cbdde37ea2bb2802",
    "final": "a983626a888279cbb31651c6abcd1ebf4f808c8927a29a8349d03ffc35bd4e5a",
}
FIT_CACHE_SEALS = {
    "initial": "8dc6d855aed1fa4c8ea6f2c47ded63802fc3a09b747ca28a79de5807c4ce391a",
    "final": "89a9a6c103f7b6c439567f7291863a737f2abf3150ac1fffa79fca0ced9ca617",
}
FIT_TAG, EVAL_TAG = 0x46454154555245, 0x524541444F5554
RIDGE, EPSILON, MARGIN_MIN = 0.01, 1e-6, 1e-8
CORES, ARMS = ("initial", "final"), ("true", "null")


def require(condition, message):
    if not condition:
        raise ValueError(message)


class NumericalControlFailure(ValueError):
    pass


def numerical_require(condition, message):
    if not condition:
        raise NumericalControlFailure(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value, name):
    value = np.asarray(value)
    numerical_require(value.dtype.kind in "bifu" and np.isfinite(value).all(), f"nonfinite/malformed {name}")
    return value


def read_json(path):
    with Path(path).open() as file:
        return json.load(file)


def write_json(path, value):
    with Path(path).open("x") as file:
        json.dump(value, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())


def save_arrays(path, **arrays):
    arrays = {name: finite(value, name) for name, value in arrays.items()}
    with Path(path).open("xb") as file:
        np.savez(file, **arrays)
        file.flush()
        os.fsync(file.fileno())


def local_now():
    return datetime.now().astimezone().isoformat()


def role_targets(cells):
    cells = np.asarray(cells)
    require(cells.ndim == 2 and cells.shape[1] == 64 and cells.dtype.kind in "iu", "visible cells must be integer [N,64]")
    require(((cells >= 0) & (cells <= 3)).all(), "invalid nonterminal toy color")
    require(((cells == 2).sum(1) == 1).all() and ((cells == 3).sum(1) == 1).all(), "agent/goal must each occur exactly once")
    return np.stack([(cells == color).argmax(1) for color in (2, 3)], axis=1)


def geometry_labels(targets):
    delta = np.stack([targets[:, 1] % 8 - targets[:, 0] % 8,
                      targets[:, 1] // 8 - targets[:, 0] // 8], axis=1)
    require((np.abs(delta).sum(1) == 1).all(), "roles are not adjacent")
    return np.stack([-delta[:, 1], delta[:, 1], -delta[:, 0], delta[:, 0]], axis=1).argmax(1)


def validate_targets(targets, n, tokens):
    targets = np.asarray(targets)
    require(targets.shape == (n, 2) and targets.dtype.kind in "iu", "role target shape/type")
    require(((targets >= 0) & (targets < tokens)).all(), "role target out of range")
    require((targets[:, 0] != targets[:, 1]).all(), "two roles cannot occupy one cell")
    return targets.astype(np.int64)


def margins(scores, targets):
    """Return all 63 competitors in ascending token order, excluding the target."""
    n, tokens, roles = scores.shape
    targets = validate_targets(targets, n, tokens)
    values = scores[np.arange(n)[:, None], targets, np.arange(roles)]
    differences = values[:, None, :] - scores
    mask = np.arange(tokens)[None, :, None] != targets[:, None, :]
    return differences.transpose(0, 2, 1)[mask.transpose(0, 2, 1)].reshape(n, roles, tokens - 1)


def fit_selector(features, targets):
    h = finite(features, "fit features").astype(np.float64)
    require(h.ndim == 3 and h.shape[1] >= 2, "fit feature shape")
    n, tokens, width = h.shape
    targets = validate_targets(targets, n, tokens)
    x = h.reshape(n * tokens, width)
    y = np.zeros((n, tokens, 2), dtype=np.float64)
    y[np.arange(n)[:, None], targets, np.arange(2)] = 1.0
    y = y.reshape(n * tokens, 2)
    mean, std = x.mean(0), x.std(0, ddof=0)
    scale = np.maximum(std, 1e-6)
    z = (x - mean) / scale
    intercept = y.mean(0)
    matrix = z.T @ z / len(z) + RIDGE * np.eye(width)
    rhs = z.T @ (y - intercept) / len(z)
    weights = np.linalg.solve(matrix, rhs)
    residual = float(np.linalg.norm(matrix @ weights - rhs) / max(np.linalg.norm(rhs), np.finfo(np.float64).tiny))
    numerical_require(residual <= 1e-10, "ridge normal-equation residual exceeded")
    raw_weights = weights / scale[:, None]
    raw_bias = intercept - mean @ raw_weights
    standardized = z @ weights + intercept
    raw = x @ raw_weights + raw_bias
    error = float(np.max(np.abs(standardized - raw)))
    tolerance = 1e-9 * (1.0 + float(np.max(np.abs(standardized))))
    numerical_require(error <= tolerance, "raw/standardized role scores disagree")
    scores = raw.reshape(n, tokens, 2)
    min_margins = margins(scores, targets).min(axis=(0, 2))
    eligible = min_margins > MARGIN_MIN
    alpha = np.ones(2)
    alpha[eligible] = np.log((tokens - 1) * (1 - EPSILON) / EPSILON) / min_margins[eligible]
    arrays = dict(mean=mean, population_std=std, scale=scale, weights=weights,
                  intercept=intercept, raw_weights=raw_weights, raw_bias=raw_bias,
                  alpha=alpha, eligible=eligible, min_fitted_margins=min_margins,
                  queries=np.sqrt(width) * (raw_weights * alpha).T)
    for name, value in arrays.items():
        finite(value, name)
    info = {"lambda": RIDGE, "token_rows": len(x), "width": width,
            "normal_equation_relative_residual": residual,
            "raw_score_max_error": error, "raw_score_tolerance": tolerance,
            "clamped_dimensions": int((std < 1e-6).sum()),
            "objective": float(np.square(standardized-y).sum()/(2*len(x)) + RIDGE*np.square(weights).sum()/2),
            "minimum_fitted_margins": min_margins.tolist(), "alpha": alpha.tolist(),
            "mass_guarantee_eligible": eligible.tolist()}
    return arrays, info


def softmax(logits):
    logits = finite(logits, "softmax logits")
    exponent = np.exp(logits - logits.max(axis=1, keepdims=True))
    return finite(exponent / exponent.sum(axis=1, keepdims=True), "softmax weights")


def apply_selector(features, selector):
    """No target/label argument: both learned pools are computed feature-only."""
    h = finite(features, "inference features").astype(np.float64)
    scores = finite(h @ selector["raw_weights"] + selector["raw_bias"], "role scores64")
    standardized = (h-selector["mean"])/selector["scale"] @ selector["weights"] + selector["intercept"]
    error = float(np.max(np.abs(scores-standardized)))
    numerical_require(error <= 1e-9*(1+float(np.max(np.abs(standardized)))), "inference raw-score reconstruction failed")
    # Role bias is constant within a query and cancels from softmax.
    logits64 = finite((h @ selector["raw_weights"]) * selector["alpha"], "role logits64")
    queries32 = finite(selector["queries"].astype(np.float32), "queries32")
    h32 = finite(h.astype(np.float32), "features32")
    raw_weights32 = finite(selector["raw_weights"].astype(np.float32), "raw_weights32")
    raw_bias32 = finite(selector["raw_bias"].astype(np.float32), "raw_bias32")
    scores32 = finite(h32 @ raw_weights32 + raw_bias32, "role raw scores32")
    logits32 = finite((h32 @ queries32.T) / np.float32(np.sqrt(h.shape[2])), "C9 reference logits32")
    weights64, weights32 = softmax(logits64), softmax(logits32)
    positions = scores.argmax(axis=1)
    hard = h[np.arange(len(h))[:, None], positions, :].reshape(len(h), -1)
    soft = np.einsum("ntr,ntd->nrd", weights64, h).reshape(len(h), -1)
    return dict(scores=scores, scores32=scores32, logits64=logits64, logits32=logits32,
                weights64=weights64, weights32=weights32, positions=positions,
                hard=hard, soft=finite(soft, "soft pooled features"),
                argmax_ties=(scores == scores.max(1, keepdims=True)).sum(1)-1), error


def validate_policy(policy, width=256):
    for name, shape in [("mean", (width,)), ("scale", (width,)),
                        ("coefficients", (width, 4)), ("label_mean", (4,))]:
        value = finite(policy[name], f"C8 policy {name}")
        require(value.shape == shape, f"C8 policy {name} shape")
    require((np.asarray(policy["scale"]) > 0).all(), "C8 policy scale must be positive")


def policy_scores(pools, policy):
    return finite(((pools-np.asarray(policy["mean"]))/np.asarray(policy["scale"])) @
                  np.asarray(policy["coefficients"]) + np.asarray(policy["label_mean"]), "policy scores")


def mass_checks(result, targets, eligible):
    weights64 = finite(result["weights64"], "mass weights64")
    weights32 = finite(result["weights32"], "mass weights32")
    require(weights64.ndim == 3 and weights64.shape[-1] == 2 and
            weights32.shape == weights64.shape, "mass weight shape")
    targets = validate_targets(targets, weights64.shape[0], weights64.shape[1])
    eligible = np.asarray(eligible)
    require(eligible.dtype.kind == "b" and eligible.shape == (2,), "mass eligibility must be two booleans")
    require(((weights64 >= 0) & (weights64 <= 1)).all() and
            ((weights32 >= 0) & (weights32 <= 1)).all(), "invalid attention mass")
    n = len(targets)
    masses64 = weights64[np.arange(n)[:, None], targets, np.arange(2)]
    masses32 = weights32[np.arange(n)[:, None], targets, np.arange(2)].astype(np.float64)
    strict = 1-EPSILON
    failed64 = eligible & (masses64.min(0) < strict-1e-12)
    failed32 = eligible & (masses32.min(0) < strict-2*np.finfo(np.float32).eps)
    info = {"eligible": eligible.tolist(), "minimum_mass64": masses64.min(0).tolist(),
            "minimum_mass32": masses32.min(0).tolist(),
            "strict_mass_failures64": ((masses64 < strict) & eligible).sum(0).tolist(),
            "strict_mass_failures32": ((masses32 < strict) & eligible).sum(0).tolist(),
            "tolerated_failure64": failed64.tolist(), "tolerated_failure32": failed32.tolist(),
            "passed": not bool(failed64.any() or failed32.any())}
    return info, masses64, masses32


def localization(result, targets):
    correct = result["positions"] == targets
    all_margins = margins(result["scores"], targets)
    return {"rows": len(targets), "agent_correct": int(correct[:, 0].sum()),
            "goal_correct": int(correct[:, 1].sum()), "joint_correct": int(correct.all(1).sum()),
            "minimum_margins": all_margins.min(axis=(0, 2)).tolist(),
            "target_tie_counts": (all_margins == 0).sum(axis=(0, 2)).tolist(),
            "argmax_tie_counts": result["argmax_ties"].sum(0).tolist(),
            "predicted_role_position_counts": [np.bincount(result["positions"][:, r], minlength=result["scores"].shape[1]).tolist()
                for r in range(2)]}, all_margins


def bootstrap_indices(n=256):
    return np.random.Generator(np.random.PCG64(1918)).integers(0, n, size=(10000, n), dtype=np.int64)


def accuracy_interval(correct, indices):
    correct = np.asarray(correct, dtype=bool)
    values = correct[indices].mean(1)
    return {"correct": int(correct.sum()), "rows": len(correct), "accuracy": float(correct.mean()),
            "ci95": np.quantile(values, [0.025, 0.975], method="linear").tolist()}, values


def evaluation_metrics(result, roles, actions, policy, indices):
    correct = result["positions"] == roles
    metrics = {}
    for name, values in [("agent", correct[:, 0]), ("goal", correct[:, 1]), ("joint", correct.all(1))]:
        metrics[name], _ = accuracy_interval(values, indices)
    constant = np.stack([(actions[indices] == a).mean(1) for a in range(4)]).max(0)
    best = float(np.bincount(actions, minlength=4).max()/len(actions))
    arrays = {}
    for name in ("hard", "soft"):
        scores = policy_scores(result[name], policy)
        prediction = scores.argmax(1)
        metric, draws = accuracy_interval(prediction == actions, indices)
        metric["advantage_over_resampled_best_constant"] = {
            "estimate": metric["accuracy"]-best,
            "ci95": np.quantile(draws-constant, [0.025, 0.975], method="linear").tolist()}
        metrics[name] = metric
        arrays[f"{name}_policy_scores"] = scores
        arrays[f"{name}_policy_predictions"] = prediction
    return metrics, arrays


def classify(arms, controls):
    require(set(controls) == {"synthetic", "fit_mass", "oracle_initial", "oracle_final"} and
            all(type(value) is bool for value in controls.values()), "missing/malformed mandatory controls")
    require(set(arms) == {core+"/"+arm for core in CORES for arm in ARMS}, "missing evaluation arm")
    for arm in arms.values():
        for name in ("agent", "goal", "joint", "hard", "soft"):
            accuracy = arm[name]["accuracy"]
            require(type(accuracy) in (int, float) and np.isfinite(accuracy) and 0 <= accuracy <= 1,
                    "invalid accuracy in decision")
        for name in ("hard", "soft"):
            interval = finite(arm[name]["advantage_over_resampled_best_constant"]["ci95"], "policy advantage CI")
            require(interval.shape == (2,) and -1 <= interval[0] <= interval[1] <= 1, "invalid policy CI")
    null_pass = all(arms[f"{core}/null"]["joint"]["accuracy"] <= .10 and
                    arms[f"{core}/null"][name]["accuracy"] <= .50
                    for core in CORES for name in ("hard", "soft"))
    valid = all(controls.values()) and null_pass
    decisions = {}
    for core in CORES:
        arm = arms[f"{core}/true"]
        numerical = (arm["joint"]["accuracy"] >= .95 and
                     all(arm[name]["accuracy"] >= .90 and
                         arm[name]["advantage_over_resampled_best_constant"]["ci95"][0] > .25
                         for name in ("hard", "soft")))
        decisions[core] = {"criteria_met": numerical, "routing_supported": bool(valid and numerical),
            "status": "inconclusive_failed_control" if not valid else
                           "selection_only_support" if numerical else "selection_only_not_supported"}
    return {"controls_valid": valid, "null_controls_passed": null_pass, "cores": decisions,
            "method_promotion": False, "automatic_training": False}


def synthetic_controls():
    # Known role coordinates are embedded in artificial features, not parsed by inference.
    rng = np.random.Generator(np.random.PCG64(901))
    targets = np.stack([rng.permutation(64)[:2] for _ in range(48)])
    h = rng.normal(0, .1, (48, 64, 6))
    h[:, :, :2] = 0
    h[np.arange(48)[:, None], targets, np.arange(2)] = 1
    selector, fit = fit_selector(h, targets)
    output, _ = apply_selector(h, selector)
    require(np.array_equal(output["positions"], targets), "synthetic selector failed")
    require((selector["min_fitted_margins"] > MARGIN_MIN).all(), "synthetic margins failed")
    mass, _, _ = mass_checks(output, targets, selector["eligible"])
    require(mass["passed"], "synthetic mass premise failed")
    permutation = rng.permutation(64)
    inverse = np.argsort(permutation)
    moved, _ = apply_selector(h[:, permutation], selector)
    require(np.allclose(output["scores"][:, permutation], moved["scores"], atol=1e-12, rtol=1e-12), "score equivariance failed")
    require(np.allclose(output["weights64"][:, permutation], moved["weights64"], atol=1e-12, rtol=1e-12), "weight equivariance failed")
    require(np.array_equal(moved["positions"], inverse[targets]), "unique-argmax equivariance failed")
    zero, zero_fit = fit_selector(np.ones((48, 64, 6)), targets)
    zero_out, _ = apply_selector(np.ones((48, 64, 6)), zero)
    require(np.all(zero_out["positions"] == 0) and np.all(zero_out["argmax_ties"] == 63), "identical-token tie control failed")
    require(not zero["eligible"].any() and np.all(zero["alpha"] == 1), "identical-token mass eligibility failed")
    return {"known_separator": True, "identical_tokens": True, "joint_permutation": True,
            "normal_equation": True, "raw_score": True,
            "positive_residual": fit["normal_equation_relative_residual"],
            "identical_residual": zero_fit["normal_equation_relative_residual"]}


class Outer:
    """The pinned inventory can be checked before fitting without reading eval data."""
    def __init__(self, root, manifest_path, expected):
        self.root, self.path, self.expected = Path(root), Path(manifest_path), expected
        require(self.root.is_absolute() and self.path.is_absolute(), "parent paths must be absolute")
        require(self.path.is_file() and not self.path.is_symlink(), "parent manifest must be regular")
        require(file_sha(self.path) == expected, "parent external seal mismatch")
        self.document = read_json(self.path)
        require(datetime.fromisoformat(self.document["created_local"]).utcoffset() is not None,
                "parent seal timestamp must be timezone-aware")
        require(self.document["campaign"] == str(self.root), "parent campaign path mismatch")
        self.files = self.document["files"]
        require(isinstance(self.files, dict) and self.files, "empty parent inventory")
        for name, entry in self.files.items():
            path = Path(name)
            require(not path.is_absolute() and ".." not in path.parts, "unsafe parent inventory path")
            require(isinstance(entry["bytes"], int) and entry["bytes"] >= 0 and
                    len(entry["sha256"]) == 64, "malformed parent inventory")
        self.check_inventory()

    def check_inventory(self):
        actual = set()
        require(not self.root.is_symlink(), "parent root is a symlink")
        for directory, dirs, files in os.walk(self.root, followlinks=False):
            for name in dirs + files:
                path = Path(directory)/name
                require(not path.is_symlink(), "symlink in sealed parent")
            for name in files:
                path = Path(directory)/name
                require(path.is_file(), "nonregular parent file")
                actual.add(str(path.relative_to(self.root)))
        require(actual == set(self.files), "parent inventory/tree differs")
        for name, entry in self.files.items():
            require((self.root/name).stat().st_size == entry["bytes"], f"parent length changed: {name}")

    def verified_file(self, name):
        require(name in self.files, f"unsealed input: {name}")
        path = self.root/name
        require(file_sha(path) == self.files[name]["sha256"], f"parent artifact mismatch: {name}")
        return path

    def verify_all(self, barrier):
        barrier.require_sealed()
        require(file_sha(self.path) == self.expected, "external parent manifest changed")
        self.check_inventory()
        for name in self.files:
            self.verified_file(name)


class Barrier:
    def __init__(self, output):
        self.output = Path(output)
        self.path = self.output.with_suffix(".fit-seal.json")
        self.digest = None
        self.events = []
        self.started = time.monotonic()

    def event(self, name, **extra):
        self.events.append({"event": name, "created_local": local_now(),
                            "elapsed_seconds": time.monotonic()-self.started, **extra})

    def seal(self, names):
        require(self.digest is None, "fit barrier already sealed")
        require(set(names) == {f"selector-{core}-{arm}.npz" for core in CORES for arm in ARMS} |
                {f"policy-{core}.json" for core in CORES}, "incomplete four-arm freeze")
        document = {"schema": "c10-four-fit-freeze-v1", "created_local": local_now(),
                    "root": str(self.output), "files": {name: file_sha(self.output/name) for name in names},
                    "registration_sha256": REG_SHA, "events": self.events.copy()}
        write_json(self.path, document)
        self.digest = file_sha(self.path)
        with self.output.with_suffix(".fit-seal.sha256").open("x") as file:
            file.write(self.digest+"\n")
            file.flush()
            os.fsync(file.fileno())
        self.event("all_four_fits_sealed", manifest=str(self.path), sha256=self.digest)
        self.require_sealed()

    def require_sealed(self):
        require(self.digest is not None and self.path.is_file(), "evaluation attempted before all four fits sealed")
        require(file_sha(self.path) == self.digest, "fit freeze manifest changed")
        document = read_json(self.path)
        for name, expected in document["files"].items():
            require(file_sha(self.output/name) == expected, "frozen selector/policy changed")


def verify_inner(outer, folder):
    path = outer.verified_file(folder+"/manifest.json")
    document = read_json(path)
    files = document["files"]
    actual = {p.name for p in (outer.root/folder).iterdir() if p.is_file() and p.name != "manifest.json"}
    require(actual == set(files), "inner manifest/tree differs")
    for name, expected in files.items():
        entry = outer.files.get(folder+"/"+name)
        expected_sha = expected["sha256"] if isinstance(expected, dict) else expected
        require(entry is not None and entry["sha256"] == expected_sha, "inner/outer seal disagreement")
        if isinstance(expected, dict):
            require(entry["bytes"] == expected["bytes"], "inner/outer length disagreement")
    return document


def prefix_rows(path, count):
    rows, raw = [], bytearray()
    # Unbuffered readline consumes exactly the permitted prefix, with no eval-row prefetch.
    with Path(path).open("rb", buffering=0) as file:
        for _ in range(count):
            line = file.readline()
            require(bool(line), "missing permitted fitting row")
            raw.extend(line)
            rows.append(json.loads(line))
    return rows, sha(raw)


def validate_rows(rows, count, evaluation=False):
    seed, tag, partition = (20260916, EVAL_TAG, "fresh_eval") if evaluation else (20260915, FIT_TAG, "fit")
    require(len(rows) == count, "wrong query population")
    cells = []
    for i, row in enumerate(rows):
        require(row["schema"] == "looped-known-features-v1" and row["input_index"] == i and
                row["layout_index"] == i and row["episode_id"] == tag+i and row["data_seed"] == seed and
                row["partition"] == partition and row["permutation_id"] == 0 and
                row["condition"] == "factual" and row["support_cleared"] is False and
                row["inferred_controls"] == [0, 1, 2, 3] and row["evaluation_loops"] == 4 and
                row["min_distance"] == row["max_distance"] == row["oracle_distance"] == 1,
                "feature row population/mapping mismatch")
        visible = row["visible_cells"]
        require(isinstance(visible, list) and len(visible) == 64 and
                all(type(x) is int and 0 <= x <= 3 for x in visible), "visible cell categorical shape")
        pixels = np.repeat(np.asarray(visible, dtype="<u4"), 64)
        require(sha(pixels.tobytes()) == row["query_sha256"], "visible-cell/token-order query digest mismatch")
        require(type(row["correct_action"]) is int and 0 <= row["correct_action"] < 4 and
                sha(np.asarray(row["correct_action"], dtype="<u4").tobytes()) == row["label_sha256"], "action digest mismatch")
        for name, shape in [("current", [64, 128]), ("cls", [128]), ("policy", [4])]:
            descriptor = row["arrays"][name]
            length = int(np.prod(shape))*4
            require(descriptor == {"file": f"known-features-{name}.f32", "dtype": "F32LE",
                                   "shape": shape, "byte_offset": i*length, "byte_length": length},
                    "feature row byte offsets differ")
        cells.append(visible)
    roles = role_targets(cells)
    actions = geometry_labels(roles)
    require(np.array_equal(actions, [r["correct_action"] for r in rows]), "visible geometry/action mismatch")
    require(len({r["query_sha256"] for r in rows}) == count and
            len({r["input_sha256"] for r in rows}) == count, "duplicate input/query")
    require((np.bincount(actions, minlength=4) > 0).all(), "missing action class")
    return roles, actions


def identity(row):
    return {key: row[key] for key in ("input_index", "episode_id", "partition", "input_sha256",
                                     "query_sha256", "label_sha256", "correct_action")}


def load_fit(core, outer8, outer9):
    folder = f"caches/fit-{core}"
    manifest = verify_inner(outer9, folder)
    require(file_sha(outer9.root/folder/"manifest.json") == FIT_CACHE_SEALS[core], "wrong registered fit cache")
    require(manifest["schema"] == "looped-learned-readout-cache-v1" and
            manifest["core_checkpoint_sha256"] == CHECKPOINTS[core] and manifest["rows"] == 512 and
            manifest["partition"] == "fit", "fit cache population/checkpoint mismatch")
    require(set(manifest["files"]) == {"current.f32", "cls.f32", "rows.jsonl"}, "fit cache must be materialized prefix only")
    for name in manifest["files"]:
        outer9.verified_file(folder+"/"+name)
    source = manifest["source"]
    source_folder = f"features-{core}"
    verify_inner(outer8, source_folder)
    require(source["root"] == str(C8/source_folder) and source["source_revision"] == C8_SOURCE and
            source["binary_sha256"] == C8_BINARY and source["data_seed"] == 20260915 and
            source["episode_id_base"] == FIT_TAG and
            source["manifest_sha256"] == outer8.files[source_folder+"/manifest.json"]["sha256"], "C8 cache source mismatch")
    array = manifest["source_arrays"]["current"]
    require(array["byte_offset"] == 0 and array["byte_length"] == 512*64*128*4 and
            array["sha256"] == outer8.files[source_folder+"/known-features-current.f32"]["sha256"], "C8 source feature prefix mismatch")
    data = (outer9.root/folder/"current.f32").read_bytes()
    require(len(data) == 512*64*128*4, "fit feature length mismatch")
    with (outer8.root/source_folder/"known-features-current.f32").open("rb", buffering=0) as file:
        require(file.read(len(data)) == data, "C8/cache raw prefix differs")
    rows, role_prefix_sha = prefix_rows(outer8.root/source_folder/"known-features-rows.jsonl", 512)
    roles, actions = validate_rows(rows, 512)
    cache_rows, _ = prefix_rows(outer9.root/folder/"rows.jsonl", 512)
    require([identity(r) for r in rows] == cache_rows, "C8 role sidecar/cache identities differ")
    features = finite(np.frombuffer(data, dtype="<f4").reshape(512, 64, 128), "cached fitting features")
    metadata = read_json(outer8.verified_file(source_folder+"/metadata.json"))
    require(metadata["provenance"]["checkpoint"]["sha256"] == CHECKPOINTS[core], "C8 feature checkpoint differs")
    for name in ("initial.safetensors", "final.safetensors"):
        require(outer8.files[source_folder+"/"+name]["sha256"] == CHECKPOINTS[core], "C8 frozen checkpoint changed")
    return features, rows, roles, actions, {"cache_manifest_sha256": FIT_CACHE_SEALS[core],
        "current_sha256": sha(data), "role_prefix_sha256": role_prefix_sha,
        "feature_file": str(outer9.root/folder/"current.f32"), "byte_offset": 0, "byte_length": len(data),
        "role_source": str(outer8.root/source_folder/"known-features-rows.jsonl"), "role_rows": 512}


def load_eval_features(core, outer, barrier):
    barrier.require_sealed()
    folder = f"features-fresh-{core}"
    verify_inner(outer, folder)
    report = read_json(outer.verified_file(folder+"/report.json"))
    metadata = read_json(outer.verified_file(folder+"/metadata.json"))
    require(report["schema"] == "looped-known-features-v1" and
            report["status"] == "complete_pending_analysis" and report["optimizer_updates"] == 0 and
            report["model_forwards"] == 256 and report["input_rows"] == 256 and
            report["fit_rows"] == 0 and report["data_seed"] == 20260916 and report["episode_id_base"] == EVAL_TAG and
            report["loops"] == 4 and report["checkpoint_sha256"] == CHECKPOINTS[core] and
            metadata["provenance"]["source_revision"] == C9_SOURCE and
            metadata["provenance"]["binary_sha256"] == C9_EXTRACTOR and
            metadata["provenance"]["checkpoint"]["sha256"] == CHECKPOINTS[core], "evaluation extraction provenance mismatch")
    for name in ("initial.safetensors", "final.safetensors"):
        require(outer.files[folder+"/"+name]["sha256"] == CHECKPOINTS[core], "evaluation frozen checkpoint changed")
    raw = outer.verified_file(folder+"/known-features-current.f32").read_bytes()
    native = outer.verified_file(folder+"/known-features-policy.f32").read_bytes()
    require(len(raw) == 256*64*128*4 and len(native) == 256*4*4, "evaluation raw byte shape mismatch")
    return finite(np.frombuffer(raw, dtype="<f4").reshape(256, 64, 128), "evaluation features"), \
        finite(np.frombuffer(native, dtype="<f4").reshape(256, 4), "native policy"), folder


def runtime_environment():
    libraries = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        path = line.split()[-1]
        if path.startswith("/") and ("openblas" in path.lower() or "mkl_rt" in path.lower()):
            libraries[path] = None
    require(libraries, "unverified BLAS runtime: expected a discoverable OpenBLAS/MKL library")
    symbols = ("scipy_openblas_get_num_threads64_", "scipy_openblas_get_num_threads", "openblas_get_num_threads64_", "openblas_get_num_threads", "MKL_Get_Max_Threads")
    for path in libraries:
        library = ctypes.CDLL(path)
        getter = next((getattr(library, name) for name in symbols if hasattr(library, name)), None)
        require(getter is not None, "cannot verify active BLAS thread count")
        getter.restype = ctypes.c_int
        require(getter() == 1, "BLAS is not single threaded")
        libraries[path] = {"sha256": file_sha(path), "threads": getter()}
    return {"python_version": sys.version, "python_executable": sys.executable,
            "python_sha256": file_sha(sys.executable), "numpy_version": np.__version__,
            "numpy_module": np.__file__, "numpy_module_sha256": file_sha(np.__file__), "blas": libraries,
            "thread_environment": {key: os.environ[key] for key in
             ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS")}}


def distribution(roles, actions):
    return {"action_counts": np.bincount(actions, minlength=4).tolist(),
            "role_position_counts": [np.bincount(roles[:, r], minlength=64).tolist() for r in range(2)]}


def save_result(path, result, roles, actions, policy, fitted_roles=None):
    true, true_margins = localization(result, roles)
    fitted, fitted_margins = localization(result, roles if fitted_roles is None else fitted_roles)
    arrays = dict(result, true_roles=roles, fitted_roles=roles if fitted_roles is None else fitted_roles,
                  actions=actions, true_margins=true_margins, fitted_margins=fitted_margins)
    policy_metrics = {}
    for name in ("hard", "soft"):
        scores = policy_scores(result[name], policy)
        predictions = scores.argmax(1)
        arrays[name+"_policy_scores"] = scores
        arrays[name+"_policy_predictions"] = predictions
        policy_metrics[name] = {"correct": int((predictions == actions).sum()), "rows": len(actions),
                                "accuracy": float((predictions == actions).mean())}
    save_arrays(path, **arrays)
    return {"true_localization": true, "fitted_localization": fitted, "policy": policy_metrics}


def run(output, freeze):
    started = time.monotonic()
    barrier = Barrier(output)
    def deadline():
        require(time.monotonic()-started <= 60, "registered 60-second diagnostic budget exceeded")
    environment = runtime_environment()
    outer8 = Outer(C8, R8/"completed-campaign.manifest.json", C8_SEAL)
    outer9 = Outer(C9, R9/"completed-campaign.manifest.json", C9_SEAL)
    parent_binary_hashes = {entry["sha256"] for entry in outer9.files.values()}
    require(C9_BINARY in parent_binary_hashes and C9_EXTRACTOR in parent_binary_hashes,
            "registered C9 binaries missing from completed outer seal")
    # Only manifests/inventory and fitting inputs are touched until the barrier seals.
    analysis_path = R8/"completed-analysis.json"
    require(file_sha(analysis_path) == freeze["c8_analysis_sha256"], "C8 frozen policy result digest differs")
    analysis = read_json(analysis_path)
    require(analysis["status"] == "valid_completed_feature_diagnostic" and
            analysis["source_revision"] == C8_SOURCE and analysis["binary_sha256"] == C8_BINARY,
            "C8 policy analysis provenance mismatch")
    controls = synthetic_controls()
    barrier.event("synthetic_controls_passed")
    policies, populations, input_provenance = {}, {}, {}
    for core in CORES:
        policies[core] = analysis["fit"]["probes"][core+"/visible_roles"]["real"]
        validate_policy(policies[core])
        h, rows, roles, actions, provenance = load_fit(core, outer8, outer9)
        oracle = h[np.arange(512)[:, None], roles].reshape(512, 256).astype(np.float64)
        oracle_correct = int((policy_scores(oracle, policies[core]).argmax(1) == actions).sum())
        require(oracle_correct/512 >= .90, "inconclusive_failed_control: C8 fit oracle compatibility")
        input_provenance[core] = provenance | {"fit_oracle_correct": oracle_correct}
        populations[core] = h, rows, roles, actions
        write_json(output/f"policy-{core}.json", policies[core])
        deadline()
    require([identity(r) for r in populations["initial"][1]] ==
            [identity(r) for r in populations["final"][1]], "fit core populations differ")
    permutation = np.random.Generator(np.random.PCG64(1917)).permutation(512)
    require(np.array_equal(np.sort(permutation), np.arange(512)), "invalid null permutation")
    save_arrays(output/"null-targets.npz", permutation=permutation,
                true_targets=populations["initial"][2], null_targets=populations["initial"][2][permutation])
    fit_reports, selectors, mass_valid = {}, {}, True
    freeze_names = [f"policy-{core}.json" for core in CORES]
    for core in CORES:
        h, rows, roles, actions = populations[core]
        for arm in ARMS:
            name = core+"/"+arm
            targets = roles if arm == "true" else roles[permutation]
            barrier.event("fit_started", core=core, arm=arm)
            selector, fit = fit_selector(h, targets)
            result, reconstruction_error = apply_selector(h, selector)
            mass, masses64, masses32 = mass_checks(result, targets, selector["eligible"])
            mass_valid = mass_valid and mass["passed"]
            selector["output_weight"] = (np.asarray(policies[core]["coefficients"])/np.asarray(policies[core]["scale"])[:, None]).T
            selector["output_bias"] = np.asarray(policies[core]["label_mean"]) - \
                (np.asarray(policies[core]["mean"])/np.asarray(policies[core]["scale"])) @ np.asarray(policies[core]["coefficients"])
            filename = f"selector-{core}-{arm}.npz"
            save_arrays(output/filename, **selector)
            freeze_names.append(filename)
            scored = save_result(output/f"fit-{core}-{arm}.npz", result, roles, actions, policies[core], targets)
            fit_reports[name] = fit | scored | {"mass_check": mass, "raw_inference_error": reconstruction_error,
                "finite_fit_witness": bool(selector["eligible"].all() and mass["passed"])}
            selectors[name] = selector
            barrier.event("fit_completed", core=core, arm=arm, selector_sha256=file_sha(output/filename))
            deadline()
    write_json(output/"fit-report.json", fit_reports)
    barrier.seal(freeze_names)
    # Evaluation consumes the serialized, hash-bound values that were actually sealed.
    for core in CORES:
        policies[core] = read_json(output/f"policy-{core}.json")
        for arm in ARMS:
            with np.load(output/f"selector-{core}-{arm}.npz", allow_pickle=False) as arrays:
                selectors[core+"/"+arm] = {name: arrays[name] for name in arrays.files}
    barrier.event("evaluation_integrity_read_started")
    # Full parent-tree hashing necessarily reads eval bytes, so it occurs after the seal.
    outer8.verify_all(barrier)
    outer9.verify_all(barrier)
    deadline()
    indices = bootstrap_indices()
    save_arrays(output/"bootstrap.npz", indices=indices)
    eval_reports, population_reports = {}, {}
    evaluation_identities = None
    oracle_controls = {}
    for core in CORES:
        h, native, folder = load_eval_features(core, outer9, barrier)
        barrier.event("evaluation_features_opened", core=core)
        # All four selectors are frozen. For this core, inference precedes label loading.
        results = {arm: apply_selector(h, selectors[core+"/"+arm])[0] for arm in ARMS}
        learned_policy = {arm: {pool: policy_scores(results[arm][pool], policies[core])
                               for pool in ("hard", "soft")} for arm in ARMS}
        barrier.event("evaluation_predictions_computed_before_labels", core=core)
        rows, row_sha = prefix_rows(outer9.root/folder/"known-features-rows.jsonl", 256)
        roles, actions = validate_rows(rows, 256, True)
        require(len((outer9.root/folder/"known-features-rows.jsonl").read_text().splitlines()) == 256, "extra evaluation rows")
        ids = [identity(r) for r in rows]
        if evaluation_identities is None:
            evaluation_identities = ids
        require(ids == evaluation_identities, "evaluation core populations differ")
        require(not {r["query_sha256"] for r in rows} &
                {r["query_sha256"] for r in populations[core][1]}, "fit/evaluation query overlap")
        write_json(output/f"evaluation-identities-{core}.json", ids)
        oracle_pools = h[np.arange(256)[:, None], roles].reshape(256, 256).astype(np.float64)
        oracle_scores = policy_scores(oracle_pools, policies[core])
        oracle_metric, _ = accuracy_interval(oracle_scores.argmax(1) == actions, indices)
        native_metric, _ = accuracy_interval(native.argmax(1) == actions, indices)
        oracle_controls["oracle_"+core] = oracle_metric["accuracy"] >= .90
        save_arrays(output/f"controls-{core}.npz", oracle_scores=oracle_scores, native_logits=native,
                    actions=actions, roles=roles)
        population_reports[core] = {"fit": distribution(populations[core][2], populations[core][3]),
            "evaluation": distribution(roles, actions), "oracle": oracle_metric, "native": native_metric,
            "evaluation_row_sha256": row_sha}
        for arm in ARMS:
            result = results[arm]
            metrics, _ = evaluation_metrics(result, roles, actions, policies[core], indices)
            for pool in ("hard", "soft"):
                require(np.array_equal(policy_scores(result[pool], policies[core]), learned_policy[arm][pool]),
                        "policy scoring depended on evaluation labels")
            details = save_result(output/f"evaluation-{core}-{arm}.npz", result, roles, actions, policies[core])
            eval_reports[core+"/"+arm] = metrics | details
        barrier.event("evaluation_scored", core=core)
        deadline()
    numerical_controls = {"synthetic": all(v for k,v in controls.items() if k not in
        ("positive_residual", "identical_residual")), "fit_mass": bool(mass_valid), **oracle_controls}
    decision = classify(eval_reports, numerical_controls)
    barrier.require_sealed()
    require(file_sha(analysis_path) == freeze["c8_analysis_sha256"], "C8 policy source changed")
    for outer in (outer8, outer9):
        outer.verify_all(barrier)
    deadline()
    return {"schema": "c10-role-selector-witness-v1", "status": "complete_selection_only" if decision["controls_valid"] else "inconclusive_failed_control",
        "registration_sha256": REG_SHA, "runtime": environment, "elapsed_seconds": time.monotonic()-started,
        "parent_seals": {"c8": C8_SEAL, "c9": C9_SEAL}, "c8_analysis_sha256": freeze["c8_analysis_sha256"],
        "input_provenance": input_provenance, "synthetic_controls": controls, "fit": fit_reports,
        "evaluation": eval_reports, "populations": population_reports, "decision": decision,
        "numerical_controls": numerical_controls,
        "fit_seal": {"path": str(barrier.path), "sha256": barrier.digest}, "chronology": barrier.events,
        "real_ridge_solves": 4, "synthetic_ridge_solves": 2, "fit_token_rows": 32768,
        "model_forwards": 0, "optimizer_updates": 0, "null_seed": 1917, "bootstrap_seed": 1918,
        "bootstrap_draws": 10000, "scope": "privileged fitting supervision; selection-only C9 query reuse",
        "limits": ["Failed ridge is not infeasibility or information absence.",
                   "C9-class translation is real-arithmetic; F32 CPU checks do not validate a CUDA transplant.",
                   "No causal localization to AdamW, model promotion, new training, or ARC claim.",
                   "NumPy/BLAS work is not CandleGraph-instrumented; parent extraction profiles are retained."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir
    require(output.is_absolute() and output.parent.resolve() == HERE and not output.exists(), "new output directory must be directly under R10")
    require(args.freeze.is_absolute() and args.freeze.is_file() and not args.freeze.is_symlink(), "missing reviewed source freeze")
    frozen = read_json(args.freeze)
    freeze_digest = file_sha(args.freeze)
    source_revision = frozen.get("source_revision", "")
    require(isinstance(source_revision, str) and len(source_revision) == 40 and
            all(c in "0123456789abcdef" for c in source_revision), "missing reviewed source revision")
    require(frozen.get("accepted") is True and frozen.get("registration_sha256") == REG_SHA and
            file_sha(HERE/"registration.md") == REG_SHA and
            frozen.get("script_sha256") == file_sha(__file__) and
            frozen.get("tests_sha256") == file_sha(HERE/"role_selector_witness_tests.py") and
            frozen.get("synthetic_tests_passed", 0) > 0 and frozen.get("c9_seal_sha256") == C9_SEAL,
            "reviewed registration/source/tests/parent freeze mismatch")
    require(all(not output.with_suffix(suffix).exists() for suffix in
                (".fit-seal.json", ".fit-seal.sha256", ".manifest.sha256")), "reused output evidence sibling")
    output.mkdir()
    write_json(output/"launch.json", {"pid": os.getpid(), "created_local": local_now(), "args": sys.argv,
        "registration_sha256": REG_SHA, "script_sha256": file_sha(__file__),
        "tests_sha256": file_sha(HERE/"role_selector_witness_tests.py"), "source_revision": source_revision,
        "freeze": str(args.freeze), "freeze_sha256": freeze_digest, "status": "running"})
    wall_started = time.monotonic()
    def timeout(_signum, _frame):
        raise TimeoutError("registered 90-second external-equivalent wall deadline")
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 90)
    exit_code = 0
    try:
        report = run(output, frozen)
    except Exception as error:
        exit_code = 1
        report = {"status": "incomplete_timeout" if isinstance(error, TimeoutError) or "budget exceeded" in str(error)
                  else "inconclusive_failed_control" if isinstance(error, NumericalControlFailure) or "inconclusive_failed_control" in str(error)
                  else "failed_integrity_or_numerical_control", "error": f"{type(error).__name__}: {error}",
                  "elapsed_seconds": time.monotonic()-wall_started, "model_forwards": 0, "optimizer_updates": 0}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    report["total_wall_seconds"] = time.monotonic()-wall_started
    report["script_sha256"] = file_sha(__file__)
    report["tests_sha256"] = file_sha(HERE/"role_selector_witness_tests.py")
    report["registration_sha256"] = REG_SHA
    report["source_freeze_sha256"] = freeze_digest
    report["source_revision"] = source_revision
    if (file_sha(args.freeze) != freeze_digest or report["script_sha256"] != frozen["script_sha256"] or
            report["tests_sha256"] != frozen["tests_sha256"] or file_sha(HERE/"registration.md") != REG_SHA):
        report["status"] = "failed_integrity_source_changed"
        exit_code = 1
    write_json(output/"report.json", report)
    files = {path.name: {"sha256": file_sha(path), "bytes": path.stat().st_size} for path in output.iterdir()}
    write_json(output/"manifest.json", {"schema": "c10-artifacts-v1", "created_local": local_now(), "files": files})
    digest = file_sha(output/"manifest.json")
    with output.with_suffix(".manifest.sha256").open("x") as file:
        file.write(digest+"\n")
    for name, entry in files.items():
        require(file_sha(output/name) == entry["sha256"], "final output seal verification failed")
    print(json.dumps({"status": report["status"], "report": str(output/"report.json"), "manifest_sha256": digest}))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
