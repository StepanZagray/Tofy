#!/usr/bin/env python3
"""Independent C13 numerics; pinned receipt owns source/profile/lifecycle provenance."""
import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import signal
import struct
import time
import numpy as np

MAPS = (0, 2, 4, 5, 7, 8, 9, 10, 13, 14, 15, 16, 18, 19, 21, 23)
PERMS = tuple(itertools.permutations(range(4)))
SHAPES = {"queries": (2, 128), "output.weight": (4, 256), "output.bias": (4,)}
EXTRAS = {"logits", "attention", "pooled", "checkpointstage"}

def check(ok, reason):
    if not ok:
        raise ValueError(reason)

def decode(raw):
    def pairs(items):
        out = {}
        for k, v in items:
            check(k not in out, "duplicate JSON key")
            out[k] = v
        return out
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: check(False, "nonfinite JSON"))

def same(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)

def read_bound(binding):
    p = Path(binding["path"])
    check(p.is_absolute() and p.is_file() and not p.is_symlink() and p.stat().st_size <= 128 * 1024**2, "bound file type/size")
    raw = p.read_bytes()
    check(hashlib.sha256(raw).hexdigest() == binding["sha256"], "file hash: " + str(p))
    return raw

def array(value, shape):
    def typed(x):
        return all(typed(y) for y in x) if isinstance(x, list) else type(x) in (int, float)
    check(typed(value), "array scalar type")
    a = np.asarray(value, dtype=np.float64)
    check(a.shape == shape and np.isfinite(a).all() and (np.abs(a) <= np.finfo(np.float32).max).all(), "array shape/finiteness")
    return a

def head(raw):
    check(len(raw) >= 8, "head truncated")
    size = struct.unpack_from("<Q", raw)[0]; start = 8 + size
    check(2 <= size <= 1024 * 1024 and start <= len(raw), "head header size")
    header = decode(raw[8:start]); check(type(header) is dict, "head header")
    if "__metadata__" in header:
        check(type(header["__metadata__"]) is dict and all(type(k) is type(v) is str for k, v in header["__metadata__"].items()), "head metadata")
    check(set(header) - {"__metadata__"} == set(SHAPES), "head tensor names")
    regions, tensors = [], {}
    for name, shape in SHAPES.items():
        spec = header[name]
        check(set(spec) == {"dtype", "shape", "data_offsets"} and spec["dtype"] == "F32", "head dtype/schema")
        check(same(spec["shape"], list(shape)), "head tensor shape")
        offsets = spec["data_offsets"]
        check(type(offsets) is list and len(offsets) == 2 and all(type(x) is int for x in offsets), "head offsets type")
        lo, hi = offsets
        check(0 <= lo < hi <= len(raw) - start and hi-lo == math.prod(shape)*4, "head offsets/range")
        tensors[name] = np.frombuffer(raw[start+lo:start+hi], dtype="<f4").reshape(shape).copy()
        check(np.isfinite(tensors[name]).all(), "head nonfinite")
        regions.append((lo, hi))
    regions.sort()
    check(regions[0][0] == 0 and regions[-1][1] == len(raw)-start and all(a[1] == b[0] for a, b in zip(regions, regions[1:])), "head overlap/gap/trailing bytes")
    return tensors, header, start

def verify_heads(original, scaled):
    a, ha, start = head(original); b, hb, other = head(scaled)
    check(start == other and original[:start] == scaled[:start] and same(ha, hb), "changed header/layout")
    lo, hi = ha["queries"]["data_offsets"]
    with np.errstate(over="ignore", invalid="ignore"):
        expected = (a["queries"] * np.float32(16)).astype("<f4")
    check(np.isfinite(expected).all() and b["queries"].tobytes() == expected.tobytes(), "queries not exact finite F32 x16")
    check(original[start:start+lo] == scaled[start:start+lo] and original[start+hi:] == scaled[start+hi:], "foreign bytes changed")
    return a, b

def identity(row, index):
    check(type(row["group_index"]) is int and row["group_index"] == index//16 and row["row_index"] == index, "query order")
    check(type(row["permutation_id"]) is int and row["permutation_id"] == MAPS[index%16], "map order")
    check(row["cohort"] == "seen" and row["condition"] == "factual" and row["support_cleared"] is False, "seen factual only")
    c = row["query_cells"]
    check(len(c) == 64 and all(type(x) is int and 0 <= x <= 3 for x in c) and c.count(2) == c.count(3) == 1, "visible cells")
    a, g = c.index(2), c.index(3); delta = (g//8-a//8, g%8-a%8)
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
    check(delta in directions, "adjacent query roles")
    label = PERMS[row["permutation_id"]].index(directions.index(delta))
    check((a, g, label) == (row["agent_patch"], row["goal_patch"], row["correct_action"]), "independent role/label mismatch")
    check(same(row["inferred_controls"], list(PERMS[row["permutation_id"]])), "audited bijection mismatch")
    actions = row["observed_support_action_ids"]
    check(len(actions) == len(set(actions)) == 3 and all(type(x) is int and 0 <= x < 4 for x in actions), "support actions")
    omitted = next(iter(set(range(4))-set(actions)))
    check(omitted == row["omitted_action"], "omitted action")
    return a, g, label, omitted

def score(rows, audit):
    check(len(rows) == len(audit) == 1024, "complete 64x16 population required")
    ids, z, att, pooled = [], [], [], []
    for i, (row, original) in enumerate(zip(rows, audit)):
        check(set(row) == set(original) | EXTRAS and same(original, {k: row[k] for k in original}), "audit/output identity")
        check(row["checkpointstage"] == "final", "terminal checkpoint stage")
        ids.append(identity(original, i)); z.append(array(row["logits"], (4,)))
        att.append(array(row["attention"], (2, 64))); pooled.append(array(row["pooled"], (256,)))
    ids, z, att, pooled = map(np.asarray, (ids, z, att, pooled))
    check((att >= 0).all() and (att <= 1).all() and np.all(abs(att.sum(2)-1) <= 2e-5), "attention probabilities")
    labels, omitted, roles = ids[:, 2], ids[:, 3], ids[:, :2]
    predictions = np.asarray([max(range(4), key=list(x).__getitem__) for x in z])
    hit = predictions == labels
    ce = np.asarray([math.log(math.fsum(math.exp(v-max(x)) for v in x)) + (max(x)-x[y]) for x, y in zip(z, labels)])
    role_mass = np.take_along_axis(att, roles[:, :, None], 2).squeeze(2)
    check(np.all(att.argmax(2) == roles) and np.all((att == att.max(2, keepdims=True)).sum(2) == 1), "correct unique role winners")
    check(all(np.all((labels == a).reshape(64,16).sum(1) == 4) for a in range(4)), "within-group label balance")
    subsets = {}
    for name, mask, size in (("omitted", labels == omitted, 4), ("demonstrated", labels != omitted, 12)):
        check(np.all(mask.reshape(64,16).sum(1) == size), "within-group subset balance")
        subsets[name] = dict(rows=int(mask.sum()), correct=int(hit[mask].sum()), accuracy=float(hit[mask].mean()))
    summary = dict(rows=1024, groups=64, maps_per_group=16, correct=int(hit.sum()), accuracy=float(hit.mean()),
                   ce=math.fsum(ce)/1024, label_histogram=np.bincount(labels, minlength=4).tolist(),
                   prediction_histogram=np.bincount(predictions, minlength=4).tolist(),
                   all_maps_correct_groups=int(hit.reshape(64,16).all(1).sum()), subsets=subsets)
    return summary, hit.reshape(64,16).mean(1), ce.reshape(64,16).mean(1), att, role_mass, pooled, z

def power(attention):
    with np.errstate(divide="ignore"):
        x = 16 * np.log(attention)
    x -= x.max(2, keepdims=True); x = np.exp(x); x /= x.sum(2, keepdims=True)
    check(np.isfinite(x).all(), "nonfinite attention power")
    return x

def interval(values, weights):
    samples = np.sort(weights @ values / 64)
    def q(p):
        location = 9999*p; lo = int(location)
        return float(samples[lo] + (location-lo)*(samples[lo+1]-samples[lo]))
    return dict(estimate=float(values.mean()), ci95=[q(.025), q(.975)])

def decide(accuracy, lower):
    return ("frozen_sharpening_recovers_registered_fit" if accuracy >= .9 else "partial_accuracy_gain_only") if lower > 0 else "concentration_insufficient_for_accuracy_recovery"

def compute(baseline, treatment, audit, old_raw, new_raw):
    old, new = verify_heads(old_raw, new_raw)
    b, ba, bc, bp, bm, bx, bz = score(baseline, audit)
    t, ta, tc, tp, tm, tx, tz = score(treatment, audit)
    predicted = power(bp); predicted_mass = predicted.max(2)
    check((predicted_mass >= .99).all(), "premise_not_met: predicted mass below .99")
    check((tm >= .99).all(), "actual target mass below .99")
    error = abs(tp-predicted)
    check(np.all(error <= 1e-4 + 1e-4*abs(predicted)), "attention power mismatch")
    affine_errors = []
    for params, x, z in ((old, bx, bz), (new, tx, tz)):
        reconstructed = x @ params["output.weight"].astype(float).T + params["output.bias"].astype(float)
        residual = abs(z-reconstructed)
        check(np.all(residual <= 2e-5 + 2e-5*abs(reconstructed)), "affine reconstruction mismatch")
        affine_errors.append(float(residual.max()))
    draws = np.random.Generator(np.random.PCG64(1940)).integers(0, 64, (10000,64))
    weights = np.stack([np.bincount(x, minlength=64) for x in draws]).astype(float)
    contrasts = {k: interval(v, weights) for k, v in dict(accuracy_delta=ta-ba, ce_delta=tc-bc, accuracy_minus_constant=ta-.25).items()}
    controls = dict(power_max_absolute_error=float(error.max()), affine_max_absolute_error=affine_errors[1],
                    role_mass_min=tm.min(0).tolist(), role_mass_mean=tm.mean(0).tolist(),
                    predicted_role_mass_min=predicted_mass.min(0).tolist(), predicted_role_mass_mean=predicted_mass.mean(0).tolist())
    extra = dict(baseline_affine_max_absolute_error=affine_errors[0], baseline_role_mass_min=bm.min(0).tolist(),
                 baseline_role_mass_mean=bm.mean(0).tolist(), baseline_role_correct_counts=[1024,1024],
                 treatment_role_correct_counts=[1024,1024], draws_u64le_sha256=hashlib.sha256(draws.astype("<u8").tobytes()).hexdigest())
    return dict(baseline=b, treatment=t, contrasts=contrasts, controls=controls,
                decision=decide(t["accuracy"], contrasts["accuracy_delta"]["ci95"][0])), extra

def compare(expected, actual, prefix=""):
    errors = []
    if isinstance(expected, dict):
        check(isinstance(actual, dict), "report object " + prefix)
        for k, v in expected.items(): errors.extend(compare(v, actual[k], prefix+"/"+k))
    elif isinstance(expected, list):
        check(isinstance(actual, list) and len(actual) == len(expected), "report list " + prefix)
        for i, (a, b) in enumerate(zip(expected, actual)): errors.extend(compare(a, b, prefix+"/"+str(i)))
    elif type(expected) is float:
        check(type(actual) in (int,float) and math.isfinite(actual), "report finite scalar " + prefix)
        errors.append(abs(expected-actual)); check(abs(expected-actual) <= 1e-10, "numerical report mismatch " + prefix)
    else: check(type(expected) is type(actual) and expected == actual, "exact report mismatch " + prefix)
    return errors

def review(config_path, config_sha, report_path, report_sha):
    start = time.monotonic(); config = decode(read_bound(dict(path=str(config_path), sha256=config_sha)))
    report = decode(read_bound(dict(path=str(report_path), sha256=report_sha)))
    receipt = decode(read_bound(config["integrity_receipt"]))
    check(all(receipt[k] is True for k in ("accepted", "zero_updates", "source_verified", "profile_verified")), "integrity receipt rejected")
    registration = dict(path=config["registration"], sha256=config["registration_sha256"])
    check(report["accepted"] is True and report["config_sha256"] == config_sha and report["registration_sha256"] == registration["sha256"], "report/config binding")
    raw = {}
    for key in ("baseline", "treatment", "original_head", "scaled_head", "audit"):
        binding = config[key]; check(receipt["files"][binding["path"]] == binding["sha256"], "receipt file closure")
        raw[key] = read_bound(binding)
    check(receipt["files"][registration["path"]] == registration["sha256"], "receipt registration closure"); read_bound(registration)
    rows = {k: [decode(x) for x in raw[k].splitlines()] for k in ("baseline", "treatment", "audit")}
    result, extra = compute(rows["baseline"], rows["treatment"], rows["audit"], raw["original_head"], raw["scaled_head"])
    errors = compare(result, report)
    check(time.monotonic()-start < 120, "CPU deadline")
    return dict(accepted=True, recomputed=result, additional=extra, config_sha256=config_sha, report_sha256=report_sha,
                reviewer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                maximum_absolute_difference=max(errors, default=0), compared_float_fields=len(errors),
                elapsed_seconds=time.monotonic()-start,
                scope="Independent numerical and head-byte agreement. Pinned receipt owns source/core/runtime/profiler/parent-seal provenance; reused seen examples, one exploratory contrast, no promotion.")

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "config-sha256", "report", "report-sha256", "output"): p.add_argument("--"+name, required=True)
    a = p.parse_args(); check(not Path(a.output).exists(), "output exists")
    signal.signal(signal.SIGALRM, lambda *_: check(False, "120-second CPU deadline")); signal.alarm(120)
    try:
        result = review(a.config, a.config_sha256, a.report, a.report_sha256); code = 0
    except (ValueError, TypeError, KeyError, OSError, OverflowError) as error:
        result = dict(accepted=False, decision="premise_not_met" if str(error).startswith("premise_not_met:") else "failed_integrity", error=str(error)); code = 1
    finally: signal.alarm(0)
    with open(a.output, "x") as f: json.dump(result, f, indent=2, allow_nan=False); f.write("\n")
    return code

if __name__ == "__main__": raise SystemExit(main())
