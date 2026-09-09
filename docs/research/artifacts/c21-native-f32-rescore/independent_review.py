#!/usr/bin/env python3
"""Independent C19 public-data/scalar review; no model or primary scorer import.

Frozen neural replay and runtime provenance are separately hash-bound evidence,
not an independent execution of the neural network by this reviewer.
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse
from collections import Counter
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
R = Path(__file__).resolve().parent
SEED = 20260923
BASE = 0x4E415449564542
MAPS = tuple(itertools.permutations(range(4)))
FIT = (0, 2, 4, 5, 7, 8, 9, 10, 13, 14, 15, 16, 18, 19, 21, 23)
DIRECTIONS = ((0, -1), (0, 1), (-1, 0), (1, 0))
FRAMES = ("before0", "after0", "before1", "after1", "before2", "after2", "current")
PANEL_KEYS = set("input_index query_index episode_id data_seed permutation_id policy_label input_sha256 query_sha256 metadata_sha256 public_cells public_metadata".split())
OUTPUT_KEYS = set("evaluation_index learned_attention adapter_records logits prediction control".split())
CHECKS = set("source build checkpoints zero_updates unchanged_parameters qualification profiles cleanup".split())
DEPENDENCY = "1ea5cc5c5998a1c2bd388eb158653dd0d1b7452a"
VISION = "4cd502f7a76fe3dd9729f6693e9d707c670972f7ebe58ad1cbdde37ea2bb2802"
SELECTOR = "a43ad78fe7a6b1594d23c205fb191c8ac81610deaaff25198cb421d423edc678"
BINDER = "d2deeba7b0fafc2386c9a19d39bc99b7534b91a0155d66e77792a5c4037bd717"
HISTORY = (
    ("09559eb4dd1b933de724da81aef3bb80547e010a00e38e39de31c9f72fab1c6d", 768),
    ("bd72fd9707759dd19074a22b9d62899419b28711722b3617c21cc277a6a25c0d", 256),
    ("09a7aa423b17f32bbae14ae9178a15d5fff8428936afdf23ebc4f09878a6ddca", 256),
    ("61ae87a267cc545b05884dc412160c0313a0c2bc1015cfa2323bde256f91eaeb", 256),
    ("55f200b3a136476896d57f284a7d5db4e8f8fc2b7b4fa54ef309d1837f3c5910", 73600),
    ("fbb137383a16a27d72141ef66028c1c9cb470afccc2b03864ec0a961a94d73f2", 1024),
)


class Invalid(ValueError):
    pass


def need(condition, message):
    if not condition:
        raise Invalid(message)


def exact(a, b):
    if type(a) is not type(b):
        return False
    if type(a) is dict:
        return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if type(a) is list:
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


def keys(value, expected, where):
    need(type(value) is dict and set(value) == expected, f"{where}: exact keys required")


def decode(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            need(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    def constant(value):
        raise Invalid("nonfinite JSON constant: " + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def regular(path):
    p = Path(path)
    need(p.is_absolute() and p == p.resolve(strict=True) and p.is_file(), "absolute regular nonsymlink file required")
    return p


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    p = regular(path)
    before = p.stat()
    with p.open("rb") as f:
        result = hashlib.file_digest(f, "sha256").hexdigest()
    after = p.stat()
    need((before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
         (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), "file changed during hashing")
    return result


def digest(value):
    need(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value), "malformed SHA256")
    return value


def bound(record):
    keys(record, {"path", "sha256"}, "binding")
    need(file_sha(record["path"]) == digest(record["sha256"]), "bound file changed")
    return regular(record["path"])


def load(path):
    p = regular(path)
    need(p.stat().st_size <= 128 * 1024 * 1024, "JSON size limit")
    return decode(p.read_bytes())


def jsonl(path):
    with regular(path).open("rb") as f:
        while line := f.readline(512 * 1024 + 1):
            need(len(line) <= 512 * 1024 and line.strip(), "empty/oversized JSONL row")
            yield decode(line)


def frozen(files):
    need(type(files) is dict and files, "frozen files required")
    for path, value in files.items():
        bound({"path": path, "sha256": value})


def array(value, shape):
    if not shape:
        need(type(value) in (float, int) and math.isfinite(value) and abs(value) <= 3.4028234663852886e38,
             "nonnumeric/nonfinite/out-of-F32-range array")
        return struct.unpack("<f", struct.pack("<f", float(value)))[0]
    need(type(value) is list and len(value) == shape[0], "array shape")
    return [array(v, shape[1:]) for v in value]


def flat(values):
    for value in values:
        if type(value) is list:
            yield from flat(value)
        else:
            yield value


def close(actual, expected, atol=1e-4, rtol=1e-5):
    left, right = list(flat(actual)), list(flat(expected))
    need(len(left) == len(right), "parity element count")
    error = 0.0
    for a, b in zip(left, right):
        need(math.isfinite(a) and math.isfinite(b) and abs(a-b) <= atol + rtol*abs(b), "numerical parity")
        error = max(error, abs(a-b))
    return error


def parity(actual, expected):
    error = close(actual, expected)
    ratio = max(abs(a-b)/(1e-4+1e-5*abs(b)) for a, b in zip(flat(actual), flat(expected)))
    return {"maximum_absolute_error": error, "maximum_tolerance_ratio": ratio}


def logit_parity(actual, reference):
    result = parity(actual, reference); eligible = mismatches = 0
    need(len(actual) == len(reference), "logit parity row count")
    for scores, ref in zip(actual, reference):
        winner = max(range(4), key=ref.__getitem__)
        margin = ref[winner]-max(ref[a] for a in range(4) if a != winner)
        error = max(abs(a-b) for a, b in zip(scores, ref))
        if margin > 0 and margin > 2*error:
            eligible += 1
            mismatches += int(max(range(4), key=scores.__getitem__) != winner)
    need(mismatches == 0, "eligible logit parity winner mismatch")
    return {**result, "rows": len(actual), "eligible_winners": eligible, "ineligible_winners": len(actual)-eligible,
            "eligible_winner_mismatches": mismatches,
            "eligibility": "reference winner margin > 0 and > 2 * row maximum absolute logit error"}


@lru_cache(maxsize=24)
def metadata(actions):
    values = []
    for frame in range(7):
        for cell in range(64):
            row = [0.0] * 10
            row[frame % 2 if frame < 6 else 2] = 1.0
            if frame < 6:
                row[3 + actions[frame//2]] = 1.0
            row[7:] = [cell % 8 / 7, cell // 8 / 7, frame//2 / 3 if frame < 6 else 1.0]
            values.extend(row)
    return struct.pack("<4480f", *values)


def pixels(cells):
    return b"".join(struct.pack("<I", value) * 64 for value in cells)


def delta(a, b):
    return b % 8 - a % 8, b // 8 - a // 8


def truth(row, index):
    keys(row, PANEL_KEYS, "panel row")
    group, mapping = divmod(index, 24)
    expected = {"input_index": index, "query_index": group, "permutation_id": mapping,
                "episode_id": BASE + group, "data_seed": SEED}
    need(all(exact(row[k], v) for k, v in expected.items()), "panel index/seed/tag/map identity")
    cells = row["public_cells"]
    need(type(cells) is list and len(cells) == 7, "seven public frames required")
    locations = []
    for frame, values in enumerate(cells):
        need(type(values) is list and len(values) == 64 and all(type(v) is int and 0 <= v <= 3 for v in values), "public categorical cells")
        need(values.count(2) == values.count(3) == 1 and values[0] not in (2, 3), "unique visible roles")
        if frame < 6:
            need(1 not in values and values[63] == 3, "support walls/fixed goal")
        else:
            need(all(values[c] == 1 for c in range(64) if c % 8 in (0, 7) or c//8 in (0, 7)), "query boundary walls")
        locations.append([values.index(2), values.index(3)])
    x, y = locations[0][0] % 8, locations[0][0]//8
    need(2 <= x <= 5 and 2 <= y <= 5, "support start geometry")
    meta = array(row["public_metadata"], (4480,))
    actions = []
    for pair in range(3):
        onehot = meta[pair*2*640+3:pair*2*640+7]
        need(onehot.count(1.0) == 1 and onehot.count(0.0) == 3, "public action one-hot")
        actions.append(onehot.index(1.0))
    need(len(set(actions)) == 3, "distinct public actions")
    packed_meta = struct.pack("<4480f", *meta)
    need(packed_meta == metadata(tuple(actions)), "public frame/coordinate/action/time metadata")
    for pair, action in enumerate(actions):
        before, after = cells[2*pair:2*pair+2]
        start, end = locations[2*pair][0], locations[2*pair+1][0]
        need(delta(start, end) == DIRECTIONS[MAPS[mapping][action]] and before[end] == 0, "support action effect")
        changed = list(before); changed[start], changed[end] = 0, 2
        need(exact(after, changed), "support modified non-agent pixels")
        if pair:
            need(exact(before, cells[2*pair-1]), "support continuity")
    direction = delta(*locations[6])
    need(direction in DIRECTIONS, "distance-one query required")
    label = MAPS[mapping].index(DIRECTIONS.index(direction))
    need(exact(row["policy_label"], label), "visible query action label")
    query_bytes = pixels(cells[6])
    expected = {"query_sha256": sha(query_bytes), "metadata_sha256": sha(packed_meta),
                "input_sha256": sha(b"".join(pixels(frame) for frame in cells) + packed_meta)}
    need(all(row[k] == v for k, v in expected.items()), "raw public input/query/metadata hash")
    return {"locations": locations, "actions": actions, "omitted": next(a for a in range(4) if a not in actions),
            "demonstrated": label in actions, "direction": DIRECTIONS.index(direction)}


def histories(record):
    value = load(bound(record))
    keys(value, {"history", "created_local", "scope"}, "history manifest")
    need(type(value["history"]) is list and len(value["history"]) == 6, "six complete historical sources")
    union, paths, counts = set(), set(), []
    for source, (expected_sha, expected_count) in zip(value["history"], HISTORY):
        path = bound(source)
        need(source["sha256"] == expected_sha and path not in paths, "wrong/duplicate historical source")
        paths.add(path); count = 0; local = set()
        for row in jsonl(path):
            query = digest(row.get("query_sha256")); union.add(query); local.add(query); count += 1
            if "id" in row and "episode_id" in row:
                need(exact(row["id"], row["episode_id"]), "conflicting historical typed identities")
        need(count == expected_count, "complete historical row count")
        counts.append({"path": str(path), "sha256": source["sha256"], "parsed_rows": count,
                       "hash_occurrences": count, "unique_query_hashes": len(local), "conflicts": 0})
    return union, counts


def panel(path, history):
    rows, truths = [], []
    for index, row in enumerate(jsonl(path)):
        need(index < 768, "excess panel rows")
        truths.append(truth(row, index)); rows.append(row)
    need(len(rows) == 768, "32x24 panel required")
    queries, inputs, directions = set(), set(), Counter()
    for start in range(0, 768, 24):
        block = rows[start:start+24]; facts = truths[start:start+24]; first = block[0]
        query = first["query_sha256"]
        need(query not in history and query not in queries, "historical/internal query overlap; no replacement")
        queries.add(query); directions[facts[0]["direction"]] += 1
        need(all(exact(r["public_cells"][6], first["public_cells"][6]) and r["query_sha256"] == query and
                 r["metadata_sha256"] == first["metadata_sha256"] for r in block), "unpaired query/public metadata")
        need(all(f["locations"][0] == facts[0]["locations"][0] for f in facts), "unpaired support start")
        need(Counter(r["policy_label"] for r in block) == Counter({a: 6 for a in range(4)}), "group action balance")
        need(sum(f["demonstrated"] for f in facts) == 18, "group demonstrated balance")
        for r in block:
            need(r["input_sha256"] not in inputs, "duplicate full panel input")
            inputs.add(r["input_sha256"])
    need(set(directions) == set(range(4)), "four query directions required")
    summary = {"rows": 768, "groups": 32, "label_histogram": [192]*4,
               "direction_histogram": [directions[d] for d in range(4)], "demonstrated": 576, "omitted": 192}
    return rows, truths, summary


def probabilities(value):
    attention = array(value, (7, 2, 64))
    for frame in attention:
        for role in frame:
            need(all(0 <= a <= 1 for a in role) and abs(math.fsum(role)-1) <= 1e-5, "invalid normalized attention")
    return attention


def positions(attention):
    return [[[math.fsum(weight*(cell%8 if axis == 0 else cell//8) for cell, weight in enumerate(role))
              for axis in range(2)] for role in frame] for frame in attention]


def records(attention, actions):
    xy = positions(attention)
    result = []
    for pair, action in enumerate(actions):
        result.append([xy[2*pair+1][0][axis]-xy[2*pair][0][axis] for axis in range(2)] +
                      [float(a == action) for a in range(4)] + [0.0])
    result.append([xy[6][1][axis]-xy[6][0][axis] for axis in range(2)] + [0.0]*4 + [1.0])
    return result


def metric(logits, label):
    winner = max(range(4), key=logits.__getitem__)
    peak = max(logits)
    return {"correct": int(winner == label), "prediction": winner,
            "ce": peak + math.log(math.fsum(math.exp(x-peak) for x in logits)) - logits[label],
            "margin": logits[label]-max(logits[a] for a in range(4) if a != label),
            "winner_margin": logits[winner]-max(logits[a] for a in range(4) if a != winner)}


def stream(path, panel_rows, truths, control):
    learned, effective, actual_records, logits, metrics = [], [], [], [], []
    maximum_record_error = 0.0
    for index, (row, reference, factual) in enumerate(itertools.zip_longest(jsonl(path), panel_rows, truths)):
        need(row is not None and reference is not None and factual is not None, "runtime stream count")
        keys(row, PANEL_KEYS | OUTPUT_KEYS, "runtime row")
        need(all(exact(row[k], reference[k]) for k in PANEL_KEYS - {"public_metadata"}), "runtime public input differs from panel")
        # JSON decimal spellings may differ; the declared input identity is F32LE.
        actual_metadata = struct.pack("<4480f", *array(row["public_metadata"], (4480,)))
        reference_metadata = struct.pack("<4480f", *array(reference["public_metadata"], (4480,)))
        need(actual_metadata == reference_metadata, "runtime public metadata F32 bits differ from panel")
        need(exact(row["evaluation_index"], index) and row["control"] == control, "runtime index/control")
        attention = probabilities(row["learned_attention"])
        used = attention if control == "factual" else [[[1/64]*64 for _ in range(2)] for _ in range(7)]
        observed = array(row["adapter_records"], (4, 7))
        expected = records(used, factual["actions"])
        maximum_record_error = max(maximum_record_error, close(observed, expected))
        # Public routing/record-kind columns are exact identities, not tolerances.
        need(all(observed[i][2:] == expected[i][2:] for i in range(4)), "changed public routing/record kind")
        scores = array(row["logits"], (4,))
        measured = metric(scores, reference["policy_label"])
        need(exact(row["prediction"], measured["prediction"]), "runtime prediction differs from first argmax")
        learned.append(attention); effective.append(used); actual_records.append(observed)
        logits.append(scores); metrics.append(measured)
    need(len(metrics) == 768, "runtime requires768 rows")
    return {"learned": learned, "effective": effective, "records": actual_records,
            "logits": logits, "metrics": metrics, "maximum_record_error": maximum_record_error}


def action_summary(rows, truths, metrics):
    need(len(rows) == len(truths) == len(metrics) == 768, "action population size")
    def subset(indices):
        selected = [metrics[i] for i in indices]
        correct = sum(m["correct"] for m in selected)
        return {"rows": len(indices), "correct": correct, "accuracy": correct/len(indices),
                "ce": math.fsum(m["ce"] for m in selected)/len(indices),
                "minimum_true_margin": min(m["margin"] for m in selected)}
    result = subset(list(range(768)))
    result.update(groups=32, minimum_true_margin=min(m["margin"] for m in metrics),
                  label_histogram=[sum(r["policy_label"] == a for r in rows) for a in range(4)],
                  prediction_histogram=[sum(m["prediction"] == a for m in metrics) for a in range(4)],
                  all_maps_correct_groups=sum(all(m["correct"] for m in metrics[i:i+24]) for i in range(0, 768, 24)),
                  missing_id_predictions=sum(m["prediction"] == t["omitted"] for m, t in zip(metrics, truths)))
    result["subsets"] = {name: subset([i for i, t in enumerate(truths) if t["demonstrated"] == demonstrated])
                         for name, demonstrated in (("demonstrated", True), ("omitted", False))}
    result["per_map"] = []
    def detailed(indices):
        return {**subset(indices), "minimum_true_margin": min(metrics[i]["margin"] for i in indices)}
    for mapping in range(24):
        result["per_map"].append({"map_id": mapping, **detailed(list(range(mapping, 768, 24)))})
    result["map_splits"] = {name: detailed([i for i, row in enumerate(rows) if (row["permutation_id"] in FIT) == familiar])
                            for name, familiar in (("familiar", True), ("heldout", False))}
    result["per_group"] = [{"query_index": group, **detailed(list(range(group*24, group*24+24)))} for group in range(32)]
    result["per_direction"] = [{"direction": d, **detailed([i for i, t in enumerate(truths) if t["direction"] == d])} for d in range(4)]
    result["group_accuracy_range"] = [min(g["accuracy"] for g in result["per_group"]), max(g["accuracy"] for g in result["per_group"])]
    return result


def uniform_groups(result):
    maximum_logit_error, maximum_record_error, eligible, disagreements = 0.0, 0.0, 0, 0
    for start in range(0, 768, 24):
        scores = result["logits"][start:start+24]
        base = scores[0]
        errors = [close([z], [base]) for z in scores]
        maximum_logit_error = max(maximum_logit_error, *errors)
        for record in result["records"][start:start+24]:
            need(exact(record, result["records"][start]), "uniform records differ within group")
            maximum_record_error = max(maximum_record_error, close(record, result["records"][start]))
        margins = [m["winner_margin"] for m in result["metrics"][start:start+24]]
        compare_winners = min(margins) > 0 and min(margins) > 2*max(errors)
        if compare_winners:
            eligible += 1
            disagrees = len({m["prediction"] for m in result["metrics"][start:start+24]}) != 1
            disagreements += int(disagrees)
        # The fixed identical-input control must be deterministic even on ties.
        need(len({m["prediction"] for m in result["metrics"][start:start+24]}) == 1,
             "uniform identical inputs have different predictions")
        need(sum(m["correct"] for m in result["metrics"][start:start+24]) == 6, "uniform group is not exact quarter")
    return {"maximum_logit_error": maximum_logit_error, "maximum_record_error": maximum_record_error,
            "eligible_groups": eligible, "winner_disagreements": disagreements}


def grounding(truths, attentions):
    n = len(truths)
    need(n == len(attentions) == 768, "grounding population size")
    expected = [t["locations"] for t in truths]
    winners = [[[max(range(64), key=role.__getitem__) for role in frame] for frame in attention]
               for attention in attentions]
    coordinates = [positions(attention) for attention in attentions]
    frames = []
    for frame in range(7):
        record = {"frame_index": frame}
        for role, name in enumerate(("agent", "goal")):
            correct = sum(winners[i][frame][role] == expected[i][frame][role] for i in range(n))
            masses = [attentions[i][frame][role][expected[i][frame][role]] for i in range(n)]
            errors = [[abs(coordinates[i][frame][role][axis] -
                           (expected[i][frame][role] % 8 if axis == 0 else expected[i][frame][role]//8))
                       for axis in range(2)] for i in range(n)]
            record[name] = {"correct": correct, "accuracy": correct/n,
                            "mean_mass": math.fsum(masses)/n, "minimum_mass": min(masses),
                            "mean_position_l1_error": math.fsum(math.fsum(e) for e in errors)/n,
                            "maximum_position_linf_error": max(max(e) for e in errors)}
        frames.append(record)
    displacements = []
    for record in range(4):
        before, after = ((record*2, 0), (record*2+1, 0)) if record < 3 else ((6, 0), (6, 1))
        f0, r0 = before; f1, r1 = after
        deltas, locations, errors = [], [], []
        for i in range(n):
            target = delta(expected[i][f0][r0], expected[i][f1][r1])
            deltas.append(delta(winners[i][f0][r0], winners[i][f1][r1]) == target)
            locations.append(winners[i][f0][r0] == expected[i][f0][r0] and winners[i][f1][r1] == expected[i][f1][r1])
            errors.extend(abs(coordinates[i][f1][r1][axis]-coordinates[i][f0][r0][axis]-target[axis]) for axis in range(2))
        displacements.append({"record_index": record, "correct_argmax_delta": sum(deltas),
                              "correct_argmax_locations": sum(locations),
                              "mean_absolute_error": math.fsum(errors)/(n*2), "maximum_absolute_error": max(errors)})
    return {"rows": n,
            "all_roles_correct": sum(all(winners[i][f][r] == expected[i][f][r] for f in range(7) for r in range(2)) for i in range(n)),
            "consumed_roles_correct": sum(all(winners[i][f][0] == expected[i][f][0] for f in range(7)) and
                                            winners[i][6][1] == expected[i][6][1] for i in range(n)),
            "frames": frames, "displacements": displacements}


def compare(actual, expected, where="report", stats=None):
    if stats is None:
        stats = {"compared_float_fields": 0, "compared_discrete_fields": 0, "max_absolute_float_difference": 0.0}
    if type(expected) is dict:
        keys(actual, set(expected), where)
        for key in expected:
            compare(actual[key], expected[key], where+"/"+key, stats)
    elif type(expected) is list:
        need(type(actual) is list and len(actual) == len(expected), where+": list shape")
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, where+f"/{i}", stats)
    elif type(expected) is float:
        need(type(actual) in (int, float) and math.isfinite(actual), where+": finite number required")
        error = abs(actual-expected)
        need(error <= 1e-12 + 1e-10*abs(expected), where+": numerical mismatch")
        stats["compared_float_fields"] += 1
        stats["max_absolute_float_difference"] = max(error, stats["max_absolute_float_difference"])
    else:
        need(exact(actual, expected), where+": exact discrete mismatch")
        stats["compared_discrete_fields"] += 1
    return stats


def revision(value):
    need(type(value) is str and re.fullmatch(r"[0-9a-f]{40}", value), "source revision")
    return value


def audit_manifest(audit, manifest):
    p = bound(manifest); root = p.parent
    need(p.name == "manifest.json" and regular(audit["path"]) == root/"panel-rows.jsonl", "audit manifest scope")
    value = load(p)
    keys(value, {"schema", "files"}, "audit manifest")
    need(value["schema"] == "looped-grounded-policy-artifacts-v1", "audit manifest schema")
    expected = {"launch.json", "metadata.json", "panel-rows.jsonl", "profiles.json", "report.json"}
    keys(value["files"], expected, "audit manifest inventory")
    need({child.name for child in root.iterdir()} == expected | {"manifest.json"}, "audit root inventory differs")
    for name, checksum in value["files"].items():
        bound({"path": str(root/name), "sha256": checksum})
    need(value["files"]["panel-rows.jsonl"] == audit["sha256"], "audit entry differs")
    pin = regular(str(root)+".manifest.sha256")
    need(pin.read_text() == manifest["sha256"]+"\n", "external audit manifest pin differs")
    report = load(root/"report.json")
    expected = {"status": "complete_pending_analysis", "classification": "input_audit", "model_forwards": 0,
                "optimizer_updates": 0, "input_rows": 768, "query_groups": 32, "panel_seed": SEED, "panel_tag": BASE,
                "label_counts": [192]*4}
    need(all(exact(report.get(k), v) for k, v in expected.items()), "audit zero-work/data report identity")
    return report


def premise(config_path):
    started = time.monotonic(); config_sha = file_sha(config_path); config = load(config_path)
    keys(config, {"schema", "audit", "audit_manifest", "history", "registration", "source_revision", "binary_sha256", "frozen_files"}, "premise config")
    need(config["schema"] == "looped-native-binding-premise-v1", "premise schema")
    revision(config["source_revision"]); digest(config["binary_sha256"]); frozen(config["frozen_files"])
    for record in [config[k] for k in ("audit", "audit_manifest", "history", "registration")]:
        path = bound(record)
        need(config["frozen_files"].get(str(path)) == record["sha256"], "premise selected file not frozen")
    need(config["binary_sha256"] in config["frozen_files"].values(), "premise binary not frozen")
    for name in ("independent_review.py", "independent_review_tests.py"):
        need(config["frozen_files"].get(str(R/name)) == file_sha(R/name), "premise reviewer/tests not frozen")
    audit = audit_manifest(config["audit"], config["audit_manifest"])
    root = Path(config["audit_manifest"]["path"]).parent
    for name, checksum in load(config["audit_manifest"]["path"])["files"].items():
        need(config["frozen_files"].get(str(root/name)) == checksum, "audit artifact not frozen")
    pin = str(root)+".manifest.sha256"
    need(config["frozen_files"].get(pin) == file_sha(pin), "audit outer pin not frozen")
    metadata = load(root/"metadata.json")
    expected = {"schema": "looped-native-binding-v1", "device": "cpu", "objective": "public_input_audit",
                "optimizer_updates": 0, "model_forwards": 0}
    need(all(exact(metadata.get(k), v) for k, v in expected.items()), "CPU metadata work/mode")
    provenance = metadata.get("provenance")
    need(type(provenance) is dict and provenance.get("source_revision") == config["source_revision"] and
         provenance.get("binary_sha256") == config["binary_sha256"] and
         provenance.get("candle_graph_revision") == DEPENDENCY, "CPU audit source/binary/dependency identity")
    history, sources = histories(config["history"])
    _, _, panel_summary = panel(config["audit"]["path"], history)
    compare(audit.get("history"), sources, "audit historical sources")
    need(exact(audit.get("excluded_unique_queries"), len(history)), "audit historical union differs")
    frozen(config["frozen_files"])
    need(file_sha(config_path) == config_sha, "premise config drift")
    need(time.monotonic()-started <= 60, "premise exceeds60seconds")
    return {"schema": "looped-native-binding-independent-premise-result-v1", "accepted": True,
            "config_sha256": config_sha, "source_revision": config["source_revision"], "binary_sha256": config["binary_sha256"],
            "registration_sha256": config["registration"]["sha256"], "audit_sha256": config["audit"]["sha256"],
            "audit_manifest_sha256": config["audit_manifest"]["sha256"], "history_sha256": config["history"]["sha256"],
            "reviewer_sha256": file_sha(Path(__file__).resolve()), "panel": panel_summary,
            "history": history_summary(history, sources),
            "elapsed_seconds": time.monotonic()-started,
            "scope": "Independent public geometry, metadata, hashes,32x24 pairing, full six-source history union and direction/action coverage. No model outcomes or neural execution. Source/binary provenance is a frozen external binding."}


def authority(config):
    keys(config, {"schema", "registration", "history", "audit", "checkpoint", "integrity_receipt", "streams", "frozen_files"}, "analysis config")
    need(config["schema"] == "looped-native-binding-analysis-config-v1", "analysis config schema")
    keys(config["streams"], {"factual", "uniform"}, "scientific streams")
    receipt = load(bound(config["integrity_receipt"]))
    need(receipt.get("schema") == "looped-native-binding-integrity-v1" and receipt.get("accepted") is True, "external integrity rejected")
    keys(receipt.get("checks"), CHECKS, "runtime checks")
    need(all(v is True for v in receipt["checks"].values()), "external integrity component failed")
    revision(receipt.get("source_revision")); digest(receipt.get("binary_sha256"))
    expected = {"dependency_revision": DEPENDENCY, "vision_checkpoint_sha256": VISION,
                "selector_sha256": SELECTOR, "binder_checkpoint_sha256": BINDER, "vision_loops": 4, "binder_loops": 4}
    need(all(exact(receipt.get(k), v) for k, v in expected.items()), "registered frozen component identity")
    need(config["checkpoint"]["sha256"] == BINDER, "binder checkpoint differs")
    expected_files = {**receipt["frozen_files"], config["integrity_receipt"]["path"]: config["integrity_receipt"]["sha256"]}
    need(exact(config["frozen_files"], expected_files), "receipt/config frozen closure")
    frozen(config["frozen_files"])
    selected = [config[k] for k in ("registration", "history", "audit", "checkpoint", "integrity_receipt")]
    selected += list(config["streams"].values())
    paths = []
    for record in selected:
        p = bound(record); paths.append(p)
        need(config["frozen_files"].get(str(p)) == record["sha256"], "analysis input not frozen")
    need(len(paths) == len(set(paths)), "distinct analysis inputs required")
    for name in ("analysis.py", "analysis_tests.py", "numpy_binding.py", "numpy_binding_tests.py", "independent_review.py", "independent_review_tests.py"):
        need(config["frozen_files"].get(str(R/name)) == file_sha(R/name), "scorer/reviewer/replay sources and tests not frozen")
    need(VISION in config["frozen_files"].values() and SELECTOR in config["frozen_files"].values(), "vision/selector payloads not frozen")
    return receipt


def history_summary(union, sources):
    return {"files": [{"path": row["path"], "sha256": row["sha256"], "rows": row["parsed_rows"],
                        "hash_occurrences": row["hash_occurrences"], "unique_queries": row["unique_query_hashes"]} for row in sources],
            "source_rows": sum(row["parsed_rows"] for row in sources), "unique_queries": len(union), "overlap_queries": 0}


def controls(rows, truths, result, factual):
    uniform_groups(result)
    expected = [records([[[1/64]*64 for _ in range(2)] for _ in range(7)], t["actions"]) for t in truths]
    need(all(value == 0 for record in expected for r in record for value in r[:2]), "uniform arithmetic displacement")
    reference_logits = [result["logits"][(i//24)*24] for i in range(768)]
    # Independent ideal one-hot arithmetic must recover the visible cardinal records.
    ideal_correct = 0
    for row, t in zip(rows, truths):
        onehot = [[[float(c == t["locations"][f][role]) for c in range(64)] for role in range(2)] for f in range(7)]
        ideal = records(onehot, t["actions"])
        mapping = {}
        for r, action in zip(ideal[:3], t["actions"]):
            need(tuple(r[:2]) in DIRECTIONS, "ideal support movement not cardinal")
            mapping[action] = DIRECTIONS.index(tuple(r[:2]))
        mapping[t["omitted"]] = next(d for d in range(4) if d not in mapping.values())
        desired = DIRECTIONS.index(tuple(ideal[3][:2]))
        answer = next(a for a in range(4) if mapping[a] == desired)
        ideal_correct += int(answer == row["policy_label"])
    need(ideal_correct == 768, "ideal one-hot geometry control failed")
    return {"constant_action_0_correct": sum(r["policy_label"] == 0 for r in rows),
            "always_missing_correct": sum(r["policy_label"] == t["omitted"] for r, t in zip(rows, truths)),
            "ideal_geometry_correct": ideal_correct, "uniform_six_correct_per_query": True,
            "uniform_identical_records_within_query": True, "uniform_identical_winners_within_query": True,
            "uniform_records": parity(result["records"], expected),
            "uniform_logits_within_query": logit_parity(result["logits"], reference_logits),
            "paired_learned_attention": parity(factual["learned"], result["learned"])}


def decision(summaries, control):
    f, u = summaries["factual"], summaries["uniform"]
    gates = {"factual_all_actions": f["correct"] == 768 and f["all_maps_correct_groups"] == 32 and
             f["subsets"]["demonstrated"]["correct"] == 576 and f["subsets"]["omitted"]["correct"] == 192,
             "factual_minimum_true_margin": f["minimum_true_margin"] >= .001,
             "uniform_control": u["correct"] == 192 and control["uniform_identical_records_within_query"] and
             control["uniform_identical_winners_within_query"] and control["uniform_six_correct_per_query"]}
    return gates, "supported_frozen_native_composition" if all(gates.values()) else "frozen_native_composition_not_supported"


def trusted_neural_replay(value):
    expected = {"maximum_absolute_error", "maximum_tolerance_ratio", "rows", "eligible_winners", "ineligible_winners",
                "eligible_winner_mismatches", "eligibility"}
    keys(value, expected, "shared neural replay")
    for key in ("maximum_absolute_error", "maximum_tolerance_ratio"):
        need(type(value[key]) in (int, float) and math.isfinite(value[key]) and value[key] >= 0, "invalid shared replay error")
    need(value["maximum_tolerance_ratio"] <= 1, "shared neural replay tolerance failed")
    for key in ("rows", "eligible_winners", "ineligible_winners", "eligible_winner_mismatches"):
        need(type(value[key]) is int and 0 <= value[key] <= 768, "invalid shared replay coverage")
    need(value["rows"] == 768 and value["eligible_winners"]+value["ineligible_winners"] == 768 and
         value["eligible_winner_mismatches"] == 0, "shared replay winner/row failure")
    need(value["eligibility"] == "reference winner margin > 0 and > 2 * row maximum absolute logit error", "shared replay eligibility differs")
    return value


def reconstruction(config, shared_replay):
    union, sources = histories(config["history"])
    rows, truths, panel_report = panel(config["audit"]["path"], union)
    arrays, summaries, ground, replay = {}, {}, {}, {}
    keys(shared_replay, {"factual", "uniform"}, "replay streams")
    for name, control in (("factual", "factual"), ("uniform", "uniform_attention")):
        result = stream(config["streams"][name]["path"], rows, truths, control); arrays[name] = result
        summaries[name] = action_summary(rows, truths, result["metrics"])
        ground[name] = {which: grounding(truths, result[which]) for which in ("learned", "effective")}
        expected = [records(a, t["actions"]) for a, t in zip(result["effective"], truths)]
        keys(shared_replay[name], {"records", "logits"}, "replay component")
        replay[name] = {"records": parity(result["records"], expected), "logits": trusted_neural_replay(shared_replay[name]["logits"])}
    control = controls(rows, truths, arrays["uniform"], arrays["factual"])
    gates, verdict = decision(summaries, control)
    return {"history": history_summary(union, sources), "panel": panel_report, "summaries": summaries,
            "grounding": ground, "replay": replay, "controls": control, "gates": gates, "decision": verdict}


def review(config_path, report_path):
    started = time.monotonic(); config_sha, report_sha = file_sha(config_path), file_sha(report_path)
    config = load(config_path); receipt = authority(config); report = load(report_path)
    header = {"schema": "looped-native-binding-analysis-v1", "accepted": True, "config_sha256": config_sha,
              "registration_sha256": config["registration"]["sha256"], "source_revision": receipt["source_revision"],
              "binary_sha256": receipt["binary_sha256"], "binder_checkpoint_sha256": BINDER}
    keys(report, set(header) | {"history", "panel", "summaries", "grounding", "replay", "controls", "gates", "decision", "limits", "elapsed_seconds", "pid"}, "primary report")
    compare({k: report[k] for k in header}, header)
    need(type(report["limits"]) is list and report["limits"] and all(type(v) is str and v for v in report["limits"]), "primary scope limits")
    need(type(report["elapsed_seconds"]) in (int, float) and math.isfinite(report["elapsed_seconds"]) and
         0 <= report["elapsed_seconds"] <= 60 and type(report["pid"]) is int and report["pid"] > 0, "primary runtime record")
    expected = reconstruction(config, report["replay"])
    # Neural replay errors are shared evidence. Do not count comparing their
    # values back to themselves as independent numerical verification.
    actual = {k: report[k] for k in expected}
    actual = {**actual, "replay": {name: {"records": v["records"]} for name, v in report["replay"].items()}}
    independent = {**expected, "replay": {name: {"records": v["records"]} for name, v in expected["replay"].items()}}
    stats = compare(actual, independent)
    authority(config)
    need(file_sha(config_path) == config_sha and file_sha(report_path) == report_sha, "config/report drift")
    need(time.monotonic()-started <= 60, "review exceeds60seconds")
    return {"schema": "looped-native-binding-independent-review-v1", "accepted": True, "report_sha256": report_sha,
            "config_sha256": config_sha, "reviewer_sha256": file_sha(Path(__file__).resolve()), **stats,
            "decision": expected["decision"], "gates": expected["gates"], "panel": expected["panel"],
            "shared_neural_replay": {name: v["logits"] for name, v in expected["replay"].items()},
            "shared_replay_source_sha256": file_sha(R/"numpy_binding.py"), "elapsed_seconds": time.monotonic()-started,
            "scope": "Independent complete six-source novelty,32x24 public geometry/metadata/hashes, F32 scalar attention/coordinate/record reconstruction, role and action metrics, controls and exact gates. Neural binder replay error/coverage is typed, range-checked, source-bound shared primary evidence and is excluded from independent comparison counts; no neural forward is rerun. Runtime source/build/device/weights/zero-update/profiler/cleanup evidence shares the pinned external receipt. No population, unprivileged-learning, recurrence or ARC claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--premise", action="store_true")
    args = parser.parse_args()
    need((args.report is None) == args.premise, "--report is required only for full review")
    need(args.output.is_absolute() and args.output.parent == args.output.parent.resolve(strict=True) and
         not args.output.exists() and not args.output.is_symlink(), "new absolute output required")
    def timeout(_signal, _frame):
        raise Invalid("independent CPU stage exceeds60seconds")
    signal.signal(signal.SIGALRM, timeout); signal.alarm(60)
    result = premise(args.config) if args.premise else review(args.config, args.report)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write("\n")
    signal.alarm(0)


if __name__ == "__main__":
    main()
