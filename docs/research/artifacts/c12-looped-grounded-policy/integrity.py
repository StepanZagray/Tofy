#!/usr/bin/env python3
"""C12 terminal execution certificate. No fitting, generation, or metric selection.

The launch specification pins this program, its tests, and the operator modules.
The certificate is deliberately compatible with analyze.py's closed schema.
Recorded cleanup is checked against current process identities; scientific action
and role accuracies remain the independent analyzer's responsibility.
"""
import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import csv
import datetime
import functools
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import re
import signal
import struct
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
FIT = (0, 2, 4, 5, 7, 8, 9, 10, 13, 14, 15, 16, 18, 19, 21, 23)
HELD = (1, 3, 6, 11, 12, 17, 20, 22)
MAPS = {"seen": FIT, "familiar": FIT, "heldout": HELD}
PERMUTATIONS = tuple(itertools.permutations(range(4)))
CORE = "4cd502f7a76fe3dd9729f6693e9d707c670972f7ebe58ad1cbdde37ea2bb2802"
HEAD = "a43ad78fe7a6b1594d23c205fb191c8ac81610deaaff25198cb421d423edc678"
IMPORT = "d2f9a7a99917d180646a5e28e26acd92cbd3fe53919fa009da3e11747e0c158e"
DEPENDENCY = "1ea5cc5c5998a1c2bd388eb158653dd0d1b7452a"
CHECKS = {"data", "source", "gradient", "numerical", "device", "profiler",
          "unused_heads", "oracle", "initialization", "completed_training"}
HEAD_SHAPES = {"queries": (2, 128), "output.weight": (4, 256), "output.bias": (4,)}
IDENTITY = {"schema", "cohort", "condition", "support_cleared", "row_index", "group_index",
            "training_group_index", "original_update", "data_seed", "episode_id", "permutation_id",
            "correct_action", "query_direction", "agent_patch", "goal_patch",
            "observed_support_action_ids", "omitted_action", "correct_action_demonstrated",
            "inferred_controls", "input_sha256", "factual_input_sha256", "cleared_input_sha256",
            "metadata_sha256", "query_sha256", "targets_sha256", "label_sha256"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def object_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key {key}")
        result[key] = value
    return result


def decode(raw):
    def invalid(value):
        raise ValueError(f"nonfinite JSON {value}")
    return json.loads(raw, object_pairs_hook=object_pairs, parse_constant=invalid)


def read(path):
    return decode(Path(path).read_bytes())


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def same(actual, expected, where):
    require(encoded(actual) == encoded(expected), f"{where}: identity differs")


def keys(value, expected, where):
    require(type(value) is dict and set(value) == set(expected), f"{where}: exact keys required")


def integer(value, low, high, where):
    require(type(value) is int and low <= value <= high, f"{where}: invalid integer")
    return value


def number(value, where, positive=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), f"{where}: invalid numeric value")
    return float(value)


def sha(value):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value), "invalid SHA256")
    return value


def digest(path):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and not path.is_symlink(), f"nonregular file {path}")
    before = path.stat()
    signature = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    result = _digest(str(path), signature)
    after = path.stat()
    require(signature == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            f"file changed while hashing {path}")
    return result


@functools.lru_cache(maxsize=None)
def _digest(path, _signature):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_files(files):
    require(type(files) is dict and files, "empty frozen files")
    for path, expected in files.items():
        require(digest(path) == sha(expected), f"file hash differs {path}")


def binding(path):
    return {"path": str(path), "sha256": digest(path)}


def lines(path):
    with Path(path).open("rb") as handle:
        for line in handle:
            require(line.strip(), f"blank JSONL row {path}")
            yield decode(line)


def finite(value, shape, where):
    def numeric(x):
        return all(numeric(v) for v in x) if type(x) is list else type(x) in (int, float)
    require(numeric(value), f"{where}: nonnumeric array")
    array = np.asarray(value, dtype=np.float64)
    require(array.shape == shape and np.isfinite(array).all()
            and np.max(np.abs(array), initial=0) <= np.finfo(np.float32).max, f"{where}: shape/nonfinite")
    return array


def safetensors(path):
    """Minimal strict F32 parser: named tensors, contiguous storage, no pickle."""
    raw = Path(path).read_bytes()
    require(len(raw) >= 8, "truncated safetensors")
    length = struct.unpack("<Q", raw[:8])[0]
    require(2 <= length <= min(16 * 1024 * 1024, len(raw) - 8), "invalid safetensors header")
    table = decode(raw[8:8 + length])
    require(type(table) is dict and table, "empty safetensors")
    payload, result, intervals = raw[8 + length:], {}, []
    for name, item in table.items():
        if name == "__metadata__":
            require(type(item) is dict and all(type(k) is str and type(v) is str for k, v in item.items()),
                    "invalid safetensors metadata")
            continue
        keys(item, {"dtype", "shape", "data_offsets"}, "tensor descriptor")
        require(item["dtype"] == "F32" and type(item["shape"]) is list, "only F32 tensors")
        shape = tuple(integer(d, 1, 100_000_000, "tensor dimension") for d in item["shape"])
        require(type(item["data_offsets"]) is list and len(item["data_offsets"]) == 2, "offset pair")
        start, stop = [integer(x, 0, len(payload), "tensor offset") for x in item["data_offsets"]]
        require(stop - start == math.prod(shape) * 4, "tensor length differs")
        data = payload[start:stop]
        require(np.isfinite(np.frombuffer(data, dtype="<f4")).all(), "nonfinite tensor")
        result[name] = (shape, data)
        intervals.append((start, stop))
    end = 0
    for start, stop in sorted(intervals):
        require(start == end, "tensor overlap or storage gap")
        end = stop
    require(result and end == len(payload), "unclaimed tensor bytes")
    return result


def canonical_head(path):
    raw = Path(path).read_bytes()
    require(len(raw) == 5136 and np.isfinite(np.frombuffer(raw, dtype="<f4")).all(), "canonical import invalid")
    result, offset = {}, 0
    for name, shape in HEAD_SHAPES.items():
        size = 4 * math.prod(shape)
        result[name] = (shape, raw[offset:offset + size])
        offset += size
    return result


def inactive(name):
    return bool(re.fullmatch(r"(?:policy_head|value_head|reward_head|next_head_[0-3])\.(?:weight|bias)", name))


def changes(initial_core, initial_head, final_core, final_head):
    require(set(initial_core) == set(final_core) and set(initial_head) == set(final_head) == set(HEAD_SHAPES),
            "parameter population changed")
    unused = {name for name in initial_core if inactive(name)}
    require(unused == {f"{family}.{part}" for family in
                      ("policy_head", "value_head", "reward_head", *(f"next_head_{i}" for i in range(4)))
                      for part in ("weight", "bias")}, "unused head population differs")
    active = sorted(["core." + n for n in initial_core if n not in unused]
                    + ["spatial_policy." + n for n in initial_head])
    report = dict(active_parameter_names=active, changed_body_names=[], changed_head_names=[],
                  changed_unused_names=[], unused_heads_unchanged=True, all_parameters_unchanged=True)
    norms = dict(body_l2=0., head_l2=0., unused_l2=0., body_changed_tensors=0,
                 head_changed_tensors=0, unused_changed_tensors=0)
    for prefix, old, new in (("core", initial_core, final_core), ("spatial_policy", initial_head, final_head)):
        for name in sorted(old):
            require(old[name][0] == new[name][0], "parameter shape differs")
            if prefix == "spatial_policy":
                require(old[name][0] == HEAD_SHAPES[name], "spatial head shape differs")
            kind = "head" if prefix == "spatial_policy" else "unused" if name in unused else "body"
            if old[name][1] != new[name][1]:
                report[f"changed_{kind}_names"].append(f"{prefix}.{name}")
                report["all_parameters_unchanged"] = False
                norms[f"{kind}_changed_tensors"] += 1
            delta = np.frombuffer(new[name][1], dtype="<f4").astype(np.float64) - np.frombuffer(old[name][1], dtype="<f4")
            norms[f"{kind}_l2"] += float(delta @ delta)
    report["unused_heads_unchanged"] = not report["changed_unused_names"]
    for key in ("body_l2", "head_l2", "unused_l2"):
        norms[key] = math.sqrt(number(norms[key], key))
    return report, norms


def check_updates(path, count, physical, positive=False):
    values, last, elapsed = [], None, -1.
    for index, row in enumerate(lines(path), 1):
        keys(row, {"update", "metrics", "elapsed_seconds"}, "update row")
        same(row["update"], index, "update order")
        require(number(row["elapsed_seconds"], "cumulative update elapsed") >= elapsed, "time went backwards")
        elapsed = row["elapsed_seconds"]
        m = row["metrics"]
        keys(m, {"rows", "physical_batch", "microbatches", "tail_batch", "mean_ce", "pre_update_correct",
                 "pre_clip_norm", "clip_scale", "body_gradient_norm", "head_gradient_norm",
                 "query_gradient_norms", "elapsed_seconds"}, "update metrics")
        for key, expected in dict(rows=64, physical_batch=physical, microbatches=math.ceil(64 / physical),
                                  tail_batch=63 % physical + 1).items():
            same(m[key], expected, key)
        integer(m["pre_update_correct"], 0, 64, "pre-update correct")
        for key in ("mean_ce", "pre_clip_norm", "body_gradient_norm", "head_gradient_norm", "elapsed_seconds"):
            number(m[key], key, positive=positive and key in ("body_gradient_norm", "head_gradient_norm"))
        require(type(m["query_gradient_norms"]) is list and len(m["query_gradient_norms"]) == 2, "query norm shape")
        for value in m["query_gradient_norms"]:
            number(value, "query gradient")
        require(0 < number(m["clip_scale"], "clip scale") <= 1, "clip outside range")
        require(math.isclose(m["pre_clip_norm"], math.hypot(m["body_gradient_norm"], m["head_gradient_norm"]),
                             rel_tol=2e-5, abs_tol=1e-7), "gradient family/global norm mismatch")
        require(math.isclose(m["clip_scale"], min(1., 1 / (m["pre_clip_norm"] + 1e-6)),
                             rel_tol=2e-5, abs_tol=1e-7), "clip does not implement global norm one")
        values.append([m["body_gradient_norm"], m["head_gradient_norm"], *m["query_gradient_norms"]])
        last = m
    require(len(values) == count, "incomplete/excess optimizer updates")
    array = np.asarray(values)
    return last, {name: {"min": float(array[:, i].min()), "max": float(array[:, i].max()),
                        "mean": float(array[:, i].mean()), "nonzero_updates": int((array[:, i] > 0).sum())}
                  for i, name in enumerate(("body", "head", "query_0", "query_1"))}


def process_gone(pid, start=None):
    integer(pid, 1, 2**31 - 1, "process ID")
    path = Path(f"/proc/{pid}/stat")
    if not path.exists():
        return
    # A reused PID is not an owned survivor when the recorded start tick differs.
    observed = path.read_text().rsplit(")", 1)[1].split()[19]
    require(start is not None and str(start) != observed, f"owned process still exists {pid}")


def group_gone(pgid):
    integer(pgid, 1, 2**31 - 1, "process group")
    try:
        os.killpg(pgid, 0)  # Existence probe only; never signals or kills a process.
    except ProcessLookupError:
        return
    raise ValueError(f"owned process group still exists {pgid}")


def cleanup(state, success=True, model=False):
    for name in ("pid_gone", "group_gone") + (("model_pid_gone",) if model else ()):
        require(state.get(name) is True, f"cleanup missing/untyped {name}")
    for name in ("owned_survivors", "group_survivors"):
        same(state.get(name), [], name)
    require(state.get("cleanup_error", "missing") is None, "cleanup error/absent")
    if success:
        same(state.get("returncode"), 0, "return code")
        require(state.get("failure" if model else "error", "missing") is None, "lifecycle failure")
    starts = state.get("owned_process_start_ticks", {})
    require(type(starts) is dict, "process start table")
    for pid in state.get("owned_pids", []):
        process_gone(pid, starts.get(str(pid)))
    if model and state.get("model_pid") is not None:
        process_gone(state["model_pid"], starts.get(str(state["model_pid"])))


def operation(path, success=True):
    state = read(path)
    cleanup(state, success=success)
    record = read(path.with_name(path.name.replace(".exit.json", ".process.json")))
    process_gone(record["pid"], record["process_start_ticks"])
    if success and "accepted" in state:
        require(state["accepted"] is True, "outer operation rejected")
    return state


def timestamp(value):
    result = datetime.datetime.fromisoformat(value)
    require(result.tzinfo is not None, "unqualified recorded time")
    return result


def typed_checks(checks):
    keys(checks, CHECKS, "certificate gates")
    require(all(value is True for value in checks.values()), "certificate requires literal true gates")


def profile(root, captures, source):
    bound = root.with_suffix(".bound")
    summaries = read(bound / "summary.json")
    traces = sorted(root.with_suffix(".profiles").glob("*/trace.jsonl"))
    raw = sorted(root.with_suffix(".nsight").glob("*.nsys-rep"))
    require(len(summaries) == len(traces) == len(raw) == captures, "profiler population differs")
    expected_steps = (2, 100, 1150) if captures == 3 else (1,)
    require({t.parent.name for t in traces} == {f"update-{step:012}" for step in expected_steps},
            "registered capture updates differ")
    require({s["trace"] for s in summaries} == {str(t) for t in traces}, "profile trace coverage")
    raw_seen, correlations = set(), set()
    for item in summaries:
        for field in ("structurally_valid", "capture_complete"):
            require(item["health"].get(field) is True, "profile health failed")
        require(item["raw_application_labels_verified"] is True, "raw Nsight label verification absent")
        require(item["gpu"]["status"] == "available" and item["gpu"]["provenance_binding"] == "bound", "GPU evidence not bound")
        trace = Path(item["trace"])
        with trace.open("rb") as handle:
            meta = decode(handle.readline())
        correlation = meta["correlation_id"]
        require(correlation not in correlations, "duplicate profile correlation")
        correlations.add(correlation)
        manual_path = Path(item["manual_check"])
        require(manual_path.is_relative_to(bound) and Path(item["bundle"]).is_relative_to(bound), "bound profile escapes root")
        manual = read(manual_path)
        same(manual["trace"], str(trace), "manual trace")
        raw_path = Path(manual["raw_report"])
        require(raw_path in raw and raw_path not in raw_seen, "raw Nsight report reused")
        raw_seen.add(raw_path)
        expected = meta["capture_contract"]["gpu_expected_semantic_labels"]
        require(expected and len(set(expected)) == len(expected), "empty/duplicate GPU labels")
        same(manual["application_labels_each_once"], {label: 1 for label in expected}, "application labels")
        index = integer(int(manual_path.parent.name.removeprefix("export-")), 0, captures - 1, "capture index")
        attachment = bound / f"attachment-{index}"
        manifest = read(attachment / "capture-manifest.json")
        same(manifest["source_revisions"], {"tofy": source["revision"], "candle_graph": DEPENDENCY}, "Nsight source")
        same(manifest["correlation"], {"id": correlation}, "Nsight correlation")
        same(manifest["gpu_expected_semantic_labels"], expected, "Nsight GPU contract")
        require(digest(raw_path) == digest(attachment / "capture.nsys-rep"), "raw Nsight copy differs")
        artifacts = manifest["artifacts"]
        require({a["path"] for a in artifacts} == {p.name for p in attachment.iterdir() if p.name != "capture-manifest.json"},
                "Nsight inventory differs")
        for artifact in artifacts:
            path = attachment / artifact["path"]
            require(path.parent == attachment and path.stat().st_size == artifact["size_bytes"]
                    and digest(path) == sha(artifact["sha256"]), "Nsight artifact differs")
        with (attachment / "stats_nvtx_gpu_proj_trace.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        application = [r for r in rows if r["Name"].startswith(":tofy.looped/")]
        require(sorted(r["Name"][1:] for r in application) == sorted(expected)
                and all(int(r["NumGPUOps"]) > 0 for r in application), "Nsight executed GPU labels differ")
        for name in ("export", "publication", "verification", "overview", "gpu-correlation"):
            operation(manual_path.parent / f"{name}.exit.json")


def operators(spec):
    required = ("integrity.py", "integrity_tests.py", "driver.py", "supervise.py", "bind_nsight.py", "analyze.py", "registration.md")
    require(all(str(HERE / name) in spec["frozen_files"] for name in required), "operator source not frozen")
    verify_files(spec["frozen_files"])
    loaded = {}
    for name in ("supervise", "driver"):
        module_spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[name] = module
        module_spec.loader.exec_module(module)
        # Identical hash semantics with per-invocation stat-guarded memoization.
        module.digest, module.verify_files, module.read = digest, verify_files, read
        loaded[name] = module
    return loaded["driver"], loaded["supervise"]


def audit_identity(row, cohort, condition, index):
    keys(row, IDENTITY | ({"query_cells"} if cohort != "training" else set()), "audit identity")
    maps = HELD if cohort == "heldout" else FIT
    group, slot = divmod(index, len(maps))
    original = group if cohort == "training" else group * 4599 // 63 if cohort == "seen" else None
    training = original is not None
    expected = dict(schema="looped-grounded-policy-data-v1", cohort=cohort, condition=condition,
                    support_cleared=condition == "cleared", row_index=index, group_index=group,
                    training_group_index=original, original_update=original // 4 + 1 if training else None,
                    data_seed=20260920 if training else 20260921,
                    episode_id=(0x47524F554E445452 + original) if training else 0x47524F554E444556 + group,
                    permutation_id=maps[slot])
    for name, value in expected.items():
        same(row[name], value, name)
    for name in IDENTITY:
        if name.endswith("_sha256"):
            sha(row[name])
    controls = list(PERMUTATIONS[maps[slot]])
    same(row["inferred_controls"], controls, "inferred controls")
    agent = integer(row["agent_patch"], 0, 63, "agent")
    goal = integer(row["goal_patch"], 0, 63, "goal")
    deltas = ((0, -1), (0, 1), (-1, 0), (1, 0))
    delta = goal % 8 - agent % 8, goal // 8 - agent // 8
    require(delta in deltas, "query is not one step")
    direction = deltas.index(delta)
    same(row["query_direction"], direction, "visible direction")
    action = controls.index(direction)
    same(row["correct_action"], action, "oracle label")
    actions = row["observed_support_action_ids"]
    require(type(actions) is list and len(actions) == 3, "support action count")
    require(len({integer(a, 0, 3, "support action") for a in actions}) == 3, "support action uniqueness")
    same(row["omitted_action"], next(a for a in range(4) if a not in actions), "omitted action")
    same(row["correct_action_demonstrated"], action in actions, "demonstrated flag")
    same(row["label_sha256"], hashlib.sha256(struct.pack("<I", action)).hexdigest(), "label hash")
    same(row["input_sha256"], row[f"{condition}_input_sha256"], "condition input hash")
    if cohort != "training":
        cells = row["query_cells"]
        require(type(cells) is list and len(cells) == 64 and all(type(x) is int and 0 <= x <= 3 for x in cells), "visible cells")
        require(cells.count(2) == cells.count(3) == 1 and cells[agent] == 2 and cells[goal] == 3, "visible role oracle")
        require(all(cells[i] == 1 for i in range(64) if i % 8 in (0, 7) or i // 8 in (0, 7)), "boundary walls")
        query = np.repeat(np.asarray(cells, dtype="<u4"), 64).tobytes()
        same(row["query_sha256"], hashlib.sha256(query).hexdigest(), "visible query hash")


def check_groups(rows, maps):
    require(len(rows) % len(maps) == 0, "incomplete map group")
    for offset in range(0, len(rows), len(maps)):
        group = rows[offset:offset + len(maps)]
        first = group[0]
        for row in group:
            for key in ("query_sha256", "metadata_sha256", "observed_support_action_ids", "omitted_action",
                        "cleared_input_sha256", "agent_patch", "goal_patch"):
                same(row[key], first[key], f"within-group {key}")
        require(len({r["factual_input_sha256"] for r in group}) == len(maps), "duplicate factual tuples")
        same([sum(r["correct_action"] == a for r in group) for a in range(4)], [len(maps) // 4] * 4, "group action balance")
        require(sum(not r["correct_action_demonstrated"] for r in group) == len(maps) // 4, "omitted subset balance")


def population(campaign, spec, report):
    audit = report["audit"]
    for key, value in dict(schema="looped-grounded-policy-data-v1", status="complete_pending_analysis",
                           evidence_class="data_audit", optimizer_updates=0, model_forwards=0,
                           training_groups=4600, training_rows=73600, training_action_counts=[18400] * 4,
                           training_map_counts=[4600 if i in FIT else 0 for i in range(24)],
                           seen_input_target_label_parity=True, historical_selected_unique_queries=1280,
                           new_unique_queries=64, new_query_overlap=0).items():
        same(audit[key], value, f"data report {key}")
    require(number(audit["elapsed_seconds"], "audit elapsed", positive=True) <= 120, "audit runtime")
    generation = number(audit["training_generation_and_audit_seconds"], "generation elapsed", positive=True)
    require(math.isclose(number(audit["training_rows_per_second"], "generation rate", positive=True), 73600 / generation, rel_tol=1e-12), "generation rate mismatch")
    paths = {"training": campaign / "audit/training-audit.jsonl"}
    paths.update({f"{cohort}/{condition}": campaign / f"audit/{cohort}-{condition}-audit.jsonl"
                  for cohort in MAPS for condition in ("factual", "cleared")})
    artifacts = audit["artifacts"]
    require(type(artifacts) is list and len(artifacts) == 7 and {a["path"] for a in artifacts} == {str(p) for p in paths.values()}, "seven audit artifacts required")
    for artifact in artifacts:
        path = Path(artifact["path"])
        rows = 73600 if path == paths["training"] else 512 if "heldout" in path.name else 1024
        same(artifact, dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size, rows=rows), "audit file binding")
    train_queries, train_inputs, selected = set(), set(), {}
    selected_groups = {i * 4599 // 63 for i in range(64)}
    group, total = [], 0
    for index, row in enumerate(lines(paths["training"])):
        audit_identity(row, "training", "factual", index)
        group.append(row)
        train_queries.add(row["query_sha256"])
        train_inputs.add(row["input_sha256"])
        if row["group_index"] in selected_groups:
            selected[(row["group_index"], row["permutation_id"])] = row
        if len(group) == 16:
            check_groups(group, FIT)
            group.clear()
        total += 1
    require(total == 73600 and not group, "training population count")
    for key, value in dict(training_unique_query_images=len(train_queries), training_duplicate_query_groups=4600-len(train_queries),
                           training_unique_input_tuples=len(train_inputs)).items():
        same(audit[key], value, key)
    history, history_reports = set(), []
    require(type(spec["history"]) is list and len(spec["history"]) == 4, "historical source population")
    history_hashes = ("09559eb4dd1b933de724da81aef3bb80547e010a00e38e39de31c9f72fab1c6d",
                      "bd72fd9707759dd19074a22b9d62899419b28711722b3617c21cc277a6a25c0d",
                      "09a7aa423b17f32bbae14ae9178a15d5fff8428936afdf23ebc4f09878a6ddca",
                      "61ae87a267cc545b05884dc412160c0313a0c2bc1015cfa2323bde256f91eaeb")
    for index, source in enumerate(spec["history"]):
        keys(source, {"path", "sha256", "first_rows"}, "history source")
        same(source["first_rows"], 512 if index == 0 else None, "history selection")
        require(source["sha256"] == history_hashes[index] == spec["frozen_files"].get(source["path"]), "history source not frozen")
        rows = list(lines(source["path"]))
        require(len(rows) == (768 if index == 0 else 256), "history rows")
        subset = rows[:512] if index == 0 else rows
        queries = {sha(r["query_sha256"]) for r in subset}
        require(len(queries) == len(subset), "duplicate historical selected queries")
        history.update(queries)
        history_reports.append(dict(path=source["path"], sha256=digest(source["path"]), bytes=Path(source["path"]).stat().st_size,
                                    rows=len(rows), first_rows=source["first_rows"], selected_rows=len(subset), selected_unique_queries=len(queries)))
    require(len(history) == 1280, "historical union count")
    same(audit["histories"], history_reports, "history report")
    audits = {}
    for cohort, maps in MAPS.items():
        for condition in ("factual", "cleared"):
            selector = f"{cohort}/{condition}"
            rows = list(lines(paths[selector]))
            require(len(rows) == 64 * len(maps), "evaluation audit count")
            for index, row in enumerate(rows):
                audit_identity(row, cohort, condition, index)
            check_groups(rows, maps)
            audits[selector] = rows
        factual, cleared = audits[f"{cohort}/factual"], audits[f"{cohort}/cleared"]
        for row, clear in zip(factual, cleared):
            expected = dict(row, condition="cleared", support_cleared=True, input_sha256=row["cleared_input_sha256"])
            same(clear, expected, "factual/cleared identity")
            if cohort == "seen":
                old = selected[(row["training_group_index"], row["permutation_id"])]
                for key in IDENTITY - {"cohort", "row_index", "group_index"}:
                    same(row[key], old[key], "seen training identity")
        summary = audit["cohorts"][cohort]
        expected = dict(groups=64, rows_per_condition=len(factual), map_ids=list(maps),
                        map_counts=[64 if i in maps else 0 for i in range(24)],
                        unique_query_images=len({r["query_sha256"] for r in factual}), action_counts=[len(factual)//4]*4,
                        omitted_action_rows=len(factual)//4, demonstrated_action_rows=3*len(factual)//4,
                        oracle_accuracy=1.0, constant_or_action_id_only_accuracy=0.25, always_omitted_accuracy=0.25,
                        always_omitted_on_omitted_accuracy=1.0, always_omitted_on_demonstrated_accuracy=0.0,
                        best_action_id_only_on_demonstrated_accuracy=1.0/3.0, cleared_inputs_identical_within_group=True)
        same(summary, expected, "cohort audit controls")
    familiar = audits["familiar/factual"][::16]
    heldout = audits["heldout/factual"][::8]
    for f, h in zip(familiar, heldout):
        for key in ("query_sha256", "query_cells", "metadata_sha256", "observed_support_action_ids", "cleared_input_sha256"):
            same(f[key], h[key], "familiar/heldout pairing")
    unseen = {r["query_sha256"] for r in familiar}
    require(len(unseen) == 64 and not unseen & (train_queries | history), "new query collision: no replacement allowed")
    require(digest(paths["training"]) == digest(campaign / "train-seed0/training-stream.jsonl"), "consumed training stream differs")
    return {key: binding(path) for key, path in paths.items() if key != "training"}, audits


def verify_parity(campaign, driver, spec):
    driver.verify_reference_bindings(spec)
    references = {"logits": (driver.C11 / "qual-initial-c10_true/logits.f32", (512, 4)),
                  "attention": (driver.C11 / "qual-initial-c10_true/attention.f32", (512, 2, 64)),
                  "pooled": (driver.C11 / "qual-initial-c10_true/pooled.f32", (512, 256)),
                  "current": (driver.C8 / "features-initial/known-features-current.f32", (768, 64, 128)),
                  "cls": (driver.C8 / "features-initial/known-features-cls.f32", (768, 128))}
    checks = {}
    for batch in (1, 4):
        rows = list(lines(campaign / f"parity-b{batch}/qualification-rows.jsonl"))
        same([r["index"] for r in rows], list(range(4)), "parity row order")
        for name, (path, shape) in references.items():
            require(path.stat().st_size == math.prod(shape) * 4, "reference byte layout")
            ref = np.fromfile(path, dtype="<f4", count=4 * math.prod(shape[1:])).reshape((4, *shape[1:]))
            require(np.isfinite(ref).all(), "nonfinite reference")
            actual = finite([r[name] for r in rows], ref.shape, name)
            atol = 1e-5 if name == "attention" else 1e-4
            error = np.abs(actual - ref)
            require(np.all(error <= atol + 1e-5 * np.abs(ref)), "registered warm-start parity failed")
            if name in ("logits", "attention"):
                require(np.array_equal(actual.argmax(axis=-1), ref.argmax(axis=-1)), "parity argmax differs")
            checks[f"{batch}/{name}"] = dict(max_absolute_error=float(error.max()), atol=atol, rtol=1e-5)
    report = read(campaign / "parity-analysis.json")
    require(report["accepted"] is True, "parity not accepted")
    same(report["checks"], checks, "retained parity recomputation")


def check_weight_report(root, report, initial_core, initial_head, restore):
    core_path, head_path = root / "final-core.safetensors", root / "final-head.safetensors"
    require(digest(core_path) == sha(report["final_core_sha256"]) and digest(head_path) == sha(report["final_head_sha256"]), "saved final checkpoints differ")
    actual, numeric = changes(initial_core, initial_head, safetensors(core_path), safetensors(head_path))
    same(report["changes"], actual, "raw parameter changes")
    require(actual["unused_heads_unchanged"] and actual["changed_body_names"] and actual["changed_head_names"], "inactive heads changed or active family unchanged")
    if restore:
        core, head = root / "restored-core.safetensors", root / "restored-head.safetensors"
        require(digest(core) == sha(report["restored_core_sha256"]) and digest(head) == sha(report["restored_head_sha256"]), "restored checkpoint hashes")
        restored, _ = changes(initial_core, initial_head, safetensors(core), safetensors(head))
        require(restored["all_parameters_unchanged"] is True, "restoration changed named F32 bytes")
        same(report["restored_changes"], restored, "restoration report")
    else:
        for key in ("restored_changes", "restored_core_sha256", "restored_head_sha256"):
            require(report[key] is None, "training cannot restore initialization")
    return numeric


def invocation(spec, name, mode, updates, physical, source, operator, cohort=None, cleared=False):
    campaign, root = Path(spec["campaign"]), Path(spec["campaign"]) / name
    cfg_path = campaign / f"invocations/{name}.json"
    auth_path = campaign / f"invocations/{name}-authority.json"
    cfg, auth, state = read(cfg_path), read(auth_path), read(root.with_suffix(".exit.json"))
    require(state["accepted"] is True and state["bindings_unchanged"] is True, "invocation not accepted")
    cleanup(state, model=True)
    proc = read(root.with_suffix(".process.json"))
    process_gone(proc["pid"], state.get("owned_process_start_ticks", {}).get(str(proc["pid"])))
    group_gone(proc["pgid"])
    for key, value in dict(schema="looped-grounded-policy-config-v1", source_revision=source["revision"],
                           registration=str(HERE / "registration.md"), registration_sha256=spec["registration_sha256"],
                           mode=mode, output_dir=str(root), updates=updates, physical_batch=physical,
                           cohort=cohort, cleared=cleared).items():
        same(cfg[key], value, f"invocation {name}/{key}")
    integer(cfg["max_seconds"], 1, 3600 if mode == "train" else 120 if mode in ("audit", "qualify") else 600, "invocation deadline")
    for key, value in dict(schema="looped-grounded-policy-launch-v1", accepted=True, config_sha256=digest(cfg_path),
                           mode=mode, campaign=str(campaign), name=name, binary=str(campaign / "grounded_policy_probe"),
                           binary_sha256=source["binary_sha256"], repository=spec["repository"], source=source["revision"]).items():
        same(auth[key], value, f"invocation authority {key}")
    require(all(auth["frozen_files"].get(p) == value for p, value in spec["frozen_files"].items()), "launch omitted source freeze")
    verify_files(auth["frozen_files"])
    same(proc["config_sha256"], digest(cfg_path), "process config")
    same(proc["authority_sha256"], digest(auth_path), "process authority")
    same(proc["binary_sha256"], source["binary_sha256"], "process binary")
    same(proc["config"], str(cfg_path), "process config path")
    for key, value in dict(TOFY_PERF_TRACE=str(root.with_suffix(".host.json")), NSYS_NVTX_PROFILER_REGISTER_ONLY="0",
                           NVIDIA_TF32_OVERRIDE="0", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1").items():
        same(proc["environment"][key], value, "invocation environment")
    require(timestamp(auth["created_local"]) <= timestamp(proc["started_local"]) <= timestamp(state["finished_local"]), "invocation chronology")
    outer = operation(campaign / f"operations/{name}-supervisor.exit.json")
    require(outer["accepted"] is True, "supervisor outer failure")
    manifest_sha, profiles, host = operator.root_manifest(root)
    require(manifest_sha == state["manifest_sha256"], "root exit pin differs")
    report, metadata = read(root / "report.json"), read(root / "metadata.json")
    same(metadata["config"], cfg, "model invocation metadata")
    provenance = metadata["provenance"]
    for key, value in (("source_revision", source["revision"]), ("binary_sha256", source["binary_sha256"]),
                       ("candle_graph_revision", DEPENDENCY)):
        same(provenance[key], value, "model provenance")
    for key, value in dict(loops=4, hidden=128, layers=2, heads=4, max_loops=8, effective_batch=64,
                           seed=0, objective="policy_cross_entropy_only", privileged_role_warm_start=True).items():
        same(metadata[key], value, "registered numerical configuration")
    same(report["status"], "complete_pending_analysis", "report completion")
    same(report["optimizer_updates"], updates, "report update count")
    expected_report = dict(status="complete_pending_analysis", optimizer_updates=updates)
    if mode != "audit":
        expected_report["physical_batch"] = physical
        same(report["physical_batch"], physical, "report batch")
    same(auth["expected_report"], expected_report, "expected report authority")
    require(number(report["elapsed_seconds"], "reported elapsed") <= cfg["max_seconds"], "reported deadline exceeded")
    same(state["reported_model_elapsed_seconds"], report["elapsed_seconds"], "lifecycle reported elapsed")
    require(number(state["model_phase_seconds"], "model phase") <= cfg["max_seconds"], "model phase deadline")
    require(number(state["finalization_seconds"], "finalization") <= 120, "finalization deadline")
    captures = 0 if mode == "audit" else 3 if mode == "train" else 1
    same(state["captures_expected"], captures, "capture budget")
    if mode == "audit":
        same(report["model_forwards"], 0, "audit forwards")
        same(host, [], "audit host trace")
        same(profiles["files"], {}, "audit profiles")
        require(not root.with_suffix(".nsight").exists() and not root.with_suffix(".bound").exists(), "audit GPU artifacts")
        same(cfg["history"], spec["history"], "audit history")
    else:
        require(type(host) is list and host and profiles["files"], "GPU profile work absent")
        require(number(state["headroom_mib"], "GPU reserve") >= 512
                and number(state["max_temperature_c"], "GPU temperature") < 85, "device limits violated")
        require("--trace=cuda,nvtx,osrt,cudnn,cublas" in proc["command"] and "--sample=process-tree" in proc["command"], "required device profiling missing")
        operation(campaign / f"operations/{name}-bind.exit.json")
        profile(root, captures, source)
        same(cfg["history"], [], "non-audit history forbidden")
    require(digest(cfg["core_checkpoint"]) == cfg["core_sha256"] and digest(cfg["head_checkpoint"]) == cfg["head_sha256"], "invocation checkpoint differs")
    require(digest(cfg["import_manifest"]) == IMPORT, "warm-start import differs")
    for key in ("core_checkpoint", "head_checkpoint"):
        require(auth["frozen_files"].get(cfg[key]) == digest(cfg[key]), "launch checkpoint not frozen")
    if mode != "eval_final":
        same((cfg["core_sha256"], cfg["head_sha256"]), (CORE, HEAD), "registered initial checkpoint")
    if mode in ("train", "eval_initial", "eval_final"):
        same(cfg["audit_root"], str(campaign / "audit"), "audit root")
        same(cfg["audit_manifest_sha256"], digest(campaign / "audit/manifest.json"), "audit root pin")
    else:
        require(cfg["audit_root"] is None and cfg["audit_manifest_sha256"] is None, "unexpected audit input")
    return cfg, report, state


def capacity_search(campaign, qualification, spec, supervisor):
    tests = qualification["tests"]
    require(type(tests) is dict and tests and all(type(v) is bool for v in tests.values()), "capacity test types")
    low, high, batch, visited, passed = 0, 65, 64, set(), []
    while True:
        require(str(batch) in tests and batch not in visited, "capacity search path missing/repeated")
        visited.add(batch)
        name = f"batch-{batch}-u2"
        state = read(campaign / f"{name}.exit.json")
        same(state["accepted"], tests[str(batch)], "capacity observation")
        if tests[str(batch)]:
            low = max(low, batch)
            passed.append(name)
        else:
            cleanup(state, success=False, model=True)
            operation(campaign / f"operations/{name}-supervisor.exit.json", success=False)
            require(state["capacity_failure"] is True and "recovery_error" not in state
                    and supervisor.capacity_only(state, (campaign / f"{name}.stdout.log").read_text()), "invalid capacity exclusion")
            cfg = read(campaign / f"invocations/{name}.json")
            for key, value in dict(mode="batch_smoke", physical_batch=batch, updates=2,
                                   core_sha256=CORE, head_sha256=HEAD, source_revision=spec["source"]).items():
                same(cfg[key], value, "failed capacity trial configuration")
            auth_path = campaign / f"invocations/{name}-authority.json"
            auth, proc = read(auth_path), read(campaign / f"{name}.process.json")
            require(proc["authority_sha256"] == digest(auth_path) and auth["config_sha256"] == digest(campaign / f"invocations/{name}.json"), "failed capacity authority")
            verify_files(auth["frozen_files"])
            process_gone(proc["pid"], state.get("owned_process_start_ticks", {}).get(str(proc["pid"])))
            group_gone(proc["pgid"])
            high = min(high, batch)
        if high - low <= 1:
            break
        batch = (low + high) // 2
    require(low > 0 and {str(x) for x in visited} == set(tests), "incomplete/extraneous capacity search")
    same(qualification["physical_batch"], low, "selected physical batch")
    same(qualification["upper_excluded"], high if high <= 64 else None, "adjacent excluded capacity")
    same(qualification["accumulation"], math.ceil(64 / low), "selected accumulation")
    same(qualification["effective_batch"], 64, "selected effective batch")
    return ["parity-b1", "parity-b4", *passed, f"confirm-{low}-u5"], visited


def build(spec, driver, supervisor):
    campaign = Path(spec["campaign"])
    require(campaign.is_absolute() and campaign.resolve() == campaign and campaign.is_dir(), "canonical campaign required")
    require(re.fullmatch(r"[0-9a-f]{40}", spec["source"]), "invalid source revision")
    source = dict(revision=spec["source"], binary_sha256=sha(spec["binary_sha256"]), dependency_revision=DEPENDENCY)
    require(digest(campaign / "grounded_policy_probe") == source["binary_sha256"], "binary differs")
    require(spec["initial_core_sha256"] == CORE and spec["initial_head_sha256"] == HEAD, "initialization spec differs")
    require(digest(HERE / "registration.md") == sha(spec["registration_sha256"]), "registration differs")
    stages = {name: driver.completed_stage(spec, name) for name in ("audit", "qualification", "frozen", "train", "final")}
    qualification = read(campaign / "qualification-analysis.json")
    require(qualification["accepted"] is True, "qualification not accepted")
    qualification_names, tested = capacity_search(campaign, qualification, spec, supervisor)
    expected_stages = dict(audit=["audit"], qualification=qualification_names,
                           frozen=[f"frozen-{c}" for c in MAPS], train=["train-seed0"],
                           final=[f"final-{c}-{q}" for c in MAPS for q in ("factual", "cleared")])
    previous = None
    for name, expected in expected_stages.items():
        stage = stages[name]
        same(stage["names"], expected, "closed stage population")
        at = timestamp(stage["created_local"])
        require(previous is None or previous <= at, "stage order differs")
        previous = at
        # A stage must bind every current file in its finalized Nsight bundles.
        for invocation_name in expected:
            for path in (campaign / f"{invocation_name}.bound").rglob("*"):
                if path.is_file():
                    require(stage["frozen_files"].get(str(path)) == digest(path), "stage omitted bound profile artifact")
    physical = integer(qualification["physical_batch"], 1, 64, "selected batch")
    cases = [("audit", "audit", 0, 1, None, False), ("parity-b1", "qualify", 0, 1, None, False),
             ("parity-b4", "qualify", 0, 4, None, False)]
    cases += [(f"batch-{b}-u2", "batch_smoke", 2, b, None, False) for b in tested if qualification["tests"][str(b)]]
    cases += [(f"confirm-{physical}-u5", "batch_smoke", 5, physical, None, False)]
    cases += [(f"frozen-{c}", "eval_initial", 0, physical, c, False) for c in MAPS]
    cases += [("train-seed0", "train", 1150, physical, None, False)]
    cases += [(f"final-{c}-{q}", "eval_final", 0, physical, c, q == "cleared") for c in MAPS for q in ("factual", "cleared")]
    expected_names = {x[0] for x in cases} | {f"batch-{b}-u2" for b in tested}
    require({p.name[:-10] for p in campaign.glob("*.exit.json")} == expected_names, "missing/extra invocation roots")
    require({p.stem for p in (campaign / "invocations").glob("*.json") if not p.stem.endswith("-authority")} == expected_names, "invocation configuration population")
    evidence = {}
    for name, mode, updates, batch, cohort, cleared in cases:
        evidence[name] = invocation(spec, name, mode, updates, batch, source, supervisor, cohort, cleared)
    # Every stage's model work begins after its predecessor seal; audit precedes clock.
    order = ("audit", "qualification", "frozen", "train", "final")
    for previous_name, current in zip(order, order[1:]):
        cutoff = timestamp(stages[previous_name]["created_local"])
        for name in expected_stages[current]:
            require(timestamp(read(campaign / f"{name}.process.json")["started_local"]) >= cutoff, "stage crossed predecessor barrier")
    initial_cfg = evidence["parity-b1"][0]
    initial_core, initial_head = safetensors(initial_cfg["core_checkpoint"]), canonical_head(initial_cfg["head_checkpoint"])
    audit_report = evidence["audit"][1]
    audits, identities = population(campaign, spec, audit_report)
    verify_parity(campaign, driver, spec)
    gradient_norms, parameter_changes = {}, {}
    for name, mode, count, batch, _, _ in cases:
        if mode not in ("train", "batch_smoke"):
            continue
        report = evidence[name][1]
        for key, expected in dict(effective_batch=64, accumulation=math.ceil(64 / batch), input_rows=count * 64).items():
            same(report[key], expected, key)
        last, norms = check_updates(campaign / name / "updates.jsonl", count, batch, positive=mode == "batch_smoke")
        same(report["last_update"], last, "terminal update metrics")
        number(report["updates_elapsed_seconds"], "all updates elapsed", positive=True)
        number(report["checkpoint_seconds"], "checkpoint elapsed")
        gradient_norms[name] = norms
        parameter_changes[name] = check_weight_report(campaign / name, report, initial_core, initial_head, mode == "batch_smoke")
    admitted = driver.admission(evidence[f"confirm-{physical}-u5"][1], audit_report["audit"])
    require(admitted["accepted"] is True, "runtime admission fails on recomputation")
    for key, value in admitted.items():
        same(qualification[key], value, "admission result")
    totals = {"qualification": 0., "training": 0., "evaluation": 0.}
    for name in sorted(expected_names):
        state = read(campaign / f"{name}.exit.json")
        mode = read(campaign / f"invocations/{name}.json")["mode"]
        key = "qualification" if mode in ("qualify", "batch_smoke") else "training" if mode == "train" else "evaluation" if mode.startswith("eval_") else None
        if key:
            totals[key] += number(state["model_phase_seconds"], "cumulative model budget")
    for name, ceiling in (("qualification", 600), ("training", 3600), ("evaluation", 600)):
        require(totals[name] <= ceiling, f"{name} budget exceeded")
        require(math.isclose(number(stages["final"]["budgets"][name], "recorded phase budget"), totals[name],
                             rel_tol=1e-12, abs_tol=1e-8), "final recorded phase budget differs")
    clock = read(campaign / "clock.json")
    require(clock["source"] == spec["source"], "clock source")
    wall = (timestamp(stages["final"]["created_local"]) - timestamp(clock["started_local"])).total_seconds()
    require(0 <= wall <= 5400 and abs(wall - number(stages["final"]["budgets"]["wall"], "recorded wall")) < 1,
            "campaign wall budget")
    train = evidence["train-seed0"][1]
    checkpoints = dict(frozen=dict(core_sha256=CORE, policy_sha256=HEAD, updates=0),
                       final=dict(core_sha256=train["final_core_sha256"], policy_sha256=train["final_head_sha256"], updates=1150))
    streams = {}
    for stage in ("frozen_factual", "final_factual", "final_cleared"):
        checkpointstage = "frozen" if stage == "frozen_factual" else "final"
        condition = "cleared" if stage == "final_cleared" else "factual"
        for cohort in MAPS:
            name = f"frozen-{cohort}" if checkpointstage == "frozen" else f"final-{cohort}-{condition}"
            cfg, report, _ = evidence[name]
            checkpoint = checkpoints[checkpointstage]
            same((cfg["core_sha256"], cfg["head_sha256"]), (checkpoint["core_sha256"], checkpoint["policy_sha256"]), "evaluation selected checkpoints")
            if checkpointstage == "final":
                same((cfg["core_checkpoint"], cfg["head_checkpoint"]),
                     (str(campaign / "train-seed0/final-core.safetensors"), str(campaign / "train-seed0/final-head.safetensors")), "terminal checkpoint paths")
            expected = identities[f"{cohort}/{condition}"]
            same(report["input_rows"], len(expected), "evaluation report rows")
            same(report["cohort"], cohort, "evaluation report cohort")
            same(report["cleared"], condition == "cleared", "evaluation report condition")
            path = campaign / name / "evaluation-rows.jsonl"
            actual = list(lines(path))
            require(len(actual) == len(expected), "evaluation stream count")
            for row, identity in zip(actual, expected):
                keys(row, set(identity) | {"checkpointstage", "logits", "attention", "pooled"}, "evaluation output")
                same({key: row[key] for key in identity}, identity, "evaluation audited row")
                same(row["checkpointstage"], checkpointstage, "evaluation checkpoint marker")
                finite(row["logits"], (4,), "policy logits")
                attention = finite(row["attention"], (2, 64), "attention")
                require(((attention >= 0) & (attention <= 1)).all()
                        and np.allclose(attention.sum(axis=1), 1, atol=1e-5, rtol=1e-5), "attention probabilities")
                finite(row["pooled"], (256,), "pooled features")
            streams[f"{stage}/{cohort}"] = dict(**binding(path), cohort=cohort, condition=condition,
                checkpointstage=checkpointstage, source=source, checkpoint=checkpoint,
                audit_sha256=audits[f"{cohort}/{condition}"]["sha256"])
    checks = {name: True for name in CHECKS}
    typed_checks(checks)
    certificate = dict(schema="looped-grounded-policy-integrity-v1", accepted=True, campaign=str(campaign),
                       source=source, registration_sha256=spec["registration_sha256"], checkpoints=checkpoints,
                       audit_sha256={k: v["sha256"] for k, v in audits.items()},
                       stream_sha256={k: v["sha256"] for k, v in streams.items()}, checks=checks,
                       parameter_changes=parameter_changes, gradient_norms=gradient_norms)
    config = dict(schema="looped-grounded-policy-analysis-config-v1", campaign=str(campaign), source=source,
                  registration=binding(HERE / "registration.md"), checkpoints=checkpoints, audits=audits, streams=streams)
    return certificate, config


def main():
    require(__debug__, "Python optimization forbidden")
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    def deadline(_number, _frame):
        raise TimeoutError("registered 120-second integrity budget exceeded")
    previous = signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, 120)
    try:
        require(digest(args.spec) == sha(args.sha256), "campaign specification differs")
        spec = read(args.spec)
        campaign = Path(spec["campaign"])
        out = args.output_dir
        require(out.is_absolute() and out.parent.resolve(strict=True) == out.parent
                and out.is_relative_to(campaign) and not out.exists(), "output must be new and inside campaign")
        out.mkdir()
        driver, supervisor = operators(spec)
        certificate, config = build(spec, driver, supervisor)
        # Revalidate touched immutable files before creating an accepted certificate.
        verify_files(spec["frozen_files"])
        require(digest(args.spec) == args.sha256, "specification changed during verification")
        for filename, value in (("integrity.json", certificate),):
            with (out / filename).open("x") as handle:
                handle.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
        config["integrity"] = binding(out / "integrity.json")
        with (out / "analysis-config.json").open("x") as handle:
            handle.write(json.dumps(config, sort_keys=True, indent=2, allow_nan=False) + "\n")
        print(json.dumps(dict(accepted=True, integrity=binding(out / "integrity.json"),
                              analysis_config=binding(out / "analysis-config.json"))), flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


if __name__ == "__main__":
    main()
