#!/usr/bin/env python3
"""Synthetic contract tests only: no registered C12 rows, checkpoints, or runs."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("c12_integrity", HERE / "integrity.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def save(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def tensor_file(path, tensors):
    table, raw = {}, bytearray()
    for name, (shape, data) in tensors.items():
        table[name] = dict(dtype="F32", shape=list(shape), data_offsets=[len(raw), len(raw) + len(data)])
        raw.extend(data)
    header = json.dumps(table).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + raw)


def models():
    scalar = ((1,), struct.pack("<f", 0.0))
    core = {f"{family}.{part}": scalar for family in
            ("policy_head", "reward_head", "value_head", *(f"next_head_{i}" for i in range(4)))
            for part in ("weight", "bias")}
    core["blocks.0.weight"] = scalar
    head = {name: (shape, bytes(4 * check.math.prod(shape))) for name, shape in check.HEAD_SHAPES.items()}
    return core, head


def update(index=1, norm=0.5):
    return dict(update=index, elapsed_seconds=float(index), metrics=dict(
        rows=64, physical_batch=33, microbatches=2, tail_batch=31, mean_ce=7.0,
        pre_update_correct=0, pre_clip_norm=norm, clip_scale=min(1., 1 / (norm + 1e-6)),
        body_gradient_norm=norm * .6, head_gradient_norm=norm * .8,
        query_gradient_norms=[norm * .1, norm * .2], elapsed_seconds=.2))


def fixture_rows():
    cells = [1 if i % 8 in (0, 7) or i // 8 in (0, 7) else 0 for i in range(64)]
    cells[27], cells[19] = 2, 3
    query = b"".join(struct.pack("<I", c) * 64 for c in cells)
    rows = []
    for slot, map_id in enumerate(check.FIT):
        controls = list(check.PERMUTATIONS[map_id])
        action = controls.index(0)
        row = {field: "a" * 64 for field in check.IDENTITY if field.endswith("_sha256")}
        row.update(schema="looped-grounded-policy-data-v1", cohort="familiar", condition="factual",
                   support_cleared=False, row_index=slot, group_index=0, training_group_index=None,
                   original_update=None, data_seed=20260921, episode_id=0x47524F554E444556,
                   permutation_id=map_id, correct_action=action, query_direction=0, agent_patch=27,
                   goal_patch=19, observed_support_action_ids=[0, 1, 2], omitted_action=3,
                   correct_action_demonstrated=action != 3, inferred_controls=controls, query_cells=cells,
                   query_sha256=hashlib.sha256(query).hexdigest(),
                   label_sha256=hashlib.sha256(struct.pack("<I", action)).hexdigest(),
                   input_sha256=f"{slot:064x}", factual_input_sha256=f"{slot:064x}")
        rows.append(row)
    return rows


class Integrity(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="c12-integrity-fixture-")
        self.root = Path(self.temp.name).resolve()

    def tearDown(self):
        path = self.root
        self.temp.cleanup()
        self.assertFalse(path.exists())

    def test_strict_json_rejects_duplicate_and_nonfinite(self):
        for raw in ('{"a":1,"a":2}', '{"v":NaN}', '{"v":Infinity}'):
            with self.assertRaises(ValueError):
                check.decode(raw)

    def test_literal_boolean_gate_population(self):
        valid = {name: True for name in check.CHECKS}
        check.typed_checks(valid)
        for bad in ({}, dict(valid, gradient=1), dict(valid, oracle=False), dict(valid, extra=True)):
            with self.assertRaises(ValueError):
                check.typed_checks(bad)

    def test_tensor_roundtrip_offsets_dtype_and_finite(self):
        path = self.root / "weights.safetensors"
        core, _ = models()
        tensor_file(path, core)
        self.assertEqual(check.safetensors(path), core)
        raw = path.read_bytes()
        length = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8:8 + length])
        first = next(iter(header))
        for change in (dict(dtype="F64"), dict(data_offsets=[0, 3]), dict(data_offsets=[4, 8])):
            broken = copy.deepcopy(header)
            broken[first].update(change)
            h = json.dumps(broken).encode()
            path.write_bytes(struct.pack("<Q", len(h)) + h + raw[8 + length:])
            with self.assertRaises(ValueError):
                check.safetensors(path)
        tensor_file(path, {"x": ((1,), struct.pack("<f", float("nan")))})
        with self.assertRaises(ValueError):
            check.safetensors(path)

    def test_canonical_named_import_and_signed_zero_identity(self):
        path = self.root / "parameters.f32"
        path.write_bytes(bytes(5136))
        core, head = models()
        self.assertEqual(check.canonical_head(path), head)
        changed = dict(core)
        changed["policy_head.bias"] = ((1,), struct.pack("<f", -0.0))
        audit, numeric = check.changes(core, head, changed, head)
        self.assertFalse(audit["unused_heads_unchanged"])
        self.assertEqual(audit["changed_unused_names"], ["core.policy_head.bias"])
        self.assertEqual(numeric["unused_l2"], 0.)  # Bit identity is stronger than norm equality.
        path.write_bytes(bytes(5135))
        with self.assertRaises(ValueError):
            check.canonical_head(path)

    def test_saved_changes_and_restoration_are_raw_verified(self):
        core, head = models()
        changed_core = dict(core, **{"blocks.0.weight": ((1,), struct.pack("<f", .25))})
        changed_head = dict(head)
        changed_head["output.bias"] = ((4,), struct.pack("<4f", .1, 0, 0, 0))
        for name, values in (("final-core", changed_core), ("final-head", changed_head),
                             ("restored-core", core), ("restored-head", head)):
            tensor_file(self.root / f"{name}.safetensors", values)
        audit, _ = check.changes(core, head, changed_core, changed_head)
        restored, _ = check.changes(core, head, core, head)
        report = dict(changes=audit, restored_changes=restored)
        for kind in ("final", "restored"):
            for family in ("core", "head"):
                report[f"{kind}_{family}_sha256"] = check.digest(self.root / f"{kind}-{family}.safetensors")
        numeric = check.check_weight_report(self.root, report, core, head, True)
        self.assertEqual(numeric["body_changed_tensors"], 1)
        tensor_file(self.root / "restored-core.safetensors", changed_core)
        report["restored_core_sha256"] = check.digest(self.root / "restored-core.safetensors")
        with self.assertRaisesRegex(ValueError, "restoration changed"):
            check.check_weight_report(self.root, report, core, head, True)

    def test_update_count_tail_and_negative_performance_remain_valid(self):
        path = self.root / "updates.jsonl"
        values = [update(1), update(2, 2.)]
        path.write_text("\n".join(json.dumps(v) for v in values) + "\n")
        last, norms = check.check_updates(path, 2, 33, positive=True)
        self.assertEqual(last["pre_update_correct"], 0)
        self.assertEqual(norms["body"]["nonzero_updates"], 2)
        for field, bad in (("tail_batch", 32), ("clip_scale", 1.), ("body_gradient_norm", 0.)):
            corrupted = copy.deepcopy(values)
            corrupted[1]["metrics"][field] = bad
            path.write_text("\n".join(json.dumps(v) for v in corrupted))
            with self.assertRaises(ValueError):
                check.check_updates(path, 2, 33, positive=True)
        path.write_text(json.dumps(values[0]))
        with self.assertRaises(ValueError):
            check.check_updates(path, 2, 33)

    def test_zero_terminal_gradient_allowed_but_smoke_requires_connection(self):
        path = self.root / "updates.jsonl"
        path.write_text(json.dumps(update(norm=0.)))
        check.check_updates(path, 1, 33)
        with self.assertRaises(ValueError):
            check.check_updates(path, 1, 33, positive=True)

    def test_visible_oracle_map_balance_and_corruption(self):
        rows = fixture_rows()
        for index, row in enumerate(rows):
            check.audit_identity(row, "familiar", "factual", index)
        check.check_groups(rows, check.FIT)
        wrong = copy.deepcopy(rows[0])
        wrong["query_cells"][0] = 0
        with self.assertRaises(ValueError):
            check.audit_identity(wrong, "familiar", "factual", 0)
        wrong = copy.deepcopy(rows[0])
        wrong["correct_action"] = (wrong["correct_action"] + 1) % 4
        with self.assertRaises(ValueError):
            check.audit_identity(wrong, "familiar", "factual", 0)
        rows[1]["observed_support_action_ids"] = [2, 1, 0]
        with self.assertRaises(ValueError):
            check.check_groups(rows, check.FIT)

    def test_hash_memo_invalidates_after_same_length_edit(self):
        path = self.root / "pin"
        path.write_bytes(b"a")
        old = check.digest(path)
        path.write_bytes(b"b")
        self.assertNotEqual(old, check.digest(path))

    def test_cleanup_requires_typed_flags_and_no_survivors(self):
        state = dict(pid_gone=True, group_gone=True, owned_survivors=[], group_survivors=[],
                     cleanup_error=None, error=None, returncode=0)
        check.cleanup(state)
        for change in (dict(pid_gone=1), dict(group_survivors=[9]), dict(cleanup_error="error"), dict(error="failed")):
            with self.assertRaises(ValueError):
                check.cleanup(dict(state, **change))
        start = Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19]
        with self.assertRaises(ValueError):
            check.process_gone(os.getpid(), start)

    def test_finite_array_rejects_bool_nan_and_shape(self):
        for value in ([True] * 4, [float("nan")] * 4, [0.] * 3):
            with self.assertRaises(ValueError):
                check.finite(value, (4,), "fixture")

    def test_leader_exit_does_not_hide_live_process_group(self):
        with patch.object(check.os, "killpg", return_value=None) as probe:
            with self.assertRaisesRegex(ValueError, "process group still exists"):
                check.group_gone(12345)
            probe.assert_called_once_with(12345, 0)
        with patch.object(check.os, "killpg", side_effect=ProcessLookupError):
            check.group_gone(12345)

    def test_capacity_search_requires_closed_maximal_bracket(self):
        save(self.root / "batch-64-u2.exit.json", {"accepted": True})
        qualification = dict(tests={"64": True}, physical_batch=64, upper_excluded=None,
                             accumulation=1, effective_batch=64)
        names, tested = check.capacity_search(self.root, qualification, {}, None)
        self.assertEqual(tested, {64})
        self.assertIn("confirm-64-u5", names)
        for change in (dict(tests={"64": 1}), dict(physical_batch=63), dict(tests={"64": True, "32": True})):
            with self.assertRaises(ValueError):
                check.capacity_search(self.root, dict(qualification, **change), {}, None)

    def test_module_freeze_precedes_operator_import(self):
        with patch.object(check.importlib.util, "spec_from_file_location") as importer:
            with self.assertRaises(ValueError):
                check.operators({"frozen_files": {str(HERE / "integrity.py"): check.digest(HERE / "integrity.py")}})
            importer.assert_not_called()

    def test_missing_terminal_stage_stops_before_output_read(self):
        class Driver:
            def completed_stage(self, _spec, stage):
                if stage == "final":
                    raise ValueError("final stage absent")
                return {}
        campaign_spec = dict(campaign=str(self.root), source="a" * 40, binary_sha256="b" * 64,
                             initial_core_sha256=check.CORE, initial_head_sha256=check.HEAD,
                             registration_sha256="b" * 64)
        with patch.object(check, "digest", return_value="b" * 64), patch.object(check, "read") as reader:
            with self.assertRaisesRegex(ValueError, "final stage absent"):
                check.build(campaign_spec, Driver(), None)
            reader.assert_not_called()

    def test_config_and_certificate_match_analyzer_closed_schema(self):
        module_spec = importlib.util.spec_from_file_location("c12_analyzer_contract", HERE / "analyze.py")
        analyzer = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(analyzer)
        source = dict(revision="a" * 40, binary_sha256="b" * 64, dependency_revision=check.DEPENDENCY)
        checkpoints = dict(frozen=dict(core_sha256=check.CORE, policy_sha256=check.HEAD, updates=0),
                           final=dict(core_sha256="c" * 64, policy_sha256="d" * 64, updates=1150))
        audits = {f"{c}/{q}": dict(path=str(self.root / f"{c}-{q}-audit"), sha256="e" * 64)
                  for c in check.MAPS for q in ("factual", "cleared")}
        streams = {}
        for stage in ("frozen_factual", "final_factual", "final_cleared"):
            condition = "cleared" if stage == "final_cleared" else "factual"
            checkpoint = "frozen" if stage == "frozen_factual" else "final"
            for cohort in check.MAPS:
                streams[f"{stage}/{cohort}"] = dict(path=str(self.root / f"{stage}-{cohort}"), sha256="f" * 64,
                    cohort=cohort, condition=condition, checkpointstage=checkpoint, source=source,
                    checkpoint=checkpoints[checkpoint], audit_sha256=audits[f"{cohort}/{condition}"]["sha256"])
        certificate = dict(schema="looped-grounded-policy-integrity-v1", accepted=True, campaign=str(self.root),
            source=source, registration_sha256="a" * 64, checkpoints=checkpoints,
            audit_sha256={k:v["sha256"] for k,v in audits.items()}, stream_sha256={k:v["sha256"] for k,v in streams.items()},
            checks={name:True for name in check.CHECKS}, parameter_changes={"body_changed_tensors":1}, gradient_norms={"body_max":.1})
        save(self.root / "integrity.json", certificate)
        config = dict(schema="looped-grounded-policy-analysis-config-v1", campaign=str(self.root), source=source,
            registration=dict(path=str(HERE / "registration.md"), sha256="a" * 64), checkpoints=checkpoints,
            audits=audits, streams=streams, integrity=check.binding(self.root / "integrity.json"))
        analyzer.validate_config(config, self.root)
        self.assertEqual(analyzer.certificate(config), certificate)
        config["streams"]["final_cleared/seen"]["checkpoint"] = checkpoints["frozen"]
        with self.assertRaises(ValueError):
            analyzer.validate_config(config, self.root)

    def test_never_reused_output_is_rejected_before_operators(self):
        spec_path = self.root / "spec.json"
        output = self.root / "certificate"
        output.mkdir()
        save(spec_path, {"campaign": str(self.root)})
        argv = ["integrity.py", "--spec", str(spec_path), "--sha256", check.digest(spec_path), "--output-dir", str(output)]
        with patch.object(sys, "argv", argv), patch.object(check, "operators") as importer:
            with self.assertRaisesRegex(ValueError, "output must be new"):
                check.main()
            importer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
