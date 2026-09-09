#!/usr/bin/env python3
"""Synthetic only: no checkpoint, generator, primary metric or campaign reads."""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"
import copy
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
import numpy as np

_spec = importlib.util.spec_from_file_location("c15_independent_under_test", Path(__file__).with_name("independent_review.py"))
I = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(I)


def fixture(index=0):
    """Hand-built public tiles; registered identity fields, never task generation."""
    group, slot = divmod(index, 16)
    maps = [0, 2, 4, 5, 7, 8, 9, 10, 13, 14, 15, 16, 18, 19, 21, 23]
    controls = list(itertools.permutations(range(4)))[maps[slot]]
    steps = [-8, 8, -1, 1]
    frames, position = [], 27
    for action in range(3):
        before = [0] * 64
        before[63], before[position] = 3, 2
        position += steps[controls[action]]
        after = [0] * 64
        after[63], after[position] = 3, 2
        frames.extend([before, after])
    query = [1 if x in (0, 7) or y in (0, 7) else 0 for y in range(8) for x in range(8)]
    query[27], query[19] = 2, 3
    frames.append(query)
    metadata = []
    for frame in range(7):
        for p in range(64):
            row = [0.0] * 10
            row[frame % 2 if frame < 6 else 2] = 1.0
            if frame < 6:
                row[3 + frame // 2] = 1.0
            row[7], row[8], row[9] = p % 8 / 7, p // 8 / 7, frame // 2 / 3 if frame < 6 else 1.0
            metadata.extend(row)
    packed_metadata = struct.pack("<4480f", *metadata)
    metadata = list(struct.unpack("<4480f", packed_metadata))
    packed = [struct.pack("<4096I", *(cell for cell in frame for _ in range(64))) for frame in frames]
    h = lambda data: hashlib.sha256(data).hexdigest()
    label = controls.index(0)
    training = group * 4599 // 63
    row = {"schema": "looped-grounded-policy-data-v1", "cohort": "seen", "condition": "factual", "support_cleared": False,
           "row_index": index, "group_index": group, "training_group_index": training, "original_update": training // 4 + 1,
           "data_seed": 20260920, "episode_id": 0x47524F554E445452 + training, "permutation_id": maps[slot],
           "correct_action": label, "query_direction": 0, "agent_patch": 27, "goal_patch": 19,
           "observed_support_action_ids": [0, 1, 2], "omitted_action": 3, "correct_action_demonstrated": label < 3,
           "inferred_controls": list(controls), "input_sha256": h(b"".join(packed) + packed_metadata),
           "factual_input_sha256": h(b"".join(packed) + packed_metadata),
           "cleared_input_sha256": h(bytes(6 * 16384) + packed[6] + packed_metadata), "metadata_sha256": h(packed_metadata),
           "query_sha256": h(packed[6]), "targets_sha256": h(b"synthetic inherited target identity"),
           "label_sha256": h(struct.pack("<I", label)), "query_cells": query,
           "logits": [float(a == label) for a in range(4)], "pooled": [0.0] * 256, "checkpointstage": "frozen",
           "frames": [], "public_metadata": metadata}
    for cells in frames:
        attention = [[float(p == cells.index(role)) for p in range(64)] for role in (2, 3)]
        row["frames"].append({"cells": cells, "attention": attention, "pooled": [0.0] * 256})
    row["attention"] = copy.deepcopy(row["frames"][6]["attention"])
    reference = {k: copy.deepcopy(v) for k, v in row.items() if k not in ("frames", "public_metadata")}
    return row, reference


def metric_fixture(n=1024, kind="perfect"):
    truth = np.asarray([[(27, 63), (19, 63), (19, 63), (27, 63), (27, 63), (26, 63), (27, 19)]] * n)
    attention = np.zeros((n, 7, 2, 64))
    if kind == "uniform":
        attention.fill(1/64)
    else:
        for row in range(n):
            for frame in range(7):
                for role in range(2):
                    target = int(truth[row, frame, role])
                    if kind == "shifted":
                        target = target + 1 if role == 0 else 1
                    attention[row, frame, role, target] = 1.0
    return truth, attention


class Inputs(unittest.TestCase):
    def test_public_bytes_and_all_registered_map_slots(self):
        for index in (0, 1, 15, 16, 511, 1023):
            row, reference = fixture(index)
            truth, _, correct, errors = I.validate_row(row, reference, "initial", index)
            self.assertEqual(truth[6], (27, 19))
            self.assertEqual(correct, 1)
            self.assertEqual(errors, {"logits": 0.0, "attention": 0.0, "pooled": 0.0})
        metadata = struct.unpack("<4480f", I.metadata((0, 1, 2)))
        self.assertEqual(metadata[0:7], (1, 0, 0, 1, 0, 0, 0))
        self.assertEqual(metadata[640:647], (0, 1, 0, 1, 0, 0, 0))
        self.assertEqual(metadata[1289], struct.unpack("<f", struct.pack("<f", 1/3))[0])
        self.assertEqual(metadata[3840:3847], (0, 0, 1, 0, 0, 0, 0))

    def test_public_input_hash_tampering(self):
        changes = (
            lambda r: r.__setitem__("input_sha256", "0" * 64),
            lambda r: r.__setitem__("query_sha256", "0" * 64),
            lambda r: r.__setitem__("metadata_sha256", "0" * 64),
            lambda r: r.__setitem__("cleared_input_sha256", "0" * 64),
            lambda r: r.__setitem__("label_sha256", "0" * 64),
            lambda r: r["public_metadata"].__setitem__(9, .25),
            lambda r: r["frames"][0]["cells"].__setitem__(1, 1),
            lambda r: r["frames"][1]["cells"].__setitem__(2, 3),
            lambda r: r.__setitem__("permutation_id", 1),
            lambda r: r.__setitem__("support_cleared", 0),
            lambda r: r.__setitem__("correct_action", True),
        )
        for mutate in changes:
            with self.subTest(mutate=mutate):
                row, _ = fixture()
                mutate(row)
                with self.assertRaises(I.Invalid):
                    I.public_truth(row, 0)

    def test_reference_parity_and_alias_fail_closed(self):
        for key in ("logits", "attention", "pooled"):
            row, ref = fixture()
            if key == "attention":
                row[key][0][27] -= .001
                row[key][0][26] += .001
            else:
                row[key][0] += .001
            with self.assertRaises(I.Invalid):
                I.validate_row(row, ref, "initial", 0)
        row, ref = fixture()
        row["frames"][6]["pooled"][0] = 1e-8
        with self.assertRaisesRegex(I.Invalid, "alias"):
            I.validate_row(row, ref, "initial", 0)
        row, ref = fixture()
        row["frames"][0]["attention"][0][27] = float("nan")
        with self.assertRaises(I.Invalid):
            I.validate_row(row, ref, "initial", 0)

    def test_exact_argmax_within_float_tolerance(self):
        row, ref = fixture()
        ref["logits"] = [0.0] * 4
        row["logits"] = [0.0, 1e-7, 0.0, 0.0]
        with self.assertRaisesRegex(I.Invalid, "winner"):
            I.validate_row(row, ref, "initial", 0)

    def test_array_shapes_types_and_normalization(self):
        for value in ([True], [float("inf")], [1e100], [[1.0]], ["1"]):
            with self.assertRaises((I.Invalid, ValueError)):
                I.array(value, (1,))
        for probs in ([.5, .4], [-.1, 1.1], [0, 0]):
            with self.assertRaises(I.Invalid):
                I.array(probs, (2,), True)

    def test_strict_json_and_binding_mutations(self):
        for raw in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
            with self.assertRaises(I.Invalid):
                I.decode(raw)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.json"
            path.write_text('{"value":1}')
            record = {"path": str(path), "sha256": I.file_sha(path)}
            self.assertEqual(I.bound(record), path)
            path.write_text('{"value":2}')
            with self.assertRaises(I.Invalid):
                I.bound(record)
            link = Path(folder) / "alias.json"
            link.symlink_to(path)
            with self.assertRaises(I.Invalid):
                I.regular(link)


class Metrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.weights, cls.bootstrap = I.resampling()

    def test_perfect_and_uniform(self):
        for kind, accuracy, mass in (("perfect", 1, 1), ("uniform", 0, 1/64)):
            truth, attention = metric_fixture(kind=kind)
            summary, _, gate = I.reconstruct(truth, attention, [0] * 1024, {"logits": 0.0, "attention": 0.0, "pooled": 0.0}, self.weights)
            for key, metric in summary["endpoints"].items():
                target = 0 if key == "current/action_accuracy" else accuracy if key.endswith("accuracy") else mass
                self.assertEqual(metric, {"estimate": float(target), "ci95": [float(target), float(target)]})
            self.assertEqual(summary["correct_counts"]["support/pooled_displacement_accuracy"], 3072 * accuracy)
            self.assertEqual(gate["selector_reuse_supported"], kind == "perfect")
            self.assertEqual(len(gate["components"]), 21)

    def test_wrong_locations_can_have_correct_displacements(self):
        truth, attention = metric_fixture(2, "shifted")
        values, minima, counts = I.score(truth, attention, [0, 1])
        self.assertEqual(counts["support/pooled_displacement_accuracy"], 6)
        self.assertEqual(counts["support/all_three_displacements_accuracy"], 2)
        self.assertEqual(counts["pair0/both_agent_accuracy"], 0)
        self.assertTrue(all(x == 0 for x in minima.values()))
        self.assertEqual(values["current/action_accuracy"].tolist(), [0, 1])

    def test_conjunction_and_mass_is_not_gate(self):
        truth, attention = metric_fixture()
        attention[:] = .49/63
        for frame in range(7):
            for role in range(2):
                attention[:, frame, role, truth[0, frame, role]] = .51
        summary, _, gate = I.reconstruct(truth, attention, [0]*1024, {}, self.weights)
        self.assertTrue(gate["selector_reuse_supported"])
        self.assertAlmostEqual(summary["minimum_mass"]["before0/agent_mass"], .51)
        attention[:11, 0, 0, :] = 0
        attention[:11, 0, 0, 1] = 1
        _, _, gate = I.reconstruct(truth, attention, [0]*1024, {}, self.weights)
        self.assertFalse(gate["selector_reuse_supported"])
        self.assertFalse(gate["components"]["before0/agent_accuracy"])
        self.assertTrue(gate["components"]["before0/goal_accuracy"])

    def test_bootstrap_matches_explicit_group_draws_and_paired_difference(self):
        draws = np.random.Generator(np.random.PCG64(1944)).integers(0, 64, (10000, 64))
        initial = np.arange(64, dtype=float) / 64
        final = (np.arange(64) % 7).astype(float) / 7
        result = I.intervals({"initial": initial, "delta": final-initial}, self.weights)
        for key, values in (("initial", initial), ("delta", final-initial)):
            expected = np.quantile(values[draws].mean(axis=1), [.025, .975], method="linear")
            np.testing.assert_allclose(result[key]["ci95"], expected, atol=1e-14, rtol=0)
            self.assertAlmostEqual(result[key]["estimate"], values.mean(), places=14)
        self.assertEqual(self.bootstrap["draws_sha256"], hashlib.sha256(draws.astype("<u8").tobytes()).hexdigest())
        self.assertFalse(np.array_equal(draws[:, 0], draws[:, 1]))

    def test_controls_require_cell0_nonrole_and_nonzero_moves(self):
        truth, _ = metric_fixture(1)
        self.assertTrue(I.controls(truth)["uniform"]["passed"])
        truth[0, 0, 1] = 0
        with self.assertRaises(I.Invalid):
            I.controls(truth)
        truth, _ = metric_fixture(1)
        truth[0, 1, 0] = truth[0, 0, 0]
        with self.assertRaises(I.Invalid):
            I.controls(truth)

    def test_report_missing_nonfinite_numeric_count_and_gate_mutations(self):
        truth, attention = metric_fixture()
        summary, _, gate = I.reconstruct(truth, attention, [1]*1024, {}, self.weights)
        expected = {"summary": summary, "gate": gate}
        verified = I.compare(copy.deepcopy(expected), expected)
        self.assertGreater(verified["compared_float_fields"], 100)
        self.assertEqual(verified["max_absolute_float_difference"], 0)
        changes = (
            lambda x: x["summary"]["endpoints"].pop("before0/agent_mass"),
            lambda x: x["summary"]["endpoints"]["before0/agent_mass"].__setitem__("estimate", float("nan")),
            lambda x: x["summary"]["endpoints"]["before0/agent_mass"].__setitem__("estimate", .99),
            lambda x: x["summary"]["correct_counts"].__setitem__("before0/agent_accuracy", 1024.0),
            lambda x: x["gate"].__setitem__("selector_reuse_supported", False),
            lambda x: x["gate"]["components"].__setitem__("pair0/displacement_accuracy", 1),
        )
        for mutate in changes:
            modified = copy.deepcopy(expected)
            mutate(modified)
            with self.assertRaises(I.Invalid):
                I.compare(modified, expected)

    def test_review_seam_checks_report_pin_and_all_numeric_sections(self):
        truth, attention = metric_fixture()
        parity = {"logits": 0.0, "attention": 0.0, "pooled": 0.0}
        summary, groups, gate = I.reconstruct(truth, attention, [1]*1024, parity, self.weights)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config_path, report_path = root/"config.json", root/"report.json"
            config = {"registration": {"sha256": "a"*64}, "integrity": {"sha256": "b"*64}, "arms": {"initial": {}, "final": {}}}
            config_path.write_text(json.dumps(config))
            receipt = {"source_revision": "c"*40, "binary_sha256": "d"*64}
            report = {"schema": I.SCHEMA, "accepted": True, "classification": "exploratory_frozen_selector_reuse",
                      "config_sha256": I.file_sha(config_path), "registration_sha256": "a"*64,
                      "analysis_sha256": "e"*64, "helper_sha256": I.HELPER_SHA, "integrity_sha256": "b"*64,
                      **receipt, "summaries": {"initial": summary, "final": summary}, "gates": {"initial": gate, "final": gate},
                      "final_minus_initial": I.intervals({k: np.zeros(64) for k in groups}, self.weights),
                      "bootstrap": self.bootstrap, "limits": ["synthetic limitation"]*6, "elapsed_seconds": .1}
            report_path.write_text(json.dumps(report))
            real_hash = I.file_sha
            def test_hash(path):
                return "e"*64 if Path(path).name == "analysis.py" else real_hash(path)
            collected = ([{"synthetic": True}], truth, attention, [1]*1024, parity)
            with mock.patch.object(I, "authority", return_value=receipt), mock.patch.object(I, "collect", return_value=collected), mock.patch.object(I, "file_sha", side_effect=test_hash):
                result = I.review(config_path, report_path)
                self.assertTrue(result["accepted"])
                self.assertEqual(result["report_sha256"], real_hash(report_path))
                self.assertGreater(result["compared_float_fields"], 400)
                report["gates"]["initial"]["selector_reuse_supported"] = False
                report_path.write_text(json.dumps(report))
                with self.assertRaisesRegex(I.Invalid, "discrete"):
                    I.review(config_path, report_path)


if __name__ == "__main__":
    unittest.main()
