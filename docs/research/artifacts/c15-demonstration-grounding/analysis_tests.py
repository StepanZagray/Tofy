#!/usr/bin/env python3
"""Hand-built public frames and corrupted artifacts; no episode generation/model."""
import copy
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("c15_scorer_under_test", HERE / "analysis.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)
np = A.np


def fixture(map_id=0):
    """Independently encode one artificial, legal first-group public input."""
    permutation = tuple(itertools.permutations(range(4)))[map_id]
    directions = ((0, -1), (0, 1), (-1, 0), (1, 0))
    cells, agent = [], 27
    for action in (0, 1, 2):
        before = [0] * 64
        before[agent], before[63] = 2, 3
        dx, dy = directions[permutation[action]]
        agent += dx + 8 * dy
        after = [0] * 64
        after[agent], after[63] = 2, 3
        cells.extend((before, after))
    query = [int(x in (0, 7) or y in (0, 7)) for y in range(8) for x in range(8)]
    query[27], query[28] = 2, 3
    cells.append(query)
    metadata = []
    for frame in range(7):
        for y in range(8):
            for x in range(8):
                record = [0.] * 10
                record[frame % 2 if frame < 6 else 2] = 1.
                if frame < 6:
                    record[3 + frame // 2] = 1.
                record[7:10] = [x / 7, y / 7, (frame // 2) / 3 if frame < 6 else 1.]
                metadata.extend(struct.unpack("<10f", struct.pack("<10f", *record)))
    raw_metadata = struct.pack("<4480f", *metadata)
    patches = b"".join(struct.pack("<I", color) * 64 for frame in cells for color in frame)
    query_raw = patches[-16384:]
    sha = lambda raw: hashlib.sha256(raw).hexdigest()
    label = permutation.index(3)
    audit = dict(schema="looped-grounded-policy-data-v1", cohort="seen", condition="factual",
                 support_cleared=False, row_index=A.H.FIT.index(map_id), group_index=0,
                 training_group_index=0, original_update=1, data_seed=20260920,
                 episode_id=0x47524F554E445452, permutation_id=map_id, correct_action=label,
                 query_direction=3, agent_patch=27, goal_patch=28, observed_support_action_ids=[0, 1, 2],
                 omitted_action=3, correct_action_demonstrated=label != 3, inferred_controls=list(permutation),
                 input_sha256=sha(patches + raw_metadata), factual_input_sha256=sha(patches + raw_metadata),
                 cleared_input_sha256=sha(bytes(6 * 16384) + query_raw + raw_metadata),
                 metadata_sha256=sha(raw_metadata), query_sha256=sha(query_raw),
                 targets_sha256=sha(b"synthetic target identity; no simulator"),
                 label_sha256=sha(struct.pack("<I", label)), query_cells=query)
    frames = []
    for frame in cells:
        attention = [[float(i == frame.index(role)) for i in range(64)] for role in (2, 3)]
        frames.append(dict(cells=frame, attention=attention, pooled=[i / 256 for i in range(256)]))
    reference = dict(audit, checkpointstage="frozen", logits=[float(i == label) for i in range(4)],
                     attention=copy.deepcopy(frames[6]["attention"]), pooled=frames[6]["pooled"].copy())
    row = dict(copy.deepcopy(reference), frames=frames, public_metadata=metadata)
    return row, reference


class PublicFrameTests(unittest.TestCase):
    def setUp(self):
        self.row, self.reference = fixture()

    def validate(self, row=None, reference=None):
        return A.validate_row(row or self.row, reference or self.reference, "initial", 0)

    def test_full_input_reconstruction_and_every_fit_map(self):
        for slot, map_id in enumerate(A.H.FIT):
            row, reference = fixture(map_id)
            _, truth, attention, correct, errors = A.validate_row(row, reference, "initial", slot)
            self.assertEqual(correct, 1)
            self.assertEqual(errors, dict(logits=0., attention=0., pooled=0.))
            self.assertTrue(np.array_equal(attention.argmax(-1), truth))
            self.assertEqual(truth[:6, 1].tolist(), [63] * 6)

    def test_current_boundary_support_open_floor_distinction(self):
        self.assertEqual(self.row["frames"][0]["cells"][0], 0)
        self.assertEqual(self.row["query_cells"][0], 1)
        self.validate()
        self.row["frames"][0]["cells"][0] = 1
        with self.assertRaisesRegex(ValueError, "support topology"):
            self.validate()

    def test_hash_corruption_and_wrong_frame_order(self):
        self.row["frames"][2]["cells"][10] = 2
        with self.assertRaises(ValueError):
            self.validate()
        self.row, self.reference = fixture()
        self.row["frames"][0], self.row["frames"][1] = self.row["frames"][1], self.row["frames"][0]
        with self.assertRaises(ValueError):
            self.validate()
        self.row, self.reference = fixture()
        for field in ("input_sha256", "factual_input_sha256"):
            self.row[field] = self.reference[field] = "f" * 64
        with self.assertRaisesRegex(ValueError, "full public input hash"):
            self.validate()

    def test_wrong_label_map_and_row_order(self):
        for field, value in (("correct_action", 2), ("permutation_id", 2), ("row_index", True)):
            row, ref = fixture()
            row[field] = ref[field] = value
            with self.assertRaises(ValueError):
                self.validate(row, ref)

    def test_public_metadata_has_exact_f32_bytes(self):
        self.row["public_metadata"][9] = 1 / 3
        with self.assertRaisesRegex(ValueError, "public metadata"):
            self.validate()

    def test_finite_range_normalization_bool_and_dimensions(self):
        for invalid in (float("nan"), float("inf"), -0.1, True):
            row, reference = fixture()
            row["frames"][0]["attention"][0][0] = invalid
            with self.assertRaises(ValueError):
                self.validate(row, reference)
        row, reference = fixture()
        row["frames"][0]["attention"][0][0] = .00002
        with self.assertRaisesRegex(ValueError, "normalization"):
            self.validate(row, reference)
        row, reference = fixture()
        row["frames"][0]["pooled"].pop()
        with self.assertRaises(ValueError):
            self.validate(row, reference)

    def test_current_tolerance_never_permits_changed_winner(self):
        self.row["logits"] = [0., 0., 0., .000001]
        self.reference["logits"] = [.000001, 0., 0., 0.]
        with self.assertRaisesRegex(ValueError, "winner parity"):
            self.validate()

    def test_current_alias_is_exact_even_inside_parity_tolerance(self):
        self.row["frames"][6]["pooled"][0] = .00000001
        with self.assertRaisesRegex(ValueError, "alias"):
            self.validate()


class MetricTests(unittest.TestCase):
    def setUp(self):
        row, ref = fixture()
        _, truth, attention, _, _ = A.validate_row(row, ref, "initial", 0)
        self.truth, self.attention = truth[None], attention[None]

    def test_onehot_and_uniform_controls(self):
        result = A.controls(self.truth)
        self.assertEqual(result["onehot"]["role_and_displacement_accuracy"], 1)
        self.assertEqual(result["uniform"]["role_and_displacement_accuracy"], 0)
        self.assertEqual(result["uniform"]["minimum_mass"], 1 / 64)

    def test_correct_displacement_can_have_wrong_absolute_locations(self):
        translated = self.attention.copy()
        for frame in range(6):
            translated[0, frame, 0] = 0
            translated[0, frame, 0, self.truth[0, frame, 0] + 1] = 1
        values, _, counts = A.measurements(translated, self.truth, [1])
        self.assertEqual(values["support/all_three_displacements_accuracy"].tolist(), [1])
        self.assertEqual(values["support/all_six_joint_accuracy"].tolist(), [0])
        self.assertEqual(counts["support/pooled_displacement_accuracy"], 3)
        self.assertEqual(counts["pair0/both_agent_accuracy"], 0)

    def test_wrong_after_location_corrupts_only_corresponding_displacement(self):
        self.attention[0, 3, 0] = self.attention[0, 2, 0]
        values, _, counts = A.measurements(self.attention, self.truth, [0])
        self.assertEqual([values[f"pair{i}/displacement_accuracy"][0] for i in range(3)], [1, 0, 1])
        self.assertEqual(counts["support/pooled_displacement_accuracy"], 2)
        self.assertEqual(counts["support/all_three_displacements_accuracy"], 0)

    def test_whole_group_weighting_and_paired_bootstrap(self):
        rows = np.repeat(np.arange(64, dtype=float) / 63, 16)
        groups = A.grouped({"x": rows})
        draws = np.repeat(np.arange(64)[:, None], 64, axis=1)
        estimate = A.bootstrap(groups, draws)["x"]
        self.assertAlmostEqual(estimate["estimate"], .5)
        np.testing.assert_allclose(estimate["ci95"], [.025, .975], atol=1e-15)
        difference = A.bootstrap({"x": groups["x"] - groups["x"]}, draws)["x"]
        self.assertEqual(difference, {"estimate": 0., "ci95": [0., 0.]})

    def test_gate_uses_all_components_but_not_mass_or_ci(self):
        values, _, _ = A.measurements(self.attention, self.truth, [0])
        endpoints = {key: {"estimate": float(value.mean()), "ci95": [0., 1.]} for key, value in values.items()}
        endpoints["before0/agent_mass"]["estimate"] = .01
        endpoints["before0/agent_accuracy"]["estimate"] = .99
        endpoints["pair1/displacement_accuracy"]["estimate"] = .98
        self.assertTrue(A.gate(endpoints)["selector_reuse_supported"])
        self.assertEqual(len(A.gate(endpoints)["components"]), 21)
        endpoints["after2/goal_accuracy"]["estimate"] = .989999
        self.assertFalse(A.gate(endpoints)["selector_reuse_supported"])


class BindingTests(unittest.TestCase):
    def test_complete_synthetic_stream_and_paired_group_endpoints(self):
        templates = [fixture(map_id) for map_id in A.H.FIT]
        def population(reference=False):
            for group in range(64):
                original = group * 4599 // 63
                for slot, pair in enumerate(templates):
                    row = copy.deepcopy(pair[int(reference)])
                    row.update(row_index=group * 16 + slot, group_index=group,
                               training_group_index=original, original_update=original // 4 + 1,
                               episode_id=0x47524F554E445452 + original)
                    yield row
        # All frames are hand-built repeated fixtures, not generated task episodes.
        with mock.patch.object(A, "jsonl", side_effect=lambda p: population(p == "reference")):
            audits, truth, attention, actions, errors = A.collect("initial", {
                "rows": {"path": "rows"}, "reference": {"path": "reference"}})
        self.assertEqual(len(audits), 1024)
        self.assertEqual(errors, dict(logits=0., attention=0., pooled=0.))
        values, minima, counts = A.measurements(attention, truth, actions)
        self.assertEqual(counts["support/pooled_displacement_accuracy"], 3072)
        self.assertEqual(counts["support/all_six_joint_accuracy"], 1024)
        draws = np.random.Generator(np.random.PCG64(1944)).integers(0, 64, (10000, 64))
        endpoints = A.bootstrap(A.grouped(values), draws)
        self.assertTrue(all(value == {"estimate": 1., "ci95": [1., 1.]} for value in endpoints.values()))
        self.assertTrue(all(value == 1 for value in minima.values()))
        self.assertTrue(A.gate(endpoints)["selector_reuse_supported"])
        A.controls(truth)

    def test_missing_extra_or_reordered_stream_rows_reject(self):
        row, reference = fixture()
        for new, old in (([row], []), ([row], [reference]), ([row, row], [reference, reference])):
            with mock.patch.object(A, "jsonl", side_effect=lambda p: iter(new if p == "rows" else old)):
                with self.assertRaises(ValueError):
                    A.collect("initial", {"rows": {"path": "rows"}, "reference": {"path": "reference"}})

    def test_strict_json_hash_and_nonsymlink_paths(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}'):
            with self.assertRaises(ValueError):
                A.H.decode(raw)
        with tempfile.TemporaryDirectory(prefix="c15-scorer-") as directory:
            path = Path(directory) / "input"
            path.write_bytes(b"fixture")
            sha = A.file_sha(path)
            self.assertEqual(A.bound(dict(path=str(path), sha256=sha)), path)
            with self.assertRaises(ValueError):
                A.bound(dict(path=str(path), sha256="0" * 64))
            alias = Path(directory) / "alias"
            alias.symlink_to(path)
            with self.assertRaises(ValueError):
                A.file_sha(alias)

    def test_receipt_and_checkpoint_guards_before_numerical_reads(self):
        with tempfile.TemporaryDirectory(prefix="c15-scorer-") as directory:
            root = Path(directory)
            def record(name, value):
                path = root / name
                path.write_text(json.dumps(value))
                return dict(path=str(path), sha256=A.file_sha(path))
            registration = record("registration", {"synthetic": True})
            arms = {a: {k: record(a + k, {"no_model_values": True}) for k in ("rows", "reference")}
                    for a in ("initial", "final")}
            files = {str(A.HELPER): A.HELPER_SHA, str(HERE / "analysis.py"): A.file_sha(HERE / "analysis.py")}
            receipt_files = dict(files, **{r["path"]: r["sha256"] for a in arms.values() for r in a.values()})
            receipt_files[registration["path"]] = registration["sha256"]
            receipt = dict(schema="looped-demonstration-grounding-integrity-v1", accepted=True,
                           checks=dict.fromkeys(A.CHECKS, True), frozen_files=receipt_files,
                           source_revision="a" * 40, binary_sha256="b" * 64,
                           dependency_revision=A.DEPENDENCY, arms=copy.deepcopy(A.CHECKPOINTS))
            config = dict(schema=A.SCHEMA, registration=registration, frozen_files=files,
                          arms=arms, integrity=record("receipt", receipt))
            A.validate_config(config)
            for key in A.CHECKS:
                bad = copy.deepcopy(receipt)
                bad["checks"][key] = 1  # Reject truthy, non-boolean assertions.
                config["integrity"] = record("receipt", bad)
                with self.assertRaises(ValueError):
                    A.validate_config(config)
            bad = copy.deepcopy(receipt)
            bad["arms"]["final"]["head_sha256"] = "f" * 64
            config["integrity"] = record("receipt", bad)
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                A.validate_config(config)
            bad = copy.deepcopy(receipt)
            del bad["frozen_files"][arms["initial"]["reference"]["path"]]
            config["integrity"] = record("receipt", bad)
            with self.assertRaisesRegex(ValueError, "does not bind"):
                A.validate_config(config)


if __name__ == "__main__":
    print(f"synthetic test PID={os.getpid()}", flush=True)
    unittest.main(verbosity=2)
