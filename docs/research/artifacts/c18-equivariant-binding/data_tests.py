#!/usr/bin/env python3
"""Finite enumeration and analytic controls; no learned-model calls or fits."""
import copy
import importlib.util
import itertools
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
_spec = importlib.util.spec_from_file_location("c18_data_under_test", Path(__file__).with_name("data.py"))
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)
np = D.np


class FiniteDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = D.dataset(512)

    def test_complete_independent_enumeration_and_unique_gap_one(self):
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        for partition, maps, count in (("fit", D.FIT, 1536), ("heldout", D.HELD, 768)):
            rows = self.data[partition]
            self.assertEqual(len(rows), count)
            expected = itertools.product(itertools.permutations(range(4), 3), range(4), maps)
            for i, (row, (actions, desired, map_id)) in enumerate(zip(rows, expected, strict=True)):
                mapping = tuple(itertools.permutations(range(4)))[map_id]
                effects = {a: directions[mapping[a]] for a in actions}
                missing = next(a for a in range(4) if a not in actions)
                effects[missing] = tuple(-sum(e[j] for e in effects.values()) for j in (0, 1))
                q = directions[desired]
                scores = [sum(effects[a][j] * q[j] for j in (0, 1)) for a in range(4)]
                expected_label = next(a for a in range(4) if mapping[a] == desired)
                self.assertEqual(row["index"], i)
                self.assertEqual(row["correct_action"], expected_label)
                self.assertEqual(row["observed_actions"], list(actions))
                self.assertEqual(scores[expected_label], 1)
                self.assertEqual(sorted(scores), [-1, 0, 0, 1])
                self.assertEqual(D.analytic_scores(row["features"]).tolist(), scores)
        self.assertTrue(D.validate(self.data))

    def test_canonical_serialization_ignores_nested_insertion_order(self):
        first = {"z": [{"b": 2, "a": 1}], "a": True}
        second = {"a": True, "z": [{"a": 1, "b": 2}]}
        self.assertEqual(D.canonical_bytes(first), D.canonical_bytes(second))
        self.assertEqual(D.canonical_bytes(first), b'{"a":true,"z":[{"a":1,"b":2}]}\n')
        with self.assertRaises(ValueError):
            D.canonical_bytes({"bad": float("nan")})

    def test_balanced_constants_and_missing_action_fallback(self):
        for partition in ("fit", "heldout"):
            rows = self.data[partition]
            labels = [r["correct_action"] for r in rows]
            self.assertEqual([labels.count(a) for a in range(4)], [len(rows) // 4] * 4)
            fallback = [next(a for a in range(4) if a not in r["observed_actions"]) for r in rows]
            self.assertEqual(sum(p == y for p, y in zip(fallback, labels)), len(rows) // 4)
            self.assertTrue(all((p == y) == (not r["demonstrated"]) for p, y, r in zip(fallback, labels, rows)))

    def test_semantic_orbits_are_six_exact_support_permutations(self):
        for partition, expected in (("fit", 256), ("heldout", 128)):
            rows = self.data[partition]
            canonical_inputs = {}
            for row in rows:
                records = tuple(sorted(tuple(x) for x in row["features"][:3])) + (tuple(row["features"][3]),)
                canonical_inputs.setdefault(records, []).append(row)
            self.assertEqual(len(canonical_inputs), expected)
            self.assertTrue(all(len(v) == 6 for v in canonical_inputs.values()))
            self.assertTrue(all(sum(r["canonical"] for r in v) == 1 for v in canonical_inputs.values()))
            self.assertTrue(all(len({r["correct_action"] for r in v}) == 1 for v in canonical_inputs.values()))

    def test_schedule_complete_epochs_and_exact_exposures(self):
        indices = np.asarray(self.data["updates"])
        self.assertEqual(indices.shape, (1150, 512))
        for epoch in range(383):
            self.assertEqual(sorted(indices[3 * epoch:3 * (epoch + 1)].ravel()), list(range(1536)))
        self.assertEqual(len(set(indices[1149:].ravel())), 512)
        counts = np.bincount(indices.ravel(), minlength=1536)
        self.assertEqual((int((counts == 383).sum()), int((counts == 384).sum())), (1024, 512))
        self.assertEqual(sum(o["training_presentations"] for o in self.data["audit"]["abstract"]["fit"]["orbits"]), 588800)
        self.assertEqual(D.schedule(512), self.data["updates"])
        rng = np.random.Generator(np.random.PCG64(20260922))
        self.assertEqual(indices[:3].ravel().tolist(), rng.permutation(1536).tolist())

    def test_every_power_of_two_preserves_flat_order_and_exact_tail(self):
        original = [i for batch in self.data["updates"] for i in batch]
        self.assertEqual(D.sha(np.asarray(original, dtype="<u4").tobytes()),
                         "a46eac4347bfe16d55d220240452cb67643ccd007a97d2e56da16be45cb6f51e")
        for batch in (512, 1024, 2048, 4096, 8192, 16384, 32768):
            updates = D.schedule(batch)
            self.assertEqual([i for values in updates for i in values], original)
            self.assertTrue(all(len(values) == batch for values in updates[:-1]))
            self.assertEqual(len(updates[-1]), 588800 % batch or batch)
        self.assertEqual(len(D.schedule(1024)), 575)
        self.assertEqual((len(D.schedule(2048)), len(D.schedule(2048)[-1])), (288, 1024))

    def test_regrouping_preserves_all_cohorts_and_exposures(self):
        regrouped = D.dataset(2048)
        for key in ("fit", "heldout", "cached_visual"):
            self.assertEqual(D.canonical_bytes(regrouped[key]), D.canonical_bytes(self.data[key]))
        self.assertEqual(regrouped["audit"]["abstract"], self.data["audit"]["abstract"])
        self.assertEqual(regrouped["audit"]["schedule"]["tail_rows"], 1024)
        self.assertEqual(regrouped["audit"]["schedule"]["full_updates"], 287)
        self.assertTrue(D.validate(regrouped))

    def test_smoke_is_full_batch_exact_prefix_with_own_schedule_identity(self):
        original = np.asarray(self.data["updates"]).ravel()
        for batch, updates in ((512, 2), (2048, 5), (32768, 5)):
            smoke = D.dataset(batch, "smoke", updates)
            indices = np.asarray(smoke["updates"])
            self.assertEqual(indices.shape, (updates, batch))
            np.testing.assert_array_equal(indices.ravel(), original[:updates * batch])
            self.assertEqual(smoke["schedule"], dict(kind="smoke", effective_batch=batch,
                presentations=batch * updates, indices_sha256=D.sha(indices.astype("<u4").tobytes())))
            self.assertEqual(smoke["audit"]["schedule"]["tail_rows"], 0)
            self.assertNotEqual(smoke["schedule"]["indices_sha256"], D.C17_FLAT_SHA)

    def test_schedule_admission_and_rehashed_tail_corruptions(self):
        for batch in (True, 512., 256, 513, 65536):
            with self.assertRaises(ValueError):
                D.schedule(batch)
        for kind, updates in (("training", 2), ("smoke", None), ("smoke", True), ("smoke", 3), ("other", 2)):
            with self.assertRaises(ValueError):
                D.schedule(1024, kind, updates)
        damaged = D.dataset(2048)
        damaged["updates"][-1].append(damaged["updates"][-1][-1])
        damaged["schedule"]["presentations"] += 1
        damaged["schedule"]["indices_sha256"] = D.sha(np.asarray([i for b in damaged["updates"] for i in b], dtype="<u4").tobytes())
        with self.assertRaises(ValueError):
            D.validate(damaged)

    def test_zero_effects_and_zero_query_collapse_correct_groups(self):
        for partition in ("fit", "heldout"):
            effect_groups, query_groups = {}, {}
            for row in self.data[partition]:
                x = np.asarray(row["features"])
                effect_zero = x.copy(); effect_zero[:3, :2] = 0
                query_zero = x.copy(); query_zero[3, :2] = 0
                effect_groups.setdefault(effect_zero.tobytes(), []).append(row["correct_action"])
                query_groups.setdefault(query_zero.tobytes(), []).append(row["correct_action"])
                self.assertEqual(D.analytic_scores(effect_zero).tolist(), [0.] * 4)
                self.assertEqual(D.analytic_scores(query_zero).tolist(), [0.] * 4)
            for labels in (*effect_groups.values(), *query_groups.values()):
                self.assertEqual([labels.count(a) for a in range(4)], [len(labels) // 4] * 4)

    def test_label_feature_schedule_schema_and_hash_corruption(self):
        edits = [lambda d: d["fit"][0].update(correct_action=True),
                 lambda d: d["fit"][0]["features"][0].__setitem__(0, .25),
                 lambda d: d["updates"][0].__setitem__(0, d["updates"][0][1]),
                 lambda d: d["fit"][0].update(input_sha256="0" * 64),
                 lambda d: d["fit"][0].update(orbit_id=999),
                 lambda d: d.update(unregistered=True)]
        for edit in edits:
            damaged = copy.deepcopy(self.data)
            edit(damaged)
            with self.assertRaises(ValueError):
                D.validate(damaged)


class MathematicalLimitsTests(unittest.TestCase):
    def test_two_observations_do_not_identify_missing_actions(self):
        maps = [(0, 1, 2, 3), (0, 1, 3, 2)]
        self.assertEqual(maps[0][:2], maps[1][:2])
        self.assertNotEqual(maps[0].index(2), maps[1].index(2))
        with self.assertRaises(ValueError):
            D.records(D.DIRECTIONS[:3], [0, 0, 1], D.DIRECTIONS[0])

    def test_cardinality_and_bijection_assumptions_are_material(self):
        # Zero-sum effects alone do not guarantee self-dot-product is maximal.
        x = D.records([[1, 0], [2, 0], [-1, 0]], [0, 1, 2], [1, 0])
        self.assertEqual(int(D.analytic_scores(x).argmax()), 1)  # Actual desired action is0.
        # Three observed actions in a nonbijective environment leave action3 free.
        effects_a = ((0, -1), (0, 1), (-1, 0), (1, 0))
        effects_b = ((0, -1), (0, 1), (-1, 0), (0, -1))
        self.assertEqual(effects_a[:3], effects_b[:3])
        self.assertNotEqual(effects_a[3], effects_b[3])

    def test_normalized_coordinate_bound_is_tight_at_opposite_corners(self):
        attention = np.zeros((7, 2, 64))
        attention[:, 0, 0], attention[:, 0, 63] = .8, .2
        attention[:, 1, 63] = 1
        # Make after positions truth63, with goal at7, to test summed endpoints.
        cells = []
        for frame in range(7):
            c = [0] * 64
            if frame in (1, 3, 5):
                attention[frame, 0] = 0
                attention[frame, 0, 63], attention[frame, 0, 0] = .8, .2
                attention[frame, 1] = 0; attention[frame, 1, 7] = 1
                c[63], c[7] = 2, 3
            else:
                c[0], c[63] = 2, 3
            cells.append(c)
        x, normalized, coords, rounding = D.soft_features(attention.tolist(), [0, 1, 2])
        np.testing.assert_allclose(coords[0, 0], [1.4, 1.4])
        np.testing.assert_allclose(coords[1, 0] - coords[0, 0], [4.2, 4.2])
        report = D.coordinate_audit(normalized, coords, cells, rounding)
        self.assertAlmostEqual(report["maximum_coordinate_absolute_error"], 1.4)
        self.assertLess(report["analytic_gap_lower_bound"], 0)
        self.assertEqual(x.shape, (4, 7))

    def test_soft_features_do_not_consume_truth_and_normalize_first(self):
        attention = np.full((7, 2, 64), 1 / 64)
        x, _, _, _ = D.soft_features(attention.tolist(), [2, 0, 3])
        y, _, _, _ = D.soft_features((attention * 1.000001).tolist(), [2, 0, 3])
        np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(x[:, :2], np.zeros((4, 2)))
        for invalid in (float("nan"), float("inf"), -1., True):
            damaged = attention.tolist(); damaged[0][0][0] = invalid
            with self.assertRaises(ValueError):
                D.soft_features(damaged, [2, 0, 3])
        with self.assertRaises(ValueError):
            D.soft_features((attention * .5).tolist(), [2, 0, 3])

    def test_frozen_parent_and_json_guards_fail_before_loading_values(self):
        with mock.patch.object(D, "file_sha", return_value="0" * 64):
            with self.assertRaisesRegex(ValueError, "outer seal"):
                D.load_visual()
        with tempfile.TemporaryDirectory(prefix="c18-data-") as folder:
            path = Path(folder) / "file"
            path.write_text("fixture")
            alias = Path(folder) / "alias"; alias.symlink_to(path)
            with self.assertRaises(ValueError):
                D.file_sha(alias)
        for raw in ('{"x":1,"x":2}', '{"x":NaN}'):
            with self.assertRaises(ValueError):
                D.decode(raw)


if __name__ == "__main__":
    print(f"synthetic test PID={os.getpid()}", flush=True)
    unittest.main(verbosity=2)
