"""Synthetic fixtures only: these tests never open C8/C9 real features or labels."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import role_selector_witness as w
np = w.np


def artificial(n=24, width=8):
    rng = np.random.default_rng(123)
    roles = np.stack([rng.permutation(64)[:2] for _ in range(n)])
    h = rng.normal(0, .01, (n, 64, width))
    h[:, :, :2] = 0
    h[np.arange(n)[:, None], roles, np.arange(2)] = 1
    return h, roles


def policy(width):
    rng = np.random.default_rng(78)
    return {"mean": rng.normal(size=width).tolist(),
            "scale": rng.uniform(.2, 2, size=width).tolist(),
            "coefficients": rng.normal(size=(width, 4)).tolist(),
            "label_mean": [.1, .2, .3, .4]}


def feature_rows():
    roles = [(18, 10), (18, 26), (18, 17), (18, 19)]
    rows = []
    for i, (agent, goal) in enumerate(roles):
        cells = [int(j//8 in (0, 7) or j%8 in (0, 7)) for j in range(64)]
        cells[agent], cells[goal] = 2, 3
        row = {"schema": "looped-known-features-v1", "input_index": i, "layout_index": i,
               "episode_id": w.FIT_TAG+i, "data_seed": 20260915, "partition": "fit",
               "permutation_id": 0, "condition": "factual", "support_cleared": False,
               "inferred_controls": [0, 1, 2, 3], "evaluation_loops": 4,
               "min_distance": 1, "max_distance": 1, "oracle_distance": 1,
               "visible_cells": cells, "correct_action": i,
               "input_sha256": w.sha(bytes([i])),
               "query_sha256": w.sha(np.repeat(np.asarray(cells, dtype="<u4"), 64).tobytes()),
               "label_sha256": w.sha(np.asarray(i, dtype="<u4").tobytes()), "arrays": {}}
        for name, shape in [("current", [64, 128]), ("cls", [128]), ("policy", [4])]:
            length = int(np.prod(shape))*4
            row["arrays"][name] = {"file": f"known-features-{name}.f32", "shape": shape,
                                   "dtype": "F32LE", "byte_offset": i*length, "byte_length": length}
        rows.append(row)
    return rows


class NumericalTests(unittest.TestCase):
    def test_registered_synthetic_controls(self):
        controls = w.synthetic_controls()
        for name in ("known_separator", "identical_tokens", "joint_permutation", "normal_equation", "raw_score"):
            self.assertIs(controls[name], True)

    def test_ridge_matches_independent_augmented_least_squares(self):
        h, roles = artificial()
        result, info = w.fit_selector(h, roles)
        x = h.reshape(-1, h.shape[-1])
        z = (x-result["mean"])/result["scale"]
        y = np.zeros((len(h), 64, 2))
        for row, targets in enumerate(roles):
            for role, token in enumerate(targets):
                y[row, token, role] = 1
        y = y.reshape(-1, 2)-result["intercept"]
        augmented = np.concatenate([z, np.sqrt(len(z)*.01)*np.eye(z.shape[1])])
        labels = np.concatenate([y, np.zeros((z.shape[1], 2))])
        independent = np.linalg.lstsq(augmented, labels, rcond=None)[0]
        np.testing.assert_allclose(result["weights"], independent, atol=1e-11, rtol=1e-11)
        self.assertLess(info["normal_equation_relative_residual"], 1e-10)
        np.testing.assert_allclose(z@independent+result["intercept"], x@result["raw_weights"]+result["raw_bias"], atol=1e-10)

    def test_identical_tokens_have_ties_and_no_margin_witness(self):
        roles = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        h = np.ones((4, 64, 4))
        selector, info = w.fit_selector(h, roles)
        result, _ = w.apply_selector(h, selector)
        np.testing.assert_array_equal(result["positions"], 0)
        np.testing.assert_array_equal(result["argmax_ties"], 63)
        np.testing.assert_array_equal(selector["alpha"], [1, 1])
        self.assertEqual(info["mass_guarantee_eligible"], [False, False])
        perm = np.arange(64)[::-1]
        moved, _ = w.apply_selector(h[:, perm], selector)
        np.testing.assert_array_equal(result["scores"][:, perm], moved["scores"])
        np.testing.assert_array_equal(result["weights64"][:, perm], moved["weights64"])
        # First-index tie breaking intentionally does not follow a moved index zero.
        self.assertFalse(np.array_equal(moved["positions"], np.argsort(perm)[result["positions"]]))

    def test_margins_exclude_self_and_keep_competitor_order(self):
        scores = np.array([[[1., 4.], [3., 2.], [2., 1.]]])
        roles = np.array([[1, 0]])
        np.testing.assert_array_equal(w.margins(scores, roles), [[[2., 1.], [2., 3.]]])

    def test_soft_query_and_frozen_affine_translation(self):
        h, roles = artificial(8, 128)
        selector, _ = w.fit_selector(h, roles)
        result, _ = w.apply_selector(h, selector)
        logits = np.einsum("ntd,rd->ntr", h, selector["queries"])/np.sqrt(128)
        np.testing.assert_allclose(logits, result["logits64"], atol=1e-10, rtol=1e-10)
        weights = np.exp(logits-logits.max(1, keepdims=True))
        weights /= weights.sum(1, keepdims=True)
        pools = np.einsum("ntr,ntd->nrd", weights, h).reshape(8, 256)
        old = policy(256)
        w.validate_policy(old)
        folded_w = np.asarray(old["coefficients"])/np.asarray(old["scale"])[:, None]
        folded_b = np.asarray(old["label_mean"])-np.asarray(old["mean"])@folded_w
        np.testing.assert_allclose(pools@folded_w+folded_b, w.policy_scores(result["soft"], old), atol=1e-10, rtol=1e-10)
        self.assertTrue(np.isfinite(result["scores32"]).all())
        self.assertTrue(np.isfinite(result["logits32"]).all())

    def test_mass_guard_rejects_nonfinite_wrong_shapes_and_false_eligibility(self):
        h, roles = artificial(4)
        selector, _ = w.fit_selector(h, roles)
        result, _ = w.apply_selector(h, selector)
        self.assertTrue(w.mass_checks(result, roles, selector["eligible"])[0]["passed"])
        for key in ("weights64", "weights32"):
            broken = {k: v.copy() for k, v in result.items()}
            broken[key][0, 0, 0] = np.nan
            with self.assertRaises(w.NumericalControlFailure):
                w.mass_checks(broken, roles, selector["eligible"])
        for invalid in ([1, 1], [True], [True, True, True]):
            with self.assertRaises(ValueError):
                w.mass_checks(result, roles, invalid)
        flat = dict(result, weights64=np.full_like(result["weights64"], 1/64),
                    weights32=np.full_like(result["weights32"], 1/64))
        self.assertFalse(w.mass_checks(flat, roles, np.array([True, True]))[0]["passed"])
        self.assertTrue(w.mass_checks(flat, roles, np.array([False, False]))[0]["passed"])

    def test_nonfinite_features_and_bad_roles_fail_closed(self):
        h, roles = artificial(4)
        h[0, 0, 0] = np.inf
        with self.assertRaises(w.NumericalControlFailure):
            w.fit_selector(h, roles)
        for bad in (np.array([[0, 0]]), np.array([[0, 64]]), np.array([[0., 1.]])):
            with self.assertRaises(ValueError):
                w.validate_targets(bad, 1, 64)

    def test_null_permutation_preserves_pairs_and_repeats_across_cores(self):
        pairs = np.array([[i%64, (i+1)%64] for i in range(512)])
        a = np.random.Generator(np.random.PCG64(1917)).permutation(512)
        b = np.random.Generator(np.random.PCG64(1917)).permutation(512)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(np.sort(a), np.arange(512))
        self.assertEqual(sorted(map(tuple, pairs)), sorted(map(tuple, pairs[a])))
        self.assertTrue((pairs[a, 1] == (pairs[a, 0]+1)%64).all())


class IntegrityTests(unittest.TestCase):
    def test_visible_role_geometry_and_byte_layout_are_independent(self):
        rows = feature_rows()
        roles, actions = w.validate_rows(rows, 4)
        np.testing.assert_array_equal(actions, [0, 1, 2, 3])
        np.testing.assert_array_equal(roles[:, 0], 18)
        for mutation in ("duplicate_role", "offset", "order", "label", "fraction", "bool"):
            changed = copy.deepcopy(rows)
            if mutation == "duplicate_role":
                changed[0]["visible_cells"][19] = 2
                changed[0]["query_sha256"] = w.sha(np.repeat(np.asarray(changed[0]["visible_cells"], dtype="<u4"), 64).tobytes())
            elif mutation == "offset":
                changed[1]["arrays"]["current"]["byte_offset"] += 4
            elif mutation == "order":
                changed[0]["visible_cells"] = changed[0]["visible_cells"][::-1]
            elif mutation == "label":
                changed[0]["correct_action"] = 1
            elif mutation == "fraction":
                changed[0]["visible_cells"][0] = .5
            else:
                changed[0]["visible_cells"][0] = True
            with self.assertRaises(ValueError, msg=mutation):
                w.validate_rows(changed, 4)

    def test_prefix_reader_does_not_consume_evaluation_row(self):
        with tempfile.TemporaryDirectory(dir=w.HERE) as temporary:
            path = Path(temporary)/"rows.jsonl"
            path.write_bytes(b'{"fit":1}\n'+b'not-valid-evaluation-json\n')
            rows, digest = w.prefix_rows(path, 1)
            self.assertEqual(rows, [{"fit": 1}])
            self.assertEqual(digest, w.sha(b'{"fit":1}\n'))

    def test_four_fit_barrier_and_serialization_tampering(self):
        with tempfile.TemporaryDirectory(dir=w.HERE) as temporary:
            output = Path(temporary)/"output"
            output.mkdir()
            barrier = w.Barrier(output)
            with self.assertRaises(ValueError):
                w.load_eval_features("initial", None, barrier)
            with self.assertRaises(ValueError):
                barrier.seal([])
            names = []
            for core in w.CORES:
                name = f"policy-{core}.json"
                w.write_json(output/name, policy(256))
                names.append(name)
                for arm in w.ARMS:
                    name = f"selector-{core}-{arm}.npz"
                    w.save_arrays(output/name, weights=np.arange(6).reshape(3, 2))
                    names.append(name)
            barrier.seal(names)
            barrier.require_sealed()
            with np.load(output/names[-1], allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["weights"], np.arange(6).reshape(3, 2))
            (output/names[-1]).write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                barrier.require_sealed()

    def test_outer_inventory_does_not_read_evaluation_bytes_before_barrier(self):
        with tempfile.TemporaryDirectory(dir=w.HERE) as temporary:
            root = Path(temporary)/"parent"
            root.mkdir()
            (root/"eval.bin").write_bytes(b"hidden evaluation")
            manifest = Path(temporary)/"manifest.json"
            w.write_json(manifest, {"campaign": str(root), "created_local": w.local_now(),
                "files": {"eval.bin": {"sha256": w.file_sha(root/"eval.bin"), "bytes": 17}}})
            expected = w.file_sha(manifest)
            original = w.file_sha
            opened = []
            def tracked(path):
                opened.append(Path(path))
                return original(path)
            with mock.patch.object(w, "file_sha", side_effect=tracked):
                outer = w.Outer(root, manifest, expected)
                self.assertNotIn(root/"eval.bin", opened)
                barrier = w.Barrier(Path(temporary)/"output")
                with self.assertRaises(ValueError):
                    outer.verify_all(barrier)
                self.assertNotIn(root/"eval.bin", opened)
            (root/"extra.bin").write_bytes(b"x")
            with self.assertRaises(ValueError):
                w.Outer(root, manifest, expected)


class StatisticalTests(unittest.TestCase):
    def test_bootstrap_reselects_constant_in_each_whole_query_draw(self):
        roles = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])
        result = {"positions": roles.copy(), "hard": np.eye(4), "soft": np.eye(4)}
        old = {"mean": [0]*4, "scale": [1]*4, "coefficients": np.eye(4).tolist(), "label_mean": [0]*4}
        actions = np.array([0, 1, 2, 3])
        draws = np.array([[0, 0, 0, 0], [0, 1, 2, 3], [1, 1, 2, 2]])
        metrics, _ = w.evaluation_metrics(result, roles, actions, old, draws)
        # Perfect predictions minus per-draw best constant: 0, .75, .50.
        expected = np.quantile([0, .75, .50], [.025, .975], method="linear")
        np.testing.assert_array_equal(metrics["hard"]["advantage_over_resampled_best_constant"]["ci95"], expected)
        self.assertEqual(metrics["joint"]["accuracy"], 1)
        np.testing.assert_array_equal(w.bootstrap_indices(), w.bootstrap_indices())

    @staticmethod
    def passing():
        arms = {}
        for core in w.CORES:
            for arm in w.ARMS:
                rate = 1. if arm == "true" else 0.
                record = {name: {"accuracy": rate} for name in ("agent", "goal", "joint", "hard", "soft")}
                for name in ("hard", "soft"):
                    record[name]["advantage_over_resampled_best_constant"] = {"ci95": [.5, .8]}
                arms[core+"/"+arm] = record
        controls = {name: True for name in ("synthetic", "fit_mass", "oracle_initial", "oracle_final")}
        return arms, controls

    def test_mandatory_controls_reject_missing_wrongtype_or_incomplete_arms(self):
        arms, controls = self.passing()
        self.assertTrue(w.classify(arms, controls)["cores"]["initial"]["routing_supported"])
        for malformed in ({}, {**controls, "fit_mass": 1.0}, {**controls, "extra": True},
                          {k: v for k,v in controls.items() if k != "oracle_initial"}):
            with self.assertRaises(ValueError):
                w.classify(arms, malformed)
        with self.assertRaises(ValueError):
            w.classify({k:v for k,v in arms.items() if k != "final/null"}, controls)

    def test_either_null_policy_and_either_oracle_failure_is_inconclusive(self):
        for core in w.CORES:
            for pool in ("hard", "soft"):
                arms, controls = self.passing()
                arms[core+"/null"][pool]["accuracy"] = .51
                decision = w.classify(arms, controls)
                self.assertFalse(decision["controls_valid"])
                self.assertEqual(decision["cores"]["initial"]["status"], "inconclusive_failed_control")
            arms, controls = self.passing()
            controls["oracle_"+core] = False
            self.assertFalse(w.classify(arms, controls)["cores"]["final"]["routing_supported"])

    def test_routing_gate_has_strict_ci_and_separate_accuracy_thresholds(self):
        for key, value in (("joint", .949), ("hard", .899), ("soft", .899)):
            arms, controls = self.passing()
            arms["initial/true"][key]["accuracy"] = value
            self.assertEqual(w.classify(arms, controls)["cores"]["initial"]["status"], "selection_only_not_supported")
        arms, controls = self.passing()
        arms["initial/true"]["hard"]["advantage_over_resampled_best_constant"]["ci95"][0] = .25
        self.assertFalse(w.classify(arms, controls)["cores"]["initial"]["routing_supported"])


if __name__ == "__main__":
    unittest.main()
