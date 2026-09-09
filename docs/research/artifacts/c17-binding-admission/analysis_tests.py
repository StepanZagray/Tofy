#!/usr/bin/env python3
"""Synthetic finite outputs only; actual model checkpoints/results are never read."""
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("c16_analysis_under_test", HERE / "analysis.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)
np, D = A.np, A.D


def output_row(row, spec, logits):
    # Independent F32 byte construction for actual ablated input identity.
    x = copy.deepcopy(row["features"])
    if spec["cleared"]:
        for record in x[:3]:
            record[0] = record[1] = 0.
    if spec["query_cleared"]:
        x[3][0] = x[3][1] = 0.
    raw = struct.pack("<28f", *(v for record in x for v in record))
    return dict(copy.deepcopy(row), logits=logits, stage=spec["stage"], loops=spec["loops"],
                cleared=spec["cleared"], query_cleared=spec["query_cleared"], model_input_sha256=hashlib.sha256(raw).hexdigest())


def logits_for(rows, kind="oracle"):
    result = np.zeros((len(rows), 4))
    if kind == "oracle":
        for i, row in enumerate(rows):
            result[i, row["correct_action"]] = 5.
    elif kind == "missing":
        for i, row in enumerate(rows):
            result[i, next(a for a in range(4) if a not in row["observed_actions"])] = 5.
    return result


class NumericTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fit, cls.held = D.abstract_rows(D.FIT), D.abstract_rows(D.HELD)

    def test_scalar_ce_signed_margin_ties_and_shift_invariance(self):
        logits = np.array([[2., 1., -1., 0.], [-1000., 0., 0., -1000.], [0., 0., 0., 0.]])
        labels = np.array([0, 0, 3])
        result = A.vectors(logits, labels)
        expected = [math.log(sum(math.exp(float(v - max(row))) for v in row)) - float(row[y] - max(row))
                    for row, y in zip(logits, labels)]
        np.testing.assert_allclose(result["ce"], expected, atol=1e-14, rtol=0)
        self.assertEqual(result["predictions"].tolist(), [0, 1, 0])
        self.assertEqual(result["margin"].tolist(), [1., -1000., 0.])
        self.assertEqual(result["winner_margin"].tolist(), [1., 0., 0.])
        shifted = A.vectors(logits + 10000, labels)
        np.testing.assert_array_equal(shifted["ce"], result["ce"])

    def test_oracle_raw_canonical_subset_and_map_counts(self):
        result, _, _, _ = A.score(self.fit, logits_for(self.fit), A.SPECS["final-fit-l4"])
        self.assertEqual((result["raw"]["rows"], result["raw"]["correct"]), (1536, 1536))
        self.assertEqual((result["canonical"]["rows"], result["canonical"]["correct"]), (256, 256))
        self.assertEqual(result["canonical"]["subsets"]["omitted"]["rows"], 64)
        self.assertEqual(result["canonical"]["subsets"]["demonstrated"]["rows"], 192)
        self.assertTrue(all(v["rows"] == 16 and v["correct"] == 16 for v in result["canonical"]["per_map"].values()))
        self.assertEqual(result["ordering"], dict(applicable=True, orbits=256, inconsistent_orbits=0,
                                                winner_invariant=True, max_logit_deviation=0.))
        self.assertAlmostEqual(result["canonical"]["ce"], math.log(1 + 3 * math.exp(-5)), places=14)

    def test_missing_id_fallback_is_not_demonstrated_success(self):
        result, _, _, _ = A.score(self.held, logits_for(self.held, "missing"), A.SPECS["final-heldout-query-zero"])
        self.assertEqual(result["canonical"]["correct"], 32)
        self.assertEqual(result["canonical"]["subsets"]["omitted"]["correct"], 32)
        self.assertEqual(result["canonical"]["subsets"]["demonstrated"]["correct"], 0)
        self.assertTrue(result["control"]["winner_invariant"] and result["control"]["exact_quarter"])

    def test_external_controls_report_counts_without_fake_ce(self):
        controls = A.external_controls({"fit": self.fit, "heldout": self.held})
        for cohort, canonical_rows in (("fit", 256), ("heldout", 128)):
            result = controls[cohort]
            self.assertEqual(result["analytic"]["minimum_true_score_margin"], 1.)
            self.assertEqual(result["analytic"]["canonical"]["correct"], canonical_rows)
            self.assertEqual(result["constant_action0"]["canonical"]["correct"], canonical_rows // 4)
            self.assertEqual(result["always_missing_id"]["canonical"]["subsets"]["omitted"]["correct"], canonical_rows // 4)
            self.assertEqual(result["always_missing_id"]["canonical"]["subsets"]["demonstrated"]["correct"], 0)
            self.assertNotIn("ce", result["analytic"]["canonical"])

    def test_pair_counts_are_on_same_cases_and_signed(self):
        labels = np.array([0, 1, 2, 3])
        first = A.vectors(np.eye(4) * 5, labels)
        second = A.vectors(np.array([[5., 0., 0., 0.]] * 4), labels)
        difference = A.paired(first, second, np.array([True] * 4))
        self.assertEqual(difference["accuracy_delta"], .75)
        self.assertEqual((difference["both_correct"], difference["first_only_correct"], difference["second_only_correct"]), (1, 3, 0))
        self.assertEqual(difference["changed_predictions"], 3)
        reverse = A.paired(second, first, np.ones(4, bool))
        self.assertEqual(reverse["accuracy_delta"], -.75)
        self.assertAlmostEqual(reverse["ce_delta"], -difference["ce_delta"])

    def test_orbit_logit_range_checks_all_pairs_and_exact_winners(self):
        logits = logits_for(self.fit)
        members = [i for i, r in enumerate(self.fit) if r["orbit_id"] == 0]
        logits[members[0]] += .5
        logits[members[1]] -= .5
        result, _, _, _ = A.score(self.fit, logits, A.SPECS["final-fit-l4"])
        self.assertEqual(result["ordering"]["max_logit_deviation"], 1.)
        self.assertTrue(result["ordering"]["winner_invariant"])
        logits[members[2]] = 0
        logits[members[2], (self.fit[members[2]]["correct_action"] + 1) % 4] = 10
        result, _, _, _ = A.score(self.fit, logits, A.SPECS["final-fit-l4"])
        self.assertEqual(result["ordering"]["inconsistent_orbits"], 1)

    def test_inconsistent_ablated_predictions_are_detected_even_at_quarter_accuracy(self):
        logits = logits_for(self.fit, "zero")
        # On one identical-input group, changing a correct and an incorrect answer
        # can preserve the aggregate quarter; determinism remains an independent gate.
        groups = {}
        spec = A.SPECS["final-fit-effects-zero"]
        for i, row in enumerate(self.fit):
            groups.setdefault(A.actual_input(row, spec).tobytes(), []).append(i)
        members = next(iter(groups.values()))
        right = next(i for i in members if self.fit[i]["correct_action"] == 0)
        wrong = next(i for i in members if self.fit[i]["correct_action"] == 1)
        logits[right, 1] = logits[wrong, 1] = 1
        result, _, _, _ = A.score(self.fit, logits, spec)
        self.assertTrue(result["control"]["exact_quarter"])
        self.assertFalse(result["control"]["winner_invariant"])

    def test_integer_gates_and_unselected_depth_are_separate(self):
        summaries = {}
        for name, spec in A.SPECS.items():
            if spec["cohort"] == "cached_visual":
                continue
            rows = self.fit if spec["cohort"] == "fit" else self.held
            logits = logits_for(rows, "zero" if spec["cleared"] or spec["query_cleared"] else "oracle")
            summaries[name] = A.score(rows, logits, spec)[0]
        fit, held = summaries["final-fit-l4"]["canonical"], summaries["final-heldout-l4"]["canonical"]
        fit["correct"], held["correct"] = 254, 116
        held["subsets"]["demonstrated"]["correct"], held["subsets"]["omitted"]["correct"] = 87, 29
        summaries["final-fit-l1"]["ordering"]["winner_invariant"] = False
        self.assertEqual(A.decide(summaries)[0], "supported_single_seed_binding_prerequisite")
        for target in (fit, held, held["subsets"]["demonstrated"], held["subsets"]["omitted"]):
            target["correct"] -= 1
            self.assertEqual(A.decide(summaries)[0], "registered_binding_screen_not_supported")
            target["correct"] += 1


class GuardTests(unittest.TestCase):
    def test_obsolete_batch64_schedule_rejects_before_cache_read(self):
        dataset = D.dataset()
        prefix = np.asarray(dataset["updates"]).ravel()[:73600]
        dataset["updates"] = prefix.reshape(1150, 64).tolist()
        with mock.patch.object(A.D, "load_visual") as cache:
            with self.assertRaisesRegex(ValueError, "schedule differs"):
                A.validate_dataset(dataset)
            cache.assert_not_called()

    def test_post_clamp_input_hash_and_identity_fields(self):
        row = D.abstract_rows(D.FIT)[0]
        for name in ("final-fit-l4", "final-fit-effects-zero", "final-fit-query-zero"):
            spec = A.SPECS[name]
            output = output_row(row, spec, [1., 2., 3., 4.])
            np.testing.assert_array_equal(A.row_logits(output, row, spec), [1., 2., 3., 4.])
            for key, value in (("model_input_sha256", "f" * 64), ("loops", True), ("stage", "initial"), ("correct_action", 9)):
                bad = copy.deepcopy(output); bad[key] = value
                with self.assertRaises(ValueError):
                    A.row_logits(bad, row, spec)
        bad_spec = dict(spec, cleared=True, query_cleared=True)
        with self.assertRaises(ValueError):
            A.actual_input(row, bad_spec)

    def test_nonfinite_bool_and_extra_missing_reordered_rows(self):
        rows = D.abstract_rows(D.FIT)[:2]
        spec = A.SPECS["final-fit-l4"]
        for value in (float("inf"), float("nan"), True, 1e100):
            output = output_row(rows[0], spec, [value, 0., 0., 0.])
            with self.assertRaises(ValueError):
                A.row_logits(output, rows[0], spec)
        outputs = [output_row(r, spec, [0., 0., 0., 0.]) for r in rows]
        with tempfile.TemporaryDirectory(prefix="c16-analysis-") as folder:
            path = Path(folder) / "rows"
            for variant in (outputs[:1], outputs + outputs[:1], outputs[::-1]):
                path.write_bytes(b"".join(D.canonical_bytes(r) for r in variant))
                with self.assertRaises(ValueError):
                    A.read_stream(path, rows, spec)

    def test_independent_dataset_membership_rejects_rehashed_corruption(self):
        rows = D.abstract_rows(D.HELD)
        A.validate_abstract(rows, D.HELD)
        for key, value in (("map_id", 0), ("canonical", 1), ("orbit_id", 0), ("correct_action", 0)):
            bad = copy.deepcopy(rows)
            # Ensure mutation differs from the selected original.
            bad[1][key] = value if value != bad[1][key] else 999
            with self.assertRaises(ValueError):
                A.validate_abstract(bad, D.HELD)
        bad = copy.deepcopy(rows)
        bad[0]["features"][0][0] += .25
        bad[0].update(D.identity(bad[0]["features"], bad[0]["correct_action"]))
        with self.assertRaises(ValueError):
            A.validate_abstract(bad, D.HELD)

    def test_complete_synthetic_config_analysis_and_runtime_rejections(self):
        # Synthetic cache fixtures deliberately bypass only the external C15
        # read adapter. No real visual/model output is read by this test.
        fit = D.abstract_rows(D.FIT)
        visual = [dict(index=i, audit={"permutation_id": r["map_id"], "observed_support_action_ids": r["observed_actions"]}, features=r["features"],
                       correct_action=r["correct_action"], demonstrated=r["demonstrated"],
                       input_sha256=r["input_sha256"], label_sha256=r["label_sha256"])
                  for i, r in enumerate(fit[:1024])]
        visual_audit = {"synthetic_fixture": True}
        dataset = D.dataset(visual, visual_audit)
        with tempfile.TemporaryDirectory(prefix="c16-analysis-") as folder:
            root = Path(folder)
            def save(name, value):
                path = root / name
                path.write_bytes(D.canonical_bytes(value))
                return str(path), A.file_sha(path)
            data_path, data_sha = save("dataset", dataset)
            reg_path, reg_sha = save("registration", {"synthetic_fixture": True})
            sources = {str(HERE / n): A.file_sha(HERE / n) for n in ("data.py", "analysis.py", "analysis_tests.py")}
            files = dict(sources, **{data_path: data_sha, reg_path: reg_sha})
            streams = {}
            for name, spec in A.SPECS.items():
                rows = dataset[spec["cohort"]]
                logits = logits_for(rows, "zero" if spec["stage"] == "initial" or spec["cleared"] or spec["query_cleared"] else "oracle")
                path = root / name
                path.write_bytes(b"".join(D.canonical_bytes(output_row(row, spec, value.tolist())) for row, value in zip(rows, logits)))
                streams[name] = dict(spec, rows=str(path), sha256=A.file_sha(path))
                files[str(path)] = A.file_sha(path)
            checkpoints = {"initial": {"sha256": "1" * 64, "parameter_sha256": "2" * 64},
                           "final": {"sha256": "3" * 64, "parameter_sha256": "4" * 64}}
            receipt = dict(schema="looped-action-binding-integrity-v1", accepted=True, checks=dict.fromkeys(A.CHECKS, True),
                source_revision="a" * 40, binary_sha256="b" * 64, dependency_revision=A.DEPENDENCY,
                dataset_sha256=data_sha, registration_sha256=reg_sha, checkpoints=checkpoints, frozen_files=files,
                streams={name: dict(record, checkpoint_sha256=checkpoints[record["stage"]]["sha256"],
                                   parameter_sha256=checkpoints[record["stage"]]["parameter_sha256"]) for name, record in streams.items()})
            receipt_path, receipt_sha = save("receipt", receipt)
            config = dict(schema=A.SCHEMA, dataset=data_path, dataset_sha256=data_sha,
                registration=dict(path=reg_path, sha256=reg_sha), frozen_files=dict(files, **{receipt_path: receipt_sha}),
                streams=streams, integrity_receipt=receipt_path, integrity_receipt_sha256=receipt_sha)
            config_path, _ = save("config", config)
            with mock.patch.object(A.D, "load_visual", return_value=(visual, visual_audit)):
                report = A.analyze(config_path)
            self.assertTrue(report["accepted"])
            self.assertEqual(report["decision"], "supported_single_seed_binding_prerequisite")
            self.assertEqual(len(report["summaries"]), 15)
            self.assertEqual(report["external_controls"]["cached_visual"]["analytic"]["raw"]["correct"], 1024)
            self.assertEqual(report["contrasts"]["final_minus_initial/heldout"]["canonical"]["accuracy_delta"], .75)
            self.assertEqual(report["summaries"]["final-cached-visual-l4"]["ordering"]["applicable"], False)
            for key in A.CHECKS:
                bad = copy.deepcopy(receipt); bad["checks"][key] = 1
                new_path, new_sha = save("receipt", bad)
                config["integrity_receipt_sha256"] = new_sha
                config["frozen_files"][new_path] = new_sha
                with self.assertRaises(ValueError):
                    A.configuration(config)
            bad = copy.deepcopy(receipt)
            bad["streams"]["final-fit-l4"]["parameter_sha256"] = "e" * 64
            new_path, new_sha = save("receipt", bad)
            config["integrity_receipt_sha256"] = new_sha; config["frozen_files"][new_path] = new_sha
            with self.assertRaisesRegex(ValueError, "checkpoint binding"):
                A.configuration(config)


if __name__ == "__main__":
    print(f"synthetic scorer test PID={os.getpid()}", flush=True)
    unittest.main(verbosity=2)
