#!/usr/bin/env python3
"""Synthetic C12 evaluator fixtures. No episode generator or actual campaign data."""
import copy
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location("c12_analyze", Path(__file__).with_name("analyze.py"))
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)
np = A.np


def binding(path):
    return {"path": str(path), "sha256": A.sha_bytes(path.read_bytes())}


def write_json(path, value):
    path.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")


def fixture_row(cohort, group, map_id, slot):
    """Hand-constructed visible cells and audit records, in a fake seed/id namespace."""
    cells = [1 if i % 8 in (0, 7) or i // 8 in (0, 7) else 0 for i in range(64)]
    agent = 27
    direction = group % 4
    goal = agent + (-8, 8, -1, 1)[direction]
    wall_slots = [i for i, color in enumerate(cells) if color == 0 and i not in (agent, 19, 35, 26, 28)][:8]
    pattern = group + (128 if cohort == "seen" else 0)
    for bit, cell in enumerate(wall_slots):
        cells[cell] = (pattern >> bit) & 1
    cells[agent], cells[goal] = 2, 3
    controls = list(A.PERMUTATIONS[map_id])
    action = controls.index(direction)
    actions = [(group + i) % 4 for i in range(3)]
    omitted = (group + 3) % 4
    patches = b"".join(struct.pack("<I", color) * 64 for color in cells)
    metadata = A.public_metadata(tuple(actions))
    cleared = A.sha_bytes(bytes(6 * 4096 * 4) + patches + metadata)
    factual = A.sha_bytes(f"synthetic-factual-{cohort}-{group}-{map_id}".encode())
    training_group = group * 4599 // 63 if cohort == "seen" else None
    return {
        "schema": A.DATA_SCHEMA, "cohort": cohort, "condition": "factual", "support_cleared": False,
        "row_index": group * len(A.MAPS[cohort]) + slot, "group_index": group,
        "training_group_index": training_group,
        "original_update": training_group // 4 + 1 if training_group is not None else None,
        "data_seed": 101 if cohort == "seen" else 102,
        "episode_id": 100000 + training_group if cohort == "seen" else 200000 + group,
        "permutation_id": map_id, "correct_action": action, "query_direction": direction,
        "agent_patch": agent, "goal_patch": goal, "observed_support_action_ids": actions,
        "omitted_action": omitted, "correct_action_demonstrated": action in actions,
        "inferred_controls": controls, "input_sha256": factual, "factual_input_sha256": factual,
        "cleared_input_sha256": cleared, "metadata_sha256": A.sha_bytes(metadata),
        "query_sha256": A.sha_bytes(patches), "targets_sha256": A.sha_bytes(f"targets-{cohort}-{group}-{map_id}".encode()),
        "label_sha256": A.sha_bytes(struct.pack("<I", action)), "query_cells": cells,
    }


def fake_output(identity, stage):
    row = copy.deepcopy(identity)
    pred = row["correct_action"] if stage == "final_factual" else 0
    row["logits"] = [2.0 if action == pred else 0.0 for action in range(4)]
    row["attention"] = [[1.0 if p == row[field] else 0.0 for p in range(64)]
                        for field in ("agent_patch", "goal_patch")]
    row["pooled"] = [0.0] * 256
    row["checkpointstage"] = "frozen" if stage == "frozen_factual" else "final"
    return row


class Fixture:
    def __init__(self, root):
        self.root = root
        self.rows = {}
        audits, streams = {}, {}
        for cohort, maps in A.MAPS.items():
            factual = [fixture_row(cohort, group, map_id, slot)
                       for group in range(64) for slot, map_id in enumerate(maps)]
            cleared = [dict(row, condition="cleared", support_cleared=True,
                            input_sha256=row["cleared_input_sha256"]) for row in factual]
            for condition, population in (("factual", factual), ("cleared", cleared)):
                selector = f"{cohort}/{condition}"
                self.rows[selector] = population
                path = root / f"audit-{cohort}-{condition}.jsonl"
                self.write_rows(path, population)
                audits[selector] = binding(path)
        source = {"revision": "a" * 40, "binary_sha256": "b" * 64, "dependency_revision": A.DEPENDENCY}
        checkpoints = {"frozen": {"core_sha256": A.INITIAL_CORE, "policy_sha256": A.INITIAL_POLICY, "updates": 0},
                       "final": {"core_sha256": "c" * 64, "policy_sha256": "d" * 64, "updates": 1150}}
        for stage in A.STAGES:
            for cohort in A.MAPS:
                condition = "cleared" if stage == "final_cleared" else "factual"
                checkpointstage = "frozen" if stage == "frozen_factual" else "final"
                selector = f"{stage}/{cohort}"
                self.rows[selector] = [fake_output(row, stage) for row in self.rows[f"{cohort}/{condition}"]]
                path = root / f"output-{stage}-{cohort}.jsonl"
                self.write_rows(path, self.rows[selector])
                streams[selector] = dict(binding(path), cohort=cohort, condition=condition,
                                         checkpointstage=checkpointstage, source=source,
                                         checkpoint=checkpoints[checkpointstage],
                                         audit_sha256=audits[f"{cohort}/{condition}"]["sha256"])
        reg = root / "synthetic-registration.md"
        reg.write_text("Synthetic fixture; no experiment population or model output.\n")
        self.certificate = {
            "schema": "looped-grounded-policy-integrity-v1", "accepted": True, "campaign": str(root),
            "source": source, "registration_sha256": binding(reg)["sha256"], "checkpoints": checkpoints,
            "audit_sha256": {k: v["sha256"] for k, v in audits.items()},
            "stream_sha256": {k: v["sha256"] for k, v in streams.items()},
            "checks": {k: True for k in A.CHECKS},
            "parameter_changes": {"core_changed_elements": 1, "policy_changed_elements": 1, "unused_changed_elements": 0},
            "gradient_norms": {"qualification_body": 1.0, "qualification_head": 1.0},
        }
        cert = root / "integrity.json"
        write_json(cert, self.certificate)
        self.config = {"schema": A.SCHEMA, "campaign": str(root), "source": source,
                       "registration": binding(reg), "checkpoints": checkpoints,
                       "audits": audits, "streams": streams, "integrity": binding(cert)}
        self.config_path = root / "config.json"
        self.seal_config()

    @staticmethod
    def write_rows(path, population):
        path.write_text("".join(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n" for row in population))

    def seal_config(self):
        write_json(self.config_path, self.config)
        self.config_sha = binding(self.config_path)["sha256"]


class Numerics(unittest.TestCase):
    def test_public_metadata_matches_independent_scalar_encoding(self):
        actual = A.public_metadata((2, 0, 3))
        expected = bytearray()
        for frame in range(7):
            step, role = (frame // 2, frame % 2) if frame < 6 else (3, 2)
            for patch in range(64):
                row = [0.0] * 10
                row[role] = 1.0
                if frame < 6:
                    row[3 + (2, 0, 3)[step]] = 1.0
                row[7], row[8], row[9] = patch % 8 / 7, patch // 8 / 7, step / 3
                expected.extend(struct.pack("<10f", *row))
        self.assertEqual(actual, bytes(expected))

    def test_registered_balance(self):
        for maps in A.MAPS.values():
            for action in range(4):
                self.assertEqual([sum(A.PERMUTATIONS[p][action] == d for p in maps) for d in range(4)], [len(maps) // 4] * 4)

    def test_stable_ce_and_role_metrics(self):
        identities = [fixture_row("heldout", g, p, i) for g in range(64) for i, p in enumerate(A.HELDOUT)]
        outputs = [fake_output(row, "final_factual") for row in identities]
        for row in outputs:
            row["logits"] = [10000.0 if a == row["correct_action"] else -10000.0 for a in range(4)]
        summary, grouped = A.measure(outputs, identities, "final")
        self.assertEqual(summary["accuracy"], 1.0)
        self.assertEqual(summary["ce"], 0.0)
        self.assertEqual(grouped["joint_role_accuracy"].tolist(), [1.0] * 64)

    def test_bootstrap_resamples_whole_query_groups_with_shared_draws(self):
        values = np.arange(64, dtype=float) / 63
        draws = np.random.Generator(np.random.PCG64(1932)).integers(0, 64, (10000, 64))
        result = A.bootstrap({"x": values, "twice": 2 * values}, draws)
        manual = np.quantile([sum(values[int(i)] for i in draw) / 64 for draw in draws], [.025, .975], method="linear")
        np.testing.assert_allclose(result["x"]["ci95"], manual, rtol=0, atol=1e-15)
        np.testing.assert_array_equal(result["twice"]["ci95"], 2 * np.asarray(result["x"]["ci95"]))

    def test_classification_conjunction_and_strict_bounds(self):
        summaries = {f"final_factual/{c}": {"accuracy": 1.0} for c in A.MAPS}
        contrasts = {c: {"final_minus_frozen": {"accuracy": {"ci95": [.1, .3]}},
                         "final_minus_cleared": {"accuracy": {"ci95": [.3, .5]}}} for c in A.MAPS}
        self.assertEqual(A.decide(summaries, contrasts)[0], "supported_single_seed_screen")
        summaries["final_factual/seen"]["accuracy"] = .89
        self.assertEqual(A.decide(summaries, contrasts)[0], "training_feasibility_not_supported")
        summaries["final_factual/seen"]["accuracy"] = .90
        contrasts["heldout"]["final_minus_frozen"]["accuracy"]["ci95"][0] = 0
        self.assertEqual(A.decide(summaries, contrasts)[0], "transfer_not_supported")
        contrasts["heldout"]["final_minus_frozen"]["accuracy"]["ci95"][0] = .1
        contrasts["familiar"]["final_minus_cleared"]["accuracy"]["ci95"][0] = .25
        self.assertEqual(A.decide(summaries, contrasts)[0], "transfer_not_supported")


class FullSeam(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="c12-analyzer-synthetic-")
        cls.fixture = Fixture(Path(cls.temp.name))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.patches = mock.patch.multiple(A, TRAIN_SEED=101, EVAL_SEED=102, TRAIN_BASE=100000, EVAL_BASE=200000)
        self.patches.start()

    def tearDown(self):
        self.patches.stop()

    def test_complete_pinned_nine_stream_positive(self):
        result = A.analyze(self.fixture.config_path, self.fixture.config_sha)
        self.assertIs(result["accepted"], True)
        self.assertIs(result["supported"], True)
        self.assertEqual(len(result["summaries"]), 9)
        self.assertTrue(all(result["gates"].values()))
        self.assertEqual(result["summaries"]["final_factual/heldout"]["correct"], 512)
        self.assertEqual(result["summaries"]["final_cleared/seen"]["correct"], 256)
        self.assertEqual(result["contrasts"]["heldout"]["final_minus_frozen"]["accuracy"]["ci95"], [.75, .75])

    def test_hash_tamper_rejected_before_streams(self):
        with self.assertRaisesRegex(A.Invalid, "hash mismatch"):
            A.analyze(self.fixture.config_path, "0" * 64)

    def test_config_selectors_source_updates_and_alias_tampering(self):
        for mutate in (
            lambda c: c["streams"].pop("final_factual/heldout"),
            lambda c: c["streams"]["final_factual/heldout"].update(checkpointstage="frozen"),
            lambda c: c["source"].update(dependency_revision="0" * 40),
            lambda c: c["checkpoints"]["final"].update(updates=True),
            lambda c: c["streams"]["final_factual/heldout"].update(path=c["streams"]["frozen_factual/heldout"]["path"]),
        ):
            config = copy.deepcopy(self.fixture.config)
            mutate(config)
            with self.subTest(config=config["checkpoints"]), self.assertRaises(A.Invalid):
                A.validate_config(config, self.fixture.root)

    def test_integrity_boolean_and_closure_fail_closed(self):
        config = copy.deepcopy(self.fixture.config)
        for field, replacement in (("accepted", 1), ("accepted", False), ("source", {}), ("stream_sha256", {})):
            cert = copy.deepcopy(self.fixture.certificate)
            cert[field] = replacement
            with mock.patch.object(A, "pinned", return_value=json.dumps(cert).encode()), self.assertRaises(A.Invalid):
                A.certificate(config)
        for check in A.CHECKS:
            cert = copy.deepcopy(self.fixture.certificate)
            cert["checks"][check] = 1
            with mock.patch.object(A, "pinned", return_value=json.dumps(cert).encode()), self.assertRaises(A.Invalid):
                A.certificate(config)

    def test_identity_geometry_order_dtype_and_hash_tampering(self):
        original = self.fixture.rows["seen/factual"][0]
        for field, replacement in (("correct_action", (original["correct_action"] + 1) % 4),
                                    ("episode_id", 123), ("permutation_id", 23), ("group_index", True),
                                    ("metadata_sha256", "0" * 64), ("cleared_input_sha256", "0" * 64),
                                    ("query_cells", [0] * 64), ("agent_patch", True),
                                    ("observed_support_action_ids", [0, 0, 1])):
            row = copy.deepcopy(original); row[field] = replacement
            with self.subTest(field=field), self.assertRaises(A.Invalid):
                A.validate_identity(row, "seen", "factual", 0)

    def test_output_shapes_nonfinite_and_identity_tampering(self):
        identity = self.fixture.rows["heldout/factual"]
        base = self.fixture.rows["final_factual/heldout"]
        for field, replacement in (("logits", [0, 1, 2]), ("logits", [True, 0, 0, 0]),
                                    ("logits", [float("nan"), 0, 0, 0]),
                                    ("attention", [[0.0] * 64] * 2), ("pooled", [0.0] * 255),
                                    ("input_sha256", "0" * 64), ("checkpointstage", "frozen")):
            outputs = list(base); outputs[0] = dict(base[0], **{field: replacement})
            with self.subTest(field=field), self.assertRaises(A.Invalid):
                A.measure(outputs, identity, "final")

    def test_cleared_numeric_tolerance_and_exact_argmax(self):
        identity = self.fixture.rows["heldout/cleared"]
        base = self.fixture.rows["final_cleared/heldout"]
        outputs = copy.deepcopy(base)
        outputs[1]["logits"][0] += 1e-6
        self.assertEqual(A.measure(outputs, identity, "final")[0]["accuracy"], .25)
        outputs[1]["logits"][0] += .01
        with self.assertRaisesRegex(A.Invalid, "cleared logits"):
            A.measure(outputs, identity, "final")
        outputs = copy.deepcopy(base)
        for row in outputs:
            row["logits"] = [0.0] * 4
        outputs[1]["logits"][1] = 1e-7
        with self.assertRaisesRegex(A.Invalid, "cleared predictions"):
            A.measure(outputs, identity, "final")

    def test_json_duplicates_and_nonfinite_rejected(self):
        for raw in ('{"a":1,"a":2}', '{"value":NaN}', '{"value":Infinity}'):
            with self.assertRaises(A.Invalid):
                A.decode(raw)


if __name__ == "__main__":
    unittest.main()
