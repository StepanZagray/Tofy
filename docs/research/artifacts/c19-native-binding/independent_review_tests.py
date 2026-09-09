#!/usr/bin/env python3
"""C19 scalar/data fixtures. No generator, model, historical or C19 outcome reads."""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import signal
import struct
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
_spec = importlib.util.spec_from_file_location("c19_independent_under_test", Path(__file__).with_name("independent_review.py"))
I = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(I)
FIXTURE_SEED, FIXTURE_BASE = 12345, 0x54455354


def frame(agent, goal=63, walls=False):
    values = [int(walls and (c % 8 in (0, 7) or c//8 in (0, 7))) for c in range(64)]
    values[agent], values[goal] = 2, 3
    return values


def fixture_panel(query_direction=None):
    rows = []
    actions = (0, 1, 2)
    metadata = []
    for f in range(7):
        for c in range(64):
            m = [0.0]*10; m[f%2 if f < 6 else 2] = 1.0
            if f < 6: m[3+actions[f//2]] = 1.0
            m[7:] = [c%8/7, c//8/7, f//2/3 if f < 6 else 1.0]
            metadata.extend(m)
    packed = struct.pack("<4480f", *metadata)
    metadata = list(struct.unpack("<4480f", packed))
    for group in range(32):
        direction = group % 4 if query_direction is None else query_direction
        dx, dy = I.DIRECTIONS[direction]
        query = frame(27, 27+dx+8*dy, True)
        for bit, cell in enumerate((9, 10, 11, 12, 13)):
            query[cell] = (group >> bit) & 1
        for mapping, controls in enumerate(I.MAPS):
            cells = []; agent = 27
            for action in actions:
                cells.append(frame(agent)); dx, dy = I.DIRECTIONS[controls[action]]
                agent += dx+8*dy; cells.append(frame(agent))
            cells.append(list(query))
            raw = b"".join(struct.pack("<I", c)*64 for f in cells for c in f)
            rows.append(dict(input_index=group*24+mapping, query_index=group, episode_id=FIXTURE_BASE+group,
                             data_seed=FIXTURE_SEED, permutation_id=mapping, policy_label=controls.index(direction),
                             public_cells=cells, public_metadata=metadata,
                             input_sha256=hashlib.sha256(raw+packed).hexdigest(),
                             query_sha256=hashlib.sha256(raw[-16384:]).hexdigest(),
                             metadata_sha256=hashlib.sha256(packed).hexdigest()))
    return rows


def attention(row, soft=False, shifted=False):
    result = []
    for f, cells in enumerate(row["public_cells"]):
        roles = []
        for role in (2, 3):
            index = cells.index(role)
            if shifted and (role == 2 or f == 6): index += 1
            roles.append([(0.9 if c == index else 0.0)+0.1/64 if soft else float(c == index) for c in range(64)])
        result.append(roles)
    return result


def runtime(rows, control="factual", soft=False):
    result = []
    for i, row in enumerate(rows):
        learned = attention(row, soft)
        used = learned if control == "factual" else [[[1/64]*64 for _ in range(2)] for _ in range(7)]
        # Independent scalar coordinate formula in the fixture as well.
        points = [[[sum(role[c]*(c%8 if axis == 0 else c//8) for c in range(64)) for axis in range(2)] for role in f] for f in used]
        records = []
        for step in range(3):
            records.append([points[2*step+1][0][a]-points[2*step][0][a] for a in range(2)]+[float(a == step) for a in range(4)]+[0.0])
        records.append([points[6][1][a]-points[6][0][a] for a in range(2)]+[0.0]*4+[1.0])
        winner = row["policy_label"] if control == "factual" else 3
        result.append({**row, "evaluation_index": i, "learned_attention": learned, "adapter_records": records,
                       "logits": [4.0 if a == winner else 0.0 for a in range(4)], "prediction": winner, "control": control})
    return result


class FixtureCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = fixture_panel()

    def setUp(self):
        self.seed = mock.patch.multiple(I, SEED=FIXTURE_SEED, BASE=FIXTURE_BASE)
        self.seed.start(); self.addCleanup(self.seed.stop)

    def validated(self, rows=None, history=None):
        with mock.patch.object(I, "jsonl", return_value=iter(rows or self.rows)):
            return I.panel("synthetic", history or set())


class PublicData(FixtureCase):
    def test_complete_geometry_hashes_and_balance(self):
        rows, truths, report = self.validated()
        self.assertEqual(report, dict(rows=768, groups=32, label_histogram=[192]*4, direction_histogram=[8]*4, demonstrated=576, omitted=192))
        self.assertEqual(len({r["input_sha256"] for r in rows}), 768)
        self.assertEqual(sum(t["demonstrated"] for t in truths), 576)

    def test_hash_label_metadata_and_hidden_geometry_mutations(self):
        mutations = [lambda r: r.__setitem__("policy_label", (r["policy_label"]+1)%4),
                     lambda r: r.__setitem__("policy_label", True),
                     lambda r: r.__setitem__("data_seed", FIXTURE_SEED+1),
                     lambda r: r.__setitem__("episode_id", FIXTURE_BASE+1),
                     lambda r: r.__setitem__("input_sha256", "0"*64),
                     lambda r: r.__setitem__("query_sha256", "0"*64),
                     lambda r: r["public_metadata"].__setitem__(9, .1),
                     lambda r: r["public_cells"][0].__setitem__(9, 1),
                     lambda r: r["public_cells"][6].__setitem__(0, 0),
                     lambda r: r["public_cells"][1].__setitem__(9, 3)]
        for mutate in mutations:
            row = copy.deepcopy(self.rows[0]); mutate(row)
            with self.assertRaises(I.Invalid): I.truth(row, 0)

    def test_collision_incomplete_and_unpaired_panels(self):
        with self.assertRaisesRegex(I.Invalid, "overlap"): self.validated(history={self.rows[0]["query_sha256"]})
        with self.assertRaises(I.Invalid): self.validated(self.rows[:-1])
        rows = list(self.rows); rows[1] = rows[0]
        with self.assertRaises(I.Invalid): self.validated(rows)
        with self.assertRaises(I.Invalid): self.validated(rows + [rows[0]])
        # Valid geometry with a repeated query in a different group still fails.
        rows = copy.deepcopy(self.rows)
        for i in range(24):
            source = copy.deepcopy(rows[i]); source.update(input_index=96+i, query_index=4, episode_id=FIXTURE_BASE+4)
            rows[96+i] = source
        with self.assertRaisesRegex(I.Invalid, "overlap"): self.validated(rows)

    def test_strict_json_and_array_types(self):
        for raw in ('{"x":1,"x":1}', '{"x":NaN}'):
            with self.assertRaises(I.Invalid): I.decode(raw)
        for value in ([True]*4, [float("inf")]*4, [1.0]*3):
            with self.assertRaises(I.Invalid): I.array(value, (4,))

    def test_action_balance_does_not_replace_physical_direction_coverage(self):
        with self.assertRaisesRegex(I.Invalid, "four query directions"):
            self.validated(fixture_panel(query_direction=0))


class Scoring(FixtureCase):
    def test_stable_ce_margin_and_first_tie(self):
        m = I.metric([1000.0]*4, 0)
        self.assertAlmostEqual(m["ce"], math.log(4)); self.assertEqual(m["winner_margin"], 0)
        self.assertEqual(m["prediction"], 0); self.assertEqual(m["margin"], 0)
        self.assertEqual(I.metric([3.,2.,1.,0.], 3)["margin"], -3)

    def test_perfect_and_uniform_full_scalar_pipeline(self):
        rows, truths, _ = self.validated()
        summaries = {}
        for control in ("factual", "uniform_attention"):
            with mock.patch.object(I, "jsonl", return_value=iter(runtime(rows, control))):
                result = I.stream("synthetic", rows, truths, control)
            summary = I.action_summary(rows, truths, result["metrics"]); summaries[control] = summary
            self.assertEqual(summary["correct"], 768 if control == "factual" else 192)
            self.assertEqual(summary["all_maps_correct_groups"], 32 if control == "factual" else 0)
            self.assertEqual(summary["map_splits"]["familiar"]["rows"], 512)
            self.assertEqual(summary["map_splits"]["heldout"]["rows"], 256)
            if control != "factual":
                self.assertEqual(summary["missing_id_predictions"], 768)
                self.assertEqual(summary["subsets"]["demonstrated"]["correct"], 0)
                self.assertEqual(summary["subsets"]["omitted"]["correct"], 192)
                self.assertEqual(I.uniform_groups(result)["winner_disagreements"], 0)
            ground = I.grounding(truths, result["effective"])
            self.assertEqual(ground["all_roles_correct"], 768 if control == "factual" else 0)
            self.assertEqual(ground["frames"][0]["agent"]["mean_mass"], 1 if control == "factual" else 1/64)
        self.assertLess(summaries["factual"]["ce"], summaries["uniform_attention"]["ce"])

    def test_soft_locations_and_correct_delta_do_not_imply_correct_locations(self):
        _, truths, _ = self.validated()
        soft = I.grounding(truths, [attention(r, soft=True) for r in self.rows])
        self.assertEqual(soft["all_roles_correct"], 768)
        self.assertGreater(soft["frames"][0]["agent"]["mean_position_l1_error"], 0)
        shifted = I.grounding(truths, [attention(r, shifted=True) for r in self.rows])
        self.assertEqual(shifted["consumed_roles_correct"], 0)
        self.assertTrue(all(d["correct_argmax_delta"] == 768 and d["correct_argmax_locations"] == 0 for d in shifted["displacements"]))

    def test_runtime_identity_record_prediction_and_attention_tamper(self):
        rows, truths, _ = self.validated(); values = runtime(rows)
        mutations = [lambda r: r.__setitem__("control", "uniform_attention"),
                     lambda r: r.__setitem__("evaluation_index", True),
                     lambda r: r.__setitem__("prediction", (r["prediction"]+1)%4),
                     lambda r: r["adapter_records"][0].__setitem__(0, 0.125),
                     lambda r: r["adapter_records"][0].__setitem__(2, 1.000001),
                     lambda r: r["learned_attention"][0][0].__setitem__(0, -1e-7),
                     lambda r: r["logits"].__setitem__(0, float("nan"))]
        for mutate in mutations:
            values[0] = copy.deepcopy(runtime(rows[:1])[0]); mutate(values[0])
            with mock.patch.object(I, "jsonl", return_value=iter(values)):
                with self.assertRaises(I.Invalid): I.stream("synthetic", rows, truths, "factual")

    def test_uniform_wrong_record_and_winner_are_integrity_failures(self):
        rows, truths, _ = self.validated()
        with mock.patch.object(I, "jsonl", return_value=iter(runtime(rows, "uniform_attention"))):
            result = I.stream("synthetic", rows, truths, "uniform_attention")
        changed = copy.deepcopy(result); changed["records"][1][0][0] = 1e-7
        with self.assertRaises(I.Invalid): I.uniform_groups(changed)
        changed = copy.deepcopy(result); changed["logits"][1] = [4.,0.,0.,0.]
        changed["metrics"][1] = I.metric(changed["logits"][1], rows[1]["policy_label"])
        with self.assertRaises(I.Invalid): I.uniform_groups(changed)


class Files(unittest.TestCase):
    def test_complete_history_union_and_hash_pin(self):
        with tempfile.TemporaryDirectory(prefix="c19-history-test-") as temporary:
            root = Path(temporary); sources = []; pins = []
            for i in range(6):
                p = root/f"source-{i}.jsonl"
                p.write_text('\n'.join(json.dumps(dict(query_sha256=I.sha(str(j).encode()), episode_id=j)) for j in (i, i+1))+'\n')
                checksum = I.file_sha(p); sources.append(dict(path=str(p), sha256=checksum)); pins.append((checksum, 2))
            manifest = root/"history.json"; manifest.write_text(json.dumps(dict(history=sources, created_local="fixture", scope="synthetic")))
            record = dict(path=str(manifest), sha256=I.file_sha(manifest))
            with mock.patch.object(I, "HISTORY", tuple(pins)):
                union, counts = I.histories(record)
                self.assertEqual(len(union), 7); self.assertEqual(len(counts), 6)
                self.assertTrue(all(r["parsed_rows"] == r["unique_query_hashes"] == 2 for r in counts))
                (root/"source-0.jsonl").write_text('{}\n')
                with self.assertRaises(I.Invalid): I.histories(record)

    def test_changed_report_discrete_float_and_symlink_fail(self):
        expected = {"correct": 3, "ce": 1.2, "gate": True}
        self.assertEqual(I.compare(expected, expected)["compared_float_fields"], 1)
        for changed in ({**expected, "correct": 3.0}, {**expected, "ce": 1.21}, {**expected, "gate": 1}, {**expected, "ce": float("nan")}):
            with self.assertRaises(I.Invalid): I.compare(changed, expected)
        with tempfile.TemporaryDirectory(prefix="c19-file-test-") as temporary:
            root = Path(temporary); p = root/"a"; p.write_text("fixture"); link = root/"link"; link.symlink_to(p)
            with self.assertRaises(I.Invalid): I.file_sha(link)


def write(root, name, value):
    p = root/name
    p.write_text(json.dumps(value) if not isinstance(value, str) else value)
    return {"path": str(p), "sha256": I.file_sha(p)}


def replay_evidence():
    numerical = dict(maximum_absolute_error=0.0, maximum_tolerance_ratio=0.0)
    logits = dict(numerical, rows=768, eligible_winners=768, ineligible_winners=0, eligible_winner_mismatches=0,
                  eligibility="reference winner margin > 0 and > 2 * row maximum absolute logit error")
    return {name: {"records": dict(numerical), "logits": dict(logits)} for name in ("factual", "uniform")}


class ReviewSeams(FixtureCase):
    def test_gate_conjunction_is_separate_from_grounding_and_shared_replay(self):
        rows, truths, _ = self.validated(); summaries = {}
        for name, control in (("factual", "factual"), ("uniform", "uniform_attention")):
            values = runtime(rows, control)
            metrics = [I.metric(r["logits"], r["policy_label"]) for r in values]
            summaries[name] = I.action_summary(rows, truths, metrics)
        control = dict(uniform_identical_records_within_query=True, uniform_identical_winners_within_query=True, uniform_six_correct_per_query=True)
        self.assertEqual(I.decision(summaries, control)[1], "supported_frozen_native_composition")
        for field, value in (("correct", 767), ("minimum_true_margin", .00099), ("all_maps_correct_groups", 31)):
            changed = copy.deepcopy(summaries); changed["factual"][field] = value
            self.assertEqual(I.decision(changed, control)[1], "frozen_native_composition_not_supported")
        self.assertEqual(I.decision(summaries, {**control, "uniform_six_correct_per_query": False})[1], "frozen_native_composition_not_supported")
        proof = replay_evidence()["factual"]["logits"]
        for key, value in (("maximum_tolerance_ratio", 1.01), ("rows", True), ("eligible_winner_mismatches", 1), ("eligible_winners", 767)):
            with self.assertRaises(I.Invalid): I.trusted_neural_replay({**proof, key: value})
        ties = I.logit_parity([[0.,0.,0.,0.]], [[0.,0.,0.,0.]])
        self.assertEqual(ties["eligible_winners"], 0)
        near = I.logit_parity([[0.,1e-6,0.,0.]], [[1e-6,0.,0.,0.]])
        self.assertEqual(near["eligible_winners"], 0)

    def test_full_review_provenance_and_scalar_report_seam(self):
        with tempfile.TemporaryDirectory(prefix="c19-review-seam-") as temporary:
            root = Path(temporary)
            specs = {name: write(root, name, "synthetic") for name in ("registration.md", "history.json", "panel.jsonl", "checkpoint", "vision", "selector", "factual.jsonl", "uniform.jsonl")}
            for name in ("analysis.py", "analysis_tests.py", "numpy_binding.py", "numpy_binding_tests.py", "independent_review.py", "independent_review_tests.py"):
                specs[name] = write(root, name, "synthetic source "+name)
            files = {s["path"]: s["sha256"] for s in specs.values()}
            # These are file-binding fixtures, never model parameters or model execution.
            with mock.patch.multiple(I, R=root, BINDER=specs["checkpoint"]["sha256"], VISION=specs["vision"]["sha256"], SELECTOR=specs["selector"]["sha256"]):
                receipt = dict(schema="looped-native-binding-integrity-v1", accepted=True, checks=dict.fromkeys(I.CHECKS, True), source_revision="1"*40,
                               binary_sha256="2"*64, dependency_revision=I.DEPENDENCY, vision_checkpoint_sha256=I.VISION, selector_sha256=I.SELECTOR,
                               binder_checkpoint_sha256=I.BINDER, vision_loops=4, binder_loops=4, frozen_files=files)
                rec = write(root, "receipt.json", receipt)
                config = dict(schema="looped-native-binding-analysis-config-v1", registration=specs["registration.md"], history=specs["history.json"],
                              audit=specs["panel.jsonl"], checkpoint=specs["checkpoint"], integrity_receipt=rec,
                              streams={name: specs[name+".jsonl"] for name in ("factual", "uniform")}, frozen_files={**files, rec["path"]: rec["sha256"]})
                cfg = write(root, "config.json", config)
                self.assertTrue(I.authority(config)["accepted"])
                changed = copy.deepcopy(config); changed["frozen_files"].pop(specs["selector"]["path"])
                with self.assertRaises(I.Invalid): I.authority(changed)
                values = {specs["panel.jsonl"]["path"]: self.rows, specs["factual.jsonl"]["path"]: runtime(self.rows), specs["uniform.jsonl"]["path"]: runtime(self.rows, "uniform_attention")}
                source = lambda p: iter(values[str(p)])
                with mock.patch.object(I, "jsonl", side_effect=source), mock.patch.object(I, "histories", return_value=(set(), [])):
                    expected = I.reconstruction(config, replay_evidence())
                    report = {**expected, "schema": "looped-native-binding-analysis-v1", "accepted": True, "config_sha256": cfg["sha256"],
                              "registration_sha256": config["registration"]["sha256"], "source_revision": "1"*40, "binary_sha256": "2"*64,
                              "binder_checkpoint_sha256": I.BINDER, "limits": ["synthetic, no neural replay"], "elapsed_seconds": 1.0, "pid": 12345}
                    rep = write(root, "report.json", report)
                    answer = I.review(Path(cfg["path"]), Path(rep["path"]))
                    self.assertTrue(answer["accepted"])
                    self.assertGreater(answer["compared_float_fields"], 100)
                    # Expected numerical fields exclude both shared neural replay dictionaries.
                    independent = {**expected, "replay": {n: {"records": v["records"]} for n, v in expected["replay"].items()}}
                    self.assertEqual(answer["compared_float_fields"], I.compare(independent, independent)["compared_float_fields"])
                    mutated = copy.deepcopy(report); mutated["summaries"]["factual"]["correct"] = 767
                    write(root, "report.json", mutated)
                    with mock.patch.object(I, "reconstruction", return_value=expected):
                        with self.assertRaises(I.Invalid): I.review(Path(cfg["path"]), Path(rep["path"]))

    def test_premise_real_manifest_seam_and_changed_inventory(self):
        with tempfile.TemporaryDirectory(prefix="c19-premise-seam-") as temporary:
            root = Path(temporary); audit_root = root/"audit"; audit_root.mkdir()
            sources = []
            audit = dict(status="complete_pending_analysis", classification="input_audit", model_forwards=0, optimizer_updates=0,
                         input_rows=768, query_groups=32, panel_seed=FIXTURE_SEED, panel_tag=FIXTURE_BASE,
                         label_counts=[192]*4, history=sources, excluded_unique_queries=0)
            metadata = dict(schema="looped-native-binding-v1", device="cpu", objective="public_input_audit", optimizer_updates=0, model_forwards=0,
                            provenance=dict(source_revision="1"*40, binary_sha256=I.sha(b"synthetic executable binding"), candle_graph_revision=I.DEPENDENCY))
            entries = {name: write(audit_root, name, audit if name == "report.json" else metadata if name == "metadata.json" else "synthetic") for name in ("launch.json", "metadata.json", "panel-rows.jsonl", "profiles.json", "report.json")}
            manifest = write(audit_root, "manifest.json", dict(schema="looped-grounded-policy-artifacts-v1", files={name: s["sha256"] for name, s in entries.items()}))
            pin = write(root, "audit.manifest.sha256", manifest["sha256"]+'\n')
            history = write(root, "history.json", "synthetic"); registration = write(root, "registration.md", "synthetic")
            binary = write(root, "binary", "synthetic executable binding")
            scripts = [write(root, name, "synthetic source") for name in ("independent_review.py", "independent_review_tests.py")]
            frozen = {s["path"]: s["sha256"] for s in [*entries.values(), manifest, pin, history, registration, binary, *scripts]}
            config = dict(schema="looped-native-binding-premise-v1", audit=entries["panel-rows.jsonl"], audit_manifest=manifest, history=history,
                          registration=registration, source_revision="1"*40, binary_sha256=binary["sha256"], frozen_files=frozen)
            cfg = write(root, "config.json", config)
            with mock.patch.object(I, "R", root), mock.patch.object(I, "jsonl", return_value=iter(self.rows)), mock.patch.object(I, "histories", return_value=(set(), sources)):
                result = I.premise(Path(cfg["path"]))
                self.assertTrue(result["accepted"]); self.assertEqual(result["panel"]["rows"], 768)
                write(root, "config.json", {**config, "source_revision": "2"*40})
                with self.assertRaisesRegex(I.Invalid, "source/binary/dependency"):
                    I.premise(Path(cfg["path"]))
            (audit_root/"extra").write_text("unexpected")
            with self.assertRaises(I.Invalid): I.audit_manifest(config["audit"], manifest)


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("60-second fixture bound")))
    signal.alarm(60)
    unittest.main()
