#!/usr/bin/env python3
"""Synthetic C18 reviewer tests; no model, campaign or C15 outcome reads."""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
import copy
from collections import Counter
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np

sys.dont_write_bytecode = True


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


I = module("c18_independent_tested", "independent_review.py")
# Only the pure finite-data factory is compared. No data.load_visual, primary
# scoring function, or actual artifact is called by these fixtures.
D = module("c18_data_fixture_factory", "data.py")


def visual_source():
    cells, position = [], 27
    for offset in (-8, 8, -1):
        before = [0]*64; before[63] = 3; before[position] = 2
        position += offset
        after = [0]*64; after[63] = 3; after[position] = 2
        cells.extend((before, after))
    query = [1 if x in (0, 7) or y in (0, 7) else 0 for y in range(8) for x in range(8)]
    query[27], query[19] = 2, 3
    cells.append(query)
    metadata = [0.0]*4480
    patches = b"".join(struct.pack("<I", c)*64 for frame in cells for c in frame)
    meta_bytes = struct.pack("<4480f", *metadata)
    audit = {k: "0"*64 for k in I.AUDIT_FIELDS}
    audit.update(schema="looped-grounded-policy-data-v1", cohort="seen", condition="factual", support_cleared=False,
                 row_index=0, group_index=0, training_group_index=0, original_update=1, data_seed=20260920,
                 episode_id=0x47524F554E445452, permutation_id=0, correct_action=0, query_direction=0,
                 agent_patch=27, goal_patch=19, observed_support_action_ids=[0,1,2], omitted_action=3,
                 correct_action_demonstrated=True, inferred_controls=[0,1,2,3], query_cells=query,
                 input_sha256=I.sha(patches+meta_bytes), factual_input_sha256=I.sha(patches+meta_bytes),
                 metadata_sha256=I.sha(meta_bytes), query_sha256=I.sha(patches[-16384:]), label_sha256=I.sha(struct.pack("<I",0)))
    frames = [{"cells": c, "attention": [[float(i==c.index(role)) for i in range(64)] for role in (2,3)], "pooled": [0.0]*256} for c in cells]
    return {**audit, "checkpointstage": "frozen", "frames": frames, "public_metadata": metadata,
            "logits": [0.0]*4, "attention": copy.deepcopy(frames[6]["attention"]), "pooled": [0.0]*256}


def synthetic_cached(fit):
    return [{"index": i, "audit": {"permutation_id": r["map_id"], "observed_support_action_ids": r["observed_actions"]},
             "features": copy.deepcopy(r["features"]), "correct_action": r["correct_action"], "demonstrated": r["demonstrated"],
             "input_sha256": r["input_sha256"], "label_sha256": r["label_sha256"]} for i, r in enumerate(fit[:1024])]


def fixture_outcomes(spec, rows, model_kind="legacy"):
    logits = []
    for row in rows:
        if spec["cleared"] or spec["query_cleared"]:
            actions = row["observed_actions"]
            winner = next(a for a in range(4) if a not in actions)
        elif spec["stage"] == "initial" and model_kind == "equivariant":
            logits.append([0.0] * 4)
            continue
        elif spec["stage"] == "initial":
            winner = 0
        else:
            winner = row["correct_action"]
        logits.append([4.0 if a == winner else 0.0 for a in range(4)])
    return logits, [I.sha(I.feature_bytes(r["features"], spec["cleared"], spec["query_cleared"])) for r in rows]


class Dataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fit, cls.held = I.abstract_rows(I.FIT), I.abstract_rows(I.HELD)

    def test_all_finite_rows_and_complete_schedule_match_separate_factory(self):
        self.assertEqual(self.fit, D.abstract_rows(D.FIT))
        self.assertEqual(self.held, D.abstract_rows(D.HELD))
        self.assertEqual(self.fit[0]["features"], [[0.,-1.,1.,0.,0.,0.,0.],[0.,1.,0.,1.,0.,0.,0.],[-1.,0.,0.,0.,1.,0.,0.],[0.,-1.,0.,0.,0.,0.,1.]])
        updates = I.schedule(512)
        self.assertEqual(updates, D.schedule(512))
        counts = Counter(itertools.chain.from_iterable(updates))
        self.assertEqual(Counter(counts.values()), {383:1024,384:512})
        self.assertEqual(len(updates), 1150)
        self.assertTrue(all(len(x)==512 for x in updates))
        self.assertEqual(I.sha(np.asarray(updates,dtype="<u4").tobytes()), I.SCHEDULE_SHA)
        self.assertEqual(I.abstract_audit(self.fit, counts)["unique_semantic_orbits"], 256)
        self.assertEqual(I.abstract_audit(self.held)["unique_semantic_orbits"], 128)

    def test_full_dataset_audit_and_mutated_hash_or_schedule(self):
        cached = synthetic_cached(self.fit)
        audit = {"synthetic": True}
        value = D.dataset(2048, visual=cached, visual_audit=audit)
        with mock.patch.object(I, "cached_rows", side_effect=AssertionError("no actual cache reads")):
            rebuilt = I.dataset(value, (cached,audit))
            self.assertEqual(rebuilt["schedule"]["presentations"], 588800)
            for mutate in (
                lambda d: d["fit"][0].__setitem__("input_sha256", "0"*64),
                lambda d: d["fit"][0].__setitem__("canonical", 1),
                lambda d: d["fit"][0].__setitem__("orbit_id", 1),
                lambda d: d["updates"][0].reverse(),
                lambda d: d["updates"][-1].append(0),
                lambda d: d["schedule"].__setitem__("presentations", 588800.0),
                lambda d: d["schedule"].__setitem__("indices_sha256", "f"*64),
                lambda d: d["audit"]["schedule"].__setitem__("tail_rows", 0),
                lambda d: d["audit"]["abstract"]["fit"]["orbits"][0]["member_indices"].pop(),
            ):
                changed = copy.deepcopy(value); mutate(changed)
                with self.assertRaises(I.Invalid):
                    I.dataset(changed, (cached,audit))

    def test_regrouped_flat_schedule_all_batches_and_explicit_tail(self):
        original = I.flat_schedule().tolist()
        for batch in (512,1024,2048,4096,8192,16384,32768):
            with self.subTest(batch=batch):
                updates = I.schedule(batch)
                self.assertEqual(list(itertools.chain.from_iterable(updates)), original)
                self.assertEqual(updates, D.schedule(batch))
                self.assertTrue(all(len(update)==batch for update in updates[:-1]))
                self.assertEqual(len(updates[-1]), 588800 % batch or batch)
                self.assertEqual(len(updates), (588800 + batch-1)//batch)
                self.assertEqual(I.completed_schedule(batch)["tail_rows"],588800 % batch)
        self.assertEqual(I.completed_schedule(2048)["updates"],288)
        self.assertEqual(I.completed_schedule(2048)["tail_rows"],1024)
        for batch in (True,512.,256,768,65536):
            with self.assertRaises(I.Invalid): I.schedule(batch)

    def test_smoke_prefix_only_and_unseen_row_exposures(self):
        cached=synthetic_cached(self.fit); visual_audit={"synthetic":True}
        original=I.flat_schedule().tolist()
        for batch, steps in ((512,2),(1024,5),(32768,2),(32768,5)):
            updates=I.schedule(batch,"smoke",steps)
            self.assertEqual(len(updates),steps)
            self.assertTrue(all(len(x)==batch for x in updates))
            self.assertEqual(list(itertools.chain.from_iterable(updates)),original[:batch*steps])
        value=D.dataset(512,"smoke",2,visual=cached,visual_audit=visual_audit)
        rebuilt=I.dataset(value,(cached,visual_audit))
        self.assertEqual(rebuilt["schedule"]["minimum_presentations_per_row"],0)
        self.assertEqual(rebuilt["schedule"]["tail_rows"],0)
        for kind,steps in (("training",2),("smoke",None),("smoke",3),("smoke",True),("unknown",2)):
            with self.assertRaises(I.Invalid): I.schedule(512,kind,steps)

    def test_postclamp_little_endian_bits_and_only_requested_values(self):
        row = self.fit[0]
        original = np.asarray(row["features"],dtype="<f4")
        for effects, query in ((False,False),(True,False),(False,True)):
            raw = I.feature_bytes(row["features"],effects,query)
            actual = np.frombuffer(raw,dtype="<f4").reshape(4,7)
            expected = original.copy()
            if effects: expected[:3,:2]=0
            if query: expected[3,:2]=0
            self.assertEqual(actual.tobytes(),expected.tobytes())
            self.assertTrue(np.array_equal(actual[:,2:],original[:,2:]))
        with self.assertRaises(I.Invalid): I.feature_bytes(row["features"],True,True)
        with self.assertRaises(I.Invalid): I.feature_bytes(row["features"],1,False)

    def test_cached_adapter_uses_attention_expectations_and_rounds_once(self):
        source = visual_source()
        actual, diagnostic = I.visual_row(source,0)
        self.assertEqual(actual["features"], self.fit[0]["features"])
        self.assertEqual(actual["input_sha256"],self.fit[0]["input_sha256"])
        self.assertEqual(diagnostic["maximum_coordinate_absolute_error"],0)
        # A controlled diffuse source shifts expectations; public role positions
        # stay unchanged and are never substituted into the feature calculation.
        source["frames"][0]["attention"][0][27] = .999
        source["frames"][0]["attention"][0][28] = .001
        actual, diagnostic = I.visual_row(source,0)
        self.assertNotEqual(actual["features"],self.fit[0]["features"])
        self.assertAlmostEqual(actual["features"][0][0],-.001,places=8)
        self.assertGreater(diagnostic["maximum_coordinate_absolute_error"],0)
        self.assertGreater(diagnostic["analytic_gap_lower_bound"],0)
        for field in ("input_sha256","query_sha256","metadata_sha256","label_sha256"):
            changed=copy.deepcopy(source);changed[field]="f"*64
            with self.assertRaises(I.Invalid): I.visual_row(changed,0)


class Metrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fit,cls.held=I.abstract_rows(I.FIT),I.abstract_rows(I.HELD)
        cls.populations={"fit":cls.fit,"heldout":cls.held,"cached_visual":synthetic_cached(cls.fit)}
        cls.specs=I.selectors()

    def test_stable_ce_signed_margin_winner_margin_and_first_tie(self):
        metric=I.row_metrics([10000.,10000.,9999.,9998.],1)
        self.assertEqual(metric["winner"],0)
        self.assertEqual(metric["correct"],0)
        self.assertEqual(metric["margin"],0)
        self.assertEqual(metric["winner_margin"],0)
        self.assertAlmostEqual(metric["ce"],math.log(2+math.exp(-1)+math.exp(-2)),places=14)
        self.assertEqual(I.row_metrics([4.,3.,2.,1.],3)["margin"],-3)
        self.assertTrue(math.isfinite(I.row_metrics([3e38,-3e38,0.,1.],1)["ce"]))
        for logits in ([True,0,0,0],[float("nan"),0,0,0],[1,2,3],[1e100,0,0,0]):
            with self.assertRaises(I.Invalid): I.row_metrics(logits,0)

    def test_raw_canonical_subsets_permap_and_full_six_ordering(self):
        spec=self.specs["final-fit-l4"]
        logits,_=fixture_outcomes(spec,self.fit)
        result,_,canonical=I.summary(self.fit,logits)
        self.assertEqual(result["raw"]["correct"],1536)
        self.assertEqual(result["canonical"]["correct"],256)
        self.assertEqual(result["canonical"]["subsets"]["omitted"]["rows"],64)
        self.assertEqual(result["canonical"]["subsets"]["demonstrated"]["rows"],192)
        self.assertTrue(all(r["correct"]==16 for r in result["canonical"]["per_map"].values()))
        members=next(indices for _,indices in I.ordered_orbits(self.fit) if canonical[0] in indices)
        # Maximum pair range differs from canonical-reference maximum here.
        logits[members[1]][0] += .2
        logits[members[2]][0] -= .3
        result,_,_=I.summary(self.fit,logits)
        self.assertAlmostEqual(result["ordering"]["max_logit_deviation"],.5)
        logits[members[1]]=[10.,0.,0.,0.] if self.fit[members[1]]["correct_action"]!=0 else [0.,10.,0.,0.]
        result,_,_=I.summary(self.fit,logits)
        self.assertEqual(result["ordering"]["inconsistent_orbits"],1)

    def test_missing_id_shortcut_and_both_ablations_exact_quarter(self):
        for cohort,rows in (("fit",self.fit),("heldout",self.held)):
            expected_groups={"effects-zero":96,"query-zero":24*(16 if cohort=="fit" else 8)}
            for suffix in ("effects-zero","query-zero"):
                spec=self.specs[f"final-{cohort}-{suffix}"]
                logits,hashes=fixture_outcomes(spec,rows)
                summary,_,canonical=I.summary(rows,logits)
                control=I.ablation_control(rows,logits,hashes,canonical)
                self.assertEqual(control["identical_input_groups"],expected_groups[suffix])
                self.assertTrue(control["exact_quarter"])
                self.assertEqual(summary["canonical"]["subsets"]["omitted"]["accuracy"],1)
                self.assertEqual(summary["canonical"]["subsets"]["demonstrated"]["accuracy"],0)
                logits[0]=[10. if a==rows[0]["correct_action"] else 0. for a in range(4)]
                with self.assertRaisesRegex(I.Invalid,"different winners"):
                    I.ablation_control(rows,logits,hashes,canonical)

    def test_external_controls_no_fake_probability_or_cached_orbits(self):
        for cohort,rows in self.populations.items():
            controls=I.external_controls(rows,cohort=="cached_visual")
            self.assertEqual(controls["analytic"]["minimum_true_score_margin"],1)
            self.assertEqual(controls["constant_action0"]["canonical"]["accuracy"],.25)
            self.assertEqual(controls["always_missing_id"]["canonical"]["subsets"]["omitted"]["accuracy"],1)
        rows=self.populations["cached_visual"]
        logits,_=fixture_outcomes(self.specs["final-cached-visual-l4"],rows)
        result,_,_=I.summary(rows,logits,True)
        self.assertEqual(result["canonical"],result["raw"])
        self.assertFalse(result["ordering"]["applicable"])
        self.assertIsNone(result["ordering"]["orbits"])

    def test_complete_fifteen_stream_reconstruction_and_gate_scope(self):
        with mock.patch.object(I,"stream",side_effect=fixture_outcomes):
            report=I.reconstruction(self.populations,self.specs,"legacy")
        self.assertEqual(len(report["summaries"]),15)
        self.assertEqual(report["decision"],"supported_single_seed_binding_prerequisite")
        self.assertTrue(all(report["gates"]["selection"].values()))
        self.assertEqual(report["contrasts"]["final_minus_initial/fit"]["canonical"]["accuracy_delta"],.75)
        self.assertEqual(report["contrasts"]["factual_minus_query_zero/heldout"]["canonical"]["subsets"]["omitted"]["accuracy_delta"],0)
        summaries=report["summaries"]
        summaries["initial-fit-l4"]["ordering"]["winner_invariant"]=False
        summaries["final-fit-l1"]["ordering"]["winner_invariant"]=False
        self.assertTrue(all(I.legacy_gates(summaries)[1].values()))
        summaries["final-heldout-l4"]["canonical"]["subsets"]["omitted"]["correct"]=28
        self.assertFalse(I.legacy_gates(summaries)[1]["heldout_omitted_29_of_32"])
        self.assertEqual(I.legacy_gates(summaries)[0],"registered_binding_screen_not_supported")
        summaries["final-fit-l4"]["ordering"]["winner_invariant"]=False
        self.assertFalse(I.legacy_gates(summaries)[1]["terminal_l4_ordering"])


class Equivariance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows={"fit":I.abstract_rows(I.FIT),"heldout":I.abstract_rows(I.HELD)}
        cls.populations={**cls.rows,"cached_visual":synthetic_cached(cls.rows["fit"])}
        cls.specs=I.selectors()

    def values(self, margin=4.0):
        # Construct effect-ordered scores first; route to action IDs explicitly.
        result={}
        for cohort,rows in self.rows.items():
            result[cohort]=[]
            for row in rows:
                mapping=I.MAPS[row["map_id"]]
                physical=[margin if d==row["desired_direction"] else 0.0 for d in range(4)]
                result[cohort].append([physical[mapping[action]] for action in range(4)])
        return result

    def member(self, map_index, missing=0, desired=0):
        for cohort,rows in self.rows.items():
            for i,row in enumerate(rows):
                absent=next(a for a in range(4) if a not in row["observed_actions"])
                if row["canonical"] and row["map_id"]==map_index and I.MAPS[map_index][absent]==missing and row["desired_direction"]==desired:
                    return cohort,i
        self.fail("synthetic map member missing")

    def test_all16_orbits_align_all24_maps_and_inverse_mapping(self):
        values=self.values()
        result=I.action_equivariance(self.rows,values)
        self.assertEqual(set(result["by_orbit"]),{f"{a}/{b}" for a in range(4) for b in range(4)})
        self.assertEqual(result["strict_true_margin_orbits"],16)
        self.assertEqual(result["maximum_absolute_logit_error"],0)
        self.assertEqual(result["canonical_rows"],384)
        for name,orbit in result["by_orbit"].items():
            desired=int(name[-1]);self.assertEqual(orbit["correct"],24)
            self.assertEqual(orbit["reference_logits_effect_order"],[4. if d==desired else 0. for d in range(4)])
            self.assertTrue(orbit["winner_invariant"])
        # A three-cycle makes direct indexing differ from inverse reindexing.
        cycle=I.MAPS.index((1,2,0,3));cohort,index=self.member(cycle,0,0)
        self.assertEqual(values[cohort][index],[0.,0.,4.,0.])

    def test_missing_duplicate_map_or_nonfinite_fails_closed(self):
        values=self.values()
        changed=copy.deepcopy(self.rows);cohort,index=self.member(0)
        changed[cohort][index]["canonical"]=False
        with self.assertRaisesRegex(I.Invalid,"all24 maps"):I.action_equivariance(changed,values)
        changed=copy.deepcopy(self.rows);cohort,index=self.member(1)
        changed[cohort][index]["map_id"]=0
        with self.assertRaises(I.Invalid):I.action_equivariance(changed,values)
        values[cohort][index][0]=float("nan")
        with self.assertRaises(I.Invalid):I.action_equivariance(self.rows,values)

    def test_fixed_map0_reference_and_elementwise_tolerance(self):
        values=self.values(1.0);cohort,index=self.member(1)
        mapping=I.MAPS[1];action=mapping.index(0)
        tolerance=1e-4+1e-5
        values[cohort][index][action]+=tolerance*(1-1e-6)
        good=I.action_equivariance(self.rows,values)
        self.assertTrue(good["numerically_equivariant"])
        values[cohort][index][action]=1+tolerance*(1+1e-6)
        bad=I.action_equivariance(self.rows,values)
        self.assertFalse(bad["numerically_equivariant"])
        self.assertGreater(bad["maximum_tolerance_ratio"],1.)
        self.assertEqual(bad["strict_true_margin_orbits"],16)
        self.assertTrue(bad["all_eligible_winners_invariant"])

    def test_ties_and_near_tie_flip_are_ineligible_without_relaxing_margin(self):
        ties=I.action_equivariance(self.rows,self.values(0.0))
        self.assertTrue(ties["numerically_equivariant"])
        self.assertEqual(ties["positive_margin_orbits"],0)
        self.assertEqual(ties["strict_true_margin_orbits"],0)
        self.assertTrue(all(v["winner_invariant"] is None for v in ties["by_orbit"].values()))
        values={cohort:[[1e-6 if I.MAPS[row["map_id"]][a]==0 else 0. for a in range(4)] for row in rows]
                for cohort,rows in self.rows.items()}
        cohort,index=self.member(1)
        values[cohort][index]=[1e-6 if I.MAPS[1][a]==1 else 0. for a in range(4)]
        flipped=I.action_equivariance(self.rows,values)
        self.assertTrue(flipped["numerically_equivariant"])
        self.assertEqual(flipped["winner_inconsistent_orbits"],0)
        self.assertEqual(flipped["positive_margin_orbits"],15)
        self.assertIsNone(flipped["by_orbit"]["0/0"]["winner_invariant"])
        self.assertTrue(flipped["all_eligible_winners_invariant"])
        self.assertEqual(flipped["strict_true_margin_orbits"],0)
        self.assertEqual(flipped["winner_comparison_criterion"],
                         "minimum_winner_margin > 2*maximum_absolute_logit_error and >0")

    def test_winner_comparison_requires_strict_twice_error_margin(self):
        error=2**-17
        values=self.values(3*error);cohort,index=self.member(1)
        loser=I.MAPS[1].index(1)
        values[cohort][index][loser]=error
        boundary=I.action_equivariance(self.rows,values)["by_orbit"]["0/0"]
        self.assertTrue(boundary["numerically_equivariant"])
        self.assertEqual(boundary["minimum_winner_margin"],2*boundary["maximum_absolute_logit_error"])
        self.assertFalse(boundary["positive_margin_winner_comparison"])
        values[cohort][index][loser]=error/2
        resolved=I.action_equivariance(self.rows,values)["by_orbit"]["0/0"]
        self.assertTrue(resolved["positive_margin_winner_comparison"])
        self.assertTrue(resolved["winner_invariant"])

    def test_strict_margin_and_kind_specific_conjunction(self):
        with mock.patch.object(I,"stream",side_effect=fixture_outcomes):
            result=I.reconstruction(self.populations,self.specs,"equivariant")
        self.assertEqual(result["decision"],"supported_single_seed_action_equivariant_selection")
        self.assertTrue(all(result["gates"]["selection"].values()))
        self.assertEqual(result["action_equivariance"]["initial"]["positive_margin_orbits"],0)
        weak=I.action_equivariance(self.rows,self.values(.0009))
        self.assertEqual(sum(o["correct"] for o in weak["by_orbit"].values()),384)
        self.assertEqual(weak["strict_true_margin_orbits"],0)
        eq={**result["action_equivariance"],"final":weak}
        decision,components=I.gates(result["summaries"],"equivariant",eq)
        self.assertEqual(decision,"action_equivariant_selection_not_supported")
        self.assertFalse(components["selection"]["all16_strict_true_margin"])
        self.assertEqual(I.gates(result["summaries"],"legacy",eq)[0],"supported_single_seed_binding_prerequisite")
        self.assertEqual(I.action_equivariance(self.rows,self.values(.001))["strict_true_margin_orbits"],16)
        for stage,key in (("initial","numerically_equivariant"),("final","numerically_equivariant"),("initial","all_eligible_winners_invariant")):
            changed=copy.deepcopy(result["action_equivariance"]);changed[stage][key]=False
            self.assertEqual(I.gates(result["summaries"],"equivariant",changed)[0],"action_equivariant_selection_not_supported")
        with self.assertRaises(I.Invalid):I.gates(result["summaries"],"unknown",eq)


class Integrity(unittest.TestCase):
    def test_flattened_output_identity_and_postclamp_hash(self):
        row=I.abstract_rows(I.FIT)[0]
        spec=I.selectors()["final-fit-effects-zero"]
        output={**row,**{k:spec[k] for k in ("stage","loops","cleared","query_cleared")},"logits":[0.,0.,0.,1.],
                "model_kind":"equivariant",
                "model_input_sha256":I.sha(I.feature_bytes(row["features"],True,False))}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'rows.jsonl'
            record={**spec,"rows":str(path)}
            path.write_text(json.dumps(output)+'\n')
            I.stream(record,[row],"equivariant")
            mutations=(lambda r:r.__setitem__("model_input_sha256",row["input_sha256"]),
                       lambda r:r.__setitem__("loops",True),lambda r:r.__setitem__("stage","initial"),
                       lambda r:r.__setitem__("correct_action",1),lambda r:r.__setitem__("extra",1),
                       lambda r:r.__setitem__("model_kind","legacy"),lambda r:r.pop("model_kind"),
                       lambda r:r.__setitem__("logits",[float("inf"),0,0,0]))
            for mutate in mutations:
                changed=copy.deepcopy(output);mutate(changed);path.write_text(json.dumps(changed)+'\n')
                with self.assertRaises((I.Invalid,ValueError)): I.stream(record,[row],"equivariant")
            path.write_text('')
            with self.assertRaises(I.Invalid): I.stream(record,[row],"equivariant")

    def test_hash_symlink_strict_json(self):
        for raw in ('{"x":1,"x":2}','{"x":NaN}'):
            with self.assertRaises(I.Invalid): I.decode(raw)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'file';path.write_bytes(b'original');digest=I.file_sha(path)
            I.bound(path,digest);path.write_bytes(b'changed')
            with self.assertRaises(I.Invalid): I.bound(path,digest)
            link=Path(folder)/'alias';link.symlink_to(path)
            with self.assertRaises(I.Invalid): I.file_sha(link)

    def test_receipt_seven_field_streams_and_exact_selfhash_closure(self):
        h='a'*64
        config={"schema":I.SCHEMA,"dataset":"/fixture/dataset","dataset_sha256":h,
                "model_kind":"equivariant","effective_batch":2048,
                "registration":{"path":"/fixture/registration","sha256":h},"integrity_receipt":"/fixture/receipt","integrity_receipt_sha256":h,
                "streams":{name:{**s,"rows":"/fixture/"+name,"sha256":h} for name,s in I.selectors().items()}}
        frozen={str(I.R/name):h for name in ("data.py","analysis.py","independent_review.py","independent_review_tests.py")}
        frozen.update({"/fixture/dataset":h,"/fixture/registration":h})
        frozen.update({r["rows"]:h for r in config["streams"].values()})
        config["frozen_files"]={**frozen,"/fixture/receipt":h}
        checkpoints={"initial":{"sha256":h,"parameter_sha256":'b'*64},"final":{"sha256":h,"parameter_sha256":'c'*64}}
        receipt={"schema":"looped-action-binding-integrity-v2","accepted":True,"checks":dict.fromkeys(I.CHECKS,True),
                 "model_kind":"equivariant","effective_batch":2048,"completed_schedule":I.completed_schedule(2048),
                 "source_revision":'d'*40,"binary_sha256":h,"dependency_revision":I.DEPENDENCY,"dataset_sha256":h,
                 "registration_sha256":h,"checkpoints":checkpoints,"frozen_files":frozen,
                 "streams":{name:{**r,"checkpoint_sha256":h,"parameter_sha256":checkpoints[r["stage"]]["parameter_sha256"]} for name,r in config["streams"].items()}}
        with mock.patch.object(I,"frozen"),mock.patch.object(I,"bound",side_effect=lambda p,h:Path(p)),mock.patch.object(I,"file_sha",return_value=h),mock.patch.object(I,"load",return_value=receipt):
            self.assertIs(I.authority(config),receipt)
            for mutate in (lambda c:c["streams"]["final-fit-l4"].__setitem__("checkpoint_sha256",h),
                           lambda c:c["streams"]["final-fit-l4"].__setitem__("loops",1),
                           lambda c:c.__setitem__("model_kind","legacy"),lambda c:c.__setitem__("model_kind","unknown"),
                           lambda c:c.__setitem__("effective_batch",2048.0),lambda c:c.__setitem__("effective_batch",4096),
                           lambda c:c["frozen_files"].__setitem__("/fixture/unclosed",h)):
                changed=copy.deepcopy(config);mutate(changed)
                with self.assertRaises(I.Invalid): I.authority(changed)
            for field,value in (("updates",287),("tail_rows",0),("presentations",588799),("effective_batch",512),("indices_sha256",'f'*64)):
                saved=receipt["completed_schedule"][field];receipt["completed_schedule"][field]=value
                with self.assertRaises(I.Invalid): I.authority(config)
                receipt["completed_schedule"][field]=saved
            receipt["checks"]["profiles"]=1
            with self.assertRaises(I.Invalid): I.authority(config)

    def test_numeric_report_mutations_and_real_review_seam(self):
        fit,held=I.abstract_rows(I.FIT),I.abstract_rows(I.HELD)
        data={"fit":fit,"heldout":held,"cached_visual":synthetic_cached(fit)}
        specs=I.selectors()
        with mock.patch.object(I,"stream",side_effect=fixture_outcomes):
            expected=I.reconstruction(data,specs,"legacy")
        stats=I.compare(copy.deepcopy(expected),expected)
        self.assertGreater(stats["compared_float_fields"],1000)
        for mutate in (lambda r:r["gates"]["selection"].__setitem__("terminal_l4_ordering",False),
                       lambda r:r["summaries"]["final-fit-l4"]["canonical"].__setitem__("correct",256.0),
                       lambda r:r["summaries"]["final-fit-l4"]["raw"].__setitem__("ce",float("nan")),
                       lambda r:r["summaries"]["final-fit-l4"]["raw"]["margin"].__setitem__("mean",3.99),
                       lambda r:r["external_controls"].pop("heldout")):
            changed=copy.deepcopy(expected);mutate(changed)
            with self.assertRaises(I.Invalid): I.compare(changed,expected)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);config_path=root/'config.json';report_path=root/'report.json';data_path=root/'data.json'
            data["schedule"]={"kind":"training","effective_batch":2048,"presentations":588800,"indices_sha256":I.SCHEDULE_SHA}
            data_path.write_text(json.dumps(data))
            config={"dataset":str(data_path),"dataset_sha256":I.file_sha(data_path),"registration":{"sha256":'a'*64},
                    "model_kind":"legacy","effective_batch":2048,
                    "integrity_receipt_sha256":'b'*64,"streams":specs}
            config_path.write_text(json.dumps(config))
            receipt={"source_revision":'c'*40,"binary_sha256":'d'*64}
            report={**expected,"schema":"looped-action-binding-analysis-v2","accepted":True,"classification":"single_seed_action_equivariance_screen",
                    "model_kind":"legacy","effective_batch":2048,"completed_schedule":I.completed_schedule(2048),
                    "config_sha256":I.file_sha(config_path),"dataset_sha256":config["dataset_sha256"],"registration_sha256":'a'*64,
                    "analysis_sha256":I.file_sha(I.R/'analysis.py'),"data_helper_sha256":I.file_sha(I.R/'data.py'),
                    "integrity_receipt_sha256":'b'*64,**receipt,"limits":["synthetic"],"elapsed_seconds":.1}
            report_path.write_text(json.dumps(report))
            with mock.patch.object(I,"authority",return_value=receipt),mock.patch.object(I,"dataset",return_value={"schedule":{"indices_sha256":I.SCHEDULE_SHA}}),mock.patch.object(I,"stream",side_effect=fixture_outcomes):
                result=I.review(config_path,report_path)
                self.assertTrue(result["accepted"])
                self.assertEqual(result["report_sha256"],I.file_sha(report_path))
                self.assertEqual(result["decision"],"supported_single_seed_binding_prerequisite")
                report["gates"]["selection"]["heldout_116_of_128"]=False;report_path.write_text(json.dumps(report))
                with self.assertRaises(I.Invalid): I.review(config_path,report_path)


if __name__ == "__main__":
    unittest.main()
