#!/usr/bin/env python3
"""Synthetic independent C13 controls; no actual heads, attention or episodes."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("independent", Path(__file__).with_name("independent_review.py"))
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
np = R.np

def head_bytes(scale=1):
    # Foreign tensors precede AND follow queries to exercise both preservation ranges.
    weights = [float(i % 256 == i // 256) for i in range(1024)]
    queries = [scale * (i+1) / 256 for i in range(256)]
    header = {"output.weight": dict(dtype="F32", shape=[4,256], data_offsets=[0,4096]),
              "queries": dict(dtype="F32", shape=[2,128], data_offsets=[4096,5120]),
              "output.bias": dict(dtype="F32", shape=[4], data_offsets=[5120,5136])}
    encoded = json.dumps(header, separators=(",", ":")).encode(); encoded += b" "*((-len(encoded))%8)
    return struct.pack("<Q", len(encoded)) + encoded + struct.pack("<1284f", *(weights+queries+[0.]*4))

def fixture():
    audit, baseline, treatment = [], [], []
    p = .2; other = .8/63
    sharp = p**16/(p**16 + 63*other**16)
    for i in range(1024):
        map_id = R.MAPS[i%16]; label = R.PERMS[map_id].index(0)
        cells = [0]*64; cells[27] = 2; cells[19] = 3
        row = dict(group_index=i//16, row_index=i, permutation_id=map_id, cohort="seen", condition="factual",
                   support_cleared=False, query_cells=cells, agent_patch=27, goal_patch=19,
                   correct_action=label, inferred_controls=list(R.PERMS[map_id]),
                   observed_support_action_ids=[0,1,2], omitted_action=3, synthetic=True)
        audit.append(row)
        for mass, destination in ((p, baseline), (sharp, treatment)):
            attention = [[mass if j == role else (1-mass)/63 for j in range(64)] for role in (27,19)]
            pooled = [0.]*256
            pooled[0] += 1-mass; pooled[label] += 2*mass
            destination.append(dict(row, checkpointstage="final", attention=attention, pooled=pooled, logits=pooled[:4]))
    return baseline, treatment, audit, head_bytes(), head_bytes(16)

class Controls(unittest.TestCase):
    def test_oracle_recovery_and_exact_foreign_weight_preservation(self):
        inputs = fixture(); result, extra = R.compute(*inputs)
        self.assertEqual(result["baseline"]["correct"], 256)
        self.assertEqual(result["treatment"]["correct"], 1024)
        self.assertEqual(result["baseline"]["all_maps_correct_groups"], 0)
        self.assertEqual(result["treatment"]["all_maps_correct_groups"], 64)
        self.assertEqual(result["contrasts"]["accuracy_delta"]["ci95"], [.75,.75])
        self.assertEqual(result["contrasts"]["accuracy_minus_constant"]["ci95"], [.75,.75])
        base_ce = (math.log(math.exp(1.2)+3)-1.2 + 3*(math.log(math.exp(.8)+math.exp(.4)+2)-.4))/4
        self.assertAlmostEqual(result["baseline"]["ce"], base_ce, places=14)
        self.assertAlmostEqual(result["treatment"]["ce"], math.log(1+3*math.exp(-2)), places=14)
        self.assertEqual(result["decision"], "frozen_sharpening_recovers_registered_fit")
        self.assertEqual(extra["baseline_affine_max_absolute_error"], 0)
        altered = copy.deepcopy(result); altered["contrasts"]["accuracy_delta"]["ci95"][0] = .749
        with self.assertRaisesRegex(ValueError, "numerical report mismatch"): R.compare(result, altered)
        # Separate transport check: synthetic receipt is not source-provenance evidence.
        with tempfile.TemporaryDirectory(prefix="c13-independent-synthetic-") as tmp:
            root = Path(tmp)
            def put(name, raw):
                path = root/name; path.write_bytes(raw)
                return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())
            bindings = {k: put(k, ("\n".join(json.dumps(r) for r in v)+"\n").encode() if isinstance(v,list) else v)
                        for k,v in zip(("baseline","treatment","audit","original_head","scaled_head"), inputs)}
            reg = put("registration", b"synthetic fixture only")
            receipt = dict(accepted=True,zero_updates=True,source_verified=True,profile_verified=True,
                           files={v["path"]:v["sha256"] for v in [*bindings.values(),reg]})
            bindings["integrity_receipt"] = put("receipt", json.dumps(receipt).encode())
            config = put("config", json.dumps(dict(bindings,registration=reg["path"],registration_sha256=reg["sha256"])).encode())
            report = put("report", json.dumps(dict(result,accepted=True,config_sha256=config["sha256"],registration_sha256=reg["sha256"])).encode())
            reviewed = R.review(config["path"],config["sha256"],report["path"],report["sha256"])
            self.assertTrue(reviewed["accepted"])
            with self.assertRaisesRegex(ValueError,"file hash"):
                R.review(config["path"],"0"*64,report["path"],report["sha256"])

    def test_group_bootstrap_and_strict_decision_boundary(self):
        values = np.asarray([-.25]*16+[0.]*16+[.25]*16+[.75]*16)
        draws = np.random.Generator(np.random.PCG64(1940)).integers(0,64,(10000,64))
        weights = np.stack([np.bincount(x, minlength=64) for x in draws])
        got = R.interval(values, weights)
        scalar = [math.fsum(float(values[i]) for i in draw)/64 for draw in draws]
        self.assertEqual(got["ci95"], np.quantile(scalar,[.025,.975],method="linear").tolist())
        self.assertEqual(R.decide(.9,.001), "frozen_sharpening_recovers_registered_fit")
        self.assertEqual(R.decide(.899,.001), "partial_accuracy_gain_only")
        self.assertEqual(R.decide(1.,0), "concentration_insufficient_for_accuracy_recovery")

    def test_corruptions_and_unmet_concentration_fail_closed(self):
        b,t,a,original,scaled = fixture()
        wrong = bytearray(scaled); wrong[-1] ^= 1
        with self.assertRaisesRegex(ValueError, "foreign bytes"): R.verify_heads(original, bytes(wrong))
        with self.assertRaisesRegex(ValueError, "x16"): R.verify_heads(original, head_bytes(8))
        start = 8+struct.unpack_from("<Q", original)[0]
        wrong = bytearray(original); wrong[start:start+4] = struct.pack("<f",float("nan"))
        with self.assertRaisesRegex(ValueError, "nonfinite"): R.head(bytes(wrong))
        wrong = bytearray(original); wrong[8:start] = wrong[8:start].replace(b'4096,5120',b'4095,5120')
        with self.assertRaisesRegex(ValueError, "offsets"): R.head(bytes(wrong))
        changed = copy.deepcopy(t); changed[0]["attention"][0][27] = .999
        changed[0]["attention"][0][0] += .001
        with self.assertRaisesRegex(ValueError, "power mismatch"): R.compute(b,changed,a,original,scaled)
        changed = copy.deepcopy(t); changed[0]["logits"][0] += .01
        with self.assertRaisesRegex(ValueError, "affine reconstruction"): R.compute(b,changed,a,original,scaled)
        bad_a = copy.deepcopy(a); bad_a[0]["correct_action"] = (bad_a[0]["correct_action"]+1)%4
        bad_b, bad_t = copy.deepcopy(b), copy.deepcopy(t)
        bad_b[0]["correct_action"] = bad_t[0]["correct_action"] = bad_a[0]["correct_action"]
        with self.assertRaisesRegex(ValueError, "role/label"): R.compute(bad_b,bad_t,bad_a,original,scaled)
        diffuse = copy.deepcopy(b)
        for row in diffuse:
            row["attention"] = [[.02 if i == role else .98/63 for i in range(64)] for role in (27,19)]
        with self.assertRaisesRegex(ValueError, "premise_not_met"): R.compute(diffuse,t,a,original,scaled)

if __name__ == "__main__": unittest.main()
