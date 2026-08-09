#!/usr/bin/env python3
"""Regression tests for the public P2 A/B gate CLI."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GATE = REPO_ROOT / "scripts" / "p2_ab_gate.py"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def config_for(arm: str, output_dir: str) -> dict[str, object]:
    config: dict[str, object] = {
        "physical_batch": 1024,
        "grad_accum": 1,
        "sigreg_max_rows": 32768,
        "lessons": ["dynamics"],
        "output_dir": output_dir,
    }
    if arm == "control":
        config.update(sigreg_spatial=True, sigreg_spatial_pool=True)
    else:
        config.update(sigreg_spatial=True, sigreg_pre_rms_spatial=True)
    return config


def report() -> dict[str, object]:
    shuffle = {"ratio": 1.2, "ratio_ci95_low": 1.1, "ratio_ci95_high": 1.3}
    return {
        "schema": "p2.eval_report.v9",
        "synthetic_dynamics": {
            "representation": {
                "sigreg_raw": 1.0,
                "sigreg_bounded": 0.5,
                "sigreg_near_bound": False,
                "noncollapse_pass": True,
                "mean_encoder_variance": 0.1,
                "effective_rank": 4.0,
                "effective_rank_fraction": 0.5,
            },
            "action_diagnostics": {
                "aggregate": {"shuffle": shuffle},
                "by_source": {"random_one_step": {"shuffle": shuffle}},
            },
            "changed_transitions": {
                "n": 2,
                "improvement_fraction": 0.2,
                "improvement_ci95_low": 0.1,
                "improvement_ci95_high": 0.3,
            },
            "rollout": {"mse_8": 0.1},
        },
    }


class AbGateCliTests(unittest.TestCase):
    def run_gate(self, root: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "python3",
                str(GATE),
                "--root",
                str(root),
                "--treatment-arm",
                "pre-rms-spatial",
                "--seeds",
                "1",
                "--final-update",
                "2000",
                "--output-json",
                str(output_dir / "gate.json"),
                "--output-md",
                str(output_dir / "gate.md"),
            ],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_partial_run_emits_structured_validation_failure_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            root = temp_path / "ab-sigreg-geometry-v2"
            for arm in ("control", "pre-rms-spatial"):
                arm_dir = root / "seed-1" / arm
                config = config_for(arm, f"/old/pod/{root.name}/seed-1/{arm}")
                write_json(arm_dir / "config.json", config)
                if arm == "control":
                    write_json(
                        arm_dir / "manifest.json",
                        {
                            "git_sha": "a" * 40,
                            "candle_git_sha": "b" * 40,
                            "binary_sha256": "c" * 64,
                        },
                    )

            completed = self.run_gate(root, temp_path / "out")

            self.assertNotEqual(completed.returncode, 0)
            self.assertNotIn("Traceback", completed.stderr)
            failure = json.loads((temp_path / "out" / "gate.json").read_text())
            self.assertEqual(failure["schema"], "p2.ab_gate_validation_failure.v1")
            self.assertTrue(failure["errors"])

    def test_relocated_legacy_absolute_artifact_paths_are_resolved_under_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            root = temp_path / "ab-sigreg-geometry-v2"
            legacy_root = temp_path / "old-pod" / root.name
            provenance = {
                "git_sha": "a" * 40,
                "candle_git_sha": "b" * 40,
                "binary_sha256": "c" * 64,
            }
            for arm in ("control", "pre-rms-spatial"):
                arm_dir = root / "seed-1" / arm
                legacy_arm = legacy_root / "seed-1" / arm
                config = config_for(arm, str(legacy_arm))
                write_json(arm_dir / "config.json", config)
                config_sha = hashlib.sha256((arm_dir / "config.json").read_bytes()).hexdigest()
                geometry = (
                    "post_rms_pooled_spatial_cells"
                    if arm == "control"
                    else "pre_rms_unpooled_spatial_cells"
                )
                write_json(
                    arm_dir / "manifest.json",
                    {
                        "schema": "p2.sigreg_geometry_arm.v1",
                        "arm": arm,
                        "seed": 1,
                        "physical_batch": 1024,
                        "grad_accum": 1,
                        "sigreg_geometry": geometry,
                        "complete_through_update": 2000,
                        "config_sha256": config_sha,
                        **provenance,
                    },
                )
                phases = []
                for update in (1000, 2000):
                    checkpoint = arm_dir / "checkpoints" / f"step-{update:012d}"
                    for name in (
                        "model.safetensors",
                        "optimizer.safetensors",
                        "trainer_state.json",
                    ):
                        path = checkpoint / name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(f"{arm}-{update}-{name}".encode())
                    eval_dir = arm_dir / f"eval-update-{update:04d}"
                    write_json(eval_dir / "eval_report.json", report())
                    hashed = [
                        checkpoint / "model.safetensors",
                        checkpoint / "optimizer.safetensors",
                        checkpoint / "trainer_state.json",
                        arm_dir / "config.json",
                        eval_dir / "eval_report.json",
                    ]
                    lines = []
                    for path in hashed:
                        relative = path.relative_to(root)
                        legacy_path = legacy_root / relative
                        legacy_path.parent.mkdir(parents=True, exist_ok=True)
                        legacy_path.write_bytes(path.read_bytes())
                        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {legacy_path}")
                    (eval_dir / "sha256.txt").write_text("\n".join(lines) + "\n")
                    for kind in ("train", "eval"):
                        log_path = arm_dir / f"{kind}-{update}.log"
                        log_path.write_text("completed successfully\n")
                        phases.append(
                            {
                                "stage": f"{kind}-{update}",
                                "status": "0",
                                "log_path": str(legacy_root / log_path.relative_to(root)),
                                **provenance,
                            }
                        )
                (arm_dir / "phases.jsonl").write_text(
                    "".join(json.dumps(phase) + "\n" for phase in phases)
                )

            completed = self.run_gate(root, temp_path / "out")

            self.assertEqual(
                completed.returncode,
                0,
                completed.stderr + (temp_path / "out" / "gate.json").read_text(),
            )
            result = json.loads((temp_path / "out" / "gate.json").read_text())
            self.assertEqual(result["decision"]["action"], "replicate_both_arms_seeds_2_and_3")

            local_model = (
                root
                / "seed-1"
                / "control"
                / "checkpoints"
                / "step-000000001000"
                / "model.safetensors"
            )
            local_model.write_bytes(b"corrupt-local-copy")
            rejected = self.run_gate(root, temp_path / "out-after-corruption")
            self.assertNotEqual(rejected.returncode, 0)
            failure = json.loads(
                (temp_path / "out-after-corruption" / "gate.json").read_text()
            )
            self.assertEqual(failure["schema"], "p2.ab_gate_validation_failure.v1")
            self.assertIn("sha256 mismatch", json.dumps(failure["errors"]))


if __name__ == "__main__":
    unittest.main()
