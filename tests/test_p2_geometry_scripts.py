#!/usr/bin/env python3
"""Black-box regression tests for the geometry experiment shell entrypoints."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OVERNIGHT = REPO_ROOT / "scripts" / "p2_sigreg_geometry_overnight.sh"
SMOKE = REPO_ROOT / "scripts" / "p2_sigreg_geometry_smoke.sh"
ACTION_RUNNER = REPO_ROOT / "scripts" / "p2_sigreg_action_ab.sh"


def executable(path: Path, contents: str) -> None:
    path.write_text(textwrap.dedent(contents).lstrip())
    path.chmod(0o755)


class GeometryOvernightTests(unittest.TestCase):
    def fixture(self, temp_path: Path) -> tuple[Path, dict[str, str], Path]:
        scripts = temp_path / "scripts"
        scripts.mkdir()
        shutil.copy2(OVERNIGHT, scripts / OVERNIGHT.name)
        log = temp_path / "events.log"
        executable(
            scripts / "p2_sigreg_geometry_ab.sh",
            r"""
            #!/usr/bin/env bash
            set -euo pipefail
            arm="$1"
            seed="$2"
            printf 'start %s blocking=%s\n' "$arm" "${CUDA_LAUNCH_BLOCKING:-0}" >>"$P2_TEST_LOG"
            if [[ "$arm" == control && "${P2_TEST_FAIL_WITHOUT_PHASE:-0}" == 1 ]]; then
              printf 'end %s failed-without-phase\n' "$arm" >>"$P2_TEST_LOG"
              exit 3
            fi
            if [[ "$arm" == pre-rms-spatial && "${P2_TEST_FAIL_ONCE:-0}" == 1 && ! -e "$P2_TEST_MARKER" ]]; then
              touch "$P2_TEST_MARKER"
              mkdir -p "$P2_AB_ROOT/seed-$seed/$arm"
              printf '{"stage":"eval-1000","status":"134"}\n' >>"$P2_AB_ROOT/seed-$seed/$arm/phases.jsonl"
              printf 'end %s failed\n' "$arm" >>"$P2_TEST_LOG"
              exit 134
            fi
            sleep 0.1
            printf 'end %s ok\n' "$arm" >>"$P2_TEST_LOG"
            """,
        )
        executable(
            scripts / "p2_ab_gate.py",
            r"""
            #!/usr/bin/env python3
            import argparse, json
            from pathlib import Path
            parser = argparse.ArgumentParser()
            parser.add_argument('--root')
            parser.add_argument('--treatment-arm')
            parser.add_argument('--seeds', nargs='+')
            parser.add_argument('--final-update')
            parser.add_argument('--output-json', type=Path)
            parser.add_argument('--output-md', type=Path)
            args = parser.parse_args()
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps({'decision': {'action': 'stop_after_pilot'}}))
            args.output_md.write_text('# gate\n')
            """,
        )
        env = {
            **os.environ,
            "P2_AB_ROOT": str(temp_path / "ab-root"),
            "P2_EXPECTED_SHA": "a" * 40,
            "P2_EXPECTED_CANDLE_SHA": "b" * 40,
            "P2_EXPECTED_BINARY_SHA": "c" * 64,
            "P2_GEOMETRY_SKIP_SMOKE": "1",
            "P2_TEST_LOG": str(log),
            "P2_TEST_MARKER": str(temp_path / "failed-once"),
        }
        return scripts / OVERNIGHT.name, env, log

    def test_arms_run_sequentially(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            script, env, log = self.fixture(Path(temp))
            completed = subprocess.run([str(script)], env=env, text=True, capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                log.read_text().splitlines(),
                [
                    "start control blocking=0",
                    "end control ok",
                    "start pre-rms-spatial blocking=0",
                    "end pre-rms-spatial ok",
                ],
            )

    def test_failed_eval_is_retried_once_with_cuda_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            script, env, log = self.fixture(Path(temp))
            env["P2_TEST_FAIL_ONCE"] = "1"
            completed = subprocess.run([str(script)], env=env, text=True, capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            treatment = [line for line in log.read_text().splitlines() if "pre-rms-spatial" in line]
            self.assertEqual(
                treatment,
                [
                    "start pre-rms-spatial blocking=0",
                    "end pre-rms-spatial failed",
                    "start pre-rms-spatial blocking=1",
                    "end pre-rms-spatial ok",
                    "start pre-rms-spatial blocking=1",
                    "end pre-rms-spatial ok",
                    "start pre-rms-spatial blocking=0",
                    "end pre-rms-spatial ok",
                ],
            )

    def test_stale_failed_eval_does_not_trigger_retry_for_a_new_pre_phase_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            script, env, log = self.fixture(temp_path)
            phase_log = temp_path / "ab-root" / "seed-1" / "control" / "phases.jsonl"
            phase_log.parent.mkdir(parents=True)
            phase_log.write_text(
                '{"stage":"eval-1000","status":"134"}\n'
                '{"stage":"eval-1000","status":"0"}\n'
            )
            env["P2_TEST_FAIL_WITHOUT_PHASE"] = "1"
            completed = subprocess.run([str(script)], env=env, text=True, capture_output=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertEqual(
                log.read_text().splitlines(),
                ["start control blocking=0", "end control failed-without-phase"],
            )
            self.assertIn("failed outside evaluation; not retrying", completed.stderr)

    def test_pending_failed_eval_is_recovered_and_repeated_before_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            script, env, log = self.fixture(temp_path)
            treatment_phase = (
                temp_path
                / "ab-root"
                / "seed-1"
                / "pre-rms-spatial"
                / "phases.jsonl"
            )
            treatment_phase.parent.mkdir(parents=True)
            treatment_phase.write_text('{"stage":"eval-1000","status":"134"}\n')
            completed = subprocess.run([str(script)], env=env, text=True, capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            treatment = [line for line in log.read_text().splitlines() if "pre-rms-spatial" in line]
            self.assertEqual(
                treatment,
                [
                    "start pre-rms-spatial blocking=1",
                    "end pre-rms-spatial ok",
                    "start pre-rms-spatial blocking=1",
                    "end pre-rms-spatial ok",
                    "start pre-rms-spatial blocking=0",
                    "end pre-rms-spatial ok",
                ],
            )


class GeometryActionRunnerTests(unittest.TestCase):
    def test_recovery_repeat_requires_identical_report_before_returning(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            repo = temp_path / "repo"
            scripts = repo / "scripts"
            scripts.mkdir(parents=True)
            shutil.copy2(ACTION_RUNNER, scripts / ACTION_RUNNER.name)
            shutil.copy2(REPO_ROOT / "scripts" / "p2_artifacts.py", scripts / "p2_artifacts.py")
            candle = temp_path / "candle_graph" / ".git"
            candle.mkdir(parents=True)
            shim = temp_path / "bin"
            shim.mkdir()
            executable(
                shim / "git",
                r"""
                #!/usr/bin/env bash
                if [[ "$*" == *"status --porcelain"* ]]; then
                  exit 0
                elif [[ "$*" == *"candle_graph"* ]]; then
                  printf '%s\n' "$P2_EXPECTED_CANDLE_SHA"
                else
                  printf '%s\n' "$P2_EXPECTED_SHA"
                fi
                """,
            )
            executable(shim / "nvidia-smi", "#!/usr/bin/env bash\nprintf '0,fake,0,1,0,0\\n'\n")
            fake_tofy = temp_path / "tofy"
            executable(
                fake_tofy,
                r"""
                #!/usr/bin/env bash
                set -euo pipefail
                subcommand="$1"
                shift
                while (($#)); do
                  case "$1" in
                    --output) output="$2"; shift 2 ;;
                    --episode-jsonl) episodes="$2"; shift 2 ;;
                    *) shift ;;
                  esac
                done
                [[ "$subcommand" == p2-eval ]]
                printf '{"schema":"p2.eval_report.v9","value":1}\n' >"$output"
                printf '{}\n' >"$episodes"
                """,
            )
            ab_root = temp_path / "ab-sigreg-geometry-v2"
            arm = ab_root / "seed-1" / "control"
            checkpoint = arm / "checkpoints" / "step-000000001000"
            eval_dir = arm / "eval-update-1000"
            eval_dir.mkdir(parents=True)
            checkpoint.mkdir(parents=True)
            artifacts = []
            for name, contents in (
                ("model.safetensors", b"model"),
                ("optimizer.safetensors", b"optimizer"),
                ("trainer_state.json", b"{}\n"),
            ):
                path = checkpoint / name
                path.write_bytes(contents)
                artifacts.append(path)
            config = arm / "config.json"
            config.write_text("{}\n")
            report = eval_dir / "eval_report.json"
            report.write_text('{"schema":"p2.eval_report.v9","value":1}\n')
            artifacts.extend((config, report))
            (eval_dir / "sha256.txt").write_text(
                "".join(
                    f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(ab_root)}\n"
                    for path in artifacts
                )
            )
            (arm / "phases.jsonl").write_text(
                json.dumps({"stage": "eval-1000", "status": "0"}) + "\n"
            )
            expected_sha = "a" * 40
            expected_candle_sha = "b" * 40
            binary_sha = hashlib.sha256(fake_tofy.read_bytes()).hexdigest()
            env = {
                **os.environ,
                "PATH": f"{shim}:{os.environ['PATH']}",
                "TOFY_BIN": str(fake_tofy),
                "P2_AB_ROOT": str(ab_root),
                "P2_CANDLE_ROOT": str(candle.parent),
                "P2_AB_EXPERIMENT": "geometry",
                "P2_AB_TARGET_UPDATE": "2000",
                "P2_AB_RECOVERY_ONLY_UPDATE": "1000",
                "P2_AB_REPEAT_EVAL_UPDATE": "1000",
                "P2_GPU_SAMPLE_INTERVAL": "0.05",
                "P2_EXPECTED_SHA": expected_sha,
                "P2_EXPECTED_CANDLE_SHA": expected_candle_sha,
                "P2_EXPECTED_BINARY_SHA": binary_sha,
            }
            completed = subprocess.run(
                [str(scripts / ACTION_RUNNER.name), "control", "1"],
                env=env,
                text=True,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            verification = json.loads((eval_dir / "repeat-verification.json").read_text())
            self.assertTrue(verification["identical"])
            self.assertFalse((eval_dir / "repeat-required").exists())

class GeometrySmokeTests(unittest.TestCase):
    def test_smoke_exercises_train_and_eval_for_both_arms_in_order(self) -> None:
        self.assertTrue(SMOKE.is_file(), f"missing smoke entrypoint: {SMOKE}")
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            log = temp_path / "commands.log"
            fake_tofy = temp_path / "tofy"
            executable(
                fake_tofy,
                r"""
                #!/usr/bin/env bash
                set -euo pipefail
                subcommand="$1"
                shift
                physical_batch=""
                synthetic_episodes=""
                while (($#)); do
                  case "$1" in
                    --output-dir) output_dir="$2"; shift 2 ;;
                    --output) output="$2"; shift 2 ;;
                    --episode-jsonl) episodes="$2"; shift 2 ;;
                    --physical-batch) physical_batch="$2"; shift 2 ;;
                    --synthetic-episodes) synthetic_episodes="$2"; shift 2 ;;
                    *) shift ;;
                  esac
                done
                if [[ "$subcommand" == p2-train ]]; then
                  printf 'p2-train batch=%s\n' "$physical_batch" >>"$P2_TEST_LOG"
                  mkdir -p "$output_dir/checkpoints/step-000000000004"
                  printf model >"$output_dir/checkpoints/step-000000000004/model.safetensors"
                  printf '{}\n' >"$output_dir/config.json"
                else
                  printf 'p2-eval episodes=%s batch=%s\n' \
                    "$synthetic_episodes" "$physical_batch" >>"$P2_TEST_LOG"
                  mkdir -p "$(dirname "$output")"
                  printf '{"schema":"p2.eval_report.v9"}\n' >"$output"
                  printf '{}\n' >"$episodes"
                fi
                """,
            )
            env = {
                **os.environ,
                "TOFY_BIN": str(fake_tofy),
                "P2_SMOKE_ROOT": str(temp_path / "smoke"),
                "P2_TEST_LOG": str(log),
            }
            completed = subprocess.run([str(SMOKE)], env=env, text=True, capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            repeated = subprocess.run([str(SMOKE)], env=env, text=True, capture_output=True)
            self.assertEqual(repeated.returncode, 0, repeated.stderr)
            self.assertIn("preserving verified geometry smoke", repeated.stdout)
            self.assertEqual(
                log.read_text().splitlines(),
                ["p2-train batch=1024", "p2-eval episodes=64 batch=1024"] * 2,
            )


if __name__ == "__main__":
    unittest.main()
