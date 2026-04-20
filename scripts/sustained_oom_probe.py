#!/usr/bin/env python3
"""Sustained CUDA OOM probe for Tofy training stages.

This is intentionally separate from unit tests and runtime smoke tests. OOM
behavior depends on the GPU, driver, CUDA allocator, dtype, batch shape, and how
long the training loop runs. A one-step smoke test can prove that a shape starts;
it cannot prove that the shape is safe for a long run.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "local_models"


def run_checked(
    cmd: list[str],
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
) -> None:
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def query_vram() -> dict[str, int] | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    line = out.splitlines()[0].strip()
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 3:
        return None
    try:
        used, free, total = (int(float(part)) for part in parts)
    except ValueError:
        return None
    return {"used_mb": used, "free_mb": free, "total_mb": total}


def latest_artifact(pattern: str) -> Path:
    candidates = [
        path
        for path in MODEL_DIR.glob(pattern)
        if path.is_file() and path.name.count(".safetensors") == 1
    ]
    if not candidates:
        raise RuntimeError(f"no artifact found for {pattern}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def backup_local_models(probe_dir: Path) -> Path:
    backup = probe_dir / "local_models.backup"
    if backup.exists():
        shutil.rmtree(backup)
    backup.mkdir(parents=True)
    if not MODEL_DIR.exists():
        return backup
    for src in MODEL_DIR.rglob("*"):
        if not src.is_file():
            continue
        dst = backup / src.relative_to(MODEL_DIR)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return backup


def restore_local_models(backup: Path) -> None:
    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        return
    for src in backup.rglob("*"):
        if not src.is_file():
            continue
        dst = MODEL_DIR / src.relative_to(backup)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def write_probe_data(path: Path, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for i in range(rows):
            prompt = " ".join(
                [
                    f"user asks rust task {i}",
                    f"value_{i}",
                    f"condition_{i % 97}",
                    f"context_token_{i}",
                    f"helper_name_{i}",
                    "return only rust code",
                ]
            )
            answer = " ".join(
                [
                    "<action:code>",
                    f"fn function_{i}(value_{i}: i32) -> i32 {{",
                    f"let shifted_{i} = value_{i} + {i % 17};",
                    f"if shifted_{i} > {i % 31} {{ shifted_{i} }} else {{ shifted_{i} + 1 }}",
                    "}",
                    f"unique_code_token_{i}",
                ]
            )
            handle.write(f"{prompt}\t{answer}\tcode\n")


def base_env(args: argparse.Namespace, stage_name: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "TOFY_TRAIN_DTYPE": args.dtype,
            "TOFY_LATENT_CONTEXT_SEGMENTS": "4",
            "TOFY_LATENT_RECENT_FULL_SEGMENTS": "1",
            "TOFY_LATENT_HISTORY_RATIO": "0.35",
            "TOFY_LATENT_WARMUP_STEPS": "0",
            "TOFY_WORLD_CONTEXT_SEGMENTS": "2",
            "TOFY_WORLD_RECENT_FULL_SEGMENTS": "1",
            "TOFY_RECURSIVE_PLANNER_MEMORY": "1",
            "TOFY_WORLD_TRAIN_ROLLOUT_STEPS": "2",
            "TOFY_WORLD_ROLLOUT_STEPS": "2",
            "TOFY_RUN_GROUP": args.run_group,
            "TOFY_RUN_STAGE_NAME": stage_name,
        }
    )
    return env


def world_env(args: argparse.Namespace, stage_name: str) -> dict[str, str]:
    env = base_env(args, stage_name)
    env["TOFY_WORLD_WARMUP_BATCH"] = str(args.world_warmup_batch)
    env["TOFY_WORLD_WARMUP_GRAD_ACCUM"] = str(args.world_warmup_accum)
    if args.world_warmup_steps is None:
        env.pop("TOFY_WORLD_WARMUP_STEPS", None)
    else:
        env["TOFY_WORLD_WARMUP_STEPS"] = str(args.world_warmup_steps)
    return env


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {
            "sample_count": 0,
            "peak_used_mb": None,
            "min_free_mb": None,
            "total_mb": None,
            "late_growth_mb": None,
        }
    peak_used = max(sample["used_mb"] for sample in samples)
    min_free = min(sample["free_mb"] for sample in samples)
    total = max(sample["total_mb"] for sample in samples)
    half_idx = max(0, len(samples) // 2)
    late_growth = samples[-1]["used_mb"] - samples[half_idx]["used_mb"]
    return {
        "sample_count": len(samples),
        "peak_used_mb": peak_used,
        "min_free_mb": min_free,
        "total_mb": total,
        "peak_fraction": peak_used / total if total else None,
        "late_growth_mb": late_growth,
        "first_used_mb": samples[0]["used_mb"],
        "last_used_mb": samples[-1]["used_mb"],
    }


def run_measured(
    name: str,
    cmd: list[str],
    env: dict[str, str],
    probe_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    log_path = probe_dir / f"{name}.log"
    samples: list[dict[str, Any]] = []
    stop = False
    start = time.time()

    def sampler() -> None:
        while not stop:
            snap = query_vram()
            if snap is not None:
                snap["elapsed_sec"] = round(time.time() - start, 3)
                samples.append(snap)
            time.sleep(args.sample_interval)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    stop = True
    thread.join(timeout=2)

    text = log_path.read_text(encoding="utf-8", errors="replace")
    summary = summarize_samples(samples)
    oom = "CUDA_ERROR_OUT_OF_MEMORY" in text or "out of memory" in text.lower()
    headroom_ok = (
        summary["min_free_mb"] is not None
        and summary["min_free_mb"] >= args.min_headroom_mb
    )
    growth_ok = (
        summary["late_growth_mb"] is None
        or args.max_late_growth_mb <= 0
        or summary["late_growth_mb"] <= args.max_late_growth_mb
    )
    passed = proc.returncode == 0 and not oom and headroom_ok and growth_ok
    result = {
        "name": name,
        "cmd": cmd,
        "return_code": proc.returncode,
        "seconds": round(time.time() - start, 2),
        "log": str(log_path),
        "oom": oom,
        "headroom_ok": headroom_ok,
        "growth_ok": growth_ok,
        "passed": passed,
        "min_headroom_mb": args.min_headroom_mb,
        "max_late_growth_mb": args.max_late_growth_mb,
        **summary,
        "tail": "\n".join(text.splitlines()[-30:]),
    }
    sample_path = probe_dir / f"{name}.vram_samples.jsonl"
    with sample_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample) + "\n")
    result["samples_path"] = str(sample_path)
    print(json.dumps({k: v for k, v in result.items() if k != "tail"}, indent=2), flush=True)
    return result


def latent_cmd(args: argparse.Namespace, data_path: Path, steps: int) -> list[str]:
    return [
        str(args.binary),
        "--latent",
        str(data_path),
        str(steps),
        str(args.latent_batch),
        str(args.dim),
        str(args.max_seq),
        str(args.layers),
        str(args.heads),
        str(args.vocab),
        "--grad-accum",
        str(args.latent_accum),
    ]


def world_cmd(
    args: argparse.Namespace,
    data_path: Path,
    latent: Path,
    vocab: Path,
    steps: int,
) -> list[str]:
    return [
        str(args.binary),
        "--train-world",
        str(latent),
        str(vocab),
        str(data_path),
        str(steps),
        str(args.world_batch),
        str(args.dim),
        str(args.max_seq),
        str(args.layers),
        str(args.heads),
        str(args.bridge_dim),
        str(args.planner_slots),
        "--lambda",
        str(args.world_lambda),
        "--lr",
        str(args.world_lr),
        "--grad-accum",
        str(args.world_accum),
        "--action-loss-weight",
        str(args.world_action_loss_weight),
        "--router-warmup",
        str(args.world_router_warmup),
    ]


def decoder_cmd(
    args: argparse.Namespace,
    data_path: Path,
    latent: Path,
    vocab: Path,
    world: Path,
    steps: int,
    output_path: Path,
) -> list[str]:
    return [
        str(args.binary),
        "--train-decoder",
        str(latent),
        str(vocab),
        str(world),
        str(data_path),
        str(steps),
        str(args.decoder_batch),
        str(args.decoder_max_seq),
        str(args.dim),
        str(args.layers),
        str(args.heads),
        str(args.bridge_dim),
        str(args.planner_slots),
        "--decoder-kind",
        "code",
        "--decoder-max-vocab",
        str(args.decoder_max_vocab),
        "--decoder-output",
        str(output_path),
        "--grad-accum",
        str(args.decoder_accum),
    ]


def ensure_setup_latent(args: argparse.Namespace, data_path: Path) -> tuple[Path, Path]:
    if args.latent_model:
        latent = Path(args.latent_model)
        vocab = Path(args.encoder_vocab) if args.encoder_vocab else MODEL_DIR / "vocabs/vocab_encoder.txt"
        return latent, vocab
    run_checked(
        latent_cmd(args, data_path, args.setup_latent_steps),
        env=base_env(args, "setup_latent"),
    )
    return latest_artifact("model_latent_*.safetensors"), MODEL_DIR / "vocabs/vocab_encoder.txt"


def ensure_setup_world(
    args: argparse.Namespace, data_path: Path, latent: Path, vocab: Path
) -> Path:
    if args.world_model:
        return Path(args.world_model)
    run_checked(
        world_cmd(args, data_path, latent, vocab, args.setup_world_steps),
        cwd=ROOT,
        env=world_env(args, "setup_world"),
    )
    return latest_artifact("model_world_*.safetensors")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sustained CUDA OOM probes with external nvidia-smi sampling."
    )
    parser.add_argument("--stage", choices=["all", "latent", "world", "decoder"], default="all")
    parser.add_argument("--quick", action="store_true", help="Use very short step counts to validate the probe itself.")
    parser.add_argument("--probe-dir", type=Path, default=None)
    parser.add_argument("--keep-local-models", action="store_true", help="Do not restore local_models after the probe.")
    parser.add_argument("--build", action="store_true", help="Run cargo build --release before probing.")
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/jepa_ai")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "f16", "f32"])
    parser.add_argument("--sample-interval", type=float, default=0.10)
    parser.add_argument("--min-headroom-mb", type=int, default=512)
    parser.add_argument("--max-late-growth-mb", type=int, default=512, help="0 disables late-growth failure.")
    parser.add_argument("--data-rows", type=int, default=24000)
    parser.add_argument("--run-group", default=f"oom_probe_{time.strftime('%Y-%m-%d_%H-%M-%S')}")

    parser.add_argument("--dim", type=int, default=640)
    parser.add_argument("--max-seq", type=int, default=256)
    parser.add_argument("--layers", type=int, default=7)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--bridge-dim", type=int, default=640)
    parser.add_argument("--planner-slots", type=int, default=64)
    parser.add_argument("--vocab", type=int, default=8000)

    parser.add_argument("--latent-steps", type=int, default=1000)
    parser.add_argument("--latent-batch", type=int, default=32)
    parser.add_argument("--latent-accum", type=int, default=1)

    parser.add_argument("--world-steps", type=int, default=8000)
    parser.add_argument("--world-batch", type=int, default=64)
    parser.add_argument("--world-accum", type=int, default=2)
    parser.add_argument("--world-warmup-batch", type=int, default=64)
    parser.add_argument("--world-warmup-accum", type=int, default=1)
    parser.add_argument("--world-warmup-steps", type=int, default=None)
    parser.add_argument("--world-lambda", type=float, default=0.2)
    parser.add_argument("--world-lr", type=float, default=2e-4)
    parser.add_argument("--world-action-loss-weight", type=float, default=1.0)
    parser.add_argument("--world-router-warmup", type=int, default=0)

    parser.add_argument("--decoder-steps", type=int, default=1000)
    parser.add_argument("--decoder-batch", type=int, default=12)
    parser.add_argument("--decoder-accum", type=int, default=2)
    parser.add_argument("--decoder-max-seq", type=int, default=224)
    parser.add_argument("--decoder-max-vocab", type=int, default=16000)

    parser.add_argument("--setup-latent-steps", type=int, default=1)
    parser.add_argument("--setup-world-steps", type=int, default=2)
    parser.add_argument("--latent-model", type=Path, default=None)
    parser.add_argument("--encoder-vocab", type=Path, default=None)
    parser.add_argument("--world-model", type=Path, default=None)
    args = parser.parse_args()
    if args.quick:
        args.latent_steps = min(args.latent_steps, 3)
        args.world_steps = min(args.world_steps, 4)
        args.decoder_steps = min(args.decoder_steps, 3)
        args.min_headroom_mb = min(args.min_headroom_mb, 128)
        args.max_late_growth_mb = 0
    return args


def main() -> int:
    args = parse_args()
    if query_vram() is None:
        print("ERROR: nvidia-smi is required for sustained OOM probes.", file=sys.stderr)
        return 2
    if args.build:
        run_checked(["cargo", "build", "--release"])
    if not args.binary.exists():
        print(f"ERROR: release binary not found: {args.binary}", file=sys.stderr)
        print("Run: cargo build --release", file=sys.stderr)
        return 2

    probe_dir = args.probe_dir or Path(tempfile.mkdtemp(prefix="tofy-oom-probe-"))
    probe_dir.mkdir(parents=True, exist_ok=True)
    data_path = probe_dir / "world_code_synth.txt"
    write_probe_data(data_path, args.data_rows)
    backup = backup_local_models(probe_dir)
    print(f"OOM probe dir: {probe_dir}")
    print(f"Stage: {args.stage}")
    print(f"Minimum required free VRAM: {args.min_headroom_mb} MB")
    print(f"Maximum allowed late growth: {args.max_late_growth_mb} MB")

    results: list[dict[str, Any]] = []
    try:
        latent: Path | None = None
        vocab: Path | None = None
        world: Path | None = None

        def write_summary() -> Path:
            summary_path = probe_dir / "summary.json"
            summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            return summary_path

        if args.stage in ("all", "latent"):
            result = run_measured(
                "latent",
                latent_cmd(args, data_path, args.latent_steps),
                base_env(args, "latent"),
                probe_dir,
                args,
            )
            results.append(result)
            if not result["passed"]:
                print(f"Summary: {write_summary()}")
                return 1
            latent = latest_artifact("model_latent_*.safetensors")
            vocab = MODEL_DIR / "vocabs/vocab_encoder.txt"
        elif args.stage in ("world", "decoder"):
            latent, vocab = ensure_setup_latent(args, data_path)

        if args.stage in ("all", "world"):
            assert latent is not None and vocab is not None
            result = run_measured(
                "world",
                world_cmd(args, data_path, latent, vocab, args.world_steps),
                world_env(args, "world"),
                probe_dir,
                args,
            )
            results.append(result)
            if not result["passed"]:
                print(f"Summary: {write_summary()}")
                return 1
            world = latest_artifact("model_world_*.safetensors")
        elif args.stage == "decoder":
            assert latent is not None and vocab is not None
            world = ensure_setup_world(args, data_path, latent, vocab)

        if args.stage in ("all", "decoder"):
            assert latent is not None and vocab is not None
            if world is None:
                world = latest_artifact("model_world_*.safetensors")
            result = run_measured(
                "decoder",
                decoder_cmd(
                    args,
                    data_path,
                    latent,
                    vocab,
                    world,
                    args.decoder_steps,
                    probe_dir / "code_decoder_oom_probe.safetensors",
                ),
                base_env(args, "decoder"),
                probe_dir,
                args,
            )
            results.append(result)
            if not result["passed"]:
                print(f"Summary: {write_summary()}")
                return 1

        summary_path = write_summary()
        failed = [result for result in results if not result["passed"]]
        print(f"Summary: {summary_path}")
        if failed:
            print("OOM probe failed:")
            for result in failed:
                print(
                    f"- {result['name']}: rc={result['return_code']} oom={result['oom']} "
                    f"min_free={result['min_free_mb']}MB late_growth={result['late_growth_mb']}MB "
                    f"log={result['log']}"
                )
            return 1
        print("OOM probe passed.")
        return 0
    finally:
        if args.keep_local_models:
            print("Keeping local_models changes from probe.")
        else:
            restore_local_models(backup)
            print("Restored local_models after probe.")


if __name__ == "__main__":
    raise SystemExit(main())
