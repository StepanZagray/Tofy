#!/usr/bin/env python3
"""Bounded local launcher for the registered synthetic looped-agent screen."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def memory():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total,temperature.gpu",
         "--format=csv,noheader,nounits", "--id=0"],
        check=True, capture_output=True, text=True, timeout=5,
    )
    return [int(x.strip()) for x in result.stdout.strip().split(",")]


def verify_run(root):
    manifest = root / "manifest.json"
    expected = root.with_suffix(".manifest.sha256").read_text().strip()
    if digest(manifest) != expected:
        raise RuntimeError("manifest digest mismatch")
    for name, expected in json.loads(manifest.read_text())["files"].items():
        if digest(root / name) != expected:
            raise RuntimeError(f"artifact digest mismatch: {name}")


def execute(args, name, mode, batch, effective, seconds):
    root = args.root / name
    command = [str(args.binary), "--mode", mode, "--output-dir", str(root),
               "--device", "cuda:0", "--batch", str(batch), "--effective-batch", str(effective),
               "--max-seconds", str(seconds), "--search"]
    log = args.root / (name + ".stdout.log")
    started = time.monotonic()
    peak = 0
    with log.open("w") as output, (args.root / (name + ".telemetry.jsonl")).open("w") as telemetry:
        child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        save(args.root / (name + ".process.json"), {
            "pid": child.pid, "supervisor_pid": os.getpid(), "command": command,
            "started_local": datetime.datetime.now().astimezone().isoformat(),
        })
        try:
            while child.poll() is None:
                used, total, temperature = memory()
                peak = max(peak, used)
                telemetry.write(json.dumps({"elapsed_seconds": time.monotonic()-started,
                                           "memory_used_mib": used, "memory_total_mib": total,
                                           "temperature_c": temperature}) + "\n")
                telemetry.flush()
                if time.monotonic()-started > seconds + 30:
                    raise TimeoutError("external run watchdog expired")
                time.sleep(0.2)
            code = child.wait()
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            save(args.root / (name + ".exit.json"), {
                "returncode": child.returncode, "pid_gone": not Path(f"/proc/{child.pid}").exists(),
                "elapsed_seconds": time.monotonic()-started,
            })
    if not (root / "report.json").exists() or not (root / "manifest.json").exists():
        raise RuntimeError(f"unsealed infrastructure failure; inspect {log}")
    verify_run(root)
    report = json.loads((root / "report.json").read_text())
    metadata = json.loads((root / "metadata.json").read_text()) if (root / "metadata.json").exists() else {}
    if metadata.get("provenance", {}).get("binary_sha256") != digest(args.binary):
        raise RuntimeError("probe binary provenance mismatch")
    if code and not any(word in str(report).lower() for word in ("out_of_memory", "out of memory", "cuda_error_out_of_memory")):
        raise RuntimeError(f"probe failed for a non-capacity reason: {report}")
    _, total, _ = memory()
    accepted = code == 0 and report["status"] == "complete_pending_analysis" and total-peak >= 512
    result = {"name": name, "batch": batch, "accepted": accepted, "peak_memory_mib": peak,
              "memory_total_mib": total, "headroom_mib": total-peak, "returncode": code,
              "seconds": time.monotonic()-started, "manifest_sha256": digest(root / "manifest.json")}
    save(args.root / (name + ".summary.json"), result)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["capacity", "screen"])
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--capacity-root", type=Path)
    args = parser.parse_args()
    args.binary = args.binary.resolve(strict=True)
    args.root.mkdir()
    save(args.root / "launcher.json", {"pid": os.getpid(), "binary": str(args.binary),
                                     "binary_sha256": digest(args.binary), "kind": args.kind})
    if args.kind == "screen":
        if args.capacity_root is None:
            raise RuntimeError("screen requires a completed capacity root")
        capacity = json.loads((args.capacity_root / "selected.json").read_text())
        if capacity["binary_sha256"] != digest(args.binary):
            raise RuntimeError("capacity was measured on another binary")
        if capacity["conservative_training_seconds"] > 1500:
            raise RuntimeError("measured runtime leaves insufficient evaluation reserve; revise registration")
        save(args.root / "selection.json", capacity)
        execute(args, "screen-seed0", "train", capacity["physical_batch"], 64, 1800)
        return
    low, high, candidate = 0, 65, 8
    results = []
    while high-low > 1:
        if len(results) >= 11:
            raise RuntimeError("registered capacity probe count exhausted")
        result = execute(args, f"capacity-{len(results):02d}-b{candidate}", "smoke", candidate, candidate, 300)
        results.append(result)
        if result["accepted"]:
            low = candidate
            candidate = min(64, candidate*2) if high == 65 else (low+high)//2
        else:
            high = candidate
            candidate = (low+high)//2
        if candidate == 0:
            raise RuntimeError("no feasible physical batch")
    repeat = execute(args, f"repeat-b{low}", "smoke", low, low, 300)
    if not repeat["accepted"]:
        raise RuntimeError("selected physical batch was not stable on repetition")
    timings = [json.loads(line)["update_seconds"] for line in
               (args.root / repeat["name"] / "updates.jsonl").read_text().splitlines()]
    selected = {"binary_sha256": digest(args.binary), "physical_batch": low,
                "effective_batch": 64, "accumulation": math.ceil(64/low),
                "last_microbatch": 64-low*(math.ceil(64/low)-1),
                "conservative_training_seconds": max(timings)*math.ceil(64/low)*256,
                "repeat": repeat, "probes": results,
                "classification": "implementation_smoke_only"}
    save(args.root / "selected.json", selected)
    print(json.dumps(selected), flush=True)


if __name__ == "__main__":
    main()
