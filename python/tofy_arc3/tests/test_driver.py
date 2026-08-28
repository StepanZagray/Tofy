#!/usr/bin/env python3
"""Offline self-test for the ARC-AGI-3 local toolkit driver."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "python" / "tofy_arc3" / "run_local.py"
MOCK = Path(__file__).with_name("mock_bridge.py")
ENVIRONMENTS = Path("/tmp/arcagi-workdir/environment_files")


def main() -> None:
    assert ENVIRONMENTS.joinpath("ls20", "9607627b", "metadata.json").is_file()
    with tempfile.TemporaryDirectory(prefix="tofy-arc3-driver-") as temp:
        temp_path = Path(temp)
        shim = temp_path / "mock-tofy"
        shim.write_text(
            "#!/bin/sh\n"
            f'exec "{sys.executable}" "{MOCK}" "$@"\n',
            encoding="utf-8",
        )
        shim.chmod(shim.stat().st_mode | stat.S_IXUSR)
        output_dir = temp_path / "output"
        environment = os.environ.copy()
        environment["OPERATION_MODE"] = "competition"
        result = subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "--bin",
                str(shim),
                "--device",
                "cpu",
                "--checkpoint",
                str(temp_path / "unused.safetensors"),
                "--train-config",
                str(temp_path / "unused-config.json"),
                "--games",
                "ls20-9607627b",
                "--environments-dir",
                str(ENVIRONMENTS),
                "--output-dir",
                str(output_dir),
            ],
            env=environment,
            text=True,
            capture_output=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"driver failed with {result.returncode}\nstdout:\n{result.stdout}"
            f"\nstderr:\n{result.stderr}"
        )

        report_path = output_dir / "arc3_local_report.json"
        scorecard_path = output_dir / "toolkit_scorecard.json"
        summary_path = output_dir / "local_summary.json"
        assert report_path.is_file()
        assert scorecard_path.is_file()
        assert summary_path.is_file()

        report = json.loads(report_path.read_text(encoding="utf-8"))
        scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert report["card_id"] == scorecard["card_id"]
        assert summary["totals"]["games_played"] == 1
        assert summary["totals"]["games_won"] in (0, 1)
        assert summary["totals"]["total_levels_completed"] == sum(
            game["levels_completed"] for game in summary["games"]
        )
        assert summary["totals"]["total_levels"] == scorecard["total_levels"]
        assert summary["totals"]["toolkit_scorecard_score"] == scorecard["score"]
        assert summary["games"][0]["game_id"] == "ls20-9607627b"
        assert summary["games"][0]["actions"] == 3
        assert Path(summary["arc3_local_report"]) == report_path
        assert Path(summary["toolkit_scorecard"]) == scorecard_path
        assert Path(summary["recordings_dir"]).is_dir()
        assert Path(summary["profile_dir"]).is_dir()

    print("test_driver.py: PASS")


if __name__ == "__main__":
    main()
