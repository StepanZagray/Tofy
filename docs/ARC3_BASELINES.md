# Offline public-game reference agents

`python/tofy_arc3/run_baseline.py` runs both reference agents through the same
official local engine loop. Existing `run_local.py` and `python/kaggle/my_agent.py`
already provide Tofy integration; this runner adds a controlled comparison with
a local pretrained model, explicit budgets, provenance and failure artifacts.

The engine owns score calculation. `engine_score` preserves its root `score`
on the native 0–100 scale for the explicitly registered local public subset.
It is not a Private Kaggle score. Do not average every entry in `runs`: an empty
initialization run can otherwise halve the value. This runner calls `make` once
per game and reads `observation_space`, avoiding the legacy second opening RESET.
Action budgets and report `actions` count policy actions plus charged GAME_OVER
retry resets. `non_reset_actions` and `resets` separate those components. The
runner rejects disagreement with the engine's per-game or total action count.
The [charged-action correction](research/2026-09-06-arc3-charged-actions-confirmation.md)
records the earlier screen's accounting deviation and the fresh verification.

## Setup and execution

Use Python 3.12 with pinned `arc-agi==0.9.9`, `arcengine==0.9.3`, and the exact
resolved NumPy version recorded in each run. Download only the registered public
game versions into a separate directory before evaluation. Never pass game
source, metadata, a scorecard, or another arm's trajectories to a policy.

```bash
PYTHONPATH=python /path/to/arcagi-venv/bin/python -m unittest -v \
  tofy_arc3.tests.test_baseline tofy_arc3.tests.test_local_chat
PYTHONPATH=python /path/to/arcagi-venv/bin/python -m tofy_arc3.run_baseline \
  --config /path/to/frozen-reference.json --output-dir /path/to/new-run
```

The output directory must not exist. Run from a clean, reviewed, pushed commit.
Before a model screen, bind the exact configuration and artifact hashes to its
preregistration and run the bounded engine/device smoke. Required fields:

```json
{
  "evidence_class": "exploratory",
  "games": ["ar25-0c556536", "bp35-0a0ad940", "cd82-fb555c5d"],
  "environments_dir": "/absolute/path/to/only-these-game-versions",
  "seed": 0,
  "expected_screen_contract_sha256": "905416cf8ac752e34628cd3f9bed796e503cf7f5ad96fcd0313c1c61d72f8a76",
  "max_actions_per_game": 128,
  "max_actions_per_level": 128,
  "max_level_retries": 3,
  "max_seconds_per_game": 180,
  "max_seconds_total": 600,
  "decision_timeout": 30,
  "limitations": ["Single-seed public-subset screen; no generalization promotion"],
  "agent": {
    "kind": "local_chat",
    "model": "prize-baseline-qwen",
    "model_file": "/absolute/path/to/model.gguf",
    "server_binary": "/usr/bin/llama-server",
    "max_tokens": 1024,
    "history_turns": 4,
    "context_size": 32768,
    "prompt_batch": 1024,
    "physical_batch": 512,
    "gpu_layers": 99,
    "reasoning_budget": 256,
    "seed": 0
  },
  "artifacts": [
    {"path": "/absolute/path/to/model.gguf", "sha256": "REQUIRED_DIGEST"},
    {"path": "/usr/bin/llama-server", "sha256": "REQUIRED_DIGEST"}
  ]
}
```

For Tofy, `agent` instead contains `kind: tofy`, `binary`, `checkpoint`,
`train_config`, `device` and `policy` (`greedy` or `phase-a`). All three source
files require matching artifact hashes. Optional `phase_a_calibration` must be
recorded and hashed. A fresh process is created for each game and its runtime
config redirects transient GPU locks outside the sealed training root. This
version uses frozen weights without online adaptation. The bridge's
`finish_observation` operation ingests the final capped frame and closes its
stream without computing another decision or executing another action.

The local-chat baseline uses a fixed prompt, lossless palette text, bounded
recent history and strict JSON actions. It has no vision encoder, executable
code tools or explicit long-term rule ledger yet. These are separate treatments.
Only numeric loopback endpoints are accepted; no provider key, proxy, redirect,
random fallback or hidden-state access is used. Model-serving parameters and
loaded-model identity are recorded by the runner, which starts and stops the
exact hashed server binary and weights for each game. Startup and gameplay
times are separate; the suite wall-time budget includes both and cleanup.

## Evidence and limitations

- `config.json`: exact configuration; `provenance.json`: source, package,
  artifact and environment-file hashes.
- Per-game `trajectory.jsonl`: every received animation frame, actual action,
  terminal feedback and reset. Tofy stderr/runtime config/process identity are
  kept in that game's directory.
- `toolkit_scorecard.json`: canonical local engine score, with credential field
  removed. `report.json`: completion, actions, resets, elapsed time and stop reason.
- `manifest.json`: finalized artifact hashes after agent subprocess shutdown.
  Record its digest outside the run before analysis; this is point-in-time integrity.
- Time/action limits produce explicit bounded outcomes. Transport, invalid
  actions, unexpected resets and integrity errors produce failed reports and
  may not become model-performance evidence.

The first three-game screen only establishes a reproducible operational
reference. Its 128-action/180-second caps are not the official prize budget.
One seed and previously public games cannot support a generalization claim.
Fresh larger-population and multi-seed comparisons must follow a preregistered
decision; do not tune on this screen and present the same games as held out.

Qwen3.5-9B is the initial local candidate, not a claim of the best eligible model.
Upstream revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` uses Apache-2.0.
The Unsloth Q4_K_M artifact at revision
`3885219b6810b007914f3a7950a8d1b469d598a5` is pinned separately; quantization,
text-only input and the laptop budget limit what its score means. Source:
<https://huggingface.co/Qwen/Qwen3.5-9B> and
<https://huggingface.co/unsloth/Qwen3.5-9B-GGUF>. Retrieval: September 6, 2026 CDT.
This candidate license review does not certify the entire repository or a
competition submission; the binding payout/all-component release review remains.
