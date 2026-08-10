# P2 live ARC-AGI-3 evaluation

`p2-arc3-live-eval` runs a frozen P2 checkpoint against the public games returned by
the official ARC-AGI-3 API. With no `--games` filter it evaluates every discovered
game under one scorecard, then closes the scorecard and writes
`p2.arc3_live_report.v1`.

The command loads `ARC_API_KEY` from the environment or `.env`. The repository's
existing aliases `ARC_AGI_3_API_KEY` and `ARC_AGI_API` are also accepted. The key is
never written to the report or command output.

```bash
# Read-only discovery: authenticates, but opens no scorecard and submits no actions.
cargo run --release -- p2-arc3-live-eval --list-only

# Full public suite (omit --games to keep the all-public-games contract).
cargo run --release --features cudnn -- p2-arc3-live-eval \
  --checkpoint runs/p2/readiness-v2/model.best.safetensors \
  --train-config runs/p2/readiness-v2/config.json \
  --device cuda \
  --max-actions-per-game 512 \
  --output runs/p2/readiness-v2/arc3_live_report.json

# Cheap end-to-end smoke on one exact public game ID.
cargo run --release -- p2-arc3-live-eval \
  --checkpoint runs/p2/readiness-v2/model.best.safetensors \
  --train-config runs/p2/readiness-v2/config.json \
  --games '<game-id-from-list-only>' \
  --max-actions-per-game 1 \
  --output /tmp/tofy-arc3-live-smoke.json
```

## Evaluation boundary

Public observations are held-out evaluation only. The live module can load weights
and make inferences, but it has no training, gradient, optimizer, checkpoint-selection,
or curriculum interface. A source-boundary test rejects imports of the live/recording
modules, HTTP client, or API key from `src/p2/train.rs`.

Each report records SHA-256 hashes of the checkpoint and training config, the complete
discovered-game manifest, selected games, per-action masks and model scores, stop/error
reasons, the raw closed scorecard, recomputed official RHAE when the response contains
the required baselines (or an explicit parse error), and both `held_out_only=true` and
`public_data_used_for_fitting=false`.

Do not use public results to tune the policy, thresholds, curriculum, or select a
checkpoint. Freeze those choices on synthetic held-out evaluation first.

## Policy scope

The current P2 checkpoint has no trained reward or action-value head. The live policy
therefore ranks only actions allowed by the current response using predicted transition
fidelity, reliability, no-op probability, and latent action effect. It is an honest
closed-loop transfer evaluation of the existing world model, not a claim that the model
has a complete hidden-goal solver. This limitation is embedded in every report.

ACTION6 coordinates are generated from visible-object representatives plus a uniform
grid and scored in bounded batches. Server frames are fail-closed validated, and only
the last animation frame is treated as the settled observation. The HTTP client keeps
the affinity cookies established by RESET. It retries only idempotent reads such as
game discovery on transport failures, `429`, and `5xx` responses. ACTION and other
POST mutations are sent once; an ACTION transport, response-body, `429`, or `5xx`
failure is reported as an ambiguous mutation and is not retried.
