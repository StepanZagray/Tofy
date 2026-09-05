# V6 CurrentDouble frozen-checkpoint trajectory preregistration

Date: 2026-09-05  
Evidence class: diagnostic / selection-only  
Promotion authority: none  
Training authorized: no

## Bounded claim

On the exact seed-5 CurrentDouble trajectory sealed at
`baseline-floor-current-double-seed5-20260905T121849-CDT`, the raw world-core
prediction pathway either never departed meaningfully from the background
zero-control after update 256, or departed and later collapsed before update
2,048. A secondary branch tests whether raw and EMA weights materially differ
at the same checkpoint.

This experiment does not select or promote a checkpoint, weighting mode,
training budget, architecture, or ARC-3 agent. It uses no training and no
public ARC data. Any checkpoint singled out by this diagnostic must be
confirmed on a fresh population before its numbers are cited as evidence for a
treatment.

## Frozen source and artifacts

- Source checkpoint run: `/home/stepan/Coding/Personal/.tofy-build/baseline-floor-current-double-seed5-20260905T121849-CDT`.
- Source Tofy revision: `d0468a808d0d3cd2754dc1b31e4a85ab636fcc36`.
- Source run recursive seal-manifest digest:
  `e686a783b13eadc9f099d2ba5bdce68ecc10923e686db6fc86cc9bb4c02abfc5`.
- Evaluator binary:
  `/home/stepan/Coding/Personal/.tofy-build/target-baseline-floor-d0468a80/release/tofy`.
- Evaluator binary SHA-256:
  `aa4e2498a4f83295fe4f65be1a99ec1194f7f04e7a6a8926166d3d016b447611`.
- Train-config SHA-256:
  `874d53e53e68cfb5dbaada83bf25b5558f2874ae23f3af62997e13ec1263f3c1`.
- Fable judgment/correction SHA-256:
  `f9062fb3ea3afdf714fb771a49d4a539e9bc3da86163ccfca7b4b777f48def63` /
  `1b724b536891c68b502708a037cf53d5f7e982f96d5fbabc1b2bd94d4ded25cf`.

## Fixed evaluations and order

Use the ten checkpoint steps `0,2,256,512,768,1024,1280,1536,1792,2048`
and both `ema.safetensors` and `model.safetensors`, for exactly twenty
evaluations. Run step-2,048 EMA first; it is the endpoint replay gate. If it
passes, run the other nineteen in ascending step order with EMA before model,
skipping the already completed endpoint combination.

Every invocation is:

```bash
<BINARY> p2-eval \
  --checkpoint <P1>/checkpoints/step-<S>/<kind>.safetensors \
  --train-config <P1>/config.json \
  --seed 15 --iid-seed 6 --synthetic-episodes 64 --physical-batch 64 \
  --ptrm-k 1,2,4,8 --q-mse-threshold 0.05 --device cuda \
  --eval-mode representation --identifiability false --profile-eval false \
  --output <NEWROOT>/rescore-<S>-<kind>.json
```

No episode JSONL is requested. No resume, treatment, changed-budget, profile,
or public-data option is allowed. Each command gets a five-minute hard timeout.
The first endpoint replay must finish within 90 seconds or the campaign stops
because the total 30-minute budget would be at risk.

## Fail-closed identity and population gates

All twenty reports must have:

- evaluator binary SHA `aa4e2498...`, train-config SHA `874d53e5...`, CUDA,
  full `world_core_v6`, representation mode, and `research_claim=false`;
- all nine `identity.population_sha256` values identical across reports and
  equal to the sealed full P2 report, including OOD dynamics H1
  `af6572ae...` and IID dynamics H1 `811fe666...`;
- synthetic-dynamics semantic counts exactly 3,502 changed pixels, 1,394
  changed transitions, 119,219 foreground pixels, and 1,408 transitions;
- zero-control changed pixel accuracy exactly `0.6781838949171902` and changed
  exact accuracy exactly `0.2417503586800574`;
- no non-finite JSON values, hash mismatch, missing sidecar, evaluator error,
  surviving process/lock, or dirty/reused output root.

Do not compare `command_sha256` or `eval_config_sha256` across reports because
checkpoint/output identity differs by construction.

## Step-2,048 EMA endpoint replay gate

The first report's checkpoint SHA must be
`8ef9eb4cb49be16ce7e93937afe52d75431a9cf99ef0ff6760dfd02ea207b115`.
Against the sealed full P2 report on the identical population:

- one-step-prediction changed/foreground pixel and exact accuracies must match
  exactly;
- one-step-prediction changed-mask mean NLL must match within absolute `1e-4`;
- one-step latent MSE, changed learned MSE, and copy-forward MSE must match
  within relative `1e-4`;
- learned-copy-control foreground pixel accuracy must match exactly.

Any integer-ratio mismatch stops the campaign. Tolerances are not widened
after observation.

## Metrics and registered branch

For every set and checkpoint record:

1. Primary: signed changed-mask pixel-accuracy difference,
   `one_step_prediction - zero_control`.
2. One-step-prediction foreground pixel accuracy.
3. Changed-transition learned latent MSE and copy-forward MSE.
4. Raw-versus-EMA differences at the same step.

Steps 0 and 2 are reported as initialization baselines but excluded from the
never-learned foreground threshold because random decoding can attain small
chance foreground accuracy.

- **Never learned:** for every step from 256 through 2,048 and both weight
  kinds, primary difference `<= 0.005` and foreground accuracy `< 0.01`.
- **Collapsed:** some step has primary difference `>= 0.02`, or foreground
  accuracy at least its same-kind step-0 value plus `0.05`, and a later step
  returns inside both never-learned bands.
- **EMA masking:** at any same-step pair, one kind crosses a branch threshold
  while the other does not. This takes precedence and routes first to an EMA
  binding/smoothing diagnostic.
- **Unresolved:** none of those conjunctions holds. Do not improvise a fourth
  conclusion; inspect the trajectory and preregister a new discriminator.

The thresholds correspond to about 18 and 70 of 3,502 changed pixels for the
primary bands. Classification is a conjunction across all required reports;
no best-checkpoint or average selection is allowed.

## Budget, execution, and next decision

- One fresh, never-reused campaign root; sequential execution under the global
  GPU lock; exact PID/session tracking.
- Maximum 90 seconds for the first replay, five minutes for any later report,
  and 30 minutes total wall-clock.
- Seal every completed report and the recursive campaign root after Fable and
  integrator verification.
- If never-learned, the next preregistration targets gradient delivery into the
  world core and must include a tiny positive-control overfit before long
  training. If collapsed, target the first loss interval and recursion/residual
  mechanism. If EMA masking, inspect EMA update/binding first. If unresolved,
  stop for a new discriminator.

A/C/D quality training remains blocked under every result of this diagnostic.
