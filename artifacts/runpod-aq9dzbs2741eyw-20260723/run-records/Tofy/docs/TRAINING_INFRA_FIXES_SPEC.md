# Spec: Training-Loop and Infra Fixes (Bridge + Knowledge + Latent Stages)

Status: actionable change list, all findings verified against the working
tree. Complements the [Qwen knowledge-injection specification](QWEN_KNOWLEDGE_INJECTION_SPEC.md) (P0 items there —
regime split, function split, world-weight saving — have landed; this spec
covers the next layer: training-loop correctness, resumability, and eval
plumbing).

Priorities: P0 = wrong results or blocked experiment, P1 = crash/cost
exposure, P2 = performance and selection quality.

---

## P0.1 Bridge gradient accumulation is a no-op that wastes compute

`src/tasks/bridge.rs:637-645`: one batch is loaded, ONE loss is computed,
then `accumulate_scaled_gradients` backprops that same loss `grad_accum`
times scaled by `1/grad_accum`. The summed gradient is mathematically
identical to a single backward pass — effective batch stays `args.batch`
(16), not `batch × grad_accum` (128), while paying `grad_accum` extra
backward passes (~8× wasted step time with the minimal profile).

Change: restructure to true microbatching — load `grad_accum` DISTINCT
row microbatches per optimizer step; compute conditioning, token loss, and
margin loss per microbatch; accumulate scaled gradients across them; then
step. Step count semantics: `steps` counts optimizer steps.

AC: debug log prints `effective_batch = batch * grad_accum`; a two-step
run with grad_accum=2 consumes 4 distinct microbatches (assert on row
indices in a test with a tiny corpus).

## P0.2 Implement `--eval-bridge` (currently a stub)

`src/tasks/bridge.rs:700-703` returns `Ok(false)` — the advertised command
falls through to usage/error. This is the last missing piece of the ladder
(P1.3 in the Qwen knowledge-injection specification).

Change: implement per that spec — load `eval/veclab_eval.jsonl`, build
prompts per regime (Step-1 RAG mode pastes docs into the prompt with
zeroed conditioning), greedy decode via `Qwen3Bridge::generate`, write the
candidate into the harness dir, run `go build` + `go test`, aggregate
`compile_rate`/`suite_pass_rate` per subset (`seen`/`heldout`) × condition
(matched/zeroed/shuffled/swapped). Emit a JSON report; append summary to
`docs/RESULTS.md`.

AC: end-to-end eval on gold solutions reports 100% pass (harness sanity);
on stub solutions 0%.

## P0.3 World-model checkpoint selection ignores validation

`src/tasks/knowledge.rs:252-292` builds `_val_stream` / `_cached_val_stream`
and never uses them; selection at `knowledge.rs:519` uses the training-batch
snapshot (`recon + assoc + 0.2*sigreg` on the LAST microbatch of the log
window) — a stochastic single-batch metric. "Best" checkpoints can be noise
or memorization.

Change: every `log_every` (or a separate `val_every`), run a fixed-size
validation pass (e.g. 8 batches, deterministic seed) from the val stream in
no-grad mode; compute the same composite on val; select checkpoints on the
val metric. Log both train and val curves to TensorBoard.

AC: `_val_stream` no longer underscore-bound; log shows `val_sel` and best
checkpoints save only when val improves.

---

## P1.1 Knowledge stage: periodic resumable checkpoints + resume state

Verified: train/optimizer sidecars are written only AFTER the loop
(`knowledge.rs:553-557`); resume state is loaded (`:397`) but never saved
anywhere — `best_metric`/`saved_checkpoint` are lost, and a crash loses the
whole run since startup.

Change: every `TOFY_CHECKPOINT_EVERY` steps (default 500): save
`train_checkpoint`, optimizer state, and a resume sidecar
(step, best_metric, saved_checkpoint) via `util::save_resume_state`; keep
end-of-loop saves. Resume restores all three.

AC: kill a run at step ~700, resume; training continues from the last
checkpoint boundary with the same best_metric; final metrics match an
uninterrupted run within noise.

## P1.2 Bridge stage: resume support

Verified: the bridge loop (`bridge.rs:569-691`) has best/latest model saves
but no optimizer state, no resume sidecar, no `--resume`. Long bridge runs
(15k steps × 4 forwards/step with hard negatives) are crash-exposed.

Change: mirror the knowledge-stage mechanism — optimizer state + resume
sidecar (step, best_ce) saved on the same `val_every` cadence as `latest`;
`--resume` flag restores and fast-forwards the deterministic batch cursor
(after P2.2's shuffle: restore the epoch RNG seed + position).

AC: same kill/resume test as P1.1.

## P1.3 KV-cache generation (eval is quadratic without it)

`bridge.rs:364-392` `generate` re-uploads the full token vector and runs a
full forward per new token. Eval is 600 tasks × ≤512 new tokens: quadratic
attention plus per-token host→device copies makes eval hours instead of
minutes. `candle_transformers` qwen3 layers already support incremental
decoding with a KV cache; `Qwen3Bridge` bypasses it.

Change: add cached decode: forward the prompt once (`seqlen_offset=0`),
then feed one token at a time with growing offset; cross-attention K/V of
the conditioning slots are position-independent — precompute once per task.
Clear caches between tasks.

AC: parity test — cached and uncached greedy decode produce identical token
sequences on 3 prompts; eval wall-clock drops >10×.

## P1.4 Token-cache streams can hang forever

`src/data/data.rs:919-941` (`read_next_pair`): on EOF the stream resets and
loops. A zero-record cache, an all-filtered split (e.g. modulus split that
matches nothing), or an all-empty-rows file spins forever without error.
Same pattern in the other cached/raw stream `next`/`refill` loops.

Change: count consecutive resets without yielding a row; after 2 full
passes bail with a clear error naming the file and split parameters. Apply
to all stream loop sites in `data.rs`.

AC: unit tests — empty cache file and impossible split both return errors,
not hangs.

---

## P2.1 Bridge validation set is a fixed tiny sample

`bridge.rs:655`: `let sample = &val_rows[..val_rows.len().min(args.batch)]`
— always the FIRST `batch` val rows. Selection metric (P0 of the previous
spec) rides on ~16 fixed rows: noisy and biased toward whatever functions
sort first.

Change: evaluate the full val set in batches (it is small), or if capped,
a deterministic rotating sample ≥128 rows. Report mean CE over the whole
pass.

AC: val metric stable (< a few % jitter) across adjacent evals of an
unchanged model.

## P2.2 Shuffle bridge training rows

`bridge.rs:570-573` walks rows in file order with wrap-around; adjacent
batches are correlated (same function's paraphrase block) and every epoch
repeats the same order. Shuffle indices per epoch with a seeded RNG
(deterministic given `--seed`).

AC: two consecutive epochs visit different batch compositions; runs remain
reproducible under a fixed seed.

## P2.3 Device-side gradient clipping in latent stage

`src/tasks/latent.rs:928` uses host-synchronizing
`clip_accumulated_gradients`; knowledge training already uses
`clip_accumulated_gradients_device`. The sync stalls the GPU every step of
a 20k-step stage.

Change: switch latent to the device-side variant (same epsilon/norm
semantics); keep the scalar grad-norm logging on the existing `log_every`
cadence only (one sync per log, not per step).

AC: loss curves match the host version on a short A/B run; step time drops
measurably on GPU.

## P2.4 Observability: make each link of the knowledge chain debuggable

Goal: when eval fails, logs must identify WHICH link broke —
data → slots → conditioning → decoder-read — without rerunning anything.

Bridge stage (currently println-only, `bridge.rs`):
- Add `AsyncSummaryWriter` (same as knowledge stage). Scalars per log step:
  `loss/total`, `loss/positive_ce`, `loss/margin_zero`, `loss/margin_shuffle`,
  `loss/margin_hard` (separately, not just the sum), LR, grad norm.
- Per val step: `val/ce_matched`, `val/ce_zeroed`, `val/gap` (already
  printed; also to TB).
- **Gate telemetry**: mean and max sigmoid gate per injection site
  (`gate/site_{i}_mean`), every val step. Gates are computed in
  `GatedCrossAttention::forward` and discarded — expose via a lightweight
  stats hook (only when logging, to avoid per-step syncs).
- **Conditioning health**: mean L2 norm and per-dim std of adapter output
  (`cond/norm_mean`, `cond/std`), every val step.

Knowledge stage (`knowledge.rs`):
- `assoc/top1_acc`: in-batch retrieval accuracy (argmax over InfoNCE logits
  == diagonal), the interpretable twin of the association loss.
- `data/duplicate_fn_in_batch`: count of InfoNCE batches violating
  function-ID uniqueness (should be 0; nonzero = corpus interleave bug).
- Val twins of all train scalars once P0.3 lands.

Data/tokenizer sanity (one-time prints at stage start):
- Byte-fallback/OOV rate of fictional identifiers under the encoder vocab;
  warn if any function name exceeds 4 encoder tokens.
- Row counts per split and per row type (docs/knowledge/tasks) — extends
  the existing `TOFY_PRINT_SPLIT_STATS` output.

Eval (`--eval-bridge`, with P0.2):
- Per-task failure category in the JSON report: `compile_error`,
  `tests_failed`, `must_call_violation`, `timeout`, `pass`; aggregate
  counts per subset × condition. Store per-task generated code for the
  first N failures for inspection.

AC: a deliberately-broken run of each kind (shuffled conditioning file,
frozen gates, corrupt interleave) is identifiable from logs alone.

---

## Suggested landing order

1. P0.1 (grad accum) + P2.2 (shuffle) — same loop surgery, do together.
2. P0.3 (world val selection) + P1.1 (knowledge resume) — same file.
3. P1.2 (bridge resume) + P2.1 (full val pass) — same loop.
4. P1.3 (KV cache) then P0.2 (eval-bridge), since eval depends on fast
   generation.
5. P1.4 and P2.3 anytime (independent).

Each lands with `cargo check` + clippy clean and the listed ACs; nothing
here changes model architecture or data formats, so no retraining
invalidation — but P0.1 changes effective batch size, so bridge
hyperparameters (lr 2e-4) should be re-validated after it lands.
