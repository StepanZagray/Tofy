# Training-loop and infra fixes (archived)

Status: **completed / archived (July 2026)**. The P0–P2 items in this note
(true bridge microbatching, `--eval-bridge`, world validation selection,
knowledge/bridge resume sidecars, KV-cached generate, token-cache EOF safety,
shuffle, device-side latent clip) are implemented in the current tree.

Do not treat the historical line numbers or “still broken” wording below as
current. For live design see
[QWEN_KNOWLEDGE_INJECTION_SPEC.md](QWEN_KNOWLEDGE_INJECTION_SPEC.md) and
[RUNPOD.md](RUNPOD.md).

---

## Original finding list (historical)

The remainder of this file is retained only as a record of what was fixed.
It is not an actionable backlog.

### P0.1 Bridge gradient accumulation — fixed

True microbatching across distinct rows; `effective_batch = batch * grad_accum`.

### P0.2 `--eval-bridge` — fixed

Implemented in `src/tasks/eval.rs` / `src/tasks/bridge.rs` (ladder regimes,
Go compile/test, JSON report).

### P0.3 World-model validation selection — fixed

`src/tasks/knowledge.rs` selects on validation MSE + SIGReg.

### P1.x Resume / KV cache / token-cache EOF — fixed

Periodic sidecars and resume state for knowledge and bridge; cached generate
path; cached stream bails after two empty passes.

### P2.x Shuffle / clip / val coverage — fixed or largely fixed

Bridge epoch shuffle; device-side latent gradient clip; full validation loss
iteration for selection (first-batch telemetry may still be logged separately).
