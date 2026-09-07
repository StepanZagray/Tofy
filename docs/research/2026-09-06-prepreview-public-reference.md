# Prepreview learned-agent development reference

Hypothesis: the pinned offline pretrained language agent can complete at least
one native ARC-AGI-3 development level on each of two fixed seeds within the
registered action/time budget. This is an empirical reference, not a promise
of success, an architecture ablation or the campaign's substantial-performance
target. The existing G reference completed0/23 at128 charged actions/game;
it lacked a trained task-value objective. The present agent changes model,
representation and decoding together. Compare absolute native outcomes and
costs, not attribution to an individual component or matched training compute.

Use only the previously inspected development population, in fixed order:
ar25-0c556536, bp35-0a0ad940, cd82-fb555c5d. Seeds0 and1 both run; no selection.
Use the exact three-game cache and official OFFLINE arc-agi0.9.9/arcengine0.9.3
engine via the common corrected runner. Full population is25 games/183 levels;
the other22 remain reserved for the first locked confirmation. Development
levels and their assets, trajectories, mechanics and solutions stay out of
pretraining. The agent receives only permitted factual observations/action
feedback, never game code, hidden state, identifiers or external game knowledge.

Prerequisite: complete and verify the repaired non-ARC resource/protocol
qualification before any engine action. Use its pinned model/server identities
and first resource-qualified context/batch configuration. Model is official
Qwen/Qwen3-8B-GGUF revision7c41481f57cb95916b40956ab2f0b139b296d974,
Q4_K_M SHA256d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785.
The May2025 artifact predates the public preview; this is chronology evidence,
not a complete original-corpus audit. Record exact clean reviewed pushed source,
server/model/script/configuration/registration hashes and cache file manifest.

Fresh model-serving process and cleared four-turn history for every game and
seed. Compact lossless observations preserve all animation layers and rows,
with generic connected-component geometry; no goal/object labels are supplied.
Nonthinking mode, max1024 output tokens, strict available-action JSON schema,
temperature0.7/top_p0.8/top_k20/min_p0, fixed seed. No optimizer updates, no
cross-game memory, no Python/tool execution, search, persistent rule memory,
or connected online effect CNN. The validated CNN prerequisite is independent
and cannot be credited for these actions. Nonthinking mode can limit reasoning.

Maximum128 charged actions/game and128/level,20 GAME_OVER retries. Initial
engine reset observation is reused; retry RESET consumes the charged budget.
No voluntary agent RESET; that limits recovery from nonterminal dead ends.
Maximum900 seconds/game,2700 seconds/suite,60 seconds/decision, two suites
at most5400 seconds total including integrity analysis. Before launch, estimate
the likely runtime from the actual qualification latency; if these bounds
cannot exercise the action budget, explicitly label the reference time-limited.
There is no silent fallback action or mid-run decoding/budget change.

Run an exact one-game/two-action engine smoke first using the dedicated one-game
cache. It is implementation smoke only; exclude its outcomes from the reference.
Then run both complete suites. Native scorecard counts must reconcile with
policy actions plus retries; forward terminal and final observations without
an extra decision. Preserve per-action observations, model outputs/token usage,
latency, native levels/scores and stop reasons. A strict format/timeout failure
is infrastructure/protocol evidence, not a completed zero-score model result.
Stop a failed suite for analysis rather than silently continuing or retrying it.

Report all six game/seed outcomes, total and per-level native scores, charged
actions/resets, per-decision/token/elapsed cost, model/GPU identity and inference
batch. No training effective batch/accumulation applies. Two sampled seeds are
a development screen, not a population confidence interval. Inspect full traces
for generic failure modes only; do not encode game-specific solutions.

Reference PASS requires at least one completed native level on each seed. It
does not satisfy the campaign target (>=92/183 levels and>=20/100 on each fixed
seed over all25 games). PASS admits a separately registered longer development
confirmation or controller intervention; FAIL requires behavior/model-capability
analysis before an intervention. Never choose the better seed or merely extend
an exhausted run. The autonomous campaign continues after either outcome.

Own exact runner/server/telemetry PIDs, with hard external suite watchdog and
no concurrent GPU training. Stop all telemetry/children before sealing. Verify
file sets/digests and source/cache/model identity, retain failed launch roots,
and record manifest digests outside each finalized root. No deployment or prize
submission is implied by this local reference.
