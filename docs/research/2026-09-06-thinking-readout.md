# Controlled thinking readout

The first controlled readout passed grounding at 12/12 for both seeds, while
raw history scored 11/24 for both. Separately, the strict-schema thinking
protocol completed ten valid actions with nonempty reasoning. Before inference,
this experiment was narrowed from a reasoning-by-history design to raw history
only because summary message role order is a separate ambiguity. This isolates
whether enabled deliberation improves factual action-effect readout. It does not
test public gameplay, hidden goals, or weight updates.

Claim: on deterministic layouts from Python `Random(2060907)`, thinking mode
can meet the registered raw-history readout threshold and improve over a matched
fresh nonthinking arm. The intervention is `reasoning_mode` (`off`, `on`), with
model seeds 0 and 1. All other model, sampler, schema, representation, context,
and resource settings remain fixed: Qwen3-8B Q4_K_M, temperature 0.7, top-p
0.8, top-k 20, min-p 0, maximum output 1024, context 16384, compact lossless
current and historical frames, two raw historical groups, prompt/physical
batches 1024/1024, and 37-layer CUDA offload. Thinking uses the already
qualified 256-token reasoning budget.

Generate twelve grounding and twenty-four counterbalanced effect fixtures with
the existing visible/oracle boundary and controls, using `layout_seed=2060907`.
Hash visible and oracle fixtures together before inference. The 24 effect items
form twelve same-current, opposite-answer counterfactual pairs. Record exact
correctness by mode and seed, paired off/on discordances, prediction changes
between modes, and for each mode the both/one/neither-correct and
prediction-changed counts within counterfactual pairs. Report target-side and
unique-target-action population controls without inferring population-level
superiority.

Run all grounding cells first: 12 items for each reasoning mode and model seed,
48 completions. Use a fresh server for every stage/mode/seed and a fresh client
per item. Order modes `off,on` for seed 0 and `on,off` for seed 1. If any cell
scores below 11/12 exact centroids, stop B by design and report a completed
negative A.

If A passes, run the 24 raw-history items for every mode and seed, 96 additional
completions. Each off/on pair has the same current frame, instruction, raw
historical messages, question, model seed, sampler, and population. The only
intervention is reasoning mode. Nonthinking responses must have empty
reasoning; each thinking server must produce nonempty reasoning for at least
one response. All completions must use the strict action schema, end with
`finish_reason=stop`, and pass production context/token accounting.

The sole gate requires raw-thinking accuracy at least 21/24 on each model seed
and a gain of at least 3/24 over matched raw-off on each seed. Failure rejects
the intervention. Report every row, absolute count, paired discordance,
counterfactual-pair correctness and prediction-change counts, gate margins,
token count, latency, and gate result. These fixed synthetic counts do not
establish population superiority.

Thinking uses more actual decoding compute despite the equal maximum token
budget, so this is not a matched-compute comparison. The cap is 144 completions,
120 seconds per server startup, 60 seconds per decision, and 900 seconds total.
The expected duration is about nine minutes: roughly 72 thinking completions at
6.5 seconds and 72 nonthinking completions at 0.8 seconds, plus startup and
integrity checks.

Before inference, require clean reviewed pushed source with production core
bytes identical to qualified revision `92cce245`, verified sealed context and
thinking-protocol qualifications, exact model/server/config/helper/source/
fixture identities, no concurrent GPU compute, all-layer CUDA proof, peak VRAM
at most 95%, complete request/response capture, and exact owned-process cleanup.
Seal the final root and preserve its external manifest digest.

This completed local diagnostic can only admit a separately registered later
public development screen. It cannot promote a policy, trigger a public run, or
support broad reasoning claims. No public observations, game assets, game
source, solutions, or training data are used.
