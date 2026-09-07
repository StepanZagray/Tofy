# Controlled grounding and effect readout

This is a diagnostic proposal, not a public policy improvement. The observed
ar25 click loop does not distinguish parsing, effect attribution, or hidden-goal
inference. Test the first two with explicit synthetic instructions. No ARC
environment, public frame, game source or model update is used. Existing
connected-component centroids are part of the deployed representation; success
may be metadata readout and must not be called raw-pixel reasoning.

Fixed model/server and context-budgeted compact representation are those of
92cce245's registered repair. Same nonthinking sampler, max1024 tokens,16k
context,prompt/physical1024/1024,allCUDA, both model seeds0/1, no online training.
No concurrent GPU work. The diagnostic appends an explicit task instruction to
the normal system message, retains the normal strict action schema, and starts
fresh client history for every item. A fresh serving process starts each seed.
These instructions are diagnostic tasks, not hidden public-game goal discovery.

Use12 deterministic layouts from Python Random(2060906), generated once and
hashed before inference. Keep odd3x3 or5x5 rectangles inside64x64 with nonoverlap,
two distinct components and categorical nonzero palettes. Randomize positions
and palette identities independent of labels; prohibit targetcenter(31,31) so
the previously observed fixed click cannot pass. All fixture metadata and target
answers stay outside prompts. Available action isACTION6 only.

A:12 single-observation grounding items. Explicitly name the target component's
palette and ask to click its exact centroid. Other component is a distractor.
Both seeds run all12; pass requires>=11/12 exact clicks on EACH seed. Record
all predictions, target-membership accuracy as secondary, strict centroid as
primary. A fail stops the subsequent GPU diagnostic; analyze representation.

B:24 action-effect items (two counterbalanced histories for each layout).
Both final rectangles have the same palette, with exactly identical current
frames across each paired history. Initially one rectangle has a different
palette. Always execute left-component click, then right-component click; only
the initially different rectangle changes to final palette when clicked.
The paired history swaps which click changed pixels. Ask to repeat the prior
action that changed the visible board. This is factual effect attribution, not
causal generalization or hidden-goal inference. A last-frame-only deterministic
policy must give the same answer to both paired histories and cannot exceed50%
on that balanced pair. Verify this aliasing control before inference.

Compare raw factual history (the deployed compact full observation/action
groups) with an exact factual summary of those same two transitions: action
tuple and changed-pixel count only, followed by the same full current frame.
No labels such as correct/effective/successful or target answers enter summaries.
The summary is a representation of existing information, not extra simulator
knowledge. The intervention is historical representation only; current pixels,
question, model, seed, output budget and all other settings match. Both arms
run all24 items on both seeds, alternating arm order by layout parity.

Local summary gate:>=21/24 correct on EACH seed and>=3/24 paired gain over raw
history on EACH seed. If the raw arm already reaches>=21/24 on either seed,
this gate still requires gain; failure means summaries lack the registered
benefit, not that the model lacks attribution. No ceiling exception may be
invented after seeing results. Report all48 predictions per seed, discordant
pairs and absolute accuracies; no population or public confidence claim.
No seed/checkpoint/item subset selection. Max120 completions including A,
60 seconds/decision,120startup,900 total for the complete diagnostic. A protocol
or context failure is infrastructure failure, not an incorrect model answer.

Before inference: CPU fixtures/oracle scoring/aliasing controls, exact pinned
source/config/script/fixture hashes, clean reviewed pushed checkout and passing
repaired device qualification. Preserve all prompts/responses/costs and lifecycle,
exact PID cleanup and verified manifests. Treat results as completed local
diagnostic evidence only after all registered rows and integrity checks pass.

Decision: only a passed A and B permits a separately registered generic factual
summary policy intervention. It does not justify replacing all old frames,
claiming hidden-goal inference or broad public competence. Failure narrows the
next question; no bigger model, longer public budget or arbitrary planner is
automatically admitted. Keep all22 reserved public games untouched.
