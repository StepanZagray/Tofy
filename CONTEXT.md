# Tofy World Modeling

Tofy learns reusable world dynamics for interactive ARC-AGI environments. This language separates factual experience from model predictions and names the representation properties required for action-faithful planning.

## Language

**Factual Action Branch**:
One confirmed transition in a group that shares the same current state but applies a different legal action or coordinate and records its factual outcome.
_Avoid_: Imagined branch, shuffled action

**Branch Group**:
The complete set of factual action branches collected from one current state.
_Avoid_: Batch, trajectory

**Board Effect**:
The task-relevant state change caused by an action, excluding deterministic status-display progression such as consumed action budget.
_Avoid_: Frame difference, pixel change

**Outcome Equivalence**:
The relation between factual action branches whose actions produce the same board effect from the same current state.
_Avoid_: No-op equivalence, action equality

**Action-Conditioned Transition**:
A predicted next world state whose outcome depends on both the current state and the selected action, including its coordinate when present.
_Avoid_: State prediction, frame forecast

**Spatial Action Field**:
An action representation that preserves where a coordinate action applies while retaining the action identity shared across the state.
_Avoid_: Coordinate bias, action token

**Consumer Latent**:
The learned state representation directly consumed by transition prediction, recurrence, and planning.
_Avoid_: Projection head, regularizer embedding

**Consumer Transition**:
The current, predicted-next, and factual target Consumer Latents for one Action-Conditioned Transition, including their predicted displacement.
_Avoid_: Latent tuple, transition tensors

**Representation Health**:
The conjunction of non-vanishing variation, adequate dimensional diversity, finite values, and retained information at every consumer seam.
_Avoid_: Non-collapse score, latent quality

**Changed Transition**:
A factual action branch with a non-empty board effect.
_Avoid_: Non-noop frame

**Raw Observation Identity**:
An immutable identity over every observed pixel and observation field, used for audit and corruption detection.
_Avoid_: State key, board hash

**Mechanics State Identity**:
A game- and level-scoped identity over task mechanics that may equate observations differing only in proven presentation or status progression.
_Avoid_: Raw hash, visual identity

**Factual Memory**:
The append-only history of confirmed observations, confirmed actions, and their observed results; predictions never enter it.
_Avoid_: Replay buffer, imagined memory

**Ambiguous Action Attempt**:
An at-most-once action submission whose application status cannot be confirmed from the transport result.
_Avoid_: Failed action, confirmed transition

**Hidden Rule**:
The episode-stable operator family, colour bindings, and goal family that determine transitions and never appear in any observation or conditioning input.
_Avoid_: Operator token, revealed rule

**Learning History**:
A chronological sequence of factual transitions from one hidden-rule episode produced by an exploratory policy whose competence improves along the sequence.
_Avoid_: Expert trajectory, demonstration

**Context Window**:
The bounded set of most recent factual transitions from the same episode that is supplied to the model as evidence about the hidden rule.
_Avoid_: History buffer, memory tokens

**Twin Episode**:
A generated episode that shares a byte-identical initial frame with another episode but is bound to a different hidden rule.
_Avoid_: Augmented copy, shuffled control

**Fast Weights**:
The named parameter subset that test-time adaptation may update; every other parameter is frozen at inference.
_Avoid_: Adapter, LoRA

**Whole-Frame Content**:
The v6 convention that all 4096 pixels of a frame are board content with no reserved status row or unsupervised padding.
_Avoid_: Content rectangle, playfield
