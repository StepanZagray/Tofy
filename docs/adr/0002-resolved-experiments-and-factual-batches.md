# ADR 0002: Resolved experiments and complete factual batches

Status: Accepted (2026-08-12)

## Context

World-core identity and compatibility are currently expressed as independent booleans across the CLI, persisted training configuration, model construction, resume contract, report schema, and evaluation. This admits invalid states, including a V2/V3 run that allocates the rejected loss-only SIGReg projector. Factual Branch Groups are also flattened into generic rows, truncated to a physical batch, reconstructed from adjacency, and activated from the first row's family. A partial group can therefore be treated as complete.

ADR 0001 deliberately made every V2 intervention independently switchable for its causal campaign. Those switches have now served that experiment; the V2/V3 treatments were rejected by the frozen gates, and keeping the switch bag as the source of new experiment semantics creates more invalid combinations than useful arms.

## Decision

For new training, resolve every CLI/config request through one typed experiment definition before model construction. The resolved definition owns:

- world-core family and report/checkpoint schema;
- action conditioning topology;
- regularizer statistic and population;
- whether factual Branch Group learning is required;
- compatibility validation and persisted experiment identity.

The CLI remains an adapter. Legacy V2/V3 flags may be read only to construct the typed request and to evaluate historical checkpoints; they are not independent semantics after resolution. V2/V3 categorically reject a loss-only projector.

Factual training operates on complete Branch Groups through a Factual Batch. The batch owns stable group identity, deterministic group ranges, Board Effects, and recoverability. Tensor construction is an adapter over its rows. Physical batch selection admits whole groups only and never silently truncates one.

Training and evaluation use a named Consumer Transition as their shared test surface. The board-probe module owns spatial-row layout and changed-patch derivation.

## Consequences

ADR 0001 is superseded for new training but retained as the rationale and compatibility record for legacy V2/V3 checkpoints. New experiment arms are explicit typed variants rather than arbitrary boolean products. Some formerly parseable combinations now fail before parameters or output directories are created.

The resolved experiment identity becomes part of exact resume comparison and reports. Changing its topology, statistic, population, or factual-learning contract is trajectory-changing and cannot exact-resume.

Factual batches may contain fewer branch rows than a requested physical row limit because completeness takes priority. Experiment reports record both Branch Group and branch-row populations.

This decision improves interface depth, locality, and causal validity. It does not claim that any new predictive-state treatment improves ARC-AGI-3 performance.
