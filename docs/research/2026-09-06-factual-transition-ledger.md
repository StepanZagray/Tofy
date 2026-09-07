# Factual transition ledger: later diagnostic registration

Date: 2026-09-06

## Claim boundary

This change implements no inference and runs no diagnostic. It adds a default-off,
question-independent, deterministic ledger of experienced visible actions and raw outcomes.
The ledger is not a trained goal solver. This registration makes no public-performance,
population, matched-compute, promotion, or ARC-AGI claim.

The later diagnostic is limited to whether enabling the ledger improves
same-current/opposite-history effect readout over unchanged raw history in the registered
non-ARC fixtures. Ledger facts may use only the visible before observation, the canonical
selected action, and the confirmed next visible observation. They may not use hidden state,
fixture identity or semantics, inferred success or progress, recommendations, external data,
public traces, game source, or assets.

The prerequisite raw-history thinking comparison at source `d85d837c` completed all 144
requests and failed its gate: thinking-off scored `14/24` and `12/24`, while thinking-on
scored `17/24` and `11/24` on seeds 0 and 1. Grounding and protocol/integrity checks passed.
This result fixes reasoning OFF for both ledger arms; it does not change the fresh fixture
seed or thresholds below.

## Frozen comparison

- Use a fresh diagnostic with fixed `layout_seed=2060908`.
- Compare ledger off against ledger on with reasoning OFF and model seeds `0` and `1`.
- Use the same raw history in both arms. Hold every other model, prompt, decoding, fixture,
  ordering, budget, and evaluator setting fixed within each seed.
- Keep paired counterfactual controls: the current observation is identical within a pair and
  only the experienced action/history differs.
- Before phase B, every grounding arm must reach at least `11/12`.
- On each model seed separately, ledger on must reach at least `21/24` and must gain at least
  `3/24` over ledger off.

Failure of any threshold rejects this bounded diagnostic claim. Passing every threshold only
supports the registered non-ARC readout claim; it does not authorize promotion or a public
performance claim. Any later inference run, threshold change, fixture change, or broader claim
requires a separate reviewed registration.
