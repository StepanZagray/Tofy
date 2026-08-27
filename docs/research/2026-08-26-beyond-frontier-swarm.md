# Beyond-frontier improvement research — GPT-5.6 Sol xhigh swarm (2026-08-26)

Five GPT-5.6 Sol agents (reasoning effort xhigh, web search, full repo access) ran in
parallel on disjoint angles, briefed with the v5 recipe, live foundation-v2 telemetry
(51.4% changed-exact at step 9216, foreground reconstruction ~0.67), and an explicit
novelty bar: do not re-recommend anything already in ADR 0003/0004 or the 2026-08-24
research corpus. Raw reports: `raw-2026-08-26-swarm/` (sA objectives, sB planning,
sC in-episode adaptation, sD architecture, sE theory). Effect sizes in the raw reports
are preregistration priors, not measurements.

Claims below marked **[verified]** were independently checked against the code by the
orchestrating agent; everything else is agent-reported with file:line evidence in the
raw reports.

## 1. Code findings that matter for the CURRENT run

1. **Changed-pixel weighting is applied twice** (found independently by sA and sE).
   `split_weighted_ce` (`src/p2/train.rs:3459`) takes per-stratum means — which already
   normalize away class frequency — and then multiplies the changed mean by
   `changed_weight = (1-p)/p`. Aggregate changed:unchanged loss mass is `(1-p)/p : 1`
   (~24:1 at p=0.04) instead of the balanced 1:1 the copy-attractor theorem derived;
   the per-pixel effective ratio is `((1-p)/p)^2`. The copy-gate BCE directly below
   (`train.rs:3646`) uses the correct pooled construction. **[verified]**
   Candidate explanation for the foreground-reconstruction plateau (static pixels are
   gradient-starved). Falsifier: three matched 512–1024-step arms from one checkpoint
   (current / equal split means / gradient-budgeted). sE proposes `SplitWeighting.lean`.

2. **The headline metric is changed-only, raw-logits.** `one_step_changed_exact`
   (`src/p2/eval.rs:2260`) skips unchanged pixels and scores raw palette argmax — no
   copy-gate composition, no penalty for hallucinated edits to unchanged pixels.
   **[verified]** So 51.4% is neither full-frame exactness nor the deployed decoder's
   accuracy. All of sA/sD/sE ask for a full-frame exact metric + false-change rate,
   and raw-vs-composed persisted side by side.

3. **The 0.67 foreground plateau imposes NO hard cap on changed-exact** (sE). The two
   metrics use different pixel sets (`target != EMPTY` vs `current != target`); neither
   contains the other. The decisive offline diagnostic is sE's four-term decomposition
   on the changed set: B (target-latent decode exact), P (predicted-latent decode
   exact), Q (decode disagreement), G (gate-open rate); `A = P ∩ G`,
   `Pr(B)-Pr(Q)-Pr(¬G) ≤ Pr(A) ≤ Pr(B)+Pr(Q)`. Decision rule: low B → decoder binding;
   high B, big Q → predictor off-manifold; high P, low A → gate binding.

4. **EP clipping**: no separately clipped EP accumulation exists; the controller bounds
   EP's encoder contribution, then one combined clip at 1.0 — documented as ADR 0003's
   approximation at `train.rs:5976` **[verified]**, but sA notes EP can still rotate
   the shared clipped update. Second-tier concern.

5. **The ambiguity census cannot support the conclusions drawn from it** (sC, sE, sA).
   It groups by exact visible pixels (singletons report ceiling 1.0), covers only
   history-1/2, omits previous actions/outcomes from the key, and uses a top-left
   anchored rectangle while v5 translates boards (`src/p2/semantic_eval.rs:461,358`).
   `repeated_groups` must gate validity. Replacement: matched-counterfactual census —
   hold frame+action fixed, apply all operator families, report `q_sep` and Bayes
   ceilings for h ∈ {0,1,2,4,8,16}. CPU-only.

6. **Temporal streams never teach adaptation** (sC). The episode operator only affects
   ACTION5/ACTION6 (single call site `data.rs:1728` **[verified]**), and no stream
   contains an operator-revealing transition followed by another operator-dependent
   query in the same episode — so no recurrent state or adapter ever receives the
   meta-signal "use this fact to revise later predictions." Operator-bearing rows are
   ~20–23% of the mixture; on fully separating queries a history-free model is capped
   at 1/4 (four training families). Worst-case synthetic frozen-model ceiling ≈ 84%;
   the true value is `1 - 0.75·q_sep` with `q_sep` unmeasured (see item 5).

## 2. Where the five agents converge

- **The output seam before everything else.** sA, sD, sE independently rank the
  decoder/grounding/loss-normalization seam above capacity, EP, recurrence depth, or
  planning for changed-exact gains. sD refuses to convict the CNN grid before ~80–85%
  changed-exact with ≥95% target reconstruction.
- **Stop treating a sparse deterministic transition as 4,096 independent
  classifications.** The single strongest convergent design idea (sA #2/#4, sD #1/#2/#4,
  sE #6): predict the transition as an explicit edit object — cardinality × changed-set
  × values — composed over an exact copy of the current frame. Set-level exactness
  bound mirrors the existing pixel-CE Lean bound. sD adds the pointer/copy variant
  (K rigid transports that read source pixels; grid CopyNet) and the lossless-palette
  register variant appears in sA #4. All have zero-GPU oracle screens.
- **Planning must be selection-aware and shallow.** sB and sE independently: marginal
  calibration does not cover argmax over 4,102 roots (union bound makes ≥1 corrupted
  finalist near-certain); prefer ~32–64 transition-equivalents at horizon 2 with
  certificates (`margin > 2·D_H`), exact factual edges as zero-error, probes justified
  exactly when they flip a finalist's certificate. ADR 0004 Phase A exhaustive search
  should be a teacher/oracle configuration, not the deployed controller.
- **Exactness via memory is free and provable.** sB #4 (retrodiction-certified local
  edit cache, backed by `Locality.lean`) and sC #1 (canonical causal edit memory with
  D4/color canonicalization) are the same mechanism at two generality levels: zero
  training cost, exact-by-construction on matches, abstain on conflict.
- **Inverse-action consistency is not a trust certificate** (sE #5, echoed by sB's
  verification stance): an action-code model achieves zero inverse loss while wrong
  everywhere. ADR 0004's irreversible-edge trust gate should drop it as a sufficient
  condition.

## 3. Disagreements / tension

- **History conditioning**: sA #5 (mechanics-posterior GRU into base dynamics) vs sC
  (memory + fast-weight writer first, tiny in-context transformer only after one-fact
  causal gain is demonstrated) vs sE (measure `C_k - C_1` first; reject history if
  lower bound < 2pp). Resolution all three accept: run the corrected census first.
- **Equivariance**: sD gives the provable hypothesis-class bound (≤128× vs current
  stride-4) but demands the cheap 8-view ensemble test before building G-CNNs; sA's
  phase-preserving unshuffle (#6) is the lighter-weight cousin. Neither is tier-0.
- **Effect on live score vs changed-exact**: sB expects planning work to move live
  score, not changed-exact; sA/sD/sE expect objective/decoder work to move
  changed-exact. Not actually contradictory — different targets.

## 4. Recommended sequence (cheapest-decisive-first)

Tier 0 — offline only, no GPU training, run on existing checkpoints/data:
1. sE's B/P/Q/G decoder decomposition (rescoring; settles "decoder first?").
2. Joint-Bayes copy/color compositor rescore (sA #1; zero params, Bayes-optimal
   composition vs hard gate>0.5).
3. Matched-counterfactual ambiguity census, h=0..16 (settles history and the
   adaptation ceiling; validity-gated by repeated_groups).
4. Oracle screens for the edit grammar: count-oracle, K-flow displacement coverage,
   connected-component grammar coverage (sD #1/#2/#3).
5. D4 8-view + 16-phase ensemble probes (sD #5); depth sweep with semantic
   stabilization per outer step (sD #6, existing `model.rs:1720` path).
6. Add full-frame exact + false-change metrics to eval.

Tier 1 — short matched training arms (each ≤ ~2k steps):
7. Split-weighting A/B/C (item 1.1) — could feed the current run's lineage directly.
8. Frozen-backbone screens in oracle-screen-pass order: cardinality+set head →
   K-flow compositor → nonlinear decoder probe (sA #3; only if B was low in tier 0).
9. Canonical edit memory + orthogonal one-shot fast-weight writer on the frozen
   checkpoint (sC #1/#2) — requires adding operator-bearing temporal meta-episodes
   to the generator first (sC's data change; also needed by any future history model).

Tier 2 — planning phase (post-run, per ADR 0004 revision):
10. Factorized goal/rule EIG (sB #1), copy-gate predicate-support pruning (sB #2),
    interval dominance certificates (sB #3), retrodiction edit cache (sB #4).
11. Amortized discriminating probes; planner CEGIS repair loop (sB #5/#6).

Lean queue (sE): `SplitWeighting.lean`, `PlanningSelection.lean`, `HistoryCeiling.lean`
first; sE's audit table of proof-vs-ADR-claim overreach is in its §"Audit".

## 5. Report index

| Agent | File | Focus | Status |
|---|---|---|---|
| sA | `raw-2026-08-26-swarm/sA-next-constraints-objectives.md` | next constraints, objectives, representations | DONE_WITH_CONCERNS |
| sB | `raw-2026-08-26-swarm/sB-planning-hidden-goals.md` | latent planning, hidden goals, verification | DONE |
| sC | `raw-2026-08-26-swarm/sC-inepisode-adaptation.md` | in-episode learning, memory, fast weights | DONE_WITH_CONCERNS |
| sD | `raw-2026-08-26-swarm/sD-architecture.md` | architecture, edit grammars, equivariance | DONE_WITH_CONCERNS |
| sE | `raw-2026-08-26-swarm/sE-new-theory.md` | new theorems, Lean audit | DONE_WITH_CONCERNS |
