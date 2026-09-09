# Finite action binding and the soft-coordinate seam

## Answer

The finite cardinal-bijection premise is verified, and the pinned initial C15 soft-coordinate cache preserves a positive external analytic margin. Learning the retrieval/complement relation remains an empirical C16 question; no learned-binder result is included here.

Before any C16 model invocation, the user requested width256 (1,580,804 parameters) and effective batch512. This supersedes the100,292parameter/batch64 implementation preparation. The abstract and cached examples, mathematical controls and scientific thresholds stay fixed; the1150-update schedule now presents588800 examples. The earlier configuration produced no learning outcome.

## Findings

### Verified finite premise

- Status: established.
- Confidence: high for the stated finite assumptions.
- Claim: Three distinct, nonblocked observations determine a bijection between four action IDs and the four cardinal unit effects. A finite polynomial readout recovers the desired action with unique score gap1.
- Source tier: local-primary; independent algebra and exhaustive analytic controls.
- Source: [C16 implementation contract](/home/stepan/Research/_runs/2026-09-09T182014Z-tofy-looped-action-binding/implementation-claim.md), [data implementation](/home/stepan/Research/_runs/2026-09-09T182014Z-tofy-looped-action-binding/data.py), [independent enumeration fixtures](/home/stepan/Research/_runs/2026-09-09T182014Z-tofy-looped-action-binding/data_tests.py).
- Source date or revision: September9,2026; preparation before any C16 model invocation.
- Applicability: Exactly four cardinal directions, a bijection, three distinct observed action IDs, truthful nonblocked effects, and a desired cardinal effect.
- Limits: Representability is not a convergence, generalization, recurrence-utility or ARC theorem. The formula is an external audit control and must not enter the learned forward.

Let A be the3×4 matrix of observed action one-hots and E the3×2 matrix of their cardinal effects. Let q∈R² be the desired cardinal effect. Because the observed IDs are distinct, O=AᵀE puts each observed effect into its action row and zero in the one missing row. The vector m=1−Aᵀ1 is exactly the one-hot indicator of that missing action. All four cardinal effects sum to zero, so the missing effect is −ΣᵢEᵢ. Therefore

F = AᵀE − m(ΣᵢEᵢ)ᵀ

is precisely the4×2 action-effect table. The four entries of Fq are1,0,0,−1: one matching direction, two orthogonal directions and its opposite. The correct action is the unique maximum, gap1. E is a matrix of **two-dimensional displacement vectors**, not direction one-hots; replacing its meaning would invalidate this derivation.

The assumptions matter. Two observed actions leave two unobserved effects exchangeable. Repeated observed IDs do not produce a single missing-action indicator. A nonbijective mapping permits arbitrary unobserved effects. Blocked transitions can hide the underlying movement. Even a zero-sum collection of arbitrary vectors need not have correct dot-product retrieval: effects(1,0),(2,0),(−1,0),(−2,0) with desired(1,0) give the larger score to(2,0). These are explicit fixture counterexamples, not claims about the registered population.

## Population, schedule and controls

- Status: established.
- Confidence: high for deterministic construction and analytic controls.
- Evidence: All2304 raw abstract rows pass independent enumeration and the gap1 control. Dataset checks reconstruct the exact ordering, features, labels, F32LE input hashes and U32LE label hashes.
- Source tier: local-primary.
- Source: [data.py, abstract_rows/schedule/validate](/home/stepan/Research/_runs/2026-09-09T182014Z-tofy-looped-action-binding/data.py), [data_tests.py](/home/stepan/Research/_runs/2026-09-09T182014Z-tofy-looped-action-binding/data_tests.py).
- Source date or revision: September9,2026.
- Applicability: Fixed16 fit maps and8 held-out maps,24 ordered action triples and4 desired directions.
- Limits: Exhausting this finite representation is not broad distributional coverage. No independence claim applies to the six support-order variants.

Rows are ordered action-triple-major, desired-direction-major, map-slot-last. The1536 fit and768 held-out rows comprise256 and128 semantic set-input orbits, each with six orderings. `orbit_id=map_slot*16+omitted_action*4+desired_direction`; `canonical` selects the ascending action triple, which is the first raw member. The artifact records every canonical index, member list and fit exposure total. A no-position-encoding model should give invariant predictions across the six variants; canonical metrics plus that explicit invariance check avoid counting copies as independent observations.

PCG64 seed20260922 independently shuffles each complete1536-row epoch.1150 batches of512 produce588800 presentations:383 full epochs plus512 rows of epoch384.1024 rows appear383 times;512 appear384 times. The U32LE schedule hash is `a46eac4347bfe16d55d220240452cb67643ccd007a97d2e56da16be45cb6f51e`. The256 semantic fit orbits receive2298–2303 presentations each, averaging2300. These are repeated cases, not fresh independent observations.

For each observed-action order and desired direction, either map list has balanced labels. A constant action and any deterministic model after **zeroing only the three observed effects** therefore score exactly25% overall. Public action IDs, query flag and desired direction stay intact. Zeroing only the desired query displacement instead makes all four desired-direction cases identical at fixed map/action triple, again with one label each. A missing-ID fallback can then score100% on omitted cases and0% on demonstrated cases while still scoring25% overall. Omitted-only query-zero success does not demonstrate query use. The learned forward may receive only the4×7 features; row/map/orbit IDs, controls and labels remain outside it.

## Cached visual inputs: derivation and verified bounds

- Status: established for data preparation; learned transfer is unknown.
- Confidence: high for these sealed1024 reused rows and the external analytic rule.
- Claim: Normalized attention expectations from the C15 initial checkpoint produce a sufficiently small perturbation of the abstract effects for the analytic gap to remain positive.
- Evidence: All1024 source rows were rehashed and their public input/query/metadata/label identities reconstructed; every cached analytic prediction is correct. Maximum coordinate error is4.679284703712483e−6, maximum endpoint bound8.990584075885266e−6, maximum F32 rounding error2.9800709455685137e−8. Minimum normalized true-role mass is0.9999987156308463. The conservative minimum analytic margin lower bound is0.9999238406209142.
- Source tier: local-primary.
- Source: [C15 initial retained rows](/home/stepan/Projects/code/.tofy-runs/looped-demonstration-grounding-20260909T191228-IST/frames-initial/evaluation-rows.jsonl), [C15 external seal](/home/stepan/Research/_runs/2026-09-09T173927Z-tofy-looped-demonstration-grounding/completed-campaign.manifest.json), [data.py, soft_features/coordinate_audit/load_visual](/home/stepan/Research/_runs/2026-09-09T182014Z-tofy-looped-action-binding/data.py).
- Source date or revision: C15 source `61e1335670239627772088a2fb6c81c2df499ed2`; seal SHA256 `6007293c5f51d263b0e3f40dcd819124efa8ffdaf2809202eda0c60142ffd40d`; initial-row SHA256 `91a08ec737c5b8b784086e730f637ec08afe60f84281ad9fc05380f723a3b25a`.
- Applicability: Initial frozen core plus its privileged role head, exact C15 reused training-query panel. Only this initial cache is packaged.
- Limits: No new visual evidence, learned-binder outcome, native-policy improvement, or training credit for an ability already present initially. The selected seal/row bindings are rechecked; this preparation does not re-audit every external profiler/source sibling in the parent campaign.

For a normalized distribution p over coordinates x∈{0,…,7}, and true coordinate x*,

|Σᵢpᵢxᵢ−x*| ≤ Σᵢ≠*pᵢ|xᵢ−x*| ≤ 7(1−p*).

The same bound holds separately for y. Opposite corners attain it. Before/after displacement error is bounded coordinatewise by the sum of their endpoint bounds; desired goal-minus-agent error obeys the same rule. Normalization is mandatory: divide the retained attention by its own sum before taking expectations. Float64 computes expectations and differences; only final4×7 model features are rounded to F32. The artifact separately records the measured rounding error.

Let a bound the largest coordinate error of any reconstructed action effect (including the sum of all three observed-effect errors for the omitted action), and b bound the desired-effect error. For cardinal true effects and desired direction, each perturbed dot-product score differs by at most a+b+2ab. Thus the correct-versus-rival margin is at least1−2(a+b+2ab). Per-row bounds include F32 rounding and a conservative1e−10 numerical allowance. This guarantees the **external analytic rule** on this cache; it supplies no Lipschitz or optimization guarantee for the learned transformer.

`soft_features` accepts only attention[7][2][64] and the three public action IDs. It has no access to cells, role positions, controls or labels. Public cells are consumed later by `coordinate_audit` and label/hash validation only. This separation is part of the schema/runner contract, not evidence of an end-to-end learned action policy.

## Verification and decision

Twelve data fixtures pass in1.345seconds, covering all2304 abstract cases, semantic orbits, both zero-input controls, corruption checks, assumption counterexamples, tight soft-coordinate bounds, the588800-presentation schedule and deterministic nested JSON key ordering. No C16 model invocation or fit occurred. The batch512 canonical JSON is5362577bytes, SHA256 `bc1413652b6bf38cadde28c8f1fd3d88d9cf781ab55413300bc155e341c47295`. Authorized in-memory regeneration and full comparison completed in1.967seconds (process1820038/session54997 exited0). Every abstract/cached row, feature, label, map, semantic orbit and provenance value matched the previous canonical dataset exactly; only update indices, schedule audit and fit-orbit exposure counts changed. Primary owns creation of the new `dataset-b512.json`; this task wrote no dataset artifact.

Earlier prelaunch serialization correction: the original byte checksum `22e89e6aa8e5c1479ecb09e513595d8da247fe6315248bdf784a01d9bb3989c6` was process-specific because cached audit dictionaries inherited set iteration order. Primary's preparation-only artifact instead hashed `8e35569ad5439844ad6c82c4596d42e54e3da0024ef5329e5e02ee386b60c6f7`. Read-only comparison established identical values in every root field; sorting keys produced batch64 canonical checksum `8c05315a3f0c312ae43afe37caf7988cffaae456d4b5a3b8a49dd3dd2174d048` from both objects. That serialization-only repair changed no numerical value or gate. Both older dataset files remain untouched and are obsolete implementation preparation after the separate user-requested width/batch change. The current data.py SHA is `2e62f1a455695ddb00a0457e568fc93b486f622809f06e662f1c56d78e18098e`; source-freeze records must bind this batch512 version.

The finite retrieval-plus-set-complement premise is verified, and the fixed soft-coordinate cache is justified as a reused evaluation seam. Whether the specified recurrent binder learns it, transfers across held-out mappings, tolerates the visual perturbation, or benefits from depth remains empirical. Primary registration and independent runtime analysis own those decisions. No automatic training or method promotion follows this preparation.

## Contradictions and gaps

The action effects are two-dimensional cardinal vectors; treating E as direction one-hots would change the problem and invalidate the proposed sign criticism. Six support orders are repeated semantic cases, not independent examples. The initial visual selectors had privileged role supervision, and the cached cases are reused; neither fact transfers learning credit to C16. The analytic continuity bound applies to the external polynomial control, not an untested learned transformer. The prelaunch serialization issue was byte ordering only and was corrected without changing numerical data; the preserved preparation artifact is not a model result.
