# C18 action-equivariant binding: finite selection passes, with structural limits

**Completed September 9, 2026, machine-local IST. Both arms have accepted
integrity and independent scoring.** The equivariant treatment passes its
registered single-seed component gate; the matched legacy binder fails its
held-map thresholds. Native policy and ARC records are unchanged.

Both models fit **256/256 canonical cases**. The treatment scores **128/128
held-map cases**, including **96/96 demonstrated and 32/32 omitted**; legacy
scores **81/128**, including **76/96 demonstrated and 5/32 omitted**. The
treatment's held-map success is coverage of **16 action-renaming orbit types**:
under the imposed symmetry, correct fitting already entails the renamed cases.
It is not independent semantic generalization or discovery of a general algorithm.
Both models also score **1024/1024 on the reused, privileged C15 visual cache**,
without executing or training the vision core.

The treatment chooses the omitted action on every effects-zero and query-zero
case: **0% demonstrated, 100% omitted and 25% overall**. Overall 25% is forced
by balanced labels on identical clamped inputs. Query-zero CE is **19.919956**,
including **26.423501 on demonstrated cases**, so the strong factual result
does not imply calibrated behavior when required information is removed.

The [legacy analysis](/home/stepan/Projects/code/.tofy-runs/looped-action-equivariant-20260909T212250-IST/legacy-analysis.json),
[treatment analysis](/home/stepan/Projects/code/.tofy-runs/looped-action-equivariant-20260909T212250-IST/equivariant-analysis.json)
and [paired comparison](/home/stepan/Projects/code/.tofy-runs/looped-action-equivariant-20260909T212250-IST/comparison.json)
retain all endpoints. The [registration](artifacts/c18-equivariant-binding/registration.md)
and [finite proof](/home/stepan/Research/_runs/2026-09-09T194505Z-tofy-looped-action-equivariance/findings/symmetry.md)
were frozen before these outcomes. This is a component selection result, not
method promotion or evidence for an LLM, native controller, planner or ARC solver.

## Historical motivation and matched comparison

[C17](2026-09-09-looped-binding-screen-results.md), at source
`9d4827d23c50aa12d8d49b36705e67dbea480371`, fitted all 256 canonical cases,
including 64/64 omitted-action cases. Its held-map result was 71/128 overall,
69/96 demonstrated and 2/32 omitted. This separates successful fitting from
failed transfer, especially for absent actions. C17 also scored 1024/1024 on
the reused C15 visual cache; that did not establish new visual learning or
held-map visual control.

C17's support-order invariance did not enforce action-ID renaming equivariance.
Its input projection can treat action one-hot channels differently, and its
CLS head has four independently parameterized output rows. C18 retains that
binder as a new, matched comparator. Historical C17 is context: both C18 arms
use the same selected batch and resulting optimizer-step count.

## Task, model and inference boundary

Each example supplies three distinct public action IDs, their observed
two-dimensional displacements, and a requested cardinal direction. A mapping
is a bijection from four action IDs to up, down, left and right. The label is
the action producing the requested direction. Three quarters of cases request
a demonstrated action; the remainder request the omitted action.

The public input remains F32 `[4,7]`: three support records containing
displacement, four action indicators and a query flag, followed by the query
record. Labels are used for cross-entropy and scoring. Map IDs, correct labels,
role truth and grouping fields do not enter learned inference.

| Component | Legacy comparator | Equivariant treatment |
|---|---|---|
| Width / attention heads | 256 / 4 | 256 / 4 |
| Recurrent body | Two shared transformer blocks | The same block architecture |
| Training depth | Four loops | Four loops |
| Tokens processed | Four projected records plus learned CLS | Four action tokens |
| Readout | CLS to four action scores | One shared scalar head applied to each action token |
| Parameters | 1,580,804 | 1,579,265 |

Both models start recurrent state at zero and reinject the projected input on
every loop. Each four-loop forward executes both shared blocks four times;
weights are reused across iterations. Shared block initialization matches
exactly between arms, as verified by the recorded parameter digests. Projection
and head shapes differ, so full initial parameter files differ. This changes
the input representation and output sharing together. The
[binder implementation](../../src/p2/looped_agent/binding.rs) contains both
architectures at source `ec6ba27c05fa1032a82362a05dbd15fe51cc043a`.

For the treatment, let `A` be the three-by-four public action-indicator matrix,
`E` the three-by-two observed displacement matrix, and `q` the requested
two-dimensional displacement. The four-by-five action-token matrix is

```text
X = [AᵀE, Aᵀ1, 1qᵀ].
```

Each action token contains its observed effect, whether it was observed, and
the broadcast query. The omitted action retains zero observed effect and a
zero observation mask. The projection, transformer operations and scalar head
share weights across action slots, without positional/action embeddings or
slot-specific readout parameters.

`AᵀE` supplies the association between an observed public action ID and its
effect. This is an explicit representation prior; differentiability alone
does not make that association learned. The forward path does not analytically
complete the missing effect or run a direction matcher. Selection must still
be learned from policy cross-entropy. A valid solution can match an observed
effect and otherwise choose the omitted slot, without constructing an explicit
missing displacement internally.

No pretrained LLM, LLM controller or learned vision execution is used in this
experiment. The cached diagnostic reuses C15 initial role-attention coordinates
derived from privileged selectors and familiar cases. Those selectors are
inherited, not trained in C18. Rewards, values, dynamics, planning, rollouts and
persistent episode memory are outside this component screen.

## Why the finite task has 16 action-renaming orbits

An orbit is a collection of cases related solely by renaming action IDs.
For an action permutation matrix `P`, the public indicator matrix becomes
`A' = APᵀ`, giving `X' = PX`. Shared row operations and full self-attention
commute with that permutation. Zero initialization and shared input reinjection
preserve the property at each recurrent step. Thus the treatment satisfies

```text
L(PX) = P L(X)
```

in exact arithmetic at every allowed depth. This is a finite architectural
property, not a theorem that optimization will find correct parameters.

Let `g` and `h` map actions to physical directions. Renaming actions by
`π = h⁻¹∘g` changes `g` into `h`, while preserving both the omitted physical
direction and requested physical direction. Conversely, cases with the same
pair of physical directions are related by such a renaming. There are exactly
`4 × 4 = 16` orbits: four omitted-action types and 12 demonstrated-action types.
Action renaming does not rotate or rename the physical directions themselves.

The complete abstract population contains 384 cases after removing support
order: 24 mappings times four omitted actions times four queries. Each of the
16 action-renaming orbits contains 24 map arrangements, of which 16 are in the
fit split and eight in the held split. These are different from the 256/128
canonical support-order cases reported for historical continuity.

Strict-margin correctness on one representative of each of the 16 orbits,
together with exact equivariance, implies correctness on all 384 cases and
all 2,304 ordered rows. In particular, full fit correctness implies full held
correctness. First-index argmax does not commute with permutations on ties:
if the true token shares a maximum with `r ≥ 2` tokens, it is selected in only
`24/r ≤ 12` map arrangements, insufficient for all 16 fit arrangements to be
correct. The numerical experiment checks finite tolerances and margins directly.

Consequently, even a perfect held-map result would be redundant symmetry
coverage. A lookup over the 16 types could solve this task. Neither convergence,
useful recurrent depth nor broader algorithm discovery follows from the proof.

## Exact paired data, batch and optimizer rule

Fit map IDs are `[0,2,4,5,7,8,9,10,13,14,15,16,18,19,21,23]`; held IDs are
`[1,3,6,11,12,17,20,22]`, using lexicographically ordered permutations of four
directions. All 24 ordered triples of distinct observed actions and four query
directions give 1,536 fit and 768 held rows. Six support orderings describe each
canonical case. The 1,024 cached visual rows are unchanged.

Both arms consume the exact same **588,800 ordered presentations** as C17.
PCG64 seed 20260922 supplies 383 complete shuffled epochs of 1,536 rows, then
512 rows from the next epoch. Of the ordered fit rows, 1,024 receive 383
presentations and 512 receive 384. Repetition creates no additional distinct
semantic cases.

The fixed flat index stream is regrouped into the selected effective batch
`B`, retaining the actual final tail. There is no padding, dropping or extra
duplication. The selected **physical/effective batch is 4,096, accumulation 1**.
Each arm executes **144 updates: 143 full batches plus 3,072 rows in the final
update**, exactly 588,800 presentations. Both use identical batch sizes, tail,
order and update count. This changes optimizer history relative to C17's
1,150 updates; the two C18 arms are the matched comparison.

The registered initializer uses seed 0. AdamW uses learning rate 0.0003,
betas 0.9/0.999, epsilon `1e-8`, weight decay 0.01 and global L2 clip 1.
Policy CE is averaged over actual rows, including the tail. There is no warm-up,
learning-rate schedule or automatic learning-rate scaling.

The registered candidate ladder was 512, 1024, 2048, 4096, 8192, 16384 and
32768. Both architectures completed disposable two- and five-update trials at
512, 1024, 2048 and 4096, with initialization restored. The legacy two-update
trial at 8192 failed with `CUDA_ERROR_OUT_OF_MEMORY`, stopping enlargement.
The equivariant 8192 trial and larger candidates were not run. Thus 4096 is
the largest tested stable joint power-of-two batch under this contract, not
a universal hardware or treatment-only maximum. The accepted
[capacity selection](/home/stepan/Projects/code/.tofy-runs/looped-action-equivariant-20260909T212250-IST/capacity-selection.json)
retains the failed root separately from learned-model evidence.

## Four-loop gates and all 15 streams per arm

The treatment passes `supported_single_seed_action_equivariant_selection`;
legacy remains `registered_binding_screen_not_supported`. Both have accepted
integrity. All checkpoints are terminal or initial checkpoints prescribed in
advance; no best seed, checkpoint or inference depth was selected.

| Registered treatment requirement | Observed |
|---|---|
| Initial action equivariance | Pass; maximum absolute logit error 1.6689301e-6, maximum tolerance ratio 0.0153789 |
| Terminal action equivariance | Pass; maximum absolute logit error 1.3828278e-5, maximum tolerance ratio 0.0795661 |
| Eligible renamed winners agree | All 16 initial and 16 terminal orbit types agree |
| All 16 types have true margin ≥0.001 across all 24 maps | Pass; minimum terminal margin 16.7352557182 |
| Terminal support-order winners agree | All 256 fit and 128 held canonical cases |
| Effects-zero/query-zero integrity | All four streams valid; balanced identical-input groups and exact 25% overall |

Alignment uses physical directions only in scoring and compares each map with
fixed map 0. Numerical tolerance is `1e-4 + 1e-5 × abs(reference)`. Winner
comparisons require positive minimum winning margin greater than twice the
maximum measured logit discrepancy. Treatment is numerically equivariant at
initialization, when only four of the 16 types have a strict correct margin;
all 16 are learned by the terminal checkpoint. Initial symmetry alone was not
a learning result.

Legacy's maximum renaming discrepancy grows from 2.868513 to 21.444884. It has
three terminal orbit types with all-member true margin ≥0.001, and fails all
three held accuracy thresholds despite perfect fitting. Its reported zero
inconsistent eligible winner orbits must not be mistaken for action symmetry:
**zero legacy orbits satisfy the winner-comparison eligibility margin bound**,
and its numerical equivariance checks fail.

The tables include every registered stream. CE is mean cross-entropy scored
from retained F32 logits using F64 arithmetic; lower is better. Counts are exact,
CE is rounded. Raw rows include the six support orderings; canonical rows use
one ascending-action representative. Cached rows use all 1,024 retained cases
for both summaries, without invented support-order variants.

### Legacy comparator

| Stream | Canonical correct | Raw correct | CE | Demonstrated correct | Omitted correct |
|---|---:|---:|---:|---:|---:|
| `initial-fit-l4` | 58/256 | 348/1536 | 1.58663101 | 53/192 | 5/64 |
| `final-fit-l1` | 231/256 | 1386/1536 | 0.285603668 | 189/192 | 42/64 |
| `final-fit-l2` | 256/256 | 1536/1536 | 0.00581885425 | 192/192 | 64/64 |
| `final-fit-l4` | 256/256 | 1536/1536 | 3.93393771e-05 | 192/192 | 64/64 |
| `final-fit-l8` | 256/256 | 1536/1536 | 0.0011417984 | 192/192 | 64/64 |
| `final-fit-effects-zero` | 64/256 | 384/1536 | 8.24492974 | 40/192 | 24/64 |
| `final-fit-query-zero` | 64/256 | 384/1536 | 8.07563903 | 51/192 | 13/64 |
| `initial-heldout-l4` | 31/128 | 186/768 | 1.59067732 | 24/96 | 7/32 |
| `final-heldout-l1` | 83/128 | 498/768 | 1.86932001 | 77/96 | 6/32 |
| `final-heldout-l2` | 81/128 | 486/768 | 2.3039351 | 78/96 | 3/32 |
| `final-heldout-l4` | 81/128 | 486/768 | 2.57308553 | 76/96 | 5/32 |
| `final-heldout-l8` | 75/128 | 450/768 | 2.74606451 | 69/96 | 6/32 |
| `final-heldout-effects-zero` | 32/128 | 192/768 | 8.24492955 | 20/96 | 12/32 |
| `final-heldout-query-zero` | 32/128 | 192/768 | 7.76535654 | 26/96 | 6/32 |
| `final-cached-visual-l4` | 1024/1024 | 1024/1024 | 3.88643595e-05 | 768/768 | 256/256 |

### Equivariant treatment

| Stream | Canonical correct | Raw correct | CE | Demonstrated correct | Omitted correct |
|---|---:|---:|---:|---:|---:|
| `initial-fit-l4` | 64/256 | 384/1536 | 1.37607665 | 64/192 | 0/64 |
| `final-fit-l1` | 256/256 | 1536/1536 | 0.00110486091 | 192/192 | 64/64 |
| `final-fit-l2` | 256/256 | 1536/1536 | 4.99197347e-06 | 192/192 | 64/64 |
| `final-fit-l4` | 256/256 | 1536/1536 | 1.64135086e-08 | 192/192 | 64/64 |
| `final-fit-l8` | 256/256 | 1536/1536 | 5.84794626e-11 | 192/192 | 64/64 |
| `final-fit-effects-zero` | 64/256 | 384/1536 | 4.88237207 | 0/192 | 64/64 |
| `final-fit-query-zero` | 64/256 | 384/1536 | 19.9199557 | 0/192 | 64/64 |
| `initial-heldout-l4` | 32/128 | 192/768 | 1.37607666 | 32/96 | 0/32 |
| `final-heldout-l1` | 128/128 | 768/768 | 0.0011048613 | 96/96 | 32/32 |
| `final-heldout-l2` | 128/128 | 768/768 | 4.99197939e-06 | 96/96 | 32/32 |
| `final-heldout-l4` | 128/128 | 768/768 | 1.64135303e-08 | 96/96 | 32/32 |
| `final-heldout-l8` | 128/128 | 768/768 | 5.84795268e-11 | 96/96 | 32/32 |
| `final-heldout-effects-zero` | 32/128 | 192/768 | 4.88237207 | 0/96 | 32/32 |
| `final-heldout-query-zero` | 32/128 | 192/768 | 19.9199557 | 0/96 | 32/32 |
| `final-cached-visual-l4` | 1024/1024 | 1024/1024 | 1.52485928e-08 | 768/768 | 256/256 |

### Paired contrasts, negative evidence and depth limits

On terminal four-loop held cases, treatment gains **47/128 cases /36.71875
percentage points** over legacy, with CE lower by **2.573085514**. The subset
gains are 20/96 demonstrated and 27/32 omitted. Both arms fit 256/256 and score
1024/1024 on the reused cache, so their accuracy difference there is zero.
These finite matched differences support the representation/output-sharing
package on this task; action-renamed held cases remain structural coverage.

Legacy improves held accuracy from 31/128 to 81/128, gaining 61 formerly wrong
cases while losing 11 formerly correct ones. Yet held CE worsens from
**1.590677 to 2.573086**. Demonstrated CE improves **1.562717 → 0.847399**,
while omitted CE worsens **1.674560 → 7.750146** and omitted accuracy falls
**7/32 → 5/32**. Its minimum held true-label margin is **−19.356327**.
This is worse wrong-answer confidence within that subset, not proof that its
representation contains no usable information.

Treatment improves from 64/256 fit and 32/128 held to complete accuracy, with
no initially correct case lost. Factual-minus-either-clamp accuracy is +75
percentage points in both cohorts; within the omitted subset it is **zero**,
because the clamped model already always chooses the missing action. The
matched legacy factual-minus-clamp held gain is +38.28125 percentage points,
but its omitted factual score is below both clamped omitted scores. Full paired
case counts, CE and margin differences are in each arm's `contrasts` records.

Both treatment clamps choose the omitted action on every row. The 25% overall
score follows from balanced identical inputs, and the omitted-only 100% is
therefore a missing-ID shortcut. Effects-zero CE is **4.882372**, split into
**6.465062 demonstrated /0.134303 omitted**. Query-zero CE is **19.919956**,
split into **26.423501 demonstrated /0.409319 omitted**. This is substantially
worse query-zero CE than legacy's **8.075639 fit /7.765357 held**, despite equal
overall accuracies. No ablation score is promoted as learned inference.

The terminal treatment already scores every factual case correctly at **one
loop**. Extra evaluation loops reduce CE, but these results do not establish
that recurrence was necessary, that deeper training would help, or that the
model discovered a general iterative algorithm. Legacy held counts at depths
1/2/4/8 are **83/81/81/75**; alternative depth never selects the reported model.
Both models were trained only at four shared loops.

Every abstract stream has exact winner agreement under all six support
orderings. Treatment ordering logits are numerically identical in these retained
streams (maximum deviation 0); this is distinct from its small nonzero
**action-renaming** discrepancies. Legacy's largest pairwise support-order
logit deviation across the registered streams is **2.4318695e-5**.
Final canonical prediction histograms are `[64,64,64,64]` on fit for both arms;
held histograms are `[27,36,29,36]` for legacy and `[32,32,32,32]` for treatment.
All per-map counts and remaining histograms are retained in the analyses.

### Cached adapter and external controls

Both frozen terminal binders score **1024/1024**, including **768/768
demonstrated and 256/256 omitted**, on cached C15 initial-selector features.
Legacy CE is **3.8864360e-5**, treatment CE **1.5248593e-8**; respective minimum
true margins are **7.599625** and **16.735213**. Both predict `[256,256,256,256]`.
The cache uses familiar mappings and reused visual cases, with nearly one-hot
privileged initial role attention. No fresh images, vision forward pass or
jointly trained image-to-action system is evaluated here. Existing selector
capability cannot be credited to either C18 fit.

The separate analytic control is correct on **256/256 fit, 128/128 held and
1024/1024 cached**, with minimum true-score margin 1 on abstract inputs and
0.999995992 on the cache. Constant action 0 scores **64/256, 32/128 and
256/1024**. Always choosing the omitted ID has those same overall counts,
with 100% omitted and 0% demonstrated accuracy. The analytic solver remains
absent from learned forward execution.

## Runtime, capacity and profiling

Both fits consumed all **588,800 presentations** at physical/effective 4096,
accumulation 1, ending at update 144 with a 3,072-row tail. Legacy ran first,
then treatment. The initial shared-block parameter digest is identical:
`14f4228a2b708e72cd733e6ecd0393f863e7ebdc9ec613f7f51d51310f31c402`.
The legacy full initializer matches its registered C17 identity; complete
initial/final hashes are recorded below. All frozen evaluations preserve their
selected checkpoint identities.

| Runtime field | Legacy | Equivariant |
|---|---:|---:|
| Empirical admission forecast | 101.060607 s | 87.805209 s |
| Model-reported lifetime | 74.687315 s | 62.137947 s |
| Reported update loop | 73.830382 s | 61.276547 s |
| Externally measured model phase | 75.132051 s | 62.660544 s |
| External finalization | 0.580150 s | 0.574757 s |
| Maximum sampled training memory | 7,102 MiB | 5,758 MiB |
| Minimum sampled training reserve | 1,049 MiB | 2,393 MiB |
| Maximum sampled training temperature | 67°C | 68°C |

The admission rule uses component maxima from disposable two- and five-update
trials and forecasts `1.25 × (141u + 3c + s + h) + k`; the unchanged model cap
is 600 seconds per arm. It is an empirical screen, not an upper-bound theorem.
Training captures are updates **2, 72 and 144**. The lower observed treatment
time is an instrumented one-run cost, not production throughput superiority.

The [execution record](/home/stepan/Projects/code/.tofy-runs/looped-action-equivariant-20260909T212250-IST/execution.json) reports **536.166003 s**
of operator execution, including capacity trials, both fits and evaluations.
The outer supervisor records 536.315138 s for that operation.
The separately supervised analysis stage takes **15.928933 s**. Build time is
**25.535856 s**. These stage durations exclude pauses between stages; the seal
was reverified at **21:32:53 IST on September 9, 2026**.

The [failed legacy 8192 trial](/home/stepan/Projects/code/.tofy-runs/looped-action-equivariant-20260909T212250-IST/legacy-smoke-b8192.exit.json) is
preserved with return code 1 and an explicit CUDA out-of-memory message in its
[log](/home/stepan/Projects/code/.tofy-runs/looped-action-equivariant-20260909T212250-IST/legacy-smoke-b8192.stdout.log). It spent **5.177419 s** in the
supervised stage and is excluded from learned-model evidence. Its sampled
memory was only 398 MiB: failed allocation requests and unsampled peaks are not
measured by that telemetry. Joint capacity selection does not establish the
equivariant model's separate maximum.

The device is the local RTX 5060 Laptop GPU with 8,151 MiB reported memory;
execution uses CUDA F32 with TF32 override 0. Sampled memory is not allocator
peak. Software records Python 3.14.7, NumPy 2.4.2/OpenBLAS, Rust/Cargo 1.95.0
and Nsight Systems CLI 2026.4.1; exact tool and binary files are hash-bound.

The seal verifies **54 healthy CUDA bundles**: two four-row qualification
captures, 16 successful disposable trial captures, six training captures and
30 frozen evaluation captures. Host/NVTX, available CandleGraph tensor/scalar/
gradient records and bound Nsight evidence are present. Operation/activation
linkage, allocation lifetimes, instrumented physical-memory checkpoints and
device-event intervals remain unwired; automatic GPU correlation is incomplete.
`profiled_work` captures do not support a production timing verdict.

## Independent reconstruction and exact identities

Both independent reviews accept all 15 streams, reconstructing the finite
population, complete schedule/tail, cached adapter, actual post-clamp F32
hashes, scalar CE/margins, counts, paired contrasts, controls and action orbits.
Each agrees on **3,195 floating fields and 5,309 discrete fields**. Maximum
absolute floating differences are **1.7763568394002505e-15 legacy** and
**7.105427357601002e-15 treatment**. Independent code imports neither primary
metrics nor the producer and performs no fitting or model inference.
Source/build/initialization/training/gradient/device/profile/checkpoint/cleanup
checks share each pinned runtime receipt; this is independent numerical
reconstruction, not a second training run or independently observed CUDA execution.

| Common identity | Revision or SHA-256 |
|---|---|
| Tofy source | `ec6ba27c05fa1032a82362a05dbd15fe51cc043a` |
| candle_graph | `1ea5cc5c5998a1c2bd388eb158653dd0d1b7452a` |
| Launch binary | `843a44f227dc066ea7d3471d605acc099d2c6bb6be6ee21f2940e78d66b1d9f0` |
| Registration | `3e9d759c154e097d93372300a5fb252c3e3ffa2494243d63e45bdf32c64191f9` |
| Source C17 dataset | `bc1413652b6bf38cadde28c8f1fd3d88d9cf781ab55413300bc155e341c47295` |
| Regrouped B4096 dataset | `b08bb964dcd2260ebea5fefde15cece5d6264ed1d540e13ca1573864f17f95f2` |
| Flat U32LE schedule | `a46eac4347bfe16d55d220240452cb67643ccd007a97d2e56da16be45cb6f51e` |
| Launch specification | `86b3e47c5680633ec49739c2a679bb129e5ecef22ae7247a943225165b82c6aa` |
| Pair comparison | `f0794eb53da402f033830dd8a47d65fd95fd4c4778f78cf379b180a16b72faac` |
| Outer campaign manifest | `6535a3283029f2d5ad211ad70030fb3e2ad21b1dca1530e985406a94465c7b4c` |

| Per-arm identity | Legacy | Equivariant |
|---|---|---|
| Initial checkpoint | `1f59081cc8dd74b0b6d1ddc59e33a80fe4a98fe3c2640c74e4a3b6eecd0f1a6e` | `69338ea5dc20e0e09a4ccaae2c32bc84fb11863bbb00d06f954977c9b77e0fc3` |
| Final checkpoint | `ae556667d6b2e0daf3c00f93d04a6ba36738e6e9301f23c902a80bfdc0f672ca` | `d2deeba7b0fafc2386c9a19d39bc99b7534b91a0155d66e77792a5c4037bd717` |
| Initial parameter digest | `55d6d89a7e8a22049d074ae828cabb2f114093ec88886df67744a453c0cd364e` | `40722ba2215083d3b660eaeeb53870597e2378cc959446539ae841fdde3ff53f` |
| Final parameter digest | `f1999e849ef92e9954c7f44ad20f0b5652c2ac1309ebb412b8df4d8398c2a3eb` | `86dd13d3998598d54acc1d97167c75d09d1d00f12b29b9073154d19a29a9e491` |
| Analysis config | `c499702cd5ffe0d4934a266d09252b283d3498b6ace656b7c2d96a1bcb775e9f` | `31b6ce8ee6893fefae202e801553bcedb3b7a0a7f37f8cd57f5457062e454dc7` |
| Numerical analysis | `175178dd638d2d3491ef99c1923ba6eb8bd8364b4a8d8462516531918b93dbef` | `a0678019444f75e116eb63f8db994acb339b7bc6e6d4ecf845eb9191248b1036` |
| Independent review | `4ecb979c4d35f4e4bc75730f2f9bb760de77784abdb03e5f52db6bb845421fbf` | `85635581134617e97061a413a9ed64f00ee0f2440332997d2820c39fc07ba12a` |
| Integrity receipt | `d2f19e301ecae442433489c254dbbab5f147d829f37e0800b0c09415718b6abb` | `167a40f55b52a06aff66826612336104924147c492a995206cc93b72cd63fefd` |

The [completed manifest](/home/stepan/Research/_runs/2026-09-09T194505Z-tofy-looped-action-equivariance/completed-campaign.manifest.json) and
[accepted verification](/home/stepan/Research/_runs/2026-09-09T194505Z-tofy-looped-action-equivariance/completed-campaign-verification.json) record
**3,692 files /1,564,624,017 bytes, 2,610 external bindings, 1,356 recorded PIDs
gone and 54 CUDA bundles**. This is point-in-time integrity, not immutable
storage. The failed capacity trial remains in the sealed provenance. No
experiment process is intentionally retained.

## Historical commands and bounded next decision

These commands reproduce the recorded invocation, not permission to reuse the
completed root. Any new execution requires a new registered root and its own
qualification. The exact process records preserve the working directory,
arguments, deadlines and PIDs:

- [Execution](/home/stepan/Research/_runs/2026-09-09T194505Z-tofy-looped-action-equivariance/operations/execute.process.json)
- [Analysis](/home/stepan/Research/_runs/2026-09-09T194505Z-tofy-looped-action-equivariance/operations/analysis-stage.process.json)
- [Sealing](/home/stepan/Research/_runs/2026-09-09T194505Z-tofy-looped-action-equivariance/operations/outer-seal.process.json)

```bash
cd /home/stepan/Projects/code/Tofy-equivariant-binding
/home/stepan/venvs/tensorboard/bin/python3 \
  /home/stepan/Research/_runs/2026-09-09T194505Z-tofy-looped-action-equivariance/campaign_operator.py \
  execute \
  --campaign /home/stepan/Projects/code/.tofy-runs/looped-action-equivariant-20260909T212250-IST \
  --spec-sha256 86b3e47c5680633ec49739c2a679bb129e5ecef22ae7247a943225165b82c6aa
/home/stepan/venvs/tensorboard/bin/python3 \
  /home/stepan/Research/_runs/2026-09-09T194505Z-tofy-looped-action-equivariance/campaign_operator.py \
  analyze \
  --campaign /home/stepan/Projects/code/.tofy-runs/looped-action-equivariant-20260909T212250-IST \
  --spec-sha256 86b3e47c5680633ec49739c2a679bb129e5ecef22ae7247a943225165b82c6aa
```

[Build provenance](/home/stepan/Research/_runs/2026-09-09T194505Z-tofy-looped-action-equivariance/build.json) records the exact reviewed-source
command, target directory, environment and separately hashed binary:

```bash
CARGO_TARGET_DIR=/home/stepan/Coding/Personal/.tofy-build/v6-recipe-repair/target \
TOFY_BUILD_COMMAND='C18 source ec6ba27c05fa1032a82362a05dbd15fe51cc043a; CARGO_TARGET_DIR=/home/stepan/Coding/Personal/.tofy-build/v6-recipe-repair/target cargo build --release --locked --offline --features cudnn,profiling,serde_json/float_roundtrip --example action_binding_probe' \
cargo build --release --locked --offline \
  --features cudnn,profiling,serde_json/float_roundtrip --example action_binding_probe
```

The result supports a learned finite selector after explicit observed-action
routing, with exact shared recurrence retained and an unchanged presentation
budget in the matched comparison. Its 16-type symmetry closure, single seed,
slightly different head/projection sizes and privileged reused visual cache
bound the claim. It does not establish useful recurrence, a general binder,
a native image-to-action system, an LLM controller or ARC performance.

The registered pass permits preparing a fresh native image-to-action adapter
check. That next experiment requires its own claim and qualification; no
subsequent training result is claimed here. Native-policy and public best
metrics remain unchanged.
