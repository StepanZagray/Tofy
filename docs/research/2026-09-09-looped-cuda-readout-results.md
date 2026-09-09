# C11 fixed-readout CUDA confirmation — September 9, 2026 IST

Both frozen cores' fixed C10 true heads score **768/768 actions and joint agent/goal selections** on three registered 256-query panels. All six core/panel gates pass: `confirmed_on_registered_synthetic_populations`. Independent raw calculations reproduce the result. This is a known-control, one-step synthetic readout result: C10 used privileged role labels during fitting, and initial-core success receives no C7 training credit. Native policy remains **188/768 initial, 209/768 final**. No native-controller, ARC, architecture or useful-recurrence improvement is established. [accepted confirmation analysis](/home/stepan/Research/_runs/2026-09-09T141234Z-tofy-looped-cuda-readout-retry/confirmation-analysis.json); [independent raw review](/home/stepan/Research/_runs/2026-09-09T141234Z-tofy-looped-cuda-readout-retry/findings/independent-review-01.json).

## Frozen comparison and data boundary

Ten fixed head/core combinations use the common initial and C7-final cores, each with C10 true-role, C10 permuted-role, C9 policy-trained spatial, C9 policy-trained CLS and C9 permuted-policy spatial heads. C11 fits nothing and executes zero optimizer updates. C10 canonical tensors are cast once to F32; C9 named F32 tensors retain their exact bytes. The shared core runs four loops; the imported policy head has no recurrence. Core extraction and cached-head inference are separate CUDA stages, not an integrated acting controller. [unchanged registration](/home/stepan/Research/_runs/2026-09-09T141234Z-tofy-looped-cuda-readout-retry/registration.md).

The 768 distinct queries use seeds 20260917/20260918/20260919, episode bases `0x43554441434f4e46 + panel*0x10000`, fixed mapping 0, distance one and factual support. Audits verified all three panels before six extractions, excluding six historical JSONLs with a 5,688-query union and rejecting within/cross-panel query/input/episode collisions without replacement. The six caches were sealed before thirty head evaluations. Geometry independently validates 256/256 labels per panel. The new wall/support combinations remain within the same 120 adjacent-role configurations; no structural extrapolation claim follows. [panel seal](/home/stepan/Research/_runs/2026-09-09T141234Z-tofy-looped-cuda-readout-retry/panel-seal.json); [cache seal](/home/stepan/Research/_runs/2026-09-09T141234Z-tofy-looped-cuda-readout-retry/cache-seal.json); [independent raw review](/home/stepan/Research/_runs/2026-09-09T141234Z-tofy-looped-cuda-readout-retry/findings/independent-review-01.json).

Physical/effective batches are 1 for core extraction, 512 for old-fit-cache head qualification and 256 for each full head panel, accumulation 1. Each inference batch exhausts its available population; there was no training-batch search or optimization in C11.

## Counts, controls and uncertainty

| Endpoint | Initial core action accuracy | C7-final core action accuracy | Mean CE, initial / final |
|---|---:|---:|---:|
| c 10_true | 768/768 (100.00%) | 768/768 (100.00%) | 0.954857 / 0.954897 |
| c 10_null | 188/768 (24.48%) | 188/768 (24.48%) | 2.593450 / 22.141767 |
| c 9_spatial | 166/768 (21.61%) | 256/768 (33.33%) | 1.435523 / 1.341918 |
| c 9_cls | 166/768 (21.61%) | 199/768 (25.91%) | 1.441581 / 1.402342 |
| c 9_null | 197/768 (25.65%) | 180/768 (23.44%) | 1.405972 / 1.401547 |
| native | 188/768 (24.48%) | 209/768 (27.21%) | 1.505785 / 1.391007 |
| oracle_role | 768/768 (100.00%) | 768/768 (100.00%) | 0.954856 / 0.954897 |
| Best constant | 209/768 (27.21%) | 209/768 (27.21%) | No probabilistic fit |

| Panel / seed | Action-label counts [0,1,2,3] | Best constant /256 | C10 true, initial / final | C9 spatial, initial / final | Native, initial / final |
|---|---|---:|---:|---:|---:|
| 0 / 20260917 | [78, 51, 71, 56] | 78 | 256 / 256 | 55 / 81 | 51 / 78 |
| 1 / 20260918 | [57, 67, 69, 63] | 69 | 256 / 256 | 54 / 93 | 67 / 57 |
| 2 / 20260919 | [74, 70, 60, 52] | 74 | 256 / 256 | 57 / 82 | 70 / 74 |

C10 true agent, goal and joint counts are all 768/768 for each core. C10 null joint counts are 0/768, but individual localization is **initial agent 121/768, goal 0/768; final agent 0/768, goal 12/768**. Nulls are not uniformly devoid of role correlations. C10 null and C9 null action gates pass on every panel; oracle-role actions are perfect throughout. CUDA true-role attention has no argmax ties; its smallest retained target mass is 0.9999990463. The finite fitting-margin bound was not assumed to prove new-panel behavior. [accepted confirmation analysis](/home/stepan/Research/_runs/2026-09-09T141234Z-tofy-looped-cuda-readout-retry/confirmation-analysis.json), `panels`, `pooled.roles`.

The minimum per-panel lower 95% paired advantage is **63.671875 percentage points over constant** and **57.8125 points over C9 spatial**, both above the strict registered 25-point gates. Joint localization >=95%, action >=90%, positive/oracle/null controls and all integrity checks pass in every core/panel cell. Pooled metrics cannot rescue a failed cell.

C9 final spatial has a real positive pooled contrast: **33.33% versus constant 27.21%, +6.1198 pp [1.3021,9.5052]**. Its per-panel advantages are +1.1719 pp [−7.4219,7.4316], +9.3750 pp [0.3906,14.4531], and +3.1250 pp [−5.4785,8.9844]; only panel 1 excludes zero in its pointwise interval. Initial spatial is 21.61%; final-minus-initial spatial is +11.7188 pp [7.9427,15.6250]. This is a frozen joint core/head contrast, not a causal isolation of core training or a reversal of C9's failed original-panel decision. C10 true final-minus-initial accuracy is 0 pp [0,0]. [accepted confirmation analysis](/home/stepan/Research/_runs/2026-09-09T141234Z-tofy-looped-cuda-readout-retry/confirmation-analysis.json), `pooled`, `panels.*.statistics`.

Intervals use 10,000 paired whole-query PCG64 draws, seeds 1920/1921/1922 per panel and stratified seed 1919 pooled with 256 queries retained per panel. Best constant is reselected on every draw. All intervals are pointwise, not simultaneous. The true-head empirical accuracy interval [1,1] is not population perfection: descriptive Wilson 95% is [99.5023%,100%] pooled and[98.5216%,100%] per panel, assuming independent queries. C10 true CE≈0.955 is descriptive; affine-ridge softmax calibration is unproven. One fixed null permutation does not estimate a general chance distribution. [unchanged registration](/home/stepan/Research/_runs/2026-09-09T141234Z-tofy-looped-cuda-readout-retry/registration.md); [accepted confirmation analysis](/home/stepan/Research/_runs/2026-09-09T141234Z-tofy-looped-cuda-readout-retry/confirmation-analysis.json).

## Numerical qualification and independent verification

Eleven device qualifications (ten heads on the previously accessed 512 fitting rows plus one old-input core smoke) are implementation evidence only. Logits and pooled tensors must satisfy `abs_error <= 1e-4 + 1e-5*abs(reference)`; attention uses `1e-5 + 1e-5*abs(reference)`. Every action must match exactly. F32 reduced-precision GEMM is disabled and `NVIDIA_TF32_OVERRIDE=0`; no tolerance changed after outputs. Qualification max absolute logits/pooled/attention errors are 2.4795532e-5/3.8146973e-6/3.5762787e-7; old core CLS/current/policy raw tensors match exactly. [qualification report](/home/stepan/Research/_runs/2026-09-09T141234Z-tofy-looped-cuda-readout-retry/qualification-analysis.json).

The independent verifier imported no production analyzer or fitting helper. It checked original/imported/initial/final head tensors, both frozen core hashes, F32 offsets and finiteness, visible geometry, query hashes, all 50 root manifests and lifecycle barriers. All CUDA actions match independent named-tensor reconstruction; maximum logits/attention/pooled error is 3.0517578e-5/8.3446503e-7/3.8146973e-6. Seven synthetic fixtures passed. All 247 report comparisons agree, max mean-CE difference 3.5527136788e-15. Independent computation checks this same experiment, not a second experiment. It rehashed parent manifests and original parameter artifacts, not every historical campaign file itself. [independent raw review](/home/stepan/Research/_runs/2026-09-09T141234Z-tofy-looped-cuda-readout-retry/findings/independent-review-01.json); [247-field comparison](/home/stepan/Research/_runs/2026-09-09T141234Z-tofy-looped-cuda-readout-retry/findings/independent-comparison.json); [independent commands and limits](/home/stepan/Research/_runs/2026-09-09T141234Z-tofy-looped-cuda-readout-retry/findings/independent-review.md).

## Failed first attempt and retry provenance

Original source `a2f1125d15bb5fff2f86f8a47887aeb7e575973d` passed all 11 qualification runs, then generated valid panel 0 with 256 rows and zero model forwards/updates. The pinned C8 GPU-oriented verifier treated parsed empty CPU host trace `[]` as failure at line 43. The backend returned 0; the wrapper correctly withheld acceptance after its verifier failed. No fresh core/head inference happened in that attempt. Original panel 0 bytes are identical in the retry. Original model qualification cost 47.8101346710 s, finalization 9.8008899700 s, build 32.4851528430 s outside its clock, CPU audit backend 0.291418498 s, and start-to-failed-outer-wrapper 104.820758 s remain retained. [original failed synthesis](/home/stepan/Research/_runs/2026-09-09T131652Z-tofy-looped-cuda-readout-confirmation/synthesis.md); [original exact accounting](/home/stepan/Research/_runs/2026-09-09T131652Z-tofy-looped-cuda-readout-confirmation/verification.md).

A separate original analyzer bug reused `sha` while checking frozen inputs, returning the last input hash instead of the validated root-manifest hash. The original numerical checks and actual exit/manifest comparisons were correct; ten summary manifest fields were wrong. Retry preserves `root_manifest_sha` and uses a CPU-only verifier requiring exact registered argv, five artifacts plus manifest, empty profiles/trace and zero work; GPU verification is unchanged. All Rust, registration, ten canonical parameter payloads, head/data mathematics and gates are unchanged. The retry regression suite passes 30 operator+10 package/sequence+28 analyzer fixtures; its initial unpinned BLAS test invocation failed the thread guard before passing in the exact supervised single-thread environment. No original artifact was repaired in place. [audit/manifest fixes](/home/stepan/Research/_runs/2026-09-09T141234Z-tofy-looped-cuda-readout-retry/findings/audit-fix.md); [source review](/home/stepan/Research/_runs/2026-09-09T141234Z-tofy-looped-cuda-readout-retry/findings/final-source-review.json).

| Identity | Exact value |
|---|---|
| Retry campaign | `/home/stepan/Projects/code/.tofy-runs/looped-cuda-readout-retry-20260909T151301-IST` |
| Retry source | `306582c713f51a4c9ff461375f928abca2408444` |
| Dependency | `1ea5cc5c5998a1c2bd388eb158653dd0d1b7452a` |
| Head binary SHA256 | `38fd0bf3dcc2bb65102e6533b0afa4c677fe977bda543e467a8e2afa8813cbe3` |
| Core binary SHA256 | `39bef5f8c2d85501458cda14e69fe67ff379e06a3882970567cebfa4a8c11127` |
| Initial core SHA256 | `4cd502f7a76fe3dd9729f6693e9d707c670972f7ebe58ad1cbdde37ea2bb2802` |
| C7-final core SHA256 | `a983626a888279cbb31651c6abcd1ebf4f808c8927a29a8349d03ffc35bd4e5a` |
| Unchanged registration SHA256 | `ce687312edae9ed9edb914217d0c6383fca8b6c9ece1c3a72a6d8e12a68d6a2b` |
| Primary analysis SHA256 | `ccec3a08f926b34b9e4b96bfdcef575978250cbab0f2219dbf4f790f07fcf73c` |
| Independent analysis SHA256 | `45a9a5149ccbd1d391031d0da3c724847d9014dba5de2bc5e95592dfd2619a26` |
| Original failed seal SHA256 | `a534bd999136be79e7a8cb962bb89a9371eceecc7bf60c44bf152388546ba508` |
| Retry completed seal SHA256 | `fa1eea8210155135e0afd5acc9ac06c35c1611569a362dd7cc64438185083e52` |

The original seal preserves 821 files/105,571,414bytes and 94 bindings with 211 PIDs gone. Retry sealing at **15:42:09 IST** preserves 3,427 files/373,015,350bytes and 109 bindings with 903 recorded PIDs gone. This is point-in-time integrity, not immutable storage. Source snapshots, scripts, checkpoints and campaign outputs remain unchanged. [original seal](/home/stepan/Research/_runs/2026-09-09T131652Z-tofy-looped-cuda-readout-confirmation/failed-campaign.manifest.json); [completed campaign seal](/home/stepan/Research/_runs/2026-09-09T141234Z-tofy-looped-cuda-readout-retry/completed-campaign.manifest.json); [seal revalidation](/home/stepan/Research/_runs/2026-09-09T141234Z-tofy-looped-cuda-readout-retry/completed-campaign-verification.json).

## Compute and profiler limits

On the NVIDIA GeForce RTX5060 Laptop GPU (8151 MiB, driver 610.57.04), retry build took 24.7017204030 s outside its campaign clock. Qualification+confirmation model phases consumed 216.1990418920/360 s; profiler/supervisor finalization 44.5346857100/600 s. First-qualification to last-head completion was 396.099776 s; primary analysis 6.7435698890 s and independent numerical review 1.865727 s. The analysis wall snapshot was 445.699280 s; the final seal was 1,221.576642 s after the first qualification launch, below 1800 s. Preparation/build and original failed-attempt costs remain separate. [independent raw review](/home/stepan/Research/_runs/2026-09-09T141234Z-tofy-looped-cuda-readout-retry/findings/independent-review-01.json); [accepted confirmation analysis](/home/stepan/Research/_runs/2026-09-09T141234Z-tofy-looped-cuda-readout-retry/confirmation-analysis.json); [completed campaign seal](/home/stepan/Research/_runs/2026-09-09T141234Z-tofy-looped-cuda-readout-retry/completed-campaign.manifest.json).

All 47 CUDA first-forward captures and three empty CPU traces pass their declared contract. Wired evidence includes semantic spans, labelled tensor/seam statistics, host scalars, host/NVTX traces and bound Nsight CUDA/NVTX/OSRT/cuDNN/cuBLAS/CPU sampling. No training gradients are expected at zero updates. Operation/activation linkage, allocation lifetimes, instrumented physical-memory checkpoints and device-event intervals remain unwired. Automatic GPU correlation is incomplete. Sampled GPU memory is not allocator peak, host spans are not kernel durations, and `profiled_work` captures are not production timing evidence. NumPy reference operations have no CandleGraph instrumentation. [qualification capabilities](/home/stepan/Research/_runs/2026-09-09T141234Z-tofy-looped-cuda-readout-retry/qualification-analysis.json); [independent raw review](/home/stepan/Research/_runs/2026-09-09T141234Z-tofy-looped-cuda-readout-retry/findings/independent-review-01.json).

## Exact historical commands

These commands are records, not instructions to reuse sealed roots. Working directory: `/home/stepan/Projects/code/Tofy-cuda-readout`. Exact build:

```bash
CARGO_TARGET_DIR=/home/stepan/Coding/Personal/.tofy-build/v6-recipe-repair/target cargo build --release --locked --offline --package tofy --features cudnn,profiling,serde_json/float_roundtrip --example learned_readout_probe --example looped_agent_probe
```

Exact supervised true-head invocations for all six confirmed cells, copied from campaign `operations/*-supervisor.process.json`:

```bash
# head-0-initial-c10_true
/home/stepan/venvs/tensorboard/bin/python3 -B /home/stepan/Research/_runs/2026-09-09T141234Z-tofy-looped-cuda-readout-retry/supervise_stage.py \
  --config /home/stepan/Projects/code/.tofy-runs/looped-cuda-readout-retry-20260909T151301-IST/invocations/head-0-initial-c10_true.json \
  --sha256 c03680adb5b11ffd35d825faf61131641f2056a3f8ed74e6ba2abcadbcb15c10

# head-0-final-c10_true
/home/stepan/venvs/tensorboard/bin/python3 -B /home/stepan/Research/_runs/2026-09-09T141234Z-tofy-looped-cuda-readout-retry/supervise_stage.py \
  --config /home/stepan/Projects/code/.tofy-runs/looped-cuda-readout-retry-20260909T151301-IST/invocations/head-0-final-c10_true.json \
  --sha256 67e989d449d143502e99ee88c815b65b4d00c86cabd0f79fba8ba19ed66147e5

# head-1-initial-c10_true
/home/stepan/venvs/tensorboard/bin/python3 -B /home/stepan/Research/_runs/2026-09-09T141234Z-tofy-looped-cuda-readout-retry/supervise_stage.py \
  --config /home/stepan/Projects/code/.tofy-runs/looped-cuda-readout-retry-20260909T151301-IST/invocations/head-1-initial-c10_true.json \
  --sha256 01e6d3418d9d8537e87c65008e733c4a1ffe2869ef2a6a8e63b30ac745eb45cd

# head-1-final-c10_true
/home/stepan/venvs/tensorboard/bin/python3 -B /home/stepan/Research/_runs/2026-09-09T141234Z-tofy-looped-cuda-readout-retry/supervise_stage.py \
  --config /home/stepan/Projects/code/.tofy-runs/looped-cuda-readout-retry-20260909T151301-IST/invocations/head-1-final-c10_true.json \
  --sha256 e0e1cd95ac658b2ccd97dbf619cba9952aea540e24156256ed6fcf65c447e83c

# head-2-initial-c10_true
/home/stepan/venvs/tensorboard/bin/python3 -B /home/stepan/Research/_runs/2026-09-09T141234Z-tofy-looped-cuda-readout-retry/supervise_stage.py \
  --config /home/stepan/Projects/code/.tofy-runs/looped-cuda-readout-retry-20260909T151301-IST/invocations/head-2-initial-c10_true.json \
  --sha256 9c827dd7a2628f327f6b5bc02c9386ded5dc2fafa542e9ca89544a76879ec53c

# head-2-final-c10_true
/home/stepan/venvs/tensorboard/bin/python3 -B /home/stepan/Research/_runs/2026-09-09T141234Z-tofy-looped-cuda-readout-retry/supervise_stage.py \
  --config /home/stepan/Projects/code/.tofy-runs/looped-cuda-readout-retry-20260909T151301-IST/invocations/head-2-final-c10_true.json \
  --sha256 b3f59f3438f24b8b176f1082157a7d3f44341b0cf87747f50fad3d432a1eb736
```

Each hashed invocation fixes import/cache hashes, panel, batch and deadline. The supervisor binds the fresh host trace and invokes Nsight with CUDA/NVTX/OSRT/cuDNN/cuBLAS and process-tree CPU sampling; `NVIDIA_TF32_OVERRIDE=0` and `NSYS_NVTX_PROFILER_REGISTER_ONLY=0` are recorded. The complete native argv/environment for the first extraction and true head are retained in [extraction process record](/home/stepan/Projects/code/.tofy-runs/looped-cuda-readout-retry-20260909T151301-IST/features-0-initial.process.json) and [head process record](/home/stepan/Projects/code/.tofy-runs/looped-cuda-readout-retry-20260909T151301-IST/head-0-initial-c10_true.process.json). All 30 names and budget are in [head sequence record](/home/stepan/Projects/code/.tofy-runs/looped-cuda-readout-retry-20260909T151301-IST/heads-sequence.json); individual process/config files are covered by the outer seal.

Exact aggregate analysis command (single-thread BLAS environment: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`):

```bash
/home/stepan/venvs/tensorboard/bin/python3 -B /home/stepan/Research/_runs/2026-09-09T141234Z-tofy-looped-cuda-readout-retry/analyze_cuda_readout.py --config /home/stepan/Research/_runs/2026-09-09T141234Z-tofy-looped-cuda-readout-retry/confirmation-analysis-config.json --sha256 f857e832d2682622dfddd732f0d3385778bf5e45c89d19e8bcffbb1b053e99ca --stage confirmation --output /home/stepan/Research/_runs/2026-09-09T141234Z-tofy-looped-cuda-readout-retry/confirmation-analysis.json
```

[Analysis process record](/home/stepan/Projects/code/.tofy-runs/looped-cuda-readout-retry-20260909T151301-IST/operations/confirmation-analysis.process.json) binds this command; the output root is sealed. Full research sources and verification: [source ledger](/home/stepan/Research/_runs/2026-09-09T141234Z-tofy-looped-cuda-readout-retry/sources.md) and [verification](/home/stepan/Research/_runs/2026-09-09T141234Z-tofy-looped-cuda-readout-retry/verification.md).

## Decision and next falsifier

Confirm the fixed role-supervised spatial readout on these registered populations only. The successful initial arm prevents attributing the capability to C7 learning. C9's modest positive pooled final spatial contrast is retained, but no causal optimizer/core conclusion follows. Successor fidelity, unknown-rule inference, planner value, episode memory, autonomous probing and ARC integration remain separate unmet capabilities.

Next is an **integrated trainable current-token spatial policy**, qualified for the same forward and nonzero policy gradients into both core and head, followed by a separately registered policy-loss variable-control screen. The common initial core and its privileged C11 true-head warm start retain four shared loops; no true role/map indices, parser, LLM or external planner enter learned inference. The C12 claim describes parity/gradient prerequisites and a candidate one-seed screen; no completed training result or launch authorization is implied. A CLS affine probe is not a mandatory intermediate step. [C12 implementation claim and unfinished training hypothesis](/home/stepan/Research/_runs/2026-09-09T143629Z-tofy-looped-grounded-policy-learning/implementation-claim.md).
