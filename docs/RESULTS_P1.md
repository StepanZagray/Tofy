# Results P1

Prior P1 metrics. Archived pre-P1 numbers are in [`RESULTS_P0.md`](RESULTS_P0.md).

## Best So Far

P0 was skipped, so these are **exploratory**, not confirmatory, results. P1A/P1B
settings were frozen before either report was inspected: seeds `1..8`, 60 episodes
per split and seed, all six hidden families round-robin, and 9,999 paired stratified
bootstrap replicates. P1C is development evidence: its always-parallel version used
seeds `1..8`; later falsification, hybrid, and routing studies used seeds `101..108`
after the earlier behavior was inspected. Those runs intentionally reuse identical
scenarios to isolate policy changes, not to claim fresh confirmatory results.
The hard retargeting study froze seeds `201..208` before its full run, but a six-task
seed-201 implementation smoke was inspected first; it also remains developmental.

| Phase / held-out agent | N | Success | Oracle-normalized efficiency | Other | Gate |
|---|---:|---:|---:|---:|---|
| P1A `candidate_goal_discrimination` | 480 | 0.8396 | 0.3686 | objective ID 0.6479 | PASS vs strongest exploration; success CI low 0.4500, efficiency CI low 0.2067 |
| P1B `beam_search` | 480 | 0.5542 | 0.5536 | 0 terminal failures | PASS vs strongest one-step controls; success CI low 0.1813, efficiency CI low 0.2083 |
| P1B `oracle_optimal` | 480 | 1.0000 | 1.0000 | mean 12.64 actions | PASS vs reactive; success/efficiency CI low 0.6563 |
| P1C falsification-only `set_aware_parallel_planning` | 480 | 0.9500 | **0.5564** | 0 failures; multi-goal probe-action rate 0.4093 | **FAIL** configured +0.05 gate; success lift +0.0188, CI [-0.0083, 0.0458]; efficiency lift +0.0087, CI [-0.0166, 0.0342] |
| P1C reversed router `broad_falsify_narrow_progress` | 480 | **0.9521** | 0.5509 | 0 failures; 0.60 method switches/episode | **FAIL** configured +0.05 gate; success lift +0.0208, CI [-0.0042, 0.0458]; efficiency lift +0.0032, CI [-0.0234, 0.0293] |
| P1C hard falsification-only | 480 | **0.9771** | 0.4629 | 0 failures; sequential baseline retargeted 4.42 times/episode | **PASS** vs sequential; success CI low +0.0750, efficiency CI low +0.3137 |
| P1C hard reversed router | 480 | **0.9771** | **0.4645** | 0 failures; 0.58 method switches/episode | **PASS** vs sequential; success CI low +0.0750, efficiency CI low +0.3149 |

P1A report SHA-256:
`92a4f1e310e13026b800abfcf81218f65b7eac8cfc2ee0ba2f2cc03683daa631`.
P1B report SHA-256:
`083e8feaac81fa4875cf8517a1d6b321e2936a696bd10c07c442c4fb9c720602`.
P1C routing-study report SHA-256:
`e1eb8b737669d5951fd0b5278ae7f95f1fc4a9c64f005c1cd127567a38a55af9`.
P1C hard retargeting report SHA-256:
`10e10d2b829a7ec3f12e430be18238427787deda4034b036feae877a26a3c004`.
Historical P1C falsification-only report SHA-256:
`64ed2fc65fa8fd95ea100902e91c950fab7c866b391608b0648a81a2921a7801`.
Historical always-parallel P1C report SHA-256:
`e997c51beaefc4d34a02c2cb6aa0a0c4cd1857a566a116e1faf9e05917889086`.
Historical shared-prefix/commitment P1C report SHA-256:
`a15fd941b9532ed41b83da4c39c03c41c15ad1be29fc405b645098e19e453991`.

The first P1C policy continually maximized shared progress. It achieved only 0.3125
held-out success and 0.0380 efficiency because it kept hypotheses alive and often
cycled. A strict-majority shared-prefix plus finite-commitment policy raised success
to 0.9229 but remained inefficient at 0.2214: it averaged 29.47 environment actions
and spent 36.74% of them in shared-prefix mode.

The routing study compares both parallel primitives and five ways of deciding between
them. Shared progress takes a jointly safe canonical first action supported by at
least two and a strict majority of exact-live shortest plans. Falsification executes a
safe probe whose endpoint makes at least two live goals predict success, so a
nonterminal observation can reject several goals at once. The exact-live posterior is
uniform: one live goal is a clear winner, two or three are a narrow tie, and four or
more are broad uncertainty.

| Held-out strategy | Routing while ambiguous | Success | Mean actions | Efficiency | Internal work | Shared action rate | Probe action rate | Switches / episode |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Sequential baseline | One goal at a time | 0.9313 | 20.29 | 0.5477 | 2,615 | — | — | — |
| Falsification-only | Always falsify if possible | 0.9500 | 21.64 | **0.5564** | 271,317 | 0.000 | 0.409 | 0.00 |
| Shared-progress-only | Always seek majority progress | 0.9313 | 28.71 | 0.2094 | 353,777 | 0.406 | 0.000 | 0.00 |
| Proposed router | Progress broad; falsify narrow | 0.9417 | 26.70 | 0.1696 | 349,953 | 0.414 | 0.272 | 2.51 |
| Reversed router | Falsify broad; progress narrow | **0.9521** | 21.95 | 0.5509 | 276,584 | 0.181 | 0.404 | 0.60 |
| Dumb alternation | Alternate actual parallel choices | 0.9458 | 23.64 | 0.2364 | 314,321 | 0.234 | 0.393 | 2.86 |
| Cost-aware | Compare predicted falsifications/action with distance reduction/action | 0.9417 | 26.74 | 0.1698 | 500,631 | 0.411 | 0.271 | 2.58 |
| Capped broad progress | Proposed router; force falsification after two progress actions | 0.9458 | 25.47 | 0.1813 | 346,051 | 0.282 | 0.374 | 2.85 |

The result reverses the initial routing hypothesis. With many equally plausible goals,
shared progress is easy to find but weak evidence: it preserves too many hypotheses
and adds actions. Falsification is most valuable under that broad uncertainty because
one observation can remove several candidates. Shared progress is less damaging only
after the set has narrowed. Accordingly, the reversed router has the best success,
while falsification-only has the best efficiency and is the simpler default. Dumb
alternation and the two anti-overuse heuristics reduce some damage relative to the
proposed router but remain far less efficient than falsification-only.

Sequential discrimination scored 0.9313 success, 20.29 actions, and 0.5477 efficiency,
but incurred a 0.0375 terminal-failure rate. Every parallel policy eliminated terminal
failures. No strategy passed the configured gate, which requires the *lower* paired
confidence bound to exceed +0.05 for both success and efficiency. For the two leading
strategies, both intervals include zero; this is evidence of parity with a small point
advantage, not proof of superiority. These `101..108` scenarios were already used
during P1C development, so this is developmental model selection rather than a fresh
confirmatory test.

The family breakdown explains much of the trade-off. Falsification-only reaches
1.000 success and 0.7825 efficiency on avoid-hazard tasks, versus sequential's 0.775
success and 0.5807 efficiency, and exactly preserves the one-action resource probe.
It is weaker on switch-order success (0.775 versus 0.8125) and less efficient on
collect-all tasks. The reversed router trades a little falsification efficiency for
narrow-set shared progress; its 0.9521 aggregate success is one episode above the
falsification-only policy.

## P1C hard: research before greedy commitment

This separate challenge asks whether the agent should commit to one easy-looking
live hypothesis immediately or first gather evidence that can reject several
hypotheses. It runs exactly three agents: sequential discrimination,
falsification-only, and broad-falsify/narrow-progress. P1 has a uniform exact-live
belief rather than a graded likelihood model, so “greedy” here means the sequential
policy's minimum-heuristic candidate, not a literal maximum-posterior goal.

The hard generator creates a fork with a deep, recoverable side corridor and a safe
multi-goal endpoint probe. A deterministic attempt stream accepts a scenario only if
sequential discrimination retargets after at least three falsified commitments.
Neither sequential success nor either parallel policy's behavior is used for task
selection. Every accepted hidden goal is exact-solvable, each split retains all six
families round-robin, and every accepted task has a safe probe of width at least two.

| Held-out agent | Success | Terminal failures | Mean actions | Efficiency | Internal work | Key diagnostic |
|---|---:|---:|---:|---:|---:|---|
| Sequential discrimination | 0.8750 | 0.0542 | 66.47 | 0.1205 | 5,200 | 4.42 wrong commitments/episode |
| Falsification-only | **0.9771** | 0.0000 | 34.67 | 0.4629 | 527,179 | probe-action rate 0.5148 |
| Broad-falsify/narrow-progress | **0.9771** | 0.0000 | **34.46** | **0.4645** | 522,300 | shared-action rate 0.1515; 0.58 switches/episode |

Both research-first policies beat sequential by +0.1021 success and about +0.343
oracle-normalized efficiency. Their paired 95% lower bounds clear the configured
+0.05 thresholds: falsification-only has success CI [0.0750, 0.1313] and efficiency
CI [0.3137, 0.3702]; the reversed router has success CI [0.0750, 0.1313] and
efficiency CI [0.3149, 0.3720]. Both therefore pass their exploratory hard-slice
gates. Sequential retarget counts range from three to seven, with all 480 held-out
episodes satisfying the contract and a mean of 4.42.

The two research-first policies are indistinguishable here: their success is exactly
equal, and the reversed router's +0.0016 efficiency lift over falsification-only has
CI [-0.0018, 0.0050]. Narrow-set shared progress therefore adds no demonstrated value
after broad falsification. Falsification-only remains the simpler recommendation.

The largest family gains appear where early commitment is dangerous or especially
wasteful. On avoid-hazard tasks, sequential succeeds 0.675 with 0.325 terminal
failure and 0.033 efficiency; both research policies succeed 1.000 with no terminal
failure and about 0.80 efficiency. On preserve-resource tasks, sequential succeeds
0.850 at 0.014 efficiency, while both parallel policies score 1.000 success and 1.000
efficiency. Switch-order and collect-all remain the hardest research-first families.

This is strong mechanism evidence but deliberately selected developmental evidence,
not an unbiased estimate of performance on arbitrary ARC-like tasks. Selection is
conditioned on the sequential policy's retargeting failure mode. It proves that
multi-goal research can substantially help when early single-goal commitments are
misleading; it does not establish how often that condition occurs in ARC-AGI-3. The
compute trade-off is also large: parallel policies use roughly 100 times the internal
search work even though they halve environment actions.

An earlier P1C execution was discarded before interpretation because five transformed
held-out preserve-resource tasks were oracle-unsolvable. The generator now places a
side-branch resource pickup, exact-checks every transformed hidden goal, and assigns
the required plan length plus eight actions of slack. The recorded report above has
zero unsolvable oracle episodes.

## Reproduction commands

### Recorded exploratory P1A

```bash
cargo run --release -- p1a \
  --seeds 1,2,3,4,5,6,7,8 \
  --episodes-per-split 60 \
  --bootstrap-samples 9999 \
  --output runs/p1/p1a_exploratory_s1-8_n60.json
```

### Recorded exploratory P1B

```bash
cargo run --release -- p1b \
  --seeds 1,2,3,4,5,6,7,8 \
  --episodes-per-split 60 \
  --bootstrap-samples 9999 \
  --output runs/p1/p1b_exploratory_s1-8_n60.json
```

### Recorded developmental P1C routing study

```bash
cargo run --release -- p1c \
  --seeds 101,102,103,104,105,106,107,108 \
  --episodes-per-split 60 \
  --bootstrap-samples 9999 \
  --output runs/p1/p1c_routing_s101-108_n60.json
```

### Recorded developmental P1C hard retargeting study

```bash
cargo run --release -- p1c-hard \
  --seeds 201,202,203,204,205,206,207,208 \
  --episodes-per-split 60 \
  --bootstrap-samples 9999 \
  --output runs/p1/p1c_hard_retarget_s201-208_n60.json
```

### Smoke (quick local)

```bash
cargo run --release -- all \
  --seed 1 \
  --episodes-per-split 2 \
  --output runs/p1/smoke_all.json
```

This one-seed smoke is expected to fail the exploratory gates closed. Passing gates
requires at least two seeds and replicated coverage of all six objective families.
It includes all seven P1C strategies and the sequential discrimination baseline.

Multi-seed implementation validation:

```bash
cargo run --release -- all \
  --seeds 1,2 \
  --episodes-per-split 6 \
  --bootstrap-samples 999 \
  --output runs/p1/validation_all.json
```

This remains validation rather than experimental evidence.

Confirmatory settings are intentionally absent because P0 was skipped. Do not label
a run confirmatory until its seed set, episode count, lift thresholds, and statistical
protocol are frozen in advance.

## Update rule

When a better metric is reported:

1. Replace the relevant **Best So Far** row with the new metric.
2. Paste the **exact command** used.
3. Keep oracle-normalized efficiency labeled as such (never call it RHAE).
4. Do not mix archived VecLab/Qwen numbers into this table; that experiment is
   archived in [`RESULTS_P0.md`](RESULTS_P0.md).
5. Do not promote implementation smoke tests into this document.
