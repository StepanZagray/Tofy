# C11 infrastructure retry

The original C11 device qualification passed its numerical checks, but the first zero-forward CPU audit failed a reused GPU verifier that required a nonempty host trace. Review then found a report-only variable shadowing error: the analyzer returned an input hash instead of the already validated root-manifest hash. The original artifacts and reports remain sealed without repair.

This directory snapshots only the four corrected operator/evaluator source and test files plus the retry contract. All other C11 registration, translation, packaging and sequencing source remains byte-identical to the adjacent `2026-09-09-looped-cuda-readout` snapshots. Runtime copies and exact source/export/config bindings live in `/home/stepan/Research/_runs/2026-09-09T141234Z-tofy-looped-cuda-readout-retry`.

CPU audits now require their exact zero-work artifact contract, including a hash-bound empty JSON host trace. Every CUDA invocation retains the existing populated-capture validator. The analyzer retains the original root hash independently of input-file checks. These corrections change no model, generator, data seed, fit, numerical tolerance, or scientific gate. The retry runs from a new campaign root and requalifies the exact newly built binaries.

Qualification remains implementation smoke. The fixed synthetic role-supervised readout is not an integrated controller or ARC result; the original C11 registration retains all interpretation and profiling limits.
