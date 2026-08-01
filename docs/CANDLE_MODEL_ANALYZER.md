# candleModelAnalyzer integration

Tofy integrates the sibling `../candleModelAnalyzer` repository as an external
audit, not a training dependency. This keeps the learned model runnable without the
analyzer while making structure, checkpoint keys, and first-step gradients
machine-checkable.

## Run it

Every `p2-train` run writes `runtime.json` using the analyzer's
`candle-graph/runtime/1` schema. Audit source only:

```bash
scripts/audit_p2.sh runs/p2/analyzer
```

Audit source, checkpoint keys, and runtime gradients together:

```bash
scripts/audit_p2.sh \
  runs/p2/analyzer \
  runs/p2/smoke/model.safetensors \
  runs/p2/smoke/runtime.json
```

The output directory contains `summary.json`, `doctor.json`, `model-ir.json`, and
`world-model.json`, plus `checkpoint.json` and `runtime.json` when those optional
inputs are supplied. The script fails on missing inputs or analyzer command errors.

## Verified smoke result

The first P2 smoke audit found:

- Candle `0.11.0` matched the analyzer catalog;
- all 25 certain `WorldModel` checkpoint tensors matched, with no missing or
  unclaimed tensors;
- all 25 parameters had finite, non-zero first-step gradients;
- no dtype conflict or dtype-risk finding was reported.

The crate-wide doctor still reports source analysis rather than compiler-resolved
evidence. In unified model-IR mode it does not reconstruct components/parameters;
the explicit legacy `--root WorldModel --entry WorldModel::forward` audit does. The
script runs both views so this limitation stays visible instead of silently turning
unknown evidence into a pass.

## What candleModelAnalyzer can improve

1. Recognize crate-wide Candle components built through `VarBuilder` without needing
   an explicit legacy root; the current unified summary reports zero components and
   parameters even though the root audit proves 25 parameters.
2. Finish its compiler-resolved frontend so names/types, macros, cfg expansion,
   dispatch, and value flow use compiler identities rather than source heuristics.
3. Improve recursive/control-flow transfer rules. `WorldModel::forward` currently
   leaves loops, result macros, collection operations, and some tensor receivers as
   `Unknown`, producing many dataflow diagnostics despite complete parameter-key
   discovery.
4. Correlate runtime records to static tensor IDs more ergonomically. Tofy's trace
   can prove gradient presence by `(root,key)`, but its tensor observation has no
   stable static ID because obtaining that ID currently requires a separate scan.
5. Keep Candle operation catalogs version-gated and expand them deliberately beyond
   `0.11.0`; unknown future APIs must not inherit old semantics silently.
6. Add a strict mode that distinguishes analyzer coverage gaps from actual model
   defects. Today `--strict` can reject a valid recursive model because unsupported
   source constructs remain unknown.
