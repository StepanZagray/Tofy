#!/usr/bin/env python3
"""ADR 0005 §5.3 analysis: residual-vs-reliability AUROC (numpy only).

Input: JSONL emitted by `tofy p2-residual-probe` with one row per transition:
  source ("real"|"synthetic"), game_id, residual_changed_pixels (int),
  residual_pixel_ce (float, mean per-pixel CE of the composed decode against the
  actual next frame), reliability (sigmoid), q (sigmoid), actual_noop (bool),
  composed_exact (bool).

Outputs analysis.json and prints a Markdown table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney U) with tie handling. labels: 1=positive."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i : j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    r = np.empty_like(ranks)
    r[order] = ranks
    return float((r[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def bootstrap_auroc(scores, labels, rng, n_boot=1000):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    point = auroc(scores, labels)
    n = len(scores)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(auroc(scores[idx], labels[idx]))
    vals = np.array(vals, dtype=np.float64)
    vals = vals[~np.isnan(vals)]
    lo, hi = (np.percentile(vals, [2.5, 97.5]) if len(vals) else (np.nan, np.nan))
    return {"auroc": point, "ci95": [float(lo), float(hi)], "n": int(n),
            "n_pos": int(labels.sum()), "n_neg": int((~labels).sum())}


def quantiles(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return {}
    q = np.percentile(x, [0, 5, 25, 50, 75, 95, 100])
    return {"n": int(len(x)), "mean": float(x.mean()), "min": float(q[0]), "p05": float(q[1]),
            "p25": float(q[2]), "median": float(q[3]), "p75": float(q[4]), "p95": float(q[5]),
            "max": float(q[6]),
            "frac_below_1e-6": float((x < 1e-6).mean()),
            "frac_below_1e-3": float((x < 1e-3).mean()),
            "frac_above_0.5": float((x > 0.5).mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.rows).read_text().splitlines() if l.strip()]
    src = np.array([r["source"] for r in rows])
    real = src == "real"
    synth = src == "synthetic"
    ce = np.array([r["residual_pixel_ce"] for r in rows], dtype=np.float64)
    cp = np.array([r["residual_changed_pixels"] for r in rows], dtype=np.float64)
    rel = np.array([r["reliability"] for r in rows], dtype=np.float64)
    q = np.array([r["q"] for r in rows], dtype=np.float64)
    noop = np.array([bool(r["actual_noop"]) for r in rows])
    exact = np.array([bool(r["composed_exact"]) for r in rows])
    wrong = ~exact
    # Trainer-comparable label: exact on the pixels that actually changed
    # (defined only for changed transitions). None -> excluded via mask.
    cexact_raw = [r.get("composed_changed_exact") for r in rows]
    cexact_defined = np.array([v is not None for v in cexact_raw])
    cexact = np.array([bool(v) if v is not None else False for v in cexact_raw])
    cwrong = ~cexact

    rng = np.random.default_rng(args.seed)
    res = {"n_real": int(real.sum()), "n_synthetic": int(synth.sum()), "bootstrap_resamples": args.boot,
           "seed": args.seed}

    # Q1: real vs synthetic (positive = real). Higher residual / lower reliability should flag real.
    res["real_vs_synthetic"] = {
        "residual_pixel_ce": bootstrap_auroc(ce, real, rng, args.boot),
        "residual_changed_pixels": bootstrap_auroc(cp, real, rng, args.boot),
        "one_minus_reliability": bootstrap_auroc(1.0 - rel, real, rng, args.boot),
        "one_minus_q": bootstrap_auroc(1.0 - q, real, rng, args.boot),
    }
    # Same, restricted to transitions where the frame actually changed (guards against
    # "real is mostly no-op" confound).
    ch = ~noop
    res["real_vs_synthetic_changed_only"] = {
        "residual_pixel_ce": bootstrap_auroc(ce[ch], real[ch], rng, args.boot),
        "residual_changed_pixels": bootstrap_auroc(cp[ch], real[ch], rng, args.boot),
        "one_minus_reliability": bootstrap_auroc(1.0 - rel[ch], real[ch], rng, args.boot),
    }
    # No-op-only stratum: the ideal residual is exactly zero for BOTH sources,
    # so content density cannot explain separation here (spurious-edit residual).
    res["real_vs_synthetic_noop_only"] = {
        "residual_pixel_ce": bootstrap_auroc(ce[noop], real[noop], rng, args.boot),
        "residual_changed_pixels": bootstrap_auroc(cp[noop], real[noop], rng, args.boot),
        "one_minus_reliability": bootstrap_auroc(1.0 - rel[noop], real[noop], rng, args.boot),
    }
    # Q2b: within-set prediction of "changed-exact wrong" on changed rows
    # (the trainer's one_step_changed_exact label). NOTE: residual_changed_pixels
    # is NOT tautological here (it counts errors anywhere; changed-exact only
    # scores the changed pixels), unlike full-exact where residual==0 <=> exact.
    for name, mask in (("real", real & cexact_defined), ("synthetic", synth & cexact_defined)):
        res[f"predict_changed_wrong_within_{name.split('_')[0] if False else name}"] = {
            "residual_pixel_ce": bootstrap_auroc(ce[mask], cwrong[mask], rng, args.boot),
            "residual_changed_pixels": bootstrap_auroc(cp[mask], cwrong[mask], rng, args.boot),
            "one_minus_reliability": bootstrap_auroc(1.0 - rel[mask], cwrong[mask], rng, args.boot),
            "one_minus_q": bootstrap_auroc(1.0 - q[mask], cwrong[mask], rng, args.boot),
            "base_rate_changed_wrong": float(cwrong[mask].mean()) if mask.any() else float("nan"),
        }
    res["rates_changed_exact"] = {
        "real": float(cexact[real & cexact_defined].mean()) if (real & cexact_defined).any() else None,
        "synthetic": float(cexact[synth & cexact_defined].mean()) if (synth & cexact_defined).any() else None,
    }
    # Q2: within-set prediction of "composed decode (full-frame) wrong" (positive = wrong).
    # residual_changed_pixels == 0 <=> composed_exact, so its AUROC here is tautological.
    for name, mask in (("real", real), ("synthetic", synth)):
        res[f"predict_wrong_within_{name}"] = {
            "residual_pixel_ce": bootstrap_auroc(ce[mask], wrong[mask], rng, args.boot),
            "residual_changed_pixels": bootstrap_auroc(cp[mask], wrong[mask], rng, args.boot),
            "one_minus_reliability": bootstrap_auroc(1.0 - rel[mask], wrong[mask], rng, args.boot),
            "one_minus_q": bootstrap_auroc(1.0 - q[mask], wrong[mask], rng, args.boot),
            "base_rate_wrong": float(wrong[mask].mean()) if mask.any() else float("nan"),
            "base_rate_noop": float(noop[mask].mean()) if mask.any() else float("nan"),
        }
    # Controls.
    perm = rng.permutation(len(rows))
    res["negative_control_shuffled_labels"] = {
        "residual_pixel_ce_real_vs_synth": bootstrap_auroc(ce, real[perm], rng, args.boot),
        "one_minus_reliability_real_vs_synth": bootstrap_auroc(1.0 - rel, real[perm], rng, args.boot),
    }
    res["positive_control"] = {
        "synthetic_reliability": quantiles(rel[synth]),
        "synthetic_composed_exact_rate": float(exact[synth].mean()) if synth.any() else float("nan"),
        "synthetic_residual_pixel_ce": quantiles(ce[synth]),
        # residual on a no-op transition when the decode is exact must be 0 by construction
        "synthetic_exact_rows_have_zero_changed_pixels": bool(
            (cp[synth & exact] == 0).all()) if (synth & exact).any() else None,
    }
    res["reliability_distribution"] = {"real": quantiles(rel[real]), "synthetic": quantiles(rel[synth])}
    res["q_distribution"] = {"real": quantiles(q[real]), "synthetic": quantiles(q[synth])}
    res["residual_distribution"] = {
        "real_pixel_ce": quantiles(ce[real]), "synthetic_pixel_ce": quantiles(ce[synth]),
        "real_changed_pixels": quantiles(cp[real]), "synthetic_changed_pixels": quantiles(cp[synth]),
    }
    res["rates"] = {
        "real_composed_exact": float(exact[real].mean()) if real.any() else None,
        "real_noop": float(noop[real].mean()) if real.any() else None,
        "synthetic_composed_exact": float(exact[synth].mean()) if synth.any() else None,
        "synthetic_noop": float(noop[synth].mean()) if synth.any() else None,
    }
    # Per-game reliability + residual on real.
    games = sorted(set(r["game_id"] for r in rows if r["source"] == "real"))
    per_game = {}
    for g in games:
        m = np.array([r["source"] == "real" and r["game_id"] == g for r in rows])
        mc = m & cexact_defined
        per_game[g] = {"n": int(m.sum()), "reliability_median": float(np.median(rel[m])),
                       "changed_exact_rate": float(cexact[mc].mean()) if mc.any() else None,
                       "changed_rows": int(mc.sum()),
                       "reliability_mean": float(rel[m].mean()),
                       "pixel_ce_median": float(np.median(ce[m])),
                       "composed_exact_rate": float(exact[m].mean()),
                       "noop_rate": float(noop[m].mean())}
    res["per_game_real"] = per_game

    Path(args.out).write_text(json.dumps(res, indent=2))

    def fmt(d):
        return f"{d['auroc']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}] (n={d['n']}, pos={d['n_pos']})"
    print("| Task | Score | AUROC [95% CI] |")
    print("|---|---|---|")
    for task in ("real_vs_synthetic", "real_vs_synthetic_changed_only", "real_vs_synthetic_noop_only",
                 "predict_changed_wrong_within_real", "predict_changed_wrong_within_synthetic",
                 "predict_wrong_within_real", "predict_wrong_within_synthetic"):
        for score, d in res[task].items():
            if isinstance(d, dict) and "auroc" in d:
                print(f"| {task} | {score} | {fmt(d)} |")
    for score, d in res["negative_control_shuffled_labels"].items():
        print(f"| negative_control_shuffled | {score} | {fmt(d)} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
