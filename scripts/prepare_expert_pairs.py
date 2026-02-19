#!/usr/bin/env python3
"""
Prepare technical / wiki-like / expert Q&A pairs for world-model training.

Output format (same as UltraChat):
    context<TAB>next_turn

Supported datasets (--dataset):
    sciq       Science Q&A: User: <question> [Context: <support>] -> short answer
    squad      SQuAD-style: User: <question> Context: <paragraph> -> answer span
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def _sanitize(s: str) -> str:
    return s.strip().replace("\t", " ").replace("\n", " ")


def prepare_sciq(
    output_path: Path,
    split: str = "train",
    max_rows: int | None = None,
    include_support: bool = True,
    min_answer_words: int = 1,
) -> tuple[int, int]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets: pip install datasets") from exc

    ds = load_dataset("sciq", split=split)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0

    with output_path.open("w", encoding="utf-8") as out:
        for row in ds:
            if max_rows is not None and written >= max_rows:
                break
            question = _sanitize(str(row.get("question", "")))
            answer = _sanitize(str(row.get("answer", "")))
            support = _sanitize(str(row.get("support", "")))

            if not question or not answer:
                skipped += 1
                continue
            if len(answer.split()) < min_answer_words:
                skipped += 1
                continue

            if include_support and support:
                context = f"User: {question}\nContext: {support}"
            else:
                context = f"User: {question}"

            out.write(f"{context}\t{answer}\n")
            written += 1

    return written, skipped


def prepare_squad(
    output_path: Path,
    split: str = "train",
    max_rows: int | None = None,
    min_answer_words: int = 1,
) -> tuple[int, int]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets: pip install datasets") from exc

    # SQuAD 2.0 has context, question, answers (list of text); some have 'no_answer'
    ds = load_dataset("rajpurkar/squad_v2", split=split)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0

    with output_path.open("w", encoding="utf-8") as out:
        for row in ds:
            if max_rows is not None and written >= max_rows:
                break
            context_para = _sanitize(str(row.get("context", "")))
            question = _sanitize(str(row.get("question", "")))
            answers = row.get("answers", {})
            if isinstance(answers, dict) and "text" in answers:
                texts = answers["text"]
            elif isinstance(answers, list):
                texts = [a.get("text", a) if isinstance(a, dict) else str(a) for a in answers]
            else:
                skipped += 1
                continue

            if not context_para or not question or not texts:
                skipped += 1
                continue

            answer = _sanitize(texts[0]) if texts else ""
            if not answer or len(answer.split()) < min_answer_words:
                skipped += 1
                continue

            context = f"User: {question}\nContext: {context_para}"
            out.write(f"{context}\t{answer}\n")
            written += 1

    return written, skipped


def main() -> None:
    p = argparse.ArgumentParser(
        description="Prepare expert/technical Q&A pairs for world-model training (context TAB next_turn)."
    )
    p.add_argument(
        "--dataset",
        choices=["sciq", "squad"],
        default="sciq",
        help="Dataset to use (default: sciq)",
    )
    p.add_argument("--output", type=Path, default=Path("data/sciq_pairs.txt"), help="Output TSV path")
    p.add_argument("--split", default="train", help="Dataset split (default: train)")
    p.add_argument("--max_rows", type=int, default=None, help="Max pairs to write (default: all)")
    p.add_argument("--no_support", action="store_true", help="(sciq) omit support paragraph from context")
    p.add_argument("--min_answer_words", type=int, default=1, help="Min words in answer (default: 1)")
    args = p.parse_args()

    if args.dataset == "sciq":
        written, skipped = prepare_sciq(
            args.output,
            split=args.split,
            max_rows=args.max_rows,
            include_support=not args.no_support,
            min_answer_words=args.min_answer_words,
        )
    elif args.dataset == "squad":
        written, skipped = prepare_squad(
            args.output,
            split=args.split,
            max_rows=args.max_rows,
            min_answer_words=args.min_answer_words,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"Wrote {written} pairs, skipped {skipped}, output={args.output}")
    print("Train with: cargo run --release -- --train-world", args.output, "... --init-encoder model_latent_*.safetensors")


if __name__ == "__main__":
    main()
