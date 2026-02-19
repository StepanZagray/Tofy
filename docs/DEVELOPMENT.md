# Development Guide

## Repository conventions

- Keep changes minimal and targeted.
- Avoid unrelated refactors.
- Prefer explicit CLI and module boundaries.

## Local validation

After code changes:

```bash
cargo check --no-default-features
```

With CUDA available:

```bash
cargo check
```

## Where to add new work

- New task pipelines: `src/tasks/`
- New model blocks: `src/model/`
- New data parsing/batching: `src/data/`
- New config: `src/config/`
- Docs: `README.md` and `docs/*.md`

## Agent-oriented notes

- World-model action is fixed to `reply`; extend in `src/tasks/world.rs` and `src/model/world_transition.rs` if you add more actions.
- Decoder boundary: `LocalDecoderRuntime` in `src/model/decoder_runtime.rs`.

## Suggested workflow

1. Implement or change code.
2. Update docs in the same change.
3. Run `cargo check` (and optionally tests).
4. Keep CLI usage strings in README and main.rs accurate.
