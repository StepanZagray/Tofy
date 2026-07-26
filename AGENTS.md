## Scope
- These rules apply to this repository.
- Read this file before making edits.

## Chat
- If the user asks for explanation only, do not modify files.
- Prefer concrete, implementation-level explanations.

## Edits
- Use repository-native tooling and normal editor operations for file changes.
- Keep changes minimal and targeted; avoid unrelated refactors.
- Do not preserve backward compatibility unless the user explicitly asks for it.
- Write as little code as possible while still solving the problem clearly.
- Always use available hardware to its safe maximum without reducing quality.
  For accelerator training, measure the largest stable physical batch, then
  minimize gradient accumulation while preserving the intended effective batch
  and optimizer schedule. Record the selected batch/accumulation pair in the
  experiment results.

## Validation
- After code edits, run lint/compile checks when possible.
- Resolve all errors and warnings.

## Delegation
- Capable primary agents should delegate bounded, independent coding, documentation, analysis, and research work when doing so improves speed or supplies an independent review.
- Use Composer 2.5 through Cursor CLI for focused implementation, test writing, and code review when it is available.
- Use GPT-5.6 Luna High through Codex CLI model `gpt-5.6-luna` with `--config 'model_reasoning_effort="high"'` for bounded analysis, research, documentation, or alternate implementation work when the current Codex provider supports it.
- Keep delegated tasks narrow, safe, and verifiable. The primary agent owns scope, safety, integration decisions, review of every delegated change, and final validation.
- Never claim a delegated model was used if the CLI or account rejected it; record the failure and continue with an available agent or locally.

## RunPod SSH
- RunPod pods may reject non-PTY SSH commands with `Error: Your SSH client doesn't support PTY`.
- To connect on the first try, use the Connect-tab user/host and allocate a TTY:
  `ssh -tt <pod-user>@ssh.runpod.io -i ~/.ssh/id_ed25519`
- After connecting, run inspection commands inside the session instead of using one-shot `ssh ... 'command'`.

## Results Tracking
- Keep `docs/RESULTS.md` up to date.
- When a better metric is reported, update the "Best So Far" section with the new metric and exact command used.
