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
- Make sure program uses maximum hardware capabilities, for speed without affecting quality.

## Validation
- After code edits, run lint/compile checks when possible.
- Resolve all errors and warnings.

## RunPod SSH
- RunPod pods may reject non-PTY SSH commands with `Error: Your SSH client doesn't support PTY`.
- To connect on the first try, use the Connect-tab user/host and allocate a TTY:
  `ssh -tt <pod-user>@ssh.runpod.io -i ~/.ssh/id_ed25519`
- After connecting, run inspection commands inside the session instead of using one-shot `ssh ... 'command'`.

## Results Tracking
- Keep `docs/RESULTS.md` up to date.
- When a better metric is reported, update the "Best So Far" section with the new metric and exact command used.
