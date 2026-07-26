# Claude guidance

Follow the repository instructions in [`AGENTS.md`](AGENTS.md).

## RunPod deployment

Deploy repository changes through Git: create and review a local commit, push
it, then fetch/pull or check out that exact commit on the pod. Do not send
repository files directly with `scp`, `rsync`, tar-over-SSH, terminal pastes,
or similar mechanisms unless Git transfer is genuinely impossible or a
non-repository artifact is strictly required. State the necessity before
using an exception.
