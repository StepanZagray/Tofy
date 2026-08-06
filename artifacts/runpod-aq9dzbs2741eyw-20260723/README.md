# RunPod failure-forensics archive

Pod: `aq9dzbs2741eyw-644122a3@ssh.runpod.io`

Captured: 2026-07-23

The archive contains run logs, TensorBoard event files, JSON reports,
manifests, configuration, and the remote source snapshot needed to audit the
failed experiment. Large model/optimizer checkpoint tensors were deliberately
excluded; the existing qualified checkpoint metrics and paths remain recorded
in the logs and reports.

Archive:

```text
tofy-forensics-aq9dzbs2741eyw-20260723.tar.gz
size:   2,417,216 bytes
sha256: e4e7099b08377549c16001bd277f3e0c13b20e65a58d11236a7e6428ce9b98d9
```

The extracted `run-records/` directory is provided for direct searching.

Full-suite RAG ceiling reports captured after repairing the prompt contract and
executable-call validator:

```text
rewrite-rag-strict-reports.tar.gz
sha256: 1a2232fbbf58ad14f8021170bbea7e946d8b6264b4fbc82070cda2787084abf0

rewrite-rag-reports/rewrite_rag_strict_seen300.json
sha256: 1dd513f25a647322bbccd34c0147f257fc2d73f2fd50430d7708e660a3b44b64

rewrite-rag-reports/rewrite_rag_strict_heldout300.json
sha256: cb40f6512b9a3b4ebc051645e6107b26f6498e4cb179c01b8a40f51c72588e2a
```

The final exact Go AST selector validator was rerun over both 300-task splits.
Its reports were byte-identical to the archived files above (the same two
SHA-256 hashes), confirming that the hardening did not reclassify any task.
