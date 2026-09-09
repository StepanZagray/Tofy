"""Seal C15 after both analysis processes exit; keep this process outside its tree."""
import datetime
import hashlib
import json
from pathlib import Path

R = Path(__file__).resolve().parent
C = Path((R / 'campaign-path.txt').read_text().strip())
def read(p): return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def save(p, v):
    with Path(p).open('x') as f: json.dump(v, f, indent=2); f.write('\n')
def now(): return datetime.datetime.now().astimezone().isoformat()
def require(ok, message):
    if not ok: raise ValueError(message)

a = read(C / 'analysis.json')
v = read(C / 'independent-review.json')
require(a['accepted'] is True and v['accepted'] is True and v['report_sha256'] == sha(C / 'analysis.json'), 'independent analysis not accepted')
bindings = dict(read(C / 'launch-spec.json')['frozen_files'])
bindings.update(read(R / 'source-freeze.json')['files'])
bindings[str(R / 'source-freeze.json')] = sha(R / 'source-freeze.json')
bindings[str(R / 'seal_completed.py')] = sha(R / 'seal_completed.py')
pids = set()
def scan(o):
    if isinstance(o, dict):
        for key, value in o.items():
            if key in ('pid', 'supervisor_pid', 'model_pid', 'pgid') and isinstance(value, int) and value > 0: pids.add(value)
            if key == 'owned_pids': pids.update(value)
            if key in ('owned_survivors', 'group_survivors'): require(not value, 'owned survivors recorded')
            if key == 'cleanup_error': require(value is None, 'cleanup error recorded')
            scan(value)
    elif isinstance(o, list):
        for value in o: scan(value)
for base in (C, R / 'operations'):
    for p in base.rglob('*.json'):
        if p.name.startswith('outer-seal.'): continue
        if p.name.endswith(('.process.json', '.exit.json')):
            scan(read(p))
            if base != C: bindings[str(p)] = sha(p)
for p in (R / 'operations').glob('*.stdout.log'):
    if not p.name.startswith('outer-seal.'): bindings[str(p)] = sha(p)
for pid in pids: require(not Path(f'/proc/{pid}').exists(), f'process still exists: {pid}')
for path, digest in bindings.items(): require(sha(path) == digest, f'binding changed: {path}')
captures = 0
for name in ('qualify-b4', 'frames-initial', 'frames-final'):
    rows = read(C / f'{name}.bound/summary.json')
    require(len(rows) == 1 and all(x['health']['structurally_valid'] and x['health']['capture_complete'] and x['raw_application_labels_verified'] and x['gpu']['status'] == 'available' and x['gpu']['provenance_binding'] == 'bound' for x in rows), 'incomplete wired capture')
    captures += len(rows)
decision = {arm: value['selector_reuse_supported'] for arm, value in a['gates'].items()}
save(C / 'lifecycle.json', dict(state='complete', classification='completed_exploratory_grounding_evidence', decision=decision, created_local=now(), optimizer_updates=0, all_owned_processes_gone=True, analysis_sha256=sha(C / 'analysis.json'), independent_review_sha256=sha(C / 'independent-review.json')))
files = {}
for path in sorted(C.rglob('*')):
    require(not path.is_symlink(), 'artifact symlink')
    if path.is_file(): files[str(path.relative_to(C))] = dict(bytes=path.stat().st_size, sha256=sha(path))
manifest = dict(schema='tofy-demonstration-grounding-evidence-v1', campaign=str(C), created_local=now(), classification='completed_exploratory_grounding_evidence', source=a['source_revision'], binary_sha256=a['binary_sha256'], decision=decision, files=files, external_bindings=bindings, pids_verified_gone=sorted(pids), cuda_bundles=captures, integrity_scope='Point-in-time verification, not immutable storage.')
path = R / 'completed-campaign.manifest.json'
save(path, manifest)
digest = sha(path)
require({str(p.relative_to(C)) for p in C.rglob('*') if p.is_file()} == set(files), 'inventory changed')
for rel, row in files.items(): require(sha(C / rel) == row['sha256'] and (C / rel).stat().st_size == row['bytes'], 'artifact changed')
for path, expected in bindings.items(): require(sha(path) == expected, 'external binding changed')
(R / 'completed-campaign.manifest.sha256').write_text(digest + '\n')
verification = dict(accepted=True, created_local=now(), manifest_sha256=digest, files=len(files), bytes=sum(x['bytes'] for x in files.values()), bindings=len(bindings), pids_gone=len(pids), cuda_bundles=captures)
save(R / 'completed-campaign-verification.json', verification)
state = read(R / 'operator-state.json')
state.update(state='complete', decision=decision, manifest_sha256=digest, active_processes=[], updated_local=now())
(R / 'operator-state.json').write_text(json.dumps(state, indent=2) + '\n')
print(json.dumps(verification))
