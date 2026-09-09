"""C11 shared immutable input checks; no fitting, generation, or launches."""
import datetime
import hashlib
import json
import math
from pathlib import Path

R = Path(__file__).resolve().parent
REPO = Path('/home/stepan/Projects/code/Tofy-cuda-readout')
RUNS = Path('/home/stepan/Projects/code/.tofy-runs')
RESEARCH = Path('/home/stepan/Research/_runs')
C8 = RUNS / 'looped-frozen-features-20260909T115455-IST'
C9 = RUNS / 'looped-learned-readout-20260909T124221-IST'
R8 = RESEARCH / '2026-09-09T104547Z-tofy-looped-frozen-features'
R9 = RESEARCH / '2026-09-09T113612Z-tofy-looped-learned-readout'
R10 = RESEARCH / '2026-09-09T124206Z-tofy-looped-role-selector-witness'
C10 = R10 / 'evidence-01'
PARENTS = {
    'c8': (R8, 'ebae6f94562879ed343895dc09e9243ab616900f5f8f5e64b54f7ac408cf9e60'),
    'c9': (R9, 'ea0fb1e773400363e52e3b6966600fde5d8e0ffe37a14547e49772d2186fe477'),
    'c10': (R10, '76565de6508b903ff19537a7f63352f1d649debef89abf1773b93096f8b727e8'),
}
CORES = {
    'initial': '4cd502f7a76fe3dd9729f6693e9d707c670972f7ebe58ad1cbdde37ea2bb2802',
    'final': 'a983626a888279cbb31651c6abcd1ebf4f808c8927a29a8349d03ffc35bd4e5a',
}
ARMS = ('c10_true', 'c10_null', 'c9_spatial', 'c9_cls', 'c9_null')
DEPENDENCY = '1ea5cc5c5998a1c2bd388eb158653dd0d1b7452a'
NSYS = '/home/stepan/Projects/code/.tofy-tools/nsight-2026.4.1/opt/nvidia/nsight-systems-cli/2026.4.1/bin/nsys'
LIFECYCLE = C8 / 'supervise.py'
LIFECYCLE_SHA = 'e047b88eed16fb9d8459d82f26a5d5f3ae2dbfc6765ccfa8325c4e9f8ee7de41'
FEATURES = ['cudnn', 'profiling', 'serde_json/float_roundtrip']
FIELDS = ('input_index', 'episode_id', 'partition', 'input_sha256', 'query_sha256', 'label_sha256', 'correct_action')
REGISTRATION_SHA = 'ce687312edae9ed9edb914217d0c6383fca8b6c9ece1c3a72a6d8e12a68d6a2b'

def require(value, reason):
    if not value:
        raise RuntimeError(reason)

def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f'duplicate JSON key: {key}')
        result[key] = value
    return result

def read(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_object,
                      parse_float=finite_float,
                      parse_constant=lambda value: require(False, f'nonfinite JSON: {value}'))

def finite_float(value):
    parsed = float(value)
    require(math.isfinite(parsed), 'nonfinite JSON number')
    return parsed

def moment(value):
    require(type(value) is str, 'timestamp must be text')
    parsed = datetime.datetime.fromisoformat(value)
    require(parsed.utcoffset() is not None, 'timestamp must have a timezone')
    require(parsed <= datetime.datetime.now(datetime.timezone.utc), 'timestamp is in the future')
    return parsed

def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')

def now():
    return datetime.datetime.now().astimezone().isoformat()

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def campaign():
    path = Path((R / 'campaign-path.txt').read_text().strip())
    require(path.parent == RUNS and path.is_dir() and not path.is_symlink(), 'invalid campaign root')
    return path

def verify_files(files):
    require(type(files) is dict, 'file bindings must be an object')
    for filename, sha in files.items():
        require(type(filename) is str and type(sha) is str and len(sha) == 64
                and all(c in '0123456789abcdef' for c in sha), 'invalid file binding')
        path = Path(filename)
        require(path.is_absolute() and path.resolve() == path and path.is_file(), f'nonregular input: {path}')
        require(digest(path) == sha, f'input hash changed: {path}')

def regular_files(root):
    root = Path(root)
    require(root.is_absolute() and root.resolve() == root and root.is_dir(), 'invalid artifact root')
    result = set()
    for path in root.rglob('*'):
        require(not path.is_symlink(), f'symlink in artifact tree: {path}')
        if path.is_file():
            result.add(str(path.relative_to(root)))
        else:
            require(path.is_dir(), f'nonregular artifact: {path}')
    return result

def verified_manifest(root, expected=None):
    root = Path(root)
    path = root / 'manifest.json'
    if expected is None:
        expected = root.with_suffix('.manifest.sha256').read_text().strip()
    verify_files({str(path): expected})
    document = read(path)
    require(regular_files(root) == set(document['files']) | {'manifest.json'}, 'manifest population differs')
    for name, info in document['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'manifest path escape')
        if isinstance(info, dict):
            require(type(info['bytes']) is int and (root / name).stat().st_size == info['bytes'], 'artifact size differs')
        verify_files({str(root / name): info['sha256'] if isinstance(info, dict) else info})
    return document

def verify_parents():
    binds, trees = {}, {}
    for name, (research, expected) in PARENTS.items():
        path = research / 'completed-campaign.manifest.json'
        verify_files({str(path): expected})
        document = read(path)
        root = Path(document['campaign'])
        require(root == {'c8': C8, 'c9': C9, 'c10': C10}[name], 'parent root changed')
        moment(document['created_local'])
        require(regular_files(root) == set(document['files']), 'parent file population differs')
        for rel, info in document['files'].items():
            require(not Path(rel).is_absolute() and '..' not in Path(rel).parts, 'parent path escape')
            require(type(info['bytes']) is int and info['bytes'] >= 0
                    and (root / rel).stat().st_size == info['bytes'], 'parent size differs')
            verify_files({str(root / rel): info['sha256']})
        verify_files(document.get('bindings', {}))
        binds.update(document.get('bindings', {}))
        binds[str(path)] = expected
        trees[name] = dict(root=str(root), files=len(document['files']), manifest_sha256=expected)
    return dict(created_local=now(), parents=trees, frozen_files=binds, accepted=True)
