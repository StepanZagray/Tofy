"""Independent F32 replay of the fixed C18 equivariant binder; no Rust imports."""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import hashlib
import json
from pathlib import Path
import struct
import numpy as np

CHECKPOINT_SHA256 = 'd2deeba7b0fafc2386c9a19d39bc99b7534b91a0155d66e77792a5c4037bd717'
PARAMETER_SHA256 = '86dd13d3998598d54acc1d97167c75d09d1d00f12b29b9073154d19a29a9e491'
ATOL, RTOL = 1e-4, 1e-5


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def expected_shapes():
    result = {'input_projection.weight': (256, 5), 'input_projection.bias': (256,),
              'policy_head.weight': (1, 256), 'policy_head.bias': (1,)}
    for layer in range(2):
        for name, shape in [('attention.' + k, (256, 256)) for k in ('query', 'key', 'value', 'output')] + [('mlp_in', (1024, 256)), ('mlp_out', (256, 1024))]:
            result[f'block_{layer}.{name}.weight'] = shape
            result[f'block_{layer}.{name}.bias'] = (shape[0],)
    return result


def validate_weights(weights):
    shapes = expected_shapes()
    require(set(weights) == set(shapes), 'checkpoint tensor names')
    for name, shape in shapes.items():
        value = weights[name]
        require(value.shape == shape and value.dtype == np.dtype('<f4') and np.isfinite(value).all(), 'checkpoint tensor shape/dtype/finiteness: ' + name)


def parameter_digest(weights):
    digest = hashlib.sha256(b'looped-action-binding-parameters-v1\0')
    for name, value in sorted(weights.items()):
        encoded = name.encode()
        digest.update(struct.pack('<Q', len(encoded)) + encoded)
        digest.update(struct.pack('<Q', value.ndim))
        for size in value.shape:
            digest.update(struct.pack('<Q', size))
        digest.update(struct.pack('<Q', value.size))
        digest.update(value.astype('<f4', copy=False).tobytes())
    return digest.hexdigest()


def load_checkpoint(path, expected_sha256=CHECKPOINT_SHA256):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and not path.is_symlink(), 'checkpoint path')
    require(path.stat().st_size < 8_000_000, 'checkpoint size')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_sha256, 'checkpoint hash')
    require(len(raw) >= 8, 'checkpoint prefix')
    length = struct.unpack('<Q', raw[:8])[0]
    require(0 < length < 32_768 and 8 + length <= len(raw), 'checkpoint header length')
    header = json.loads(raw[8:8 + length], object_pairs_hook=unique_object)
    require(isinstance(header, dict), 'checkpoint header')
    header.pop('__metadata__', None)
    require(set(header) == set(expected_shapes()), 'checkpoint names')
    payload, end, weights = memoryview(raw)[8 + length:], 0, {}
    for name, item in sorted(header.items(), key=lambda pair: pair[1]['data_offsets'][0]):
        require(set(item) == {'dtype', 'shape', 'data_offsets'}, 'checkpoint fields')
        offsets, shape = item['data_offsets'], item['shape']
        require(isinstance(offsets, list) and len(offsets) == 2 and all(type(x) is int for x in offsets), 'checkpoint offsets')
        require(isinstance(shape, list) and all(type(x) is int and x > 0 for x in shape), 'checkpoint dimensions')
        require(item['dtype'] == 'F32' and tuple(shape) == expected_shapes()[name], 'checkpoint dtype/shape')
        begin, stop = offsets
        require(begin == end and stop - begin == 4 * int(np.prod(shape)) and stop <= len(payload), 'checkpoint offsets/size')
        weights[name] = np.frombuffer(payload[begin:stop], dtype='<f4').reshape(shape).copy()
        end = stop
    require(end == len(payload), 'checkpoint trailing bytes')
    validate_weights(weights)
    if expected_sha256 == CHECKPOINT_SHA256:
        require(parameter_digest(weights) == PARAMETER_SHA256, 'checkpoint parameter digest')
    return weights


def rms(x):
    return x / np.sqrt(np.maximum(np.mean(x * x, axis=-1, keepdims=True, dtype=np.float32), np.float32(1e-10)))


def linear(x, weights, name):
    return x @ weights[name + '.weight'].T + weights[name + '.bias']


def block(x, weights, name):
    batch, tokens, _ = x.shape
    z = rms(x)
    q, k, v = [linear(z, weights, name + '.attention.' + item).reshape(batch, tokens, 4, 64).transpose(0, 2, 1, 3) for item in ('query', 'key', 'value')]
    scores = (q @ k.swapaxes(-1, -2)) * np.float32(0.125)
    exponential = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attention = exponential / exponential.sum(axis=-1, keepdims=True, dtype=np.float32)
    attended = (attention @ v).transpose(0, 2, 1, 3).reshape(batch, tokens, 256)
    state = x + linear(attended, weights, name + '.attention.output')
    hidden = linear(rms(state), weights, name + '.mlp_in')
    # Stable SiLU, retaining F32 throughout and without an exponential overflow.
    exp_negative_abs = np.exp(-np.abs(hidden))
    sigmoid = np.where(hidden >= 0, 1 / (1 + exp_negative_abs), exp_negative_abs / (1 + exp_negative_abs))
    return state + linear(hidden * sigmoid, weights, name + '.mlp_out')


def replay(records, weights, loops=4, chunk_size=32):
    validate_weights(weights)
    x = np.asarray(records, dtype=np.float32)
    require(x.ndim == 3 and x.shape[1:] == (4, 7) and len(x) > 0 and np.isfinite(x).all(), 'binder records')
    require(type(loops) is int and 1 <= loops <= 8 and type(chunk_size) is int and chunk_size > 0, 'binder depth/chunk')
    result = []
    for start in range(0, len(x), chunk_size):
        part = x[start:start + chunk_size]
        a = part[:, :3, 2:6].transpose(0, 2, 1)
        tokens = np.concatenate((a @ part[:, :3, :2], a.sum(axis=2, keepdims=True), np.broadcast_to(part[:, 3:4, :2], (len(part), 4, 2))), axis=2)
        recalled = linear(tokens, weights, 'input_projection')
        state = np.zeros_like(recalled)
        for _ in range(loops):
            state = state + recalled
            for layer in range(2):
                state = block(state, weights, f'block_{layer}')
        result.append(linear(rms(state), weights, 'policy_head')[:, :, 0])
    output = np.concatenate(result)
    require(np.isfinite(output).all(), 'nonfinite replay logits')
    return output


def parity(actual, reference):
    a, r = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    require(a.shape == r.shape and a.size > 0 and np.isfinite(a).all() and np.isfinite(r).all(), 'parity arrays')
    difference = np.abs(a - r)
    ratio = difference / (ATOL + RTOL * np.abs(r))
    require((ratio <= 1).all(), 'numerical replay tolerance')
    return {'maximum_absolute_error': float(difference.max()), 'maximum_tolerance_ratio': float(ratio.max())}


def logit_parity(actual, reference):
    result = parity(actual, reference)
    a, r = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    require(a.ndim == 2 and a.shape[1] == 4, 'logit shape')
    margins = np.sort(r, axis=1)[:, -1] - np.sort(r, axis=1)[:, -2]
    discrepancy = np.abs(a - r).max(axis=1)
    eligible = (margins > 0) & (margins > 2 * discrepancy)
    mismatches = eligible & (a.argmax(axis=1) != r.argmax(axis=1))
    require(not mismatches.any(), 'eligible replay winner mismatch')
    return result | {'rows': len(a), 'eligible_winners': int(eligible.sum()), 'ineligible_winners': int((~eligible).sum()), 'eligible_winner_mismatches': int(mismatches.sum()), 'eligibility': 'reference winner margin > 0 and > 2 * row maximum absolute logit error'}
