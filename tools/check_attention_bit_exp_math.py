"""Independently emulate FP32 bit-exp, domain boundaries and normalized weights."""
import ast
import hashlib
import json
import math
from pathlib import Path
import random
import struct

ROOT = Path(__file__).resolve().parents[1]
candidate = ROOT / 'muse_attention_decode/submission_v24_bit_exp_probe.py'
tree = ast.parse(candidate.read_text())
helper = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
              and isinstance(n.targets[0], ast.Name) and n.targets[0].id == '_AQ_POLY')
assert '383.956329345703125f' in helper and '<< 8' in helper


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def value(x):
    return struct.unpack('<f', struct.pack('<I', x & 0xffffffff))[0]


def approx(x):
    if x > 120:
        return 2. ** x if x < 128 else math.inf
    return value((bits(f32(max(x, -126.) + 383.956329345703125)) << 8) + 0x7f800000)


worst = (0, None)
minimum, maximum = math.inf, -math.inf
previous = 0
count = 262145
for i in range(count):
    x = f32(-125 + 245 * i / (count - 1))
    y = approx(x)
    assert math.isfinite(y) and y > 0 and y >= previous
    previous = y
    error = y / 2. ** x - 1
    minimum, maximum = min(minimum, error), max(maximum, error)
    worst = max(worst, (abs(error), x))
assert worst[0] < .02985, worst
edges = []
for x in [-math.inf, -1e30, -1000., -128., -127., -126., -125.999,
          -2 ** -149, -0., 0., 2 ** -149, 8., 120., value(bits(120.) + 1), 127., math.inf]:
    y = approx(x)
    assert y >= 0 and not math.isnan(y)
    if x <= -126:
        assert y < 2 ** -10  # Below E4M3's half-subnormal quantum, always rounds zero.
    edges.append({'x': str(x), 'y': str(y)})
# Arbitrary FP32 bit-pattern samples in the admissible range, not only a grid.
rng = random.Random(17329)
raw_count = 0
for _ in range(100000):
    x = value(rng.getrandbits(32))
    if not math.isfinite(x) or not -125 <= x <= 120:
        continue
    error = abs(approx(x) / 2. ** x - 1)
    assert error < .02985, (x, error)
    raw_count += 1
# L1 normalized-weight difference: does not bound signed-value output relative
# error, cache quantization, or online-state error; H200 fixtures remain required.
weight_errors = []
for scale in (.01, .5, 2., 8., 40.):
    for offset in (-32., -8., 0., 7.):
        xs = [f32(rng.uniform(-4, 0) * scale + offset) for _ in range(1024)]
        a, b = [2. ** x for x in xs], [approx(x) for x in xs]
        sa, sb = sum(a), sum(b)
        weight_errors.append(sum(abs(x / sa - y / sb) for x, y in zip(a, b)))
report = dict(scope='CPU FP32 emulation only; no H200 performance/output correctness claim.',
    source_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
    helper_sha256=hashlib.sha256(helper.encode()).hexdigest(), grid_samples=count,
    random_bit_samples_in_range=raw_count, normal_range=[-125, 120],
    relative_error_min=minimum, relative_error_max=maximum, worst_abs=worst,
    masked_tail='Clamps to a positive subnormal that rounds to FP8 E4M3 zero; every key still processed.',
    edge_cases=edges, normalized_weight_l1_max=max(weight_errors),
    derivation='At biased x in [256,512), FP32 bit slope is2^15. Shift8 converts it to exponent slope2^23; modulo-add0x7f800000 restores bias127. Minimax delta=log2((1+2/(e ln2))/2).',
    limitations='NaN input is outside finite-input contract. Above120 retains hardware ex2 in CUDA. CPU does not emulate FP8 packing, online max checks, WGMMA or output RMSNorm.')
target = ROOT / 'muse_attention_decode/BIT_EXP_MATH_2026-10-31.json'
assert not target.exists()
target.write_text(json.dumps(report, indent=2) + '\n')
print('PASS', count, 'grid samples;', raw_count, 'random-bit samples; max error', worst)
print('Normalized weight L1 max:', max(weight_errors))
