"""Independently check FP32 polynomial range boundaries without CUDA."""
import ast
import ctypes
import hashlib
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
paths = [ROOT / ('muse_attention_decode/submission_v' + name) for name in
         ('21_pk2_hybrid_probe.py', '22_pk2_underflow_fix.py')]
polys = []
for path in paths:
    tree = ast.parse(path.read_text())
    poly = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id == '_AQ_POLY')
    assert all(c in poly for c in ('0.0790209f', '0.2249424f', '0.6960656f', '0.9999893f'))
    polys.append(poly)
assert polys[0].replace('-127.f', '-126.f') == polys[1]
fma = ctypes.CDLL(None).fmaf
fma.argtypes = [ctypes.c_float] * 3
fma.restype = ctypes.c_float


def f32(x):
    try:
        return struct.unpack('<f', struct.pack('<f', x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def polynomial(x, lower):
    x = f32(max(x, lower))
    j = math.floor(x)
    f = f32(x - j)
    p = fma(f, f32(.0790209), f32(.2249424))
    p = fma(f, p, f32(.6960656))
    p = fma(f, p, f32(.9999893))
    bits = struct.unpack('<I', struct.pack('<f', p))[0]
    return struct.unpack('<f', struct.pack('<I', (bits + (j << 23)) & 0xFFFFFFFF))[0]


old_masked = polynomial(-math.inf, -127)
new_masked = polynomial(-math.inf, -126)
assert math.isnan(old_masked), 'Reproduce the reported copied-helper defect'
assert math.isfinite(new_masked) and 0 <= new_masked < 2 ** -10
edge = []
for x in (-math.inf, -1000., -200., -127., -126.999, -126.5, -126., -125.999):
    y = polynomial(x, -126)
    assert math.isfinite(y) and 0 <= y < 2 ** -10
    edge.append({'input': str(x), 'corrected_value': y, 'rounds_to_e4m3_zero': True})
maxerr = 0
worst = None
for i in range(65537):
    x = f32(-120 + 240 * i / 65536)
    y = polynomial(x, -126)
    expected = 2.0 ** x
    err = abs(y / expected - 1)
    if err > maxerr:
        maxerr, worst = err, x
    assert math.isfinite(y) and y > 0
assert maxerr < .0002, maxerr
report = {
    'scope': 'CPU FP32 emulation using host C fmaf plus exact bit reconstruction; not GPU correctness or performance proof.',
    'sources': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    'old_negative_infinity_produces_nan': True,
    'corrected_negative_infinity_finite': True,
    'edge_cases': edge,
    'normal_grid_samples': 65537,
    'normal_grid_range': [-120, 120],
    'max_relative_error': maxerr,
    'worst_input': worst,
    'underflow_reason': 'p(0)<1 has biased exponent126; subtracting127 exponent units wraps the exponent into NaN. Lower clamp -126 avoids that wrap. Both true and corrected values quantize to FP8 E4M3 zero at these tiny inputs.',
    'positive_tail': 'Candidate uses original ex2 above120, preserving its overflow/max-update behavior; not part of this polynomial grid.',
}
path = ROOT / 'muse_attention_decode/PK2_MATH_BOUNDARIES_2026-10-30.json'
assert not path.exists(), 'Preserve archived analysis'
path.write_text(json.dumps(report, indent=2) + '\n')
print('PASS: old NaN reproduced, repaired masked/underflow values are finite and round to E4M3 zero.')
print('65537 normal inputs, max relative error:', maxerr, 'at', worst)
print(path.relative_to(ROOT))
