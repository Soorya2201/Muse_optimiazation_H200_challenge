"""Check conversion-free range reduction against the repaired FP32 polynomial."""
import ctypes
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
probe = ROOT / 'muse_attention_decode/exp_range_compiler_probe.cu'
source = probe.read_text()
assert 'float j=floorf(x);' in source and '__fadd_rd(x,12582912.f)' in source
fma = ctypes.CDLL(None).fmaf
fma.argtypes = [ctypes.c_float] * 3
fma.restype = ctypes.c_float


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def value(x):
    return struct.unpack('<f', struct.pack('<I', x & 0xFFFFFFFF))[0]


def add_down(a, b):
    exact = Fraction(a) + Fraction(b)
    rounded = bits(f32(float(exact)))
    # This use has positive finite sums near the bias. Rational comparisons
    # independently enforce directed rounding even for subnormal x values.
    assert 0 < rounded < 0x7F7FFFFF
    while Fraction(value(rounded)) > exact:
        rounded -= 1
    while Fraction(value(rounded + 1)) <= exact:
        rounded += 1
    assert Fraction(value(rounded)) <= exact < Fraction(value(rounded + 1))
    return value(rounded)


def poly(f):
    p = fma(f, f32(.0790209), f32(.2249424))
    p = fma(f, p, f32(.6960656))
    return fma(f, p, f32(.9999893))


def compare(raw):
    x = f32(max(raw, -126))
    assert x <= 120
    j = math.floor(x)
    f = f32(x - j)
    old = value(bits(poly(f)) + (j << 23))
    t = add_down(x, 12582912.)
    j_new = f32(t - 12582912.)
    f_new = f32(x - j_new)
    new = value(bits(poly(f_new)) + (bits(t) << 23))
    assert j_new == j and bits(f_new) == bits(f) and bits(new) == bits(old), raw
    assert math.isfinite(new)


count = 0
for i in range(65537):
    compare(f32(-126 + 246 * i / 65536))
    count += 1
for integer in range(-126, 121):
    q = bits(float(integer))
    for delta in (-2, -1, 0, 1, 2):
        x = value(q + delta)
        if math.isfinite(x) and x <= 120:
            compare(x)
            count += 1
for x in (-math.inf, -1000., -200., -127., -126.999, -126.5,
          -2 ** -149, 2 ** -149, -1e-30, 1e-30, -0., 0.):
    compare(x)
    count += 1

comp = json.loads((ROOT / 'muse_attention_decode/EXP_RANGE_COMPILER_PROBE_2026-10-30.json').read_text())
assert comp['response']['code'] == 0
device = comp['response']['devices']['SASS (compute_90a, sm_90a)']
asm = '\n'.join(row['text'] for row in device['asm'])
before, after = asm.split('old_floor:', 1)
new_sass = before.split('bias_floor:', 1)[1]
assert 'FRND.FLOOR' in after and 'F2I.FLOOR' in after
assert 'FRND' not in new_sass and 'F2I' not in new_sass and 'FADD.RM' in new_sass
report = {
    'scope': 'CPU FP32/rational rounding equivalence plus standalone Compiler Explorer SASS; not full-kernel performance evidence.',
    'source_sha256': hashlib.sha256(probe.read_bytes()).hexdigest(),
    'compared_inputs': count,
    'all_bit_identical': True,
    'range': '[-126,120] plus clamped negative tails, integer neighbors, signed zero and subnormals',
    'rounding_reference': 'Exact rational sums bracketed by adjacent FP32 numbers, toward minus infinity',
    'compiler': 'NVCC13.0, sm90a, -O3; device SASS inspected, not host assembly',
    'standalone_old': 'FRND.FLOOR + F2I.FLOOR + FADD',
    'standalone_new': 'FADD.RM + FADD + FADD; no FRND/F2I',
    'standalone_resources': '10 registers and zero stack/spill bytes for both kernels',
    'limitation': 'Same arithmetic-site count. Different instruction classes may change bottlenecks; no H200 latency gain or shared-unit causal claim established. Larger native attention resource/scheduling behavior must be measured.',
    'prefill_counterevidence': 'Existing prefill exp2_poly already uses a centered add-bias trick; its rejected hybrids are not explained by floor-conversion overhead. Do not reopen prefill solely on this finding.',
}
path = ROOT / 'muse_attention_decode/EXP_RANGE_EQUIVALENCE_2026-10-30.json'
assert not path.exists()
path.write_text(json.dumps(report, indent=2) + '\n')
print('PASS:', count, 'CPU cases bit-identical; standalone sm90a SASS conversion change verified.')
print(path.relative_to(ROOT))
