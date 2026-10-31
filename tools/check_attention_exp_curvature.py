"""Record the deferred independently derived bit-mantissa curvature form.

Equivalent ideal polynomial family appears in Fast Polynomial Transcendentals
for LLMs, appendix B.1. No coefficient sweep or GPU performance claim.
"""
import ctypes
import hashlib
import json
from pathlib import Path

from attention_direct_fp8 import f32, bits, value

lib = ctypes.CDLL(None)
fma = lib.fmaf
fma.argtypes = [ctypes.c_float] * 3
fma.restype = ctypes.c_float
A = f32(.33976602809582634)


def curved(x):
    biased = f32(x + 384.)
    m = value(((bits(biased) << 8) & 0x7fffff) | 0x3f800000)
    term = fma(m, m, fma(-3., m, 2.))
    corrected = fma(A, term, biased)
    return value((bits(corrected) << 8) + 0x7f800000)


worst, previous = (-1., 0.), 0.
for i in range(65537):
    x = f32(i / 65536)
    y = curved(x)
    assert y >= previous
    previous = y
    worst = max(worst, (abs(y / 2**x - 1), x))
assert curved(0.) == 1. and curved(1.) == 2.
report = dict(scope='CPU one-period FP32/libc-fmaf exploratory check only; no GPU test.',
    status='pass', samples=65537, coefficient=A, worst_relative_error=worst,
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    endpoints=[curved(0.), curved(1.)],
    ideal_equivalent='1 + (1-A)*f + A*f*f, f in [0,1]; not a novel polynomial family.',
    prior_art='https://arxiv.org/html/2610.00049v1#A2.SS1',
    disposition='Deferred without GPU use: accuracy improves on linear interpolation, but extra mantissa extraction/three FMAs supplies no credible cost advantage over the already losing bias-range family.')
target = Path(__file__).resolve().parents[1]/'muse_attention_decode/EXP_CURVATURE_MATH_2026-10-31.json'
assert not target.exists()
target.write_text(json.dumps(report,indent=2)+'\n')
print('PASS',worst,'coefficient',A)
