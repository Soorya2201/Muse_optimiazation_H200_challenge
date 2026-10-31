"""Validate the direct quantizer against high-precision format boundaries."""
import bisect
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import random

import attention_direct_fp8 as a

ROOT = Path(__file__).resolve().parents[1]


def reference(x):
    # Only the zero/subnormal midpoint has a dyadic log2 (x=-10).
    # All other boundaries are irrational and thus cannot equal FP32 x.
    return 0 if x <= -10 else bisect.bisect_right(a.BOUNDARIES, x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'muse_attention_decode/DIRECT_FP8_BRANCHLESS_MATH_2026-10-31.json')
    args = parser.parse_args()
    counts, samples = {}, set()
    def check(label, xs):
        count = 0
        for x in xs:
            q = a.quantize(x)
            assert q == reference(x), (label, x, q, reference(x))
            assert a.quantize_branchless(x) == q, ('branchless', label, x, q)
            assert 0 <= q <= 126
            count += 1
            samples.add(x)
        counts[label] = count

    check('ordered_grid', (a.f32(-12 + 24*i/262144) for i in range(262145)))
    check('boundary_neighbors', (a.value(a.bits(t) + j)
          for t in a.BOUNDARIES for j in range(-16, 17)))
    # All normal provisional-bin change locations (ties included).
    check('normal_bin_neighbors', (a.value(a.bits((k+.5)/8)+j)
          for k in range(-48, 72) for j in range(-16, 17)))
    # All x+384 biased-bin transitions relevant to subnormals, and adjacent
    # representable inputs. RU boundaries are especially important near -10.
    check('subnormal_bias_neighbors', (a.value(a.bits(a.value(b)-384)+j)
          for b in range(a.bits(374.), a.bits(378.)+1) for j in (-1, 0, 1)))
    rng = random.Random(514334)
    check('random_in_domain', (a.f32(rng.uniform(-12, 12)) for _ in range(250000)))
    check('random_finite_bits', (x for x in (a.value(rng.getrandbits(32))
          for _ in range(250000)) if math.isfinite(x)))
    check('tails', [-math.inf, -3.4e38, -1000., -126., -10., -6.,
                    -a.value(1), 0., a.value(1), 8., 9., 126., 3.4e38, math.inf])
    previous = -1
    for x in sorted(samples):
        q = a.quantize(x)
        assert q >= previous, ('monotonic', x, previous, q)
        previous = q
    # Pack order agrees with native e4m3x4f: x0 in low byte. Preserve the
    # parent PB overflow predicate; q<=126 prevents inter-byte carries.
    pack_cases = 10000
    for _ in range(pack_cases):
        xs = [a.f32(rng.uniform(-12, 12)) for _ in range(4)]
        qs = [a.quantize(x) for x in xs]
        word = sum(q << (8*i) for i, q in enumerate(qs))
        assert [(word >> (8*i)) & 255 for i in range(4)] == qs
        assert bool((word + 0x07070707) & 0x80808080) == any(q > 0x78 for q in qs)
    report = dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='pass', counts=counts, unique_samples=len(samples),
        packed_order_and_overflow_cases=pack_cases,
        module_sha256=hashlib.sha256(Path(a.__file__).read_bytes()).hexdigest(),
        cuda_helper_sha256=hashlib.sha256(a.cuda_helper().encode()).hexdigest(),
        cuda_branchless_helper_sha256=hashlib.sha256(a.cuda_branchless_helper().encode()).hexdigest(),
        lut_uint32=[f'0x{a.bits(x):08x}' for x in a.LUT],
        ideal_boundary_ceil_fp32=[x for x in a.BOUNDARIES],
        scope='CPU FP32 emulation vs 80-digit ideal-exp2/RNE E4M3 thresholds; no GPU execution.',
        limitations=['Not a proof over every FP32 bit pattern.',
            'ex2.approx may differ at rounding boundaries; native equivalence unproven.',
            'All-lane CUDA shuffle, resources, attention output, and runtime untested.',
            'NaN logit excluded. No key/value reads or tensor dtypes changed.'])
    path = args.output
    assert not path.exists()
    path.write_text(json.dumps(report, indent=2)+'\n')
    print('PASS', counts, 'unique', len(samples), 'pack', pack_cases)


if __name__ == '__main__':
    main()
