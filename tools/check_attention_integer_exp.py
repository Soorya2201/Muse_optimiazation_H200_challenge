"""Offline FP32/integer arithmetic and E4M3 boundary sampling, not attention proof."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random

import attention_integer_exp as model
from attention_direct_fp8 import bits, f32, value, decode

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',default='muse_attention_decode/INTEGER_EXP_MATH_2026-10-31.json')
args=parser.parse_args()
target=Path(args.output)
assert not target.exists()

# Closed integer-domain coverage, including period endpoint and carry.
max_product=max_correction=0
previous=0.
worst=(0.,0.)
for b in range(32769):
    x=b/32768
    y=model.integer_exp(x)
    assert y>=previous
    previous=y
    worst=max(worst,(abs(y/2.**x-1),x))
    if b<32768:
        p=b*(32768-b);c=(p*model.COEFFICIENT)>>32
        assert 0<=p<1<<32 and c<1<<23
        max_product=max(max_product,p);max_correction=max(max_correction,c)
assert model.integer_exp(0)==1 and model.integer_exp(1)==2

samples={bits(f32(-126+246*i/262144)) for i in range(262145)}
# Neighbors around all whole-binade seams and all finite positive E4M3 midpoints.
for integer in range(-126,121):
    u=bits(float(integer))
    for offset in range(-8,9):
        candidate=value(u+offset)
        if math.isfinite(candidate) and -126<=candidate<=120:samples.add(bits(candidate))
for code in range(1,127):
    x=f32(math.log2((decode(code-1)+decode(code))/2))
    u=bits(x)
    for offset in range(-16,17):samples.add((u+offset)&0xffffffff)
rng=random.Random(0xC0A7)
for _ in range(200000):
    x=value(rng.getrandbits(32))
    if math.isfinite(x) and -126<=x<=120:samples.add(bits(x))

for u in samples:
    x=value(u);y=model.integer_exp(x)
    assert math.isfinite(y) and y>0
    worst=max(worst,(abs(y/2.**x-1),x))
assert worst[0]<.0028,worst
assert model.integer_exp(-math.inf)==model.integer_exp(-126)
scales=[]
for e in range(-125,120):
    for b in (0,1,511,8192,16384,24576,32767):
        f=b/32768
        assert model.integer_exp(e+f)==math.ldexp(model.integer_exp(f),e)
        scales.append((e,b))
out=dict(scope='CPU scalar/core-integer coverage and FP32 sampled domain. Not native ex2.approx equivalence, complete FP32 coverage, attention-output validation or GPU timing.',
    status='pass',coefficient_A=model.A,integer_coefficient=model.COEFFICIENT,
    coefficient_interpretation='correction=floor(b*(32768-b)*round(A*2**25)/2**32)',
    full_fraction_states=32769,additional_distinct_fp32_samples=len(samples),
    worst_relative_error=worst,max_product=max_product,max_correction=max_correction,
    monotonic_on_all_fraction_states=True,endpoints=[model.integer_exp(0),model.integer_exp(1)],
    exact_scale_equivariance_cases=len(scales),
    masked_negative_infinity='clamped to 2**-126; rounds to zero under native E4M3 conversion, with no K/V skipping',
    high_tail='Caller must retain native ex2 for x>120; not covered by the core CPU model',
    model_sha256=hashlib.sha256(Path(model.__file__).read_bytes()).hexdigest(),
    prior_art='https://arxiv.org/html/2610.00049v1#A2.SS1')
target.write_text(json.dumps(out,indent=2)+'\n')
print('PASS',out['additional_distinct_fp32_samples'],'FP32 samples,32769fraction states,worst',worst,'coefficient',model.COEFFICIENT)
