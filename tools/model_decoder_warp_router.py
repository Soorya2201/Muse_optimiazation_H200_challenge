"""Independent CPU check of 32-lane stable-head routing and decoder time scenarios.

Selection layout/finite-key semantics only. Does not execute downloaded code,
model CUDA ex2/div rounding, prove GPU correctness, or predict runtime.
Primary implementation: vLLM PR64553, merge36ed4327 (Apache-2.0).
"""
import collections
import hashlib
import json
import math
from pathlib import Path
import random
import re
import struct

root=Path(__file__).resolve().parents[1]

def model(keys,scores):
    assert len(keys)==len(scores)==256 and all(math.isfinite(k) for k in keys)
    lanes=[]
    for lane in range(32):
        slots=[(keys[v*128+lane*4+j],v*128+lane*4+j,scores[v*128+lane*4+j])
               for v in range(2) for j in range(4)]
        for phase in range(8):
            for i in range(phase%2,7,2):
                if slots[i+1][0]>slots[i][0]:slots[i],slots[i+1]=slots[i+1],slots[i]
        assert slots==sorted(slots,key=lambda x:(-x[0],x[1]))
        lanes.append(slots)
    chosen=[]
    for k in range(8):
        maximum=max(a[0][0] for a in lanes)
        # Pick lowest nonempty 128-expert group, then its lowest winning lane.
        heads=[(a[0][1]//128,lane) for lane,a in enumerate(lanes) if a[0][0]==maximum]
        group,lane=min(heads);winner=lanes[lane][0];chosen.append(winner)
        assert winner[1]==min(a[0][1] for a in lanes if a[0][0]==maximum)
        for i in range(7):
            if i+k<=6:lanes[lane][i]=lanes[lane][i+1]
    expected=sorted([(keys[i],i,scores[i]) for i in range(256)],key=lambda x:(-x[0],x[1]))[:8]
    assert chosen==expected
    return [x[1] for x in chosen]

rng=random.Random(0x64553);cases=[]
for value in (0.,-0.,1.,-1.,1e30,-1e30):cases.append(([value]*256,'all_tied'))
for lane in range(32):
    keys=[-1.]*256
    for v in range(2):
        for j in range(4):keys[v*128+lane*4+j]=100.-(v*4+j)
    cases.append((keys,'all_eight_from_one_lane'))
for size in (2,4,8,16,32,64,128,256):cases.append(([float(i%size) for i in range(256)],'tie_groups'))
for _ in range(4096):
    mode=rng.randrange(4)
    if mode==0:keys=[rng.uniform(-100,100) for _ in range(256)]
    elif mode==1:keys=[float(rng.randrange(8)) for _ in range(256)]
    elif mode==2:keys=[float(rng.randrange(-1,2)) for _ in range(256)]
    else:keys=[struct.unpack('<f',struct.pack('<f',.5+rng.randrange(-8,9)*2.**-24))[0] for _ in range(256)]
    cases.append((keys,('random','eight_levels','three_levels','adjacent_fp32')[mode]))
for keys,kind in cases:
    scores=[rng.random() for _ in range(256)];ids=model(keys,scores)
    assert len(set(ids))==8

receipt=json.loads((root/'muse_decode_layer/run_receipts/eff0b59dc26c4c74.json').read_text())
routes={};pattern=re.compile(r'^(?:_gle_route|_sl_tile_route|_graft_route)\s+([0-9.]+) µs')
for row in receipt['profiles']:
    matches=[float(m.group(1)) for line in row['report'].splitlines() if (m:=pattern.match(line))]
    assert len(matches)==1
    case=row['case'];routes[(case['load'],case['layer'])]=matches[0]
assert len(routes)==12
scenarios={}
for cut in (.25,.5,1.):
    before=collections.defaultdict(float);after=collections.defaultdict(float)
    for row in receipt['benchmarks']:
        c=row['case'];us=row['median_ns']/1000;duration=routes[(c['load'],c['layer'])]
        assert us>duration
        before[c['load']]+=us*c['weight'];after[c['load']]+=(us-cut*duration)*c['weight']
    scenarios[str(cut)]=math.exp(sum(math.log(before[k]/after[k]) for k in before)/len(before))
output=dict(scope=__doc__,selection_cases=len(cases),case_kinds=dict(collections.Counter(k for _,k in cases)),
             exact_key_comparisons=True,score_associations_preserved=True,
             primary_url='https://github.com/vllm-project/vllm/blob/36ed4327d2caeb38543b75b80db2a291e88883bc/csrc/moe/muse_spark_routing.cu',
             profile_job=receipt['job'],source_sha256=receipt['source_sha256'],
             router_profile_us=[dict(load=k[0],layer=k[1],us=v) for k,v in routes.items()],
             conditional_board_proxy_for_route_profile_duration_cut=scenarios,
             limitations=['Profile duration includes waits and work other than top8; subtraction is optimistic planning, not a rigorous critical-path bound.',
               'Synthetic event time, full device numerical behavior, and adapted warp resources remain unmeasured.',
               'Current routing also normalizes expert inputs, maps device-local IDs and sometimes carries residuals; standalone selector cannot replace these.',
               'Hopper uses five shuffle steps for FP32 warp max; SM100 family redux instruction is not an H200 optimization.'])
p=root/'muse_decode_layer/WARP_ROUTER_SCREEN_2026-11-01.json';p.write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
