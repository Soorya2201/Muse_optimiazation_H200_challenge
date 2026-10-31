"""Compute exact TMA weight payloads, with explicitly conditional rate models."""
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
job='f8208a2105ab4940'
p=json.loads((ROOT/'muse_moe_w4a16'/f'PHASE_TRACE_{job}.json').read_text())
assert p['timing_attribution_gate']
fixtures=[]
for t in p['traces']:
    nj=len(t['jobs']);assert nj==len(set(j[0] for j in t['jobs']))
    a=sum(c[1]-c[0] for c in t['metadata']);b=sum(c[3]-c[2] for c in t['metadata'])
    assert a==nj*512 and b==nj*256
    # Canonical constants: four WG pieces, each8192code+1024scale bytes.
    unit_bytes=4*(8192+1024);abytes=a*unit_bytes;bbytes=b*unit_bytes
    assert abytes==nj*2*4096*4096*9//16 and bbytes==nj*4096*4096*9//16
    r=next(r for r in p['rows'] if (r['m'],r['seed'],r['pad_mib'])==(t['m'],t['seed'],t['pad_mib']))
    event_us=statistics.median(math.sqrt(x*y) for x,y in zip(r['ns']['a0'],r['ns']['a1']))/1000
    clocks=t['clocks_ns'];entry=min(c[0] for c in clocks)
    fixtures.append(dict(m=t['m'],seed=t['seed'],pad_mib=t['pad_mib'],jobs=nj,
       phase_a_stages=a,phase_b_stages=b,stage_payload_bytes=unit_bytes,
       phase_a_weight_bytes=abytes,phase_b_weight_bytes=bbytes,
       control_event_us=event_us,end_to_end_weight_payload_tb_s=(abytes+bbytes)/(event_us*1e6),
       entry_to_last_a_end_us=(max(c[2] for c in clocks)-entry)/1000,
       entry_to_last_b_end_us=(max(c[3] for c in clocks)-entry)/1000,
       a_b_overlap_envelope_us=(max(c[2] for c in clocks)-min(c[2] for c in clocks))/1000))
models=[]
for tb_s in (3.2,4.3,4.8):
    times=[(f['phase_a_weight_bytes']+f['phase_b_weight_bytes'])/(tb_s*1e6) for f in fixtures]
    models.append(dict(assumed_weight_rate_tb_s=tb_s,weight_only_target_us=times,
        two_fixtures_other_four_loads_unchanged_proxy=math.prod(f['control_event_us']/v for f,v in zip(fixtures,times))**(1/6)))
out=dict(job=job,status='planning_only',fixtures=fixtures,models=models,limits=[
 'Exact weights payload from actual producer stage partition metadata; no claim of measuring DRAM transactions or counters.',
 'Selected fixtures have one job per expert; other routings can duplicate jobs and require different accounting.',
 'Raw graph event time includes overhead; payload/event quotient is not isolated phase HBM bandwidth.',
 '3.2/4.3TB/s are measured reference stream rates in other experiments, not hard device limits. 4.8TB/s is nominal.',
 'Weight-only target omits activation/routing/output transfers and all compute, synchronization and launch work.',
 'A/B overlap envelope uses consumer boundary timestamps, not producer memory-transaction timestamps.',
 'Two synthetic M32/M256 inputs only; not a board upper bound or a prediction for M128/small loads.'
])
target=ROOT/'muse_moe_w4a16'/f'WEIGHT_PAYLOAD_MODELS_{job}.json';assert not target.exists()
target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
