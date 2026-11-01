"""Decode every ILC sample losslessly and audit the fixed predeclared policies."""
import argparse
import base64
import collections
import itertools
import json
import math
from pathlib import Path
import statistics
import struct
import zlib

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('job')
parser.add_argument('--date',required=True)
args=parser.parse_args()
track=Path(__file__).resolve().parents[1]/'muse_bf16_proj'
receipt=json.loads((track/'run_receipts'/f'{args.job}.json').read_text())
lines=receipt['log_tail'].splitlines()
wires=[line.split(' ',1)[1] for line in lines if line.startswith('BF16_ILC_Z ')]
assert len(wires)==1
raw=zlib.decompress(base64.b64decode(wires[0],validate=True))
length=struct.unpack_from('<I',raw)[0];report=json.loads(raw[4:4+length]);offset=4+length
count=0
for row in report['rows']:
    codec=row['time_codec'];assert codec in ('ns16_exact','float32_bits')
    fmt='<32H' if codec=='ns16_exact' else '<32I'
    values=list(struct.unpack_from(fmt,raw,offset));offset+=struct.calcsize(fmt)
    bits=[struct.unpack('<I',struct.pack('<f',v/1e6))[0] for v in values] if codec=='ns16_exact' else values
    ms=[struct.unpack('<f',struct.pack('<I',b))[0] for b in bits]
    assert all(math.isfinite(v) and v>0 for v in ms)
    # Exact event bits are authoritative. Decimal ns are derived, never rounded.
    row['event_ms_bits']=[bits[i:i+8] for i in range(0,32,8)]
    row['event_ns']=[[v*1e6 for v in ms[i:i+8]] for i in range(0,32,8)]
    metrics=struct.unpack_from('<16f3d',raw,offset);offset+=88
    assert all(math.isfinite(x) and x>=0 for x in metrics)
    row.update(normal=list(metrics[:2]),compressed=list(metrics[2:4]),
               fresh=[list(metrics[4:6]),list(metrics[6:8])],
               remaps=[list(metrics[i:i+2]) for i in range(8,16,2)],fp64=list(metrics[16:]))
    count+=32
assert offset==len(raw)
assert report['metric_codec']=='normal2_comp2_fresh4_remap8_f32_then_fp64_3d'
output=track/f'ILC_{args.job}_DECODED_{args.date}.json'
output.write_text(json.dumps(report,indent=2)+'\n')
audit=dict(job=args.job,source_sha256=receipt['source_sha256'],status=report['status'],
           exact_event_values=count,exact_metric_values=len(report['rows'])*19,
           payload_chars=len(wires[0]),all_frames_consumed=True,rows=[],policies=[])
if report['status']=='complete':
    assert len(report['rows'])==32
    keys={(r['pad'],r['seed'],r['id']) for r in report['rows']}
    assert keys==set(itertools.product(range(2),(3209,7909),range(8)))
    assert report['capabilities'][1:]==[1,1,9,0]
    assert [e['bytes'] for e in report['edges']]==[131072,1,15,16,17,65537]
    gates=report['gates'];by_key={};components=[]
    for row in report['rows']:
        assert row['allocation'][-2:]==[0,1]
        a,n,c,b=row['event_ns']
        ratios=dict(actual_over_compressed=[math.sqrt(x*y)/z for x,y,z in zip(a,b,c)],
                    normal_over_compressed=[x/y for x,y in zip(n,c)],
                    actual_over_normal=[math.sqrt(x*y)/z for x,y,z in zip(a,b,n)],
                    actual_a_over_b=[x/y for x,y in zip(a,b)])
        med={k:statistics.median(v) for k,v in ratios.items()}
        numeric=all(v[0]<gates['parent_global'] and v[1]<gates['parent_row']
                    for v in [row['normal'],row['compressed']]+row['fresh']+row['remaps'])
        numeric &= all(x<y for x,y in zip(row['fp64'],(gates['fp64_global'],gates['fp64_row'],gates['onehot'])))
        aa=abs(med['actual_a_over_b']-1)<gates['aa']
        matched=abs(med['actual_over_normal']-1)<gates['matched']
        gain=med['actual_over_compressed']>gates['component_speedup'] and med['normal_over_compressed']>gates['component_speedup']
        item=dict(id=row['id'],pad=row['pad'],seed=row['seed'],ratios=ratios,median=med,
                  median_ns=[statistics.median(v) for v in row['event_ns']],
                  numeric_pass=bool(numeric),aa_pass=aa,matched_pass=matched,component_pass=gain,
                  eligible=bool(numeric and aa and matched and gain))
        audit['rows'].append(item);by_key[(row['pad'],row['seed'],row['id'])]=item
    for i,spec in enumerate(report['specs']):
        rows=[r for r in audit['rows'] if r['id']==i]
        components.append(dict(id=i,layer=spec[0],m=spec[1],
          actual_over_compressed_range=[min(r['median']['actual_over_compressed'] for r in rows),max(r['median']['actual_over_compressed'] for r in rows)],
          normal_over_compressed_range=[min(r['median']['normal_over_compressed'] for r in rows),max(r['median']['normal_over_compressed'] for r in rows)],
          eligible=all(r['eligible'] for r in rows)))
    audit['components']=components
    benchmarks=receipt['benchmarks'];assert len(benchmarks)==30
    totals=collections.defaultdict(float)
    for row in benchmarks:totals[row['case']['load']]+=row['median_ns']*row['case']['weight']
    assert len(totals)==6
    cells={(s[0],s[1]):i for i,s in enumerate(report['specs'])}
    for mask in range(1,256):
        selected=[i for i in range(8) if mask&(1<<i)];proxies=[]
        for pad,seed in itertools.product(range(2),(3209,7909)):
            changed=collections.defaultdict(float)
            for row in benchmarks:
                case=row['case'];i=cells.get((case['layer'],case['m']));ratio=1.
                if i is not None and i in selected:ratio=by_key[(pad,seed,i)]['median']['actual_over_compressed']
                changed[case['load']]+=row['median_ns']*case['weight']/ratio
            proxies.append(math.exp(sum(math.log(totals[k]/changed[k]) for k in totals)/len(totals)))
        component_ok=all(components[i]['eligible'] for i in selected)
        audit['policies'].append(dict(mask=mask,selected=selected,board_proxies=proxies,
           worst_proxy=min(proxies),component_gates_pass=component_ok,
           eligible=component_ok and min(proxies)>=gates['board_speedup']))
    audit['summary']=dict(all_numeric_pass=all(r['numeric_pass'] for r in audit['rows']),
      max_aa_offset=max(abs(r['median']['actual_a_over_b']-1) for r in audit['rows']),
      max_matched_offset=max(abs(r['median']['actual_over_normal']-1) for r in audit['rows']),
      eligible_components=[c['id'] for c in components if c['eligible']],
      eligible_policy_count=sum(p['eligible'] for p in audit['policies']),
      best_changed_policy=max(audit['policies'],key=lambda p:p['worst_proxy']))
    audit['limitations']=['Parent-only official check is not a changed candidate score.',
       'Inside-graph event ratios applied to same-job scored medians are conditional proxies.',
       'Allocation compression flag does not measure actual compression ratio or physical traffic.',
       'All groups and samples retained; gates remain as declared before launch.']
(track/f'ILC_{args.job}_AUDIT_{args.date}.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps({k:audit[k] for k in ('status','exact_event_values','exact_metric_values','payload_chars')},indent=2))
if 'summary' in audit:
 print(json.dumps(audit['components'],indent=2));print(json.dumps(audit['summary'],indent=2))
else:print(report.get('error',report.get('capabilities')))
