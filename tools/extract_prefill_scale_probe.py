"""Decode exact K64 probe samples and evaluate predeclared fixed policies."""
import argparse
import base64
import itertools
import json
import math
from pathlib import Path
import re
import statistics as stats
import struct
import zlib

p=argparse.ArgumentParser(description=__doc__);p.add_argument('job');args=p.parse_args()
root=Path(__file__).resolve().parents[1];folder=root/'muse_prefill_layer'
r=json.loads((folder/'run_receipts'/f'{args.job}.json').read_text())
matches=re.findall(r'^PREFILL_SCALE_SHARE_Z (\S+)$',r['log_tail'],re.M)
assert len(matches)==1,'Require one intact report; never infer success from missing diagnostics'
wire=zlib.decompress(base64.b64decode(matches[0]));size=struct.unpack_from('<I',wire)[0]
d=json.loads(wire[4:4+size]);buf=wire[4+size:]
assert d['timing_codec']=='u32_ns_rows_a0_old_new_a1_8rounds'
count=len(d['rows'])*4*8;assert len(buf)==count*4
times=iter(struct.unpack('<'+str(count)+'I',buf))
for row in d['rows']:
    row['ns']={k:[next(times) for _ in range(8)] for k in ('a0','old','new','a1')}
    assert all(v>0 for a in row['ns'].values() for v in a)
    t=row['ns'];z=list(zip(t['a0'],t['old'],t['new'],t['a1']))
    row['ratios']=dict(old_new=stats.median(b/c for a,b,c,e in z),
        parent_new=stats.median(math.sqrt(a*e)/c for a,b,c,e in z),
        parent_old=stats.median(math.sqrt(a*e)/b for a,b,c,e in z),
        aa=stats.median(a/e for a,b,c,e in z))

out=dict(job=args.job,cluster=r['cluster'],source_sha256=r['source_sha256'],
    scope='Independent synthetic expert-chain fixtures. Timed whole layer is unchanged parent. Phase-duration proxy is not an observed board gain or critical-path bound.',
    payload_chars=len(matches[0]),diagnostic=d,summaries=[],cells=[],policies=[])
if d['status']=='complete':
    assert r['summary']['passed'] and len(d['nearby'])==5
    assert {(x['m'],x['seed'],x['pad_mib']) for x in d['rows']}==set(itertools.product((753,3012,16384),(3208,7911),(0,1,3,7)))
    assert d['movement_widths']==[32,64,96,128,160,192,224,256]
    for row in d['rows']+d['nearby']:
        assert row['change']>.001
        for field in ('error','fresh'):
            for error in row[field].values():
                assert len(error)==2 and all(math.isfinite(v) for v in error)
                assert error[0]<=.002 and error[1]<=.01
    resources=d['resources'];no_new_local=True
    for off in (0,5):
        assert resources['new'][off]==resources['old'][off]==0
        no_new_local &= resources['new'][off+2]<=resources['old'][off+2]
    for m in (753,3012,16384):
        rows=[x for x in d['rows'] if x['m']==m]
        s=dict(m=m,n=len(rows))
        for k in ('old_new','parent_new'):
            v=[x['ratios'][k] for x in rows]
            s[k]=dict(min=min(v),median=stats.median(v),max=max(v))
        for key in ('aa','parent_old'):
            s[key+'_max_offset']=max(max(x['ratios'][key],1/x['ratios'][key])-1 for x in rows)
        s['qualifies']=(no_new_local and s['old_new']['min']>1.02
            and s['aa_max_offset']<.02 and s['parent_old_max_offset']<.03)
        out['summaries'].append(s)
    profiles={x['index']:x for x in r['profiles']}
    for i,b in enumerate(r['benchmarks']):
        profile=profiles[i];assert profile['case']==b['case']
        values=[float(match[1]) for line in profile['report'].splitlines()
            if 'moe_fused_persistent' in line
            for match in [re.search(r'\s([\d.]+) µs\s',line)] if match]
        assert len(values)==1
        c=b['case'];out['cells'].append(dict(case=c,m=c['batch']*c['q_len'],
            us=b['median_ns']/1000,expert_us=values[0]))
    assert len(out['cells'])==8
    groups={(x['m'],x['seed'],x['pad_mib']):x for x in d['rows']}
    eligible={s['m'] for s in out['summaries'] if s['qualifies']}
    for mask in range(1,8):
        chosen={m for i,m in enumerate((753,3012,16384)) if mask>>i&1};factors=[]
        for seed,pad in itertools.product((3208,7911),(0,1,3,7)):
            loads={}
            for cell in out['cells']:
                c=cell['case'];saving=0
                if cell['m'] in chosen:
                    q=groups[cell['m'],seed,pad]['ratios']
                    # Conservative agreement of matched and actual-parent comparisons.
                    ratio=min(q['old_new'],q['parent_new'])
                    saving=cell['expert_us']*(1-1/ratio)
                before,after=loads.get(c['load'],(0,0))
                remaining=cell['us']-saving;assert remaining>0
                loads[c['load']]=(before+cell['us']*c['weight'],after+remaining*c['weight'])
            factors.append(math.prod(a/b for a,b in loads.values())**(1/len(loads)))
        out['policies'].append(dict(token_sizes=sorted(chosen),worst=min(factors),
            median=stats.median(factors),best=max(factors),component_gates=chosen<=eligible,
            qualifies=chosen<=eligible and min(factors)>=1.03))
path=folder/f'SCALE_SHARE_REPORT_{args.job}.json';assert not path.exists()
path.write_text(json.dumps(out,indent=2)+'\n')
print('DIAGNOSTIC',d['status'],d.get('stage'),d.get('seconds'),'payload',out['payload_chars'])
print('RESOURCES',d.get('resources'))
if d['status']!='complete':print('ERROR',d.get('error'),d.get('trace'))
for s in out['summaries']:print(json.dumps(s))
if out['policies']:
    print('BEST_WORST',max(out['policies'],key=lambda x:x['worst']))
    print('QUALIFIED',sum(x['qualifies'] for x in out['policies']))
print(path.relative_to(root))
