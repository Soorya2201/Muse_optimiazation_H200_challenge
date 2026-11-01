"""Losslessly decode topology experiment and apply predeclared gates."""
import argparse,base64,itertools,json,math,re,statistics,struct,zlib
from pathlib import Path
root=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('job');a=p.parse_args();track=root/'muse_moe_w4a16';receipt=json.loads((track/'run_receipts'/f'{a.job}.json').read_text());target=track/f'TWO_CTA_REPORT_{a.job}.json';assert not target.exists()
found=re.findall(r'^MOE_TWO_CTA_Z (\S+)$',receipt['log_tail'],re.M);assert len(found)==1,'Complete report missing; keep result unknown'
raw=zlib.decompress(base64.b64decode(found[0]));size=struct.unpack_from('<I',raw)[0];d=json.loads(raw[4:4+size]);assert d['codec']=='u32_ms_bits_32_then_f32_error_fresh_change_13_per_row';assert len(raw)==4+size+180*len(d['rows'])
for i,row in enumerate(d['rows']):
 f=struct.unpack_from('<32I13f',raw,4+size+180*i);row['event_ms_bits']={name:list(f[8*j:8*j+8]) for j,name in enumerate(('a0','old','new','a1'))}
 row['event_us']={name:[struct.unpack('<f',struct.pack('<I',v))[0]*1000 for v in b] for name,b in row['event_ms_bits'].items()}
 for field,offset in [('error',32),('fresh',38)]:row[field]={name:list(f[offset+2*j:offset+2*j+2]) for j,name in enumerate(('old','new','a1'))}
 row['change']=f[44];x=row['event_us'];assert all(len(v)==8 and min(v)>0 and all(math.isfinite(t) for t in v) for v in x.values())
 row['matched_ratio']=statistics.median(o/n for o,n in zip(x['old'],x['new']))
 row['actual_ratio']=statistics.median(math.sqrt(o*b)/n for o,b,n in zip(x['a0'],x['a1'],x['new']))
 row['aa']=statistics.median(o/b for o,b in zip(x['a0'],x['a1']))
 row['control']=statistics.median(o/math.sqrt(b*c) for o,b,c in zip(x['old'],x['a0'],x['a1']))
out=dict(job=a.job,cluster=receipt['cluster'],source_sha256=receipt['source_sha256'],scope='Independent synthetic full MoE calls on same weights/scratch. Official layer is unchanged parent. Conditional policy proxy is not an observed official-board gain.',diagnostic=d,payload_chars=len(found[0]),policies=[])
if d['status']=='complete':
 assert d['stage']=='complete'
 assert len(d['rows'])==24 and len(d['nearby'])==6 and len(d['empty'])==3
 assert set(d['resources'])=={'old','new'} and all(len(r)==18 for r in d['resources'].values())
 assert set(d['fp64'])=={'a0','old','new','a1'} and all(len(e)==2 for e in d['fp64'].values())
 assert {(r['m'],r['mode']) for r in d['nearby']}=={(17,'basis'),(33,'mixed'),(65,'hot'),(129,'mixed'),(257,'mixed'),(384,'hot')}
 assert {r['m'] for r in d['empty']}=={32,128,256}
 assert all(set(r[field])=={'old','new','a1'} and all(len(e)==2 for e in r[field].values()) for r in d['rows']+d['nearby'] for field in ('error','fresh'))
 assert {(r['m'],r['seed'],r['pad_mib']) for r in d['rows']}==set(itertools.product((32,128,256),(3101,7907),(0,1,3,7)))
 out['numeric_gate']=all(all(math.isfinite(x) for x in e) and e[0]<=.002 and e[1]<=.01 for r in d['rows']+d['nearby'] for field in ['error','fresh'] for e in r[field].values()) and all(math.isfinite(r['change']) and r['change']>.001 for r in d['rows']+d['nearby'])
 out['empty_gate']=all(r['zero'] is True and r['fresh_nonzero'] is True for r in d['empty'])
 out['resource_gate']=all(r[off]==r[off+1]==r[off+7]==0 and r[off+3]==0 and r[off+8]>=(2 if name=='new' else 1) for name,r in d['resources'].items() for off in (0,9))
 out['control_gate']=all(abs(r['aa']-1)<=.02 and abs(r['control']-1)<=.03 for r in d['rows'])
 out['fp64_gate']=all(all(math.isfinite(x) for x in e) and e[0]<=.015 and e[1]<=.075 for e in d['fp64'].values())
 out['summaries']=[]
 for m in (32,128,256):
  rows=[r for r in d['rows'] if r['m']==m];summary=dict(m=m)
  for key in ('matched_ratio','actual_ratio','aa','control'):
   vals=[r[key] for r in rows];summary[key]=dict(min=min(vals),median=statistics.median(vals),max=max(vals))
  out['summaries'].append(summary)
 lookup={(r['m'],r['seed'],r['pad_mib']):r for r in d['rows']}
 for mask in range(1,8):
  sizes=[m for i,m in enumerate((32,128,256)) if mask>>i&1];proxies=[]
  for seed,pad in itertools.product((3101,7907),(0,1,3,7)):
   proxies.append(math.prod(min(lookup[(m,seed,pad)]['matched_ratio'],lookup[(m,seed,pad)]['actual_ratio']) for m in sizes)**(1/6))
  component=all(r['matched_ratio']>=1.02 and r['actual_ratio']>=1.02 for r in d['rows'] if r['m'] in sizes)
  qualifies=component and all(out[k] for k in ['numeric_gate','empty_gate','resource_gate','control_gate','fp64_gate']) and min(proxies)>=1.03
  out['policies'].append(dict(token_sizes=sizes,component_gate=component,worst=min(proxies),median=statistics.median(proxies),best=max(proxies),qualifies=qualifies))
 print('GATES',{k:out[k] for k in ['numeric_gate','empty_gate','resource_gate','control_gate','fp64_gate']})
 for s in out['summaries']:print(s)
 print('BEST',max(out['policies'],key=lambda r:r['worst']));print('QUALIFIED',sum(r['qualifies'] for r in out['policies']))
print('DIAGNOSTIC',d['status'],d['stage'],len(d['rows']),d.get('error'))
target.write_text(json.dumps(out,indent=2)+'\n');print(target.relative_to(root))
