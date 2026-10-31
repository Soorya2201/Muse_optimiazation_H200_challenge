"""Offline complete compiler archives; distinguish predicates from executed work."""
import collections
import gzip
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1];TRACK=ROOT/'muse_attention_decode'
summaries=[]
for path in sorted(TRACK.glob('INTEGER_EXP*COMPILER*.json.gz')):
    data=json.load(gzip.open(path,'rt'));result=data['response']
    item=dict(file=path.name,compiler=data.get('compiler','nvcc130'),code=result.get('code'),
              archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              source_sha256=hashlib.sha256(data['request']['source'].encode()).hexdigest())
    if result.get('code') is None:
        item['transport_error']=result['transport_error'];summaries.append(item);continue
    assert result['code']==0 and not result['truncated']
    assert not any('truncat' in row['text'].lower() for device in result['devices'].values() for row in device['asm'])
    item['untruncated_device_rows']={name:len(device['asm']) for name,device in result['devices'].items()}
    item['stderr']=[row['text'] for row in result['stderr']]
    rows=next(device['asm'] for name,device in result['devices'].items() if name.startswith('SASS'))
    kernels={};name=None
    for row in rows:
        line=row['text'].strip()
        if line.endswith(':') and not line.startswith('.'):
            name=line[:-1];kernels[name]=[]
        elif name and line and not line.endswith(':'):
            kernels[name].append(line)
    item['kernels']={}
    for name,lines in kernels.items():
        histogram=collections.Counter();predicated=collections.Counter()
        for line in lines:
            words=line.split();pred=words[0].startswith('@');opcode=words[int(pred)]
            histogram[opcode]+=1
            if pred:predicated[opcode]+=1
        item['kernels'][name]=dict(static_sites=len(lines),opcodes=dict(histogram),predicated_opcodes=dict(predicated))
    summaries.append(item)
final=next(x for x in summaries if 'INITIALIZEDFINAL' in x['file'])
assert [final['kernels'][k]['static_sites'] for k in
        ['native_pack32','pred25_pack32','pred375_pack32','pred50_pack32','pred100_pack32']]==[112,192,224,264,408]
for name in ['pred25_pack32','pred375_pack32','pred50_pack32','pred100_pack32']:
    kernel=final['kernels'][name]
    assert kernel['opcodes'].get('SEL',0)==kernel['opcodes'].get('FSEL',0)==0
    assert kernel['opcodes'].get('BSSY',0)==kernel['opcodes'].get('BSYNC',0)==0
    fraction={'pred25_pack32':8,'pred375_pack32':12,'pred50_pack32':16,'pred100_pack32':32}[name]
    assert kernel['predicated_opcodes']['MUFU.EX2']==fraction
    assert kernel['opcodes']['MUFU.EX2']==32
out=dict(scope='Complete standalone compiler evidence, static counts include prologue and padding. No fused resources, runtime attribution or GPU speedup inferred.',archives=summaries)
target=TRACK/'INTEGER_EXP_CODEGEN_ANALYSIS_2026-10-31.json'
target.write_text(json.dumps(out,indent=2)+'\n')
print('PASS',len(summaries),'archives:2terminal502,4complete successful compiles;final 112/192/224/264/408sites')
