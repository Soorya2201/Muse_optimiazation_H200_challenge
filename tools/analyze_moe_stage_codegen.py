"""Count complete isolated SASS listings and retain resource/warning evidence."""
import argparse
import collections
import gzip
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--output',default='STAGE_CODEGEN_ANALYSIS_2026-10-31.json');args=parser.parse_args()
assert re.fullmatch('[A-Za-z0-9_.-]+',args.output)
target=ROOT/'muse_moe_w4a16'/args.output;assert not target.exists()
results=[]
for p in sorted((ROOT/'muse_moe_w4a16').glob('STAGE_CODEGEN_*2026-10-31*.json.gz')):
    a=json.load(gzip.open(p,'rt'));d=a['response']
    sass=next((v['asm'] for k,v in d.get('devices',{}).items() if k.startswith('SASS')),[])
    ops=collections.Counter()
    for row in sass:
        s=re.sub(r'^\s*@\S+\s+','',row['text']).strip()
        m=re.match(r'([A-Z][A-Z0-9_]*)(?:\.|\s)',s)
        if m:ops[m[1]]+=1
    stderr='\n'.join(x.get('text','') for x in d.get('stderr',[]))
    results.append(dict(archive=str(p.relative_to(ROOT)),variant=a['variant'],ntt=a['ntt'],regsplit=a.get('regsplit',False),
        code=d.get('code'),sass_truncated=any('[truncated' in x['text'] for x in sass),
        instructions=sum(ops.values()),counts=dict(ops),resources=dict(
          frame_spill_triplets=re.findall(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',stderr),
          registers=re.findall(r'Used (\d+) registers',stderr)),warnings=re.findall(r'\(C75\d+\)',stderr)))
out=dict(scope='Complete isolated phase-A segment SASS listings, not full native instruction counts, dynamic instructions, latency or instruction-cache metrics.',results=results)
target.write_text(json.dumps(out,indent=2)+'\n');print(target.relative_to(ROOT))
for x in results:print(Path(x['archive']).name,x['instructions'],x['counts'].get('HGMMA'),x['resources'],x['warnings'])
