"""Exhaust a narrow 7-op rotate/byte-permute decoder family; not all circuits."""
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import re

root=Path(__file__).resolve().parents[1]
mask=0x81c081c0
def rotate(value,shift):return ((value<<shift)|(value>>(32-shift if shift else 32)))&0xffffffff

# r0=q&M; t1=rot(q,a); r1=t1&M; t2=rot(q,b); r2=t2&M;
# r3=byte_perm(u,v,selector)&M. Seven nominal operations when u/v are
# q or existing t1/t2. Every arbitrary code bit must be distinct/lossless.
triples=[];examined=0;solutions=[]
for a,b in itertools.combinations(range(1,32),2):
    ma,mb=rotate(mask,32-a),rotate(mask,32-b)
    if mask&ma or mask&mb or ma&mb:continue
    free=(~(mask|ma|mb))&0xffffffff;triples.append(dict(a=a,b=b,free_mask=hex(free)))
    for u,v in itertools.combinations((0,a,b),2):
        patterns=[]
        for dst in range(4):
            choices=[]
            for selector in range(8):
                shift=u if selector<4 else v;srcbyte=selector%4;needed=0
                for bit in range(8):
                    if mask>>(dst*8+bit)&1:needed|=1<<((srcbyte*8+bit-shift)%32)
                choices.append(needed)
            patterns.append(choices)
        for selectors in itertools.product(range(8),repeat=4):
            examined+=1;needed=0
            for dst,selector in enumerate(selectors):needed|=patterns[dst][selector]
            if needed==free:solutions.append(dict(a=a,b=b,u=u,v=v,selectors=selectors))
assert len(triples)==36 and examined==442368 and not solutions
path=root/'muse_moe_w4a16/STAGE_CODEGEN_parent_NTT8_2026-10-31.json.gz'
d=json.load(gzip.open(path,'rt'));response=d['response'];assert response['code']==0
lines=[row['text'] for row in response['devices']['SASS']['asm']]
assert not any('truncated' in row for row in lines)
prmts=[row.strip() for row in lines if re.search(r'\bPRMT\b',row)]
assert len(prmts)==8
out=dict(scope='Exact bit-position feasibility for a restricted seven nominal-operation family, plus existing reduced compiler inspection. No kernel executed, no exhaustive optimality claim.',
    source_sha256=hashlib.sha256((root/'muse_moe_w4a16/submission.py').read_bytes()).hexdigest(),
    mask=hex(mask),rotations_considered=465,disjoint_rotation_triples=triples,
    byte_select_combinations_examined=examined,solutions=solutions,
    exclusions=['Other instruction sets/circuits and different BF16 encodings are outside this search.',
        'PRMT sign-replication cannot yield two independent desired bits in each destination byte, so excluded.',
        'No operation count is turned into a speedup estimate.'],
    existing_scale_hoist=dict(compiler_archive=str(path.relative_to(root)),archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        scope='Isolated phase-A NTT8 wgs_seg, two static first/steady stage bodies; not full runtime counter evidence.',
        static_prmt_sites=prmts,conclusion='Four scale-pair constructions per static stage body already hoisted across four load groups; no new explicit-hoisting saving established.'))
target=root/'muse_moe_w4a16/DECODE_SEVEN_OP_SCREEN_2026-10-31.json';assert not target.exists()
target.write_text(json.dumps(out,indent=2)+'\n')
print('No candidate in',examined,'restricted byte-select combinations;',len(prmts),'existing static PRMT sites in two stage bodies.')
