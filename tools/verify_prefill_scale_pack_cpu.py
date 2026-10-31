"""Execute the exact generated PyTorch repack on CPU; verify inverse mapping."""
import ast
import hashlib
import json
from pathlib import Path
import torch
from prefill_scale_pack import repack_source

ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/'muse_prefill_layer/submission.py').read_text();tree=ast.parse(source)
pos=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
         and any(isinstance(t,ast.Name) and t.id=='_CODE_POS' for t in n.targets))
codes=ast.get_source_segment(source,next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_codes'))
parent=ast.get_source_segment(source,next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_repack'))
new=repack_source(source)
records=[]
for n,ks,gated in [(128,64,False),(8192,1,True)]:
    env=dict(torch=torch,_CODE_POS=pos,KS=ks,WBLK=4608)
    exec(codes+'\n'+parent+'\n'+new,env)
    w=torch.randint(0,256,(1,n,ks*32),dtype=torch.uint8)
    sf=torch.randint(0,256,(1,n,ks*4),dtype=torch.uint8)
    old=env['_repack'](w,sf,gated);actual=env['_ks_repack'](w,sf,gated)
    assert torch.equal(old[...,4096:],actual[...,4096:]),'scale storage changed'
    b=actual[0,...,:4096].to(torch.int64).reshape(n//128,ks,2,4,8,4,4,4)
    words=sum(b[...,i]<<(8*i) for i in range(4))
    # Recover the eight nibble slots directly from bit positions.
    decoded=[]
    for i in [3,7,1,5,2,6,0,4]:
        decoded.append(sum(((words>>pos[i][bit])&1)<<bit for bit in range(4)))
    c=torch.stack(decoded,dim=-1).reshape(n//128,ks,2,4,8,4,4,2,2,2)
    c=c.permute(0,2,3,7,4,1,6,8,5,9).reshape(n,ks,64)
    kp=torch.arange(64);kp=(kp&1)|((kp&8)>>2)|((kp&48)>>2)|((kp&6)<<3)
    if gated:
        perm=torch.tensor([64*nt+8*q+lr if lr<8 else 4096+64*nt+8*q+lr-8
                           for nt in range(n//128) for q in range(8) for lr in range(16)])
    else:perm=torch.arange(n)
    expected=env['_codes'](w[0][perm]).reshape(n,ks,64).index_select(2,kp)
    assert torch.equal(c,expected),'actual repack source failed nibble inverse'
    records.append(dict(n=n,ks=ks,gated=gated,packed_bytes=actual.numel(),nibbles=c.numel(),scale_bytes=sf.numel()))
out=dict(scope='Actual generated _ks_repack AST executed on CPU PyTorch; model KS1 gated test exercises all8192 row positions, KS64 nongated test exercises full K4096. Not GPU numerical/performance proof.',torch_version=torch.__version__,source_sha256=hashlib.sha256(new.encode()).hexdigest(),parent_sha256=hashlib.sha256(source.encode()).hexdigest(),records=records)
(ROOT/'muse_prefill_layer/SCALE_PACK_CPU_2026-10-31.json').write_text(json.dumps(out,indent=2)+'\n')
print('PASS actual generated repack',sum(x['packed_bytes'] for x in records),'bytes and',sum(x['nibbles'] for x in records),'nibbles; scale storage unchanged')
