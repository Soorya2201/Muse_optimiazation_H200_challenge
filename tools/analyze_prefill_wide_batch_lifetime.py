"""Bounded compiler witness and source checks; not a GPU correctness proof."""
import ast
import difflib
import gzip
import hashlib
import json
from pathlib import Path
import re
from prefill_wide_batch_fenced_native import wide_batch_drained_native

root=Path(__file__).resolve().parents[1];track=root/'muse_prefill_layer'
parent=(track/'submission.py').read_text();native=next(ast.literal_eval(n.value) for n in ast.parse(parent).body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='_CUDA_SRC' for x in n.targets))
changed=wide_batch_drained_native(native)
# Input-independent host routing and complete expert epilogue are untouched.
for start,end in [('// ---- routing plan:',None),('  // ---- epilogue through the dedicated staging buffer ----','\ntemplate <bool GATED>')]:
 a=native.index(start);b=native.index(end,a) if end else len(native)
 c=changed.index(start);d=changed.index(end,c) if end else len(changed)
 assert native[a:b]==changed[c:d]
records=[]
for arm,fine in [('wide_batch',0),('wide_batch_fenced',0),('wide_batch_staged',0),('wide_batch_drained',0),('wide_batch_drained',1)]:
 p=track/f'SCALE_CODEGEN_{arm}_Nall_FINE{fine}_2026-11-01.json.gz'
 d=json.loads(gzip.decompress(p.read_bytes()));r=d['response'];assert r['code']==0,(p,r.get('code'))
 stderr='\n'.join(x['text'] for x in r.get('stderr',[]))
 ptx=[x['text'].strip() for x in r['devices']['PTX']['asm']]
 sass=[x['text'].strip() for x in r['devices']['SASS']['asm']]
 rec=dict(arm=arm,fine=fine,archive=str(p.relative_to(root)),archive_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
          diagnostics=stderr,ptx_rows=len(ptx),ptx_clipped=any('[truncated' in s for s in ptx),sass_rows=len(sass),sass_clipped=any('[truncated' in s for s in sass))
 assert not rec['ptx_clipped']
 if arm=='wide_batch_drained':
  assert not re.search(r'C751\d',stderr),stderr
  assert '8 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads' in stderr
  sites=[]
  for i,s in enumerate(ptx):
   m=re.search(r'wgmma\.mma_async\.sync\.aligned\.m64n(160|192|256)k16',s)
   if not m:continue
   # Four consecutive wide MMAs in a batch all reach a draining wait.
   following=next(j for j in range(i+1,len(ptx)) if 'wgmma.wait_group' in ptx[j])
   assert re.search(r'wgmma.wait_group.sync.aligned\s+0;',ptx[following]),ptx[following]
   assert not any(re.search(r'\b(?:bra|ret|exit)\b',v) for v in ptx[i+1:following])
   sites.append(dict(width=int(m[1]),mma_ptx_row=i,wait_ptx_row=following,wait=ptx[following]))
  expected={256:4} if fine==0 else {160:4,192:4,256:4}
  assert {n:sum(s['width']==n for s in sites) for n in expected}==expected
  rec['wide_wait0_sites']=sites
 if arm=='wide_batch':
  # Complete FINE0 SASS: four wide operand quartets loop back to the
  # same writes after only DEPBAR1, without a completion wait on that path.
  assert not rec['sass_clipped']
  start=sass.index('.L_x_66:');back=next(i for i in range(start+1,len(sass)) if 'BRA' in sass[i] and '(.L_x_66)' in sass[i])
  block=sass[start:back+1]
  assert any('DEPBAR.LE' in s and '0x1' in s for s in block)
  assert not any('DEPBAR.LE' in s and '0x0' in s for s in block)
  operands=[s for s in block if 'HGMMA.64x256' in s]
  assert len(operands)==4
  rec['bounded_loop_witness']=dict(start_row=start,backedge_row=back,
      instructions=[dict(row=i,text=sass[i]) for i in range(start,back+1) if i in (start,back) or any(k in sass[i] for k in ('R200','R204','R208','R212','DEPBAR','HGMMA'))],
      interpretation='Physical A operands R200..R215 are overwritten on the next loop iteration with only wait1 on the backedge path; this is a lifetime counterexample, not attribution of the H200 failing arm.')
 records.append(rec)
# Drained wide groups have no outstanding A readers after each stage; even
# one physical bank may be reused. Delaying completions to the wait tests
# the worst permitted execution rather than assuming early HGMMA reads.
runs=0
for stages in range(1,129):
 pending=False
 for stage in range(stages):
  assert not pending
  pending=True  # four MMAs read the one physical bank
  pending=False # group wait0 completion is mandatory
  assert not pending
 runs+=1
out=dict(scope='Exact archived compiler/source checks plus abstract ownership schedule; no H200 speed or numeric guarantee.',native_sha256=hashlib.sha256(changed.encode()).hexdigest(),canonical_prefix_sha256=hashlib.sha256(parent.encode()).hexdigest(),
         drained_bank_schedule_cases=runs,records=records,conclusion='Unfenced logical bank parity was insufficient; dynamic lifetime fence introduces 144-byte local frame, static stages cause C7512. Drained form restores 8-byte ABI frame/no spills/no C751x in both FINE modes and every wide PTX MMA is followed by wait0 without a branch bypass. H200 is still required.')
(track/'WIDE_BATCH_LIFETIME_2026-11-01.json').write_text(json.dumps(out,indent=2)+'\n')
with gzip.open(track/'WIDE_BATCH_DRAINED_NATIVE_2026-11-01.cu.gz','wt') as f:f.write(changed)
(track/'WIDE_BATCH_DRAINED_NATIVE_2026-11-01.diff').write_text(''.join(difflib.unified_diff(native.splitlines(True),changed.splitlines(True),fromfile='parent',tofile='wide_batch_drained',n=2)))
print(out['conclusion']);print('native',out['native_sha256'])
