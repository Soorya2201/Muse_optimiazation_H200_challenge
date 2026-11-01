"""Check source isolation and worst-case asynchronous A-register/ring ownership."""
import ast
import difflib
import gzip
import hashlib
import json
from pathlib import Path
import random
from prefill_wide_batch_native import wide_batch_native

root=Path(__file__).resolve().parents[1];track=root/'muse_prefill_layer'
parent=(track/'submission.py').read_text();tree=ast.parse(parent)
source=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_CUDA_SRC' for t in n.targets))
changed=wide_batch_native(source)
assert source[source.index('// ---- routing plan:'):]==changed[changed.index('// ---- routing plan:'):]
a=source.index('  // ---- epilogue through the dedicated staging buffer ----',source.index('DEVI void consumer_item('))
b=source.index('\ntemplate <bool GATED>',a)
a1=changed.index('  // ---- epilogue through the dedicated staging buffer ----',changed.index('DEVI void consumer_item('));b1=changed.index('\ntemplate <bool GATED>',a1)
assert source[a:b]==changed[a1:b1]
assert changed.count('shared_weight_native')==0

# Worst permitted completion after wait_group<1>: retain the newest group.
# The model covers every stage/kk through the existing ring parity changes.
def schedule(kps,steps=64,wrong=False,random_completion=None):
    pending=[];issue=[];releases=[];bank_reads={};slot_reads={}
    for it in range(steps//kps):
        slot=it%4
        assert slot not in slot_reads,('overwritten shared slot',it,slot,slot_reads)
        for kk in range(kps):
            bank=(kk&1) if wrong or kps!=1 else it&1
            assert bank not in bank_reads,('overwritten A bank',it,kk,bank,bank_reads)
            stage=it*kps+kk
            group=(stage,bank,slot)
            pending.append(group);bank_reads[bank]=stage;slot_reads[slot]=stage
            issue.extend((stage,j) for j in range(4))
            while len(pending)>1:
                old=pending.pop(0);bank_reads.pop(old[1]);
                if not any(p[2]==old[2] for p in pending):slot_reads.pop(old[2],None)
            if random_completion and pending and random_completion.randrange(2):
                old=pending.pop(0);bank_reads.pop(old[1]);slot_reads.pop(old[2],None)
            # Parent releases previous ring slot after kk0's wait. That slot
            # was consumed by a prior stage's group, now guaranteed complete.
            if kk==0 and it>0:
                assert (it-1)%4 not in slot_reads
                releases.append((it-1)%4)
    pending.clear();bank_reads.clear();slot_reads.clear();releases.append((steps//kps-1)%4)
    assert issue==[(stage,j) for stage in range(steps) for j in range(4)]
    assert len(releases)==steps//kps
    return len(issue),len(releases)

counterexample=None
try:schedule(1,wrong=True)
except AssertionError as error:counterexample=str(error)
assert counterexample and 'overwritten A bank' in counterexample
rng=random.Random(0xB47C);runs=0
for kps in (1,2,4):
    for steps in (4,8,16,64,128):
        schedule(kps,steps);runs+=1
        for _ in range(100):schedule(kps,steps,random_completion=rng);runs+=1
# Widths/roles/producer code are unchanged; only wide widths switch to groups.
widths=[]
for n,kps in ((32,4),(64,2),(128,2),(160,1),(192,1),(256,1)):
    widths.append(dict(n=n,kps=kps,nacc=4 if n==32 else 2 if n<=128 else 1,
                       afrag_bank_rule='it parity' if kps==1 else 'kk parity',
                       commits_per_k64_old=4 if n>128 else 1,
                       commits_per_k64_new=1))
content=''.join(difflib.unified_diff(source.splitlines(True),changed.splitlines(True),fromfile='parent',tofile='wide_batch',n=0))
(track/'WIDE_BATCH_NATIVE_2026-11-01.diff').write_text(content)
with gzip.open(track/'WIDE_BATCH_NATIVE_2026-11-01.cu.gz','wt') as f:f.write(changed)
out=dict(scope='CPU source/ownership model, not actual compiler register allocation, concurrency proof or measured speedup.',
         parent_sha256=hashlib.sha256(parent.encode()).hexdigest(),native_sha256=hashlib.sha256(changed.encode()).hexdigest(),
         source_edits=4,epilogue_unchanged=True,producer_publisher_gather_plan_attention_unchanged=True,
         weight_layout_and_precision_unchanged=True,widths=widths,schedule_cases=runs,
         naive_kps1_counterexample=counterexample,
         limitations=['The real GPU allocator must retain distinct register banks without WGMMA serialization.',
          'Grouped completion can delay ring-slot release and reduce TMA overlap; no timing inference from fewer commits.',
          'Full FINE0/FINE1 compiler resources and H200 initial/fresh/nearby outputs remain unproven.'])
(track/'WIDE_BATCH_PREFLIGHT_2026-11-01.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
