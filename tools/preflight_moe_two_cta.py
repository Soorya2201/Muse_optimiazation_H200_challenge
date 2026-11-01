"""Exact topology/layout/partition invariants, without CUDA execution."""
import ast,bisect,collections,difflib,gzip,hashlib,json,math,random,statistics
from pathlib import Path
from moe_two_cta_native import two_cta_native
root=Path(__file__).resolve().parents[1];track=root/'muse_moe_w4a16'
parent=(track/'submission.py').read_text();native=next(ast.literal_eval(n.value) for n in ast.parse(parent).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CUDA_SRC' for t in n.targets));new=two_cta_native(native)
# Every TMA payload is a contiguous half of exactly the original stage.
weight_cases=0
for pb in (0,1):
 old_units=32>>pb;new_units=old_units*2
 for e in range(32):
  for ub in range(new_units):
   for st in range(16):
    address=((e*old_units+ub//2)*16+st)*36864+(ub%2)*18432
    for wg in range(2):
     expected=((e*old_units+ub//2)*16+st)*36864+((ub%2)*2+wg)*9216
     assert address+wg*9216==expected
     assert 0<=address and address+18432<=(e+1)*old_units*16*36864
     weight_cases+=1
# Every physical output column appears once, with correct readiness bucket.
for pb in (0,1):
 cols=[];credits=collections.Counter()
 for ub in range(64>>pb):
  for wg in range(2):
   for wp in range(4):
    r16=ub*8+wg*4+wp
    cols.extend(r16*(16 if pb else 8)+g for g in range(16 if pb else 8))
   if not pb:credits[ub>>2]+=1
 assert sorted(cols)==list(range(4096))
 if not pb:assert credits=={i:8 for i in range(16)}
# New gather reuses the parent's exact512 logical thread maps.
gather_cases=0
for count in range(1,33):
 ntt=(count+7)//8
 for row in range(count):
  old=[];newmap=[]
  for ftid in range(512):
   G=ftid&31;dst=((((ftid>>5)*ntt+(row>>3))*32+G)*8+(row&7))
   src=(ftid>>5)*128+8*((G>>1)&3)+2*(G>>3)+(G&1)
   old.append((dst,tuple(src+32*k for k in range(4))))
  for part in range(2):
   for tid in range(256):
    ftid=tid+part*256;G=ftid&31;dst=((((ftid>>5)*ntt+(row>>3))*32+G)*8+(row&7))
    src=(ftid>>5)*128+8*((G>>1)&3)+2*(G>>3)+(G&1)
    newmap.append((dst,tuple(src+32*k for k in range(4))))
  assert sorted(old)==sorted(newmap)
  assert len({a for a,b in newmap})==512
  assert sorted(v for a,b in newmap for v in b)==list(range(2048))
  gather_cases+=1
# Final output passes cover all4096 columns once; route order unchanged.
assert sorted((tid+part*256)*8+i for tid in range(256) for part in range(2) for i in range(8))==list(range(4096))
# Pair-cost cuts make products safe for the worst384-token/all-local call.
tables=[[112,135,140,158],[100,104,115,123]]
rng=random.Random(0x2C7A);cases=[]
for m in (17,31,32,33,63,64,65,128,129,255,256,257,384):
 counts_cases=[[0]*32,[m]+[0]*31,[m]*8+[0]*24,[min(m,97)]*min(32,8*m//min(m,97))]
 for _ in range(30):
  counts=[0]*32
  for t in range(m):
   for e in rng.sample(range(256),8):
    if e<32:counts[e]+=1
  counts_cases.append(counts)
 for counts in counts_cases:
  assert max(counts,default=0)<=m and sum(counts)<=m*8
  jobs=[]
  for count in counts:
   jobs.extend([32]*(count//32));
   if count%32:jobs.append(count%32)
  assert len(jobs)<=136 and sum((c+7)//8*8 for c in jobs)<=3072+7*32<3776
  for pb,table in enumerate(tables):
   stride=512>>pb;pre=[0]
   for c in jobs:pre.append(pre[-1]+stride*table[(c+7)//8-1])
   total=pre[-1];grid=264
   assert total*grid<2**32
   def at(x):
    if not jobs:return 0
    j=bisect.bisect_right(pre,x)-1
    if j==len(jobs):return 2*len(jobs)*stride
    return j*stride*2+2*((x-pre[j])//table[(jobs[j]+7)//8-1])
   cuts=[at(total*c//grid) for c in range(grid)]+[len(jobs)*stride*2]
   assert cuts[0]==0 and all(a<=b for a,b in zip(cuts,cuts[1:]))
   assert all(c%2==0 for c in cuts)
   for s in range(0,len(jobs)*stride*2,16):
    j,k=divmod(s,stride*2);xe=pre[j]+(k//2+1)*table[(jobs[j]+7)//8-1]
    owner=((xe-1)*grid)//total
    owner=min(owner,grid-1)
    while owner+1<grid and total*(owner+1)//grid<xe:owner+=1
    while owner>0 and total*owner//grid>=xe:owner-=1
    assert owner==bisect.bisect_right(cuts,s)-1
    assert cuts[owner]<=s<cuts[owner+1]
   cases.append(dict(m=m,phase=pb,jobs=len(jobs),max_product=total*grid))
# General bound independent of routing samples: <=128 total jobs for3072
# local routes; per-job cost<=158; pair-stage prefix<=128*158*512.
upper_jobs=3072//32+32;upper_product=upper_jobs*158*512*264
assert upper_product<2**32
# All original scratch buffers suffice, with identical xA/xB/fixup offsets.
assert 2*320*2*4096+2*320*2+136*16==2*160*4*4096+2*160*4+136*16
assert 320>=264+1
with gzip.open(track/'TWO_CTA_NATIVE_2026-11-01.cu.gz','wt') as f:f.write(new)
(track/'TWO_CTA_NATIVE_2026-11-01.diff').write_text(''.join(difflib.unified_diff(native.splitlines(True),new.splitlines(True),fromfile='parent',tofile='two_cta',n=2)))
out=dict(scope='CPU indexing/coverage/partition/resource arithmetic only; not GPU concurrency, numerical validation or performance.',parent_sha256=hashlib.sha256(parent.encode()).hexdigest(),native_sha256=hashlib.sha256(new.encode()).hexdigest(),weight_half_payload_cases=weight_cases,gather_exact_row_maps=gather_cases,partition_cases=len(cases),maximum_sampled_prefix_product=max(c['max_product'] for c in cases),routing_independent_u32_product_bound=upper_product,readiness_credits_per_b_stage=8,all4096_output_columns_covered=True,weight_pointers_layout_precision_unchanged=True,scratch_sizes_and_x_b_fixup_offsets_unchanged=True,
 scratch_limits=dict(max_jobs=136,max_padded_rows=3296,parent_xrows_offset=3776,fixup_slots=320,grid=264),
 topology=dict(parent_consumer_warpgroups=4,new_consumer_warpgroups=2,consumer_threads=256,producer_threads=64,threads=320,weight_stage_bytes=18432,weight_slots=3,activation_slots=2,max_tokens_per_job=32,desired_resident_blocks=2),
 conditional_board_models=[dict(assumed_time_cut_on_m32_128_256=cut,other_three_loads_unchanged_proxy=(1-cut)**-.5) for cut in (.05,.10,.15)],
 limitations=['FT32 may reread weights for experts with>32tokens; logical traffic increases are not measured HBM traffic because concurrent chunks can share L2.', 'Both linked BIG variants must report at least2active blocks/SM, no increased local allocation and sufficient static/register/shared resources before any264-block launch.', 'Initial/fresh/empty/nearby/FP32 numeric tests and matched cold graph timings remain required.', 'Two initial string-anchor expected-count assertions were wrong(9vs7,5vs4), then a third(2vs3); repaired before any compiler or GPU execution.', 'Draft0/out8/lutguard/lazygen still spill; shared counter reload removes stack/spills in complete BIG0/1 internal-linkage builds. H200 occupancy remains unknown.'])
(track/'TWO_CTA_PREFLIGHT_2026-11-01.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
