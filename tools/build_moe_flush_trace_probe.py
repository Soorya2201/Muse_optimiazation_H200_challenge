"""Build a prepare-only flush tracer with no per-segment live state."""
import ast
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'muse_moe_w4a16/submission_v28_lazy_phase_trace_probe.py'
target=ROOT/'muse_moe_w4a16/submission_v30_flush_trace_probe.py'
source=base.read_text()
assert hashlib.sha256(source.encode()).hexdigest()=='678ec161b8552570fad648d04ad1af4addfa4d4b73fc73afa74aac51ca8aff91'
assert not target.exists()
tree=ast.parse(source)
def function(name):
    return ast.get_source_segment(source,[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name][-1])
build=function('_mp_build').replace('def _mp_build(st):','def _mp_build(st,unit_recording=True):')
anchor='    tag=_MPHash.sha256((cpp+native+repr(defines)).encode()).hexdigest()[:16]'
assert build.count(anchor)==1
build=build.replace(anchor,'    native=_uf_native(native,unit_recording)\n'+anchor)
probe=function('_mp_probe')
probe=probe.replace('torch.zeros(grid*16+272,device=dev,dtype=torch.int64)',
                    'torch.zeros(grid*16+272+4*2*8192*4,device=dev,dtype=torch.int64)')
probe=probe.replace('MOE_PHASE_TRACE_','MOE_FLUSH_TRACE_')
probe=probe.replace("mod=_mp_build(st);report['resources']=list(mod.okc_trace_resources())",
    "mod=_mp_build(st);report['resources']=list(mod.okc_trace_resources())\n        baseline=_mp_build(st,False);report['baseline_resources']=list(baseline.okc_trace_resources())")
probe=probe.replace("parent=_ext(tuple(_main_defines(st['sfx']))+tuple(st['sfx']));parent.okc_set_pdl(0)",
    "canonical=_ext(tuple(_main_defines(st['sfx']))+tuple(st['sfx']));canonical.okc_set_pdl(0)\n        parent=baseline")
probe=probe.replace("('a1',parent)]","('a1',parent),('canonical',canonical)]")
probe=probe.replace("['trace','a1']","['trace','a1','canonical']")
probe=probe.replace("'<19I8f3d'","'<24I12f4d'")
suffix=r'''

# Synthetic prepare-only instrumentation. No changes to timed dispatch.
def _uf_native(native,recording=True):
    native=_uf_register_split(native,2)
    start=native.index('#define DBG(k)');end=native.index('extern __shared__',start)
    dbg=native[start:end]
    assert dbg.count('if (threadIdx.x == 0)')==1
    selected='threadIdx.x == 0 && blockIdx.x < 132 && blockIdx.x % 66 < 2'
    dbg=dbg.replace('threadIdx.x == 0',selected)
    macro=r"""#define UF_SLOT(ACTION) do { \
 if (threadIdx.x == 0 && blockIdx.x < 132 && blockIdx.x % 66 < 2 && (unsigned)pd.u < 8192u) { \
  unsigned long long ptr_; \
  asm volatile("ld.volatile.shared.u64 %0, [%1];" : "=l"(ptr_) : "r"(smem_u32(&S.trace_dbg)) : "memory"); \
  if (ptr_) { \
   const unsigned sample_=(blockIdx.x/66)*2+(blockIdx.x%66); \
   unsigned long long* f_=reinterpret_cast<unsigned long long*>(ptr_)+(size_t)gridDim.x*16+272+((sample_*2+PB)*8192+(unsigned)pd.u)*4; \
   ACTION \
  } \
 } \
} while (0)
#define UF_BEGIN() UF_SLOT(f_[0]=gtimer();)
#define UF_END() UF_SLOT(f_[1]=gtimer();)
#define UF_READBACK() do {} while(0)
"""
    if not recording:
        macro+='\n#undef UF_BEGIN\n#undef UF_END\n#undef UF_READBACK\n#define UF_BEGIN() do {} while(0)\n#define UF_END() do {} while(0)\n#define UF_READBACK() do {} while(0)\n'
    native=native[:start]+dbg+macro+native[end:]
    start=native.index('__device__ __forceinline__ void wgs_flush(')
    end=native.index('\n// one segment (',start)
    body=native[start:end];anchor='  if (pd.kind == 0) return;'
    assert body.count(anchor)==1 and body.count('return;')==4
    a=body.index(anchor)+len(anchor);head,tail=body[:a],body[a:]
    tail=tail.replace('return;','{ UF_END(); return; }')
    anchor='  asm volatile("fence.acq_rel.gpu;\\n" ::: "memory");'
    assert tail.count(anchor)==1
    tail=tail.replace(anchor,'  UF_READBACK();\n'+anchor)
    assert tail.endswith('\n}')
    tail=tail[:-2]+'\n  UF_END();\n}'
    return native[:start]+head+'\n  UF_BEGIN();'+tail+native[end:]


def _mp_trace(graph,scratch,flush,sink,grid):
    assert grid==132
    scratch[3].zero_()
    sink.copy_(flush.sum(dtype=torch.float32));graph.replay();torch.cuda.synchronize()
    values=scratch[3].cpu().tolist();ctas=[0,1,66,67]
    clocks=[values[c*8:c*8+8] for c in ctas]
    assert all(all(x>0 for x in row) for row in clocks)
    origin=min(row[0] for row in clocks)
    clocks=[[x-origin for x in row] for row in clocks]
    meta=[values[grid*8+c*8:grid*8+c*8+8] for c in range(grid)]
    assert all(v[4:]==meta[0][4:] for v in meta)
    nj=meta[0][4]+meta[0][5]
    jobs=[values[grid*16+2*j:grid*16+2*j+2] for j in range(nj)]
    frames=[]
    for ci,c in enumerate(ctas):
        for pb in (0,1):
            offset=grid*16+272+(ci*2+pb)*8192*4
            for u in range(8192):
                start,end,reserved0,reserved1=values[offset+u*4:offset+(u+1)*4]
                assert reserved0==reserved1==0
                if start or end:
                    assert start and start<=end
                    lo,hi=meta[c][2*pb:2*pb+2]
                    s0=max(lo,u*16)-u*16;s1=min(hi,(u+1)*16)-u*16
                    assert 0<=s0<s1<=16
                    partial=s0!=0 or s1!=16
                    assert partial or pb==0
                    kind=(3 if pb and s0==0 else 2) if partial else 1
                    nst=s1-s0 if partial else 0
                    frames.append([ci,pb,u,kind,nst,start-origin,end-origin])
    assert frames
    cuts=[v for row in meta for v in row[:4]]
    return dict(kind='sampled_flush_trace_v1',origin_ns=origin,grid=grid,ctas=ctas,
        phase_clocks=_ut_pack(clocks),frames=_ut_pack(frames),
        frame_fields=['sample','phase','unit','expected_kind','expected_nst','start','end'],
        metadata_header=meta[0][4:],cuts_u16_le_b64=_MP64.b64encode(_MPStruct.pack('<'+str(len(cuts))+'H',*cuts)).decode(),jobs=jobs)
'''
# Reuse the already CPU-validated four-arm codec/timing functions as plain source.
unit=(ROOT/'muse_moe_w4a16/submission_v29_unit_trace_probe.py').read_text();utree=ast.parse(unit)
helper_node=[n for n in utree.body if isinstance(n,ast.FunctionDef) and n.name=='_ut_register_split'][-1]
helper=ast.get_source_segment(unit,helper_node).replace('def _ut_register_split(','def _uf_register_split(')
helpers='\n\n'.join(ast.get_source_segment(unit,[n for n in utree.body if isinstance(n,ast.FunctionDef) and n.name==name][-1])
                     for name in ['_ut_pack','_mp_measure','_mp_pack_rows'])
target.write_text(source+'\n'+helper+'\n'+suffix+'\n'+helpers+'\n'+build+'\n'+probe+'\n')
ast.parse(target.read_text())
print(target.relative_to(ROOT),hashlib.sha256(target.read_bytes()).hexdigest())
