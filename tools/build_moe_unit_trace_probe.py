"""Extend the validated zero-local trace with bounded per-unit observations."""
import ast
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'muse_moe_w4a16/submission_v28_lazy_phase_trace_probe.py'
target=ROOT/'muse_moe_w4a16/submission_v29_unit_trace_probe.py'
source=base.read_text()
assert hashlib.sha256(source.encode()).hexdigest()=='678ec161b8552570fad648d04ad1af4addfa4d4b73fc73afa74aac51ca8aff91'
assert not target.exists()
tree=ast.parse(source)
build=ast.get_source_segment(source,[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_mp_build'][-1])
anchor="    tag=_MPHash.sha256((cpp+native+repr(defines)).encode()).hexdigest()[:16]"
assert build.count(anchor)==1
build=build.replace('def _mp_build(st):','def _mp_build(st,unit_recording=True):')
build=build.replace(anchor,'    native=_ut_native(native,unit_recording)\n'+anchor)
probe=ast.get_source_segment(source,[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_mp_probe'][-1])
anchor='torch.zeros(grid*16+272,device=dev,dtype=torch.int64)'
assert probe.count(anchor)==1
probe=probe.replace(anchor,'torch.zeros(grid*16+272+4*(8+32*16),device=dev,dtype=torch.int64)')
probe=probe.replace('MOE_PHASE_TRACE_','MOE_UNIT_TRACE_')
probe=probe.replace("mod=_mp_build(st);report['resources']=list(mod.okc_trace_resources())",
                    "mod=_mp_build(st);report['resources']=list(mod.okc_trace_resources())\n        baseline=_mp_build(st,False);report['baseline_resources']=list(baseline.okc_trace_resources())")
probe=probe.replace("parent=_ext(tuple(_main_defines(st['sfx']))+tuple(st['sfx']));parent.okc_set_pdl(0)",
                    "canonical=_ext(tuple(_main_defines(st['sfx']))+tuple(st['sfx']));canonical.okc_set_pdl(0)\n        parent=baseline")
probe=probe.replace("('a1',parent)]","('a1',parent),('canonical',canonical)]")
probe=probe.replace("['trace','a1']","['trace','a1','canonical']")
probe=probe.replace("'<19I8f3d'","'<24I12f4d'")
suffix=r'''

# All instrumentation is synthetic/prepare-only; timed dispatch stays parent.
def _ut_native(native,recording=True):
    if _UT_REGISTER_SPLIT:
        native=_ut_register_split(native,2)
    def replace(before,after):
        nonlocal native
        assert native.count(before)==1,before
        native=native.replace(before,after)
    replace('  unsigned long long trace_dbg;  // diagnostic-only pointer, naturally aligned',
            '  unsigned long long trace_dbg;\n  volatile unsigned trace_count, trace_current;')
    replace('  if (tid==0) S.trace_dbg=(unsigned long long)p.dbg;',
            '  if (tid==0) { S.trace_dbg=(unsigned long long)p.dbg; S.trace_count=0; S.trace_current=0; }')
    start=native.index('#define DBG(k)')
    end=native.index('extern __shared__',start)
    dbg=native[start:end]
    assert dbg.count('if (threadIdx.x == 0)')==1
    dbg=dbg.replace('if (threadIdx.x == 0)','if (threadIdx.x == 0 && blockIdx.x < 132 && blockIdx.x % 33 == 0)')
    macros=['#define UT_SELECTED (threadIdx.x == 0 && blockIdx.x < 132 && blockIdx.x % 33 == 0)']
    def macro(name,body):
        macros.append('#define '+name+' \\\n'+' \\\n'.join(line.strip() for line in body.strip().splitlines()))
    macro('UT_RAW(ACTION)',r"""
do { if (UT_SELECTED) {
 unsigned long long ptr_;
 asm volatile("ld.volatile.shared.u64 %0, [%1];" : "=l"(ptr_) : "r"(smem_u32(&S.trace_dbg)) : "memory");
 if (ptr_) {
  unsigned long long* h_=reinterpret_cast<unsigned long long*>(ptr_)+(size_t)gridDim.x*16+272+(size_t)(blockIdx.x/33)*(8+32*16);
  ACTION
 }
} } while (0)
""")
    macro('UT_SLOT(ACTION)',r"""
UT_RAW(
 const unsigned ix_=S.trace_current;
 if (ix_<32) {
  unsigned long long* f_=h_+8+(size_t)ix_*16;
  ACTION
 }
)
""")
    macro('UT_BEGIN(PH,UU,LO,HI,NN,KD,FN)',r"""
UT_RAW(
 const unsigned ix_=S.trace_count;
 S.trace_current=ix_; S.trace_count=ix_+1;
 h_[0]=ix_+1; if (ix_==0) h_[1]=0;
 if (ix_>=32) h_[1]=1;
 else {
  unsigned long long* f_=h_+8+(size_t)ix_*16;
  f_[0]=(PH);f_[1]=(UU);f_[2]=(LO);f_[3]=(HI);f_[4]=(NN);f_[5]=(KD);
  f_[7]=0;f_[8]=0;f_[9]=0;f_[11]=0;f_[13]=0;f_[14]=0;f_[15]=(FN);
  f_[10]=gtimer();
 }
)
""")
    macro('UT_END(KD)',r"""UT_SLOT(f_[6]=(KD); f_[12]=gtimer(););""")
    macro('UT_WAIT()',r"""UT_SLOT(f_[11]=gtimer(););""")
    macro('UT_FLUSH_BEGIN()',r"""UT_SLOT(f_[7]=pd.kind;f_[8]=pd.u;f_[9]=pd.nst;f_[13]=gtimer(););""")
    macro('UT_FLUSH_END()',r"""UT_SLOT(f_[14]=gtimer(););""")
    macro('UT_READBACK()',r"""UT_SLOT(f_[9]=(unsigned)pd.nst|256u;);""")
    if not recording:
        for name in ['UT_BEGIN(PH,UU,LO,HI,NN,KD,FN)','UT_END(KD)','UT_WAIT()',
                     'UT_FLUSH_BEGIN()','UT_FLUSH_END()','UT_READBACK()']:
            macros.append('#undef '+name.split('(')[0])
            macros.append('#define '+name+' do {} while (0)')
    native=native[:start]+dbg+'\n'.join(macros)+'\n'+native[end:]
    replace('    switch (ntt) {','    UT_BEGIN(PB,u,s0,s1,ntt,pd.kind,0);\n    switch (ntt) {')
    replace('    gs += s1 - s0;','    UT_END(pd.kind);\n    gs += s1 - s0;')
    replace('  wgs_flush<PB>(p, S, pd);\n}',
            '  UT_BEGIN(PB,0,0,0,0,pd.kind,1);\n  wgs_flush<PB>(p, S, pd);\n  UT_END(pd.kind);\n}')
    replace('  wg_wait<0>();\n  if (lane == 0) mbar_arrive1(S.bars + BX_E + prev);',
            '  wg_wait<0>();\n  UT_WAIT();\n  if (lane == 0) mbar_arrive1(S.bars + BX_E + prev);')
    start=native.index('__device__ __forceinline__ void wgs_flush(')
    end=native.index('\n// one segment (',start)
    body=native[start:end]
    anchor='  if (pd.kind == 0) return;'
    assert body.count(anchor)==1 and body.count('return;')==4
    a=body.index(anchor)+len(anchor)
    head,tail=body[:a],body[a:]
    tail=tail.replace('return;','{ UT_FLUSH_END(); return; }')
    anchor='  asm volatile("fence.acq_rel.gpu;\\n" ::: "memory");'
    assert tail.count(anchor)==1
    tail=tail.replace(anchor,'  UT_READBACK();\n'+anchor)
    assert tail.endswith('\n}')
    tail=tail[:-2]+'\n  UT_FLUSH_END();\n}'
    native=native[:start]+head+'\n  UT_FLUSH_BEGIN();'+tail+native[end:]
    before='  const int tid = threadIdx.x;\n  if (tid==0) { S.trace_dbg='
    assert native.count(before)==1
    native=native.replace(before,'  static_assert((((sizeof(Smem)+127)/128)*128+(size_t)WGS_NS*UNIT_BYTES+(size_t)WGS_NX*WGS_XS)<=227*1024,"diagnostic shared allocation exceeds Hopper block cap");\n'+before)
    return native


def _ut_pack(matrix):
    assert matrix and all(len(r)==len(matrix[0]) for r in matrix)
    cols=len(matrix[0]);mins=[min(r[k] for r in matrix) for k in range(cols)]
    widths=[(max(r[k] for r in matrix)-mins[k]).bit_length() for k in range(cols)]
    packed=0;pos=0
    for k in range(cols):
        for row in matrix:
            assert isinstance(row[k],int) and row[k]>=0
            packed|=(row[k]-mins[k])<<pos;pos+=widths[k]
    return dict(rows=len(matrix),cols=cols,minima=mins,widths=widths,
                data_b64=_MP64.b64encode(packed.to_bytes((pos+7)//8,'little')).decode())


def _mp_measure(graphs,flush,sink,group):
    names=['a0','trace','a1','canonical'];ns={k:[] for k in names}
    for rep in range(5):
        i=(rep+group)%4;order=names[i:]+names[:i]
        if (rep+group)%2:order.reverse()
        for k in order:
            sink.copy_(flush.sum(dtype=torch.float32))
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();graphs[k][0].replay();b.record();b.synchronize()
            ns[k].append(round(a.elapsed_time(b)*1e6))
    return dict(ns=ns,ratio=_MPStats.median(_MPMath.sqrt(a*b)/c for a,b,c in zip(ns['a0'],ns['a1'],ns['trace'])),
                aa=_MPStats.median(a/b for a,b in zip(ns['a0'],ns['a1'])),
                baseline_speedup_vs_canonical=_MPStats.median(c/_MPMath.sqrt(a*b) for a,b,c in zip(ns['a0'],ns['a1'],ns['canonical'])))


def _mp_pack_rows(rows):
    blob=bytearray()
    for row in rows:
        ints=[row[k] for k in ['m','seed','pad_mib','scratch_mod32m']]
        for name in ['a0','trace','a1','canonical']:ints+=row['ns'][name]
        assert len(ints)==24 and all(0<=v<2**32 for v in ints)
        errors=[v for field in ['error','fresh'] for name in ['trace','a1','canonical'] for v in row[field][name]]
        blob.extend(_MPStruct.pack('<24I12f4d',*ints,*errors,row['ratio'],row['aa'],row['baseline_speedup_vs_canonical'],row['change']))
    return _MP64.b64encode(blob).decode()


def _mp_trace(graph,scratch,flush,sink,grid):
    assert grid==132,'fixed four-CTA sampling is H200-specific'
    sink.copy_(flush.sum(dtype=torch.float32));graph.replay();torch.cuda.synchronize()
    values=scratch[3].cpu().tolist();ctas=[0,33,66,99]
    clocks=[values[c*8:c*8+8] for c in ctas]
    assert all(all(x>0 for x in row) for row in clocks)
    origin=min(row[0] for row in clocks)
    clocks=[[x-origin for x in row] for row in clocks]
    assert min(x for row in clocks for x in row)>=0
    meta=[values[grid*8+c*8:grid*8+c*8+8] for c in range(grid)]
    assert all(v[4:]==meta[0][4:] for v in meta)
    nj=meta[0][4]+meta[0][5]
    jobs=[values[grid*16+2*j:grid*16+2*j+2] for j in range(nj)]
    cuts=[v for row in meta for v in row[:4]]
    frames=[];counts=[]
    for c in ctas:
        base=grid*16+272+(c//33)*(8+32*16)
        count,overflow=values[base:base+2]
        assert 0<count<=32 and overflow==0,('trace overflow',c,count,overflow)
        counts.append(count)
        for i in range(count):
            row=values[base+8+i*16:base+8+(i+1)*16]
            for k in range(10,15):
                if row[k]:row[k]=row[k]-origin+1
                assert 0<=row[k]<2**32
            frames.append(row)
    return dict(kind='sampled_unit_trace_v1',origin_ns=origin,grid=grid,ctas=ctas,
                phase_clocks=_ut_pack(clocks),frames=_ut_pack(frames),frame_counts=counts,
                frame_fields=['phase','unit','s0','s1','ntt','prior_kind','after_kind','flush_kind','flush_unit','flush_nst',
                              'segment_start','compute_wait_end','segment_end','flush_start','flush_end','final_flush'],
                metadata_header=meta[0][4:],cuts_u16_le_b64=_MP64.b64encode(_MPStruct.pack('<'+str(len(cuts))+'H',*cuts)).decode(),jobs=jobs)
'''
helper_source=(ROOT/'tools/decoder_wgs_variants.py').read_text()
helper_node=next(n for n in ast.parse(helper_source).body if isinstance(n,ast.FunctionDef) and n.name=='register_split')
helper=ast.get_source_segment(helper_source,helper_node).replace('def register_split(','def _ut_register_split(')
target.write_text(source+'\n_UT_REGISTER_SPLIT=True\n'+helper+'\n'+suffix+'\n'+build+'\n'+probe+'\n')
ast.parse(target.read_text())
print(target.relative_to(ROOT),hashlib.sha256(target.read_bytes()).hexdigest())
