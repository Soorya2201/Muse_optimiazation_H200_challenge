"""Prepare-only current-parent trace: phase clocks, routed jobs and stage cuts."""
import ast
import argparse
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--lazy-debug-pointer',action='store_true')
args=parser.parse_args()
parent=ROOT/'muse_moe_w4a16/submission_v24_alygg77_6cae49a.py'
target=ROOT/('muse_moe_w4a16/submission_v28_lazy_phase_trace_probe.py' if args.lazy_debug_pointer
             else 'muse_moe_w4a16/submission_v27_phase_trace_probe.py')
source=parent.read_text()
assert hashlib.sha256(source.encode()).hexdigest()=='b33070898927b7b92df6c7a2515f9c02dd12723c43ca63682fa164e915dd4281'
assert not target.exists()
suffix=r'''

# Independent diagnostics only; incumbent timed dispatch is unchanged.
import base64 as _MP64
import hashlib as _MPHash
import json as _MPJSON
import math as _MPMath
import statistics as _MPStats
import struct as _MPStruct
import time as _MPTime
import traceback as _MPTrace
import zlib as _MPZlib

_mp_parent_prepare=prepare


def _mp_build(st):
    native=CUDA_SRC
    assert native.count('#define OKC_DEBUG 0')==1
    native=native.replace('#define OKC_DEBUG 0','#define OKC_DEBUG 1')
    if _MP_LAZY_POINTER:
        assert native.count('struct Smem {')==1
        native=native.replace('struct Smem {','struct Smem {\n  unsigned long long trace_dbg;  // diagnostic-only pointer, naturally aligned')
        before='#define DBG(k) do { if (OKC_DEBUG && threadIdx.x == 0 && p.dbg) p.dbg[blockIdx.x * 8 + (k)] = gtimer(); } while (0)'
        assert native.count(before)==1
        native=native.replace(before,r"""#define DBG(k) do { \
  if (threadIdx.x == 0) { \
    unsigned long long ptr_; \
    asm volatile("ld.volatile.shared.u64 %0, [%1];" : "=l"(ptr_) : "r"(smem_u32(&S.trace_dbg)) : "memory"); \
    if (ptr_) reinterpret_cast<unsigned long long*>(ptr_)[blockIdx.x*8+(k)]=gtimer(); \
  } \
} while (0)""")
        before='  const int tid = threadIdx.x;\n  DBG(0);'
        assert native.count(before)==1
        native=native.replace(before,r"""  const int tid = threadIdx.x;
  if (tid==0) S.trace_dbg=(unsigned long long)p.dbg;
  DBG(0);""")
    before='  wgs_range(S, nj, 1, blockIdx.x, gridDim.x, S.wgr[2], S.wgr[3]);'
    assert native.count(before)==1
    native=native.replace(before,before+r"""
  // Idle x-producer lane0 exports metadata after computing its stage cuts.
  // No stores or clocks are added inside wgs_seg / the dequant-MMA loop.
  if (p.dbg) {
    unsigned long long* m=p.dbg+(size_t)gridDim.x*8+(size_t)blockIdx.x*8;
    for (int k=0;k<4;++k)m[k]=(unsigned long long)S.wgr[k];
    m[4]=S.njF;m[5]=S.njS;m[6]=S.wtot;m[7]=S.wtotB;
    if (blockIdx.x==0) {
      unsigned long long* jobs=p.dbg+(size_t)gridDim.x*16;
      for(int j=0;j<nj;++j){jobs[2*j]=S.jE[j];jobs[2*j+1]=S.jC[j];}
    }
  }
""")
    if _MP_LAZY_POINTER:
        # Reading p.dbg in the outlined range helper makes NVCC materialize
        # the whole by-value Params object on the kernel's local stack.
        # Keep its otherwise-unused Params argument unused, as in the parent.
        assert native.count('  if (p.dbg) {')==1
        native=native.replace('  if (p.dbg) {','  if (S.trace_dbg) {')
        assert native.count('=p.dbg+(size_t)gridDim.x')==2
        native=native.replace('=p.dbg+(size_t)gridDim.x','=(unsigned long long*)S.trace_dbg+(size_t)gridDim.x')
    native+=r"""
std::vector<int64_t> okc_trace_resources() {
  std::vector<int64_t> r;
  for(int b=0;b<2;++b){
    cudaFuncAttributes a={};
    auto e=b?cudaFuncGetAttributes(&a,okc::moe_fused<true>):cudaFuncGetAttributes(&a,okc::moe_fused<false>);
    r.insert(r.end(),{(int64_t)e,(int64_t)a.numRegs,(int64_t)a.localSizeBytes,
       (int64_t)a.maxThreadsPerBlock,(int64_t)okc_smem_bytes()});
  }
  return r;
}
"""
    cpp=CPP_SRC+'\nstd::vector<int64_t> okc_trace_resources();\n'
    defines=tuple(_main_defines(st['sfx']))+tuple(st['sfx'])
    if DQ7 and '-DOKC_DQ7=1' not in defines:defines+=('-DOKC_DQ7=1',)
    assert '-DOKC_WGS=1' in defines and st['sfx']
    tag=_MPHash.sha256((cpp+native+repr(defines)).encode()).hexdigest()[:16]
    mod=load_inline(name='moe_phase_trace_'+tag,cpp_sources=cpp,cuda_sources=native,
      functions=['okc_run','okc_set_pdl','okc_trace_resources'],
      extra_cuda_cflags=['-O3','-gencode=arch=compute_90a,code=sm_90a','-lineinfo',
                        f'-DOKC_NS={MAIN_NS}','-DOKC_NSF=3',*defines],verbose=False)
    mod.okc_set_pdl(0)
    return mod


def _mp_capture(mod,st,scratch,data):
    x,ids,tw=data['hidden_states'],data['topk_ids'],data['topk_weights']
    out=torch.empty_like(x)
    def run():mod.okc_run(st['w'],scratch,x,ids,tw,out,st['lo'],st['cnt'],st['grid'])
    run();torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):run()
    graph.replay();torch.cuda.synchronize()
    return graph,out


def _mp_error(out,ref):
    a,b=out.float(),ref.float();d=a-b
    rn=b.norm(dim=1);den=torch.maximum(rn,rn.mean()*.1).clamp_min(1e-20)
    err=[float(d.norm()/b.norm().clamp_min(1e-20)),float((d.norm(dim=1)/den).max())]
    assert bool(torch.isfinite(a).all()) and err[0]<=.002 and err[1]<=.01,err
    return err


def _mp_measure(graphs,flush,sink,group):
    names=['a0','trace','a1'];ns={k:[] for k in names}
    for rep in range(5):
        i=(rep+group)%3;order=names[i:]+names[:i]
        if (rep+group)%2:order.reverse()
        for k in order:
            sink.copy_(flush.sum(dtype=torch.float32))
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();graphs[k][0].replay();b.record();b.synchronize()
            ns[k].append(round(a.elapsed_time(b)*1e6))
    return dict(ns=ns,ratio=_MPStats.median(_MPMath.sqrt(a*b)/c for a,b,c in zip(ns['a0'],ns['a1'],ns['trace'])),
                aa=_MPStats.median(a/b for a,b in zip(ns['a0'],ns['a1'])))


def _mp_trace(graph,scratch,flush,sink,grid):
    sink.copy_(flush.sum(dtype=torch.float32));graph.replay();torch.cuda.synchronize()
    values=scratch[3].cpu().tolist()
    clocks=[values[c*8:c*8+8] for c in range(grid)]
    assert all(all(t>0 for t in ts) for ts in clocks),'missing clock slots'
    origin=min(ts[0] for ts in clocks)
    clock_order=(0,1,7,6,2,3,4,5)
    deltas=[]
    for ts in clocks:
        prev=origin;row=[]
        for col in clock_order:
            v=ts[col]-prev;prev=ts[col]
            assert 0<=v<2**32
            row.append(v)
        deltas.append(row)
    minima=[min(row[k] for row in deltas) for k in range(8)]
    widths=[(max(row[k] for row in deltas)-minima[k]).bit_length() for k in range(8)]
    packed=0;position=0
    for k in range(8):
        for row in deltas:
            packed|=(row[k]-minima[k])<<position;position+=widths[k]
    blob=packed.to_bytes((position+7)//8,'little')
    meta=[values[grid*8+c*8:grid*8+c*8+8] for c in range(grid)]
    assert all(v[4:]==meta[0][4:] for v in meta)
    nj=meta[0][4]+meta[0][5]
    jobs=[values[grid*16+2*j:grid*16+2*j+2] for j in range(nj)]
    assert all(0<=e<32 and 0<n<=64 for e,n in jobs)
    cuts=[v for row in meta for v in row[:4]]
    assert all(0<=v<65536 for v in cuts)
    cut_blob=_MPStruct.pack('<'+str(len(cuts))+'H',*cuts)
    return dict(origin_ns=origin,grid=grid,clock_bitpacked_le_b64=_MP64.b64encode(blob).decode(),
                clock_minima=minima,clock_widths=widths,clock_order=list(clock_order),metadata_header=meta[0][4:],
                cuts_u16_le_b64=_MP64.b64encode(cut_blob).decode(),jobs=jobs,
                clock_columns=['entry','routed','phase_a_end','phase_b_end','grid_barrier_end','exit','producer_join_end','gather_end'])


def _mp_pack_rows(rows):
    blob=bytearray()
    for row in rows:
        ints=[row[k] for k in ['m','seed','pad_mib','scratch_mod32m']]
        for name in ['a0','trace','a1']:ints+=row['ns'][name]
        assert len(ints)==19 and all(0<=v<2**32 for v in ints)
        errors=[v for field in ['error','fresh'] for name in ['trace','a1'] for v in row[field][name]]
        blob.extend(_MPStruct.pack('<19I8f3d',*ints,*errors,row['ratio'],row['aa'],row['change']))
    return _MP64.b64encode(blob).decode()


def _mp_probe(st):
    report=dict(status='starting',stage='build',rows=[],traces=[])
    start=_MPTime.monotonic()
    try:
        mod=_mp_build(st);report['resources']=list(mod.okc_trace_resources())
        assert report['resources'][0]==report['resources'][5]==0
        parent=_ext(tuple(_main_defines(st['sfx']))+tuple(st['sfx']));parent.okc_set_pdl(0)
        dev=st['w'][0].device;grid=st['grid']
        flush=torch.ones(64<<20,device=dev,dtype=torch.float16);sink=torch.zeros((),device=dev,dtype=torch.float32)
        retained=[]
        for pi,pad_mib in enumerate((0,1,3,7)):
            pad=torch.empty(pad_mib<<20,device=dev,dtype=torch.uint8)
            scratch=_make_scratch(dev)+[torch.zeros(grid*16+272,device=dev,dtype=torch.int64)]
            retained.append((pad,scratch))
            for si,seed in enumerate((3101,7907)):
                for m in (32,128,256):
                    report['stage']=f'fixture {m}/{seed}/{pad_mib}'
                    data=_gen_input(m,seed,dev)
                    graphs={k:_mp_capture(v,st,scratch,data) for k,v in [('a0',parent),('trace',mod),('a1',parent)]}
                    row=dict(m=m,seed=seed,pad_mib=pad_mib,
                        error={k:_mp_error(graphs[k][1],graphs['a0'][1]) for k in ['trace','a1']},
                        scratch_mod32m=scratch[1].data_ptr()%(32<<20))
                    row.update(_mp_measure(graphs,flush,sink,pi+si))
                    if pad_mib==0 and seed==3101 and m in (32,256):
                        tr=_mp_trace(graphs['trace'][0],scratch,flush,sink,grid)
                        tr.update(m=m,seed=seed,pad_mib=pad_mib);report['traces'].append(tr)
                    old=graphs['a0'][1].clone()
                    data['hidden_states'].neg_();data['topk_ids'].copy_((data['topk_ids']+13)%256);data['topk_weights'].mul_(.875)
                    for g,out in graphs.values():g.replay()
                    torch.cuda.synchronize()
                    row['fresh']={k:_mp_error(graphs[k][1],graphs['a0'][1]) for k in ['trace','a1']}
                    row['change']=float((graphs['a0'][1].float()-old.float()).norm()/old.float().norm().clamp_min(1e-20))
                    assert row['change']>.001
                    report['rows'].append(row)
                    del graphs,data,old
        report.update(status='complete',stage='complete')
    except Exception as exc:
        report.update(status='failed',error=repr(exc)[:1000],trace=_MPTrace.format_exc(limit=4)[-2000:])
    finally:
        for s in st['scratch']:s.zero_()
        torch.cuda.synchronize();report['seconds']=_MPTime.monotonic()-start
        report['row_count']=len(report['rows'])
        report['row_format']='<19I8f3d'
        report['rows_b64']=_mp_pack_rows(report.pop('rows'))
        payload=_MP64.b64encode(_MPZlib.compress(_MPJSON.dumps(report,separators=(',',':')).encode(),9)).decode()
        print('MOE_PHASE_TRACE_Z '+payload,flush=True)
        print('MOE_PHASE_TRACE_SUMMARY '+_MPJSON.dumps(dict(status=report['status'],stage=report['stage'],
           rows=report['row_count'],traces=len(report['traces']),chars=len(payload),seconds=report['seconds']),separators=(',',':')),flush=True)


def prepare(weights):
    st=_mp_parent_prepare(weights)
    _mp_probe(st)
    return st
'''
suffix='\n_MP_LAZY_POINTER='+repr(args.lazy_debug_pointer)+'\n'+suffix
target.write_text(source+suffix)
ast.parse(target.read_text())
print(target.relative_to(ROOT),hashlib.sha256(target.read_bytes()).hexdigest())
