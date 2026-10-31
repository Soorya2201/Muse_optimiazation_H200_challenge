"""Historical V33 builder. Its missing hash import failed before GPU build.

Retained for reproduction; build_moe_tiny_phase_repair.py produces V34 and
executes the repaired host build setup with a CPU compiler stub.
"""
import ast
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/'muse_moe_w4a16/submission.py').read_text()
assert hashlib.sha256(source.encode()).hexdigest()=='b33070898927b7b92df6c7a2515f9c02dd12723c43ca63682fa164e915dd4281'
target=ROOT/'muse_moe_w4a16/submission_v33_tiny_phase_probe.py';assert not target.exists()
suffix=r'''

# Independent synthetic diagnostics only; original kernel_fn stays unchanged.
import base64 as _TP64
import copy as _TPCopy
import json as _TPJSON
import math as _TPMath
import random as _TPRandom
import statistics as _TPStats
import struct as _TPStruct
import time as _TPTime
import traceback as _TPTrace
import zlib as _TPZlib

_tp_parent_prepare=prepare


def _tp_build(st,trace):
    native=CUDA_SRC
    assert native.count('#define OKC_DEBUG 0')==1
    if trace:native=native.replace('#define OKC_DEBUG 0','#define OKC_DEBUG 1')
    native+=r"""
std::vector<int64_t> okc_tiny_resources() {
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
    cpp=CPP_SRC+'\nstd::vector<int64_t> okc_tiny_resources();\n'
    defines=tuple(TINY_DEFINES)+tuple(st['sfx'])
    if DQ7:defines+=('-DOKC_DQ7=1',)
    assert '-DOKC_WGS=1' not in defines
    tag=hashlib.sha256((cpp+native+repr(defines)).encode()).hexdigest()[:16]
    ext=load_inline('moe_tiny_phase_'+tag,cpp_sources=cpp,cuda_sources=native,
        functions=['okc_run','okc_set_pdl','okc_tiny_resources'],
        extra_cuda_cflags=['-O3','-gencode=arch=compute_90a,code=sm_90a','-lineinfo',
                          f'-DOKC_NS={MAIN_NS}','-DOKC_NSF=3',*defines],verbose=False)
    ext.okc_set_pdl(0)
    return ext


def _tp_fixture(m,seed,nlocal,dev,lo):
    rng=_TPRandom.Random(seed)
    ids=[]
    remote=[i for i in range(256) if not lo<=i<lo+32]
    for _ in range(m):
        row=rng.sample(range(lo,lo+32),nlocal)+rng.sample(remote,8-nlocal)
        rng.shuffle(row);ids.append(row)
    gen=torch.Generator(device=dev).manual_seed(seed)
    x=torch.randn((m,4096),device=dev,dtype=torch.bfloat16,generator=gen)
    if seed&1:x[:,::97]*=8
    tw=torch.rand((m,8),device=dev,generator=gen)
    tw=(tw/tw.sum(dim=1,keepdim=True)*1.6).contiguous()
    return [x,torch.tensor(ids,device=dev,dtype=torch.int32),tw]


def _tp_capture(mod,st,scratch,data):
    x,ids,tw=data;out=torch.empty_like(x)
    def run():mod.okc_run(st['w'],scratch,x,ids,tw,out,st['lo'],st['cnt'],st['grid'])
    run();torch.cuda.synchronize()
    start=torch.cuda.Event(enable_timing=True,external=True)
    end=torch.cuda.Event(enable_timing=True,external=True)
    start.record();end.record();end.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):start.record();run();end.record()
    graph.replay();end.synchronize()
    return graph,out,start,end


def _tp_error(out,ref):
    a,b=out.float(),ref.float();delta=a-b
    rn=b.norm(dim=1);den=torch.maximum(rn,rn.mean()*.1).clamp_min(1e-20)
    result=[float(delta.norm()/b.norm().clamp_min(1e-20)),float((delta.norm(dim=1)/den).max())]
    assert bool(torch.isfinite(a).all()) and result[0]<=.002 and result[1]<=.01,result
    return result


def _tp_probe(st):
    report=dict(status='starting',stage='build',rows=[],nearby=[],traces=[],
        clock_columns=['entry','routed','phase_a_end','phase_b_end','grid_barrier_end','exit'],
        arms=['a0','old','trace','a1'],rounds=8)
    begin=_TPTime.monotonic()
    try:
        old=_tp_build(st,False);trace=_tp_build(st,True)
        report['resources']=dict(old=list(old.okc_tiny_resources()),trace=list(trace.okc_tiny_resources()))
        for key in ('old','trace'):
            a=report['resources'][key]
            assert a[0]==a[5]==0 and a[2]==a[7]==0,a
        parent=_ext(tuple(TINY_DEFINES)+tuple(st['sfx']))
        mods=dict(a0=parent,old=old,trace=trace,a1=parent)
        dev=st['w'][0].device;grid=st['grid']
        flush=torch.ones(64<<20,device=dev,dtype=torch.float32)
        sink=torch.zeros((),device=dev,dtype=torch.float32)
        sink.copy_(flush.sum());torch.cuda.synchronize()
        retained=[]
        def captures(data,scratch):return {k:_tp_capture(mod,st,scratch,data) for k,mod in mods.items()}
        def agreement(graphs):return {k:_tp_error(g[1],graphs['a0'][1]) for k,g in graphs.items() if k!='a0'}
        def fresh(graphs,data,seed,nlocal):
            before=graphs['a0'][1].clone()
            changed=_tp_fixture(data[0].shape[0],seed+113,max(1,nlocal),dev,st['lo'])
            for a,b in zip(data,changed):a.copy_(b)
            for g in graphs.values():g[0].replay()
            torch.cuda.synchronize()
            errors=agreement(graphs)
            change=float((graphs['a0'][1].float()-before.float()).norm())
            assert change>.001,change
            return errors,change
        report['stage']='nearby'
        scratch=_make_scratch(dev)+[torch.zeros(grid*8,device=dev,dtype=torch.int64)]
        for m in (2,3,4):
            data=_tp_fixture(m,6120+m,2,dev,st['lo']);graphs=captures(data,scratch)
            error=agreement(graphs);changed,change=fresh(graphs,data,6120+m,2)
            report['nearby'].append(dict(m=m,error=error,fresh=changed,change=change))
            del graphs,data
        del scratch
        for pad_mib in (0,1,3,7):
            pad=torch.empty(pad_mib<<20,device=dev,dtype=torch.uint8)
            scratch=_make_scratch(dev)+[torch.zeros(grid*8,device=dev,dtype=torch.int64)]
            retained.append((pad,scratch))
            for seed in (3101,7907):
                for nlocal in (0,1,2,4,8):
                    report['stage']=f'fixture {seed}/{pad_mib}/{nlocal}'
                    data=_tp_fixture(1,seed,nlocal,dev,st['lo']);graphs=captures(data,scratch)
                    error=agreement(graphs);ns={k:[] for k in mods};names=list(mods)
                    for rep in range(8):
                        order=[(v+rep//2)%4 for v in (0,1,3,2)]
                        if rep&1:order.reverse()
                        for i in order:
                            k=names[i];sink.copy_(flush.sum());g=graphs[k]
                            g[0].replay();g[3].synchronize()
                            ns[k].append(round(g[2].elapsed_time(g[3])*1e6))
                    if pad_mib==0 and nlocal==2:
                        scratch[3].zero_();sink.copy_(flush.sum());graphs['trace'][0].replay()
                        torch.cuda.synchronize()
                        raw=scratch[3].cpu().tolist()
                        clocks=[[raw[c*8+k] for k in range(6)] for c in range(grid)]
                        assert all(all(v>0 for v in row) and row==sorted(row) for row in clocks),'incomplete clocks'
                        report['traces'].append(dict(seed=seed,nlocal=nlocal,pad_mib=pad_mib,grid=grid,clocks_ns=clocks))
                    changed,change=fresh(graphs,data,seed,nlocal)
                    report['rows'].append(dict(seed=seed,nlocal=nlocal,pad_mib=pad_mib,ns=ns,
                        error=error,fresh=changed,change=change,scratch_mod32m=scratch[1].data_ptr()%(32<<20)))
                    del graphs,data
        report.update(status='complete',stage='complete')
    except Exception as exc:
        report.update(status='failed',error=repr(exc)[:1000],trace=_TPTrace.format_exc(limit=4)[-2200:])
    finally:
        # Independent scratch is discarded; incumbent state and weights remain unchanged.
        torch.cuda.synchronize();report['seconds']=_TPTime.monotonic()-begin
        payload=_tp_encode_report(report)
        print('MOE_TINY_PHASE_Z '+payload,flush=True)
        print('MOE_TINY_PHASE_SUMMARY '+_TPJSON.dumps(dict(status=report['status'],stage=report['stage'],
            rows=len(report['rows']),traces=len(report['traces']),chars=len(payload),seconds=report['seconds']),separators=(',',':')),flush=True)


def prepare(weights):
    st=_tp_parent_prepare(weights)
    _tp_probe(st)
    return st
'''
codec=(ROOT/'tools/moe_tiny_phase_codec.py').read_text()
node=next(n for n in ast.parse(codec).body if isinstance(n,ast.FunctionDef) and n.name=='encode_report')
encoder=ast.get_source_segment(codec,node).replace('def encode_report(', 'def _tp_encode_report(')
target.write_text(source+suffix+'\n'+encoder+'\n');ast.parse(target.read_text())
print(target.relative_to(ROOT),hashlib.sha256(target.read_bytes()).hexdigest())
