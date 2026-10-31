"""Build an immutable prepare-only matched scale-sharing experiment."""
import ast
import hashlib
from pathlib import Path
from prefill_scale_pack import repack_source

ROOT=Path(__file__).resolve().parents[1]
parent=ROOT/'muse_prefill_layer/submission_v20_context_bind.py'
source=parent.read_text()
assert hashlib.sha256(source.encode()).hexdigest()=='8b52dedda281b2e642bda1d9333bcf2d002fc8611e25caa10e90ed2a849021a5'
target=ROOT/'muse_prefill_layer/submission_v27_scale_share_probe.py';assert not target.exists()
modifier=(ROOT/'tools/prefill_scale_native.py').read_text();tree=ast.parse(modifier)
helper=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='HELPERS')
fn=ast.get_source_segment(modifier,next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='scale_shared_native'))
fn=fn.replace('def scale_shared_native(','def _ks_native(').replace('HELPERS','_KS_HELPERS')
suffix='''

# Independent synthetic experiments only; canonical kernel_fn is unchanged.
import base64 as _KS64
import copy as _KSCopy
import json as _KSJSON
import math as _KSMath
import random as _KSRng
import statistics as _KSStats
import struct as _KSStruct
import time as _KSTime
import traceback as _KSTrace
import zlib as _KSZlib
'''
suffix+='\n_KS_HELPERS = '+repr(helper)+'\n'+fn+'\n'+repack_source(source)+'\n'
suffix+=r'''
_ks_parent_prepare = prepare

_KS_EXTRA_CPP = r"""
std::vector<int64_t> k64_resources();
void k64_move_raw(const void*, void*, int, cudaStream_t);
void k64_move(torch::Tensor in, torch::Tensor out, int64_t n) {
  k64_move_raw(in.data_ptr(), out.data_ptr(), (int)n, c10::cuda::getCurrentCUDAStream());
}
"""
_KS_RESOURCE = r"""
#include <vector>
std::vector<int64_t> k64_resources() {
  std::vector<int64_t> out;
  for (int fine=0; fine<2; ++fine) {
    cudaFuncAttributes a={};
    cudaError_t e=fine ? cudaFuncGetAttributes(&a,okc_moe::moe_fused_persistent<1>)
                      : cudaFuncGetAttributes(&a,okc_moe::moe_fused_persistent<0>);
    out.insert(out.end(),{(int64_t)e,(int64_t)a.numRegs,(int64_t)a.localSizeBytes,
               (int64_t)a.sharedSizeBytes,(int64_t)a.maxThreadsPerBlock});
  }
  return out;
}
"""
_KS_MOVE = r"""
namespace okc_moe {
template<int N>
__global__ void k64_move_check(const uint32_t* in,uint32_t* out) {
  __shared__ uint32_t memory[N*32];
  for(int i=threadIdx.x;i<N*32;i+=256) memory[i]=in[i];
  __syncthreads();
  k64_permute_staging<N>((uint8_t*)memory,threadIdx.x);
  __syncthreads();
  for(int i=threadIdx.x;i<N*32;i+=256) out[i]=memory[i];
}
}
void k64_move_raw(const void* in,void* out,int n,cudaStream_t stream) {
#define K64_CASE(N) case N: okc_moe::k64_move_check<N><<<1,256,0,stream>>>((const uint32_t*)in,(uint32_t*)out); break
  switch(n) { K64_CASE(32); K64_CASE(64); K64_CASE(96); K64_CASE(128);
              K64_CASE(160); K64_CASE(192); K64_CASE(224); K64_CASE(256); }
#undef K64_CASE
}
"""


def _ks_build(st,fast,new):
    native=(_ks_native(_CUDA_SRC) if new else _CUDA_SRC)+GLUE+_KS_RESOURCE
    cpp=CPP+_KS_EXTRA_CPP
    if new:native+=_KS_MOVE
    else:
        # The old build exposes the same host symbol; it is never called.
        native+='\nvoid k64_move_raw(const void*,void*,int,cudaStream_t) {}\n'
    if not fast:native='#define MOE_FAST_SCALE 0\n'+native
    tag=hashlib.sha256((cpp+native).encode()).hexdigest()[:16]
    functions=['qk_prep','moe_route','moe_plan','moe_gather','moe_gather_plan','moe_combine',
               'moe_gemm','moe_init','moe_gemm_p','moe_fused','boundary','boundary_rc',
               'kv_gather','attn_prep','attn_fwd','attn_init','attn_set_prof','k64_resources','k64_move']
    ext=load_inline('prefill_scale_shared_'+tag,cpp_sources=cpp,cuda_sources=native,functions=functions,
        extra_cuda_cflags=['-O3','-gencode=arch=compute_90a,code=sm_90a','-std=c++20','--use_fast_math'],verbose=False)
    ext.moe_init(st['smem'])
    return ext


def _ks_fixture(m,seed,device,mode='mixed'):
    rng=_KSRng.Random(seed)
    lids=[[-1]*8 for _ in range(m)];positions=[[-1]*8 for _ in range(m)];counts=[0]*32
    if mode=='spectrum':
        goals=[31,32,33,63,64,65,95,96,97,127,128,129,159,160,161,191,
               192,193,223,224,225,255,256,257,511,512,513,640,767,768,769,1024]
        assert m>=1024 and sum(goals)<=8*m
        cursor=0
        for e,count in enumerate(goals):
            for j in range(count):
                tok=(cursor+j)%m;slot=next(s for s in range(8) if lids[tok][s]<0)
                lids[tok][slot]=e;positions[tok][slot]=counts[e];counts[e]+=1
            cursor+=count
    elif mode!='empty':
        for tok in range(m):
            chosen=[0] if mode=='single' else rng.sample(range(32),1)+rng.sample(range(32,256),7)
            if mode=='mixed':
                if rng.random()<.7 and 1+(tok%3) not in chosen:chosen[1]=1+(tok%3)
                if rng.random()<.25:
                    extra=4+rng.randrange(28)
                    if extra not in chosen:chosen[2]=extra
            for slot,e in enumerate(chosen):
                if e<32:
                    lids[tok][slot]=e;positions[tok][slot]=counts[e];counts[e]+=1
    gen=torch.Generator(device=device).manual_seed(seed)
    x=torch.randn((m,4096),device=device,dtype=torch.bfloat16,generator=gen)
    if seed&1:x[:,::97]*=8
    if mode=='onehot':
        x.zero_();r=torch.arange(m,device=device);x[r,(r*17+seed)%4096]=8
    weights=torch.rand((m,8),device=device,generator=gen)
    weights=(weights/weights.sum(dim=1,keepdim=True)*1.6).contiguous()
    return [x,torch.tensor(lids,device=device,dtype=torch.int32),
            torch.tensor(positions,device=device,dtype=torch.int32),
            torch.tensor(counts,device=device,dtype=torch.int32),weights]


def _ks_copy_weights(slot,packed):
    slot[0].copy_(packed[0]);slot[1].copy_(packed[1])


def _ks_call(st,data):
    x,lid,pos,cnt,weights=data
    result=_moe_forward(st,x,lid,pos,cnt,x.shape[0])
    out=torch.empty_like(x)
    st['mod'].moe_combine(result['y'],result['rrow'],lid,weights,st['g_post_expert_norm'],out)
    return out


def _ks_capture(st,data):
    _ks_call(st,data);torch.cuda.synchronize()
    start=torch.cuda.Event(enable_timing=True,external=True)
    end=torch.cuda.Event(enable_timing=True,external=True)
    start.record();end.record();end.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        start.record();out=_ks_call(st,data);end.record()
    graph.replay();end.synchronize()
    return graph,out,start,end


def _ks_error(out,ref):
    a,b=out.float(),ref.float()
    assert bool(torch.isfinite(a).all()),'nonfinite output'
    delta=a-b;rn=b.norm(dim=1)
    den=torch.maximum(rn,rn.mean()*.1).clamp_min(1e-20)
    global_=float(delta.norm()/b.norm().clamp_min(1e-20))
    row=float((delta.norm(dim=1)/den).max())
    assert global_<=.002 and row<=.01,(global_,row)
    return [global_,row]


def _ks_probe(st,weights):
    report=dict(status='starting',stage='build',rows=[],nearby=[])
    begin=_KSTime.monotonic()
    try:
        assert MOE_MODE_SMALL==MOE_MODE_LARGE=='fused' and GATHER_PLAN
        assert torch.cuda.get_device_capability()==(9,0)
        old=_ks_build(st,_fast_scales_ok(weights),False)
        new=_ks_build(st,_fast_scales_ok(weights),True)
        resources=dict(old=list(old.k64_resources()),new=list(new.k64_resources()))
        report['resources']=resources
        for k in ('old','new'):assert resources[k][0]==resources[k][5]==0,resources[k]
        # No newly introduced local allocation; standalone parent has an ABI
        # frame, so compare full native attributes rather than assume zero.
        for off in (0,5):assert resources['new'][off+2]<=resources['old'][off+2],resources
        report['stage']='movement test'
        dev=st['w13r'].device
        moves=[]
        for n in (32,64,96,128,160,192,224,256):
            raw=torch.randint(-(1<<31),(1<<31)-1,(n,32),device=dev,dtype=torch.int32)
            out=torch.empty_like(raw);new.k64_move(raw,out,n)
            rows=torch.arange(n,device=dev)[:,None];phys=torch.arange(32,device=dev)[None,:]
            sw=(rows&7)<<2;k=2*(phys^sw)
            logical=(k&1)|((k&8)>>2)|((k&48)>>2)|((k&6)<<3)
            expected=raw.gather(1,(logical//2)^sw)
            assert torch.equal(out,expected),('staging movement',n)
            moves.append(n)
        report['movement_widths']=moves
        report['stage']='repack'
        packed_old=(st['w13r'],st['w2r'])
        packed_new=(_ks_repack(weights['w13'],weights['w13_sf'],True),
                    _ks_repack(weights['w2'],weights['w2_sf'],False))
        for a,b in zip(packed_old,packed_new):assert torch.equal(a[...,4096:],b[...,4096:]),'scale storage changed'
        slot=(packed_old[0].clone(),packed_old[1].clone())
        control=dict(st,mod=old,w13r=slot[0],w2r=slot[1])
        candidate=dict(st,mod=new,w13r=slot[0],w2r=slot[1])
        arms=dict(a0=st,old=control,new=candidate,a1=st)
        flush=torch.ones(64<<20,device=dev,dtype=torch.float32)
        sink=torch.zeros((),device=dev,dtype=torch.float32)
        sink.copy_(flush.sum());torch.cuda.synchronize()
        def set_arm(name):
            if name in ('old','new'):_ks_copy_weights(slot,packed_new if name=='new' else packed_old)
        def captures(data):
            graphs={}
            for k,state in arms.items():set_arm(k);graphs[k]=_ks_capture(state,data)
            return graphs
        def replay(graphs):
            for k,g in graphs.items():set_arm(k);g[0].replay()
            torch.cuda.synchronize()
        def agreement(graphs):
            return {k:_ks_error(g[1],graphs['a0'][1]) for k,g in graphs.items() if k!='a0'}
        report['stage']='nearby'
        for m,seed,mode in [(31,6128,'empty'),(33,6144,'onehot'),(65,6160,'single'),
                            (257,6161,'mixed'),(2048,6162,'spectrum')]:
            data=_ks_fixture(m,seed,dev,mode)
            graphs=captures(data);initial=agreement(graphs)
            before=graphs['a0'][1].clone()
            changed=_ks_fixture(m,seed+97,dev,'mixed')
            for a,b in zip(data,changed):a.copy_(b)
            replay(graphs);fresh=agreement(graphs)
            change=float((graphs['a0'][1].float()-before.float()).norm()/before.float().norm().clamp_min(1e-20))
            assert change>.001,'fresh input did not change output'
            report['nearby'].append(dict(m=m,mode=mode,error=initial,fresh=fresh,change=change))
            del graphs,data,changed,before
        retained=[]
        for pad_mib in (0,1,3,7):
            pad=torch.empty(pad_mib<<20,device=dev,dtype=torch.uint8);retained.append(pad)
            for seed in (3208,7911):
                for m in (753,3012,16384):
                    report['stage']=f'timing {m}/{seed}/{pad_mib}'
                    data=_ks_fixture(m,seed,dev)
                    graphs=captures(data);errors=agreement(graphs)
                    ns={k:[] for k in arms}
                    # Balanced Williams ordering; every arm each position twice.
                    order0=[0,1,3,2];names=list(arms)
                    for rep in range(8):
                        order=[(v+rep//2)%4 for v in order0]
                        if rep&1:order.reverse()
                        for j in order:
                            k=names[j];set_arm(k)
                            sink.copy_(flush.sum())
                            g=graphs[k];g[0].replay();g[3].synchronize()
                            ns[k].append(round(g[2].elapsed_time(g[3])*1e6))
                    before=graphs['a0'][1].clone();changed=_ks_fixture(m,seed+113,dev)
                    for a,b in zip(data,changed):a.copy_(b)
                    replay(graphs);fresh=agreement(graphs)
                    change=float((graphs['a0'][1].float()-before.float()).norm()/before.float().norm().clamp_min(1e-20))
                    assert change>.001
                    report['rows'].append(dict(m=m,seed=seed,pad_mib=pad_mib,ns=ns,error=errors,fresh=fresh,
                         change=change,xmod=data[0].data_ptr()%(32<<20)))
                    del graphs,data,changed,before
        report.update(status='complete',stage='complete')
    except Exception as exc:
        report.update(status='failed',error=repr(exc)[:1000],trace=_KSTrace.format_exc(limit=4)[-2200:])
    finally:
        torch.cuda.synchronize()
        report['seconds']=_KSTime.monotonic()-begin
        framed=_KSCopy.deepcopy(report);raw=[]
        for row in framed['rows']:
            samples=row.pop('ns')
            for name in ('a0','old','new','a1'):raw.extend(samples[name])
        framed['timing_codec']='u32_ns_rows_a0_old_new_a1_8rounds'
        header=_KSJSON.dumps(framed,separators=(',',':')).encode()
        binary=_KSStruct.pack('<I',len(header))+header+_KSStruct.pack('<'+'I'*len(raw),*raw)
        payload=_KS64.b64encode(_KSZlib.compress(binary,9)).decode()
        print('PREFILL_SCALE_SHARE_Z '+payload,flush=True)
        summary=dict(status=report['status'],stage=report['stage'],rows=len(report['rows']),
                     nearby=len(report['nearby']),chars=len(payload),seconds=report['seconds'])
        if 'error' in report:summary['error']=report['error'][:240]
        print('PREFILL_SCALE_SHARE_SUMMARY '+_KSJSON.dumps(summary,separators=(',',':')),flush=True)


def prepare(weights):
    st=_ks_parent_prepare(weights)
    _ks_probe(st,weights)
    assert st['attn_choice']==ATTN_FIXED
    return st
'''
target.write_text(source+suffix);ast.parse(target.read_text())
print(target.relative_to(ROOT),hashlib.sha256(target.read_bytes()).hexdigest())
