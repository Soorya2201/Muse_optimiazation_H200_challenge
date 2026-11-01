"""Build immutable same-job two-resident-CTA diagnostic; timed parent unchanged."""
import ast,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1];track=root/'muse_moe_w4a16';parent=(track/'submission.py').read_text()
assert hashlib.sha256(parent.encode()).hexdigest()=='b33070898927b7b92df6c7a2515f9c02dd12723c43ca63682fa164e915dd4281'
from moe_two_cta_native import two_cta_native
native=next(ast.literal_eval(n.value) for n in ast.parse(parent).body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='CUDA_SRC' for x in n.targets));digest=hashlib.sha256(two_cta_native(native).encode()).hexdigest()
suffix='''
# Independent topology diagnostic only; original kernel_fn remains unchanged.
import base64 as _TC64
import copy as _TCCopy
import hashlib as _TCHash
import json as _TCJSON
import math as _TCMath
import struct as _TCStruct
import time as _TCTime
import traceback as _TCTrace
import zlib as _TCZlib
'''+(root/'tools/moe_two_cta_native.py').read_text()+'''\n_tc_parent_prepare=prepare\n'''
suffix+=r'''
_TC_RESOURCES=r"""
std::vector<int64_t> two_cta_resources() {
  std::vector<int64_t> r;
  const size_t sm=okc_smem_bytes();
  for(int b=0;b<2;++b) {
    cudaError_t se=b?cudaFuncSetAttribute(okc::moe_fused<true>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sm):
                     cudaFuncSetAttribute(okc::moe_fused<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sm);
    cudaError_t ce=b?cudaFuncSetAttribute(okc::moe_fused<true>,cudaFuncAttributePreferredSharedMemoryCarveout,100):
                     cudaFuncSetAttribute(okc::moe_fused<false>,cudaFuncAttributePreferredSharedMemoryCarveout,100);
    if(se==cudaSuccess)se=ce;
    cudaFuncAttributes a={};int blocks=0;
    cudaError_t e=b?cudaFuncGetAttributes(&a,okc::moe_fused<true>):cudaFuncGetAttributes(&a,okc::moe_fused<false>);
    cudaError_t oe=b?cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,okc::moe_fused<true>,okc::NTL,sm):
                     cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,okc::moe_fused<false>,okc::NTL,sm);
    r.insert(r.end(),{(int64_t)se,(int64_t)e,(int64_t)a.numRegs,(int64_t)a.localSizeBytes,
                     (int64_t)a.sharedSizeBytes,(int64_t)a.maxThreadsPerBlock,(int64_t)sm,(int64_t)oe,(int64_t)blocks});
  }
  return r;
}
"""
def _tc_build(new):
    native=two_cta_native(CUDA_SRC) if new else CUDA_SRC
    if new:assert _TCHash.sha256(native.encode()).hexdigest()=='__NATIVE_SHA__'
    native+=_TC_RESOURCES
    cpp=CPP_SRC+'\nstd::vector<int64_t> two_cta_resources();\n'
    defs=['-DOKC_NS=3','-DOKC_NSF=3',f'-DOKC_NW={8 if new else 16}',
          '-DOKC_FAST1=0','-DOKC_EPI4=0','-DOKC_WGS=1','-DOKC_PRMTSC=1','-DOKC_DQ7=1']
    tag=_TCHash.sha256((cpp+native+repr(defs)).encode()).hexdigest()[:16]
    mod=load_inline('moe_two_cta_'+tag,cpp_sources=cpp,cuda_sources=native,
        functions=['okc_run','okc_set_pdl','two_cta_resources'],extra_cflags=['-O2','-std=c++20'],
        extra_cuda_cflags=['-O3','-gencode=arch=compute_90a,code=sm_90a','-std=c++20','-lineinfo',*defs],verbose=False)
    mod.okc_set_pdl(0)
    return mod

def _tc_error(out,ref,tol=.002,rowtol=.01):
    a,b=out.double(),ref.double()
    assert bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()),'nonfinite comparison'
    delta=a-b;rn=b.norm(dim=1);den=torch.maximum(rn,rn.mean()*.1).clamp_min(1e-20)
    error=[float(delta.norm()/b.norm().clamp_min(1e-20)),float((delta.norm(dim=1)/den).max())]
    assert error[0]<=tol and error[1]<=rowtol,error
    # Preserve exact float32-origin metrics in compact wire frames.
    return [_TCStruct.unpack('<f',_TCStruct.pack('<f',v))[0] for v in error]

def _tc_fixture(m,seed,device,mode='mixed'):
    data=_gen_input(m,seed,device)
    if mode=='empty':data['topk_ids'].copy_(torch.arange(200,208,device=device,dtype=torch.int32).expand(m,8))
    elif mode=='hot':
        ids=torch.arange(8,device=device,dtype=torch.int32).expand(m,8).clone()
        data['topk_ids'].copy_(ids)
    elif mode=='basis':
        data['hidden_states'].zero_();r=torch.arange(m,device=device)
        data['hidden_states'][r,(r*137+seed)%4096]=4
        ids=torch.arange(200,208,device=device,dtype=torch.int32).expand(m,8).clone();ids[:,0]=0
        data['topk_ids'].copy_(ids)
    return data

def _tc_fp64_basis(data,weights):
    # One-hot GEMM1 uses selected weight columns; no huge W13 dequantization.
    x=data['hidden_states'];m=x.shape[0];columns=x.abs().argmax(dim=1)
    val=x[torch.arange(m,device=x.device),columns].double()
    grid=torch.tensor([0,.5,1,1.5,2,3,4,6],device=x.device,dtype=torch.float64)
    pack=weights['w13'][0][:,columns//2].T
    codes=(pack>>(4*(columns%2))[:,None])&15
    h=grid[(codes&7).long()]*torch.where((codes&8)!=0,-1.,1.).double()
    h*=weights['w13_sf'][0].double()[:,columns//16].T*weights['w13_gs'][0].double()*val[:,None]
    gate,up=h.chunk(2,dim=1);inter=gate*torch.sigmoid(gate)*up
    pack=weights['w2'][0];codes=torch.stack((pack&15,pack>>4),dim=-1).flatten(-2)
    w=grid[(codes&7).long()]*torch.where((codes&8)!=0,-1.,1.).double()
    w*=weights['w2_sf'][0].double().repeat_interleave(16,dim=-1)*weights['w2_gs'][0].double()
    y=inter@w.T;y*=1+weights['post_norm_w'].double()
    y*=torch.rsqrt(y.square().mean(dim=-1,keepdim=True)+1e-8)
    y*=data['topk_weights'][:,0:1].double()
    return y

def _tc_capture(mod,st,scratch,data):
    out=torch.empty_like(data['hidden_states'])
    def call():mod.okc_run(st['w'],scratch,data['hidden_states'],data['topk_ids'],data['topk_weights'],out,st['lo'],st['cnt'],st['grid'])
    call();torch.cuda.synchronize()
    a=torch.cuda.Event(enable_timing=True,external=True);b=torch.cuda.Event(enable_timing=True,external=True)
    a.record();b.record();b.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):a.record();call();b.record()
    g.replay();b.synchronize()
    return g,out,a,b

def _tc_wire(report):
    report=_TCCopy.deepcopy(report);frames=[]
    for row in report['rows']:
        samples=row.pop('event_ms_bits')
        raw=[v for name in ('a0','old','new','a1') for v in samples[name]]
        metrics=[v for key in ('error','fresh') for name in ('old','new','a1') for v in row[key][name]]+[row['change']]
        assert len(raw)==32 and len(metrics)==13
        frames.append(_TCStruct.pack('<32I13f',*(raw+metrics)))
        for key in ('error','fresh','change'):row.pop(key)
    report['codec']='u32_ms_bits_32_then_f32_error_fresh_change_13_per_row'
    header=_TCJSON.dumps(report,separators=(',',':')).encode()
    return _TC64.b64encode(_TCZlib.compress(_TCStruct.pack('<I',len(header))+header+b''.join(frames),9)).decode()

def _tc_probe(st,weights):
    report=dict(status='starting',stage='build',nearby=[],rows=[]);begin=_TCTime.monotonic()
    try:
        assert st['sfx']==('-DOKC_PRMTSC=1',),st['sfx']
        props=torch.cuda.get_device_properties(st['w'][0].device)
        assert (props.major,props.minor)==(9,0) and st['grid']==props.multi_processor_count
        old=_tc_build(False);new=_tc_build(True)
        report['resources']={k:list(v.two_cta_resources()) for k,v in [('old',old),('new',new)]}
        # Guard before any launch:264blocks must actually be simultaneously resident.
        for name,r in report['resources'].items():
            assert len(r)==18
            for off in (0,9):
                assert r[off]==r[off+1]==r[off+7]==0,(name,r)
                assert r[off+3]==0,(name,'local allocation',r)
                assert r[off+8]>=(2 if name=='new' else 1),(name,'insufficient residency',r)
        actual=_ext(tuple(_main_defines(st['sfx']))+tuple(st['sfx']));actual.okc_set_pdl(0)
        arms=dict(a0=actual,old=old,new=new,a1=actual);dev=st['w'][0].device
        flush=torch.ones(64<<20,device=dev,dtype=torch.float32);sink=torch.zeros((),device=dev,dtype=torch.float32)
        sink.copy_(flush.sum());torch.cuda.synchronize()
        def captures(data,scratch):
            out={}
            for name,mod in arms.items():
                report['capture_arm']=name
                print('MOE_TWO_CTA_STAGE '+report['stage']+' arm='+name,flush=True)
                out[name]=_tc_capture(mod,st,scratch,data)
            return out
        def replay(graphs):
            for g in graphs.values():g[0].replay()
            torch.cuda.synchronize()
        def agreement(graphs,phase):
            report['comparison_phase']=phase
            report['finite']={name:bool(torch.isfinite(g[1]).all()) for name,g in graphs.items()}
            assert all(report['finite'].values()),report['finite']
            errors={}
            for name in ('old','new','a1'):
                report['comparison_arm']=name;errors[name]=_tc_error(graphs[name][1],graphs['a0'][1])
            report.pop('comparison_arm',None)
            return errors
        def fresh(data,graphs,seed):
            before=graphs['a0'][1].clone();changed=_tc_fixture(data['hidden_states'].shape[0],seed,dev)
            for key in ('hidden_states','topk_ids','topk_weights'):data[key].copy_(changed[key])
            replay(graphs);errors=agreement(graphs,'fresh')
            change=float((graphs['a0'][1].float()-before.float()).norm()/before.float().norm().clamp_min(1e-20))
            assert change>.001,change
            return errors,_TCStruct.unpack('<f',_TCStruct.pack('<f',change))[0]
        # Empty routes are tested on new against literal zero; parent empty-WGS
        # has an unvalidated missing guard and is never launched here.
        for m in (32,128,256):
            report['stage']=f'empty/{m}';scratch=_make_scratch(dev);data=_tc_fixture(m,6110+m,dev,'empty')
            g=_tc_capture(new,st,scratch,data);assert bool((g[1]==0).all())
            changed=_tc_fixture(m,6210+m,dev)
            for key in ('hidden_states','topk_ids','topk_weights'):data[key].copy_(changed[key])
            g[0].replay();g[3].synchronize();assert bool(torch.isfinite(g[1]).all()) and bool((g[1]!=0).any())
            report.setdefault('empty',[]).append(dict(m=m,zero=True,fresh_nonzero=True));del g,scratch,data,changed
        for m,seed,mode in [(17,6170,'basis'),(33,6171,'mixed'),(65,6172,'hot'),(129,6173,'mixed'),(257,6174,'mixed'),(384,6175,'hot')]:
            report['stage']=f'nearby/{m}/{mode}';scratch=_make_scratch(dev);data=_tc_fixture(m,seed,dev,mode)
            graphs=captures(data,scratch);initial=agreement(graphs,'initial')
            if mode=='basis':
                oracle=_tc_fp64_basis(data,weights)
                report['fp64']={name:_tc_error(graphs[name][1],oracle,.015,.075) for name in ('a0','old','new','a1')}
                del oracle
            errors,change=fresh(data,graphs,seed+97)
            report['nearby'].append(dict(m=m,mode=mode,error=initial,fresh=errors,change=change));del graphs,scratch,data
        retained=[];orders=((0,1,3,2),(1,2,0,3),(2,3,1,0),(3,0,2,1));names=list(arms)
        for pi,pad_mib in enumerate((0,1,3,7)):
            pad=torch.empty(pad_mib<<20,device=dev,dtype=torch.uint8);scratch=_make_scratch(dev);retained.append((pad,scratch))
            for si,seed in enumerate((3101,7907)):
                for mi,m in enumerate((32,128,256)):
                    assert _TCTime.monotonic()-begin<200,'diagnostic200s guard'
                    report['stage']=f'timing/{m}/{seed}/{pad_mib}';data=_tc_fixture(m,seed,dev);graphs=captures(data,scratch)
                    initial=agreement(graphs,'initial');events={k:[] for k in arms}
                    for rep in range(8):
                        for index in orders[rep%4]:
                            name=names[(index+pi+si+mi)%4];g=graphs[name]
                            g[0].replay();sink.copy_(flush.sum());g[0].replay();g[3].synchronize()
                            ms=g[2].elapsed_time(g[3]);assert _TCMath.isfinite(ms) and ms>0,ms
                            events[name].append(_TCStruct.unpack('<I',_TCStruct.pack('<f',ms))[0])
                    errors,change=fresh(data,graphs,seed+113)
                    report['rows'].append(dict(m=m,seed=seed,pad_mib=pad_mib,event_ms_bits=events,error=initial,fresh=errors,change=change,scratch_mod32m=scratch[1].data_ptr()%(32<<20)))
                    del graphs,data
        report.update(status='complete',stage='complete')
    except Exception as exc:
        message=repr(exc);report.update(status='failed',error=message[:2000],error_tail=message[-700:] if len(message)>2000 else '',error_chars=len(message),trace=''.join(_TCTrace.format_tb(exc.__traceback__,limit=4))[-800:])
    finally:
        report['seconds']=_TCTime.monotonic()-begin
        payload=_tc_wire(report);assert len(payload)<12800,('report overflow',len(payload))
        print('MOE_TWO_CTA_Z '+payload,flush=True)
        print('MOE_TWO_CTA_SUMMARY '+_TCJSON.dumps({k:report[k] for k in ('status','stage','seconds')}|dict(rows=len(report['rows']),nearby=len(report['nearby']),chars=len(payload)),separators=(',',':')),flush=True)

def prepare(weights):
    st=_tc_parent_prepare(weights)
    _tc_probe(st,weights)
    return st
'''
suffix=suffix.replace('__NATIVE_SHA__',digest);result=parent+'\n'+suffix;ast.parse(result);assert result.startswith(parent)
target=track/'submission_v41_two_cta_probe.py';assert not target.exists();target.write_text(result)
print(target.relative_to(root),hashlib.sha256(result.encode()).hexdigest(),'native',digest)
