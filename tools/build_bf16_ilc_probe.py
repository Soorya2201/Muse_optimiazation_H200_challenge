"""Build an allocation-only inline-compression diagnostic; timed parent unchanged."""
import ast
import hashlib
from pathlib import Path

from bf16_ilc_allocator import CPP, CUDA

ROOT=Path(__file__).resolve().parents[1];TRACK=ROOT/'muse_bf16_proj'
parent=(TRACK/'submission.py').read_text()
assert hashlib.sha256(parent.encode()).hexdigest()=='290a86ab852c851ad0307002a55a432a1ffce44d7ef51d529e19284193647b90'
target=TRACK/'submission_v49_ilc_probe.py';assert not target.exists()
suffix='\n\n# Independent NVIDIA Hopper inline-compression allocation experiment.\n'
suffix+='# Primary docs: https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#inline-compression\n'
suffix+='_ILC_CPP = '+repr(CPP)+'\n_ILC_CUDA = '+repr(CUDA)+'\n'
suffix+=r'''
import base64 as _ILCB64
import gc as _ILCGC
import json as _ILCJSON
import math as _ILCMath
import statistics as _ILCStats
import struct as _ILCStruct
import threading as _ILCThread
import time as _ILCTime
import traceback as _ILCTraceback
import zlib as _ILCZlib

_ILC_SPECS = (
    ('qkv_gate',128,('qbig','weight')),
    ('qkv_gate',256,('raw','qkv_gate')),
    ('o',128,('oc_ol','weight')),
    ('o',256,('oc_rl','weight')),
    ('pooler',128,('klarge_rawT','pooler')),
    ('pooler',256,('klarge_rawT','pooler')),
    ('unpooler',128,('rawT','unpooler')),
    ('unpooler',256,('klarge_rawT','unpooler')),
)
_ILC_ORDERS=((0,1,3,2),(1,2,0,3),(2,3,1,0),(3,0,2,1))
_ilc_ext=None
_ilc_parent_prepare=prepare
_ilc_parent_kernel=kernel_fn

def _ilc_build():
    global _ilc_ext
    if _ilc_ext is None:
        _ilc_ext=load_inline(name='okc_bf16_inline_compression_v49',cpp_sources=_ILC_CPP,
            cuda_sources=_ILC_CUDA,
            functions=['ilc_capabilities','ilc_create','ilc_map','ilc_view','ilc_info','ilc_copy'],
            extra_cflags=['-O2','-std=c++20'],
            extra_cuda_cflags=['-O3','-std=c++20','-gencode=arch=compute_90a,code=sm_90a'],
            extra_ldflags=['-l:libcuda.so.1'],no_implicit_headers=True,verbose=False)
    return _ilc_ext

def _ilc_replace(state,path,value):
    assert len(path)==2 and isinstance(state[path[0]],dict)
    copied=dict(state);copied[path[0]]=dict(state[path[0]])
    copied[path[0]][path[1]]=value
    return copied

def _ilc_bits(ms):
    return _ILCStruct.unpack('<I',_ILCStruct.pack('<f',ms))[0]

def _ilc_wire(report):
    frames=[];meta=dict(report);rows=[]
    for original in report['rows']:
        row=dict(original);raw=row.pop('event_ms_bits')
        flat=[v for arm in raw for v in arm];assert len(flat)==32
        ns=[round(_ILCStruct.unpack('<f',_ILCStruct.pack('<I',v))[0]*1e6) for v in flat]
        exact=all(0<=n<=65535 and _ilc_bits(n/1e6)==v for n,v in zip(ns,flat))
        row['time_codec']='ns16_exact' if exact else 'float32_bits'
        frame=_ILCStruct.pack('<32H',*ns) if exact else _ILCStruct.pack('<32I',*flat)
        agreement=row.pop('normal')+row.pop('compressed')
        agreement+=sum(row.pop('fresh'),[])+sum(row.pop('remaps'),[])
        assert len(agreement)==16
        assert all(_ILCStruct.unpack('<f',_ILCStruct.pack('<f',v))[0]==v for v in agreement)
        reference=row.pop('fp64');assert len(reference)==3
        frame+=_ILCStruct.pack('<16f3d',*(agreement+reference))
        frames.append(frame);rows.append(row)
    meta['rows']=rows;meta['metric_codec']='normal2_comp2_fresh4_remap8_f32_then_fp64_3d'
    header=_ILCJSON.dumps(meta,separators=(',',':')).encode()
    envelope=_ILCStruct.pack('<I',len(header))+header+b''.join(frames)
    return _ILCB64.b64encode(_ILCZlib.compress(envelope,9)).decode()

def _ilc_agreement(out,reference):
    a,b=out.float(),reference.float();difference=a-b
    an=difference.norm(dim=-1);bn=b.norm(dim=-1);floor=bn.mean()*.1
    assert bool(torch.isfinite(a).all())
    return [float(difference.norm()/b.norm().clamp_min(1e-30)),
            float((an/bn.clamp_min(floor).clamp_min(1e-30)).max())]

def _ilc_fp64(out,x,w,hot):
    expected=x.double()@w.double().T;difference=out.double()-expected
    n=expected.norm(dim=-1);floor=n.mean()*.1
    result=[float(difference.norm()/expected.norm().clamp_min(1e-30)),
            float((difference.norm(dim=-1)/n.clamp_min(floor).clamp_min(1e-30)).max()),
            float((difference[hot].norm(dim=-1)/n[hot].clamp_min(1e-30)).max())]
    assert bool(torch.isfinite(out).all())
    assert result[0]<.01 and result[1]<.05 and result[2]<.001,result
    return result

def _ilc_capture(fn):
    fn();torch.cuda.synchronize()
    start=torch.cuda.Event(enable_timing=True,external=True)
    stop=torch.cuda.Event(enable_timing=True,external=True)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        start.record();out=fn();stop.record()
    graph.replay();torch.cuda.synchronize()
    return graph,out,start,stop

def _ilc_map_copy(ext,cap,mode,view,original):
    ext.ilc_map(cap,mode)
    ext.ilc_copy(view,original)
    torch.cuda.synchronize()

def _ilc_edge_checks(ext,device):
    # All BF16 bit patterns are treated as bytes, including NaNs and signed0.
    sources=[torch.arange(65536,device=device,dtype=torch.int32).to(torch.int16).view(torch.uint8)]
    for size in (1,15,16,17,65537):
        sources.append(torch.arange(size,device=device,dtype=torch.int64).to(torch.uint8))
    result=[]
    for original in sources:
        cap=ext.ilc_create(original.numel());info=list(ext.ilc_info(cap))
        assert info[4:6]==[0,1],info
        ext.ilc_map(cap,0);view=ext.ilc_view(cap)
        for mode,flip in ((0,False),(1,True),(0,True),(1,False)):
            desired=original.bitwise_xor(0x5a) if flip else original
            _ilc_map_copy(ext,cap,mode,view,desired)
            assert torch.equal(view,desired)
        result.append(dict(bytes=original.numel(),granularity=info[3],physical=info[2]))
        del view,cap
    return result

def _ilc_probe(weights,state):
    started=_ILCTime.perf_counter();keepers=[]
    report=dict(status='starting',stage='build',capabilities=None,edges=[],rows=[],
                specs=[(layer,m,list(path)) for layer,m,path in _ILC_SPECS],
                gates=dict(parent_global=.0002,parent_row=.002,fp64_global=.01,
                           fp64_row=.05,onehot=.001,aa=.02,matched=.03,
                           component_speedup=1.02,board_speedup=1.03))
    try:
        print('BF16_ILC_STAGE build',flush=True);ext=_ilc_build()
        report['stage']='capability';report['capabilities']=list(ext.ilc_capabilities())
        if not all(report['capabilities'][i] for i in (1,2)):
            report['status']='unsupported';return
        report['stage']='edge_checks';print('BF16_ILC_STAGE edge_checks',flush=True)
        report['edges']=_ilc_edge_checks(ext,weights['qkv_gate'].device)
        flush=torch.full((64<<20,),.125,device=weights['qkv_gate'].device,dtype=torch.float32)
        drain=torch.empty((),device=flush.device);torch.cuda.synchronize()
        pads=[];report['stage']='comparisons';print('BF16_ILC_STAGE comparisons',flush=True)
        for pad,mib in enumerate((0,7)):
            pads.append(torch.empty(mib<<20,dtype=torch.uint8,device=flush.device))
            for seed_index,seed in enumerate((3209,7909)):
                for spec,(layer,m,path) in enumerate(_ILC_SPECS):
                    if _ILCTime.perf_counter()-started>115:
                        raise RuntimeError('ILC independent diagnostic time budget exceeded')
                    report['active']=[spec,pad,seed]
                    original=state[path[0]][path[1]]
                    assert original.is_contiguous() and original.dtype==torch.bfloat16
                    if path[0]=='klarge_rawT':assert (layer,m) in state['klarge_plans']
                    if layer=='unpooler' and m==128:
                        assert state['choice'][(layer,m)][0]=='torchT'
                    gen=torch.Generator(device=flush.device).manual_seed(seed+spec*37)
                    w=weights[layer];n,k=w.shape
                    x=torch.randn((m,k),generator=gen,device=flush.device,dtype=torch.float32).to(torch.bfloat16)
                    hot=torch.arange(0,m,8,device=flush.device)
                    x[hot]=0
                    columns=(hot*131+seed)%k
                    x[hot,columns]=torch.pow(torch.full_like(hot,2,dtype=torch.float32),(hot%7-3).float()).to(torch.bfloat16)
                    data=dict(x=x,config=dict(m=m,n=n,k=k,layer=layer))
                    cap=ext.ilc_create(original.numel()*original.element_size())
                    info=list(ext.ilc_info(cap));assert info[4:6]==[0,1],info
                    ext.ilc_map(cap,0);bytes_view=ext.ilc_view(cap)
                    mapped=bytes_view.view(original.dtype).view(original.shape)
                    changed=_ilc_replace(state,path,mapped)
                    actual=lambda:_ilc_parent_kernel(data,state)
                    virtual=lambda:_ilc_parent_kernel(data,changed)
                    _ilc_map_copy(ext,cap,0,bytes_view,original)
                    assert torch.equal(bytes_view,original.view(torch.uint8).reshape(-1))
                    a0=_ilc_capture(actual);vm=_ilc_capture(virtual);a1=_ilc_capture(actual)
                    fp64=_ilc_fp64(a0[1],x,w,hot)
                    normal=_ilc_agreement(vm[1],a0[1]);assert normal[0]<.0002 and normal[1]<.002,normal
                    _ilc_map_copy(ext,cap,1,bytes_view,original)
                    assert torch.equal(bytes_view,original.view(torch.uint8).reshape(-1))
                    vm[0].replay();torch.cuda.synchronize()
                    compressed=_ilc_agreement(vm[1],a0[1]);assert compressed[0]<.0002 and compressed[1]<.002,compressed
                    # Different data at the same VA proves this exact backend
                    # reads the remapped tensor, including its TMA descriptors.
                    altered=original.clone()
                    altered.view(torch.uint8).reshape(-1)[::2].bitwise_xor_(1)
                    altered_state=_ilc_replace(state,path,altered)
                    altered_out=_ilc_parent_kernel(data,altered_state);torch.cuda.synchronize()
                    assert not torch.equal(altered_out,a0[1]),'selected weight path did not affect output'
                    remaps=[]
                    for mode,other in ((0,True),(1,False),(0,False),(1,True)):
                        _ilc_map_copy(ext,cap,mode,bytes_view,altered if other else original)
                        vm[0].replay();torch.cuda.synchronize()
                        error=_ilc_agreement(vm[1],altered_out if other else a0[1])
                        assert error[0]<.0002 and error[1]<.002,error
                        remaps.append(error)
                    original_x=x.clone();x.mul_(-.5) # preserve power-of-two one-hot rows
                    a0[0].replay();a1[0].replay();torch.cuda.synchronize()
                    assert not torch.equal(original_x,x)
                    fresh=[]
                    for mode in (0,1):
                        _ilc_map_copy(ext,cap,mode,bytes_view,original)
                        vm[0].replay();torch.cuda.synchronize()
                        error=_ilc_agreement(vm[1],a0[1]);assert error[0]<.0002 and error[1]<.002,error
                        fresh.append(error)
                    _ilc_fp64(a0[1],x,w,hot)
                    x.copy_(original_x)
                    graphs=(a0,vm,vm,a1);times=[[] for _ in graphs]
                    group=spec+seed_index+pad
                    for round_index in range(8):
                        order=tuple((arm+group)%4 for arm in _ILC_ORDERS[round_index%4])
                        for arm in order:
                            if arm in (1,2):_ilc_map_copy(ext,cap,arm-1,bytes_view,original)
                            graph,out,start,stop=graphs[arm]
                            graph.replay() # untimed warm-up, identical for every arm
                            drain.copy_(flush.sum()) # read-only cold-L2 eviction
                            graph.replay();stop.synchronize()
                            ms=start.elapsed_time(stop)
                            assert _ILCMath.isfinite(ms) and ms>0,ms
                            times[arm].append(_ilc_bits(ms))
                    report['rows'].append(dict(id=spec,pad=pad,seed=seed,
                        allocation=info[1:6],virtual_address_mod_32m=info[0]%(32<<20),
                        fp64=fp64,normal=normal,compressed=compressed,fresh=fresh,
                        remaps=remaps,event_ms_bits=times))
                    keepers.append((cap,bytes_view,mapped,changed,data,graphs,altered))
        assert len(report['rows'])==32
        report['status']='complete';report['stage']='complete'
    except Exception as error:
        report['status']='diagnostic_failed'
        message=str(error)
        report['error']=type(error).__name__+': '+message[:6000]
        report['error_tail']=message[-1500:] if len(message)>6000 else ''
        report['error_chars']=len(message)
        report['traceback']=_ILCTraceback.format_exc()[-3000:]
    finally:
        try:torch.cuda.synchronize()
        except Exception as error:
            report['cleanup_error']=type(error).__name__+': '+str(error)[-800:]
            report['status']='diagnostic_failed'
        report['elapsed_s']=_ILCTime.perf_counter()-started
        payload=_ilc_wire(report)
        assert len(payload)<8000,('ILC payload too large',len(payload))
        print('BF16_ILC_Z '+payload,flush=True)
        print('BF16_ILC_SUMMARY '+_ILCJSON.dumps(dict(status=report['status'],stage=report['stage'],
            rows=len(report['rows']),chars=len(payload),capabilities=report['capabilities']),separators=(',',':')),flush=True)
        # Views retain their arena until all graphs/tensors leave this scope;
        # no experimental allocation or input is retained in returned state.

def prepare(weights):
    state=_ilc_parent_prepare(weights)
    _ilc_probe(weights,state)
    return state
'''
source=parent+suffix
ast.parse(source)
target.write_text(source)
print(target.relative_to(ROOT),hashlib.sha256(source.encode()).hexdigest())
