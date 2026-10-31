"""Transfer the embedded MIT nine-instruction decoder to existing SK2 routes."""
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parent = ROOT / 'muse_bf16_proj/submission.py'
target = ROOT / 'muse_bf16_proj/submission_v39_c12_permute_probe.py'
raw = parent.read_bytes()
assert hashlib.sha256(raw).hexdigest() == '290a86ab852c851ad0307002a55a432a1ffce44d7ef51d529e19284193647b90'
assert not target.exists()
s = raw.decode()
t = ast.parse(s)
escape = next(n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == '_escape_source')
escape = ast.get_source_segment(s, escape)
escape = escape.replace('def _escape_source():\n    source = _sk2_src(True)', 'def _cp_escape_body(source):')
ending = "    shared = CUDA_SRC.replace('@@MBAR@@', MBAR_HELPERS).replace('@@BODY@@', '')\n    return shared + MAP_SRC + source"
assert ending in escape
escape = escape.replace(ending, '    return source')
suffix = '\n\n# SK2 layout transfer of the embedded Nathen Fernandes MIT decodediet component.\n'
suffix += '# Existing MIT/Apache notices above remain intact; not an original codec claim.\n'
suffix += escape + '\n'
suffix += r'''
import base64 as _CPB64
import copy as _CPCopy
import json as _CPJson
import math as _CPMath
import re as _CPRe
import statistics as _CPStats
import time as _CPTime
import zlib as _CPZlib

_cp_parent_prepare = prepare
_cp_parent_kernel = kernel_fn
_CP_ROUTES = (('qkv_gate',8), ('unpooler',8), ('qkv_gate',16), ('o',16),
              ('unpooler',16), ('qkv_gate',32), ('o',32), ('pooler',32), ('unpooler',32))
_CP_DECODE = r"""
__device__ __forceinline__ void decode8(uint32_t lw0, uint32_t lw1, uint32_t nw, uint32_t base4, uint32_t* r) {
  uint32_t rotated;
  asm("shf.r.wrap.b32 %0, %1, %1, 4;" : "=r"(rotated) : "r"(nw));
  const uint32_t ha = (nw & 0x87878787u) + base4;
  const uint32_t hb = (rotated & 0x87878787u) + base4;
  r[0] = prmt(lw0, ha, 0x5140);
  r[1] = prmt(lw0, ha, 0x7362);
  r[2] = prmt(lw1, hb, 0x5140);
  r[3] = prmt(lw1, hb, 0x7362);
}
"""


def _cp_specs():
    result = []
    for i, (layer, m) in enumerate(_CP_ROUTES):
        cfg = _U_CFG if (layer,m) == ('unpooler',32) else next(
            c for c in SK2_CONFIGS if c[0] == _sk2_name(layer,m))
        _, _, n, r, nt, nwr, wsk, upw, nacc, cl = cfg
        hl, stages = _sk2_geom(cfg,m)
        assert not hl and cl == 1
        mt = 2 if m == 32 else 1
        na = 1 if mt == 2 and stages >= 4 and nt == 4 else nacc
        args = f'{mt},{m},{nt},{nwr},{wsk},{upw},{stages},true,false,{na},1,__nv_bfloat16'
        result.append(dict(id=i, layer=layer,m=m,cfg=cfg,args=args,family=int(m==8),
                           pk=(m==8), n=n))
    return result


def _cp_source():
    specs = _cp_specs()
    headers = '\n'.join(dict.fromkeys(_CPRe.findall(r'^#include [^\n]+', CUDA_SRC + MAP_SRC,
                                                    flags=_CPRe.MULTILINE)))
    headers += '\n#include <torch/extension.h>\n#include <vector>\n'
    source, dispatch, resources = headers, [], []
    for packed in (0,1):
        for family in (0,1):
            ns = f'cp39_p{packed}_f{family}'
            selected = [v for v in specs if v['family']==family]
            cases = '\n'.join('if(cfg=='+str(v['id'])+') { launch_sk2<'+v['args']+
                '>(Ws,x,out,ptab,pmax,base4,M,N,K,dg,dbg); return; }' for v in selected)
            body = SK2_SRC.replace('@@SK2_DISPATCH@@',cases)
            if family:
                body = _cp_escape_body(body)
            prefix = CUDA_SRC
            if packed:
                a = prefix.index('__device__ __forceinline__ void decode8(')
                b = prefix.index('\ntemplate <typename T>',a)
                prefix = prefix[:a] + _CP_DECODE + prefix[b:]
            native = prefix.replace('@@MBAR@@', MBAR_HELPERS).replace('@@BODY@@','') + MAP_SRC + body
            native = _CPRe.sub(r'^#include [^\n]+\n?', '',native,flags=_CPRe.MULTILINE)
            source += '\nnamespace '+ns+' {\n'+native+'\n}\n'
            dispatch.append(f'if(packed=={packed} && family=={family}) {{ {ns}::sk2(Ws,x,out,tab,pmax,b4,M,N,K,id,tokr,0,dummy); return; }}')
            for v in selected:
                kern = ns+'::sk2_kernel<'+v['args']+','+str(v['pk']).lower()+'>'
                resources.append(f'if(packed=={packed} && id=={v["id"]}) {{ cudaFuncAttributes a; auto rc=cudaFuncGetAttributes(&a,{kern}); TORCH_CHECK(rc==cudaSuccess,"resource query"); return {{a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,a.maxThreadsPerBlock}}; }}')
    source += r"""
void cp_run(at::Tensor Ws,at::Tensor x,at::Tensor out,at::Tensor tab,int64_t pmax,int64_t b4,
 int64_t M,int64_t N,int64_t K,int64_t id,int64_t tokr,int64_t family,int64_t packed,at::Tensor dummy) {
""" + '\n'.join(dispatch) + '\nTORCH_CHECK(false,"invalid CP route");\n}\n'
    source += '\nstd::vector<int64_t> cp_resource(int64_t packed,int64_t id) {\n' + '\n'.join(resources)
    source += '\nTORCH_CHECK(false,"invalid CP resource");\n}\n'
    return source


def _cp_build():
    cpp = r"""
#include <torch/extension.h>
void cp_run(at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,at::Tensor);
std::vector<int64_t> cp_resource(int64_t,int64_t);
"""
    return load_inline('okc_bf16_c12_permute39_90a',cpp,cuda_sources=_cp_source(),
        functions=['cp_run','cp_resource'],extra_cuda_cflags=['-O3','-gencode=arch=compute_90a,code=sm_90a'],
        no_implicit_headers=True,verbose=False)


def _cp_repack(stream):
    assert stream.dtype == torch.uint8 and stream.is_contiguous() and stream.shape[-1] == 768
    result = stream.clone()
    old = stream.view(-1,768)[:,512:].contiguous()
    c = old.view(torch.int32).to(torch.int64) & 0xffffffff
    s1 = c & 0x80808080
    p = ((c & 0x77777777) | ((c & 0x08080808)<<4) | (s1<<4) | (s1>>28)) & 0xffffffff
    h = p & 0x08080808
    inverse = ((p & 0x77777777) | ((p & 0x80808080)>>4) | (h>>4) | (h<<28)) & 0xffffffff
    assert torch.equal(inverse,c), 'code permutation must be lossless'
    result.view(-1,768)[:,512:].copy_(torch.where(p>=2**31,p-2**32,p).to(torch.int32).view(torch.uint8))
    assert torch.equal(result.view(-1,768)[:,:512],stream.view(-1,768)[:,:512]), 'low bytes changed'
    return result


def _cp_pack(st,spec):
    layer,m = spec['layer'],spec['m']
    if (layer,m)==('unpooler',32):
        c = st['u32']
        return c['w'],c['tab'],c['pmax'],c['b4']
    _, stream, tab, pmax, b4, _ = st['sk2'][_sk2_name(layer,m)]
    return stream,tab,pmax,b4


def _cp_run(mod,st,spec,data,slot,packed):
    _,tab,pmax,b4 = _cp_pack(st,spec)
    x=data['x'];m,k=x.shape;n=data['config']['n']
    out=torch.empty((m,n),dtype=torch.bfloat16,device=x.device)
    mod.cp_run(slot,x,out,tab,pmax,b4,m,n,k,spec['id'],spec['m'],spec['family'],packed,st['dummy'])
    return out


def _cp_accuracy(out,ref,hot):
    z=out.double()-ref;rn=ref.norm(dim=1);den=rn.clamp_min(rn.mean()*.1).clamp_min(1e-30)
    e=[float(z.norm()/ref.norm().clamp_min(1e-30)),float((z.norm(dim=1)/den).max()),
       float((z[:hot].norm(dim=1)/den[:hot]).max())]
    assert bool(torch.isfinite(out).all()) and e[0]<.01 and e[1]<.05 and e[2]<.001,e
    return e


def _cp_difference(a,b):
    d=a.float()-b.float();v=b.float();rn=v.norm(dim=1);den=rn.clamp_min(rn.mean()*.1).clamp_min(1e-30)
    e=[float(d.norm()/v.norm().clamp_min(1e-30)),float((d.norm(dim=1)/den).max())]
    assert e[0]<.0002 and e[1]<.002,e
    return e


def _cp_capture(fn,init):
    init();fn();torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=fn()
    g.replay();torch.cuda.synchronize()
    return g,out


def _cp_fixture(st,spec,m,seed,outliers=False):
    n=spec['n'];w=st['raw'][spec['layer']][:n];k=w.shape[1]
    gen=torch.Generator(device=w.device).manual_seed(seed)
    x=torch.randn((m,k),generator=gen,device=w.device,dtype=torch.bfloat16)
    if outliers:x[:,::31]*=16
    hot=max(1,m//8);cols=torch.randint(k,(hot,),generator=gen,device=w.device)
    x[:hot].zero_();x[torch.arange(hot,device=w.device),cols]=2
    return dict(x=x,config=dict(layer=spec['layer'],m=m,n=n,k=k)),w,hot


def _cp_measure(graphs,inits,flush,group):
    ns=[[] for _ in graphs];a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    for rep in range(5):
        order=[(rep+group+i)%4 for i in range(4)]
        if (rep+group)%2:order.reverse()
        for arm in order:
            inits[arm]()  # Repack slot outside the event; every arm then receives a cold L2.
            flush.add_(1)
            a.record();graphs[arm][0].replay();b.record();b.synchronize()
            ns[arm].append(round(a.elapsed_time(b)*1e6))
    return ns


def _cp_probe(state):
    started=_CPTime.perf_counter()
    report=dict(status='starting',cases=[],nearby=[],resources={},specs=_cp_specs(),
                gate=dict(component=1.02,board=1.03,aa=.02,matched_control=.03),
                timing='Five rotated cold-L2 rounds. Arms parent,matched-old,new,parent. Old/new use identical slot pointer; copy outside events and flush after copy. Raw event nanoseconds.')
    try:
        specs=report['specs'];mods=_CP_MODULE
        for sid,m in ((0,7),(3,9),(7,17),(8,24),(5,31)):
            v=specs[sid];report['stage']=dict(kind='nearby',id=sid,m=m)
            stream,_,_,_=_cp_pack(state,v);new=_cp_repack(stream);slot=stream.clone()
            data,w,hot=_cp_fixture(state,v,m,1709+sid,True);ref=data['x'].double()@w.double().T
            outs=[]
            for packed,src in ((0,stream),(1,new)):
                slot.copy_(src);out=_cp_run(mods,state,v,data,slot,packed)
                outs.append(out);_cp_accuracy(out,ref,hot)
            d=_cp_difference(outs[1],outs[0])
            report['nearby'].append(dict(id=sid,m=m,d=d))
            del data,w,ref,outs,out,stream,new,slot
        flush=torch.zeros(64<<20,dtype=torch.float32,device='cuda');retained=[]
        for pad,mib in enumerate((0,1,3,7)):
            pad_tensor=torch.empty(mib<<20,dtype=torch.uint8,device='cuda')
            st=_CPCopy.deepcopy(state);retained.append((pad_tensor,st))
            for v in specs:
                stream,_,_,_=_cp_pack(st,v);new=_cp_repack(stream);slot=stream.clone()
                for group,seed in enumerate((3119,7919)):
                    if _CPTime.perf_counter()-started>120:raise RuntimeError('probe time cap120s')
                    report['stage']=dict(kind='timing',pad=pad,id=v['id'],seed=seed)
                    data,w,hot=_cp_fixture(st,v,v['m'],seed,bool(group))
                    ref=data['x'].double()@w.double().T
                    fns=[lambda:_cp_parent_kernel(data,st),
                         lambda:_cp_run(mods,st,v,data,slot,0),
                         lambda:_cp_run(mods,st,v,data,slot,1),
                         lambda:_cp_parent_kernel(data,st)]
                    inits=[lambda:None,lambda:slot.copy_(stream),lambda:slot.copy_(new),lambda:None]
                    graphs=[_cp_capture(f,i) for f,i in zip(fns,inits)]
                    errs=[_cp_accuracy(g[1],ref,hot) for g in graphs[:3]]
                    diffs=[_cp_difference(g[1],graphs[0][1]) for g in graphs[1:3]]
                    same=bool(torch.equal(graphs[1][1],graphs[2][1]))
                    old=graphs[0][1].clone();data['x'].mul_(-2)
                    for g,init in zip(graphs,inits):init();g[0].replay()
                    torch.cuda.synchronize()
                    assert not torch.equal(old,graphs[0][1]),'fresh inputs did not change output'
                    for g in graphs[:3]:_cp_accuracy(g[1],-2*ref,hot)
                    fresh=[_cp_difference(g[1],graphs[0][1]) for g in graphs[1:3]]
                    ns=_cp_measure(graphs,inits,flush,pad+group+v['id'])
                    report['cases'].append(dict(pad=pad,id=v['id'],seed=seed,ns=ns,
                        e=[[round(z,8) for z in e] for e in errs],d=diffs,fresh=fresh,
                        identical=same,slot=slot.data_ptr()%(32<<20)))
                    if pad==0 and group==0:
                        report['resources'][str(v['id'])]=[list(mods.cp_resource(p,v['id'])) for p in(0,1)]
                    del data,w,ref,graphs,fns,inits,old
                del stream,new,slot
        report['status']='complete'
    except Exception as e:
        import traceback
        report.update(status='failed',error=repr(e),traceback=traceback.format_exc(limit=4)[-2400:])
    finally:
        torch.cuda.synchronize();report['elapsed_s']=round(_CPTime.perf_counter()-started,3)
        payload=_CPB64.b64encode(_CPZlib.compress(_CPJson.dumps(report,separators=(',',':')).encode(),9)).decode()
        assert len(payload)<8000,'diagnostic must fit service tail'
        print('BF16_C12_PERMUTE_Z '+payload,flush=True)


# Compile only our native operators at import; no input or weight computation.
_CP_MODULE=_cp_build() if torch.cuda.is_available() else None


def prepare(weights):
    st=_cp_parent_prepare(weights)
    _cp_probe(st)
    torch.cuda.synchronize();torch.cuda.empty_cache()
    return st
'''
source = s + suffix
ast.parse(source)
target.write_text(source)
print(target.relative_to(ROOT),hashlib.sha256(target.read_bytes()).hexdigest())
