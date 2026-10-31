"""Independently test public Lt signatures, preserving the parent timed path."""
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parent = ROOT / "muse_prefill_layer/submission_v20_context_bind.py"
target = ROOT / "muse_prefill_layer/submission_v24_public_lt_probe.py"
source = parent.read_text()
assert hashlib.sha256(source.encode()).hexdigest() == "8b52dedda281b2e642bda1d9333bcf2d002fc8611e25caa10e90ed2a849021a5"
assert not target.exists()
suffix = r'''

# Algorithm signature observations from public ruwwww/3b15eb8. No native
# implementation copied. Parent residual recomputation and timed choices stay.
import base64 as _LPBase64
import json as _LPJSON
import math as _LPMath
import statistics as _LPStats
import time as _LPTime
import traceback as _LPTrace
import zlib as _LPZlib

_LP_PINS = {
 (753,'qkv'):(165,1,0,2), (753,'o'):(495,1,0,7),
 (753,'pooler'):(495,1,0,7),
 (3012,'qkv'):(496,1,0,7), (3012,'o'):(501,1,0,7),
 (3012,'router'):(402,1,0,3), (3012,'pooler'):(24,1,0,7),
 (3012,'unpooler'):(501,1,0,7),
 (16384,'qkv'):(24,1,0,7), (16384,'router'):(24,1,0,7),
}


def _lp_capture(run):
    run()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = run()
    g.replay()
    torch.cuda.synchronize()
    return g,out


def _lp_error(a,b):
    x,y = a.float(),b.float()
    assert bool(torch.isfinite(x).all()), 'nonfinite dense diagnostic output'
    return float((x-y).norm()/y.norm().clamp_min(1e-12))


def _lp_probe(st):
    report = dict(status='starting',stage='plans',plans=[],rows=[])
    start = _LPTime.monotonic()
    try:
        assert st.get('lt') is not None and st.get('lt_choice') is not None, 'parent Lt unavailable'
        lt = st['lt']
        selected = {}
        original_choices = dict(st['lt_choice'])
        for (T,name),pin in _LP_PINS.items():
            wn,f32 = LT_GEMMS[name]
            w = st[wn]
            N,K = w.shape
            pid = lt.lt_plan(T,N,K,f32,st['lt_ws'].numel(),32)
            idx = _lt_pin_index(lt,pid,pin)
            par = original_choices.get((T,name))
            parent_info = list(lt.lt_info(*par)) if par is not None else None
            candidate_info = list(lt.lt_info(pid,idx)) if idx >= 0 else None
            same = candidate_info == parent_info and parent_info is not None
            report['plans'].append(dict(T=T,name=name,pin=pin,offered=lt.lt_nalgo(pid),index=idx,
                                        parent_info=parent_info,candidate_info=candidate_info,same_as_parent=same))
            if idx >= 0 and not same:
                selected[T,name] = pid,idx
        flush = torch.ones(128 << 20,dtype=torch.float16,device='cuda')
        sink = torch.zeros((),dtype=torch.float32,device='cuda')
        for group,seed in enumerate((3503,8101)):
            gen = torch.Generator(device='cuda').manual_seed(seed)
            for pad in (0,1,3,7):
                keep_pad = torch.empty((pad << 20,),dtype=torch.uint8,device='cuda')
                for (T,name),choice in selected.items():
                    report['stage'] = f'{group}/{pad}/{T}/{name}'
                    wn,f32 = LT_GEMMS[name]
                    w = st[wn]
                    N,K = w.shape
                    x = (torch.randn((T,K),generator=gen,device='cuda')*.5).to(torch.bfloat16)
                    x[0,0] = 128
                    x[0,1] = -128
                    x[1].zero_()
                    x[1,K//3] = 2
                    def parent_run():
                        return _mm(st,x,name)
                    def candidate_run():
                        out = torch.empty((T,N),dtype=torch.float32 if f32 else torch.bfloat16,device='cuda')
                        lt.lt_run(choice[0],choice[1],x,w,out,st['lt_ws'])
                        return out
                    graphs = {'a0':_lp_capture(parent_run),'a1':_lp_capture(parent_run),
                              'candidate':_lp_capture(candidate_run)}
                    reference = x[:4].double() @ w.double().t()
                    tol = .00005 if f32 else .005
                    errors = {k:_lp_error(pair[1][:4],reference) for k,pair in graphs.items()}
                    assert all(e < tol for e in errors.values()), 'FP64 random/outlier/onehot gate'
                    agreement = _lp_error(graphs['candidate'][1],graphs['a0'][1])
                    assert agreement < tol, 'parent agreement gate'
                    times = {k:[] for k in graphs}
                    names = list(graphs)
                    for rep in range(9):
                        shift = (rep+group+pad)%len(names)
                        order = names[shift:]+names[:shift]
                        if (rep+group)%2:
                            order.reverse()
                        for key in order:
                            sink.copy_(flush.sum(dtype=torch.float32))
                            a,b = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                            a.record()
                            graphs[key][0].replay()
                            b.record()
                            b.synchronize()
                            times[key].append(round(a.elapsed_time(b)*1000000))
                    previous = graphs['a0'][1][:4].clone()
                    x.copy_((torch.randn(x.shape,generator=gen,device='cuda')*.5).to(torch.bfloat16))
                    for graph,out in graphs.values():
                        graph.replay()
                    torch.cuda.synchronize()
                    fresh_ref = x[:4].double() @ w.double().t()
                    fresh_errors = {k:_lp_error(pair[1][:4],fresh_ref) for k,pair in graphs.items()}
                    fresh_agreement = _lp_error(graphs['candidate'][1],graphs['a0'][1])
                    change = _lp_error(graphs['a0'][1][:4],previous)
                    assert change > .001 and fresh_agreement < tol and all(e<tol for e in fresh_errors.values()), 'fresh dense replay gate'
                    ratio = _LPStats.median(_LPMath.sqrt(a*b)/c for a,b,c in zip(times['a0'],times['a1'],times['candidate']))
                    aa = _LPStats.median(a/b for a,b in zip(times['a0'],times['a1']))
                    report['rows'].append(dict(T=T,name=name,group=group,pad_mib=pad,seed=seed,
                        ns=times,ratio=round(ratio,6),aa=round(aa,6),fp64_errors=errors,
                        agreement=agreement,fresh_errors=fresh_errors,fresh_agreement=fresh_agreement,parent_change=change))
                    del graphs,reference,fresh_ref,previous,x
                del keep_pad
        assert st['lt_choice'] == original_choices, 'probe changed timed dispatch'
        report.update(status='complete',stage='finished')
    except Exception as exc:
        report.update(status='failed',error=repr(exc),trace=_LPTrace.format_exc())
    report['elapsed_s'] = round(_LPTime.monotonic()-start,3)
    payload = _LPBase64.b64encode(_LPZlib.compress(_LPJSON.dumps(report,separators=(',',':')).encode(),9)).decode()
    print('PREFILL_PUBLIC_LT_Z '+payload,flush=True)


_lp_parent_prepare = prepare
def prepare(weights):
    st = _lp_parent_prepare(weights)
    _lp_probe(st)
    return st
'''
result = source + suffix
ast.parse(result)
target.write_text(result)
print(target.relative_to(ROOT),hashlib.sha256(result.encode()).hexdigest())
