"""Build a prepare-only WGS comparison; never change timed decoder dispatch."""
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parent = ROOT / "muse_decode_layer/submission_v17_alygg77_6cae49a.py"
target = ROOT / "muse_decode_layer/submission_v20_register_split_probe.py"
source = parent.read_text()
assert hashlib.sha256(source.encode()).hexdigest() == "8f17613bafb2369152f2739638bf8f6dfb193f6a79dbad3e6d25795cf1ebaf4a"
assert not target.exists(), "Submitted experiment files are immutable"
edits = (ROOT / "tools/decoder_wgs_variants.py").read_text()
edit_tree = ast.parse(edits)
patcher = ast.get_source_segment(edits, next(n for n in edit_tree.body if isinstance(n, ast.FunctionDef)))
suffix = '''

# Independent synthetic comparison only. Canonical kernel_fn is unchanged.
import base64 as _WG64
import hashlib as _WGHash
import json as _WGJSON
import math as _WGMath
import statistics as _WGStats
import time as _WGTime
import traceback as _WGTrace
import zlib as _WGZlib
'''
suffix += "\n" + patcher.replace("def register_split(", "def _wg_source(") + "\n"
suffix += r'''

_wg_parent_prepare = prepare


def _wg_build(st, gs):
    defines = tuple(_main_defines(st['ma_sfx'])) + tuple(st['ma_sfx'])
    if DQ7 and '-DOKC_DQ7=1' not in defines:
        defines += ('-DOKC_DQ7=1',)
    assert '-DOKC_WGS=1' in defines and '-DOKC_NW=16' in defines
    assert not any(d.startswith('@') for d in defines)
    native = _wg_source(MOE_CUDA_SRC, gs)
    native += r"""
std::vector<int64_t> okc_resources() {
  std::vector<int64_t> out;
  for (int big = 0; big < 2; ++big) {
    cudaFuncAttributes a = {};
    cudaError_t e = big ? cudaFuncGetAttributes(&a, okc::moe_fused<true>)
                        : cudaFuncGetAttributes(&a, okc::moe_fused<false>);
    out.insert(out.end(), {(int64_t)e, (int64_t)a.numRegs, (int64_t)a.localSizeBytes,
       (int64_t)a.sharedSizeBytes, (int64_t)a.maxThreadsPerBlock});
  }
  return out;
}
"""
    cpp = MOE_CPP_SRC + '\nstd::vector<int64_t> okc_resources();\n'
    tag = _WGHash.sha256((cpp + native + repr(defines)).encode()).hexdigest()[:16]
    ext = load_inline(name=f'decoder_wg_register_split_{tag}', cpp_sources=cpp,
        cuda_sources=native, functions=['okc_run','okc_set_pdl','okc_resources'],
        extra_cuda_cflags=['-O3','-gencode=arch=compute_90a,code=sm_90a','-lineinfo',
                          f'-DOKC_NS={_MA_MAIN_NS}','-DOKC_FNST=4','-DOKC_NSF=3', *defines],
        verbose=False)
    ext.okc_set_pdl(0)
    return ext


def _wg_capture(ext, st, scratch, x, ids, weights):
    out = torch.empty_like(x)
    def call():
        ext.okc_run(st['ma_w'], scratch, x, ids, weights, out, 0, 32, st['nsm'])
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    graph.replay()
    torch.cuda.synchronize()
    return graph, out


def _wg_error(out, ref):
    a,b = out.float(), ref.float()
    assert bool(torch.isfinite(a).all()), 'nonfinite candidate output'
    delta = a-b
    rn = b.norm(dim=1)
    den = torch.maximum(rn, rn.mean()*.1).clamp_min(1e-20)
    rel = float(delta.norm()/b.norm().clamp_min(1e-20))
    row = float((delta.norm(dim=1)/den).max())
    assert rel <= .002 and row <= .01, (rel,row)
    return [rel,row]


def _wg_fixture(m, seed, device):
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((m,4096), device=device, dtype=torch.bfloat16, generator=gen)
    if seed % 2:
        x[:,::97] *= 8
    # Independent skewed routing; no judge/reference imports or fixture reads.
    scores = torch.randn((m,256), device=device, generator=gen) * .7
    scores += torch.linspace(1.5,0,256,device=device)
    values, ids = scores.topk(8,dim=1)
    weights = torch.sigmoid(values)
    weights = (weights / weights.sum(dim=1,keepdim=True) * 1.6).contiguous()
    return x,ids.to(torch.int32).contiguous(),weights


def _wg_measure(graphs, flush, sink, group):
    names = list(graphs)
    ns = {k:[] for k in names}
    for rep in range(9):
        shift = (rep+group) % len(names)
        order = names[shift:] + names[:shift]
        if (rep+group)%2:
            order.reverse()
        for k in order:
            sink.copy_(flush.sum(dtype=torch.float32))
            a,b = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record(); graphs[k][0].replay(); b.record(); b.synchronize()
            ns[k].append(round(a.elapsed_time(b)*1e6))
    ratios = {k:_WGStats.median(_WGMath.sqrt(a*b)/c for a,b,c in
              zip(ns['a0'],ns['a1'],ns[k])) for k in ('gs2','gs4')}
    return dict(ns=ns,speedup=ratios,
                aa=_WGStats.median(a/b for a,b in zip(ns['a0'],ns['a1'])))


def _wg_probe(st):
    report = dict(status='starting',stage='build',rows=[],nearby=[])
    start = _WGTime.monotonic()
    try:
        assert len(st['ma_w']) >= 14 and tuple(st['ma_sfx']), 'exact packed scale path required'
        ext = {'gs2':_wg_build(st,2),'gs4':_wg_build(st,4)}
        report['resources'] = {k:list(v.okc_resources()) for k,v in ext.items()}
        assert all(r[0]==r[5]==0 for r in report['resources'].values())
        parent = _moe(128,st)
        parent.okc_set_pdl(0)
        ext = dict(a0=parent,a1=parent,**ext)
        dev = st['ma_w'][0].device
        # Share one scratch between arms: each complete kernel resets its own
        # generation/counters. Graphs replay serially, exactly as in the parent.
        flush = torch.ones(64<<20,device=dev,dtype=torch.float16)
        sink = torch.zeros((),device=dev,dtype=torch.float32)
        report['stage'] = 'nearby validation'
        for m,seed in ((40,9718),(96,9721),(192,9724)):
            scratch = _ma_make_scratch(dev)
            x,ids,weights = _wg_fixture(m,seed,dev)
            graphs = {k:_wg_capture(v,st,scratch,x,ids,weights) for k,v in ext.items()}
            errors = {k:_wg_error(graphs[k][1],graphs['a0'][1]) for k in ('a1','gs2','gs4')}
            before = graphs['a0'][1].clone()
            x.neg_(); ids.copy_((ids+37)%256); weights.mul_(.75)
            for g,out in graphs.values():g.replay()
            torch.cuda.synchronize()
            fresh = {k:_wg_error(graphs[k][1],graphs['a0'][1]) for k in ('a1','gs2','gs4')}
            change = float((graphs['a0'][1].float()-before.float()).norm()/before.float().norm().clamp_min(1e-20))
            assert change>.001, 'unchanged fresh fixture'
            report['nearby'].append(dict(m=m,seed=seed,error=errors,fresh=fresh,change=change))
            del graphs,scratch,x,ids,weights,before
        retained = []
        for pi,pad_mib in enumerate((0,1,3,7)):
            pad = torch.empty(pad_mib<<20,device=dev,dtype=torch.uint8)
            scratch = _ma_make_scratch(dev)
            retained.append((pad,scratch))
            for si,seed in enumerate((3108,7909)):
                for m in (128,256):
                    report['stage'] = f'timing {m}/{seed}/{pad_mib}'
                    x,ids,weights = _wg_fixture(m,seed,dev)
                    graphs = {k:_wg_capture(v,st,scratch,x,ids,weights) for k,v in ext.items()}
                    errors = {k:_wg_error(graphs[k][1],graphs['a0'][1]) for k in ('a1','gs2','gs4')}
                    row = dict(m=m,seed=seed,pad_mib=pad_mib,error=errors,
                               xmod=x.data_ptr()%(32<<20),smod=scratch[1].data_ptr()%(32<<20),
                               local_routes=int((ids<32).sum()),
                               distinct_local=int(ids[ids<32].unique().numel()))
                    row.update(_wg_measure(graphs,flush,sink,pi+si))
                    before = graphs['a0'][1].clone()
                    x.neg_(); ids.copy_((ids+11)%256); weights.mul_(1.125)
                    for g,out in graphs.values():g.replay()
                    torch.cuda.synchronize()
                    row['fresh'] = {k:_wg_error(graphs[k][1],graphs['a0'][1]) for k in ('a1','gs2','gs4')}
                    row['change'] = float((graphs['a0'][1].float()-before.float()).norm()/before.float().norm().clamp_min(1e-20))
                    assert row['change']>.001
                    report['rows'].append(row)
                    del graphs,x,ids,weights,before
        _check_guards()
        report.update(status='complete',stage='complete')
    except Exception as exc:
        report.update(status='failed',error=repr(exc)[:1000],trace=_WGTrace.format_exc(limit=4)[-2200:])
    finally:
        for s in st['ma_scratch']:s.zero_()
        torch.cuda.synchronize()
        report['seconds'] = _WGTime.monotonic()-start
        payload = _WG64.b64encode(_WGZlib.compress(_WGJSON.dumps(report,separators=(',',':')).encode(),9)).decode()
        # The complete small report is printed LAST, after any compiler output.
        print('DECODER_WGS_REGS_Z '+payload,flush=True)
        summary = dict(status=report['status'],stage=report['stage'],rows=len(report['rows']),
                       nearby=len(report['nearby']),chars=len(payload),seconds=report['seconds'])
        if 'error' in report:summary['error']=report['error'][:240]
        print('DECODER_WGS_REGS_SUMMARY '+_WGJSON.dumps(summary,separators=(',',':')),flush=True)


def prepare(weights):
    st = _wg_parent_prepare(weights)
    _wg_probe(st)
    return st
'''
target.write_text(source + suffix)
ast.parse(target.read_text())
print(target.relative_to(ROOT),hashlib.sha256(target.read_bytes()).hexdigest())
