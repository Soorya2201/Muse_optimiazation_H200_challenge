"""Build a same-job expert-pipeline diagnostic; canonical timed path unchanged."""
import ast
import hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1];track=root/'muse_prefill_layer'
parent=(track/'submission.py').read_text();assert hashlib.sha256(parent.encode()).hexdigest()=='8b52dedda281b2e642bda1d9333bcf2d002fc8611e25caa10e90ed2a849021a5'
prior=(track/'submission_v29_ss_pipeline_probe.py').read_text();tree=ast.parse(prior)
def function(name):return ast.get_source_segment(prior,next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name))+'\n'
append=r'''
# Independent synthetic wide RS-WGMMA batching diagnostic, not a scored change.
import base64 as _KS64
import copy as _KSCopy
import json as _KSJSON
import math as _WBMath
import random as _KSRng
import struct as _KSStruct
import time as _KSTime
import traceback as _KSTrace
import zlib as _KSZlib
'''
append+=(root/'tools/prefill_wide_batch_native.py').read_text()+'\n'
append+='\n_wb_parent_prepare=prepare\n'
append+=r'''
_WB_EXTRA_CPP='std::vector<int64_t> batch_resources();\n'
_WB_RESOURCES=r"""
#include <vector>
std::vector<int64_t> batch_resources() {
  std::vector<int64_t> out;
  for(int fine=0;fine<2;++fine) {
    cudaFuncAttributes a={};
    cudaError_t e=fine?cudaFuncGetAttributes(&a,okc_moe::moe_fused_persistent<1>):
                     cudaFuncGetAttributes(&a,okc_moe::moe_fused_persistent<0>);
    out.insert(out.end(),{(int64_t)e,(int64_t)a.numRegs,(int64_t)a.localSizeBytes,
                         (int64_t)a.sharedSizeBytes,(int64_t)a.maxThreadsPerBlock});
  }
  return out;
}
"""
def _ks_build(st,fast,new):
    core=wide_batch_native(_CUDA_SRC) if new else _CUDA_SRC
    if new:assert hashlib.sha256(core.encode()).hexdigest()=='e5b575b5edbaf30e1f88c925c2b0b0ee83579303b45c22f6334177bc16903208'
    native=core+GLUE+_WB_RESOURCES
    if not fast:native='#define MOE_FAST_SCALE 0\n'+native
    cpp=CPP+_WB_EXTRA_CPP
    tag=hashlib.sha256((cpp+native).encode()).hexdigest()[:16]
    functions=['qk_prep','moe_route','moe_plan','moe_gather','moe_gather_plan','moe_combine',
               'moe_gemm','moe_init','moe_gemm_p','moe_fused','boundary','boundary_rc',
               'kv_gather','attn_prep','attn_fwd','attn_init','attn_set_prof','batch_resources']
    ext=load_inline('prefill_wide_batch_'+tag,cpp_sources=cpp,cuda_sources=native,functions=functions,
        extra_cflags=['-O2','-std=c++20'],
        extra_cuda_cflags=['-O3','-gencode=arch=compute_90a,code=sm_90a','-std=c++20','--use_fast_math'],verbose=False)
    ext.moe_init(st['smem'])
    return ext

def _wb_wire(report):
    framed=_KSCopy.deepcopy(report);frames=[]
    for row in framed['rows']:
        samples=row.pop('event_ms_bits')
        raw=[v for name in ('a0','old','new','a1') for v in samples[name]]
        metrics=[v for field in ('error','fresh') for name in ('old','new','a1') for v in row[field][name]]+[row['change']]
        assert len(raw)==32 and len(metrics)==13
        assert all(_KSStruct.unpack('<f',_KSStruct.pack('<f',x))[0]==x for x in metrics)
        frames.append(_KSStruct.pack('<32I13f',*(raw+metrics)))
        for key in ('error','fresh','change'):row.pop(key)
    framed['timing_codec']='u32_ms_bits_32_then_f32_error_fresh_change_13_per_row'
    header=_KSJSON.dumps(framed,separators=(',',':')).encode()
    binary=_KSStruct.pack('<I',len(header))+header+b''.join(frames)
    return _KS64.b64encode(_KSZlib.compress(binary,9)).decode()
'''
for name in ('_ks_fixture','_ks_call','_ks_capture','_ks_error'):append+=function(name)+'\n'
probe=function('_ss_probe').replace('def _ss_probe(', 'def _wb_probe(').replace('.ss_resources()', '.batch_resources()')
probe=probe.replace("        report['stage']='raw weight movement'\n        dev=st['w13r'].device\n        report['movement_cases']=_ss_movement(new,st,weights)", "        report['stage']='unchanged weight layout'\n        dev=st['w13r'].device\n        report['weight_layout_unchanged']=True")
probe=probe.replace('PREFILL_SS_STAGE','PREFILL_WIDE_BATCH_STAGE')
probe=probe.replace('order0=[0,1,3,2];names=list(arms)', 'orders=((0,1,3,2),(1,2,0,3),(2,3,1,0),(3,0,2,1));names=list(arms)')
probe=probe.replace('order=[(v+rep//2)%4 for v in order0]\n                        if rep&1:order.reverse()',
                    'shift=(0,1,3,7).index(pad_mib)+(3208,7911).index(seed)+(753,3012,16384).index(m)\n                        order=[(v+shift)%4 for v in orders[rep%4]]')
probe=probe.replace('sink.copy_(flush.sum())\n                            g=graphs[k];g[0].replay();g[3].synchronize()',
                    'g=graphs[k];g[0].replay() # identical untimed warm-up before cold read\n                            sink.copy_(flush.sum())\n                            g[0].replay();g[3].synchronize()')
probe=probe.replace("                    report['stage']=f'timing {m}/{seed}/{pad_mib}'", "                    if _KSTime.monotonic()-begin>200:raise RuntimeError('wide-batch diagnostic exceeded200s')\n                    report['stage']=f'timing {m}/{seed}/{pad_mib}'")
probe=probe.replace("ns[k].append(round(g[2].elapsed_time(g[3])*1e6))", "ms=g[2].elapsed_time(g[3])\n                            assert _WBMath.isfinite(ms) and ms>0,ms\n                            ns[k].append(_KSStruct.unpack('<I',_KSStruct.pack('<f',ms))[0])")
probe=probe.replace('pad_mib=pad_mib,ns=ns,', 'pad_mib=pad_mib,event_ms_bits=ns,')
probe=probe.replace("report.update(status='failed',error=repr(exc)[:1000],trace=_KSTrace.format_exc(limit=4)[-2200:])", "message=repr(exc)\n        report.update(status='failed',error=message[:2500],error_tail=message[-1000:] if len(message)>2500 else '',error_chars=len(message),trace=''.join(_KSTrace.format_tb(exc.__traceback__,limit=4))[-1200:])")
a=probe.index('        framed=_KSCopy.deepcopy(report)');b=probe.index('        summary=dict(',a)
probe=probe[:a]+"        payload=_wb_wire(report)\n        assert len(payload)<12000,('report too large',len(payload))\n        print('PREFILL_WIDE_BATCH_Z '+payload,flush=True)\n"+probe[b:]
probe=probe.replace('PREFILL_SS_PIPE_SUMMARY','PREFILL_WIDE_BATCH_SUMMARY')
append+=probe+'\n'
append+='''
def prepare(weights):
    st=_wb_parent_prepare(weights)
    _wb_probe(st,weights)
    assert st['attn_choice']==ATTN_FIXED
    return st
'''
result=parent+'\n'+append;ast.parse(result);assert result.startswith(parent)
target=track/'submission_v33_wide_batch_probe.py';assert not target.exists();target.write_text(result)
print(target.relative_to(root),hashlib.sha256(result.encode()).hexdigest())
