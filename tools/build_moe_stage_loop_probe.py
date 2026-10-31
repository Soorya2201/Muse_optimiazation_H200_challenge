"""Build a controlled, prepare-only uniform-stage-loop experiment."""
import ast
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'muse_moe_w4a16/submission.py';source=base.read_text()
assert hashlib.sha256(source.encode()).hexdigest()=='b33070898927b7b92df6c7a2515f9c02dd12723c43ca63682fa164e915dd4281'
target=ROOT/'muse_moe_w4a16/submission_v31_stage_loop_probe.py';assert not target.exists()
def functions(path,names):
    raw=path.read_text();tree=ast.parse(raw)
    return '\n\n'.join(ast.get_source_segment(raw,[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name][-1]) for name in names)
variant=functions(ROOT/'tools/moe_stage_compact_variant.py',['compact_stage','zero_mma_stage'])
variant=variant.replace('compact_stage','_sl_compact_stage').replace('zero_mma_stage','_sl_zero_mma_stage')
register=functions(ROOT/'tools/decoder_wgs_variants.py',['register_split']).replace('register_split','_sl_register_split')
helpers=functions(ROOT/'muse_moe_w4a16/submission_v30_flush_trace_probe.py',['_mp_capture','_mp_error','_mp_measure','_mp_pack_rows'])
suffix=r'''

# Only independent synthetic preparation probes change. Timed path stays parent.
import base64 as _MP64
import hashlib as _MPHash
import json as _MPJSON
import math as _MPMath
import statistics as _MPStats
import struct as _MPStruct
import time as _MPTime
import traceback as _MPTrace
import zlib as _MPZlib

_sl_parent_prepare=prepare


def _sl_build(st,compact):
    native=_sl_register_split(CUDA_SRC,2)
    if compact:native=_sl_zero_mma_stage(native)
    native+=r"""
std::vector<int64_t> okc_stage_resources() {
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
    cpp=CPP_SRC+'\nstd::vector<int64_t> okc_stage_resources();\n'
    defines=tuple(_main_defines(st['sfx']))+tuple(st['sfx'])
    if DQ7 and '-DOKC_DQ7=1' not in defines:defines+=('-DOKC_DQ7=1',)
    assert '-DOKC_WGS=1' in defines and st['sfx']
    tag=_MPHash.sha256((cpp+native+repr(defines)).encode()).hexdigest()[:16]
    mod=load_inline(name='moe_stage_loop_'+tag,cpp_sources=cpp,cuda_sources=native,
      functions=['okc_run','okc_set_pdl','okc_stage_resources'],
      extra_cuda_cflags=['-O3','-gencode=arch=compute_90a,code=sm_90a','-lineinfo',
                        f'-DOKC_NS={MAIN_NS}','-DOKC_NSF=3',*defines],verbose=False)
    mod.okc_set_pdl(0)
    return mod


def _sl_compare(graphs,data,flush,sink,group,metadata,timing):
    row=dict(metadata)
    row['error']={k:_mp_error(graphs[k][1],graphs['a0'][1]) for k in ['trace','a1','canonical']}
    if timing:row.update(_mp_measure(graphs,flush,sink,group))
    old=graphs['a0'][1].clone()
    data['hidden_states'].neg_();data['topk_ids'].copy_((data['topk_ids']+13)%256);data['topk_weights'].mul_(.875)
    for g,out in graphs.values():g.replay()
    torch.cuda.synchronize()
    row['fresh']={k:_mp_error(graphs[k][1],graphs['a0'][1]) for k in ['trace','a1','canonical']}
    row['change']=float((graphs['a0'][1].float()-old.float()).norm()/old.float().norm().clamp_min(1e-20))
    assert row['change']>.001
    return row


def _sl_probe(st):
    report=dict(status='starting',stage='build',rows=[],nearby=[],
       arm_roles=dict(a0='canonical parent',a1='canonical parent repeat',trace='REG112 + zero-MMA uniform stage loop',canonical='REG112 original stage loop'),
       timed_dispatch='unchanged canonical parent')
    start=_MPTime.monotonic()
    try:
        candidate=_sl_build(st,True);control=_sl_build(st,False)
        report['resources']=list(candidate.okc_stage_resources())
        report['control_resources']=list(control.okc_stage_resources())
        assert all(report[k][0]==report[k][5]==0 for k in ['resources','control_resources'])
        if not all(report[k][2]==report[k][7]==0 for k in ['resources','control_resources']):
            report.update(status='rejected',stage='full native local allocation');return
        parent=_ext(tuple(_main_defines(st['sfx']))+tuple(st['sfx']));parent.okc_set_pdl(0)
        mods=[('a0',parent),('trace',candidate),('a1',parent),('canonical',control)]
        dev=st['w'][0].device
        flush=torch.ones(64<<20,device=dev,dtype=torch.float16);sink=torch.zeros((),device=dev,dtype=torch.float32)
        nearby_scratch=_make_scratch(dev)
        for m,seed in [(40,7823),(96,9533),(192,11239)]:
            report['stage']=f'nearby {m}/{seed}'
            data=_gen_input(m,seed,dev)
            graphs={k:_mp_capture(mod,st,nearby_scratch,data) for k,mod in mods}
            report['nearby'].append(_sl_compare(graphs,data,flush,sink,0,dict(m=m,seed=seed),False))
            del graphs,data
        del nearby_scratch
        retained=[]
        for pi,pad_mib in enumerate((0,1,3,7)):
            pad=torch.empty(pad_mib<<20,device=dev,dtype=torch.uint8);scratch=_make_scratch(dev)
            retained.append((pad,scratch))
            for si,seed in enumerate((3101,7907)):
                for m in (32,128,256):
                    report['stage']=f'fixture {m}/{seed}/{pad_mib}'
                    data=_gen_input(m,seed,dev)
                    graphs={k:_mp_capture(mod,st,scratch,data) for k,mod in mods}
                    row=_sl_compare(graphs,data,flush,sink,pi+si,
                      dict(m=m,seed=seed,pad_mib=pad_mib,scratch_mod32m=scratch[1].data_ptr()%(32<<20)),True)
                    report['rows'].append(row);del graphs,data
        report.update(status='complete',stage='complete')
    except Exception as exc:
        report.update(status='failed',error=repr(exc)[:1000],trace=_MPTrace.format_exc(limit=4)[-2000:])
    finally:
        for s in st['scratch']:s.zero_()
        torch.cuda.synchronize();report['seconds']=_MPTime.monotonic()-start
        report['row_count']=len(report['rows']);report['row_format']='<24I12f4d'
        report['rows_b64']=_mp_pack_rows(report.pop('rows'))
        payload=_MP64.b64encode(_MPZlib.compress(_MPJSON.dumps(report,separators=(',',':')).encode(),9)).decode()
        print('MOE_STAGE_LOOP_Z '+payload,flush=True)
        print('MOE_STAGE_LOOP_SUMMARY '+_MPJSON.dumps(dict(status=report['status'],stage=report['stage'],
          rows=report['row_count'],nearby=len(report['nearby']),chars=len(payload),seconds=report['seconds']),separators=(',',':')),flush=True)


def prepare(weights):
    st=_sl_parent_prepare(weights)
    if st.get('sfx') and '-DOKC_WGS=1' in _main_defines(st['sfx']):_sl_probe(st)
    return st
'''
target.write_text(source+'\n'+variant+'\n'+register+'\n'+helpers+'\n'+suffix+'\n')
ast.parse(target.read_text())
print(target.relative_to(ROOT),hashlib.sha256(target.read_bytes()).hexdigest())
