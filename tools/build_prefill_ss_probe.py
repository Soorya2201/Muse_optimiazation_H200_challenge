"""Build an isolated same-job diagnostic; canonical timed dispatch is unchanged."""
import ast
import argparse
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--version',type=int,choices=[28,29],default=29);args=parser.parse_args()
parent=(ROOT/'muse_prefill_layer/submission.py').read_text()
assert hashlib.sha256(parent.encode()).hexdigest()=='8b52dedda281b2e642bda1d9333bcf2d002fc8611e25caa10e90ed2a849021a5'
prior=(ROOT/'muse_prefill_layer/submission_v27_scale_share_probe.py').read_text()
tree=ast.parse(prior)
def function(name):
    return ast.get_source_segment(prior,next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name))+'\n'

append=r'''
# Independent synthetic SS pipeline diagnostic. Timed kernel_fn is unchanged.
import base64 as _KS64
import copy as _KSCopy
import json as _KSJSON
import random as _KSRng
import struct as _KSStruct
import time as _KSTime
import traceback as _KSTrace
import zlib as _KSZlib
'''
append+=(ROOT/'tools/prefill_ss_native.py').read_text()+'\n'
append+='\n_ks_parent_prepare = prepare\n'
append+=r'''
_SS_EXTRA_CPP=r"""
std::vector<int64_t> ss_resources();
void ss_move_raw(const void*,void*,cudaStream_t);
void ss_move(torch::Tensor in,torch::Tensor out) {
  ss_move_raw(in.data_ptr(),out.data_ptr(),c10::cuda::getCurrentCUDAStream());
}
"""
_SS_RESOURCES=r"""
#include <vector>
std::vector<int64_t> ss_resources() {
  std::vector<int64_t> out;
  for (int fine=0;fine<2;++fine) {
    cudaFuncAttributes a={};
    cudaError_t e=fine?cudaFuncGetAttributes(&a,okc_moe::moe_fused_persistent<1>):
                     cudaFuncGetAttributes(&a,okc_moe::moe_fused_persistent<0>);
    out.insert(out.end(),{(int64_t)e,(int64_t)a.numRegs,(int64_t)a.localSizeBytes,
                         (int64_t)a.sharedSizeBytes,(int64_t)a.maxThreadsPerBlock});
  }
  return out;
}
"""
_SS_MOVE=r"""
namespace okc_moe {
__global__ void ss_move_check(const uint32_t* in,uint32_t* out) {
  __shared__ __align__(1024) uint32_t memory[4096];
  for(int i=threadIdx.x;i<4096;i+=64) memory[i]=i<1152?in[i]:0xa5a5a5a5u;
  __syncthreads();
  ss_expand(reinterpret_cast<uint8_t*>(memory),threadIdx.x);
  __syncthreads();
  for(int i=threadIdx.x;i<4096;i+=64) out[i]=memory[i];
}
}
void ss_move_raw(const void* in,void* out,cudaStream_t stream) {
  okc_moe::ss_move_check<<<1,64,0,stream>>>((const uint32_t*)in,(uint32_t*)out);
}
"""
def _ks_build(st,fast,new):
    native=(shared_weight_native(_CUDA_SRC,184,True) if new else _CUDA_SRC)+GLUE+_SS_RESOURCES
    native+=_SS_MOVE if new else '\nvoid ss_move_raw(const void*,void*,cudaStream_t) {}\n'
    cpp=CPP+_SS_EXTRA_CPP
    if not fast:native='#define MOE_FAST_SCALE 0\n'+native
    tag=hashlib.sha256((cpp+native).encode()).hexdigest()[:16]
    functions=['qk_prep','moe_route','moe_plan','moe_gather','moe_gather_plan','moe_combine',
               'moe_gemm','moe_init','moe_gemm_p','moe_fused','boundary','boundary_rc',
               'kv_gather','attn_prep','attn_fwd','attn_init','attn_set_prof','ss_resources','ss_move']
    ext=load_inline('prefill_ss_dequant_'+tag,cpp_sources=cpp,cuda_sources=native,functions=functions,
        extra_cuda_cflags=['-O3','-gencode=arch=compute_90a,code=sm_90a','-std=c++20','--use_fast_math'],verbose=False)
    ext.moe_init(st['smem'])
    return ext

def _ss_movement(new,st,weights):
    dev=st['w13r'].device
    lut=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.],device=dev)
    rows=torch.arange(128,device=dev);k=torch.arange(64,device=dev)
    physical=k[None,:]^((rows[:,None]&7)*8)
    tests=[]
    for gated,key in ((True,'w13'),(False,'w2')):
        perm=rows if not gated else torch.where((rows%16)<8,(rows//16)*8+rows%16,4096+(rows//16)*8+rows%16-8)
        packed=st['w13r'] if gated else st['w2r']
        for ks in (0,1,17,63):
            out=torch.empty((16384,),device=dev,dtype=torch.uint8)
            new.ss_move(packed[0,0,ks],out)
            c=_codes(weights[key][0,perm,ks*32:(ks+1)*32]).long()
            scales=weights[key+'_sf'][0].float()[perm,ks*4:(ks+1)*4].repeat_interleave(16,dim=1)
            expected=(lut[c]*scales/128).to(torch.bfloat16)
            actual=out.view(torch.bfloat16).reshape(128,64).gather(1,physical)
            assert torch.equal(actual.view(torch.int16),expected.view(torch.int16)),('raw SS mapping',key,ks)
            tests.append([key,ks])
    return tests
'''
for name in ('_ks_fixture','_ks_call','_ks_capture','_ks_error'):
    append+=function(name)+'\n'
probe=function('_ks_probe')
probe=probe.replace('def _ks_probe(', 'def _ss_probe(')
probe=probe.replace('        assert MOE_MODE_SMALL==MOE_MODE_LARGE==\'fused\' and GATHER_PLAN',
                    '        assert MOE_MODE_SMALL==MOE_MODE_LARGE==\'fused\' and GATHER_PLAN\n        assert st[\'moe_flags\']==0 and st[\'smem\']==232448')
probe=probe.replace('.k64_resources()', '.ss_resources()')
a=probe.index("        report['stage']='movement test'")
b=probe.index('        control=dict(st,mod=old',a)
probe=probe[:a]+'''        report['stage']='raw weight movement'
        dev=st['w13r'].device
        report['movement_cases']=_ss_movement(new,st,weights)
        # Actual and matched arms use the same logical weights. Matched
        # old/new share one pointer pair; no format copies are necessary.
        slot=(st['w13r'].clone(),st['w2r'].clone())
'''+probe[b:]
a=probe.index('        def set_arm(name):');b=probe.index('        def captures(data):',a)
probe=probe[:a]+'        def set_arm(name): pass\n'+probe[b:]
probe=probe.replace('PREFILL_SCALE_SHARE_Z','PREFILL_SS_PIPE_Z').replace('PREFILL_SCALE_SHARE_SUMMARY','PREFILL_SS_PIPE_SUMMARY')
probe=probe.replace('            data=_ks_fixture(m,seed,dev,mode)',
                    "            report['stage']=f'nearby/{m}/{mode}'\n            data=_ks_fixture(m,seed,dev,mode)")
probe=probe.replace('            for k,state in arms.items():set_arm(k);graphs[k]=_ks_capture(state,data)',
                    "            for k,state in arms.items():\n                report['active_arm']=k\n                print('PREFILL_SS_STAGE '+report['stage']+' arm='+k,flush=True)\n                set_arm(k);graphs[k]=_ks_capture(state,data)")
probe=probe.replace("    finally:\n        torch.cuda.synchronize()\n        report['seconds']", """    finally:
        try:
            torch.cuda.synchronize()
        except Exception as sync_error:
            report['status']='failed'
            report['secondary_sync_error']=repr(sync_error)[:1000]
        report['seconds']""")
append+=probe+'\n'
append+='''
def prepare(weights):
    st=_ks_parent_prepare(weights)
    _ss_probe(st,weights)
    assert st['attn_choice']==ATTN_FIXED
    return st
'''
result=parent+'\n'+append
ast.parse(result)
assert result.startswith(parent)
target=ROOT/f'muse_prefill_layer/submission_v{args.version}_ss_pipeline_probe.py'
assert not target.exists()
target.write_text(result)
print(target.relative_to(ROOT),hashlib.sha256(result.encode()).hexdigest())
