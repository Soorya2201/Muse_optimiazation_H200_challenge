"""Exercise exact diagnostic builders, wire and FP64 oracle on CPU only."""
import ast,base64,copy,hashlib,json,random,struct,types,zlib
from pathlib import Path
root=Path(__file__).resolve().parents[1];track=root/'muse_moe_w4a16';p=track/'submission_v41_two_cta_probe.py';s=p.read_text();parent=(track/'submission.py').read_text();assert s.startswith(parent)
tree=ast.parse(s);env=dict(_TCHash=hashlib,_TCCopy=copy,_TCJSON=json,_TCStruct=struct,_TC64=base64,_TCZlib=zlib)
for n in tree.body:
 if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['CUDA_SRC','CPP_SRC','_TC_RESOURCES']:env[n.targets[0].id]=ast.literal_eval(n.value)
builds=[]
def stub(*args,**kwargs):builds.append((args,kwargs));return types.SimpleNamespace(okc_set_pdl=lambda v:None)
env['load_inline']=stub
for name in ['two_cta_native','_tc_build','_tc_wire']:
 n=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name);exec(compile(ast.Module(body=[n],type_ignores=[]),'<own exact host function>','exec'),env)
for new in (False,True):env['_tc_build'](new)
assert len(builds)==2
for args,b in builds:
 assert '-std=c++20' in b['extra_cflags'] and '-std=c++20' in b['extra_cuda_cflags']
 assert '-DOKC_WGS=1' in b['extra_cuda_cflags'] and '-DOKC_PRMTSC=1' in b['extra_cuda_cflags']
assert '-DOKC_NW=16' in builds[0][1]['extra_cuda_cflags'] and '-DOKC_NW=8' in builds[1][1]['extra_cuda_cflags']
assert 'cudaFuncAttributePreferredSharedMemoryCarveout,100' in env['_TC_RESOURCES']
probe=ast.get_source_segment(s,next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_tc_probe'))
assert probe.index("assert r[off+8]>=")<probe.index("g=_tc_capture")
assert 'grid *= 2;' in env['two_cta_native'](env['CUDA_SRC'])
rng=random.Random(0x2C7A)
def f32(v):return struct.unpack('<f',struct.pack('<f',v))[0]
def errors():return {k:[f32(rng.uniform(0,.001)),f32(rng.uniform(0,.009))] for k in ('old','new','a1')}
report=dict(status='complete',stage='complete',resources={k:[0,0,96,0,0,n,sm,0,occ]*2 for k,n,sm,occ in [('old',576,231168,1),('new',320,110976,2)]},rows=[],nearby=[])
for m in (17,33,65,129,257,384):report['nearby'].append(dict(m=m,error=errors(),fresh=errors(),change=f32(rng.random())))
report['fp64']={k:[f32(rng.uniform(0,.014)),f32(rng.uniform(0,.07))] for k in ('a0','old','new','a1')}
report['empty']=[dict(m=m,zero=True,fresh_nonzero=True) for m in (32,128,256)]
for pad in (0,1,3,7):
 for seed in (3101,7907):
  for m in (32,128,256):
   raw={k:[struct.unpack('<I',struct.pack('<f',rng.uniform(.01,1)))[0] for _ in range(8)] for k in ('a0','old','new','a1')}
   report['rows'].append(dict(m=m,seed=seed,pad_mib=pad,event_ms_bits=raw,error=errors(),fresh=errors(),change=f32(rng.random()),scratch_mod32m=rng.randrange(32<<20)))
wire=env['_tc_wire'](report);raw=zlib.decompress(base64.b64decode(wire));h=struct.unpack_from('<I',raw)[0];decoded=json.loads(raw[4:4+h]);assert len(raw)==4+h+180*24
values=[];metrics=[]
for i,row in enumerate(report['rows']):
 frame=struct.unpack_from('<32I13f',raw,4+h+180*i);want=[v for k in ('a0','old','new','a1') for v in row['event_ms_bits'][k]];assert list(frame[:32])==want;values+=want
 want=[v for key in ('error','fresh') for k in ('old','new','a1') for v in row[key][k]]+[row['change']];assert list(frame[32:])==want;metrics+=want
assert decoded['nearby']==report['nearby']
bad=copy.deepcopy(report);bad.update(status='failed',error=''.join(rng.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(2000)),error_tail=''.join(rng.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(700)),error_chars=10000,trace=''.join(rng.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(800)),comparison_arm='new',comparison_phase='fresh',finite={k:False for k in ('a0','old','new','a1')})
badwire=env['_tc_wire'](bad);assert len(wire)<12800 and len(badwire)<12800,(len(wire),len(badwire))
# Test the exact oracle against an independent full-dequantized FP64 formula.
import torch
env['torch']=torch
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_tc_fp64_basis');exec(compile(ast.Module(body=[fn],type_ignores=[]),'<own exact oracle>','exec'),env)
gen=torch.Generator().manual_seed(277);oracle_cases=0;maxerr=0
for d in (32,64):
 for m in (1,3,17):
  w13=torch.randint(0,256,(1,2*d,d//2),generator=gen,dtype=torch.uint8);w2=torch.randint(0,256,(1,d,d//2),generator=gen,dtype=torch.uint8)
  sf13=torch.randint(0,127,(1,2*d,d//16),generator=gen,dtype=torch.uint8).view(torch.float8_e4m3fn);sf2=torch.randint(0,127,(1,d,d//16),generator=gen,dtype=torch.uint8).view(torch.float8_e4m3fn)
  weights=dict(w13=w13,w2=w2,w13_sf=sf13,w2_sf=sf2,w13_gs=torch.tensor([.0001]),w2_gs=torch.tensor([.0001]),post_norm_w=torch.randn(d,generator=gen).bfloat16()*.1)
  x=torch.zeros(m,d,dtype=torch.bfloat16);col=(torch.arange(m)*13+3)%d;x[torch.arange(m),col]=4
  data=dict(hidden_states=x,topk_weights=torch.rand(m,8,generator=gen));out=env['_tc_fp64_basis'](data,weights)
  grid=torch.tensor([0,.5,1,1.5,2,3,4,6],dtype=torch.float64)
  def dq(p,sf,gs):
   q=torch.stack((p&15,p>>4),dim=-1).flatten(-2);return grid[(q&7).long()]*torch.where((q&8)!=0,-1.,1.).double()*sf.double().repeat_interleave(16,dim=-1)*gs.double()
  h=x.double()@dq(w13[0],sf13[0],weights['w13_gs'][0]).T;a,b=h.chunk(2,dim=1);y=(a*torch.sigmoid(a)*b)@dq(w2[0],sf2[0],weights['w2_gs'][0]).T;y*=1+weights['post_norm_w'].double();y*=torch.rsqrt(y.square().mean(-1,keepdim=True)+1e-8)*data['topk_weights'][:,0:1].double()
  error=float((out-y).abs().max());assert error<1e-12,error;maxerr=max(maxerr,error);oracle_cases+=1
# Synthetic fixtures also respect distinct top8 expert IDs in every row.
env.update(HIDDEN=4096,N_EXPERTS_TOTAL=256,TOP_K=8,LOCAL_EXPERTS=32,LOCAL_START=0,SHARDING='ep8')
for name in ('_gen_input','_tc_fixture'):
 fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
 exec(compile(ast.Module(body=[fn],type_ignores=[]),'<own fixture>','exec'),env)
fixture_cases=0
for m,mode in [(32,'empty'),(128,'empty'),(256,'empty'),(17,'basis'),(33,'mixed'),(65,'hot'),(129,'mixed'),(257,'mixed'),(384,'hot')]:
 d=env['_tc_fixture'](m,6180,'cpu',mode);ids=d['topk_ids'];assert ids.shape==(m,8)
 assert bool(((ids>=0)&(ids<256)).all());assert bool((ids.sort(-1).values[:,1:]!=ids.sort(-1).values[:,:-1]).all())
 if mode=='empty':assert bool((ids>=32).all())
 fixture_cases+=1
out=dict(scope='CPU-only exact builder/codec/oracle validation; no CUDA arithmetic, residency or runtime proof.',source_sha256=hashlib.sha256(s.encode()).hexdigest(),canonical_prefix_unchanged=True,exact_host_build_calls=len(builds),event_bits=len(values),exact_metric_frames=len(metrics),wire_chars=len(wire),failure_wire_chars=len(badwire),cpu_fp64_oracle_cases=oracle_cases,valid_unique_route_fixtures=fixture_cases,max_cpu_oracle_absolute_error=maxerr,pre_gpu_rejected_v37='Exact wire CPU execution raises KeyError from popping the event field inside a four-arm comprehension; V38 pops once before iteration. No native change.',pre_gpu_rejected_v36='Float8 advanced indexing precedes conversion; CPU reproduces NotImplementedError. V37 converts the scale tensor before indexing; native unchanged. CUDA support was not assumed from CPU failure.')
(track/'TWO_CTA_PROBE_PREFLIGHT_2026-11-01.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
