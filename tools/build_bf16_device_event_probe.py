"""Repair synthetic measurement order and capture device-side timestamp nodes."""
import ast
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'muse_bf16_proj/submission_v39_c12_permute_probe.py'
target=ROOT/'muse_bf16_proj/submission_v40_device_event_probe.py'
assert hashlib.sha256(base.read_bytes()).hexdigest()=='99f9deba704a7bf85e46dfe0e162e2feff49db8a18902f8bf565251ff3ea0a38'
assert not target.exists()
s=base.read_text();tree=ast.parse(s)
replacements={
'_cp_capture':r'''
def _cp_capture(fn,init):
    init();fn();torch.cuda.synchronize()
    a=torch.cuda.Event(enable_timing=True,external=True)
    b=torch.cuda.Event(enable_timing=True,external=True)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        a.record();out=fn();b.record()
    g.replay();b.synchronize()
    return g,out,a,b
''',
'_cp_measure':r'''
def _cp_measure(graphs,inits,flush,group,calibrate=False):
    ns=[[] for _ in graphs];host=[[] for _ in graphs]
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    if calibrate:a.record();b.record();b.synchronize()
    for rep in range(8):
        order=[(v+(rep+group)%4)%4 for v in (0,1,3,2)]
        if rep//4%2:order.reverse()
        for arm in order:
            inits[arm]();flush.add_(1)
            if calibrate:a.record()
            graphs[arm][0].replay()
            if calibrate:b.record();b.synchronize()
            else:graphs[arm][3].synchronize()
            ns[arm].append(round(graphs[arm][2].elapsed_time(graphs[arm][3])*1e6))
            if calibrate:host[arm].append(round(a.elapsed_time(b)*1e6))
    return ns,host if calibrate else None
'''}
for name,replacement in replacements.items():
    tree=ast.parse(s)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    segment=ast.get_source_segment(s,node)
    assert s.count(segment)==1;s=s.replace(segment,replacement.strip())
changes={
"    report=dict(status='starting',cases=[],nearby=[],resources={},specs=_cp_specs(),":
"    report=dict(status='starting',cases=[],nearby=[],resources={},specs=_cp_specs(),rounds=8,",
"                timing='Five rotated cold-L2 rounds. Arms parent,matched-old,new,parent. Old/new use identical slot pointer; copy outside events and flush after copy. Raw event nanoseconds.')":
"                timing='Eight position/adjacency-balanced cold rounds. Timestamp nodes inside each synthetic CUDA graph, external=True. Arms parent,matched-old,new,parent; same old/new slot. Raw device ns; selected simultaneous host calibration.')",
"        flush=torch.zeros(64<<20,dtype=torch.float32,device='cuda');retained=[]":
"        flush=torch.zeros(64<<20,dtype=torch.float32,device='cuda');retained=[]\n"
"        noop=[_cp_capture(lambda:None,lambda:None) for _ in range(4)]\n"
"        nn,hh=_cp_measure(noop,[lambda:None]*4,flush,0,True)\n"
"        report['event_only']={'ns':nn,'host_ns':hh}\n"
"        del noop",
"                    ns=_cp_measure(graphs,inits,flush,pad+group+v['id'])":
"                    ns,host=_cp_measure(graphs,inits,flush,pad+group+v['id'],pad==0 and group==0)",
"                        identical=same,slot=slot.data_ptr()%(32<<20)))":
"                        identical=same,slot=slot.data_ptr()%(32<<20),host_ns=host))",
"        payload=_CPB64.b64encode(_CPZlib.compress(_CPJson.dumps(report,separators=(',',':')).encode(),9)).decode()":
"        import struct as _CPStruct\n"
"        raw_times=[n for c in report['cases'] for arm in c.pop('ns') for n in arm]\n"
"        header=_CPJson.dumps(report,separators=(',',':')).encode()\n"
"        wire=_CPStruct.pack('<I',len(header))+header+_CPStruct.pack('<'+str(len(raw_times))+'I',*raw_times)\n"
"        payload=_CPB64.b64encode(_CPZlib.compress(wire,9)).decode()",
"        print('BF16_C12_PERMUTE_Z '+payload,flush=True)":
"        print('BF16_C12_DEVICE_Z '+payload,flush=True)",
}
for old,new in changes.items():
    assert s.count(old)==1,old;s=s.replace(old,new)
ast.parse(s);target.write_text(s)
print(target.relative_to(ROOT),hashlib.sha256(target.read_bytes()).hexdigest())
