"""Offline final-candidate transform, raw-event codec and primary-error retention."""
import ast
import base64
import contextlib
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import struct
import time
import traceback
import types
import zlib

from attention_integer_exp_initialized import cuda_helper

ROOT=Path(__file__).resolve().parents[1];TRACK=ROOT/'muse_attention_decode'
path=TRACK/'submission_v28_integer_exp_probe.py';source=path.read_text();tree=ast.parse(source)
assert source.startswith((TRACK/'submission.py').read_text())
env={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
     and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('CUDA_SRC','CPP_SRC','_AQ_POLY','_AQ_RESOURCE')}
assert env['_AQ_POLY']==cuda_helper(final_operands=True).replace('aq_integer_predicated','ex2_poly')
helper=env['_AQ_POLY']
assert helper.index('"@p ex2.approx.ftz.f32 z,%1;')<helper.index('"mov.f32 %0,z;')
assert helper.count('%0')==1
for name in ('_aq_source','_aq_measure','_aq_probe'):
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<inspected host functions>','exec'),env)
assert env['_aq_source'](0)==env['CUDA_SRC']+env['_AQ_RESOURCE']
for count,slots in ((1,2),(2,3)):
    native=env['_aq_source'](count)
    assert native.startswith('#define AQ_FMA_COUNT '+str(slots)+'\n')
    assert native.count('aq_exp<FLAT, ')==16
    for slot in range(8):assert native.count('aq_exp<FLAT, '+str(slot)+'>')==2

state={'arm':0,'count':0}
class Event:
    def __init__(self,enable_timing):pass
    def record(self):pass
    def synchronize(self):pass
    def elapsed_time(self,other):
        state['count']+=1
        return struct.unpack('<f',struct.pack('<f',.1*(1+.03*state['arm'])+.000032*(state['count']%17)))[0]
class Graph:
    def __init__(self,i):self.i=i
    def replay(self):state['arm']=self.i
class Flush:
    def add_(self,n):pass
env.update(torch=types.SimpleNamespace(cuda=types.SimpleNamespace(Event=Event)),
           _AQStruct=struct,_AQStats=statistics,_AQMath=math)
cases=[]
for group in range(16):
    record=env['_aq_measure']([(Graph(i),None) for i in range(4)],Flush(),group)
    raw=record['event_ms_float32_bits'];assert len(raw)==4 and all(len(a)==9 for a in raw)
    us=[[struct.unpack('<f',struct.pack('<I',x))[0]*1000 for x in a] for a in raw]
    assert round(statistics.median(a/b for a,b in zip(us[0],us[-1])),6)==record['aa']
    ratios=[statistics.median(math.sqrt(a*c)/b for a,b,c in zip(us[0],us[i],us[-1])) for i in (1,2)]
    assert [round(x,6) for x in ratios]==record['r']
    cases.append(record)
assert state['count']==576
payload=base64.b64encode(zlib.compress(json.dumps(dict(cases=cases),separators=(',',':')).encode(),9)).decode()
assert len(payload)<8000

def primary_failure(*args):raise RuntimeError('injected_primary_build_error')
class Buffer:
    def zero_(self):raise RuntimeError('injected_secondary_cleanup_error')
env.update(_AQTime=time,_AQBase64=base64,_AQZlib=zlib,_AQJson=json,_AQTraceback=traceback,
           _aq_build=primary_failure,_MOD=None,_WS={0:[Buffer()]})
output=io.StringIO()
with contextlib.redirect_stdout(output):env['_aq_probe']()
lines=output.getvalue().splitlines();payloads=[s.split('ATTN_PK2_HYBRID_Z ',1)[1] for s in lines if s.startswith('ATTN_PK2_HYBRID_Z ')]
assert len(payloads)==1
failure=json.loads(zlib.decompress(base64.b64decode(payloads[0])))
assert failure['stage']=='build_parent' and failure['status']=='diagnostic_failed'
assert 'injected_primary_build_error' in failure['error']
assert 'injected_secondary_cleanup_error' in failure['cleanup_error']
assert 'injected_primary_build_error' in failure['traceback']

report=dict(scope='CPU host/source/codec tests only; no GPU validation.',source_sha256=hashlib.sha256(source.encode()).hexdigest(),
    canonical_prefix_unchanged=True,native_source_transforms=3,slot_counts=[8,12],values_per_thread=32,
    measurement_groups=16,event_float32_values=576,event_bit_roundtrip_exact=True,ratios_reproduced=True,
    synthetic_cases_payload_chars=len(payload),primary_and_secondary_errors_both_preserved=True,
    asm_output_written_only_after_all_input_reads=True,failure_probe=failure)
target=TRACK/'INTEGER_EXP_FINAL_HOST_PREFLIGHT_2026-10-31.json'
if target.exists():assert json.loads(target.read_text())['source_sha256']==report['source_sha256']
target.write_text(json.dumps(report,indent=2)+'\n')
print('PASS V28:3transforms/576event values/primary+secondary errors/operand lifetimes')
