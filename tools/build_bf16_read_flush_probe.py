"""Match the judge's read-only cache eviction in the device-event protocol."""
import ast
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'muse_bf16_proj/submission_v40_device_event_probe.py'
target=ROOT/'muse_bf16_proj/submission_v41_read_flush_probe.py'
assert hashlib.sha256(base.read_bytes()).hexdigest()=='8a056167d53d6ee072e9d6d2a8e2a1f48e14e6199dc6023be41b41dece645250'
assert not target.exists()
s=base.read_text();t=ast.parse(s)
node=next(n for n in t.body if isinstance(n,ast.FunctionDef)and n.name=='_cp_measure')
old=ast.get_source_segment(s,node)
assert old.count('flush.add_(1)')==1
new=old.replace('    ns=[[] for _ in graphs];host=[[] for _ in graphs]',
    "    ns=[[] for _ in graphs];host=[[] for _ in graphs]\n"
    "    sink=torch.empty((),dtype=torch.float32,device=flush.device)\n"
    "    sink.copy_(flush.sum(dtype=torch.float32));torch.cuda.synchronize()")
assert new!=old
new=new.replace('inits[arm]();flush.add_(1)',
                'inits[arm]();sink.copy_(flush.sum(dtype=torch.float32))')
s=s.replace(old,new)
old="    report=dict(status='starting',cases=[],nearby=[],resources={},specs=_cp_specs(),rounds=8,"
assert s.count(old)==1;s=s.replace(old,old+"flush_kind='read_sum',")
old="        header=_CPJson.dumps(report,separators=(',',':')).encode()"
assert s.count(old)==1
s=s.replace(old,"        report['time_word_bytes']=2 if max(raw_times,default=0)<65536 else 4\n"+old)
old="_CPStruct.pack('<'+str(len(raw_times))+'I',*raw_times)"
assert s.count(old)==1
s=s.replace(old,"_CPStruct.pack('<'+str(len(raw_times))+('H' if report['time_word_bytes']==2 else 'I'),*raw_times)")
ast.parse(s);target.write_text(s)
print(target.relative_to(ROOT),hashlib.sha256(target.read_bytes()).hexdigest())
