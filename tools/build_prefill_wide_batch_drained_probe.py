"""Immutable corrected diagnostic: drain wide batches; attribute numerical failures."""
import ast
import hashlib
import inspect
from pathlib import Path
from prefill_wide_batch_fenced_native import wide_batch_drained_native

root=Path(__file__).resolve().parents[1];track=root/'muse_prefill_layer'
parent=(track/'submission.py').read_text();source=(track/'submission_v33_wide_batch_probe.py').read_text()
assert source.startswith(parent)
native=next(ast.literal_eval(n.value) for n in ast.parse(parent).body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='_CUDA_SRC' for x in n.targets))
changed=wide_batch_drained_native(native);digest=hashlib.sha256(changed.encode()).hexdigest()
needle='def _ks_build(st,fast,new):'
source=source.replace(needle,inspect.getsource(wide_batch_drained_native)+'\n'+needle)
source=source.replace('core=wide_batch_native(_CUDA_SRC) if new else _CUDA_SRC','core=wide_batch_drained_native(_CUDA_SRC) if new else _CUDA_SRC')
source=source.replace('e5b575b5edbaf30e1f88c925c2b0b0ee83579303b45c22f6334177bc16903208',digest)
source=source.replace("'prefill_wide_batch_'+tag", "'prefill_wide_batch_drained_'+tag")
old='''        def agreement(graphs):
            return {k:_ks_error(g[1],graphs['a0'][1]) for k,g in graphs.items() if k!='a0'}'''
new='''        def agreement(graphs):
            report['agreement_finite']={}
            for k,g in graphs.items():
                y=g[1].float()
                report['agreement_finite'][k]=dict(nan=int(torch.isnan(y).sum()),inf=int(torch.isinf(y).sum()))
            assert not any(report['agreement_finite']['a0'].values()),'nonfinite unchanged-parent reference'
            errors={}
            for k,g in graphs.items():
                if k=='a0':continue
                report['comparison_arm']=k
                errors[k]=_ks_error(g[1],graphs['a0'][1])
            report.pop('comparison_arm',None)
            return errors'''
assert source.count(old)==1;source=source.replace(old,new)
# Labels distinguish capture from the arm that actually failed comparison.
source=source.replace("graphs=captures(data);initial=agreement(graphs)","graphs=captures(data);report['agreement_phase']='initial';initial=agreement(graphs)")
source=source.replace("graphs=captures(data);errors=agreement(graphs)","graphs=captures(data);report['agreement_phase']='initial';errors=agreement(graphs)")
source=source.replace("replay(graphs);fresh=agreement(graphs)","replay(graphs);report['agreement_phase']='fresh';fresh=agreement(graphs)")
ast.parse(source);assert source.startswith(parent)
target=track/'submission_v34_wide_batch_drained_probe.py';assert not target.exists();target.write_text(source)
print(target.relative_to(root),hashlib.sha256(source.encode()).hexdigest(),'native',digest)
