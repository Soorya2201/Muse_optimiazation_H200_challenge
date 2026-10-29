"""Offline integration checks; does not claim CUDA correctness or performance."""
import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('builder', root / 'tools/build_rank78_m1_hybrid.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
source = (root / 'submission_v33_chen_m1_hybrid.py').read_text()
assert source.rstrip() == builder.build().rstrip()
assert source.startswith(builder.PARENT.read_text())
tree = ast.parse(source)
donor = ast.parse(builder.DONOR.read_text())

def value(tree, name):
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))

for name in ('_V30M1_CPP', '_V30M1_CUDA', '_V30M1_CONFIG'):
    assert value(tree, name) == value(donor, name)

def function(name, scope):
    node = next(n for n in reversed(tree.body) if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), name, 'exec'), scope)
    return scope[name]

class X:
    shape = (1, 8192)
    is_cuda = True
    dtype = 'bf16'
    device = 'cuda'
    def is_contiguous(self): return True
    def data_ptr(self): return 16

torch = SimpleNamespace(bfloat16='bf16')
scope = {'torch': torch, '_v30m1_launch': lambda *a: 'new', '_V33_PARENT_KERNEL': lambda *a: 'parent'}
dispatch = function('kernel_fn', scope)
state = {'v33_choices': {'qkv_gate'}, 'v33_m1': {
    'plans': {'qkv_gate': {'n': 2304, 'k': 8192}}, 'router': SimpleNamespace(device='cuda')}}
x = X()
data = {'x': x, 'config': {'layer': 'qkv_gate', 'n': 2304}}
assert dispatch(data, state) == 'new'
for shape in ((7, 8192), (16, 8192), (1, 4096)):
    x.shape = shape
    assert dispatch(data, state) == 'parent'
x.shape = (1, 8192)
data['config']['n'] = 1024
assert dispatch(data, state) == 'parent'
data['config']['n'] = 2304
x.is_cuda = False
assert dispatch(data, state) == 'parent'
x.is_cuda = True
state['v33_choices'] = set()
assert dispatch(data, state) == 'parent'

# Exercise the timing gate with fixed paired-result fixtures. A one-round win,
# a tie, nonfinite data, or a budget skip must never enable the candidate.
class Generator:
    def manual_seed(self, seed): return self

torch.Generator = lambda **kw: Generator()
torch.randn = lambda *a, **kw: X()
weight = SimpleNamespace(shape=(2304, 8192), device='cuda')
for ratios, expected in (((1.05, 1.04), True), ((1.05, 1.0), False),
                         ((1.02, 1.02), False), ((float('nan'), 1.1), False)):
    samples = iter(ratios * 2)
    scope = {'torch': torch, 'math': math, 'time': SimpleNamespace(time=lambda: 10), '_T0': 0,
             '_paired': lambda *a, **kw: (next(samples), 10, 9, 4),
             '_v30m1_launch': lambda *a: None, '_V33_PARENT_KERNEL': lambda *a: None}
    gate = function('_v33_gate', scope)
    state = {'v33_m1': {'plans': {'qkv_gate': {'n': 2304}}}}
    gate({'qkv_gate': weight, 'router': weight}, state)
    assert ('qkv_gate' in state['v33_choices']) == expected
scope['time'] = SimpleNamespace(time=lambda: 276)
state = {'v33_m1': {'plans': {'qkv_gate': {'n': 2304}}}}
gate({'qkv_gate': weight}, state)
assert state['v33_choices'] == set()
print('PASS: deterministic provenance, unchanged CUDA, dispatch, two-round margin, nonfinite and budget gates')
