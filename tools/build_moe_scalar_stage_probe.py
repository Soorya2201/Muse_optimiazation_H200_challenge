"""Bounded repair: remove the extra initialization MMA/slot wait, retain controls."""
import ast
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'muse_moe_w4a16/submission_v31_stage_loop_probe.py';source=base.read_text()
assert hashlib.sha256(source.encode()).hexdigest()=='0453862337d3ba2f04535c470ed1edc66c37207cb33f6839b6b02a7783a11363'
target=ROOT/'muse_moe_w4a16/submission_v32_scalar_stage_probe.py';assert not target.exists()
tree=ast.parse(source)
def function(name):return ast.get_source_segment(source,[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name][-1])
raw=(ROOT/'tools/moe_stage_compact_variant.py').read_text();t=ast.parse(raw)
helper=ast.get_source_segment(raw,next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='scalar_zero_stage'))
helper=helper.replace('scalar_zero_stage','_sl_scalar_zero_stage').replace('zero_mma_stage','_sl_zero_mma_stage')
build=function('_sl_build').replace('if compact:native=_sl_zero_mma_stage(native)','if compact:native=_sl_scalar_zero_stage(native)')
assert build!=function('_sl_build')
probe=function('_sl_probe').replace('REG112 + zero-MMA uniform stage loop','REG112 + scalar-zero uniform stage loop')
assert probe!=function('_sl_probe')
target.write_text(source+'\n'+helper+'\n'+build+'\n'+probe+'\n')
ast.parse(target.read_text())
print(target.relative_to(ROOT),hashlib.sha256(target.read_bytes()).hexdigest())
