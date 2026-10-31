"""Immutable global-normalization probe; canonical timed entry stays intact."""
import ast
import argparse
import hashlib
import inspect
from pathlib import Path

from bf16_global_normalization import DECODE, metadata, normalize_body, DECODE_LUT, metadata_lut, normalize_body_lut

ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'muse_bf16_proj/submission_v41_read_flush_probe.py'
parser=argparse.ArgumentParser()
parser.add_argument('--fused-lut',action='store_true')
args=parser.parse_args()
version=43 if args.fused_lut else 42
target=ROOT/f'muse_bf16_proj/submission_v{version}_global_normalization_probe.py'
assert hashlib.sha256(base.read_bytes()).hexdigest()=='53433583563e8a9227a12dfde2786c705714be720419ff016e5ec0d597eaeef0'
assert not target.exists()
s=base.read_text();t=ast.parse(s)
n=next(n for n in t.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='_CP_DECODE')
old=ast.get_source_segment(s,n)
fn=metadata_lut if args.fused_lut else metadata
bodyfn=normalize_body_lut if args.fused_lut else normalize_body
extra='\n\n'+inspect.getsource(fn).replace('def '+fn.__name__+'(', 'def _gn_metadata(')
extra+='\n\n'+inspect.getsource(bodyfn).replace('def '+bodyfn.__name__+'(', 'def _gn_body(')
s=s.replace(old,'_CP_DECODE = '+repr(DECODE_LUT if args.fused_lut else DECODE)+extra,1)
changes={
    "            if family:\n                body = _cp_escape_body(body)":
    "            if packed:\n                body = _gn_body(body)\n            if family:\n                body = _cp_escape_body(body)",
    "    x=data['x'];m,k=x.shape;n=data['config']['n']\n    out=torch.empty":
    "    if packed:b4=_gn_metadata(b4)\n    x=data['x'];m,k=x.shape;n=data['config']['n']\n    out=torch.empty",
    'okc_bf16_c12_permute39_90a':f'okc_bf16_global_normalization{version}_90a',
    "print('BF16_C12_DEVICE_Z '+payload,flush=True)":"print('BF16_C12_NORM_Z '+payload,flush=True)",
    "        specs=report['specs'];mods=_CP_MODULE":
    "        specs=report['specs'];mods=_CP_MODULE\n"
    "        report['normalization']='Global fixed-window main dot; partials unscaled before unchanged original-unit escapes.'\n"
    "        report['bases']={str(v['id']):_cp_pack(state,v)[3]&255 for v in specs}\n"
    "        assert all(1<=b<=119 for b in report['bases'].values())",
}
for old,new in changes.items():
    assert s.count(old)==1,(old,s.count(old));s=s.replace(old,new)
assert s.startswith((ROOT/'muse_bf16_proj/submission.py').read_text())
ast.parse(s);target.write_text(s)
print(target.relative_to(ROOT),hashlib.sha256(target.read_bytes()).hexdigest())
