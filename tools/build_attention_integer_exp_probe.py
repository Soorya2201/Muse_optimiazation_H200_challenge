"""Build an immutable25/37.5-percent integer-exp diagnostic; timed path parent."""
import argparse
import ast
import hashlib
from pathlib import Path

from attention_integer_exp_predicated import cuda_helper

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--initialized',action='store_true')
parser.add_argument('--safe-report',action='store_true')
parser.add_argument('--final-operands',action='store_true')
args=parser.parse_args()
assert not args.safe_report or args.initialized
assert not args.final_operands or args.safe_report
if args.initialized:
    from attention_integer_exp_initialized import cuda_helper as initialized_helper
    def cuda_helper():return initialized_helper(final_operands=args.final_operands)

ROOT=Path(__file__).resolve().parents[1]
TRACK=ROOT/'muse_attention_decode'
base=TRACK/'submission_v24_bit_exp_probe.py'
version=28 if args.final_operands else (27 if args.safe_report else (26 if args.initialized else 25))
target=TRACK/f'submission_v{version}_integer_exp_probe.py'
assert not target.exists()
assert hashlib.sha256(base.read_bytes()).hexdigest()=='3eed711fdf8586a7e92f2996c5f2740acc737d7ff737debe66c0c492cc1bb2e6'
source=base.read_text();tree=ast.parse(source)
node=next(n for n in tree.body if isinstance(n,ast.Assign) and
          isinstance(n.targets[0],ast.Name) and n.targets[0].id=='_AQ_POLY')
source=source.replace(ast.get_source_segment(source,node),
                      '_AQ_POLY = '+repr(cuda_helper().replace('aq_integer_predicated','ex2_poly')),1)
changes={
    '    if (x > 120.f) return ex2(x);\n':'',
    "'aq_exp<FLAT, ' + str(slot) + '>(fmaf('":
        "'aq_exp<FLAT, ' + str(slot + (4 if changed >= 2 else 0)) + '>(fmaf('",
    "str(2 * count) + '\\n' + source + _AQ_RESOURCE":
        "str(2 if count == 1 else 3) + '\\n' + source + _AQ_RESOURCE",
    'okc_attn_pk2_bit_v324_':f'okc_attn_integer_v3{version}_',
    "{'0': 'parent', '1': 'bit50', '2': 'bit100'}":
        "{'0': 'parent', '1': 'integer25', '2': 'integer375'}",
    'import zlib as _AQZlib':'import zlib as _AQZlib\nimport struct as _AQStruct',
    '    times = [[] for _ in graphs]':'    times = [[] for _ in graphs]\n    raw_ms_bits = [[] for _ in graphs]',
    '            times[i].append(a.elapsed_time(b) * 1000)':
        "            ms = a.elapsed_time(b)\n            times[i].append(ms * 1000)\n"
        "            raw_ms_bits[i].append(_AQStruct.unpack('<I', _AQStruct.pack('<f', ms))[0])",
    '    return dict(us=[[round(x, 3) for x in v] for v in times],':
        '    return dict(event_ms_float32_bits=raw_ms_bits,',
    "        for batch, ctx, page, scale, pairs in ((2, 6000, 1, 1., 1),":
        "        assert all(a[1] == 0 for a in report['resources'].values()), report['resources']\n"
        "        for batch, ctx, page, scale, pairs in ((2, 6000, 1, 1., 1),",
    'Independent bit-exp derivation; primary paper: https://schraudolph.org/pubs/Schraudolph99.pdf':
        'Independent integer evaluator of known endpoint quadratic; prior art https://arxiv.org/html/2610.00049v1#A2.SS1',
    '# Prior polynomial remains in v22; this helper does not copy forum/gist implementation code.':
        '# Integer evaluator, not a novel polynomial family; native high tail explicitly predicated.',
}
for old,new in changes.items():
    assert old in source,old
    source=source.replace(old,new)
if args.safe_report:
    repairs={
        'import struct as _AQStruct':'import struct as _AQStruct\nimport traceback as _AQTraceback',
        '        modules = [_aq_build(0)]':
            "        report['stage'] = 'build_parent'\n"
            "        print('ATTN_INTEGER_STAGE build_parent', flush=True)\n"
            '        modules = [_aq_build(0)]',
        '        modules.extend(_aq_build(i) for i in (1, 2))':
            "        report['stage'] = 'build_integer_arms'\n"
            "        print('ATTN_INTEGER_STAGE build_integer_arms', flush=True)\n"
            '        modules.extend(_aq_build(i) for i in (1, 2))',
        "        assert all(a[1] == 0 for a in report['resources'].values()), report['resources']":
            "        assert all(a[1] == 0 for a in report['resources'].values()), report['resources']\n"
            "        report['stage'] = 'nearby'\n"
            "        print('ATTN_INTEGER_STAGE nearby', flush=True)",
        "        flush = torch.zeros(64 << 20, dtype=torch.float32, device='cuda')":
            "        report['stage'] = 'timing'\n"
            "        print('ATTN_INTEGER_STAGE timing', flush=True)\n"
            "        flush = torch.zeros(64 << 20, dtype=torch.float32, device='cuda')",
        "        report['status'] = 'complete'":
            "        report['status'] = 'complete'\n        report['stage'] = 'complete'",
        "        report['error'] = type(error).__name__ + ': ' + str(error)[-1500:]":
            "        report['error'] = type(error).__name__ + ': ' + str(error)[-1500:]\n"
            "        report['traceback'] = _AQTraceback.format_exc()[-3000:]",
        '        for buffers in _WS.values():\n            for buffer in buffers:\n                buffer.zero_()\n        torch.cuda.synchronize()':
            '        try:\n            for buffers in _WS.values():\n'
            '                for buffer in buffers:\n                    buffer.zero_()\n'
            '            torch.cuda.synchronize()\n        except Exception as cleanup_error:\n'
            "            report['cleanup_error'] = type(cleanup_error).__name__ + ': ' + str(cleanup_error)[-800:]\n"
            "            report['status'] = 'diagnostic_failed'",
    }
    for old,new in repairs.items():
        assert old in source,old
        source=source.replace(old,new)
parent=(TRACK/'submission.py').read_text()
assert source.startswith(parent)
ast.parse(source)
# Evaluate only literal CUDA strings and the pure source transform, no module imports.
nodes=ast.parse(source).body
env={n.targets[0].id:ast.literal_eval(n.value) for n in nodes if isinstance(n,ast.Assign)
     and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('CUDA_SRC','_AQ_POLY','_AQ_RESOURCE')}
fn=next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name=='_aq_source')
exec(compile(ast.Module(body=[fn],type_ignores=[]),'<pure native source transform>','exec'),env)
assert env['_aq_source'](0)==env['CUDA_SRC']+env['_AQ_RESOURCE']
for count,expected in ((1,2),(2,3)):
    generated=env['_aq_source'](count)
    assert generated.startswith('#define AQ_FMA_COUNT '+str(expected)+'\n')
    assert generated.count('setp.gt.f32 p,%1,120.0;')==1
    # Four occurrences per slot over the four statements; both halves share it.
    for slot in range(8):assert generated.count('aq_exp<FLAT, '+str(slot)+'>')==2
    assert 'if (x > 120.f)' not in generated
    assert generated.count('constexpr bool PK2 = PCHK;')==1
    # Replacing the sixteen calls is confined to pack2, preserving every tail.
    start=generated.index('        auto pack2 = [&]()')
    stop=generated.index('        pack2();',start)
    assert generated[stop:-len(env['_AQ_RESOURCE'])]==env['CUDA_SRC'][env['CUDA_SRC'].index('        pack2();'):]
target.write_text(source)
print(target.relative_to(ROOT),hashlib.sha256(source.encode()).hexdigest())
