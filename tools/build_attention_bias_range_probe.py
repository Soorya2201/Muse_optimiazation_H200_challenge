"""Isolate scalar bias-based range reduction from the slow floor50 control."""
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parent = ROOT / 'muse_attention_decode/submission_v22_pk2_underflow_fix.py'
target = ROOT / 'muse_attention_decode/submission_v23_bias_range_probe.py'
assert hashlib.sha256(parent.read_bytes()).hexdigest() == 'e2ca7625a773febb740b7512d524f805192982afcd5b4df0cd221370742f6522'
assert not target.exists()
source = parent.read_text()
tree = ast.parse(source)
node = next(n for n in tree.body if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name) and n.targets[0].id == '_AQ_POLY')
floor_poly = ast.literal_eval(node.value)
bias_poly = floor_poly.replace('  const float j = floorf(x);\n  const float f = x - j;',
    '  const float jm = __fadd_rd(x, 12582912.f);\n'
    '  const float j = __fadd_rn(jm, -12582912.f);\n'
    '  const float f = __fsub_rn(x, j);')
assert bias_poly != floor_poly
bias_poly = bias_poly.replace('return __int_as_float(__float_as_int(p) + ((int)j << 23));',
                             'return __uint_as_float(__float_as_uint(p) + (__float_as_uint(jm) << 23));')
assert 'floorf' not in bias_poly and '(int)j' not in bias_poly
notice = '\n# Scalar range-reduction design follows FlashAttention, pinned4616f51.\n'
notice += '# Copyright (c) 2025, Tri Dao. BSD-3-Clause terms retained below.\n'
notice += '\n'.join('# ' + l for l in (ROOT / 'research/public/flash_attention_4616f51/LICENSE').read_text().splitlines())
notice += '\n# AUTHORS: ' + (ROOT / 'research/public/flash_attention_4616f51/AUTHORS').read_text().replace('\n', '; ') + '\n'
replacement = ast.get_source_segment(source, node) + notice + '_AQ_BIAS_POLY = ' + repr(bias_poly)
source = source.replace(ast.get_source_segment(source, node), replacement, 1)
changes = {
    '    helper = _AQ_POLY + r"""':
        '    helper = (_AQ_POLY if count == 1 else _AQ_BIAS_POLY) + r"""',
    "    return '#define AQ_FMA_COUNT ' + str(count) + '\\n' + source + _AQ_RESOURCE":
        "    return '#define AQ_FMA_COUNT 2\\n' + source + _AQ_RESOURCE",
    "    report = dict(status='starting', resources={}, nearby=[], cases=[])":
        "    report = dict(status='starting', resources={}, nearby=[], cases=[],\n"
        "                  variant_names={'0': 'parent', '1': 'floor50', '2': 'bias50'})",
    '                    record = _aq_measure(graphs, flush, group + pad)\n':
        '                    record = _aq_measure(graphs, flush, group + pad)\n'
        '                    record["range_error"] = _aq_error(graphs[2][1], graphs[1][1])\n'
        '                    assert record["range_error"][0] < .002 and record["range_error"][1] < .01, "range-reduction outputs disagree"\n'
        '                    record["bias_vs_floor"] = round(_AQStats.median(a / b for a, b in zip(record["us"][1], record["us"][2])), 6)\n',
}
for old, new in changes.items():
    assert source.count(old) == 1, old
    source = source.replace(old, new)
source = source.replace('okc_attn_pk2_hybrid_v322_', 'okc_attn_pk2_hybrid_v323_')
ast.parse(source)
target.write_text(source)
print(target.relative_to(ROOT), hashlib.sha256(source.encode()).hexdigest())
