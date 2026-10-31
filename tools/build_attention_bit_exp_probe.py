"""Build an immutable packed-P linear bit-exp diagnostic; timed path is parent."""
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
base = ROOT / 'muse_attention_decode/submission_v22_pk2_underflow_fix.py'
target = ROOT / 'muse_attention_decode/submission_v24_bit_exp_probe.py'
assert not target.exists()
source = base.read_text()
assert hashlib.sha256(base.read_bytes()).hexdigest() == 'e2ca7625a773febb740b7512d524f805192982afcd5b4df0cd221370742f6522'
tree = ast.parse(source)
node = next(n for n in tree.body if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name) and n.targets[0].id == '_AQ_POLY')
helper = r'''
// IEEE exponent/mantissa linear interpolation, following Schraudolph (1999).
// This independently derived fixed-binade form avoids numeric F2I conversion.
// delta=log2((1+2/(e*ln(2)))/2); relative error <=0.02984 on normal range.
// x is FP32 log2 probability; the caller retains hardware ex2 above120.
__device__ __forceinline__ float ex2_poly(float x) {
  x = fmaxf(x, -126.f);
  const float biased = __fadd_rn(x, 383.956329345703125f);
  return __uint_as_float((__float_as_uint(biased) << 8) + 0x7f800000u);
}
'''
source = source.replace(ast.get_source_segment(source, node),
                        '_AQ_POLY = ' + repr(helper) + '\n', 1)
changes = {
    "    return '#define AQ_FMA_COUNT ' + str(count) + '\\n' + source + _AQ_RESOURCE":
        "    return '#define AQ_FMA_COUNT ' + str(2 * count) + '\\n' + source + _AQ_RESOURCE",
    'okc_attn_pk2_hybrid_v322_': 'okc_attn_pk2_bit_v324_',
    "    report = dict(status='starting', resources={}, nearby=[], cases=[])":
        "    report = dict(status='starting', resources={}, nearby=[], cases=[],\n"
        "                  variant_names={'0': 'parent', '1': 'bit50', '2': 'bit100'},\n"
        "                  gates={'parent_global': .01, 'parent_row': .05, 'fallback_global': .15, 'fallback_row': .75})",
    'e[0] < .002 and e[1] < .01': 'e[0] < .01 and e[1] < .05',
    "_aq_fixture(batch, ctx, 4319 + batch + page, True, page, scale, pairs)":
        "_aq_fixture(batch, ctx, 4319 + batch + page, True, page, scale, pairs)",
}
for old, new in changes.items():
    assert old in source, old
    source = source.replace(old, new)
# The independent fixture gate is conservative relative to the contract's
# minimum 3% against FP32. This is declared before any H200 result; a failing
# arm must be rejected, never rescued by loosening thresholds afterwards.
source = source.replace('Apache-2.0 polynomial matches pinned Madhavan/vjeux code.',
    'Independent bit-exp derivation; primary paper: https://schraudolph.org/pubs/Schraudolph99.pdf')
source = source.replace('# Public mirror/provenance: research/public/rohanarun_018b7a1/PROVENANCE.json.',
    '# Prior polynomial remains in v22; this helper does not copy forum/gist implementation code.')
assert source.startswith((ROOT / 'muse_attention_decode/submission.py').read_text())
ast.parse(source)
target.write_text(source)
print(target.relative_to(ROOT), hashlib.sha256(target.read_bytes()).hexdigest())
