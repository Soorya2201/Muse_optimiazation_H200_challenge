"""Repair the masked-logit edge case; preserve the archived failed probe."""
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parent = ROOT / 'muse_attention_decode/submission_v21_pk2_hybrid_probe.py'
target = ROOT / 'muse_attention_decode/submission_v22_pk2_underflow_fix.py'
assert hashlib.sha256(parent.read_bytes()).hexdigest() == '842ebfc29731e51aeefa3bd89eba7e462018c989d41c3bddb56f10ce81b4db4a'
assert not target.exists()
source = parent.read_text()
tree = ast.parse(source)
node = next(n for n in tree.body if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name) and n.targets[0].id == '_AQ_POLY')
poly = ast.literal_eval(node.value)
assert poly.count('x = fmaxf(x, -127.f);') == 1
fixed = poly.replace('x = fmaxf(x, -127.f);', 'x = fmaxf(x, -126.f);')
source = source.replace(ast.get_source_segment(source, node), '_AQ_POLY = ' + repr(fixed), 1)
source = source.replace('okc_attn_pk2_hybrid_v321_', 'okc_attn_pk2_hybrid_v322_')
changes = {
    '    assert bool(torch.isfinite(x).all())\n':
        '    assert bool(torch.isfinite(x).all()), "nonfinite attention output"\n',
    '            data = _aq_fixture(batch, ctx, 4319 + batch + page, True, page, scale, pairs)\n':
        '            report["stage"] = {"kind": "nearby", "batch": batch, "ctx": ctx, "page": page, "scale": scale, "pairs": pairs}\n'
        '            data = _aq_fixture(batch, ctx, 4319 + batch + page, True, page, scale, pairs)\n',
    '            for mod in modules[1:]:\n':
        '            for arm, mod in enumerate(modules[1:], 1):\n'
        '                report["stage"]["arm"] = arm\n',
    '                assert all(torch.equal(a, b) for a, b in zip(bits, cache_bits()))\n':
        '                same_cache = [bool(torch.equal(a, b)) for a, b in zip(bits, cache_bits())]\n'
        '                report["latest_validation"] = {"parent_error": e, "fallback_error": f, "cache_bits_equal": same_cache}\n'
        '                assert all(same_cache), "new K/V cache bits differ from parent"\n',
    '                    data = _aq_fixture(batch, 86000, seed, bool(group))\n':
        '                    report["stage"] = {"kind": "timing", "pad": pad, "batch": batch, "seed": seed}\n'
        '                    data = _aq_fixture(batch, 86000, seed, bool(group))\n',
    "        report['error'] = type(error).__name__ + ': ' + str(error)[-1500:]\n":
        "        report['error'] = type(error).__name__ + ': ' + str(error)[-1500:]\n"
        "        import traceback\n"
        "        report['traceback'] = traceback.format_exc(limit=4)[-2400:]\n",
}
for old, new in changes.items():
    assert source.count(old) == 1, old
    source = source.replace(old, new)
ast.parse(source)
target.write_text(source)
print(target.relative_to(ROOT), hashlib.sha256(source.encode()).hexdigest())
