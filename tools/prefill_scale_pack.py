"""Generate the matching research repack without importing GPU submission."""
import ast


def repack_source(parent_source: str, function_name='_ks_repack') -> str:
    tree=ast.parse(parent_source)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_repack')
    source=ast.get_source_segment(parent_source,node)
    source=source.replace('def _repack(',f'def {function_name}(',1)
    before='    for e in range(E):\n'
    assert source.count(before)==1
    source=source.replace(before, '''    _kp = torch.arange(64, device=dev, dtype=torch.int64)
    _kp = (_kp & 1) | ((_kp & 8) >> 2) | ((_kp & 48) >> 2) | ((_kp & 6) << 3)
''' + before)
    before='        c = _codes(w[e][perm]).to(torch.int32)'
    assert source.count(before)==1
    source=source.replace(before,before+'''
        c = c.reshape(N, KS, 64).index_select(2, _kp).reshape(N, KS * 64)''')
    return source
