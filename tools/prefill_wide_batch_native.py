"""Research-only four-MMA commit groups on wide RS-WGMMA expert tiles.

Keeps the original packed weights, shared ring, K order and epilogue. Wide
KPS1 stages alternate register-fragment banks by *stage*, not constant kk.
Wide tiles retain one accumulator to avoid doubling the 128-float result.
"""

def wide_batch_native(source: str) -> str:
    def replace(before,after,count=1):
        nonlocal source
        assert source.count(before)==count,(before,source.count(before))
        source=source.replace(before,after)
    replace('#define MOE_GROUP4_MAX 128', '#define MOE_GROUP4_MAX 256')
    replace('((GROUP4 && MOE_NACC2) ?', '((GROUP4 && MOE_NACC2 && N <= 128) ?')
    # For KPS2/4, stage endpoints already alternate kk banks. For KPS1,
    # kk is always zero: using it would overwrite the previous pending A.
    replace('(DEEP ? kk : (kk & 1))',
            '(DEEP ? kk : ((KPS == 1 ? it : kk) & 1))',count=2)
    return source
