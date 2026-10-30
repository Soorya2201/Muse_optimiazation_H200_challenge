# Standalone research excerpt, no upstream module import.
# rohanarun/Auto-research-agent at 018b7a102348ea3b73701e8be52d24f82d8c91ee
# Source header credits Apache-2.0 base; modifications copyright (c)2026 Rohan Arun.
# MIT terms in adjacent LICENSE; Apache-2.0 terms at project LICENSE.

def _stored_pos_mul():
    pos={}
    for R in range(4):
        for up in range(2):
            for bit,o in enumerate([6,7,8,15]):
                pos[R,up,bit]=((o+16*up+[0,3,6][R])%32 if R<3 else [4,0,1,3][bit]+16*up)
    assert sorted(pos.values())==list(range(32))
    return pos

MUL_DQ4_CPP = 'constexpr uint32_t M=0x81C081C0u;\n  r0=q&M;\n  r1=__funnelshift_r(q,q,3)&M;\n  r2=__funnelshift_r(q,q,6)&M;\n  r3=((q&0x001b001bu)*4228u)&M;'
