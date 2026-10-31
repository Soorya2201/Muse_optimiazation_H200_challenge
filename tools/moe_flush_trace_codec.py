"""Decode bounded flush observations; preserve absolute integer clocks."""
import base64
import struct
from moe_unit_trace_codec import matrix


def trace(wire):
    assert wire['kind']=='sampled_flush_trace_v1' and wire['ctas']==[0,1,66,67]
    assert wire['frame_fields']==['sample','phase','unit','expected_kind','expected_nst','start','end']
    origin=wire['origin_ns'];frames=matrix(wire['frames']);clocks=matrix(wire['phase_clocks'])
    assert len(clocks)==4 and all(len(c)==8 for c in clocks)
    groups=[dict(cta=c,frames=[]) for c in wire['ctas']]
    for sample,pb,u,kind,nst,start,end in frames:
        assert 0<=sample<4 and pb in (0,1) and 0<=u<8192
        assert kind in (1,2,3) and nst<512 and start<=end
        groups[sample]['frames'].append([pb,u,kind,nst,origin+start,origin+end])
    data=base64.b64decode(wire['cuts_u16_le_b64']);assert len(data)==wire['grid']*8
    cuts=struct.unpack('<'+str(wire['grid']*4)+'H',data)
    metadata=[list(cuts[4*c:4*c+4])+wire['metadata_header'] for c in range(wire['grid'])]
    return dict(m=wire['m'],seed=wire['seed'],pad_mib=wire['pad_mib'],origin_ns=origin,
                ctas=wire['ctas'],phase_clocks_ns=[[origin+x for x in c] for c in clocks],
                groups=groups,metadata=metadata,jobs=wire['jobs'],frame_fields=['phase','unit','expected_kind','expected_nst','start','end'])
