"""Decode lossless, bounded sampled-unit reports without importing torch."""
import base64
import struct


def matrix(wire):
    n,k=wire['rows'],wire['cols'];mins,widths=wire['minima'],wire['widths']
    assert n>0 and len(mins)==len(widths)==k and all(0<=w<=32 for w in widths)
    data=base64.b64decode(wire['data_b64']);bits=n*sum(widths)
    assert len(data)==(bits+7)//8
    packed=int.from_bytes(data,'little');out=[[0]*k for _ in range(n)];pos=0
    for j in range(k):
        mask=(1<<widths[j])-1
        for row in out:
            row[j]=mins[j]+((packed>>pos)&mask);pos+=widths[j]
    assert packed>>pos==0
    return out


def rows(wire):
    assert wire['row_format']=='<24I12f4d'
    fmt=struct.Struct(wire['row_format']);data=base64.b64decode(wire['rows_b64'])
    assert len(data)==fmt.size*wire['row_count']
    out=[]
    for v in fmt.iter_unpack(data):
        row=dict(zip(['m','seed','pad_mib','scratch_mod32m'],v[:4]))
        row['ns']={k:list(v[4+5*i:9+5*i]) for i,k in enumerate(['a0','trace','a1','canonical'])}
        row['error']={k:list(v[24+2*i:26+2*i]) for i,k in enumerate(['trace','a1','canonical'])}
        row['fresh']={k:list(v[30+2*i:32+2*i]) for i,k in enumerate(['trace','a1','canonical'])}
        row.update(zip(['ratio','aa','baseline_speedup_vs_canonical','change'],v[36:40]))
        out.append(row)
    return out


def trace(wire):
    assert wire['kind']=='sampled_unit_trace_v1' and wire['ctas']==[0,33,66,99]
    assert len(wire['frame_fields'])==16 and len(wire['frame_counts'])==4
    frames=matrix(wire['frames']);clocks=matrix(wire['phase_clocks'])
    assert len(clocks)==4 and all(len(c)==8 for c in clocks)
    assert sum(wire['frame_counts'])==len(frames) and all(0<c<=32 for c in wire['frame_counts'])
    origin=wire['origin_ns'];groups=[];p=0
    for c,count in zip(wire['ctas'],wire['frame_counts']):
        fs=frames[p:p+count];p+=count
        for f in fs:
            for i in range(10,15):
                if f[i]:f[i]=origin+f[i]-1
        groups.append(dict(cta=c,frames=fs))
    for c in clocks:
        for i in range(8):c[i]+=origin
    data=base64.b64decode(wire['cuts_u16_le_b64']);assert len(data)==wire['grid']*8
    cuts=struct.unpack('<'+str(wire['grid']*4)+'H',data)
    metadata=[list(cuts[4*c:4*c+4])+wire['metadata_header'] for c in range(wire['grid'])]
    return dict(m=wire['m'],seed=wire['seed'],pad_mib=wire['pad_mib'],origin_ns=origin,
                ctas=wire['ctas'],phase_clocks_ns=clocks,groups=groups,metadata=metadata,
                jobs=wire['jobs'],frame_fields=wire['frame_fields'])
