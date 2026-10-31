"""Lossless CPU decoder for the bounded prepare-only MoE trace report."""
import base64
import struct


def decode_rows(wire):
    assert wire['row_format']=='<19I8f3d'
    layout=struct.Struct(wire['row_format'])
    raw=base64.b64decode(wire['rows_b64'])
    assert len(raw)==layout.size*wire['row_count']
    rows=[]
    for values in layout.iter_unpack(raw):
        row=dict(zip(['m','seed','pad_mib','scratch_mod32m'],values[:4]))
        row['ns']={k:list(values[4+5*i:9+5*i]) for i,k in enumerate(['a0','trace','a1'])}
        row['error']={k:list(values[19+2*i:21+2*i]) for i,k in enumerate(['trace','a1'])}
        row['fresh']={k:list(values[23+2*i:25+2*i]) for i,k in enumerate(['trace','a1'])}
        row.update(zip(['ratio','aa','change'],values[27:30]))
        rows.append(row)
    return rows


def decode_trace(wire):
    raw=base64.b64decode(wire['clock_bitpacked_le_b64'])
    minima,widths=wire['clock_minima'],wire['clock_widths']
    assert len(minima)==len(widths)==8 and all(0<=w<=32 for w in widths)
    assert len(raw)==(wire['grid']*sum(widths)+7)//8
    packed=int.from_bytes(raw,'little');values=[[0]*8 for _ in range(wire['grid'])];position=0
    for k in range(8):
        mask=(1<<widths[k])-1
        for row in values:
            row[k]=minima[k]+((packed>>position)&mask);position+=widths[k]
    assert packed>>position==0
    clocks=[]
    assert wire['clock_order']==[0,1,7,6,2,3,4,5]
    for c in range(wire['grid']):
        row=[0]*8;current=wire['origin_ns']
        for col,delta in zip(wire['clock_order'],values[c]):
            current+=delta;row[col]=current
        clocks.append(row)
    raw=base64.b64decode(wire['cuts_u16_le_b64'])
    assert len(raw)==wire['grid']*4*2
    cuts=struct.unpack('<'+str(wire['grid']*4)+'H',raw)
    metadata=[list(cuts[4*c:4*c+4])+wire['metadata_header'] for c in range(wire['grid'])]
    return dict(clocks_ns=clocks,metadata=metadata,jobs=wire['jobs'],
                m=wire['m'],seed=wire['seed'],pad_mib=wire['pad_mib'],
                clock_columns=wire['clock_columns'])
