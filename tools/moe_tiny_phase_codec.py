"""Exact CPU wire format for the one-token phase diagnostic."""
import base64 as _TP64
import copy as _TPCopy
import json as _TPJSON
import struct as _TPStruct
import zlib as _TPZlib


def encode_report(report):
    framed=_TPCopy.deepcopy(report);body=bytearray()
    for row in framed['rows']:
        samples=row.pop('ns');times=[]
        for name in ('a0','old','trace','a1'):times.extend(samples[name])
        assert len(times)==32 and all(0<v<2**32 for v in times)
        metrics=[]
        for field in ('error','fresh'):
            values=row.pop(field)
            for name in ('old','trace','a1'):metrics.extend(values[name])
        metrics.append(row.pop('change'))
        # These originate as torch float32 scalars. Reject rather than round
        # a value that cannot be represented exactly in this wire format.
        for value in metrics:
            assert _TPStruct.unpack('<f',_TPStruct.pack('<f',value))[0]==value
        body.extend(_TPStruct.pack('<32I13f',*times,*metrics))
    for tr in framed['traces']:
        clocks=tr.pop('clocks_ns');origin=min(row[0] for row in clocks)
        tr['origin_ns']=origin;deltas=[]
        for row in clocks:
            previous=origin;values=[]
            for clock in row:
                delta=clock-previous;assert 0<=delta<2**32
                values.append(delta);previous=clock
            assert len(values)==6;deltas.append(values)
        minima=[min(row[k] for row in deltas) for k in range(6)]
        widths=[(max(row[k] for row in deltas)-minima[k]).bit_length() for k in range(6)]
        packed=0;position=0
        for k in range(6):
            for row in deltas:
                packed|=(row[k]-minima[k])<<position;position+=widths[k]
        blob=packed.to_bytes((position+7)//8,'little');body.extend(blob)
        tr.update(minima=minima,widths=widths,clock_bytes=len(blob))
    framed['codec']='tiny_phase_v2_exact_f32_metrics_bitpacked_u64_clocks'
    header=_TPJSON.dumps(framed,separators=(',',':')).encode()
    binary=_TPStruct.pack('<I',len(header))+header+body
    return _TP64.b64encode(_TPZlib.compress(binary,9)).decode()


def decode_report(payload):
    wire=_TPZlib.decompress(_TP64.b64decode(payload));size=_TPStruct.unpack_from('<I',wire)[0]
    report=_TPJSON.loads(wire[4:4+size]);body=wire[4+size:];position=0
    assert report.pop('codec')=='tiny_phase_v2_exact_f32_metrics_bitpacked_u64_clocks'
    layout=_TPStruct.Struct('<32I13f')
    for row in report['rows']:
        values=layout.unpack_from(body,position);position+=layout.size
        row['ns']={name:list(values[8*i:8*i+8]) for i,name in enumerate(('a0','old','trace','a1'))}
        i=32
        for field in ('error','fresh'):
            row[field]={}
            for name in ('old','trace','a1'):row[field][name]=list(values[i:i+2]);i+=2
        row['change']=values[i]
    for tr in report['traces']:
        size=tr.pop('clock_bytes');blob=body[position:position+size];assert len(blob)==size
        position+=size;packed=int.from_bytes(blob,'little')
        minima,widths=tr.pop('minima'),tr.pop('widths');origin=tr.pop('origin_ns')
        assert size==(tr['grid']*sum(widths)+7)//8
        deltas=[[0]*6 for _ in range(tr['grid'])];offset=0
        for k in range(6):
            for row in deltas:
                row[k]=minima[k]+((packed>>offset)&((1<<widths[k])-1));offset+=widths[k]
        assert packed>>offset==0
        clocks=[]
        for row in deltas:
            current=origin;clock=[]
            for delta in row:current+=delta;clock.append(current)
            clocks.append(clock)
        tr['clocks_ns']=clocks
    assert position==len(body)
    return report
