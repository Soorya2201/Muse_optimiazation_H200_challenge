"""Static Muse geometry and CPU distribution/byte screen of pinned Unweight.

No downloaded code is executed. Synthetic CPU Gaussian weights use the task's
documented distribution, not the CUDA RNG stream or any judge seed.
"""
import hashlib
import heapq
import json
import math
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
rows=[]
for k in (1024,4096,8192):
    gen=torch.Generator().manual_seed(43017+k)
    w=(torch.randn((256,k),generator=gen)/math.sqrt(k)).to(torch.bfloat16)
    b=w.view(torch.int16).to(torch.int32)&65535
    ex=(b>>7)&255;hist=torch.bincount(ex.flatten(),minlength=256).tolist()
    palette=sorted(range(256),key=lambda i:(-hist[i],i))[:16]
    counts=[hist[i] for i in palette if hist[i]]
    heap=[(count,serial,[symbol]) for serial,(symbol,count) in enumerate((i,hist[i]) for i in palette if hist[i])]
    heapq.heapify(heap);lens={i:0 for i in palette};serial=256
    while len(heap)>1:
        ca,_,sa=heapq.heappop(heap);cb,_,sb=heapq.heappop(heap)
        for symbol in sa+sb:lens[symbol]+=1
        heapq.heappush(heap,(ca+cb,serial,sa+sb));serial+=1
    pl=torch.full((256,),-1,dtype=torch.int32)
    ll=torch.zeros(256,dtype=torch.int32)
    for i,p in enumerate(palette):pl[p]=i;ll[p]=lens[p]
    verb=(pl[ex].reshape(-1,64)<0).any(1)
    lens_rows=ll[ex].reshape(-1,64).sum(1)
    n=b.numel();ntiles=n//4096
    verb_count=int(verb.sum());huff_bits=int(lens_rows[~verb].sum())
    md_bytes=ntiles*16;row_offsets=n//64*2
    table_bytes=4*(1<<max(lens.values()))
    direct=n*3//2+verb_count*64+md_bytes+16
    huff=n+(huff_bits+7)//8+verb_count*64+md_bytes+row_offsets+table_bytes+16
    upper=torch.bincount(((b>>8)&127).flatten(),minlength=128)
    wins=[int(upper[i:i+8].sum()) for i in range(121)]
    base=max(range(121),key=lambda i:wins[i]);esc=n-wins[base]
    c12=n*3//2+esc*8
    assert torch.equal((((((b>>15)<<7)|(b&127))&128)<<8) | (ex<<7) | (b&127), b)
    rows.append(dict(k=k,n_rows=256,weight_count=n,seed=43017+k,
        bf16_bits_sha256=hashlib.sha256(b.to(torch.int16).numpy().tobytes()).hexdigest(),
        exponent_histogram=hist,palette=palette,huffman_lengths=lens,
        max_code_len=max(lens.values()),verbatim_rows=verb_count,total_rows=n//64,
        raw_bytes=n*2,direct_palette_bytes=direct,huffman_bytes=huff,
        c12_optimistic_bytes=c12,c12_base=base,c12_escape_count=esc,
        direct_over_c12=direct/c12,huffman_over_c12=huff/c12,
        huffman_runtime_exp_decode_logical_bytes_per_weight=3+huff_bits/(8*n)+verb_count*64/n,
        note='Huffman totals include table/row offsets; direct totals include all nibble slots and extra verbatim exponents. C12 counts8B/global escape, excludes padding/replication. No cache/DRAM counter claim.'))
geometry=[]
shapes={'qkv_gate':(2304,8192),'o':(8192,1024),'pooler':(4096,8192),
        'unpooler':(8192,4096),'router':(256,8192)}
for m in (1,8,16,32,128,256):
    for name,(n,k) in shapes.items():
        if name=='router':continue  # FP32 router is not this BF16 format.
        if name=='unpooler' and m<=8:n=1024
        for nt in (1,2,3,4):
            if n%(nt*64):continue
            tiles=math.ceil(m/128)*(n//(nt*64))
            geometry.append(dict(layer=name,m=m,n=n,k=k,tiles_n=nt,
                total_output_tasks=tiles,launched_ctas=min(tiles,132),
                padded_m=math.ceil(m/128)*128,m_fragment_fill=m/(math.ceil(m/128)*128)))
base=ROOT/'research/public/unweight_db0e6d3'
manifest=json.loads((base/'PROVENANCE.json').read_text())
for f in manifest['files']:assert hashlib.sha256((base/f['path']).read_bytes()).hexdigest()==f['sha256']
report=dict(scope='Static source-derived geometry and own CPU Gaussian byte estimates; no GPU execution or speed prediction.',
    repository_commit=manifest['commit'],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    distribution_cases=rows,geometry=geometry,
    limitations=['CPU sample is256rows, not full judged matrices or CUDA RNG.',
        'Byte estimates omit allocation alignment, activation/output traffic, transactions and timing.',
        'Source tile geometry supplies no padding initialization adapter here.',
        'Router FP32 excluded; allowed hi+lo is not a direct FP32->BF16 replacement.',
        'No whole Unweight port or new sub12-bit GPU decoder is implemented.'])
p=ROOT/'muse_bf16_proj/UNWEIGHT_FORMAT_SCREEN_2026-10-31.json';assert not p.exists()
p.write_text(json.dumps(report,indent=2)+'\n')
for r in rows:print('K',r['k'],'palette/C12',r['direct_over_c12'],'Huffman/C12',r['huffman_over_c12'],'depth',r['max_code_len'],'verbatim',r['verbatim_rows'])
print('Geometry rows',len(geometry))
