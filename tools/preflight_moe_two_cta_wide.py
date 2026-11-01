"""CPU checks for routing storage lifetimes in the FT64 two-CTA alternative.

The local layout calculation is an LP64 estimate until NVCC assertions pass;
it does not establish actual H200 occupancy or GPU memory-order correctness.
"""
import ast
import ctypes as C
import gzip
import hashlib
import json
from pathlib import Path
import random
import struct
from moe_two_cta_wide_native import two_cta_wide_native

ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT / 'muse_moe_w4a16'
parent = (TRACK / 'submission.py').read_text()
original = next(ast.literal_eval(n.value) for n in ast.parse(parent).body
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CUDA_SRC' for t in n.targets))
native = two_cta_wide_native(original)
with gzip.open(TRACK / 'TWO_CTA_WIDE_NATIVE_2026-11-01.cu.gz', 'rt') as f:
    assert f.read() == native

# Original generic-to-async handoff must include every reader/writer and the
# full producer/consumer join; the producer issues its first copy after it.
handoff = '  wgs_gather(p, S);\n  asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");\n  DBG(7);\n  WS_JOIN();'
assert native.count(handoff) == 1
producer = native.split('__device__ void wgs_wproducer', 1)[1].split('// x producer:', 1)[0]
assert producer.index('WS_JOIN();') < producer.index('cp.async.bulk')
early = native.split('__device__ void early_issue(const Params& p, Smem& S) {', 1)[1]
assert early.startswith('\n#if FAST1_ON || OKC_WGS\n  return;')
assert '#if !OKC_PRMTSC\n  uint32_t lut[256];\n#endif' in native

i, u, u64, ptr, f = C.c_int, C.c_uint, C.c_uint64, C.c_void_p, C.c_float
J = 88
fields = [('ctr_reload', u64), ('red', f*4), ('bars', u64*24), ('wglast', i*4),
          ('wcp', i*(J+1)), ('wcpB', i*(J+1)), ('jrow', i*(J+1)),
          ('wtot', u), ('wtotB', u), ('wgr', i*4), ('seq', i), ('fseq', i),
          ('preissued', i), ('preissuedB', i), ('fcnt', i*32), ('xbuf_ptr', ptr),
          ('cnt', i*64), ('off', i*64), ('wc', i*(8*32)),
          ('jE', i*J), ('jS', i*J), ('jC', i*J), ('forder', i*J),
          ('njF', i), ('njS', i), ('gen', i), ('ssp', ptr), ('uslot', i*8),
          ('dync', i*J), ('pos', C.c_short*3072)]
Smem = type('Smem', (C.Structure,), {'_fields_': fields})
metadata = (C.sizeof(Smem)+127)//128*128
assert C.sizeof(Smem) == 10976 and metadata == 11008
dynamic = metadata + 2*18432 + 2*32768
assert dynamic == 113408
assert 3072+2*3072 <= 18432


def route_overlay(ids, nw, lo):
    """Independent CPU warp histogram/scatter model with byte-addressed alias."""
    emax, nt = 3072, nw*32
    E = len(ids)
    chunk = (E+nt-1)//nt*32
    shared = bytearray([0xA5])*(metadata+2*18432+2*32768)
    ring = memoryview(shared)[metadata:]
    for x, e in enumerate(ids):
        local = e-lo
        ring[x] = local if 0 <= local < 32 else 255
    counts = [[0]*32 for _ in range(nw)]
    for w in range(nw):
        for x in range(w*chunk, min(E, (w+1)*chunk)):
            e = ring[x]
            if e != 255:
                counts[w][e] += 1
    totals = [sum(c[e] for c in counts) for e in range(32)]
    offsets, off = [], 0
    for e in range(32):
        offsets.append(off)
        off += totals[e]
    q_by_warp = [[offsets[e]+sum(counts[k][e] for k in range(w)) for e in range(32)] for w in range(nw)]
    positions = [-1]*E
    for w in range(nw):
        for x in range(w*chunk, min(E, (w+1)*chunk)):
            e = ring[x]
            if e != 255:
                q = q_by_warp[w][e]
                struct.pack_into('<h', ring, emax+2*q, x)
                positions[x] = q
                q_by_warp[w][e] += 1
    # IDs remained alive throughout the scatter; no ENT write touched them.
    assert list(ring[:E]) == [e-lo if 0 <= e-lo < 32 else 255 for e in ids]
    entries = list(struct.unpack_from('<'+'h'*off, ring, emax)) if off else []
    oracle = [x for e in range(32) for x, value in enumerate(ids) if value == e+lo]
    assert entries == oracle
    assert all(positions[x] == q for q, x in enumerate(oracle))
    assert all(positions[x] == -1 for x, e in enumerate(ids) if not lo <= e < lo+32)
    # Metadata is disjoint from the overlaid ring even when all routes local.
    assert shared[:metadata] == bytes([0xA5])*metadata
    return entries, positions, totals


rng = random.Random(0x64C7A)
fixtures = 0
for m in (1, 8, 17, 32, 33, 64, 65, 128, 256, 257, 384):
    for lo in (0, 32, 224):
        for mode in ('random', 'hot', 'empty', 'basis'):
            ids = []
            for row in range(m):
                outside = [e for e in range(256) if not lo <= e < lo+32]
                values = (rng.sample(range(256), 8) if mode == 'random' else
                          list(range(lo, lo+8)) if mode == 'hot' else
                          rng.sample(outside, 8) if mode == 'empty' else
                          [lo]+rng.sample(outside, 7))
                assert len(set(values)) == 8
                ids.extend(values)
            assert route_overlay(ids, 8, lo) == route_overlay(ids, 16, lo)
            fixtures += 1

out = dict(scope='CPU alias extent, routing scatter/order, host LP64 layout and source-handoff checks only. Not CUDA execution, actual linked occupancy, synchronization proof or performance.',
           native_sha256=hashlib.sha256(native.encode()).hexdigest(), fixtures=fixtures,
           route_working_storage_bytes=9216, estimated_metadata_bytes=metadata,
           estimated_dynamic_shared_bytes=dynamic, estimated_struct_bytes=C.sizeof(Smem),
           original_FT64_job_partition_retained=True,
           original_weight_pointers_and_global_scratch_offsets_retained=True,
           original_NTT5_through8_restored=True,
           generic_async_fence_then_full_join=True,
           limits=['NVCC static assertions and complete full-body resource reports still required.',
                   'No increased logical weight-job reads versus parent; logical X-copy bytes still double.',
                   'Two weight slots may expose copy latency; no timing evidence yet.',
                   'Actual residency/zero-local/numeric/fresh/empty/FP64/nearby/control gates remain mandatory.'])
target = TRACK / 'TWO_CTA_WIDE_PREFLIGHT_2026-11-01.json'
assert not target.exists()
target.write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out, indent=2))
