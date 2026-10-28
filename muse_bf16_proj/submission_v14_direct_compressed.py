# okc: problem=muse_bf16_proj
"""Hybrid direct-compressed and cluster GEMM for Muse Spark Open dense projections (H200).

out[m, n] = x[m, k] @ W[n, k]^T.

* Swap-AB tensor-core GEMV: W is the mma.sync "A" operand (16 weight rows per MMA), the
  tokens are the 8-wide "B"/N dimension.
* prepare() repacks W (losslessly, still BF16) into MMA-fragment order: for every 16R-row
  group g and 32-wide k-block j there is one contiguous R KB chunk in which lane t's 16 bytes
  are exactly its A-fragment registers for two k16 MMA steps (the k order inside the 32-block
  is permuted identically for x, so a 16-byte x row slice is a B fragment).
* Each CTA owns one row group and a contiguous K range. A producer warp streams that range
  with cp.async.bulk (1-D TMA) into a shared-memory ring guarded by mbarriers, together with
  the matching x slice (x rows spread over the 32 producer lanes); consumer warps read
  fragments from the ring (conflict-free) and run the MMAs.
* Split-K across CTAs uses a thread-block cluster: the CS CTAs of a cluster share a row group
  and their partial tiles are reduced through distributed shared memory (DSMEM) after one
  cluster barrier: no global atomics, no workspace, no counters, one kernel.
* Router: FP32 weights are split once into BF16 hi + lo, concatenated along K, so one pass
  computes x@hi^T + x@lo^T with FP32 accumulation (SGLang's own split, as one kernel).
* Per (layer, m) the faster of this kernel and SGLang's cuBLAS path is used (table CHOICE,
  measured on the judge's H200). Shapes not in the table use a heuristic.
"""

import os
from typing import TypedDict

import torch

DEV = False  # config sweep in prepare (check runs only)


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor
    config: Config


CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

__device__ __forceinline__ uint4 lds128(uint32_t addr) {
    uint4 r;
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "r"(addr) : "memory");
    return r;
}
__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mbar_init(uint32_t addr, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(addr), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t addr, uint32_t parity) {
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1 bra DONE;\n"
        "bra LAB_WAIT;\n"
        "DONE:\n"
        "}\n" :: "r"(addr), "r"(parity) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t addr) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(addr) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t addr, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(addr), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t mbar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(mbar) : "memory");
}
__device__ __forceinline__ void cluster_sync_all() {
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ uint32_t mapa(uint32_t addr, uint32_t rank) {
    uint32_t r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank));
    return r;
}
__device__ __forceinline__ float ld_dsmem(uint32_t addr) {
    float v;
    asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(addr) : "memory");
    return v;
}

#define BAR_BYTES 512

template <int MT, int R, int NW>
struct Cfg {
    static constexpr int NE = R * MT * 4;
    static constexpr int WST = NW * R * 1024;   // weight bytes per stage
    static constexpr int XROW = NW * 64 + 64;   // padded x row (bank-conflict free)
    static constexpr int XST = MT * 8 * XROW;   // x bytes per stage
    static constexpr int ST = WST + XST;
    static constexpr int RED = NE * 32 * 4;
};

// grid.x = G * CS (cluster = CS consecutive CTAs = one 16R-row group), grid.y = token tiles.
template <int MT, int R, int NW, bool F32, bool RP>
__global__ void __launch_bounds__(((RP ? R : NW) + 1) * 32)
cl_kernel(const uint8_t* __restrict__ Wp, const __nv_bfloat16* __restrict__ x, void* __restrict__ out,
          int m, int n, int K, int K32, int xk32, int CS, int nstage) {
    using C = Cfg<MT, R, NW>;
    extern __shared__ __align__(128) uint8_t smem[];
    const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem);
    const uint32_t full0 = sbase, empty0 = sbase + 256;
    float* red = (float*)(smem + BAR_BYTES);
    const uint32_t red_s = sbase + BAR_BYTES;
    const uint32_t ring = sbase + BAR_BYTES + C::RED;

    const int tt = blockIdx.y;
    const int g = blockIdx.x / CS, rank = blockIdx.x % CS;
    const int KPC = K32 / CS;          // k-blocks per CTA
    const int UPC = KPC / NW;          // stages per CTA
    const int jbase = rank * KPC;
    const int tok0 = tt * MT * 8;
    int ntok = m - tok0;
    if (ntok > MT * 8) ntok = MT * 8;

    if (threadIdx.x == 0) {
        for (int s = 0; s < nstage; s++) {
            mbar_init(full0 + 8 * s, 1);
            mbar_init(empty0 + 8 * s, RP ? R : NW);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    for (int i = threadIdx.x; i < C::NE * 32; i += blockDim.x) red[i] = 0.f;
    __syncthreads();

    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (w == (RP ? R : NW)) {
        // producer warp: lane 0 owns the barriers and the weight copy; the x rows are spread
        // over the 32 lanes.
        const uint32_t bytes = (uint32_t)(C::WST + ntok * NW * 64);
        const uint8_t* wsrc = Wp + ((size_t)g * K32 + jbase) * R * 1024;
        for (int i = 0; i < UPC; ++i) {
            const int slot = i % nstage;
            const uint32_t fb = full0 + 8 * slot;
            const uint32_t dst = ring + slot * C::ST;
            if (lane == 0) {
                if (i >= nstage) mbar_wait(empty0 + 8 * slot, ((i / nstage) - 1) & 1);
                mbar_expect_tx(fb, bytes);
                bulk_g2s(dst, wsrc + (size_t)i * C::WST, C::WST, fb);
            }
            __syncwarp();
            const int jb = jbase + i * NW;
            const int jx = jb >= xk32 ? jb - xk32 : jb;
            const __nv_bfloat16* xs = x + (size_t)tok0 * K + (size_t)jx * 32;
            for (int t = lane; t < ntok; t += 32)
                bulk_g2s(dst + C::WST + t * C::XROW, xs + (size_t)t * K, NW * 64, fb);
        }
    } else if constexpr (RP) {
        // Row-partitioned qkv path: each consumer warp owns one disjoint
        // 16-row slice and walks every K slice in the stage. Total MMA work
        // is unchanged, while the cross-warp shared-atomic reduction vanishes.
        const int cq = lane & 3, gr = lane >> 2;
        float acc[MT][4];
#pragma unroll
        for (int mt = 0; mt < MT; mt++)
#pragma unroll
            for (int q = 0; q < 4; q++) acc[mt][q] = 0.f;
        bool xv[MT];
#pragma unroll
        for (int mt = 0; mt < MT; mt++) xv[mt] = (mt * 8 + gr) < ntok;
        for (int i = 0; i < UPC; ++i) {
            const int slot = i % nstage;
            mbar_wait(full0 + 8 * slot, (i / nstage) & 1);
            const uint32_t sb = ring + slot * C::ST;
#pragma unroll
            for (int kw = 0; kw < NW; kw++) {
                const uint32_t wo = (uint32_t)(kw * R * 1024 + w * 1024 + lane * 16);
                const uint32_t xo = (uint32_t)(C::WST + gr * C::XROW + kw * 64 + cq * 16);
                uint4 wa0 = lds128(sb + wo);
                uint4 wa1 = lds128(sb + wo + 512);
#pragma unroll
                for (int mt = 0; mt < MT; mt++) {
                    uint4 xb = xv[mt] ? lds128(sb + xo + mt * 8 * C::XROW) : make_uint4(0, 0, 0, 0);
                    mma16816(acc[mt], wa0.x, wa1.x, wa0.y, wa1.y, xb.x, xb.y);
                    mma16816(acc[mt], wa0.z, wa1.z, wa0.w, wa1.w, xb.z, xb.w);
                }
            }
            __syncwarp();
            if (lane == 0) mbar_arrive(empty0 + 8 * slot);
        }
#pragma unroll
        for (int mt = 0; mt < MT; mt++)
#pragma unroll
            for (int q = 0; q < 4; q++)
                red[((w * MT + mt) * 4 + q) * 32 + lane] = acc[mt][q];
    } else {
        const int cq = lane & 3, gr = lane >> 2;
        float acc[R][MT][4];
#pragma unroll
        for (int r = 0; r < R; r++)
#pragma unroll
            for (int mt = 0; mt < MT; mt++)
#pragma unroll
                for (int q = 0; q < 4; q++) acc[r][mt][q] = 0.f;
        bool xv[MT];
#pragma unroll
        for (int mt = 0; mt < MT; mt++) xv[mt] = (mt * 8 + gr) < ntok;
        const uint32_t w_off = (uint32_t)(w * R * 1024 + lane * 16);
        const uint32_t x_off = (uint32_t)(C::WST + gr * C::XROW + w * 64 + cq * 16);
        for (int i = 0; i < UPC; ++i) {
            const int slot = i % nstage;
            mbar_wait(full0 + 8 * slot, (i / nstage) & 1);
            const uint32_t sb = ring + slot * C::ST;
            uint4 wa[R][2];
            uint4 xb[MT];
#pragma unroll
            for (int r = 0; r < R; r++)
#pragma unroll
                for (int h = 0; h < 2; h++) wa[r][h] = lds128(sb + w_off + (r * 2 + h) * 512);
#pragma unroll
            for (int mt = 0; mt < MT; mt++)
                xb[mt] = xv[mt] ? lds128(sb + x_off + mt * 8 * C::XROW) : make_uint4(0, 0, 0, 0);
#pragma unroll
            for (int r = 0; r < R; r++)
#pragma unroll
                for (int mt = 0; mt < MT; mt++) {
                    mma16816(acc[r][mt], wa[r][0].x, wa[r][1].x, wa[r][0].y, wa[r][1].y, xb[mt].x, xb[mt].y);
                    mma16816(acc[r][mt], wa[r][0].z, wa[r][1].z, wa[r][0].w, wa[r][1].w, xb[mt].z, xb[mt].w);
                }
            // release the slot only after the MMAs consumed the registers
            __syncwarp();
            if (lane == 0) mbar_arrive(empty0 + 8 * slot);
        }
#pragma unroll
        for (int r = 0; r < R; r++)
#pragma unroll
            for (int mt = 0; mt < MT; mt++)
#pragma unroll
                for (int q = 0; q < 4; q++)
                    atomicAdd(&red[((r * MT + mt) * 4 + q) * 32 + lane], acc[r][mt][q]);
    }
    __syncwarp();
    // every CTA's partial tile is complete and visible (cluster-wide when split)
    if (CS > 1) cluster_sync_all(); else __syncthreads();
    {
        constexpr int E = C::NE * 32;
        const int e0 = (rank * E) / CS, e1 = ((rank + 1) * E) / CS;
        const int rowbase = g * 16 * R;
        for (int e = e0 + (int)threadIdx.x; e < e1; e += blockDim.x) {
            const int i = e >> 5, ln = e & 31;
            const int r = i / (MT * 4), mt = (i >> 2) % MT, q = i & 3;
            const int row = rowbase + r * 16 + (ln >> 2) + ((q & 2) ? 8 : 0);
            const int tok = tok0 + mt * 8 + (ln & 3) * 2 + (q & 1);
            float v = 0.f;
            if (CS == 1) v = red[e];
            else for (int p = 0; p < CS; p++) v += ld_dsmem(mapa(red_s + e * 4, p));
            if (tok < m) {
                const size_t idx = (size_t)tok * n + row;
                if (F32) ((float*)out)[idx] = v;
                else ((__nv_bfloat16*)out)[idx] = __float2bfloat16(v);
            }
        }
    }
    // keep shared memory alive until every peer has read it
    if (CS > 1) cluster_sync_all();
}

template <int MT, int R, int NW, bool F32, bool RP>
static void launch(const torch::Tensor& Wp, const torch::Tensor& x, torch::Tensor& out,
                   int m, int n, int K, int K32, int xk32, int CS, int T, int ring_kb) {
    using C = Cfg<MT, R, NW>;
    TORCH_CHECK((K32 % CS) == 0 && ((K32 / CS) % NW) == 0, "bad split");
    int nstage = (ring_kb * 1024) / C::ST;
    const int upc = K32 / CS / NW;
    if (nstage > upc) nstage = upc;
    if (nstage > 32) nstage = 32;
    if (nstage < 1) nstage = 1;
    const int smem = BAR_BYTES + C::RED + nstage * C::ST;
    TORCH_CHECK(smem <= 227 * 1024, "smem too large");
    auto kfn = cl_kernel<MT, R, NW, F32, RP>;
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024);
        cudaFuncSetAttribute(kfn, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
        configured = true;
    }
    const int G = n / (16 * R);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(G * CS, T, 1);
    cfg.blockDim = dim3(((RP ? R : NW) + 1) * 32, 1, 1);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = CS;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = CS > 1 ? 1 : 0;
    cudaError_t err = cudaLaunchKernelEx(&cfg, kfn, (const uint8_t*)Wp.data_ptr(),
                                         (const __nv_bfloat16*)x.data_ptr(), out.data_ptr(),
                                         m, n, K, K32, xk32, CS, nstage);
    TORCH_CHECK(err == cudaSuccess, "launch failed: ", cudaGetErrorString(err));
}

#define ARGS Wp, x, out, m, n, K, K32, xk32, CS, T, ring_kb
#define DISPATCH_NW(MT, R, F32)                                      \
    if (NW == 4) { if (rp) return launch<MT, R, 4, F32, true>(ARGS);  \
                   return launch<MT, R, 4, F32, false>(ARGS); }       \
    if (NW == 8) { if (rp) return launch<MT, R, 8, F32, true>(ARGS);  \
                   return launch<MT, R, 8, F32, false>(ARGS); }
#define DISPATCH_R(MT, F32)                 \
    if (R == 1) { DISPATCH_NW(MT, 1, F32) } \
    if (R == 2) { DISPATCH_NW(MT, 2, F32) } \
    if (R == 4) { DISPATCH_NW(MT, 4, F32) }
#define DISPATCH_MT(F32)                \
    if (MT == 1) { DISPATCH_R(1, F32) } \
    if (MT == 2) { DISPATCH_R(2, F32) } \
    if (MT == 4) { DISPATCH_R(4, F32) }

void cl_gemm(torch::Tensor Wp, torch::Tensor x, torch::Tensor out,
             int64_t m_, int64_t n_, int64_t K32_, int64_t xk32_, int64_t CS_, int64_t T_,
             int64_t MT, int64_t R, int64_t NW, int64_t ring_kb_, bool rp) {
    const int K = (int)x.size(1);
    const int m = (int)m_, n = (int)n_, K32 = (int)K32_, xk32 = (int)xk32_, CS = (int)CS_, T = (int)T_;
    const int ring_kb = (int)ring_kb_;
    const bool f32 = out.scalar_type() == torch::kFloat32;
    if (f32) { DISPATCH_MT(true) } else { DISPATCH_MT(false) }
    TORCH_CHECK(false, "unsupported config");
}

// Direct-load exact 12-bit GEMV for the bandwidth-bound qkv/pooler m<=8
// shapes.  Sixteen warps stream disjoint K ranges.  The seventeenth warp
// evaluates the rare exact corrections concurrently, so neither a raw-block
// fallback nor an exception tail appears in the streaming loop.
__device__ __forceinline__ uint4 zload(const uint4* p) {
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}
__device__ __forceinline__ uint32_t zprmt(uint32_t a, uint32_t b, uint32_t s) {
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
    return r;
}
__device__ __forceinline__ uint32_t zmerge(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t r;
    asm("lop3.b32 %0, %1, %2, %3, 0xF8;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}
__device__ __forceinline__ uint4 zdecode(uint32_t lo0, uint32_t lo1, uint32_t q,
                                         uint32_t tl, uint32_t th) {
    const uint32_t h0 = zmerge(zprmt(tl, th, q & 0x7777u), q, 0x80808080u);
    const uint32_t h1 = zmerge(zprmt(tl, th, (q >> 16) & 0x7777u), q << 4, 0x80808080u);
    return make_uint4(zprmt(lo0, h0, 0x5140u), zprmt(lo0, h0, 0x7362u),
                      zprmt(lo1, h1, 0x5140u), zprmt(lo1, h1, 0x7362u));
}

__global__ __launch_bounds__(17 * 32, 2)
zgemv_kernel(const __nv_bfloat16* __restrict__ x, const uint4* __restrict__ zw,
             const int2* __restrict__ exc, int capr, __nv_bfloat16* __restrict__ out,
             int M, int N, int K, uint32_t tl, uint32_t th) {
    constexpr int NW = 16;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int gr = lane >> 2, tq = lane & 3;
    const int rb = blockIdx.x;
    const int kg = K / 64, kgw = kg / NW;
    __shared__ float4 partial[NW][32];
    __shared__ float corr[16][9];

    if (warp == NW) {
        const int row = lane & 15, parity = lane >> 4;
        float ea[4] = {0.f, 0.f, 0.f, 0.f};
        const int2* ep = exc + (size_t)(rb * 16 + row) * capr;
#pragma unroll 4
        for (int j = 0; j < capr; ++j) {
            const int2 e = __ldg(ep + j);
            if (e.y != 0) {
                const float delta = __int_as_float(e.y);
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int tok = parity + 2 * q;
                    if (tok < M)
                        ea[q] += delta * __bfloat162float(x[(size_t)tok * K + e.x]);
                }
            }
        }
#pragma unroll
        for (int q = 0; q < 4; ++q) corr[row][parity + 2 * q] = ea[q];
    } else {
        const int g0 = warp * kgw;
        const uint4* wp = zw + ((size_t)rb * kg + g0) * 96 + lane;
        const int tok = gr < M ? gr : M - 1;
        const __nv_bfloat16* xp = x + (size_t)tok * K + g0 * 64 + tq * 8;
        float acc[4] = {0.f, 0.f, 0.f, 0.f};

        // PIPE=2, UG=1: retain one compressed group in registers and issue
        // the following group's global loads before decoding/MMA of this one.
        uint4 a0 = zload(wp), a1 = zload(wp + 32), b = zload(wp + 64);
        for (int i = 0; i < kgw; ++i) {
            const uint4 xa = __ldg((const uint4*)(xp + (size_t)i * 64));
            const uint4 xb = __ldg((const uint4*)(xp + (size_t)i * 64 + 32));
            uint4 na0, na1, nb;
            if (i + 1 < kgw) {
                const uint4* np = wp + (size_t)(i + 1) * 96;
                na0 = zload(np); na1 = zload(np + 32); nb = zload(np + 64);
            }
            const uint4 f0 = zdecode(a0.x, a0.y, b.x, tl, th);
            const uint4 f1 = zdecode(a0.z, a0.w, b.y, tl, th);
            const uint4 f2 = zdecode(a1.x, a1.y, b.z, tl, th);
            const uint4 f3 = zdecode(a1.z, a1.w, b.w, tl, th);
            mma16816(acc, f0.x, f0.y, f0.z, f0.w, xa.x, xa.y);
            mma16816(acc, f1.x, f1.y, f1.z, f1.w, xa.z, xa.w);
            mma16816(acc, f2.x, f2.y, f2.z, f2.w, xb.x, xb.y);
            mma16816(acc, f3.x, f3.y, f3.z, f3.w, xb.z, xb.w);
            if (i + 1 < kgw) { a0 = na0; a1 = na1; b = nb; }
        }
        partial[warp][lane] = make_float4(acc[0], acc[1], acc[2], acc[3]);
    }
    __syncthreads();

    if (tid < 32) {
        float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int w = 0; w < NW; ++w) {
            const float4 v = partial[w][lane];
            sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;
        }
        sum.x += corr[gr][2 * tq];
        sum.y += corr[gr][2 * tq + 1];
        sum.z += corr[gr + 8][2 * tq];
        sum.w += corr[gr + 8][2 * tq + 1];
        const float v[4] = {sum.x, sum.y, sum.z, sum.w};
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int otok = 2 * tq + (q & 1);
            const int orow = rb * 16 + gr + ((q >> 1) << 3);
            if (otok < M) out[(size_t)otok * N + orow] = __float2bfloat16_rn(v[q]);
        }
    }
}

void zgemv(torch::Tensor x, torch::Tensor zw, torch::Tensor exc, int64_t capr,
           torch::Tensor out, int64_t N_, int64_t K_, int64_t tl_, int64_t th_) {
    const int N = (int)N_, K = (int)K_;
    TORCH_CHECK(K % (16 * 64) == 0 && N % 16 == 0, "bad zgemv shape");
    zgemv_kernel<<<N / 16, 17 * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const __nv_bfloat16*)x.data_ptr(), (const uint4*)zw.data_ptr(),
        (const int2*)exc.data_ptr(), (int)capr, (__nv_bfloat16*)out.data_ptr(),
        (int)x.size(0), N, K, (uint32_t)tl_, (uint32_t)th_);
}
"""

CPP_SRC = r"""
void cl_gemm(torch::Tensor Wp, torch::Tensor x, torch::Tensor out,
             int64_t m, int64_t n, int64_t K32, int64_t xk32, int64_t CS, int64_t T,
             int64_t MT, int64_t R, int64_t NW, int64_t ring_kb, bool rp);
void zgemv(torch::Tensor x, torch::Tensor zw, torch::Tensor exc, int64_t capr,
           torch::Tensor out, int64_t N, int64_t K, int64_t tl, int64_t th);
"""

_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
        from torch.utils.cpp_extension import load_inline
        _EXT = load_inline(
            name="muse_cluster_v14_direct_compressed",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["cl_gemm", "zgemv"],
            extra_cuda_cflags=["-O3", "-arch=sm_90a"],
            verbose=False,
        )
    return _EXT


NUM_SMS = 132
LAYERS = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096),
          "router": (256, 8192)}


def _nfor(layer, m):
    return 1024 if (layer == "unpooler" and m <= 8) else LAYERS[layer][0]


def _pack(w: torch.Tensor, R: int) -> torch.Tensor:
    """[n, k] bf16 -> [G][K32][R][2][8 rows][4 c][8 e] contiguous (fragment order)."""
    n, k = w.shape
    G, K32 = n // (16 * R), k // 32
    v = w.reshape(G, R, 2, 8, K32, 4, 8)
    return v.permute(0, 4, 1, 2, 3, 5, 6).contiguous()


def _pack_z(w: torch.Tensor):
    """Losslessly store BF16 in 12 bits plus sparse exact corrections.

    The raw low byte is retained.  The high byte becomes sign plus a 3-bit
    lookup into the best matrix-wide window of eight exponent-high values.
    Values outside that window are reconstructed to the nearest endpoint;
    their exact FP32 difference (and, if needed, one FP32 residual) is kept
    in a fixed-width per-row exception table consumed by a helper warp.
    """
    n, k = w.shape
    assert w.dtype == torch.bfloat16 and n % 16 == 0 and k % 64 == 0
    rb, kb = n // 16, k // 32
    dev = w.device
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    e7 = (bits >> 8) & 0x7F
    hist = torch.bincount(e7.flatten(), minlength=128)
    prefix = torch.cat([torch.zeros(1, dtype=hist.dtype, device=dev), torch.cumsum(hist, 0)])
    base = int(torch.argmax(prefix[8:] - prefix[:-8]))
    code = (e7 - base).clamp(0, 7)
    sign = bits >> 15
    low = bits & 0xFF

    decoded_bits = (sign << 15) | ((base + code) << 8) | low
    decoded_bits = torch.where(decoded_bits >= 32768, decoded_bits - 65536, decoded_bits)
    decoded = decoded_bits.to(torch.int16).view(torch.bfloat16)
    del decoded_bits, e7, bits

    # A BF16-to-BF16 difference is exactly representable as one or, for a
    # very large exponent separation, two FP32 values.
    d64 = w.double() - decoded.double()
    d32 = d64.float()
    residual = (d64 - d32.double()).float()
    i1, i2 = (d32 != 0).nonzero(), (residual != 0).nonzero()
    rows = torch.cat([i1[:, 0], i2[:, 0]])
    ks = torch.cat([i1[:, 1], i2[:, 1]])
    vals = torch.cat([d32[i1[:, 0], i1[:, 1]], residual[i2[:, 0], i2[:, 1]]])
    del d64, d32, residual, decoded
    order = torch.argsort(rows * k + ks, stable=True)
    rows, ks, vals = rows[order], ks[order], vals[order]
    counts = torch.bincount(rows, minlength=n)
    capr = max(1, int(counts.max()) if len(rows) else 1)
    starts = torch.cumsum(counts, 0) - counts
    pos = torch.arange(len(rows), device=dev) - starts[rows]
    exc = torch.zeros((n, capr, 2), dtype=torch.int32, device=dev)
    exc[rows, pos, 0] = ks.to(torch.int32)
    exc[rows, pos, 1] = vals.contiguous().view(torch.int32)

    def fragment_order(a):
        p = a.reshape(rb, 2, 8, kb, 4, 2, 2, 2).permute(0, 3, 5, 2, 4, 6, 1, 7)
        return p.reshape(rb, kb // 2, 4, 32, 8)

    lowp = fragment_order(low.to(torch.uint8))
    lo_sections = lowp.permute(0, 1, 3, 2, 4).reshape(rb, kb // 2, 32, 2, 16)
    lo_sections = lo_sections.permute(0, 1, 3, 2, 4)
    codp, sgnp = fragment_order(code.to(torch.int64)), fragment_order(sign.to(torch.int64))
    cshift = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28], dtype=torch.int64, device=dev)
    sshift = torch.tensor([7, 15, 23, 31, 3, 11, 19, 27], dtype=torch.int64, device=dev)
    q = (codp << cshift).sum(-1) + (sgnp << sshift).sum(-1)
    q = torch.where(q >= 2 ** 31, q - 2 ** 32, q).to(torch.int32)
    code_section = q.permute(0, 1, 3, 2).contiguous().view(torch.uint8)
    code_section = code_section.reshape(rb, kb // 2, 1, 32, 16)
    packed = torch.cat([lo_sections, code_section], dim=2).contiguous()
    lut = [base + j for j in range(8)]
    tl = sum(lut[j] << (8 * j) for j in range(4))
    th = sum(lut[j + 4] << (8 * j) for j in range(4))
    return {"w": packed, "exc": exc.contiguous(), "capr": capr,
            "tl": tl, "th": th, "nesc": int(len(rows)), "base": base}


def _mt(m):
    return 1 if m <= 8 else (2 if m <= 16 else 4)


# (layer, m) -> (R, NW, CS, ring_kb) for this kernel, or "cublas" for SGLang's path.
# From the judge-H200 sweeps (check jobs ae63bc34c7b949cf, 2ddc741cc8f5428f).
CHOICE = {
    ("qkv_gate", 1): (2, 4, 4, 48), ("o", 1): "cublas", ("pooler", 1): (2, 8, 1, 96),
    ("unpooler", 1): (1, 4, 4, 48), ("router", 1): (1, 4, 16, 96),
    ("qkv_gate", 8): (2, 4, 4, 48), ("o", 8): "cublas", ("pooler", 8): (2, 8, 2, 96),
    ("unpooler", 8): (2, 4, 8, 48), ("router", 8): (2, 8, 16, 96),
    ("qkv_gate", 16): "cublas", ("o", 16): "cublas", ("pooler", 16): "cublas",
    ("unpooler", 16): "cublas", ("router", 16): (1, 8, 16, 96),
    ("qkv_gate", 32): "cublas", ("o", 32): "cublas", ("pooler", 32): "cublas",
    ("unpooler", 32): "cublas", ("router", 32): (2, 8, 16, 96),
    ("router", 128): "cublas", ("router", 256): "cublas",
}


def _heuristic(layer, m, n):
    """Shapes outside the table: this kernel for m <= 8 (and the router up to 32), else cuBLAS."""
    if m > 32 or (m > 8 and layer != "router"):
        return "cublas"
    if layer == "o":
        return "cublas"
    K32 = (LAYERS[layer][1] * (2 if layer == "router" else 1)) // 32
    R, NW = 2, 4
    G = n // (16 * R)
    CS = 1
    while G * CS < 2 * NUM_SMS and CS < 16 and (K32 // (CS * 2)) % NW == 0:
        CS *= 2
    return (R, NW, CS, 48)


def _choice(layer, m, n):
    return CHOICE.get((layer, m)) or _heuristic(layer, m, n)


def prepare(weights):
    ext = _ext()
    dev = weights["qkv_gate"].device
    state = {"ext": ext, "packs": {}, "zpacks": {}, "w": {}}
    router = weights["router"].float()
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    eff = {name: weights[name] for name in ("qkv_gate", "o", "pooler", "unpooler")}
    eff["router"] = torch.cat([hi, lo], dim=1).contiguous()  # [256, 16384] BF16 hi | lo
    state["eff"] = eff
    for name, w in eff.items():
        for R in (1, 2, 4):
            state["packs"][(name, R)] = _pack(w, R)
    # First direct-compressed experiment is deliberately limited to the four
    # shapes where the public H200 data shows the largest bandwidth win.
    for name in ("qkv_gate", "pooler"):
        state["zpacks"][name] = _pack_z(weights[name].contiguous())
        z = state["zpacks"][name]
        print("[zpack] %s base=%d corrections=%d capr=%d" %
              (name, z["base"], z["nesc"], z["capr"]), flush=True)
    for name in ("qkv_gate", "o", "pooler", "unpooler"):
        state["w"][name] = weights[name].contiguous()
    state["router"] = weights["router"].contiguous()
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    # warm every kernel instantiation kernel_fn can pick (sets smem attributes before capture)
    for layer, (nn, kk) in LAYERS.items():
        for m in (1, 5, 7, 8, 9, 16, 32):
            xw = torch.zeros((m, kk), dtype=torch.bfloat16, device=dev)
            kernel_fn({"x": xw, "config": {"m": m, "n": _nfor(layer, m), "k": kk, "layer": layer}}, state)
    torch.cuda.synchronize()
    if DEV:
        _dev_sweep(state, weights)
    return state


def _run(state, layer, x, m, n, c, mt=None):
    K32 = state["eff"][layer].shape[1] // 32
    xk32 = x.shape[1] // 32
    R, NW, CS, ring = c
    MT = mt or _mt(m)
    T = -(-m // (8 * MT))
    dt = torch.float32 if layer == "router" else torch.bfloat16
    out = torch.empty((m, n), dtype=dt, device=x.device)
    # v12 showed the row-owned mapping helps qkv m=8, while its m=1 result
    # did not beat the original four-consumer reduction. Keep the two mappings
    # shape-specialized rather than forcing one scheduling topology globally.
    rp = layer == "qkv_gate" and m == 8
    state["ext"].cl_gemm(state["packs"][(layer, R)], x, out, m, n, K32, xk32, CS, T, MT, R, NW, ring, rp)
    return out


def _run_z(state, layer, x, n):
    z = state["zpacks"][layer]
    out = torch.empty((x.shape[0], n), dtype=torch.bfloat16, device=x.device)
    state["ext"].zgemv(x, z["w"], z["exc"], z["capr"], out, n, x.shape[1], z["tl"], z["th"])
    return out


def _cublas(state, layer, x, n):
    """SGLang's path for this shape (cuBLAS)."""
    if layer != "router":
        return x @ state["w"][layer][:n].T
    if x.shape[0] <= 8:
        return torch.nn.functional.linear(x.float(), state["router"])
    acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
    acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
    return acc


def kernel_fn(data, state):
    x = data["x"].contiguous()
    if x.data_ptr() % 16:
        x = x.clone()
    cfg = data["config"]
    layer, m, n = cfg["layer"], x.shape[0], cfg["n"]
    if layer in ("qkv_gate", "pooler") and m in (1, 8):
        return _run_z(state, layer, x, n)
    c = _choice(layer, m, n)
    if c == "cublas":
        return _cublas(state, layer, x, n)
    return _run(state, layer, x, m, n, c)


# --------------------------------------------------------------------------------------
# Development-only sweep (check runs; prints timings, never affects results)
# --------------------------------------------------------------------------------------
def _bench(fn, flush, reps=25):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(reps):
        flush.sum()
        st.record()
        g.replay()
        en.record()
        en.synchronize()
        ts.append(st.elapsed_time(en) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2]


def _dev_sweep(state, weights):
    dev = weights["qkv_gate"].device
    flush = torch.zeros(64 << 20, dtype=torch.float32, device=dev)
    dummy = torch.zeros(1, device=dev)
    ovh = _bench(lambda: dummy.add_(1), flush)
    gen = torch.Generator(device=dev).manual_seed(0)
    for m in (16, 32):
        for layer in ("qkv_gate", "o", "pooler", "unpooler"):
            n, k = _nfor(layer, m), LAYERS[layer][1]
            x = torch.randn((m, k), generator=gen, device=dev).to(torch.bfloat16)
            ref = x.double() @ weights[layer][:n].double().T
            tb = _bench(lambda: _cublas(state, layer, x, n), flush) - ovh
            K32 = k // 32
            res = []
            for R in (2, 4):
                G = n // (16 * R)
                for NW in (4, 8):
                    for CS in (1, 2, 4, 8):
                        if K32 % CS or (K32 // CS) % NW or G * CS < NUM_SMS // 2:
                            continue
                        for ring in (48, 96, 160):
                            c = (R, NW, CS, ring)
                            try:
                                out = _run(state, layer, x, m, n, c)
                                err = float((out.double() - ref).norm() / ref.norm())
                                t = _bench(lambda: _run(state, layer, x, m, n, c), flush) - ovh
                                res.append((t, c, err))
                            except Exception as e:
                                print("cfg", c, "failed", repr(e)[:200])
            res.sort(key=lambda r: r[0])
            print("m=%d %s cublas=%.2f best: %s" % (m, layer, tb, " ".join(
                "%s=%.2f(e%.0e)" % (str(c), t, e) for t, c, e in res[:4])), flush=True)
