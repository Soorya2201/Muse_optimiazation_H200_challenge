# okc: problem=muse_bf16_proj
"""Cluster split-K, TMA-streamed skinny GEMM for Muse Spark Open dense projections (H200).

out[m, n] = x[m, k] @ W[n, k]^T.

* Swap-AB tensor-core GEMV: W is the mma.sync "A" operand (16 weight rows per MMA), the
  tokens are the 8-wide "B"/N dimension.
* prepare() repacks W (losslessly, still BF16) into MMA-fragment order: for every 16R-row
  group g and 32-wide k-block j there is one contiguous R KB chunk in which lane t's 16 bytes
  are exactly its A-fragment registers for two k16 MMA steps (the k order inside the 32-block
  is permuted identically for x, so a 16-byte x row slice is a B fragment).
* Each CTA owns one row group and a contiguous K range. A producer thread streams that range
  with cp.async.bulk (1-D TMA) into a shared-memory ring guarded by mbarriers, together with
  the matching x slice; consumer warps read fragments from the ring (conflict-free) and MMA.
* Split-K across CTAs uses a thread-block cluster: the CS CTAs of a cluster share a row group,
  and their partial tiles are reduced through distributed shared memory (DSMEM) after one
  cluster barrier: no global atomics, no workspace, no counters, one kernel.
* Router: FP32 weights are split once into BF16 hi + lo, concatenated along K, so one pass
  computes x@hi^T + x@lo^T with FP32 accumulation.
"""

import os
from typing import TypedDict

import torch

DEV = True  # config sweep in prepare (check runs only)


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
template <int MT, int R, int NW, bool F32>
__global__ void __launch_bounds__((NW + 1) * 32)
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
            mbar_init(empty0 + 8 * s, NW);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    for (int i = threadIdx.x; i < C::NE * 32; i += blockDim.x) red[i] = 0.f;
    __syncthreads();

    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (w == NW) {
        // producer warp: lane 0 owns the barriers and the weight copy, the x rows are spread
        // over all 32 lanes so a 32-token stage is 1 + 1 copy issues per lane, not 33 serial ones.
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
    // cluster-wide: every CTA's partial tile is complete and visible
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
            else
                for (int p = 0; p < CS; p++) v += ld_dsmem(mapa(red_s + e * 4, p));
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

template <int MT, int R, int NW, bool F32>
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
    auto kfn = cl_kernel<MT, R, NW, F32>;
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024);
        cudaFuncSetAttribute(kfn, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
        configured = true;
    }
    const int G = n / (16 * R);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(G * CS, T, 1);
    cfg.blockDim = dim3((NW + 1) * 32, 1, 1);
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
#define DISPATCH_NW(MT, R, F32)                       \
    if (NW == 4) return launch<MT, R, 4, F32>(ARGS);  \
    if (NW == 8) return launch<MT, R, 8, F32>(ARGS);
#define DISPATCH_R(MT, F32)                 \
    if (R == 1) { DISPATCH_NW(MT, 1, F32) } \
    if (R == 2) { DISPATCH_NW(MT, 2, F32) }
#define DISPATCH_MT(F32)                \
    if (MT == 1) { DISPATCH_R(1, F32) } \
    if (MT == 2) { DISPATCH_R(2, F32) } \
    if (MT == 4) { DISPATCH_R(4, F32) } \
    if (MT == 8) { DISPATCH_R(8, F32) }

void cl_gemm(torch::Tensor Wp, torch::Tensor x, torch::Tensor out,
             int64_t m_, int64_t n_, int64_t K32_, int64_t xk32_, int64_t CS_, int64_t T_,
             int64_t MT, int64_t R, int64_t NW, int64_t ring_kb_) {
    const int K = (int)x.size(1);
    const int m = (int)m_, n = (int)n_, K32 = (int)K32_, xk32 = (int)xk32_, CS = (int)CS_, T = (int)T_;
    const int ring_kb = (int)ring_kb_;
    const bool f32 = out.scalar_type() == torch::kFloat32;
    if (f32) { DISPATCH_MT(true) } else { DISPATCH_MT(false) }
    TORCH_CHECK(false, "unsupported config");
}

// ---- dev-only: raw HBM read bandwidth probe ----
__global__ void read_kernel(const uint4* __restrict__ p, size_t n16, int* __restrict__ sink) {
    uint32_t acc = 0;
    const size_t per = (n16 + gridDim.x - 1) / gridDim.x;
    const size_t b = per * blockIdx.x, e = min(n16, b + per);
    for (size_t i = b + threadIdx.x; i < e; i += 4 * blockDim.x) {
        uint4 v[4];
#pragma unroll
        for (int u = 0; u < 4; u++) {
            size_t j = i + (size_t)u * blockDim.x;
            v[u] = j < e ? __ldg(p + j) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int u = 0; u < 4; u++) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
    }
    if (acc == 0x12345678u) sink[0] = 1;
}
__global__ void empty_kernel() {}
void read_probe(torch::Tensor buf, int64_t bytes, int64_t ctas, int64_t threads, torch::Tensor sink) {
    auto s = at::cuda::getCurrentCUDAStream();
    if (bytes == 0) { empty_kernel<<<1, 32, 0, s>>>(); return; }
    read_kernel<<<(int)ctas, (int)threads, 0, s>>>((const uint4*)buf.data_ptr(), (size_t)bytes / 16, sink.data_ptr<int>());
}
"""

CPP_SRC = r"""
void cl_gemm(torch::Tensor Wp, torch::Tensor x, torch::Tensor out,
             int64_t m, int64_t n, int64_t K32, int64_t xk32, int64_t CS, int64_t T,
             int64_t MT, int64_t R, int64_t NW, int64_t ring_kb);
void read_probe(torch::Tensor buf, int64_t bytes, int64_t ctas, int64_t threads, torch::Tensor sink);
"""

_EXT = None
_DEBUG_NAN = False


def _ext():
    global _EXT
    if _EXT is None:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
        from torch.utils.cpp_extension import load_inline
        _EXT = load_inline(
            name="muse_cluster_v5",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["cl_gemm", "read_probe"],
            extra_cuda_cflags=["-O3", "-lineinfo"],
            verbose=False,
        )
    return _EXT


NUM_SMS = 132
SKINNY_MAX_M = 32  # non-router layers above this go to cuBLAS
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


def _mt(m):
    return 1 if m <= 8 else (2 if m <= 16 else 4)


# (layer, mt) -> (R, NW, CS, ring_kb)
CFG = {}


def _default(layer, n, MT, T):
    R, NW = 1, 8
    K32 = (LAYERS[layer][1] * (2 if layer == "router" else 1)) // 32
    G = n // (16 * R)
    # aim for >= 2 CTAs per SM worth of work
    CS = 1
    while G * CS * T < 2 * NUM_SMS and CS < 16 and (K32 // (CS * 2)) % NW == 0:
        CS *= 2
    return (R, NW, CS, 64)


def _plan(layer, m, n, override=None, mt=None):
    MT = mt or _mt(m)
    T = -(-m // (8 * MT))
    R, NW, CS, ring = override or CFG.get((layer, MT, n)) or _default(layer, n, MT, T)
    return MT, T, R, NW, CS, ring


def prepare(weights):
    ext = _ext()
    dev = weights["qkv_gate"].device
    state = {"ext": ext, "packs": {}, "w": {}}
    router = weights["router"].float()
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    eff = {name: weights[name] for name in ("qkv_gate", "o", "pooler", "unpooler")}
    eff["router"] = torch.cat([hi, lo], dim=1).contiguous()  # [256, 16384]
    state["eff"] = eff
    for name, w in eff.items():
        for R in (1, 2):
            state["packs"][(name, R)] = _pack(w, R)
    for name in ("qkv_gate", "o", "pooler", "unpooler"):
        state["w"][name] = weights[name].contiguous()
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    for layer, (nn, kk) in LAYERS.items():
        for m in (1, 8, 16, 32) + ((128, 256) if layer == "router" else ()):
            xw = torch.zeros((m, kk), dtype=torch.bfloat16, device=dev)
            _run(state, layer, xw, m, _nfor(layer, m))
    torch.cuda.synchronize()
    if DEV:
        try:
            _dev_sweep(state, weights)
        except Exception as e:  # noqa
            import traceback
            print("DEV sweep failed:", repr(e))
            traceback.print_exc()
    return state


def _run(state, layer, x, m, n, override=None, mt=None):
    K32 = state["eff"][layer].shape[1] // 32
    xk32 = x.shape[1] // 32
    MT, T, R, NW, CS, ring = _plan(layer, m, n, override, mt)
    dt = torch.float32 if layer == "router" else torch.bfloat16
    out = torch.full((m, n), float("nan"), dtype=dt, device=x.device) if _DEBUG_NAN else \
        torch.empty((m, n), dtype=dt, device=x.device)
    state["ext"].cl_gemm(state["packs"][(layer, R)], x, out, m, n, K32, xk32, CS, T, MT, R, NW, ring)
    return out


def kernel_fn(data, state):
    x = data["x"].contiguous()
    if x.data_ptr() % 16:
        x = x.clone()
    cfg = data["config"]
    layer, m, n = cfg["layer"], x.shape[0], cfg["n"]
    if layer == "router" or m <= SKINNY_MAX_M:
        return _run(state, layer, x, m, n)
    return x @ state["w"][layer][:n].T


# --------------------------------------------------------------------------------------
# Development-only sweep (prints timings; never affects results)
# --------------------------------------------------------------------------------------
SCORED = [(m, l) for m in (1, 16, 32, 8, 128, 256) for l in ("qkv_gate", "o", "pooler", "unpooler", "router")]


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
    import time
    t0 = time.time()
    dev = weights["qkv_gate"].device
    ext = state["ext"]
    from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl
    flush = torch.zeros(64 << 20, dtype=torch.float32, device=dev)
    dummy = torch.zeros(1, device=dev)
    ovh = _bench(lambda: dummy.add_(1), flush)
    sink = torch.zeros(4, dtype=torch.int32, device=dev)
    print("empty graph (torch add) %.2f us; empty kernel %.2f us  (overhead below = first)" %
          (ovh, _bench(lambda: ext.read_probe(flush, 0, 1, 32, sink), flush)))
    # raw read-bandwidth floor for the weight sizes (cold L2)
    big = torch.zeros(80 << 20, dtype=torch.uint8, device=dev)
    for mb in (8, 16, 37.75, 64):
        b = int(mb * (1 << 20)) // 16 * 16
        line = "read %5.2f MB:" % mb
        for ctas, thr in ((132, 512), (264, 512), (528, 256), (1056, 256)):
            t = _bench(lambda: ext.read_probe(big, b, ctas, thr, sink), flush) - ovh
            line += " %dx%d=%.2fus(%.2fTB/s)" % (ctas, thr, t, b / t / 1e6)
        print(line, flush=True)
    del big
    gen = torch.Generator(device=dev).manual_seed(0)

    def base(layer, x, n):
        if layer != "router":
            return _bf16_gemm_dispatch_impl(x, weights[layer][:n], None)
        if x.shape[0] <= 8:
            return torch.nn.functional.linear(x.float(), weights["router"])
        acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
        acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
        return acc

    for (m, layer) in SCORED:
        n, k = _nfor(layer, m), LAYERS[layer][1]
        x = torch.randn((m, k), generator=gen, device=dev).to(torch.bfloat16)
        ref = (x.double() @ weights[layer][:n].double().T)
        tb = _bench(lambda: base(layer, x, n), flush) - ovh
        line = "m=%3d %-9s base=%6.2f" % (m, layer, tb)
        K32 = state["eff"][layer].shape[1] // 32
        if layer == "router" and m > 32:
            mts = [None, 8]
        elif layer == "router" or m <= SKINNY_MAX_M:
            mts = [None]
        else:
            mts = [8]
        res = []
        maxerr = 0.0
        for mt in mts:
            MT = mt or _mt(m)
            T = -(-m // (8 * MT))
            cands = []
            for R in (1, 2):
                G = n // (16 * R)
                for NW in (4, 8):
                    for CS in (1, 2, 4, 8, 16):
                        if K32 % CS or (K32 // CS) % NW:
                            continue
                        ctas = G * CS * T
                        if ctas < NUM_SMS // 2 or ctas > 8 * NUM_SMS:
                            continue
                        for ring in (48, 96):
                            cands.append((R, NW, CS, ring))
            for c in cands:
                try:
                    out = _run(state, layer, x, m, n, c, mt)
                    err = float((out.double() - ref).norm() / ref.norm())
                    maxerr = max(maxerr, err)
                    t = _bench(lambda: _run(state, layer, x, m, n, c, mt), flush) - ovh
                    res.append((t, c, err, mt))
                except Exception as e:
                    print("  cfg", c, "failed", repr(e)[:300])
        res.sort(key=lambda r: r[0])
        dflt = _plan(layer, m, n)
        line += " dflt%s" % (str(dflt[2:]),)
        line += " best:" + " ".join("%s/%s=%.2f" % (str(r[1]), r[3], r[0]) for r in res[:5])
        line += " maxerr=%.1e n=%d" % (maxerr, len(res))
        print(line, flush=True)
    print("sweep took %.1fs" % (time.time() - t0))
    _stress(state, weights, gen)


def _stress(state, weights, gen, reps=8):
    dev = weights["qkv_gate"].device
    global _DEBUG_NAN
    _DEBUG_NAN = True
    try:
        for (m, layer) in SCORED + [(33, "o"), (7, "pooler"), (5, "unpooler"), (9, "router")]:
            if layer != "router" and m > SKINNY_MAX_M:
                continue
            n, k = _nfor(layer, m), LAYERS[layer][1]
            worst, nans = 0.0, 0
            for _ in range(reps):
                x = torch.randn((m, k), generator=gen, device=dev).to(torch.bfloat16)
                out = kernel_fn({"x": x, "config": {"m": m, "n": n, "k": k, "layer": layer}}, state)
                ref = x.double() @ weights[layer][:n].double().T
                nans += int(torch.isnan(out).sum())
                worst = max(worst, float((out.double() - ref).norm() / ref.norm()))
            print("stress m=%d %s worst=%.2e nans=%d" % (m, layer, worst, nans), flush=True)
    finally:
        _DEBUG_NAN = False
