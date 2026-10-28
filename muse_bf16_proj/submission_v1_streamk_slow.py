# okc: problem=muse_bf16_proj
"""Stream-K TMA-bulk skinny GEMM for Muse Spark Open dense projections on H200.

out[m, n] = x[m, k] @ W[n, k]^T.

* Swap-AB tensor-core GEMV: W is the mma.sync "A" operand (16 weight rows per MMA), the
  tokens are the 8-wide "B"/N dimension.
* prepare() repacks W (losslessly, still BF16) into MMA-fragment order: for every
  16R-row group g and 32-wide k-block j there is one contiguous R KB chunk where lane t's
  16 bytes are exactly its A-fragment registers for two k16 MMA steps (the k order inside
  the 32-block is permuted identically for x, so x's 16-byte loads are B fragments).
* One persistent CTA per SM. A producer thread streams the CTA's contiguous slice of the
  packed weight with cp.async.bulk (TMA 1-D bulk copies) into a deep shared-memory ring
  guarded by mbarriers (~150-200 KB in flight per SM), consumer warps read fragments
  straight from the ring (conflict-free 512 B per warp access) and run the MMAs.
* Work is split stream-K style: the weight stream is cut into equal contiguous pieces, one
  per CTA, so every SM moves the same number of bytes regardless of n. Row groups that
  straddle two CTAs are reduced through fp32 atomics plus a last-arriver fix-up that also
  re-zeros the workspace (single kernel, graph-safe).
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

__device__ __forceinline__ uint4 ldg_x(const void* p) {
    uint4 r;
    asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}
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
__device__ __forceinline__ void cons_bar(int nthreads) {
    asm volatile("bar.sync 1, %0;" :: "r"(nthreads) : "memory");
}

#define BAR_BYTES 1024

template <int MT, int R, int NW>
struct Cfg {
    static constexpr int NE = R * MT * 4;
    static constexpr int WST = NW * R * 1024;           // weight bytes per stage
    static constexpr int XROW = NW * 64 + 64;           // padded x row (bank-conflict free)
    static constexpr int XST = MT * 8 * XROW;           // x bytes per stage
    static constexpr int ST = WST + XST;
    static constexpr int RED = NE * 32 * 4;
};

// Flush one row-group segment: reduce the NW warps' partials in smem, then either write the
// output (this CTA covered the whole K range of g) or add into the fp32 workspace; the last
// contributor (counted in k-blocks) writes the output and re-zeros the workspace + counter.
template <int MT, int R, int NW, bool F32>
__device__ __forceinline__ void flush_seg(float (&acc)[R][MT][4], float* red, unsigned* flag, int lane,
                                          int g, int seg_blocks, int K32, int tt, int T, int m, int n,
                                          float* __restrict__ ws, unsigned* __restrict__ cnt, void* __restrict__ out) {
    constexpr int NE = R * MT * 4;
    constexpr int NT = NW * 32;
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int mt = 0; mt < MT; mt++)
#pragma unroll
            for (int q = 0; q < 4; q++) {
                atomicAdd(&red[((r * MT + mt) * 4 + q) * 32 + lane], acc[r][mt][q]);
                acc[r][mt][q] = 0.f;
            }
    cons_bar(NT);
    const bool whole = (seg_blocks == K32);
    const int rowbase = g * 16 * R;
    for (int e = threadIdx.x; e < NE * 32; e += NT) {
        const int i = e >> 5, ln = e & 31;
        const int r = i / (MT * 4), mt = (i >> 2) % MT, q = i & 3;
        const int row = rowbase + r * 16 + (ln >> 2) + ((q & 2) ? 8 : 0);
        const int tok = tt * MT * 8 + mt * 8 + (ln & 3) * 2 + (q & 1);
        const float v = red[e];
        red[e] = 0.f;
        if (tok < m) {
            const size_t idx = (size_t)tok * n + row;
            if (whole) {
                if (F32) ((float*)out)[idx] = v;
                else ((__nv_bfloat16*)out)[idx] = __float2bfloat16(v);
            } else {
                atomicAdd(ws + idx, v);
            }
        }
    }
    if (!whole) {
        __threadfence();
        cons_bar(NT);
        if (threadIdx.x == 0) {
            __threadfence();
            const unsigned prev = atomicAdd(cnt + g * T + tt, (unsigned)seg_blocks);
            *flag = (prev + (unsigned)seg_blocks == (unsigned)K32) ? 1u : 0u;
            __threadfence();
        }
        cons_bar(NT);
        if (*(volatile unsigned*)flag) {
            __threadfence();
            for (int e = threadIdx.x; e < NE * 32; e += NT) {
                const int i = e >> 5, ln = e & 31;
                const int r = i / (MT * 4), mt = (i >> 2) % MT, q = i & 3;
                const int row = rowbase + r * 16 + (ln >> 2) + ((q & 2) ? 8 : 0);
                const int tok = tt * MT * 8 + mt * 8 + (ln & 3) * 2 + (q & 1);
                if (tok < m) {
                    const size_t idx = (size_t)tok * n + row;
                    const float v = __ldcg(ws + idx);
                    if (F32) ((float*)out)[idx] = v;
                    else ((__nv_bfloat16*)out)[idx] = __float2bfloat16(v);
                    __stcg(ws + idx, 0.f);
                }
            }
            if (threadIdx.x == 0) atomicExch(cnt + g * T + tt, 0u);
        }
    }
    cons_bar(NT);
}

// Wp: packed [G][K32][R][2][32] uint4; a unit = NW consecutive k-blocks of one row group g,
// which is one contiguous NW*R KB chunk. Each stage holds that chunk plus the matching x slice.
template <int MT, int R, int NW, bool F32>
__global__ void __launch_bounds__((NW + 1) * 32, 1)
stream_kernel(const uint8_t* __restrict__ Wp, const __nv_bfloat16* __restrict__ x,
              float* __restrict__ ws, unsigned* __restrict__ cnt, void* __restrict__ out,
              int m, int n, int K, int K32, int xk32, int nstage) {
    using C = Cfg<MT, R, NW>;
    extern __shared__ __align__(128) uint8_t smem[];
    const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem);
    const uint32_t full0 = sbase, empty0 = sbase + 256;
    float* red = (float*)(smem + BAR_BYTES);
    unsigned* flag = (unsigned*)(smem + BAR_BYTES - 16);
    const uint32_t ring = sbase + BAR_BYTES + C::RED;

    const int T = gridDim.y, tt = blockIdx.y;
    const int NC = gridDim.x, c = blockIdx.x;
    const int G = n / (16 * R);
    const int UPG = K32 / NW;
    const long long UT = (long long)G * UPG;
    const int u0 = (int)((UT * c) / NC), u1 = (int)((UT * (c + 1)) / NC);
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
        if (lane == 0) {
            const uint32_t bytes = (uint32_t)(C::WST + ntok * NW * 64);
            for (int i = 0, u = u0; u < u1; ++i, ++u) {
                const int slot = i % nstage;
                if (i >= nstage) mbar_wait(empty0 + 8 * slot, ((i / nstage) - 1) & 1);
                const uint32_t fb = full0 + 8 * slot;
                const uint32_t dst = ring + slot * C::ST;
                mbar_expect_tx(fb, bytes);
                bulk_g2s(dst, Wp + (size_t)u * C::WST, C::WST, fb);
                const int jb = (u % UPG) * NW;
                const int jx = jb >= xk32 ? jb - xk32 : jb;
                const __nv_bfloat16* xs = x + (size_t)tok0 * K + (size_t)jx * 32;
                for (int t = 0; t < ntok; t++)
                    bulk_g2s(dst + C::WST + t * C::XROW, xs + (size_t)t * K, NW * 64, fb);
            }
        }
        return;
    }

    // ---------------- consumers ----------------
    if (u0 >= u1) return;
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
    int cur_g = u0 / UPG, seg = 0;
    for (int i = 0, u = u0; u < u1; ++i, ++u) {
        const int g = u / UPG;
        if (g != cur_g) {
            flush_seg<MT, R, NW, F32>(acc, red, flag, lane, cur_g, seg * NW, K32, tt, T, m, n, ws, cnt, out);
            cur_g = g;
            seg = 0;
        }
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
        // release the slot only after the MMAs have consumed the registers
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * slot);
        ++seg;
    }
    flush_seg<MT, R, NW, F32>(acc, red, flag, lane, cur_g, seg * NW, K32, tt, T, m, n, ws, cnt, out);
}

template <int MT, int R, int NW, bool F32>
static void launch(const torch::Tensor& Wp, const torch::Tensor& x, torch::Tensor& ws, torch::Tensor& cnt,
                   torch::Tensor& out, int m, int n, int K, int K32, int xk32, int NC, int T, int ring_kb) {
    using C = Cfg<MT, R, NW>;
    int nstage = (ring_kb * 1024) / C::ST;
    if (nstage > 32) nstage = 32;
    if (nstage < 2) nstage = 2;
    const int smem = BAR_BYTES + C::RED + nstage * C::ST;
    auto kfn = stream_kernel<MT, R, NW, F32>;
    static int configured = 0;
    if (smem > configured) {
        cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024);
        configured = 227 * 1024;
    }
    TORCH_CHECK(smem <= 227 * 1024, "smem too large");
    auto stream = at::cuda::getCurrentCUDAStream();
    kfn<<<dim3(NC, T), (NW + 1) * 32, smem, stream>>>(
        (const uint8_t*)Wp.data_ptr(), (const __nv_bfloat16*)x.data_ptr(), ws.data_ptr<float>(),
        (unsigned*)cnt.data_ptr<int>(), out.data_ptr(), m, n, K, K32, xk32, nstage);
}

#define ARGS Wp, x, ws, cnt, out, m, n, K, K32, xk32, NC, T, ring_kb
#define DISPATCH_NW(MT, R, F32)                                   \
    if (NW == 4) return launch<MT, R, 4, F32>(ARGS);              \
    if (NW == 8) return launch<MT, R, 8, F32>(ARGS);
#define DISPATCH_R(MT, F32)                \
    if (R == 1) { DISPATCH_NW(MT, 1, F32) } \
    if (R == 2) { DISPATCH_NW(MT, 2, F32) } \
    if (R == 4) { DISPATCH_NW(MT, 4, F32) }
#define DISPATCH_MT(F32)                   \
    if (MT == 1) { DISPATCH_R(1, F32) }    \
    if (MT == 2) { DISPATCH_R(2, F32) }    \
    if (MT == 4) { DISPATCH_R(4, F32) }    \
    if (MT == 8) { if (R == 1) { DISPATCH_NW(8, 1, F32) } if (R == 2) { DISPATCH_NW(8, 2, F32) } }

void stream_gemm(torch::Tensor Wp, torch::Tensor x, torch::Tensor ws, torch::Tensor cnt, torch::Tensor out,
                 int64_t m_, int64_t n_, int64_t K32_, int64_t xk32_, int64_t NC_, int64_t T_,
                 int64_t MT, int64_t R, int64_t NW, int64_t ring_kb_) {
    const int K = (int)x.size(1);
    const int m = (int)m_, n = (int)n_, K32 = (int)K32_, xk32 = (int)xk32_, NC = (int)NC_, T = (int)T_;
    const int ring_kb = (int)ring_kb_;
    const bool f32 = out.scalar_type() == torch::kFloat32;
    if (f32) { DISPATCH_MT(true) } else { DISPATCH_MT(false) }
    TORCH_CHECK(false, "unsupported config");
}
"""

CPP_SRC = r"""
void stream_gemm(torch::Tensor Wp, torch::Tensor x, torch::Tensor ws, torch::Tensor cnt, torch::Tensor out,
                 int64_t m, int64_t n, int64_t K32, int64_t xk32, int64_t NC, int64_t T,
                 int64_t MT, int64_t R, int64_t NW, int64_t ring_kb);
"""

_EXT = None
_DEBUG_NAN = False


def _ext():
    global _EXT
    if _EXT is None:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
        from torch.utils.cpp_extension import load_inline
        _EXT = load_inline(
            name="muse_stream_v3",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["stream_gemm"],
            extra_cuda_cflags=["-O3", "-lineinfo"],
            verbose=False,
        )
    return _EXT


NUM_SMS = 132
SKINNY_MAX_M = 32  # non-router layers above this go to cuBLAS
WS_FLOATS = 1 << 22


def _pack(w: torch.Tensor, R: int) -> torch.Tensor:
    """[n, k] bf16 -> [G][K32][R][2][8 rows][4 c][8 e] contiguous (fragment order)."""
    n, k = w.shape
    G, K32 = n // (16 * R), k // 32
    v = w.reshape(G, R, 2, 8, K32, 4, 8)
    return v.permute(0, 4, 1, 2, 3, 5, 6).contiguous()


def _mt(m):
    return 1 if m <= 8 else (2 if m <= 16 else 4)


# (layer, m) -> (R, NW, ring_kb, NC). Missing entries use the default.
CFG = {}


def _default(layer, m):
    MT = _mt(m)
    R = 1 if MT == 1 else (2 if MT == 2 else 4)
    if layer == "router":
        R = min(R, 4)
    return (R, 8, 192, NUM_SMS)


def _plan(layer, m, override=None):
    MT = _mt(m)
    T = -(-m // (8 * MT))
    R, NW, ring_kb, NC = override or CFG.get((layer, m)) or _default(layer, m)
    if T > 1 and override is None:
        NC = max(1, NC // T)
    return MT, T, R, NW, ring_kb, NC


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
        for R in (1, 2, 4):
            state["packs"][(name, R)] = _pack(w, R)
    for name in ("qkv_gate", "o", "pooler", "unpooler"):
        state["w"][name] = weights[name].contiguous()
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    state["ws"] = torch.zeros(WS_FLOATS, dtype=torch.float32, device=dev)
    state["cnt"] = torch.zeros(1 << 16, dtype=torch.int32, device=dev)
    # warm every kernel instantiation used by kernel_fn (sets smem attributes outside capture)
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
    w_eff = state["eff"][layer]
    K32 = w_eff.shape[1] // 32
    xk32 = x.shape[1] // 32
    MT, T, R, NW, ring_kb, NC = _plan(layer, m, override)
    if mt is not None:
        MT, T = mt, -(-m // (8 * mt))
    dt = torch.float32 if layer == "router" else torch.bfloat16
    out = torch.full((m, n), float("nan"), dtype=dt, device=x.device) if _DEBUG_NAN else \
        torch.empty((m, n), dtype=dt, device=x.device)
    state["ext"].stream_gemm(state["packs"][(layer, R)], x, state["ws"], state["cnt"], out,
                             m, n, K32, xk32, NC, T, MT, R, NW, ring_kb)
    return out


def kernel_fn(data, state):
    x = data["x"].contiguous()
    if x.data_ptr() % 16:
        x = x.clone()
    cfg = data["config"]
    layer, m, n = cfg["layer"], x.shape[0], cfg["n"]
    if layer == "router":
        if m * n <= WS_FLOATS:
            return _run(state, layer, x, m, n)
        acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
        acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
        return acc
    if m <= SKINNY_MAX_M:
        return _run(state, layer, x, m, n)
    return x @ state["w"][layer][:n].T


# --------------------------------------------------------------------------------------
# Development-only sweep (prints timings; never affects results)
# --------------------------------------------------------------------------------------
SCORED = [(m, l) for m in (1, 16, 32, 8, 128, 256) for l in ("qkv_gate", "o", "pooler", "unpooler", "router")]
LAYERS = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096),
          "router": (256, 8192)}


def _nfor(layer, m):
    return 1024 if (layer == "unpooler" and m <= 8) else LAYERS[layer][0]


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
    from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl
    flush = torch.zeros(64 << 20, dtype=torch.float32, device=dev)
    dummy = torch.zeros(1, device=dev)
    ovh = _bench(lambda: dummy.add_(1), flush)
    print("empty graph (us): %.2f  (subtracted below)" % ovh)
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
        MT = _mt(m)
        T = -(-m // (8 * MT))
        if layer == "router" or m <= SKINNY_MAX_M:
            cands = []
            for R in (1, 2, 4):
                for NW in (4, 8):
                    cands.append((R, NW, 192, max(1, NUM_SMS // T)))
                    cands.append((R, NW, 96, max(1, 2 * NUM_SMS // T)))
            mts = [None]
        else:
            cands = [(1, 8, 192, NUM_SMS // 2), (2, 8, 192, NUM_SMS // 2), (2, 8, 96, NUM_SMS)]
            mts = [8]
        res = []
        maxerr = 0.0
        for c in cands:
            for mt in mts:
                try:
                    out = _run(state, layer, x, m, n, c, mt)
                    err = float((out.double() - ref).norm() / ref.norm())
                    maxerr = max(maxerr, err)
                    t = _bench(lambda: _run(state, layer, x, m, n, c, mt), flush) - ovh
                    res.append((t, c, err))
                except Exception as e:
                    print("  cfg", c, "failed", repr(e)[:300])
        res.sort(key=lambda r: r[0])
        line += " best:" + " ".join("%s=%.2f" % (str(r[1]), r[0]) for r in res[:5])
        line += " maxerr=%.1e" % maxerr
        print(line, flush=True)
    print("sweep took %.1fs" % (time.time() - t0))
    _stress(state, weights, gen)


def _stress(state, weights, gen, reps=12):
    """Run the default kernel_fn path repeatedly on fresh inputs with NaN-prefilled outputs."""
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
