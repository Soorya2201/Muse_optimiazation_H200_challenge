# okc: problem=muse_bf16_proj
"""Skinny BF16 GEMM for Muse Spark Open dense projections on H200.

Swap-AB tensor-core GEMV: W (n x k) is the MMA "A" operand (n fills the 16-row MMA M dim),
x (m x k) is the "B" operand (tokens fill the 8-wide MMA N dim). W is repacked once in
prepare() into a warp-fragment-ordered layout so every warp load is a contiguous 512 B
stream read straight into MMA registers (no shared memory round trip). The K order inside
each 32-wide block is permuted identically for W and x, so each thread's 16 B load of W and
of x feeds two m16n8k16 MMAs directly. Split-K across CTAs is reduced in a single kernel
(fp32 atomics + last-CTA fix-up that also re-zeros the workspace).

Router: FP32 weights are split once into BF16 hi + lo, concatenated along K, so one pass
computes x@hi^T + x@lo^T with FP32 accumulation (what SGLang does above 8 tokens).
"""

import math
import os
from typing import TypedDict

import torch

DEV = True  # run the config sweep in prepare (check runs only)


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

__device__ __forceinline__ uint4 ldg_stream(const uint4* p) {
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
        : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}
__device__ __forceinline__ uint4 ldg_x(const void* p) {
    uint4 r;
    asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
        : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}
__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Wp: packed [G][K32][R][2][32] uint4. CTA = (g, s, tt) flattened with tt fastest.
template <int MT, int R, int NW, int U, bool F32>
__global__ void __launch_bounds__(NW * 32)
skinny_kernel(const uint4* __restrict__ Wp, const __nv_bfloat16* __restrict__ x,
              float* __restrict__ ws, unsigned* __restrict__ cnt, void* __restrict__ out,
              int m, int n, int K, int K32, int xk32, int S, int T) {
    constexpr int NE = R * MT * 4;
    __shared__ float red[NW][NE][32];
    __shared__ unsigned s_last;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    int bid = blockIdx.x;
    const int tt = bid % T; bid /= T;
    const int s = bid % S;
    const int g = bid / S;
    const int j0 = (int)(((long long)s * K32) / S);
    const int j1 = (int)(((long long)(s + 1) * K32) / S);
    const int c = lane & 3, gr = lane >> 2;

    float acc[R][MT][4];
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int mt = 0; mt < MT; mt++)
#pragma unroll
            for (int q = 0; q < 4; q++) acc[r][mt][q] = 0.f;

    const uint4* wbase = Wp + (size_t)g * K32 * R * 64 + lane;
    const __nv_bfloat16* xrow[MT];
    bool xv[MT];
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
        int tok = tt * MT * 8 + mt * 8 + gr;
        xv[mt] = tok < m;
        xrow[mt] = x + (size_t)(xv[mt] ? tok : 0) * K + c * 8;
    }

    for (int jb = j0 + w; jb < j1; jb += NW * U) {
        uint4 wa[U][R][2];
        uint4 xb[U][MT];
#pragma unroll
        for (int u = 0; u < U; u++) {
            const int j = jb + u * NW;
            const bool ok = j < j1;
#pragma unroll
            for (int r = 0; r < R; r++)
#pragma unroll
                for (int h = 0; h < 2; h++)
                    wa[u][r][h] = ok ? ldg_stream(wbase + ((size_t)j * R + r) * 64 + h * 32) : make_uint4(0, 0, 0, 0);
            const int jx = j >= xk32 ? j - xk32 : j;
#pragma unroll
            for (int mt = 0; mt < MT; mt++)
                xb[u][mt] = (ok && xv[mt]) ? ldg_x(xrow[mt] + (size_t)jx * 32) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int u = 0; u < U; u++)
#pragma unroll
            for (int r = 0; r < R; r++)
#pragma unroll
                for (int mt = 0; mt < MT; mt++) {
                    mma16816(acc[r][mt], wa[u][r][0].x, wa[u][r][1].x, wa[u][r][0].y, wa[u][r][1].y,
                             xb[u][mt].x, xb[u][mt].y);
                    mma16816(acc[r][mt], wa[u][r][0].z, wa[u][r][1].z, wa[u][r][0].w, wa[u][r][1].w,
                             xb[u][mt].z, xb[u][mt].w);
                }
    }

#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int mt = 0; mt < MT; mt++)
#pragma unroll
            for (int q = 0; q < 4; q++) red[w][(r * MT + mt) * 4 + q][lane] = acc[r][mt][q];
    __syncthreads();

    const int rowbase = g * 16 * R;
    for (int e = threadIdx.x; e < NE * 32; e += NW * 32) {
        const int i = e >> 5, ln = e & 31;
        float v = 0.f;
#pragma unroll
        for (int ww = 0; ww < NW; ww++) v += red[ww][i][ln];
        const int r = i / (MT * 4), mt = (i >> 2) % MT, q = i & 3;
        const int row = rowbase + r * 16 + (ln >> 2) + ((q & 2) ? 8 : 0);
        const int tok = tt * MT * 8 + mt * 8 + (ln & 3) * 2 + (q & 1);
        if (tok < m) {
            const size_t idx = (size_t)tok * n + row;
            if (S == 1) {
                if (F32) ((float*)out)[idx] = v;
                else ((__nv_bfloat16*)out)[idx] = __float2bfloat16(v);
            } else {
                atomicAdd(ws + idx, v);
            }
        }
    }
    if (S > 1) {
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned prev = atomicAdd(cnt + g * T + tt, 1u);
            s_last = (prev == (unsigned)(S - 1));
        }
        __syncthreads();
        if (s_last) {
            __threadfence();
            for (int e = threadIdx.x; e < NE * 32; e += NW * 32) {
                const int i = e >> 5, ln = e & 31;
                const int r = i / (MT * 4), mt = (i >> 2) % MT, q = i & 3;
                const int row = rowbase + r * 16 + (ln >> 2) + ((q & 2) ? 8 : 0);
                const int tok = tt * MT * 8 + mt * 8 + (ln & 3) * 2 + (q & 1);
                if (tok < m) {
                    const size_t idx = (size_t)tok * n + row;
                    float v = __ldcg(ws + idx);
                    if (F32) ((float*)out)[idx] = v;
                    else ((__nv_bfloat16*)out)[idx] = __float2bfloat16(v);
                    __stcg(ws + idx, 0.f);
                }
            }
            if (threadIdx.x == 0) cnt[g * T + tt] = 0u;
        }
    }
}

template <int MT, int R, int NW, int U, bool F32>
static void launch(const torch::Tensor& Wp, const torch::Tensor& x, torch::Tensor& ws, torch::Tensor& cnt,
                   torch::Tensor& out, int m, int n, int K, int K32, int xk32, int S, int T) {
    const int G = n / (16 * R);
    dim3 grid(G * S * T);
    auto stream = at::cuda::getCurrentCUDAStream();
    skinny_kernel<MT, R, NW, U, F32><<<grid, NW * 32, 0, stream>>>(
        (const uint4*)Wp.data_ptr(), (const __nv_bfloat16*)x.data_ptr(), ws.data_ptr<float>(),
        (unsigned*)cnt.data_ptr<int>(), out.data_ptr(), m, n, K, K32, xk32, S, T);
}

#define DISPATCH_U(MT, R, NW, F32)                                                      \
    if (U == 1) return launch<MT, R, NW, 1, F32>(Wp, x, ws, cnt, out, m, n, K, K32, xk32, S, T); \
    if (U == 2) return launch<MT, R, NW, 2, F32>(Wp, x, ws, cnt, out, m, n, K, K32, xk32, S, T); \
    if (U == 4) return launch<MT, R, NW, 4, F32>(Wp, x, ws, cnt, out, m, n, K, K32, xk32, S, T);
#define DISPATCH_NW(MT, R, F32)                \
    if (NW == 4) { DISPATCH_U(MT, R, 4, F32) } \
    if (NW == 8) { DISPATCH_U(MT, R, 8, F32) }
#define DISPATCH_R(MT, F32)                \
    if (R == 1) { DISPATCH_NW(MT, 1, F32) } \
    if (R == 2) { DISPATCH_NW(MT, 2, F32) }
#define DISPATCH_MT(F32)                   \
    if (MT == 1) { DISPATCH_R(1, F32) }    \
    if (MT == 2) { DISPATCH_R(2, F32) }    \
    if (MT == 4) { DISPATCH_R(4, F32) }

static void dispatch(const torch::Tensor& Wp, const torch::Tensor& x, torch::Tensor& ws, torch::Tensor& cnt,
                     torch::Tensor& out, int m, int n, int K, int K32, int xk32, int S, int T,
                     int MT, int R, int NW, int U, bool f32) {
    if (f32) { DISPATCH_MT(true) } else { DISPATCH_MT(false) }
    TORCH_CHECK(false, "unsupported config");
}

void skinny(torch::Tensor Wp, torch::Tensor x, torch::Tensor ws, torch::Tensor cnt, torch::Tensor out,
            int64_t m, int64_t n, int64_t K32, int64_t xk32, int64_t S, int64_t T,
            int64_t MT, int64_t R, int64_t NW, int64_t U) {
    const int K = (int)x.size(1);
    dispatch(Wp, x, ws, cnt, out, (int)m, (int)n, K, (int)K32, (int)xk32, (int)S, (int)T,
             (int)MT, (int)R, (int)NW, (int)U, out.scalar_type() == torch::kFloat32);
}
"""

CPP_SRC = r"""
void skinny(torch::Tensor Wp, torch::Tensor x, torch::Tensor ws, torch::Tensor cnt, torch::Tensor out,
            int64_t m, int64_t n, int64_t K32, int64_t xk32, int64_t S, int64_t T,
            int64_t MT, int64_t R, int64_t NW, int64_t U);
"""

_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
        from torch.utils.cpp_extension import load_inline
        _EXT = load_inline(
            name="muse_skinny_v1",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["skinny"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
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


# (layer, m) -> (R, NW, U, S). Missing entries use the heuristic.
CFG = {}


def _heuristic(layer, m, n, K32):
    MT = 1 if m <= 8 else (2 if m <= 16 else 4)
    T = -(-m // (8 * MT))
    R, NW, U = 2, 8, 2
    G = n // (16 * R)
    S = max(1, min((2 * NUM_SMS) // (G * T), K32 // (NW * U)))
    return R, NW, U, S


def _plan(layer, m, n, K32, override=None):
    MT = 1 if m <= 8 else (2 if m <= 16 else 4)
    T = -(-m // (8 * MT))
    R, NW, U, S = override or CFG.get((layer, m)) or _heuristic(layer, m, n, K32)
    return MT, T, R, NW, U, S


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
    # cuBLAS fallbacks for large m (same as SGLang)
    for name in ("qkv_gate", "o", "pooler", "unpooler"):
        state["w"][name] = weights[name].contiguous()
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    state["ws"] = torch.zeros(WS_FLOATS, dtype=torch.float32, device=dev)
    state["cnt"] = torch.zeros(1 << 16, dtype=torch.int32, device=dev)
    if DEV:
        try:
            _dev_sweep(state, weights)
        except Exception as e:  # noqa
            import traceback
            print("DEV sweep failed:", repr(e))
            traceback.print_exc()
    return state


def _run_skinny(state, layer, x, m, n, override=None):
    w_eff = state["eff"][layer]
    k_eff = w_eff.shape[1]
    K32 = k_eff // 32
    xk32 = x.shape[1] // 32
    MT, T, R, NW, U, S = _plan(layer, m, n, K32, override)
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    state["ext"].skinny(state["packs"][(layer, R)], x, state["ws"], state["cnt"], out,
                        m, n, K32, xk32, S, T, MT, R, NW, U)
    return out


def kernel_fn(data, state):
    x = data["x"].contiguous()
    cfg = data["config"]
    layer, m, n = cfg["layer"], x.shape[0], cfg["n"]
    if layer == "router":
        if m * n <= WS_FLOATS:
            return _run_skinny(state, layer, x, m, n)
        acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
        acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
        return acc
    if m <= SKINNY_MAX_M:
        return _run_skinny(state, layer, x, m, n)
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
    print("torch", torch.__version__)
    try:
        import triton
        print("triton", triton.__version__)
    except Exception as e:
        print("no triton", e)
    try:
        import deep_gemm
        print("deep_gemm attrs:", [a for a in dir(deep_gemm) if not a.startswith("_")])
    except Exception as e:
        print("no deep_gemm", e)
    from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl
    flush = torch.zeros(64 << 20, dtype=torch.float32, device=dev)  # 256 MB
    # empty-graph overhead
    dummy = torch.zeros(1, device=dev)
    print("empty graph (us): %.2f" % _bench(lambda: dummy.add_(1), flush))
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
        tb = _bench(lambda: base(layer, x, n), flush)
        line = "m=%3d %-9s base=%7.2f" % (m, layer, tb)
        if layer == "router" or m <= SKINNY_MAX_M:
            K32 = state["eff"][layer].shape[1] // 32
            cands = [None]
            MT = 1 if m <= 8 else (2 if m <= 16 else 4)
            T = -(-m // (8 * MT))
            for R in (1, 2):
                G = n // (16 * R)
                for NW, U in ((4, 2), (8, 2), (4, 4), (8, 1)):
                    for mult in (1, 2, 4):
                        S = max(1, min((mult * NUM_SMS) // (G * T), K32 // (NW * U)))
                        c = (R, NW, U, S)
                        if c not in cands:
                            cands.append(c)
            res = []
            for c in cands:
                try:
                    out = _run_skinny(state, layer, x, m, n, c)
                    err = float((out.double() - ref).norm() / ref.norm())
                    t = _bench(lambda: _run_skinny(state, layer, x, m, n, c), flush)
                    res.append((t, c, err))
                except Exception as e:
                    print("  cfg", c, "failed", repr(e))
            res.sort(key=lambda r: r[0])
            dflt = [r for r in res if r[1] is None]
            if dflt:
                line += " dflt=%7.2f(err %.1e)" % (dflt[0][0], dflt[0][2])
            line += " best:" + " ".join("%s=%.2f" % (str(r[1]), r[0]) for r in res[:4])
            line += " maxerr=%.1e" % max(r[2] for r in res)
        print(line, flush=True)
    try:
        _triton_sweep(weights, flush, gen)
    except Exception as e:
        import traceback
        print("triton sweep failed", repr(e))
        traceback.print_exc()
    print("sweep took %.1fs" % (time.time() - t0))


def _triton_sweep(weights, flush, gen):
    import triton
    import triton.language as tl

    @triton.jit
    def _tl_gemm(W, X, O, M, N, K, BN: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rn = pid_n * BN + tl.arange(0, BN)
        rm = pid_m * BM + tl.arange(0, BM)
        rk = tl.arange(0, BK)
        wp = W + rn[:, None] * K + rk[None, :]
        xp = X + rm[None, :] * K + rk[:, None]
        acc = tl.zeros((BN, BM), dtype=tl.float32)
        for k in range(0, K, BK):
            w = tl.load(wp)
            xt = tl.load(xp, mask=rm[None, :] < M, other=0.0)
            acc = tl.dot(w, xt, acc)
            wp += BK
            xp += BK
        op = O + rm[None, :] * N + rn[:, None]
        tl.store(op, acc.to(tl.bfloat16), mask=rm[None, :] < M)

    dev = weights["qkv_gate"].device
    cfgs = [(64, 64, 64, 4, 4), (64, 128, 64, 4, 4), (128, 64, 64, 4, 4), (128, 128, 64, 4, 8),
            (64, 256, 64, 3, 8), (128, 256, 64, 3, 8), (64, 128, 128, 3, 4)]
    for m in (128, 256):
        for layer in ("qkv_gate", "o", "pooler", "unpooler"):
            n, k = LAYERS[layer]
            w = weights[layer]
            x = torch.randn((m, k), generator=gen, device=dev).to(torch.bfloat16)
            ref = x.float() @ w.float().T
            res = []
            for (BN, BM, BK, st, nw) in cfgs:
                if BM > m:
                    continue
                o = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
                grid = (triton.cdiv(m, BM), n // BN)
                fn = lambda: _tl_gemm[grid](w, x, o, m, n, k, BN=BN, BM=BM, BK=BK, num_stages=st, num_warps=nw)
                try:
                    fn()
                    err = float((o.float() - ref).norm() / ref.norm())
                    t = _bench(fn, flush)
                    res.append((t, (BN, BM, BK, st, nw), err))
                except Exception as e:
                    print("  triton cfg", (BN, BM, BK, st, nw), "failed", repr(e)[:200])
            res.sort(key=lambda r: r[0])
            print("triton m=%d %s: " % (m, layer) + " ".join("%s=%.2f(e%.0e)" % (str(c), t, e) for t, c, e in res[:4]), flush=True)
