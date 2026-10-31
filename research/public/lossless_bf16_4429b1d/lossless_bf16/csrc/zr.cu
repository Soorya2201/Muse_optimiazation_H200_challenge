
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace zr {

#define ZR_DEVI __device__ __forceinline__

constexpr int BLK = 1536;

struct Args {
    const uint8_t* wp;          // [ntiles][kcn][BLK] (zg_pack of W.reshape(N * 8, K / 8))
    const float* rscale;        // [ntiles * 16] 2^-s per virtual row
    const int4* etab;           // [ntiles][emax]: {count}, then {k in V, virtual row in tile, bits(w), bits(nv)}
    const __nv_bfloat16* x;     // [K]
    __nv_bfloat16* out;         // [2 ntiles] bf16 (PAIR == false)
    float* outf;                // [ntiles] fp32 (PAIR == true)
    int kseg, kcn, emax;        // K / 8, K / 512, escape table width
};

ZR_DEVI uint4 ld_stream(const uint8_t* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
ZR_DEVI void mma_bf16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
ZR_DEVI uint32_t word_of(const uint4& q, int i) { return i == 0 ? q.x : (i == 1 ? q.y : (i == 2 ? q.z : q.w)); }
ZR_DEVI float x_at(const __nv_bfloat16* p) {
    return __uint_as_float(((uint32_t)__ldg(reinterpret_cast<const unsigned short*>(p))) << 16);
}
// escape record {k in V, virtual row in tile (bit 3: second row, bits 0-2: segment), bits(w), bits(nv)} -> x index
ZR_DEVI int esc_xidx(const int4& r, int kseg) { return (r.y & 7) * kseg + r.x; }

template <int NW, int WPT, int U, bool PAIR, int MINB>
__global__ void __launch_bounds__(NW * 32, MINB) zr_kernel(const __grid_constant__ Args a) {
    constexpr int TPC = NW / WPT;
    static_assert(NW % WPT == 0, "zr: WPT must divide NW");
    __shared__ float sred[TPC][WPT][4];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t4 = lane & 3;
    const int tile = (int)blockIdx.x * TPC + warp / WPT;
    const int q = warp % WPT;
    const int4* et = a.etab + (size_t)tile * a.emax;
    int4 er = make_int4(0, 0, 0, 0);
    int ecnt = 0;                          // the tile's record count (bounds the tail loop below)
    if (q == 0) {
        ecnt = __ldg(&et[0].x);
        if (1 + lane < a.emax) er = __ldg(et + 1 + lane);
    }
    const float sc0 = __ldg(a.rscale + (size_t)tile * 16 + g);
    const float sc1 = __ldg(a.rscale + (size_t)tile * 16 + 8 + g);
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    float ex = 0.f;
    const uint8_t* wb = a.wp + (size_t)tile * a.kcn * BLK + lane * 16;
    const __nv_bfloat16* xs = a.x + (size_t)g * a.kseg + 16 * t4;
    bool first = true;
    for (int kb = q; kb < a.kcn; kb += WPT * U) {
        uint4 wq[U][3];
        uint4 xq[U][2];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int kc = kb + u * WPT;
            if (kc < a.kcn) {
#pragma unroll
                for (int qq = 0; qq < 3; ++qq) wq[u][qq] = ld_stream(wb + (size_t)kc * BLK + qq * 512);
                const uint4* xp = reinterpret_cast<const uint4*>(xs + kc * 64);
                xq[u][0] = __ldg(xp);
                xq[u][1] = __ldg(xp + 1);
            }
        }
        if (first) {
            // the escape record was issued before the weights, so it is back by now: fetch its x value while the
            // weights are still in flight
            first = false;
            if (q == 0) ex = x_at(a.x + esc_xidx(er, a.kseg));
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int kc = kb + u * WPT;
            if (kc < a.kcn) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint32_t La = word_of(wq[u][j >> 1], (j & 1) * 2);
                    const uint32_t Lb = word_of(wq[u][j >> 1], (j & 1) * 2 + 1);
                    const uint32_t Nw = word_of(wq[u][2], j);
                    const uint32_t Ha = (Nw & 0x87878787u) | 0x38383838u;
                    const uint32_t Hb = (__funnelshift_r(Nw, Nw, 4) & 0x87878787u) | 0x38383838u;
                    mma_bf16(acc, __byte_perm(La, Ha, 0x5140), __byte_perm(La, Ha, 0x7362), __byte_perm(Lb, Hb, 0x5140),
                             __byte_perm(Lb, Hb, 0x7362), word_of(xq[u][j >> 1], (j & 1) * 2),
                             word_of(xq[u][j >> 1], (j & 1) * 2 + 1));
                }
            }
        }
    }
    // diagonal lanes: (g, t4 = g >> 1) holds D[g][g] in acc[g & 1] and D[8 + g][g] in acc[2 + (g & 1)]
    const bool dl = t4 == (g >> 1);
    float v0 = dl ? ((g & 1) ? acc[1] : acc[0]) * sc0 : 0.f;
    float v1 = dl ? ((g & 1) ? acc[3] : acc[2]) * sc1 : 0.f;
    // escapes: v += x * nv cancels the decoded 2^-15 placeholder (x * nv = -x * 2^-15 * scale exactly), and the
    // exact products x * w are summed apart (e) and added last, so a one-hot x gives x * w bit-exactly even when w is
    // far below the row's window (zg's fmaf(x, w, x * nv) rounds such w away)
    float e0 = 0.f, e1 = 0.f;
    if (q == 0) {
        const float xn = ex * __int_as_float(er.w), xw = ex * __int_as_float(er.z);
        if (er.y & 8) { v1 += xn; e1 += xw; } else { v0 += xn; e0 += xw; }
        for (int e = 33 + lane; e <= ecnt; e += 32) {    // records past the first 32 of the tile (rare)
            const int4 r = __ldg(et + e);
            const float xv = x_at(a.x + esc_xidx(r, a.kseg));
            const float rn = xv * __int_as_float(r.w), rw = xv * __int_as_float(r.z);
            if (r.y & 8) { v1 += rn; e1 += rw; } else { v0 += rn; e0 += rw; }
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v0 += __shfl_xor_sync(0xffffffffu, v0, o);
        v1 += __shfl_xor_sync(0xffffffffu, v1, o);
        e0 += __shfl_xor_sync(0xffffffffu, e0, o);
        e1 += __shfl_xor_sync(0xffffffffu, e1, o);
    }
    if constexpr (WPT == 1) {
        if (lane == 0) {
            if constexpr (PAIR) a.outf[tile] = (v0 + e0) + (v1 + e1);
            else reinterpret_cast<__nv_bfloat162*>(a.out)[tile] = __floats2bfloat162_rn(v0 + e0, v1 + e1);
        }
    } else {
        if (lane == 0) {
            sred[warp / WPT][q][0] = v0;
            sred[warp / WPT][q][1] = v1;
            sred[warp / WPT][q][2] = e0;
            sred[warp / WPT][q][3] = e1;
        }
        __syncthreads();
        if (tid < TPC) {
            float s0 = 0.f, s1 = 0.f, f0 = 0.f, f1 = 0.f;
#pragma unroll
            for (int i = 0; i < WPT; ++i) {
                s0 += sred[tid][i][0];
                s1 += sred[tid][i][1];
                f0 += sred[tid][i][2];
                f1 += sred[tid][i][3];
            }
            const int t = (int)blockIdx.x * TPC + tid;
            if constexpr (PAIR) a.outf[t] = (s0 + f0) + (s1 + f1);
            else reinterpret_cast<__nv_bfloat162*>(a.out)[t] = __floats2bfloat162_rn(s0 + f0, s1 + f1);
        }
    }
}

// (id, NW, WPT, U, PAIR, MINB)
#define ZR_CFG_LIST(X) \
    X(0, 8, 8, 2, false, 3) X(1, 4, 4, 4, false, 4) X(2, 16, 16, 1, false, 2) \
    X(3, 8, 8, 1, false, 4) X(4, 4, 4, 2, false, 6) X(5, 8, 4, 2, false, 3) \
    X(6, 4, 2, 1, false, 8) X(7, 4, 1, 2, false, 6) X(8, 8, 2, 1, false, 4) \
    X(9, 8, 8, 2, true, 3) X(10, 16, 16, 1, true, 2) X(11, 4, 4, 4, true, 4)

template <int NW, int WPT, int U, bool PAIR, int MINB>
static void attrs(int64_t* p) {
    auto kern = zr_kernel<NW, WPT, U, PAIR, MINB>;
    cudaFuncAttributes fa;
    if (cudaFuncGetAttributes(&fa, reinterpret_cast<const void*>(kern)) == cudaSuccess) {
        p[0] = fa.numRegs;
        p[1] = (int64_t)fa.localSizeBytes;
    } else {
        cudaGetLastError();
        p[0] = -1;
        p[1] = -1;
    }
    int bps = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kern, NW * 32, 0);
    p[2] = bps;
}

}  // namespace zr

// per cfg: registers, local bytes, CTAs/SM (prepare-time diagnostic), as [n_cfg][4] {id, regs, local, bps}
torch::Tensor zr_info() {
    int n = 0;
#define X(id, nw, wpt, u, pair, minb) ++n;
    ZR_CFG_LIST(X)
#undef X
    auto t = torch::zeros({n, 4}, torch::dtype(torch::kInt64));
    int64_t* p = t.data_ptr<int64_t>();
    int i = 0;
#define X(id, nw, wpt, u, pair, minb) p[4 * i] = id; zr::attrs<nw, wpt, u, pair, minb>(p + 4 * i + 1); ++i;
    ZR_CFG_LIST(X)
#undef X
    return t;
}

void zr_gemv(torch::Tensor x, torch::Tensor wp, torch::Tensor rscale, torch::Tensor etab, int64_t emax,
             c10::optional<torch::Tensor> out, c10::optional<torch::Tensor> outf, int64_t k, int64_t ntiles,
             int64_t cfg) {
    int tpc = 0;
    bool pair = false;
    switch (cfg) {
#define X(id, nw_, wpt_, u_, pair_, minb_) case id: tpc = nw_ / wpt_; pair = pair_; break;
        ZR_CFG_LIST(X)
#undef X
        default: TORCH_CHECK(false, "zr: bad cfg ", cfg);
    }
    TORCH_CHECK(k % 512 == 0 && ntiles % tpc == 0 && x.numel() >= k, "zr: shape k=", k, " ntiles=", ntiles);
    TORCH_CHECK(pair ? (outf.has_value() && !out.has_value()) : (out.has_value() && !outf.has_value()),
                "zr: output kind does not match cfg ", cfg);
    zr::Args a;
    a.wp = wp.data_ptr<uint8_t>();
    a.rscale = rscale.data_ptr<float>();
    a.etab = reinterpret_cast<const int4*>(etab.data_ptr<int>());
    a.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    a.out = out.has_value() ? reinterpret_cast<__nv_bfloat16*>(out.value().data_ptr()) : nullptr;
    a.outf = outf.has_value() ? outf.value().data_ptr<float>() : nullptr;
    a.kseg = (int)(k / 8);
    a.kcn = (int)(k / 512);
    a.emax = (int)emax;
    cudaStream_t st_ = at::cuda::getCurrentCUDAStream();
    const int grid = (int)(ntiles / tpc);
    switch (cfg) {
#define X(id, nw_, wpt_, u_, pair_, minb_) \
        case id: zr::zr_kernel<nw_, wpt_, u_, pair_, minb_><<<grid, nw_ * 32, 0, st_>>>(a); break;
        ZR_CFG_LIST(X)
#undef X
    }
}
