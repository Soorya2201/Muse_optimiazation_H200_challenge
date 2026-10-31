
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define DEVI __device__ __forceinline__

namespace zt {

constexpr int NCW = 8;                   // consumer warps = chunks per stage
constexpr int SC = NCW;
constexpr int NTHR = 32 * (NCW + 2);     // + producer warp + escape warp
constexpr int BLK = 1536;
constexpr int XLD = SC * 128 + 16;       // smem bytes per token row of a stage (padded: conflict-free LDS.128)

struct Args {
    const uint8_t* wp;          // [rts][kcn][BLK]
    const float* rscale;        // [rows]
    const int4* etab;           // [rts][emax]: {count}, then {k, row in tile, bits(w), bits(nv)}
    const __nv_bfloat16* x;     // [m][k]
    __nv_bfloat16* out;         // [m][nout]
    int m, k, kcn, nout, emax;
};

DEVI uint32_t smem_addr(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
DEVI void mbar_init(uint64_t* b, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(smem_addr(b)), "r"(count) : "memory");
}
DEVI void mbar_arrive(uint64_t* b) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(smem_addr(b)) : "memory");
}
DEVI void mbar_arrive_tx(uint64_t* b, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(smem_addr(b)), "r"(bytes) : "memory");
}
DEVI void mbar_wait(uint64_t* b, int parity) {
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@!P1 bra WAIT_%=;\n"
        "}\n" :: "r"(smem_addr(b)), "r"(parity) : "memory");
}
DEVI void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint64_t* bar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(smem_addr(bar)) : "memory");
}
DEVI void bulk_g2s_hint(uint32_t dst, const void* src, uint32_t bytes, uint64_t* bar, uint64_t pol) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(smem_addr(bar)), "l"(pol) : "memory");
}
DEVI uint64_t evict_first_policy() {
    uint64_t p;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(p));
    return p;
}
DEVI void mma_bf16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
DEVI uint32_t word_of(const uint4& q, int i) { return i == 0 ? q.x : (i == 1 ? q.y : (i == 2 ? q.z : q.w)); }

DEVI void esc_cp_async4(uint32_t dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(dst), "l"(src) : "memory");
}
DEVI void esc_cp_wait_all() {
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
}

// s_xv: 32 * NTOK words of shared memory. scorr: [RT * 16][NTOK], zeroed, written only by this warp.
template <int NTOK, int RT>
DEVI void escape_warp(const int4* __restrict__ etab, int emax, int rt0, int klo, int khi,
                      const __nv_bfloat16* __restrict__ xg, int K, int nvalid, float (*scorr)[NTOK],
                      uint32_t* s_xv, int lane) {
    constexpr unsigned FULL = 0xffffffffu;
    const int my_cnt = lane < RT ? etab[(size_t)(rt0 + lane) * emax].x : 0;
    int cnts[RT];
    int tot = 0;
#pragma unroll
    for (int r = 0; r < RT; ++r) {
        cnts[r] = __shfl_sync(FULL, my_cnt, r);
        tot += cnts[r];
    }
    const uint32_t xv_base = static_cast<uint32_t>(__cvta_generic_to_shared(s_xv));
    for (int f0 = 0; f0 < tot; f0 += 128) {
        int4 E[4];
        bool in[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            int f = f0 + q * 32 + lane;
            in[q] = false;
            E[q] = make_int4(0, 0, 0, 0);
            if (f < tot) {
                int r = 0;
#pragma unroll
                for (int rr = 0; rr < RT - 1; ++rr)
                    if (r == rr && f >= cnts[rr]) { f -= cnts[rr]; r = rr + 1; }
                E[q] = etab[(size_t)(rt0 + r) * emax + 1 + f];
                E[q].y += r * 16;
                in[q] = E[q].x >= klo && E[q].x < khi;
            }
        }
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            if (f0 + q * 32 >= tot) break;
            if (in[q]) {
                const __nv_bfloat16* src = xg + (E[q].x & ~1);
                for (int t = 0; t < nvalid; ++t)
                    esc_cp_async4(xv_base + (uint32_t)((lane * NTOK + t) * 4), src + (size_t)t * K);
            }
            esc_cp_wait_all();
            __syncwarp();
            const unsigned inmask = __ballot_sync(FULL, in[q]);
            for (int j = 0; j < 32; ++j) {
                if (!((inmask >> j) & 1u)) continue;
                const int ek = __shfl_sync(FULL, E[q].x, j), er = __shfl_sync(FULL, E[q].y, j);
                const float w = __int_as_float(__shfl_sync(FULL, E[q].z, j));
                const float nv = __int_as_float(__shfl_sync(FULL, E[q].w, j));
                for (int t = lane; t < nvalid; t += 32) {
                    const uint32_t u = s_xv[j * NTOK + t];
                    const float xv = __uint_as_float((ek & 1) ? (u & 0xffff0000u) : (u << 16));
                    scorr[er][t] += fmaf(xv, w, xv * nv);
                }
            }
            __syncwarp();
        }
    }
}

template <int NT, int RT, int ST>
struct Cfg {
    static constexpr int NTOK = NT * 8, ROWS = RT * 16;
    static constexpr int WB = RT * SC * BLK;
    static constexpr int STAGE = WB + NTOK * XLD;
    static constexpr int SMEM = ST * STAGE;
};

template <int NT, int RT, int ST>
__global__ void __launch_bounds__(NTHR) zt_kernel(const __grid_constant__ Args a) {
    using C = Cfg<NT, RT, ST>;
    constexpr int NTOK = C::NTOK, ROWS = C::ROWS;
    extern __shared__ __align__(128) uint8_t smem[];
    __shared__ __align__(8) uint64_t full[ST], empty[ST];
    __shared__ float sred[NCW][ROWS][NTOK + 1];
    __shared__ float scorr[ROWS][NTOK];
    __shared__ uint32_t s_xv[32 * NTOK];
    __shared__ float ssc[ROWS];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int rt0 = blockIdx.x * RT;
    const int nvalid = min(NTOK, a.m);
    const int nst = (a.kcn + SC - 1) / SC;
    const uint32_t sbase = smem_addr(smem);
    for (int i = tid; i < ROWS * NTOK; i += NTHR) (&scorr[0][0])[i] = 0.f;
    if (tid == 0) {
#pragma unroll
        for (int i = 0; i < ST; ++i) {
            mbar_init(&full[i], 1);
            mbar_init(&empty[i], NCW);
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();

    if (warp == NCW) {
        // ================= producer warp =================
        const uint64_t pol = evict_first_policy();
        for (int s = 0; s < nst; ++s) {
            const int slot = s % ST;
            if (s >= ST) mbar_wait(&empty[slot], ((s / ST) - 1) & 1);
            const int kc0 = s * SC, nch = min(SC, a.kcn - kc0);
            const uint32_t st = sbase + slot * C::STAGE;
            if (lane == 0) {
                mbar_arrive_tx(&full[slot], (uint32_t)(RT * nch * BLK + nvalid * nch * 128));
#pragma unroll
                for (int r = 0; r < RT; ++r)
                    bulk_g2s_hint(st + r * SC * BLK, a.wp + ((size_t)(rt0 + r) * a.kcn + kc0) * BLK, nch * BLK, &full[slot], pol);
            }
            __syncwarp();
            for (int t = lane; t < nvalid; t += 32)
                bulk_g2s(st + C::WB + t * XLD, a.x + (size_t)t * a.k + (size_t)kc0 * 64, nch * 128, &full[slot]);
        }
    } else if (warp == NCW + 1) {
        // ================= escape warp =================
        if (lane < ROWS) ssc[lane] = a.rscale[rt0 * 16 + lane];
        escape_warp<NTOK, RT>(a.etab, a.emax, rt0, 0, a.k, a.x, a.k, nvalid, scorr, s_xv, lane);
    } else {
        // ================= consumer warps =================
        const int g = lane >> 2, t4 = lane & 3;
        float acc[RT][NT][4];
#pragma unroll
        for (int r = 0; r < RT; ++r)
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                for (int i = 0; i < 4; ++i) acc[r][nt][i] = 0.f;
        for (int s = 0; s < nst; ++s) {
            const int slot = s % ST;
            mbar_wait(&full[slot], (s / ST) & 1);
            const int nch = min(SC, a.kcn - s * SC);
            if (warp < nch) {
                const uint8_t* stp = smem + slot * C::STAGE;
                uint4 wq[RT][3];
#pragma unroll
                for (int r = 0; r < RT; ++r)
#pragma unroll
                    for (int q = 0; q < 3; ++q)
                        wq[r][q] = *reinterpret_cast<const uint4*>(stp + (r * SC + warp) * BLK + q * 512 + lane * 16);
                uint4 xq[NT][2];
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    const int tok = nt * 8 + g;
                    if (tok < nvalid) {
                        const uint4* xp = reinterpret_cast<const uint4*>(stp + C::WB + tok * XLD + warp * 128 + 32 * t4);
                        xq[nt][0] = xp[0];
                        xq[nt][1] = xp[1];
                    } else {
                        xq[nt][0] = make_uint4(0, 0, 0, 0);
                        xq[nt][1] = make_uint4(0, 0, 0, 0);
                    }
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
#pragma unroll
                    for (int r = 0; r < RT; ++r) {
                        const uint32_t La = word_of(wq[r][j >> 1], (j & 1) * 2);
                        const uint32_t Lb = word_of(wq[r][j >> 1], (j & 1) * 2 + 1);
                        const uint32_t Nw = word_of(wq[r][2], j);
                        const uint32_t Ha = (Nw & 0x87878787u) | 0x38383838u;
                        const uint32_t Hb = (__funnelshift_r(Nw, Nw, 4) & 0x87878787u) | 0x38383838u;
                        const uint32_t a0 = __byte_perm(La, Ha, 0x5140), a1 = __byte_perm(La, Ha, 0x7362);
                        const uint32_t a2 = __byte_perm(Lb, Hb, 0x5140), a3 = __byte_perm(Lb, Hb, 0x7362);
#pragma unroll
                        for (int nt = 0; nt < NT; ++nt)
                            mma_bf16(acc[r][nt], a0, a1, a2, a3, word_of(xq[nt][j >> 1], (j & 1) * 2),
                                     word_of(xq[nt][j >> 1], (j & 1) * 2 + 1));
                    }
                }
            }
            __syncwarp();
            if (lane == 0) mbar_arrive(&empty[slot]);
        }
#pragma unroll
        for (int r = 0; r < RT; ++r)
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const int tok = nt * 8 + 2 * t4;
                sred[warp][r * 16 + g][tok] = acc[r][nt][0];
                sred[warp][r * 16 + g][tok + 1] = acc[r][nt][1];
                sred[warp][r * 16 + g + 8][tok] = acc[r][nt][2];
                sred[warp][r * 16 + g + 8][tok + 1] = acc[r][nt][3];
            }
    }
    __syncthreads();
    for (int i = tid; i < ROWS * nvalid; i += NTHR) {
        const int t = i / ROWS, row = i - t * ROWS;
        float s = 0.f;
#pragma unroll
        for (int w = 0; w < NCW; ++w) s += sred[w][row][t];
        a.out[(size_t)t * a.nout + rt0 * 16 + row] = __float2bfloat16(fmaf(s, ssc[row], scorr[row][t]));
    }
}

#define ZT_CFG_LIST(X) \
    X(0, 1, 1, 4) X(1, 1, 1, 6) X(2, 1, 2, 3) X(3, 2, 1, 4) X(4, 2, 2, 3) X(5, 1, 1, 8)

template <int NT, int RT, int ST>
static void launch(const Args& a, int grid, cudaStream_t stream) {
    using C = Cfg<NT, RT, ST>;
    auto kern = zt_kernel<NT, RT, ST>;
    static bool set = false;
    if (!set) {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
        set = true;
    }
    kern<<<grid, NTHR, C::SMEM, stream>>>(a);
}

template <int NT, int RT, int ST>
static int64_t info() {
    using C = Cfg<NT, RT, ST>;
    auto kern = zt_kernel<NT, RT, ST>;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
    int bps = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kern, NTHR, C::SMEM);
    return bps;
}

}  // namespace zt

int64_t zt_bps(int64_t cfg) {
    switch (cfg) {
#define X(id, nt, rt, st) case id: return zt::info<nt, rt, st>();
        ZT_CFG_LIST(X)
#undef X
    }
    return 0;
}

void zt_gemm(torch::Tensor x, torch::Tensor wp, torch::Tensor rscale, torch::Tensor etab, int64_t emax,
             torch::Tensor out, int64_t m, int64_t k, int64_t nrt, int64_t nout, int64_t cfg) {
    int nt = 1, rt = 1;
    switch (cfg) {
#define X(id, nt_, rt_, st) case id: nt = nt_; rt = rt_; break;
        ZT_CFG_LIST(X)
#undef X
        default: TORCH_CHECK(false, "zt: bad cfg ", cfg);
    }
    TORCH_CHECK(nrt % rt == 0 && k % 64 == 0 && m <= 8 * nt, "zt: shape");
    zt::Args a;
    a.wp = wp.data_ptr<uint8_t>();
    a.rscale = rscale.data_ptr<float>();
    a.etab = reinterpret_cast<const int4*>(etab.data_ptr<int>());
    a.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    a.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    a.m = (int)m;
    a.k = (int)k;
    a.kcn = (int)(k / 64);
    a.nout = (int)nout;
    a.emax = (int)emax;
    cudaStream_t st_ = at::cuda::getCurrentCUDAStream();
    const int grid = (int)(nrt / rt);
    switch (cfg) {
#define X(id, nt_, rt_, st) case id: zt::launch<nt_, rt_, st>(a, grid, st_); break;
        ZT_CFG_LIST(X)
#undef X
    }
}
