
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define DEVI __device__ __forceinline__

namespace zg {

constexpr int NCW = 8;                   // compute warps
constexpr int NTHR = 32 * (NCW + 1);     // + one escape warp
constexpr int BLK = 1536;

struct Args {
    const uint8_t* wp;          // [rts][kcn][BLK]
    const float* rscale;        // [rows] 2^-s per matrix row
    const int4* etab;           // [rts][emax]: {count}, then {k, row in tile, bits(w), bits(-2^-15 * 2^-s)}
    const __nv_bfloat16* x;     // [m][k]
    __nv_bfloat16* out;         // [m][nout]
    int m, k, kcn, nout, emax;
};

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
DEVI uint4 ld_stream(const uint8_t* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

template <int NT, int RT, int U>
__global__ void __launch_bounds__(NTHR) zg_kernel(const __grid_constant__ Args a) {
    constexpr int NTOK = NT * 8, ROWS = RT * 16;
    __shared__ float sred[NCW][ROWS][NTOK + 1];
    __shared__ float scorr[ROWS][NTOK];
    __shared__ uint32_t s_xv[32 * NTOK];
    __shared__ float ssc[ROWS];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int rt0 = blockIdx.x * RT;
    const int nvalid = min(NTOK, a.m);
    for (int i = tid; i < ROWS * NTOK; i += NTHR) (&scorr[0][0])[i] = 0.f;
    __syncthreads();

    if (warp == NCW) {
        // ================= escape warp =================
        for (int i = lane; i < ROWS; i += 32) ssc[i] = a.rscale[rt0 * 16 + i];   // ROWS = 64 at RT = 4
        escape_warp<NTOK, RT>(a.etab, a.emax, rt0, 0, a.k, a.x, a.k, nvalid, scorr, s_xv, lane);
    } else {
        // ================= compute warps =================
        const int g = lane >> 2, t4 = lane & 3;
        float acc[RT][NT][4];
#pragma unroll
        for (int r = 0; r < RT; ++r)
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                for (int i = 0; i < 4; ++i) acc[r][nt][i] = 0.f;
        const uint8_t* wb = a.wp + (size_t)rt0 * a.kcn * BLK + lane * 16;
        for (int kb = warp; kb < a.kcn; kb += NCW * U) {
            uint4 wq[U][RT][3];
            uint4 xq[U][NT][2];
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const int kc = kb + u * NCW;
                if (kc < a.kcn) {
#pragma unroll
                    for (int r = 0; r < RT; ++r)
#pragma unroll
                        for (int q = 0; q < 3; ++q)
                            wq[u][r][q] = ld_stream(wb + ((size_t)r * a.kcn + kc) * BLK + q * 512);
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt) {
                        const int tok = nt * 8 + g;
                        if (tok < nvalid) {
                            const uint4* xp = reinterpret_cast<const uint4*>(a.x + (size_t)tok * a.k + kc * 64 + 16 * t4);
                            xq[u][nt][0] = __ldg(xp);
                            xq[u][nt][1] = __ldg(xp + 1);
                        } else {
                            xq[u][nt][0] = make_uint4(0, 0, 0, 0);
                            xq[u][nt][1] = make_uint4(0, 0, 0, 0);
                        }
                    }
                }
            }
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const int kc = kb + u * NCW;
                if (kc < a.kcn) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
#pragma unroll
                        for (int r = 0; r < RT; ++r) {
                            const uint32_t La = word_of(wq[u][r][j >> 1], (j & 1) * 2);
                            const uint32_t Lb = word_of(wq[u][r][j >> 1], (j & 1) * 2 + 1);
                            const uint32_t Nw = word_of(wq[u][r][2], j);
                            const uint32_t Ha = (Nw & 0x87878787u) | 0x38383838u;
                            const uint32_t Hb = (__funnelshift_r(Nw, Nw, 4) & 0x87878787u) | 0x38383838u;
                            const uint32_t a0 = __byte_perm(La, Ha, 0x5140), a1 = __byte_perm(La, Ha, 0x7362);
                            const uint32_t a2 = __byte_perm(Lb, Hb, 0x5140), a3 = __byte_perm(Lb, Hb, 0x7362);
#pragma unroll
                            for (int nt = 0; nt < NT; ++nt)
                                mma_bf16(acc[r][nt], a0, a1, a2, a3, word_of(xq[u][nt][j >> 1], (j & 1) * 2),
                                         word_of(xq[u][nt][j >> 1], (j & 1) * 2 + 1));
                        }
                    }
                }
            }
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

#define ZG_CFG_LIST(X) \
    X(0, 1, 1, 4) X(1, 1, 2, 2) X(2, 2, 1, 4) X(3, 2, 2, 2) X(4, 4, 1, 2) X(5, 4, 2, 1) X(6, 1, 1, 2) \
    X(7, 2, 4, 2) X(8, 4, 2, 2) X(10, 2, 4, 1)

}  // namespace zg

void zg_gemm(torch::Tensor x, torch::Tensor wp, torch::Tensor rscale, torch::Tensor etab, int64_t emax,
             torch::Tensor out, int64_t m, int64_t k, int64_t nrt, int64_t nout, int64_t cfg) {
    int nt = 1, rt = 1;
    switch (cfg) {
#define X(id, nt_, rt_, u) case id: nt = nt_; rt = rt_; break;
        ZG_CFG_LIST(X)
#undef X
        default: TORCH_CHECK(false, "zg: bad cfg ", cfg);
    }
    TORCH_CHECK(nrt % rt == 0 && k % 64 == 0 && m <= 8 * nt, "zg: shape");
    zg::Args a;
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
#define X(id, nt_, rt_, u) case id: zg::zg_kernel<nt_, rt_, u><<<grid, zg::NTHR, 0, st_>>>(a); break;
        ZG_CFG_LIST(X)
#undef X
    }
}
