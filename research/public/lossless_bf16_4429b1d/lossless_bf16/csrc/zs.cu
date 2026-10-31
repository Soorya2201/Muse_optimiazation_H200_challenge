
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cudaTypedefs.h>
#include <stdint.h>

namespace zs {

#define ZS_DEVI __device__ __forceinline__

constexpr int BLK = 1536;
constexpr int FIN_THR = 128;

struct Args {
    const uint8_t* wp;          // blocked [nrb][kcn][RT][BLK]: block b at b * BLK
    float4* ws;                 // partials [nrb][jmax][RT][NT][32] float4
    int kcn, U, G, jmax;
    unsigned long long* dbg;    // optional [G][4] %globaltimer ns: start, first stage, loop end, stores done
};

struct FinArgs {
    const float4* ws;
    const float* rscale;        // [n] 2^-s per row
    const int4* erow;           // [n][emr] {k or -1, bits(w), bits(nv), 0}
    const __nv_bfloat16* x;     // [m][k]
    __nv_bfloat16* out;         // [m][n]
    int m, k, n, nt, rt, kcn, U, G, jmax, emr, tpp, P;
    unsigned long long* dbg;    // optional [FG][4]: entry, escapes done, post-wait, end
};

ZS_DEVI uint32_t smem_addr(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
ZS_DEVI void mbar_init(uint64_t* b, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(smem_addr(b)), "r"(count) : "memory");
}
ZS_DEVI void mbar_arrive(uint64_t* b) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(smem_addr(b)) : "memory");
}
ZS_DEVI void mbar_arrive_tx(uint64_t* b, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(smem_addr(b)), "r"(bytes) : "memory");
}
ZS_DEVI void mbar_wait(uint64_t* b, int parity) {
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@!P1 bra WAIT_%=;\n"
        "}\n" :: "r"(smem_addr(b)), "r"(parity) : "memory");
}
ZS_DEVI void bulk_g2s_hint(uint32_t dst, const void* src, uint32_t bytes, uint64_t* bar, uint64_t pol) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(smem_addr(bar)), "l"(pol) : "memory");
}
ZS_DEVI void tma_load_2d(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint64_t* bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
                 :: "r"(dst), "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(smem_addr(bar)) : "memory");
}
ZS_DEVI void prefetch_tmap(const CUtensorMap* map) {
    asm volatile("prefetch.tensormap [%0];\n" :: "l"(reinterpret_cast<uint64_t>(map)) : "memory");
}
ZS_DEVI uint64_t evict_first_policy() {
    uint64_t p;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(p));
    return p;
}
ZS_DEVI void mma_bf16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
ZS_DEVI uint32_t word_of(const uint4& q, int i) { return i == 0 ? q.x : (i == 1 ? q.y : (i == 2 ? q.z : q.w)); }
ZS_DEVI unsigned long long gtimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
    return t;
}
ZS_DEVI void pdl_launch_dependents() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }
ZS_DEVI void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
ZS_DEVI void named_bar(int id, int n) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
// L2-only 16-byte load as volatile asm so every issued load stays in flight (sgx_fin_kernel's lesson)
ZS_DEVI float4 ld_cg_v4(const float4* p) {
    float4 v;
    asm volatile("ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];\n" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
    return v;
}

// the CTA whose block range [c U / G, (c+1) U / G) holds block b
ZS_DEVI int owner(int b, int G, int U) { return (int)((((long long)(b + 1)) * G + U - 1) / U) - 1; }

template <int NT, int NCW, int RW, int ST>
struct Cfg {
    static constexpr int RT = NCW * RW;
    static constexpr int NTOK = NT * 8;
    static constexpr int WB = RT * BLK;          // a multiple of 1024 (RT even)
    static constexpr int XB = NTOK * 128;        // x tile [NTOK][64] bf16, 128B-swizzled
    static constexpr int STAGE = WB + XB;
    static constexpr int SMEM = ST * STAGE + 1024;   // + slack for the 1024-B alignment
    static constexpr int NTHR = 32 * (NCW + 1);
    static constexpr int SLOT = RT * NT * 32;    // float4s per partial slot
    static_assert(WB % 1024 == 0 && XB % 1024 == 0, "stage alignment (128B swizzle needs 1024-B x tiles)");
};

template <int NT, int NCW, int RW, int ST>
__global__ void __launch_bounds__(32 * (NCW + 1), 1) zs_kernel(const __grid_constant__ Args a,
                                                               const __grid_constant__ CUtensorMap xmap) {
    using C = Cfg<NT, NCW, RW, ST>;
    constexpr int RT = C::RT;
    extern __shared__ __align__(128) uint8_t smem[];
    __shared__ __align__(8) uint64_t full[ST], empty[ST];
    pdl_launch_dependents();               // the finalize grid may launch now; it waits for this grid's completion
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    unsigned long long* dbg = a.dbg != nullptr ? a.dbg + (size_t)blockIdx.x * 4 : nullptr;
    if (dbg != nullptr && tid == 0) dbg[0] = gtimer();
    const int c = (int)blockIdx.x;
    const int b0 = (int)(((long long)c * a.U) / a.G), b1 = (int)(((long long)(c + 1) * a.U) / a.G);
    const uint32_t sbase = (smem_addr(smem) + 1023u) & ~1023u;
    uint8_t* const sgen = smem + (sbase - smem_addr(smem));
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
        // ================= producer: one lane, one bulk copy (contiguous blocks) + one x TMA per stage =================
        if (lane == 0) {
            prefetch_tmap(&xmap);
            const uint64_t pol = evict_first_policy();
            int b = b0, s = 0;
            while (b < b1) {
                const int unit = b / RT;
                const int rlo = b - unit * RT, rhi = min(RT, b1 - unit * RT);
                const int kc = unit % a.kcn;
                const int slot = s % ST;
                if (s >= ST) mbar_wait(&empty[slot], ((s / ST) - 1) & 1);
                const uint32_t st = sbase + slot * C::STAGE;
                const uint32_t wbytes = (uint32_t)(rhi - rlo) * BLK;
                mbar_arrive_tx(&full[slot], wbytes + C::XB);
                bulk_g2s_hint(st + rlo * BLK, a.wp + (size_t)b * BLK, wbytes, &full[slot], pol);
                tma_load_2d(st + C::WB, &xmap, kc * 64, 0, &full[slot]);   // tokens >= m read as 0
                b = unit * RT + rhi;
                ++s;
            }
        }
        return;
    }

    // ================= consumer warps: row tiles warp * RW .. + RW - 1 of every stage =================
    const int g = lane >> 2, t4 = lane & 3;
    float acc[RW][NT][4];
#pragma unroll
    for (int r = 0; r < RW; ++r)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[r][nt][i] = 0.f;
    int b = b0, s = 0;
    while (b < b1) {
        const int unit = b / RT;
        const int rlo = b - unit * RT, rhi = min(RT, b1 - unit * RT);
        const int rb = unit / a.kcn, kc = unit - rb * a.kcn;
        const int slot = s % ST;
        mbar_wait(&full[slot], (s / ST) & 1);
        if (dbg != nullptr && s == 0 && tid == 0) dbg[1] = gtimer();
        const uint8_t* stp = sgen + slot * C::STAGE;
        uint4 xq[NT][2];
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            // 128-B line `tok` of the swizzled x tile; its 16-B pieces 2 t4, 2 t4 + 1 sit at piece ^ (tok & 7)
            const int tok = nt * 8 + g;
            const uint8_t* xl = stp + C::WB + tok * 128;
            xq[nt][0] = *reinterpret_cast<const uint4*>(xl + (((2 * t4) ^ (tok & 7)) << 4));
            xq[nt][1] = *reinterpret_cast<const uint4*>(xl + (((2 * t4 + 1) ^ (tok & 7)) << 4));
        }
#pragma unroll
        for (int r = 0; r < RW; ++r) {
            const int R = warp * RW + r;
            if (R < rlo || R >= rhi) continue;           // warp-uniform: this row tile has no block in the stage
            uint4 wq[3];
#pragma unroll
            for (int q = 0; q < 3; ++q)
                wq[q] = *reinterpret_cast<const uint4*>(stp + R * BLK + q * 512 + lane * 16);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t La = word_of(wq[j >> 1], (j & 1) * 2);
                const uint32_t Lb = word_of(wq[j >> 1], (j & 1) * 2 + 1);
                const uint32_t Nw = word_of(wq[2], j);
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
        __syncwarp();
        if (lane == 0) mbar_arrive(&empty[slot]);
        const bool seg_end = (kc == a.kcn - 1 && rhi == RT) || (unit * RT + rhi == b1);
        if (seg_end) {
            const int j = c - owner(rb * a.kcn * RT, a.G, a.U);
            float4* dst = a.ws + ((size_t)rb * a.jmax + j) * C::SLOT + lane;
#pragma unroll
            for (int r = 0; r < RW; ++r)
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    __stcg(dst + ((warp * RW + r) * NT + nt) * 32,
                           make_float4(acc[r][nt][0], acc[r][nt][1], acc[r][nt][2], acc[r][nt][3]));
#pragma unroll
                    for (int i = 0; i < 4; ++i) acc[r][nt][i] = 0.f;
                }
        }
        b = unit * RT + rhi;
        ++s;
    }
    if (dbg != nullptr) {
        if (tid == 0) dbg[2] = gtimer();
        named_bar(1, 32 * NCW);
        if (tid == 0) dbg[3] = gtimer();
    }
}

template <int NPT>
__global__ void __launch_bounds__(FIN_THR) zs_fin_kernel(const __grid_constant__ FinArgs f) {
    const int gt = (int)blockIdx.x * FIN_THR + (int)threadIdx.x;
    const int pos = gt / f.tpp, sub = gt - pos * f.tpp;
    const bool valid = pos < f.P;
    unsigned long long* dbg = f.dbg != nullptr ? f.dbg + (size_t)blockIdx.x * 4 : nullptr;
    if (dbg != nullptr && threadIdx.x == 0) dbg[0] = gtimer();
    const int lane = pos & 31, q = pos >> 5;
    const int ntt = q % f.nt, rtile = q / f.nt;
    const int g = lane >> 2, t4 = lane & 3;
    const int row0 = rtile * 16 + g, row1 = row0 + 8;
    const int tok0 = ntt * 8 + 2 * t4, tok1 = tok0 + 1;
    const bool v0 = tok0 < f.m, v1 = tok1 < f.m;
    // escape terms summed apart (zr's arithmetic): n.. = sum x nv (cancels the decoded placeholder exactly),
    // w.. = sum x w, added last, so one-hot outputs are exact even for weights far below the 16-binade window
    float sc0 = 0.f, sc1 = 0.f;
    float n00 = 0.f, n01 = 0.f, n10 = 0.f, n11 = 0.f, w00 = 0.f, w01 = 0.f, w10 = 0.f, w11 = 0.f;
    // ---- before the wait (overlaps the main grid): scales and escape corrections of rows row0 / row1 ----
    if (valid) {
        sc0 = __ldg(f.rscale + row0);
        sc1 = __ldg(f.rscale + row1);
        const int4* e0p = f.erow + (size_t)row0 * f.emr;
        const int4* e1p = f.erow + (size_t)row1 * f.emr;
        for (int e0 = 4 * sub; e0 < f.emr; e0 += 4 * f.tpp) {
            int4 ra[4], rb[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const bool in = e0 + i < f.emr;
                ra[i] = in ? __ldg(e0p + e0 + i) : make_int4(-1, 0, 0, 0);
                rb[i] = in ? __ldg(e1p + e0 + i) : make_int4(-1, 0, 0, 0);
            }
            float xa0[4], xa1[4], xb0[4], xb1[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                xa0[i] = (ra[i].x >= 0 && v0) ? __bfloat162float(__ldg(f.x + (size_t)tok0 * f.k + ra[i].x)) : 0.f;
                xa1[i] = (ra[i].x >= 0 && v1) ? __bfloat162float(__ldg(f.x + (size_t)tok1 * f.k + ra[i].x)) : 0.f;
                xb0[i] = (rb[i].x >= 0 && v0) ? __bfloat162float(__ldg(f.x + (size_t)tok0 * f.k + rb[i].x)) : 0.f;
                xb1[i] = (rb[i].x >= 0 && v1) ? __bfloat162float(__ldg(f.x + (size_t)tok1 * f.k + rb[i].x)) : 0.f;
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (ra[i].x >= 0) {
                    const float w = __int_as_float(ra[i].y), nv = __int_as_float(ra[i].z);
                    n00 += xa0[i] * nv;
                    w00 += xa0[i] * w;
                    n01 += xa1[i] * nv;
                    w01 += xa1[i] * w;
                }
                if (rb[i].x >= 0) {
                    const float w = __int_as_float(rb[i].y), nv = __int_as_float(rb[i].z);
                    n10 += xb0[i] * nv;
                    w10 += xb0[i] * w;
                    n11 += xb1[i] * nv;
                    w11 += xb1[i] * w;
                }
            }
        }
    }
    if (dbg != nullptr) {
        __syncthreads();
        if (threadIdx.x == 0) dbg[1] = gtimer();
    }
    pdl_wait();
    if (dbg != nullptr && threadIdx.x == 0) dbg[2] = gtimer();
    // ---- after the wait: sum this position's partials (splits sub, sub + tpp, ...; clamped, all in flight) ----
    float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
    if (valid) {
        const int RT = f.rt;
        const int rbk = rtile / RT, wr = rtile - rbk * RT;
        const int blk0 = rbk * f.kcn * RT;
        const int nsp = owner(blk0 + f.kcn * RT - 1, f.G, f.U) - owner(blk0, f.G, f.U) + 1;
        const int slot = RT * f.nt * 32;
        const float4* base = f.ws + (size_t)rbk * f.jmax * slot + (wr * f.nt + ntt) * 32 + lane;
        float4 v[NPT];
#pragma unroll
        for (int i = 0; i < NPT; ++i) v[i] = ld_cg_v4(base + (size_t)min(sub + i * f.tpp, nsp - 1) * slot);
#pragma unroll
        for (int i = 0; i < NPT; ++i) {
            if (sub + i * f.tpp < nsp) {
                s.x += v[i].x;
                s.y += v[i].y;
                s.z += v[i].z;
                s.w += v[i].w;
            }
        }
    }
    for (int o = 1; o < f.tpp; o <<= 1) {
        s.x += __shfl_xor_sync(0xffffffffu, s.x, o);
        s.y += __shfl_xor_sync(0xffffffffu, s.y, o);
        s.z += __shfl_xor_sync(0xffffffffu, s.z, o);
        s.w += __shfl_xor_sync(0xffffffffu, s.w, o);
        n00 += __shfl_xor_sync(0xffffffffu, n00, o);
        n01 += __shfl_xor_sync(0xffffffffu, n01, o);
        n10 += __shfl_xor_sync(0xffffffffu, n10, o);
        n11 += __shfl_xor_sync(0xffffffffu, n11, o);
        w00 += __shfl_xor_sync(0xffffffffu, w00, o);
        w01 += __shfl_xor_sync(0xffffffffu, w01, o);
        w10 += __shfl_xor_sync(0xffffffffu, w10, o);
        w11 += __shfl_xor_sync(0xffffffffu, w11, o);
    }
    if (valid && sub == 0) {
        // fragment map: (x, y, z, w) = (row0, tok0), (row0, tok1), (row1, tok0), (row1, tok1)
        if (v0) {
            f.out[(size_t)tok0 * f.n + row0] = __float2bfloat16(fmaf(s.x, sc0, n00) + w00);
            f.out[(size_t)tok0 * f.n + row1] = __float2bfloat16(fmaf(s.z, sc1, n10) + w10);
        }
        if (v1) {
            f.out[(size_t)tok1 * f.n + row0] = __float2bfloat16(fmaf(s.y, sc0, n01) + w01);
            f.out[(size_t)tok1 * f.n + row1] = __float2bfloat16(fmaf(s.w, sc1, n11) + w11);
        }
    }
    if (dbg != nullptr) {
        __syncthreads();
        if (threadIdx.x == 0) dbg[3] = gtimer();
    }
}

// cfg id, NT (token tile / 8), NCW (consumer warps), RW (row tiles per warp), ST (ring stages, <= 216 KB).
// x is read from shared memory once per warp and stage, so its LDS bytes per weight byte are m / (12 RW): at m = 32
// RW = 1 (cfg 2) puts ~100 B / clk on shared memory with the TMA writes, RW = 4 (cfgs 6, 7: 4 warps x 4 tiles) ~45.
#define ZS_CFG_LIST(X) \
    X(0, 1, 8, 1, 16) X(1, 2, 8, 1, 15) X(2, 4, 8, 1, 13) \
    X(3, 1, 8, 2, 8) X(4, 2, 8, 2, 8) X(5, 4, 8, 2, 7) \
    X(6, 4, 4, 4, 7) X(7, 2, 4, 4, 8)

// 2D tensor map over x [m][k] bf16: box 64 columns x ntok rows, 128B swizzle, rows past m read as 0
static CUtensorMap make_xmap(const void* ptr, int k, int m, int ntok) {
    static PFN_cuTensorMapEncodeTiled_v12000 encode = nullptr;
    if (encode == nullptr) {
        cudaDriverEntryPointQueryResult q;
        cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", reinterpret_cast<void**>(&encode), 12000,
                                         cudaEnableDefault, &q);
    }
    TORCH_CHECK(encode != nullptr, "zs: cuTensorMapEncodeTiled unavailable");
    CUtensorMap mp;
    cuuint64_t dims[2] = {(cuuint64_t)k, (cuuint64_t)m};
    cuuint64_t strides[1] = {(cuuint64_t)k * 2};
    cuuint32_t box[2] = {64, (cuuint32_t)ntok};
    cuuint32_t es[2] = {1, 1};
    const CUresult r = encode(&mp, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides, box, es,
                              CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                              CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "zs: tensor map encode failed (", (int)r, ") k=", k, " m=", m, " ntok=", ntok);
    return mp;
}

template <int NT, int NCW, int RW, int ST>
static void launch_main(const Args& a, const void* xp, int k, int m, cudaStream_t stream) {
    using C = Cfg<NT, NCW, RW, ST>;
    auto kern = zs_kernel<NT, NCW, RW, ST>;
    static bool set = false;
    if (!set) {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
        set = true;
    }
    const CUtensorMap xmap = make_xmap(xp, k, m, C::NTOK);
    kern<<<a.G, C::NTHR, C::SMEM, stream>>>(a, xmap);
}

template <typename K>
static void kattr(K kern, int nthr, int smem_dyn, int64_t* p) {
    cudaFuncAttributes fa;
    if (cudaFuncGetAttributes(&fa, reinterpret_cast<const void*>(kern)) == cudaSuccess) {
        p[0] = fa.numRegs;
        p[1] = (int64_t)fa.localSizeBytes;
        p[3] = (int64_t)fa.sharedSizeBytes + smem_dyn;
    } else {
        cudaGetLastError();
        p[0] = -1;
        p[1] = -1;
        p[3] = -1;
    }
    if (smem_dyn > 0) cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_dyn);
    int bps = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kern, nthr, smem_dyn) != cudaSuccess) {
        cudaGetLastError();
        bps = -1;
    }
    p[2] = bps;
    p[4] = nthr;
}

}  // namespace zs

// [ncfg + 2][6]: per main cfg {id, regs, local bytes, CTAs/SM, smem bytes (static + dynamic), threads}, then the
// finalize kernels NPT = 4, 8 {npt, regs, local bytes, CTAs/SM alone, static smem, threads}
torch::Tensor zs_info() {
    int ncfg = 0;
#define X(id, nt, ncw, rw, st) ++ncfg;
    ZS_CFG_LIST(X)
#undef X
    auto t = torch::zeros({ncfg + 2, 6}, torch::dtype(torch::kInt64));
    int64_t* p = t.data_ptr<int64_t>();
    int row = 0;
#define X(id, nt, ncw, rw, st)                                                                              \
    p[row * 6] = id;                                                                                       \
    zs::kattr(zs::zs_kernel<nt, ncw, rw, st>, zs::Cfg<nt, ncw, rw, st>::NTHR, zs::Cfg<nt, ncw, rw, st>::SMEM, \
              p + row * 6 + 1);                                                                            \
    ++row;
    ZS_CFG_LIST(X)
#undef X
    p[row * 6] = 4;
    zs::kattr(zs::zs_fin_kernel<4>, zs::FIN_THR, 0, p + row * 6 + 1);
    ++row;
    p[row * 6] = 8;
    zs::kattr(zs::zs_fin_kernel<8>, zs::FIN_THR, 0, p + row * 6 + 1);
    return t;
}

// out[m][n] = x[m][k] . W^T on the blocked zk weights (zs_block for RT = NCW * RW). The plan (G, jmax, tpp, npt:
// zs_plan in Python) is re-checked here. ws: >= nrb * jmax * RT * NT * 128 floats, never needs zeroing. pdl: launch
// the finalize with programmatic stream serialization. dbg: optional int64 [4 G + 4 FG]. Returns FG.
int64_t zs_gemm(torch::Tensor x, torch::Tensor wp, torch::Tensor rscale, torch::Tensor erow, int64_t emr,
                torch::Tensor out, torch::Tensor ws, int64_t m, int64_t k, int64_t n, int64_t cfg, int64_t G,
                int64_t jmax, int64_t tpp, int64_t npt, int64_t pdl, c10::optional<torch::Tensor> dbg) {
    int nt = 0, ncw = 0, rw = 0;
    switch (cfg) {
#define X(id, nt_, ncw_, rw_, st) case id: nt = nt_; ncw = ncw_; rw = rw_; break;
        ZS_CFG_LIST(X)
#undef X
        default: TORCH_CHECK(false, "zs: bad cfg ", cfg);
    }
    const int RT = ncw * rw;
    TORCH_CHECK(m >= 1 && m <= 8 * nt && k % 64 == 0 && n % (16 * RT) == 0, "zs: shape m=", m, " k=", k, " n=", n,
                " cfg ", cfg);
    TORCH_CHECK(x.is_contiguous() && x.numel() == m * k && x.scalar_type() == at::kBFloat16, "zs: x");
    TORCH_CHECK(out.is_contiguous() && out.numel() == m * n && out.scalar_type() == at::kBFloat16, "zs: out");
    const int nrt = (int)(n / 16), kcn = (int)(k / 64), nrb = nrt / RT;
    const int U = nrt * kcn;
    TORCH_CHECK(G >= 1 && G <= U, "zs: G ", G, " for ", U, " blocks");
    int jm = 0;
    for (int rb = 0; rb < nrb; ++rb) {
        const long long f0 = (long long)rb * kcn * RT, f1 = f0 + (long long)kcn * RT - 1;
        const int c0 = (int)(((f0 + 1) * G + U - 1) / U) - 1, c1 = (int)(((f1 + 1) * G + U - 1) / U) - 1;
        jm = std::max(jm, c1 - c0 + 1);
    }
    TORCH_CHECK(jmax == jm, "zs: jmax ", jmax, " != ", jm);
    TORCH_CHECK(tpp >= 1 && tpp <= 32 && (tpp & (tpp - 1)) == 0 && (npt == 4 || npt == 8) && tpp * npt >= jmax,
                "zs: tpp ", tpp, " npt ", npt, " jmax ", jmax);
    TORCH_CHECK(wp.numel() >= (int64_t)U * zs::BLK && rscale.numel() >= n && emr >= 1 && erow.numel() >= n * emr * 4,
                "zs: weight tensors");
    const int64_t slot = (int64_t)RT * nt * 32;
    TORCH_CHECK(ws.scalar_type() == at::kFloat && ws.numel() >= (int64_t)nrb * jmax * slot * 4, "zs: ws too small");
    const int P = nrt * nt * 32;
    const int fg = (int)(((int64_t)P * tpp + zs::FIN_THR - 1) / zs::FIN_THR);
    unsigned long long* dg = nullptr;
    if (dbg.has_value()) {
        TORCH_CHECK(dbg.value().numel() >= 4 * (G + fg), "zs: dbg too small");
        dg = reinterpret_cast<unsigned long long*>(dbg.value().data_ptr<int64_t>());
    }
    zs::Args a;
    a.wp = wp.data_ptr<uint8_t>();
    a.ws = reinterpret_cast<float4*>(ws.data_ptr<float>());
    a.kcn = kcn;
    a.U = U;
    a.G = (int)G;
    a.jmax = (int)jmax;
    a.dbg = dg;
    cudaStream_t st_ = at::cuda::getCurrentCUDAStream();
    const void* xp = x.data_ptr();
    switch (cfg) {
#define X(id, nt_, ncw_, rw_, st) case id: zs::launch_main<nt_, ncw_, rw_, st>(a, xp, (int)k, (int)m, st_); break;
        ZS_CFG_LIST(X)
#undef X
    }
    zs::FinArgs f;
    f.ws = a.ws;
    f.rscale = rscale.data_ptr<float>();
    f.erow = reinterpret_cast<const int4*>(erow.data_ptr<int>());
    f.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    f.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    f.m = (int)m;
    f.k = (int)k;
    f.n = (int)n;
    f.nt = nt;
    f.rt = RT;
    f.kcn = kcn;
    f.U = U;
    f.G = (int)G;
    f.jmax = (int)jmax;
    f.emr = (int)emr;
    f.tpp = (int)tpp;
    f.P = P;
    f.dbg = dg != nullptr ? dg + 4 * G : nullptr;
    cudaLaunchConfig_t lc = {};
    lc.gridDim = dim3(fg);
    lc.blockDim = dim3(zs::FIN_THR);
    lc.dynamicSmemBytes = 0;
    lc.stream = st_;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    lc.attrs = attr;
    lc.numAttrs = pdl ? 1 : 0;
    const cudaError_t e = npt == 4 ? cudaLaunchKernelEx(&lc, zs::zs_fin_kernel<4>, f)
                                   : cudaLaunchKernelEx(&lc, zs::zs_fin_kernel<8>, f);
    TORCH_CHECK(e == cudaSuccess, "zs: finalize launch failed: ", cudaGetErrorString(e));
    return fg;
}
