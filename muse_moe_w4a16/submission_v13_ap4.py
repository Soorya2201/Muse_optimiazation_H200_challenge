# okc: problem=muse_moe_w4a16
"""Routed experts (NVFP4 W4A16) for H200: three launches, no host sync.

  gemm1 : per-CTA routing in smem, then (expert, 16 gate rows + 16 up rows) work items,
          4 warps split K, mma.sync bf16 with in-register NVFP4 dequant, SwiGLU epilogue
          -> inter[pair, 4096] bf16 (pair = token*8 + slot).
  gemm2 : same engine over W2 (32 output rows per item), epilogue applies (1 + w),
          writes y' bf16 and atomically accumulates sum(y'^2) per pair.
  final : out[t] = sum_slots g * y'[pair] * rsqrt(mean(y'^2) + eps).

Weights are repacked once in prepare() into a lane-ordered stream: per (expert, item, k64 step)
1152 contiguous bytes = 2 x 512 B of E2M1 codes (one 16-byte vector per lane, codes placed at
bit positions chosen so that a rotate + mask turns them into bf16 bit patterns) + 128 B of the
untouched E4M3 block scales. Still 4 bits per weight + 1 E4M3 byte per 16 weights (lossless).
"""

from typing import TypedDict

import torch
from torch.utils.cpp_extension import load_inline


class Weights(TypedDict):
    w13: torch.Tensor          # uint8 [256, 2I, H/2], gate rows [0, I) then up rows [I, 2I), packed E2M1
    w13_sf: torch.Tensor       # float8_e4m3fn [256, 2I, H/16]
    w13_gs: torch.Tensor       # float32 [256]
    w2: torch.Tensor           # uint8 [256, H, I/2]
    w2_sf: torch.Tensor        # float8_e4m3fn [256, H, I/16]
    w2_gs: torch.Tensor        # float32 [256]
    post_norm_w: torch.Tensor  # bfloat16 [H], the post-expert RMSNorm's weight (applied as 1 + w)


class Config(TypedDict):
    bs: int
    d_hidden: int
    d_expert: int
    n_experts: int
    local_start: int
    local_count: int
    top_k: int
    sharding: str


class Inputs(TypedDict):
    hidden_states: torch.Tensor  # bfloat16 [M, H]
    topk_ids: torch.Tensor       # int32 [M, 8]
    topk_weights: torch.Tensor   # float32 [M, 8]
    config: Config


H = 4096
I_DIM = 4096
LOCAL = 32
TOP_K = 8

CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#define KDIM 4096
#define NWARPS 4
#define SPW 16          // k64 steps per warp (64 steps / 4 warps)
#define STEP_U4 72      // uint4 per (item, k64 step): 32 + 32 code vectors + 8 scale vectors
#define ITEM_U4 (64 * STEP_U4)
#define WP 6            // retained weight prefetch depth
#define AP 4            // test one further activation-prefetch step

__device__ __forceinline__ uint4 ld_stream(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
      : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

__device__ __forceinline__ uint2 ld_act(const __nv_bfloat16* p) {
  uint2 r;
  asm volatile("ld.global.nc.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
  return r;
}

__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t rotl(uint32_t x, int L) { return __funnelshift_l(x, x, L); }

// w (two bf16 = e2m1 * 2^-126) times the low (H=0) or high (H=1) bf16 of s, broadcast.
template <int HI>
__device__ __forceinline__ uint32_t bmul(uint32_t w, uint32_t s) {
  __nv_bfloat162 wv = *reinterpret_cast<__nv_bfloat162*>(&w);
  __nv_bfloat162 sv = *reinterpret_cast<__nv_bfloat162*>(&s);
  __nv_bfloat162 sb = HI ? __high2bfloat162(sv) : __low2bfloat162(sv);
  __nv_bfloat162 r = __hmul2(wv, sb);
  return *reinterpret_cast<uint32_t*>(&r);
}

// Two E4M3 scales (bytes 2*HI, 2*HI+1 of word) -> bf16x2 equal to scale * 2^119.
template <int HI, bool SAFE>
__device__ __forceinline__ uint32_t cvt_scale(uint32_t word) {
  if (!SAFE) {
    uint32_t p = __byte_perm(word, 0u, HI ? 0x4342 : 0x4140);
    return p * 16u + 0x77807780u;
  } else {
    __nv_fp8x2_storage_t two = (__nv_fp8x2_storage_t)((word >> (HI * 16)) & 0xFFFFu);
    __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(two, __NV_E4M3);
    __half2 h = *reinterpret_cast<__half2*>(&hr);
    float2 f = __half22float2(h);
    const float k = 6.64613997892457936e35f;  // 2^119
    __nv_bfloat162 b = __floats2bfloat162_rn(f.x * k, f.y * k);
    return *reinterpret_cast<uint32_t*>(&b);
  }
}

// Decode two packed words (codes of sub-steps 2p, 2p+1) into 8 A-fragment registers.
__device__ __forceinline__ void dequant_pair(uint32_t A, uint32_t B, uint32_t* r0, uint32_t* r1) {
  const uint32_t M = 0x81C081C0u;
  uint32_t V = (A & 0xC180C180u) | (B & ~0xC180C180u);
  r0[0] = rotl(A, 6) & M;
  r0[1] = rotl(A, 2) & M;
  r0[2] = rotl(A, 12) & M;
  r0[3] = rotl(V, 14) & M;
  r1[0] = rotl(B, 4) & M;
  r1[1] = B & M;
  r1[2] = rotl(B, 10) & M;
  r1[3] = rotl(V, 8) & M;
}

template <int MODE, int NT, bool SAFE>
__global__ void __launch_bounds__(128) moe_gemm(
    const __nv_bfloat16* __restrict__ act, const int* __restrict__ ids, int M, int local_start,
    const uint4* __restrict__ wpk, const float* __restrict__ gsc, const float* __restrict__ normw,
    __nv_bfloat16* __restrict__ outp, float* __restrict__ ss) {
  constexpr int TILES = MODE == 1 ? 256 : 128;
  extern __shared__ int smem[];
  float* red = reinterpret_cast<float*>(smem);
  int* s_list = smem + NWARPS * 2 * NT * 4 * 32;
  __shared__ int s_cnt[32], s_off[32], s_fill[32], s_touched[32], s_ntouch;

  constexpr int CH = 8 * NT;
  __shared__ int s_ibase[32], s_nch[32], s_total;
  __shared__ int s_wcnt[NWARPS][32];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int npairs = M * 8;
  reinterpret_cast<int*>(s_wcnt)[tid] = 0;
  if (MODE == 1) {
    for (int i = blockIdx.x * 128 + tid; i < npairs; i += gridDim.x * 128) ss[i] = 0.f;
  }
  // Deterministic routing (identical list order in every CTA): warp w owns pairs [lo, hi).
  const int per = (npairs + NWARPS - 1) / NWARPS;
  const int lo = min(npairs, warp * per), hi = min(npairs, lo + per);
  __syncthreads();
  for (int i = lo + lane; i < hi; i += 32) {
    int e = ids[i] - local_start;
    if ((unsigned)e < 32u) atomicAdd(&s_wcnt[warp][e], 1);
  }
  __syncthreads();
  if (warp == 0) {
    int c = 0;
#pragma unroll
    for (int w = 0; w < NWARPS; w++) c += s_wcnt[w][lane];
    s_cnt[lane] = c;
    const int nch = (c + CH - 1) / CH;
    int incl = c, inci = nch * TILES;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      int v = __shfl_up_sync(0xffffffffu, incl, o);
      int w = __shfl_up_sync(0xffffffffu, inci, o);
      if (lane >= o) { incl += v; inci += w; }
    }
    int base = incl - c;
    s_off[lane] = base;
#pragma unroll
    for (int w = 0; w < NWARPS; w++) { const int t = s_wcnt[w][lane]; s_wcnt[w][lane] = base; base += t; }
    s_nch[lane] = nch;
    s_ibase[lane] = inci - nch * TILES;
    unsigned m = __ballot_sync(0xffffffffu, c > 0);
    if (c > 0) s_touched[__popc(m & ((1u << lane) - 1u))] = lane;
    if (lane == 31) { s_ntouch = __popc(m); s_total = inci; }
  }
  __syncthreads();
  for (int b0 = lo; b0 < hi; b0 += 32) {
    const int i = b0 + lane;
    const int e = i < hi ? ids[i] - local_start : -1;
    const bool loc = (unsigned)e < 32u;
    const unsigned peers = __match_any_sync(0xffffffffu, loc ? e : 64 + lane);
    const int rank = __popc(peers & ((1u << lane) - 1u));
    if (loc) s_list[s_wcnt[warp][e] + rank] = i;
    __syncwarp();
    if (loc && rank == 0) s_wcnt[warp][e] += __popc(peers);
    __syncwarp();
  }
  __syncthreads();

  // Items: (touched expert, tile, token chunk), chunk fastest so CTAs sharing weights run together.
  const int total = s_total;
  const int r = lane >> 2, cq = lane & 3;
  for (int item = blockIdx.x; item < total; item += gridDim.x) {
    int k_ = 0;
    while (k_ + 1 < s_ntouch && s_ibase[s_touched[k_ + 1]] <= item) k_++;
    const int e = s_touched[k_];
    const int rem = item - s_ibase[e];
    const int nch = s_nch[e];
    const int tile = rem / nch;
    const int cnt = s_cnt[e], off = s_off[e];
    const uint4* wb = wpk + ((size_t)e * TILES + tile) * ITEM_U4 + (size_t)warp * SPW * STEP_U4;
    const float g = gsc[e];
    {
      const int c0 = (rem - tile * nch) * CH;
      const int nv = min(cnt - c0, 8 * NT);
      const int ntl = (nv + 7) >> 3;
      const __nv_bfloat16* bp[NT];
      bool bv[NT];
#pragma unroll
      for (int j = 0; j < NT; j++) {
        int idx = c0 + 8 * j + r;
        bv[j] = idx < cnt;
        int p = bv[j] ? s_list[off + idx] : 0;
        int row = MODE == 1 ? (p >> 3) : p;
        bp[j] = act + (size_t)row * KDIM + warp * (SPW * 64) + cq * 4;
      }
      float acc[2][NT][4];
#pragma unroll
      for (int a = 0; a < 2; a++)
#pragma unroll
        for (int j = 0; j < NT; j++)
#pragma unroll
          for (int v = 0; v < 4; v++) acc[a][j][v] = 0.f;

      uint4 pw0[WP], pw1[WP], psc[WP];
      uint2 pa[AP][NT][4];
#pragma unroll
      for (int s = 0; s < WP; s++) {
        pw0[s] = ld_stream(wb + s * STEP_U4 + lane);
        pw1[s] = ld_stream(wb + s * STEP_U4 + 32 + lane);
        psc[s] = ld_stream(wb + s * STEP_U4 + 64 + r);
      }
#pragma unroll
      for (int s = 0; s < AP; s++)
#pragma unroll
        for (int j = 0; j < NT; j++)
#pragma unroll
          for (int q = 0; q < 4; q++)
            pa[s][j][q] = bv[j] ? ld_act(bp[j] + s * 64 + q * 16) : make_uint2(0u, 0u);

#pragma unroll
      for (int ks = 0; ks < SPW; ks++) {
        const int ws = ks % WP, as = ks % AP;
        const uint4 w0 = pw0[ws], w1 = pw1[ws], sc = psc[ws];
        uint2 xb[NT][4];
#pragma unroll
        for (int j = 0; j < NT; j++)
#pragma unroll
          for (int q = 0; q < 4; q++) xb[j][q] = pa[as][j][q];
        if (ks + WP < SPW) {
          pw0[ws] = ld_stream(wb + (ks + WP) * STEP_U4 + lane);
          pw1[ws] = ld_stream(wb + (ks + WP) * STEP_U4 + 32 + lane);
          psc[ws] = ld_stream(wb + (ks + WP) * STEP_U4 + 64 + r);
        }
        if (ks + AP < SPW) {
#pragma unroll
          for (int j = 0; j < NT; j++)
#pragma unroll
            for (int q = 0; q < 4; q++)
              pa[as][j][q] = bv[j] ? ld_act(bp[j] + (ks + AP) * 64 + q * 16) : make_uint2(0u, 0u);
        }
        // scales: [tile][row r / r+8][kb pair]
        uint32_t sc_[2][2][2];
        sc_[0][0][0] = cvt_scale<0, SAFE>(sc.x); sc_[0][0][1] = cvt_scale<1, SAFE>(sc.x);
        sc_[0][1][0] = cvt_scale<0, SAFE>(sc.y); sc_[0][1][1] = cvt_scale<1, SAFE>(sc.y);
        sc_[1][0][0] = cvt_scale<0, SAFE>(sc.z); sc_[1][0][1] = cvt_scale<1, SAFE>(sc.z);
        sc_[1][1][0] = cvt_scale<0, SAFE>(sc.w); sc_[1][1][1] = cvt_scale<1, SAFE>(sc.w);
        uint32_t R[2][4][4];  // [tile][substep][reg]
        dequant_pair(w0.x, w0.y, R[0][0], R[0][1]);
        dequant_pair(w0.z, w0.w, R[0][2], R[0][3]);
        dequant_pair(w1.x, w1.y, R[1][0], R[1][1]);
        dequant_pair(w1.z, w1.w, R[1][2], R[1][3]);
#pragma unroll
        for (int s = 0; s < 4; s++) {
#pragma unroll
          for (int t = 0; t < 2; t++) {
            const uint32_t s_r = sc_[t][0][s >> 1], s_r8 = sc_[t][1][s >> 1];
            uint32_t a0, a1, a2, a3;
            if (s & 1) {
              a0 = bmul<1>(R[t][s][0], s_r); a1 = bmul<1>(R[t][s][1], s_r8);
              a2 = bmul<1>(R[t][s][2], s_r); a3 = bmul<1>(R[t][s][3], s_r8);
            } else {
              a0 = bmul<0>(R[t][s][0], s_r); a1 = bmul<0>(R[t][s][1], s_r8);
              a2 = bmul<0>(R[t][s][2], s_r); a3 = bmul<0>(R[t][s][3], s_r8);
            }
#pragma unroll
            for (int j = 0; j < NT; j++)
              if (j < ntl) mma16816(acc[t][j], a0, a1, a2, a3, xb[j][s].x, xb[j][s].y);
          }
        }
      }

      // cross-warp split-K reduction
      __syncthreads();
#pragma unroll
      for (int t = 0; t < 2; t++)
#pragma unroll
        for (int j = 0; j < NT; j++)
          if (j < ntl)
#pragma unroll
            for (int v = 0; v < 4; v++) red[(((warp * 2 + t) * NT + j) * 4 + v) * 32 + lane] = acc[t][j][v];
      __syncthreads();
      for (int j = warp; j < ntl; j += NWARPS) {
        float h[2][4];
#pragma unroll
        for (int t = 0; t < 2; t++)
#pragma unroll
          for (int v = 0; v < 4; v++) {
            float sum = 0.f;
#pragma unroll
            for (int w = 0; w < NWARPS; w++) sum += red[(((w * 2 + t) * NT + j) * 4 + v) * 32 + lane];
            h[t][v] = sum * g;
          }
        if (MODE == 1) {
#pragma unroll
          for (int v = 0; v < 4; v++) {
            const int idx = c0 + 8 * j + 2 * cq + (v & 1);
            if (idx < cnt) {
              const int p = s_list[off + idx];
              const float gt = h[0][v], ut = h[1][v];
              const float val = gt / (1.f + __expf(-gt)) * ut;
              outp[(size_t)p * KDIM + tile * 16 + (v >> 1) * 8 + r] = __float2bfloat16(val);
            }
          }
        } else {
          float ssp[2] = {0.f, 0.f};
#pragma unroll
          for (int v = 0; v < 4; v++) {
            const int idx = c0 + 8 * j + 2 * cq + (v & 1);
            const int row0 = tile * 32 + (v >> 1) * 8 + r;
            const float y0 = h[0][v] * normw[row0];
            const float y1 = h[1][v] * normw[row0 + 16];
            if (idx < cnt) {
              const int p = s_list[off + idx];
              outp[(size_t)p * KDIM + row0] = __float2bfloat16(y0);
              outp[(size_t)p * KDIM + row0 + 16] = __float2bfloat16(y1);
            }
            ssp[v & 1] += y0 * y0 + y1 * y1;
          }
#pragma unroll
          for (int q = 0; q < 2; q++) {
            ssp[q] += __shfl_xor_sync(0xffffffffu, ssp[q], 4);
            ssp[q] += __shfl_xor_sync(0xffffffffu, ssp[q], 8);
            ssp[q] += __shfl_xor_sync(0xffffffffu, ssp[q], 16);
          }
          if (r == 0) {
#pragma unroll
            for (int q = 0; q < 2; q++) {
              const int idx = c0 + 8 * j + 2 * cq + q;
              if (idx < cnt) atomicAdd(&ss[s_list[off + idx]], ssp[q]);
            }
          }
        }
      }
    }
  }
}

__global__ void __launch_bounds__(256) moe_finalize(
    const __nv_bfloat16* __restrict__ ybuf, const float* __restrict__ ss, const int* __restrict__ ids,
    const float* __restrict__ tw, int local_start, __nv_bfloat16* __restrict__ out) {
  const int t = blockIdx.x, tid = threadIdx.x;
  float acc[16];
#pragma unroll
  for (int i = 0; i < 16; i++) acc[i] = 0.f;
#pragma unroll 1
  for (int k = 0; k < 8; k++) {
    const int p = t * 8 + k;
    const int e = ids[p] - local_start;
    if ((unsigned)e < 32u) {
      const float f = tw[p] * rsqrtf(ss[p] * (1.f / 4096.f) + 1e-8f);
      const uint4* yp = reinterpret_cast<const uint4*>(ybuf + (size_t)p * KDIM + tid * 16);
      uint4 a = yp[0], b = yp[1];
      const __nv_bfloat162* av = reinterpret_cast<const __nv_bfloat162*>(&a);
      const __nv_bfloat162* bv = reinterpret_cast<const __nv_bfloat162*>(&b);
#pragma unroll
      for (int i = 0; i < 4; i++) {
        float2 fa = __bfloat1622float2(av[i]);
        float2 fb = __bfloat1622float2(bv[i]);
        acc[2 * i] += f * fa.x; acc[2 * i + 1] += f * fa.y;
        acc[8 + 2 * i] += f * fb.x; acc[8 + 2 * i + 1] += f * fb.y;
      }
    }
  }
  uint4 o[2];
  __nv_bfloat162* ov = reinterpret_cast<__nv_bfloat162*>(o);
#pragma unroll
  for (int i = 0; i < 8; i++) ov[i] = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
  uint4* op = reinterpret_cast<uint4*>(out + (size_t)t * KDIM + tid * 16);
  op[0] = o[0];
  op[1] = o[1];
}

template <int MODE, int NT, bool SAFE>
static void launch_t(const torch::Tensor& act, const torch::Tensor& ids, int M, int ls, const torch::Tensor& wpk,
                     const torch::Tensor& gsc, const torch::Tensor& normw, torch::Tensor& outp, torch::Tensor& ss,
                     int grid) {
  auto kern = moe_gemm<MODE, NT, SAFE>;
  const int smem = (NWARPS * 2 * NT * 4 * 32 + M * 8) * 4;
  static int configured = 0;
  if (smem > configured) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem > 48 * 1024 ? smem : 48 * 1024);
    configured = smem;
  }
  kern<<<grid, 128, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(act.data_ptr()), ids.data_ptr<int>(), M, ls,
      reinterpret_cast<const uint4*>(wpk.data_ptr()), gsc.data_ptr<float>(), normw.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(outp.data_ptr()), ss.data_ptr<float>());
}

template <int MODE, int NT, bool SAFE>
static int occ_t(int M) {
  int n = 0;
  const int smem = (NWARPS * 2 * NT * 4 * 32 + M * 8) * 4;
  if (smem > 48 * 1024)
    cudaFuncSetAttribute(moe_gemm<MODE, NT, SAFE>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_gemm<MODE, NT, SAFE>, 128, smem);
  return n;
}

#define DISPATCH(FN, ...)                                                        \
  if (mode == 1) {                                                               \
    if (safe) { if (nt == 1) FN<1, 1, true>(__VA_ARGS__); else if (nt == 2) FN<1, 2, true>(__VA_ARGS__); else FN<1, 4, true>(__VA_ARGS__); } \
    else { if (nt == 1) FN<1, 1, false>(__VA_ARGS__); else if (nt == 2) FN<1, 2, false>(__VA_ARGS__); else FN<1, 4, false>(__VA_ARGS__); } \
  } else {                                                                       \
    if (safe) { if (nt == 1) FN<2, 1, true>(__VA_ARGS__); else if (nt == 2) FN<2, 2, true>(__VA_ARGS__); else FN<2, 4, true>(__VA_ARGS__); } \
    else { if (nt == 1) FN<2, 1, false>(__VA_ARGS__); else if (nt == 2) FN<2, 2, false>(__VA_ARGS__); else FN<2, 4, false>(__VA_ARGS__); } \
  }

void gemm(int64_t mode, int64_t nt, bool safe, torch::Tensor act, torch::Tensor ids, int64_t M, int64_t ls,
          torch::Tensor wpk, torch::Tensor gsc, torch::Tensor normw, torch::Tensor outp, torch::Tensor ss,
          int64_t grid) {
  DISPATCH(launch_t, act, ids, (int)M, (int)ls, wpk, gsc, normw, outp, ss, (int)grid)
}

int64_t occupancy(int64_t mode, int64_t nt, bool safe, int64_t M) {
  int n = 0;
  if (mode == 1) {
    if (safe) { n = nt == 1 ? occ_t<1, 1, true>(M) : nt == 2 ? occ_t<1, 2, true>(M) : occ_t<1, 4, true>(M); }
    else { n = nt == 1 ? occ_t<1, 1, false>(M) : nt == 2 ? occ_t<1, 2, false>(M) : occ_t<1, 4, false>(M); }
  } else {
    if (safe) { n = nt == 1 ? occ_t<2, 1, true>(M) : nt == 2 ? occ_t<2, 2, true>(M) : occ_t<2, 4, true>(M); }
    else { n = nt == 1 ? occ_t<2, 1, false>(M) : nt == 2 ? occ_t<2, 2, false>(M) : occ_t<2, 4, false>(M); }
  }
  return n;
}

std::vector<int64_t> kinfo(int64_t mode, int64_t nt, bool safe) {
  cudaFuncAttributes a;
  if (mode == 1) {
    if (safe) { if (nt == 1) cudaFuncGetAttributes(&a, moe_gemm<1, 1, true>); else if (nt == 2) cudaFuncGetAttributes(&a, moe_gemm<1, 2, true>); else cudaFuncGetAttributes(&a, moe_gemm<1, 4, true>); }
    else { if (nt == 1) cudaFuncGetAttributes(&a, moe_gemm<1, 1, false>); else if (nt == 2) cudaFuncGetAttributes(&a, moe_gemm<1, 2, false>); else cudaFuncGetAttributes(&a, moe_gemm<1, 4, false>); }
  } else {
    if (safe) { if (nt == 1) cudaFuncGetAttributes(&a, moe_gemm<2, 1, true>); else if (nt == 2) cudaFuncGetAttributes(&a, moe_gemm<2, 2, true>); else cudaFuncGetAttributes(&a, moe_gemm<2, 4, true>); }
    else { if (nt == 1) cudaFuncGetAttributes(&a, moe_gemm<2, 1, false>); else if (nt == 2) cudaFuncGetAttributes(&a, moe_gemm<2, 2, false>); else cudaFuncGetAttributes(&a, moe_gemm<2, 4, false>); }
  }
  return {(int64_t)a.numRegs, (int64_t)a.localSizeBytes};
}

void finalize(torch::Tensor ybuf, torch::Tensor ss, torch::Tensor ids, torch::Tensor tw, int64_t ls, torch::Tensor out) {
  const int M = out.size(0);
  moe_finalize<<<M, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(ybuf.data_ptr()), ss.data_ptr<float>(), ids.data_ptr<int>(),
      tw.data_ptr<float>(), (int)ls, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));
}
"""

CPP_SRC = r"""
void gemm(int64_t mode, int64_t nt, bool safe, torch::Tensor act, torch::Tensor ids, int64_t M, int64_t ls,
          torch::Tensor wpk, torch::Tensor gsc, torch::Tensor normw, torch::Tensor outp, torch::Tensor ss,
          int64_t grid);
int64_t occupancy(int64_t mode, int64_t nt, bool safe, int64_t M);
std::vector<int64_t> kinfo(int64_t mode, int64_t nt, bool safe);
void finalize(torch::Tensor ybuf, torch::Tensor ss, torch::Tensor ids, torch::Tensor tw, int64_t ls, torch::Tensor out);
"""

_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        _EXT = load_inline(
            name="muse_moe_w4a16_v9",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["gemm", "occupancy", "kinfo", "finalize"],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "--use_fast_math", "-lineinfo"],
            verbose=False,
        )
    return _EXT


# ---------------------------------------------------------------- packing (prepare only)
_FA = (7, 8, 14, 15)
# Output registers of a pair of k16 sub-steps: (word 0=A / 1=B / 2=V, rotate-left amount).
_SLOTS = [(0, 6), (0, 2), (0, 12), (2, 14), (1, 4), (1, 0), (1, 10), (2, 8)]
_TGT = (6, 7, 8, 15)  # e2m1 bit (m, e0, e1, sign) -> bit of the bf16 half


def _bitmap():
    out, used = [], set()
    for idx, (w, L) in enumerate(_SLOTS):
        s_local, reg = divmod(idx, 4)
        for half in range(2):
            for b in range(4):
                src = (16 * half + _TGT[b] - L) % 32
                ww = w if w != 2 else (0 if (src % 16) in _FA else 1)
                assert (ww, src) not in used
                used.add((ww, src))
                out.append((s_local, reg, half, b, ww, src))
    assert len(used) == 64
    return out


def _pack_gemm(wq, wsf, row_idx):
    """wq uint8 [E, N, K/2]; wsf uint8 [E, N, K/16]; row_idx long [ITEMS, 2, 16] -> uint8 [E, ITEMS, K/64, 1152]."""
    E, N, Kh = wq.shape
    K = Kh * 2
    KS = K // 64
    ITEMS = row_idx.shape[0]
    codes = torch.stack((wq & 0xF, wq >> 4), dim=-1).reshape(E, N, K)
    c = codes[:, row_idx.reshape(-1)].reshape(E, ITEMS, 2, 2, 8, KS, 2, 2, 4, 4)
    c = c.permute(0, 1, 5, 2, 4, 8, 6, 7, 3, 9).contiguous().to(torch.int32)
    words = torch.zeros((E, ITEMS, KS, 2, 8, 4, 2, 2), dtype=torch.int32, device=wq.device)
    for (sl, reg, half, b, ww, src) in _bitmap():
        rh = reg % 2
        q = (reg // 2) * 2 + half
        words[..., ww] |= ((c[..., sl, rh, q] >> b) & 1) << src
    wbytes = words.reshape(E, ITEMS, KS, 2, 32, 4).contiguous().view(torch.uint8).reshape(E, ITEMS, KS, 1024)
    s = wsf[:, row_idx.reshape(-1)].reshape(E, ITEMS, 2, 2, 8, KS, 4)
    s = s.permute(0, 1, 5, 4, 2, 3, 6).contiguous().reshape(E, ITEMS, KS, 128)
    return torch.cat((wbytes, s), dim=-1).contiguous()


def _pack_all(wq, wsf, row_idx, chunk=4):
    parts = []
    for e0 in range(0, wq.shape[0], chunk):
        parts.append(_pack_gemm(wq[e0:e0 + chunk], wsf[e0:e0 + chunk], row_idx))
    return torch.cat(parts, dim=0).contiguous()


def _needs_safe(sf_u8):
    exp = (sf_u8 >> 3) & 0xF
    return bool(((exp == 0) | (sf_u8 >= 0x7F)).any().item())


def prepare(weights):
    dev = weights["w13"].device
    ext = _ext()
    sl = slice(0, LOCAL)
    w13 = weights["w13"][sl]
    sf13 = weights["w13_sf"][sl].contiguous().view(torch.uint8)
    w2 = weights["w2"][sl]
    sf2 = weights["w2_sf"][sl].contiguous().view(torch.uint8)
    i = torch.arange(I_DIM // 16, device=dev)
    base = i[:, None] * 16 + torch.arange(16, device=dev)[None, :]
    rows1 = torch.stack((base, base + I_DIM), dim=1)
    i2 = torch.arange(H // 32, device=dev)
    rows2 = (i2[:, None] * 32 + torch.arange(32, device=dev)[None, :]).reshape(-1, 2, 16)
    pk1 = _pack_all(w13, sf13, rows1)
    pk2 = _pack_all(w2, sf2, rows2)
    safe = _needs_safe(sf13) or _needs_safe(sf2)
    state = {
        "pk1": pk1, "pk2": pk2, "safe": safe,
        "gsc1": (weights["w13_gs"][sl].float() * 128.0).contiguous(),
        "gsc2": (weights["w2_gs"][sl].float() * 128.0).contiguous(),
        "normw": (weights["post_norm_w"].float() + 1.0).contiguous(),
        "nsm": torch.cuda.get_device_properties(dev).multi_processor_count,
        "occ": {},
    }
    torch.cuda.synchronize()
    print(f"[prep] safe={safe} pk1={tuple(pk1.shape)} pk2={tuple(pk2.shape)}")
    if DEBUG:
        for mode in (1, 2):
            for nt in (1, 2, 4):
                print(f"[kinfo] mode={mode} nt={nt} regs/local={list(ext.kinfo(mode, nt, safe))} "
                      f"occ(M=256)={ext.occupancy(mode, nt, safe, 256)}")
        _selftest(state, weights)
    return state


def _nt_for(M):
    return 1 if M <= 16 or M == 32 else 2


def _grid(state, mode, nt, M):
    key = (mode, nt, M)
    g = state["occ"].get(key)
    if g is None:
        occ = max(1, int(_ext().occupancy(mode, nt, state["safe"], M)))
        tiles = 256 if mode == 1 else 128
        g = occ * state["nsm"]
        state["occ"][key] = g
    return g


DEBUG = False


def _route(bs, seed, dev):
    """Diagnostics only: synthetic skewed routing (Student-t expert popularity), GPU 0 busiest."""
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn((bs, H), generator=g, device=dev).to(torch.bfloat16)
    scores = torch.rand((bs, 256), generator=g, device=dev)
    z = torch.randn((256,), generator=g, device=dev)
    u = torch.rand((256,), generator=g, device=dev).clamp(min=1e-12)
    scores = scores + 0.03 * z / (-u.log()).sqrt()
    v, ids = scores.topk(TOP_K, dim=-1)
    pairs = torch.bincount((ids // 32).flatten(), minlength=8)
    ids = (ids - 32 * pairs.argmax()) % 256
    cfg = {"bs": bs, "d_hidden": H, "d_expert": I_DIM, "n_experts": 256, "local_start": 0,
           "local_count": 32, "top_k": 8, "sharding": "ep8"}
    return {"hidden_states": x, "topk_ids": ids.to(torch.int32),
            "topk_weights": (v / v.sum(-1, keepdim=True)).float(), "config": cfg}


def _torch_ref(data, weights):
    from nvfp4 import dequantize
    x = data["hidden_states"].float()
    ids = data["topk_ids"].long()
    out = torch.zeros_like(x)
    w = weights["post_norm_w"].float()
    for e in torch.unique(ids).tolist():
        if e >= LOCAL:
            continue
        tok, slot = (ids == e).nonzero(as_tuple=True)
        h = x[tok] @ dequantize(weights["w13"][e], weights["w13_sf"][e], weights["w13_gs"][e]).T
        it = torch.nn.functional.silu(h[:, :I_DIM]) * h[:, I_DIM:]
        y = (it @ dequantize(weights["w2"][e], weights["w2_sf"][e], weights["w2_gs"][e]).T) * (1 + w)
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-8)
        out.index_add_(0, tok, y * data["topk_weights"][tok, slot].unsqueeze(1))
    return out


def _selftest(state, weights):
    dev = weights["w13"].device
    for bs in (1, 33, 257):
        d = _route(bs, 100 + bs, dev)
        o = kernel_fn(d, state).float()
        r = _torch_ref(d, weights)
        err = ((o - r).norm() / r.norm()).item()
        rows = ((o - r).norm(dim=-1) / torch.maximum(r.norm(dim=-1), 0.1 * r.norm(dim=-1).mean())).max().item()
        print(f"[selftest] bs={bs} rel_l2={err:.3e} worst_row={rows:.3e}")
    flush = torch.empty((96 * 1024 * 1024,), dtype=torch.int32, device=dev)
    for bs in (1, 16, 32, 8, 128, 256):
        d = _route(bs, 7 + bs, dev)
        ids = d["topk_ids"]
        loc = ids[ids < 32].long()
        cnt = torch.bincount(loc, minlength=32)
        kernel_fn(d, state)
        torch.cuda.synchronize()
        gph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gph):
            kernel_fn(d, state)
        ts = []
        for _ in range(30):
            flush.add_(1)
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record()
            gph.replay()
            b.record()
            torch.cuda.synchronize()
            ts.append(a.elapsed_time(b) * 1000)
        ts.sort()
        touched = int((cnt > 0).sum())
        floor = touched * 3 * 4096 * 4096 * 0.5625 / 4.8e12 * 1e6
        print(f"[bench] bs={bs} med={ts[len(ts)//2]:.1f}us min={ts[0]:.1f}us touched={touched} "
              f"pairs={int(cnt.sum())} maxcnt={int(cnt.max())} bwfloor={floor:.1f}us")
    del flush


def kernel_fn(data, state):
    cfg = data["config"]
    ext = _ext()
    x = data["hidden_states"].contiguous()
    ids = data["topk_ids"].contiguous()
    if ids.dtype != torch.int32:
        ids = ids.to(torch.int32)
    tw = data["topk_weights"].contiguous()
    if tw.dtype != torch.float32:
        tw = tw.float()
    M = x.shape[0]
    ls = int(cfg["local_start"])
    assert ls == 0 and int(cfg["local_count"]) == LOCAL
    nt = _nt_for(M)
    dev = x.device
    inter = torch.empty((M * TOP_K, I_DIM), dtype=torch.bfloat16, device=dev)
    ybuf = torch.empty((M * TOP_K, H), dtype=torch.bfloat16, device=dev)
    ss = torch.empty((M * TOP_K,), dtype=torch.float32, device=dev)
    out = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
    safe = state["safe"]
    ext.gemm(1, nt, safe, x, ids, M, ls, state["pk1"], state["gsc1"], state["normw"], inter, ss,
             _grid(state, 1, nt, M))
    ext.gemm(2, nt, safe, inter, ids, M, ls, state["pk2"], state["gsc2"], state["normw"], ybuf, ss,
             _grid(state, 2, nt, M))
    ext.finalize(ybuf, ss, ids, tw, ls, out)
    return out
