# okc: problem=muse_attention_decode
"""Fused gated-attention decode for Muse Spark Open on H200.

One CUDA kernel (plus a tiny split-combine kernel when the KV range is split) replaces SGLang's
FusedQKNorm -> store_cache -> q.to(fp8) -> FA3 -> rms_norm_gate chain:

  * prologue: per-head RMSNorm of q and the new k (+ interleaved RoPE on sliding layers), new K/V
    quantized to E4M3 and written to the paged cache by the CTA whose key range holds them;
    the new keys are attended straight from shared memory, so no CTA reads a row being written.
  * main loop: split-KV (flash-decoding) over the paged FP8 cache, cp.async 4-stage pipeline of
    64-key tiles, both KV heads of a pair per CTA (128 contiguous bytes per cache row), the 8
    query heads x L tokens of a KV head packed into m16 tensor-core tiles (GQA packing),
    FP8 -> FP16 in registers, f32 accumulation and online softmax.
  * epilogue: RMSNorm(attention) * sigmoid(gate) -> BF16, in the combine kernel when split.
"""

import math
from typing import TypedDict

import torch
from torch.utils.cpp_extension import load_inline


class Config(TypedDict):
    batch: int
    q_len: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    window: int
    rope_theta: float
    qk_scale_factor: float
    sm_scale: float
    kv_scale: float


class Inputs(TypedDict):
    qkv: torch.Tensor
    positions: torch.Tensor
    cache_loc: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    scheduler_metadata: torch.Tensor | None
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    config: Config


CUDA_SRC = r'''
// Fused gated-attention decode for Muse Spark Open (H200 / sm_90a), single launch.
//  prologue : warp-cooperative RMSNorm (+RoPE, sin/cos computed once per token x freq) of q and the
//             new k; new K/V -> E4M3, written to the paged cache by the CTA owning their positions
//             and attended from shared memory.
//  main loop: split-KV over a thread-block cluster; 4-stage cp.async pipeline of 64-key tiles,
//             both KV heads of a pair per CTA (128 contiguous bytes per cache row); page-table
//             entries streamed through a small ring two tiles ahead of the data they address;
//             QK^T with FP8 E4M3 tensor-core MMA (m16n8k32, raw cache bytes, FP8 query),
//             online softmax with packed f16x2 exp2, PV with f16 MMA.
//  epilogue : cluster-wide (m, l, O) merge through distributed shared memory; when one cluster
//             covers the whole range the CTA finishes RMSNorm x sigmoid(gate) directly, else the
//             last-arriving cluster (per row slice) merges the clusters' partials from global.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdint.h>

namespace cg = cooperative_groups;

#define HD 64
#define TILE 64
#define NST 4
#define NTHR 128
#define IDXS 72
#define NIDX (2 * NST)
#define RMS_EPS 1e-5f

struct Params {
  const uint16_t* qkv;
  const int64_t* positions;
  const int64_t* cache_loc;
  const int* seq_lens;
  uint8_t* kc;
  uint8_t* vc;
  const int* page_table;
  const float* k_scale;
  const float* v_scale;
  const float* inv_freq;
  uint16_t* out;
  float* opart;   // [B, NHP, NC, RT, 64]
  float* lse;     // [B, NHP, NC, RT]
  int* counters;  // [B, NHP, C]
  int B, Hq, Hkv, W, ps, max_pages, window, use_rope, C, NC, chunk, nmax;
  float scale_log2;
};

__device__ __forceinline__ float bf2f(uint32_t x16) { return __uint_as_float(x16 << 16); }
__device__ __forceinline__ uint16_t f2bf(float f) {
  uint16_t r;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(f));
  return r;
}
__device__ __forceinline__ uint32_t pack_f16(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ uint16_t f2e4m3x2(float lo, float hi) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ void e4m3x4_to_f16(uint32_t x, uint32_t& lo, uint32_t& hi) {
  asm("{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, %2;\n\tcvt.rn.f16x2.e4m3x2 %0, l;\n\tcvt.rn.f16x2.e4m3x2 %1, h;\n\t}"
      : "=r"(lo), "=r"(hi) : "r"(x));
}
__device__ __forceinline__ float ex2(float x) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ uint32_t ex2h2(uint32_t x) {
  uint32_t r;
  asm("ex2.approx.f16x2 %0, %1;" : "=r"(r) : "r"(x));
  return r;
}
__device__ __forceinline__ uint32_t hadd2(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("add.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
__device__ __forceinline__ float h2sum(uint32_t x) {
  float a, b;
  asm("{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, %2;\n\tcvt.f32.f16 %0, l;\n\tcvt.f32.f16 %1, h;\n\t}"
      : "=f"(a), "=f"(b) : "r"(x));
  return a + b;
}
__device__ __forceinline__ void mma_f16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_fp8(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp16(void* sm, const void* g) {
  uint32_t s = (uint32_t)__cvta_generic_to_shared(sm);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(g));
}
__device__ __forceinline__ void cp4(void* sm, const void* g) {
  uint32_t s = (uint32_t)__cvta_generic_to_shared(sm);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(s), "l"(g));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}
__device__ __forceinline__ int swz(int key) { return ((key ^ (key >> 2)) & 1) << 2; }

template <int L>
__global__ void __launch_bounds__(NTHR, 2) attn_kernel(const Params p) {
  constexpr int ROWS = 8 * L;            // query rows per kv head
  constexpr int RT = 2 * ROWS;           // rows per CTA (2 kv heads)
  constexpr int MT = (L == 1) ? 1 : 2;
  constexpr int KW = (L == 1) ? 32 : 64;
  constexpr int NT = KW / 8;
  constexpr int QV = RT / 4;             // q vectors per warp
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* sK = smem;
  uint8_t* sV = sK + NST * TILE * 128;
  int* sIdx = (int*)(sV + NST * TILE * 128);
  uint8_t* sQ8 = (uint8_t*)(sIdx + NIDX * IDXS);
  float2* sCS = (float2*)(sQ8 + RT * HD);
  uint8_t* sNew = (uint8_t*)(sCS + L * 32);
  uint16_t* sGate = (uint16_t*)(sNew + 2 * L * 128);
  // after the main loop (aliases the K/V stages)
  float* sRed = (float*)smem;                         // L==1: [4 warps][8 rows][64]
  float* sRedML = sRed + 4 * 8 * HD;                  // [4][8][2]
  float* sFin = (float*)(smem + (L == 1 ? 16384 : 0));  // [RT][64]
  float* sFinML = sFin + RT * HD;                     // [RT][2]

  cg::cluster_group cluster = cg::this_cluster();
  const int C = p.C;
  const int s = blockIdx.x, hp = blockIdx.y, b = blockIdx.z;
  const int rank = (int)cluster.block_rank();
  const int cid = s / C;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  const int Hkv = p.Hkv, ps = p.ps, Hq = p.Hq;
  const int NHP = Hkv >> 1;
  const int tok0 = b * L;
  const bool slide = p.window > 0;
  const int RPR = RT / C;
  const int row0 = rank * RPR;
  const int* ptab = p.page_table + (size_t)b * p.max_pages;

  // ---- prefetches that do not depend on the request length
  for (int i = tid; i < RPR * 8; i += NTHR) {
    int rc = row0 + (i >> 3), c = i & 7;
    int h = rc / ROWS, r = rc % ROWS, j = r >> 3, gg = r & 7;
    int H = (hp * 2 + h) * 8 + gg;
    cp16(sGate + (i >> 3) * HD + c * 8, p.qkv + (size_t)(tok0 + j) * p.W + (Hq + 2 * Hkv) * HD + H * HD + c * 8);
  }
  int kb = s * p.chunk;
  auto issue_idx = [&](int X, int ke_lim) {
    int t0 = kb + X * TILE;
    if (t0 < ke_lim) {
      int first = t0 / ps;
      int last = min(min(t0 + TILE - 1, ke_lim - 1) / ps, p.max_pages - 1);
      int n = last - first + 1;
      int* dst = sIdx + (X % NIDX) * IDXS;
      for (int i = tid; i < n; i += NTHR) cp4(dst + i, ptab + first + i);
    }
  };
  if (!slide) {
    int lim = min(kb + p.chunk, p.nmax);
#pragma unroll
    for (int X = 0; X < 2 * NST - 2; ++X) issue_idx(X, lim);
  }
  cp_commit();

  uint32_t qraw[QV];
#pragma unroll
  for (int k = 0; k < QV; ++k) {
    int v = warp + 4 * k, h = v / ROWS, r = v % ROWS, j = r >> 3, gg = r & 7;
    int H = (hp * 2 + h) * 8 + gg;
    qraw[k] = *(const uint32_t*)(p.qkv + (size_t)(tok0 + j) * p.W + H * HD + 2 * lane);
  }

  const int seq = p.seq_lens[b];
  const int end = seq + L;
  if (slide) kb = max(0, seq - p.window + 1) + s * p.chunk;
  const int ke = min(kb + p.chunk, end);
  if (slide) {
#pragma unroll
    for (int X = 0; X < 2 * NST - 2; ++X) issue_idx(X, ke);
  }
  cp_commit();

  const bool has_new = (kb < ke) && (ke > seq);
  uint32_t nraw[L];
  if (has_new) {
#pragma unroll
    for (int k = 0; k < L; ++k) {
      int u = warp + 4 * k;  // 0..4L-1
      bool isv = u >= 2 * L;
      int ii = isv ? u - 2 * L : u;
      int j = ii >> 1, h = ii & 1;
      nraw[k] = *(const uint32_t*)(p.qkv + (size_t)(tok0 + j) * p.W + (Hq + (isv ? Hkv : 0) + hp * 2 + h) * HD +
                                   2 * lane);
    }
  }
  if (p.use_rope) {
    for (int i = tid; i < L * 32; i += NTHR) {
      int j = i >> 5, f = i & 31;
      float pos = (float)p.positions[tok0 + j];
      float sn, cs;
      sincosf(pos * p.inv_freq[f], &sn, &cs);
      sCS[i] = make_float2(cs, sn);
    }
  }
  __syncthreads();

  // ---- queries -> FP8 E4M3 (normalized, RoPE'd; qk_scale_factor folded into the softmax scale)
#pragma unroll
  for (int k = 0; k < QV; ++k) {
    int v = warp + 4 * k, r = v % ROWS, j = r >> 3;
    float x0 = bf2f(qraw[k] & 0xffff), x1 = bf2f(qraw[k] >> 16);
    float inv = rsqrtf(warp_sum(x0 * x0 + x1 * x1) * (1.f / HD) + RMS_EPS);
    x0 *= inv;
    x1 *= inv;
    if (p.use_rope) {
      float2 cs = sCS[j * 32 + lane];
      float y0 = x0 * cs.x - x1 * cs.y, y1 = x0 * cs.y + x1 * cs.x;
      x0 = y0;
      x1 = y1;
    }
    *(uint16_t*)(sQ8 + v * HD + 2 * lane) = f2e4m3x2(x0, x1);
  }
  // ---- new K/V of the block
  if (has_new) {
    const float iks = 1.f / *p.k_scale, ivs = 1.f / *p.v_scale;
#pragma unroll
    for (int k = 0; k < L; ++k) {
      int u = warp + 4 * k;
      bool isv = u >= 2 * L;
      int ii = isv ? u - 2 * L : u;
      int j = ii >> 1, h = ii & 1;
      float x0 = bf2f(nraw[k] & 0xffff), x1 = bf2f(nraw[k] >> 16);
      if (!isv) {
        float inv = rsqrtf(warp_sum(x0 * x0 + x1 * x1) * (1.f / HD) + RMS_EPS);
        x0 *= inv;
        x1 *= inv;
        if (p.use_rope) {
          float2 cs = sCS[j * 32 + lane];
          float y0 = x0 * cs.x - x1 * cs.y, y1 = x0 * cs.y + x1 * cs.x;
          x0 = y0;
          x1 = y1;
        }
        x0 *= iks;
        x1 *= iks;
      } else {
        x0 *= ivs;
        x1 *= ivs;
      }
      uint16_t q8 = f2e4m3x2(x0, x1);
      *(uint16_t*)(sNew + ((isv ? L : 0) + j) * 128 + h * HD + 2 * lane) = q8;
      int pos = seq + j;
      if (pos >= kb && pos < ke) {
        int64_t row = p.cache_loc[tok0 + j];
        *(uint16_t*)((isv ? p.vc : p.kc) + ((size_t)row * Hkv + hp * 2 + h) * HD + 2 * lane) = q8;
      }
    }
  }
  cp_wait<0>();
  __syncthreads();

  // ---- Q fragments (FP8, k32): lane reads d = 16t + 8ks .. +7
  const int h = (L == 1) ? (warp & 1) : (warp >> 1);
  const int keyoff = (L == 1) ? (warp >> 1) * 32 : 0;
  const int mtb = (L == 1) ? 0 : (warp & 1) * 2;
  uint32_t qa[MT][2][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    int r0 = (mtb + mt) * 16 + g, r1 = r0 + 8;
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      int off = 16 * t + 8 * ks;
      const uint32_t* q0 = (const uint32_t*)(sQ8 + (h * ROWS + r0) * HD + off);
      const uint32_t* q1 = (const uint32_t*)(sQ8 + (h * ROWS + r1) * HD + off);
      qa[mt][ks][0] = (r0 < ROWS) ? q0[0] : 0u;
      qa[mt][ks][1] = (r1 < ROWS) ? q1[0] : 0u;
      qa[mt][ks][2] = (r0 < ROWS) ? q0[1] : 0u;
      qa[mt][ks][3] = (r1 < ROWS) ? q1[1] : 0u;
    }
  }

  const float scale = p.scale_log2 * (*p.k_scale);
  const int ntiles = (kb < ke) ? (ke - kb + TILE - 1) / TILE : 0;
  const size_t headoff = (size_t)hp * 2 * HD;

  auto load_tile = [&](int it, int st) {
    const int t0 = kb + it * TILE;
    const int pf = t0 / ps;
    const int* idx = sIdx + (it % NIDX) * IDXS;
    uint8_t* bk = sK + st * TILE * 128;
    uint8_t* bv = sV + st * TILE * 128;
#pragma unroll
    for (int r = 0; r < (TILE * 8) / NTHR; ++r) {
      int ci = tid + r * NTHR;
      int key = ci >> 3, c = ci & 7;
      int pos = t0 + key;
      int so = key * 128 + ((c ^ swz(key)) << 4);
      if (pos < ke && pos < seq) {
        int row;
        if (ps == 1) row = idx[key];
        else {
          int pi = pos / ps;
          row = idx[pi - pf] * ps + (pos - pi * ps);
        }
        size_t off = (size_t)row * Hkv * HD + headoff + c * 16;
        cp16(bk + so, p.kc + off);
        cp16(bv + so, p.vc + off);
      } else if (pos < ke) {
        int j = pos - seq;
        *(uint4*)(bk + so) = *(const uint4*)(sNew + j * 128 + c * 16);
        *(uint4*)(bv + so) = *(const uint4*)(sNew + (L + j) * 128 + c * 16);
      } else {
        *(uint4*)(bk + so) = make_uint4(0, 0, 0, 0);
        *(uint4*)(bv + so) = make_uint4(0, 0, 0, 0);
      }
    }
  };

#pragma unroll
  for (int i = 0; i < NST - 1; ++i) {
    if (i < ntiles) load_tile(i, i);
    cp_commit();
  }

  float m[MT][2], l[MT][2], o[MT][8][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    m[mt][0] = m[mt][1] = -INFINITY;
    l[mt][0] = l[mt][1] = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) o[mt][i][0] = o[mt][i][1] = o[mt][i][2] = o[mt][i][3] = 0.f;
  }

  for (int it = 0; it < ntiles; ++it) {
    cp_wait<NST - 2>();
    __syncthreads();
    {
      int nx = it + NST - 1;
      if (nx < ntiles) load_tile(nx, nx % NST);
      issue_idx(it + 2 * NST - 2, ke);
      cp_commit();
    }
    const int st = it % NST;
    const uint8_t* K = sK + st * TILE * 128;
    const uint8_t* V = sV + st * TILE * 128;
    const int t0 = kb + it * TILE;

    float sc[MT][NT][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) sc[mt][nt][0] = sc[mt][nt][1] = sc[mt][nt][2] = sc[mt][nt][3] = 0.f;

#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      int key = keyoff + nt * 8 + g;
      uint4 kr = *(const uint4*)(K + key * 128 + (((h * 4 + t) ^ swz(key)) << 4));
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        mma_fp8(sc[mt][nt], qa[mt][0], kr.x, kr.y);
        mma_fp8(sc[mt][nt], qa[mt][1], kr.z, kr.w);
      }
    }

    const bool interior = (t0 + TILE <= ke) && (t0 + TILE - 1 <= seq) && (!slide || t0 >= seq + L - p.window);
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          float v = sc[mt][nt][e] * scale;
          if (!interior) {
            int key = t0 + keyoff + nt * 8 + 2 * t + (e & 1);
            int j = (mtb + mt) * 2 + (e >> 1);
            bool ok = key < ke && key <= seq + j && (!slide || key > seq + j - p.window);
            if (!ok) v = -INFINITY;
          }
          sc[mt][nt][e] = v;
        }

    uint32_t ph[MT][NT][2];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        if (L == 1 && hf == 1) {
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) ph[mt][nt][1] = 0u;
          continue;
        }
        float mx = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) mx = fmaxf(mx, fmaxf(sc[mt][nt][2 * hf], sc[mt][nt][2 * hf + 1]));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
        float mn = fmaxf(m[mt][hf], mx);
        float mu = (mn == -INFINITY) ? 0.f : mn;
        float corr = ex2(m[mt][hf] - mu);
        m[mt][hf] = mn;
        l[mt][hf] *= corr;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          o[mt][i][2 * hf] *= corr;
          o[mt][i][2 * hf + 1] *= corr;
        }
        uint32_t acc = 0u;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          uint32_t e2 = ex2h2(pack_f16(sc[mt][nt][2 * hf] - mu, sc[mt][nt][2 * hf + 1] - mu));
          ph[mt][nt][hf] = e2;
          acc = hadd2(acc, e2);
        }
        l[mt][hf] += h2sum(acc);
      }
    }

#pragma unroll
    for (int ks = 0; ks < NT / 2; ++ks) {
      uint32_t pa[MT][4];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        pa[mt][0] = ph[mt][2 * ks][0];
        pa[mt][1] = ph[mt][2 * ks][1];
        pa[mt][2] = ph[mt][2 * ks + 1][0];
        pa[mt][3] = ph[mt][2 * ks + 1][1];
      }
      const int k0 = keyoff + 16 * ks + 2 * t;
      const int cofs = (g & 1) * 8;
      const int cb = h * 4 + (g >> 1);
      uint2 rA = *(const uint2*)(V + (k0)*128 + ((cb ^ swz(k0)) << 4) + cofs);
      uint2 rB = *(const uint2*)(V + (k0 + 1) * 128 + ((cb ^ swz(k0 + 1)) << 4) + cofs);
      uint2 rC = *(const uint2*)(V + (k0 + 8) * 128 + ((cb ^ swz(k0 + 8)) << 4) + cofs);
      uint2 rD = *(const uint2*)(V + (k0 + 9) * 128 + ((cb ^ swz(k0 + 9)) << 4) + cofs);
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        uint32_t a = half ? rA.y : rA.x, bb = half ? rB.y : rB.x;
        uint32_t c = half ? rC.y : rC.x, dd = half ? rD.y : rD.x;
#pragma unroll
        for (int q2 = 0; q2 < 2; ++q2) {
          const uint32_t sel = q2 ? 0x7362u : 0x5140u;
          uint32_t ab0, ab1, cd0, cd1;
          e4m3x4_to_f16(__byte_perm(a, bb, sel), ab0, ab1);
          e4m3x4_to_f16(__byte_perm(c, dd, sel), cd0, cd1);
          const int i = half * 4 + q2 * 2;
#pragma unroll
          for (int mt = 0; mt < MT; ++mt) {
            mma_f16(o[mt][i], pa[mt], ab0, cd0);
            mma_f16(o[mt][i + 1], pa[mt], ab1, cd1);
          }
        }
      }
    }
  }

  cp_wait<0>();
  __syncthreads();

  // ---- CTA-level (m, l, O) per row -> sFin
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      if (L == 1 && hf == 1) continue;
      float lv = l[mt][hf];
      lv += __shfl_xor_sync(0xffffffff, lv, 1);
      lv += __shfl_xor_sync(0xffffffff, lv, 2);
      float* dst;
      float* ml;
      if (L == 1) {
        dst = sRed + (warp * 8 + g) * HD;
        ml = sRedML + (warp * 8 + g) * 2;
      } else {
        int rc = h * ROWS + (mtb + mt) * 16 + hf * 8 + g;
        dst = sFin + rc * HD;
        ml = sFinML + rc * 2;
      }
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        dst[16 * t + i] = o[mt][i][2 * hf];
        dst[16 * t + 8 + i] = o[mt][i][2 * hf + 1];
      }
      if (t == 0) {
        ml[0] = m[mt][hf];
        ml[1] = lv;
      }
    }
  __syncthreads();
  if (L == 1) {
    for (int rc = warp; rc < RT; rc += 4) {
      int hh = rc >> 3, gg = rc & 7;
      const float* m0 = sRedML + (hh * 8 + gg) * 2;
      const float* m1 = sRedML + ((hh + 2) * 8 + gg) * 2;
      float M = fmaxf(m0[0], m1[0]);
      float w0 = (M == -INFINITY) ? 0.f : ex2(m0[0] - M);
      float w1 = (M == -INFINITY) ? 0.f : ex2(m1[0] - M);
      const float* o0 = sRed + (hh * 8 + gg) * HD;
      const float* o1 = sRed + ((hh + 2) * 8 + gg) * HD;
      sFin[rc * HD + 2 * lane] = w0 * o0[2 * lane] + w1 * o1[2 * lane];
      sFin[rc * HD + 2 * lane + 1] = w0 * o0[2 * lane + 1] + w1 * o1[2 * lane + 1];
      if (lane == 0) {
        sFinML[rc * 2] = M;
        sFinML[rc * 2 + 1] = w0 * m0[1] + w1 * m1[1];
      }
    }
  }
  cluster.sync();

  // ---- cluster merge through DSMEM. Thread i owns (row kk = i/16, 4-float chunk dc = i%16) of the
  //      RPR rows this CTA finishes; every peer's (m, l) and O chunk are loaded with all loads in flight.
  float* sMer = (float*)(smem + (L == 1 ? 24576 : 34816));  // [RPR][64], then [RPR][2]
  float* sMerML = sMer + RPR * HD;
  const int items = RPR * 16;
  for (int i0 = 0; i0 < items; i0 += NTHR) {
    const int i = i0 + tid;
    if (i < items) {
      const int kk = i >> 4, dc = i & 15, rc = row0 + kk;
      float mq[16], lq[16];
      float M = -INFINITY;
#pragma unroll
      for (int q = 0; q < 16; ++q) {
        mq[q] = -INFINITY;
        lq[q] = 0.f;
        if (q < C) {
          float2 ml = *(const float2*)(cluster.map_shared_rank(sFinML, q) + rc * 2);
          mq[q] = ml.x;
          lq[q] = ml.y;
        }
      }
#pragma unroll
      for (int q = 0; q < 16; ++q) M = fmaxf(M, mq[q]);
      float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
      float den = 0.f;
      if (M != -INFINITY) {
#pragma unroll
        for (int q = 0; q < 16; ++q) {
          if (q < C) {
            float4 v = *(const float4*)(cluster.map_shared_rank(sFin, q) + rc * HD + dc * 4);
            float w = ex2(mq[q] - M);
            den += w * lq[q];
            acc.x += w * v.x; acc.y += w * v.y; acc.z += w * v.z; acc.w += w * v.w;
          }
        }
      }
      *(float4*)(sMer + kk * HD + dc * 4) = acc;
      if (dc == 0) { sMerML[kk * 2] = M; sMerML[kk * 2 + 1] = den; }
    }
  }
  cluster.sync();  // peers are done reading this CTA's sFin; sMer is visible block-wide

  const float vscale = *p.v_scale;
  // finish one (row, chunk): o = normalized attention (4 values); 16 consecutive lanes hold a row
  auto finish = [&](int kk, int dc, float4 o, bool act) {
    float ss = act ? (o.x * o.x + o.y * o.y + o.z * o.z + o.w * o.w) : 0.f;
    ss += __shfl_xor_sync(0xffffffff, ss, 1);
    ss += __shfl_xor_sync(0xffffffff, ss, 2);
    ss += __shfl_xor_sync(0xffffffff, ss, 4);
    ss += __shfl_xor_sync(0xffffffff, ss, 8);
    if (!act) return;
    const int rc = row0 + kk;
    const int hh = rc / ROWS, r = rc % ROWS, j = r >> 3, gg = r & 7;
    const int H = (hp * 2 + hh) * 8 + gg;
    float rn = rsqrtf(ss * (1.f / HD) + RMS_EPS);
    uint2 gw = *(const uint2*)(sGate + kk * HD + dc * 4);
    float y0 = o.x * rn / (1.f + __expf(-bf2f(gw.x & 0xffff)));
    float y1 = o.y * rn / (1.f + __expf(-bf2f(gw.x >> 16)));
    float y2 = o.z * rn / (1.f + __expf(-bf2f(gw.y & 0xffff)));
    float y3 = o.w * rn / (1.f + __expf(-bf2f(gw.y >> 16)));
    uint2 ov;
    ov.x = (uint32_t)f2bf(y0) | ((uint32_t)f2bf(y1) << 16);
    ov.y = (uint32_t)f2bf(y2) | ((uint32_t)f2bf(y3) << 16);
    *(uint2*)(p.out + (size_t)(tok0 + j) * Hq * HD + H * HD + dc * 4) = ov;
  };

  if (p.NC == 1) {
    for (int i0 = 0; i0 < items; i0 += NTHR) {
      const int i = i0 + tid;
      const bool act = i < items;
      const int kk = act ? i >> 4 : 0, dc = i & 15;
      float4 o = make_float4(0.f, 0.f, 0.f, 0.f);
      if (act) {
        float den = sMerML[kk * 2 + 1];
        float inv = den > 0.f ? vscale / den : 0.f;
        float4 a = *(const float4*)(sMer + kk * HD + dc * 4);
        o = make_float4(a.x * inv, a.y * inv, a.z * inv, a.w * inv);
      }
      finish(kk, dc, o, act);
    }
    return;
  }

  // ---- several clusters: partial to global; the last arriver per (b, hp, rank) merges them
  const size_t pbase = ((size_t)(b * NHP + hp) * p.NC) * RT;
  for (int i = tid; i < items; i += NTHR) {
    const int kk = i >> 4, dc = i & 15, rc = row0 + kk;
    const size_t pi = pbase + (size_t)cid * RT + rc;
    float M = sMerML[kk * 2], den = sMerML[kk * 2 + 1];
    float inv = den > 0.f ? vscale / den : 0.f;
    float4 a = *(const float4*)(sMer + kk * HD + dc * 4);
    *(float4*)(p.opart + pi * HD + dc * 4) = make_float4(a.x * inv, a.y * inv, a.z * inv, a.w * inv);
    if (dc == 0) p.lse[pi] = den > 0.f ? M + __log2f(den) : -INFINITY;
  }
  __threadfence();
  __syncthreads();
  __shared__ int s_last;
  int* ctr = p.counters + (size_t)(b * NHP + hp) * C + rank;
  if (tid == 0) s_last = (atomicAdd(ctr, 1) == p.NC - 1);
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  for (int i0 = 0; i0 < items; i0 += NTHR) {
    const int i = i0 + tid;
    const bool act = i < items;
    const int kk = act ? i >> 4 : 0, dc = i & 15, rc = row0 + kk;
    float4 o = make_float4(0.f, 0.f, 0.f, 0.f);
    if (act) {
      const float* lp = p.lse + pbase + rc;
      float M = -INFINITY;
#pragma unroll 8
      for (int q = 0; q < p.NC; ++q) M = fmaxf(M, __ldcg(lp + (size_t)q * RT));
      float den = 0.f;
#pragma unroll 8
      for (int q = 0; q < p.NC; ++q) {
        size_t pi = pbase + (size_t)q * RT + rc;
        float w = ex2(__ldcg(lp + (size_t)q * RT) - M);
        float4 v = __ldcg((const float4*)(p.opart + pi * HD + dc * 4));
        den += w;
        o.x += w * v.x; o.y += w * v.y; o.z += w * v.z; o.w += w * v.w;
      }
      float inv = 1.f / den;
      o.x *= inv; o.y *= inv; o.z *= inv; o.w *= inv;
    }
    finish(kk, dc, o, act);
  }
  if (tid == 0) *ctr = 0;
}


static size_t smem_bytes(int L) {
  int RT = 16 * L;
  return (size_t)2 * NST * TILE * 128 + NIDX * IDXS * 4 + RT * HD + L * 32 * 8 + 2 * L * 128 + RT * HD * 2;
}

template <int L>
static int max_clusters_t(int C) {
  static int cache[17] = {0};
  static bool done[17] = {false};
  if (done[C]) return cache[C];
  size_t sm = smem_bytes(L);
  cudaFuncSetAttribute(attn_kernel<L>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
  cudaFuncSetAttribute(attn_kernel<L>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(C * 64, 1, 1);
  cfg.blockDim = dim3(NTHR, 1, 1);
  cfg.dynamicSmemBytes = sm;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = C;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  int n = 0;
  if (cudaOccupancyMaxActiveClusters(&n, attn_kernel<L>, &cfg) != cudaSuccess) {
    cudaGetLastError();
    n = 0;
  }
  cache[C] = n;
  done[C] = true;
  return n;
}

int muse_func_info(int L, int which) {
  cudaFuncAttributes a;
  cudaError_t e = (L == 1) ? cudaFuncGetAttributes(&a, attn_kernel<1>) : cudaFuncGetAttributes(&a, attn_kernel<8>);
  if (e != cudaSuccess) { cudaGetLastError(); return -1; }
  return which == 0 ? a.numRegs : (int)a.localSizeBytes;
}

int muse_max_clusters(int L, int C) { return L == 1 ? max_clusters_t<1>(C) : max_clusters_t<8>(C); }

template <int L>
static int launch_t(const Params& p, cudaStream_t st) {
  size_t sm = smem_bytes(L);
  max_clusters_t<L>(p.C);  // sets the function attributes once
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(p.C * p.NC, p.Hkv / 2, p.B);
  cfg.blockDim = dim3(NTHR, 1, 1);
  cfg.dynamicSmemBytes = sm;
  cfg.stream = st;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = p.C;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  return (int)cudaLaunchKernelEx(&cfg, attn_kernel<L>, p);
}

int muse_attn_launch(const void* qkv, const void* positions, const void* cache_loc, const void* seq_lens, void* kc,
                     void* vc, const void* page_table, const void* k_scale, const void* v_scale, const void* inv_freq,
                     void* out, void* opart, void* lse, void* counters, int B, int L, int Hq, int Hkv, int ps,
                     int max_pages, int window, int use_rope, int C, int NC, int chunk, int nmax, float scale_log2,
                     int64_t stream) {
  Params p;
  p.qkv = (const uint16_t*)qkv;
  p.positions = (const int64_t*)positions;
  p.cache_loc = (const int64_t*)cache_loc;
  p.seq_lens = (const int*)seq_lens;
  p.kc = (uint8_t*)kc;
  p.vc = (uint8_t*)vc;
  p.page_table = (const int*)page_table;
  p.k_scale = (const float*)k_scale;
  p.v_scale = (const float*)v_scale;
  p.inv_freq = (const float*)inv_freq;
  p.out = (uint16_t*)out;
  p.opart = (float*)opart;
  p.lse = (float*)lse;
  p.counters = (int*)counters;
  p.B = B; p.Hq = Hq; p.Hkv = Hkv; p.W = (2 * Hq + 2 * Hkv) * HD; p.ps = ps; p.max_pages = max_pages;
  p.window = window; p.use_rope = use_rope; p.C = C; p.NC = NC; p.chunk = chunk; p.nmax = nmax;
  p.scale_log2 = scale_log2;
  cudaStream_t st = (cudaStream_t)stream;
  return L == 1 ? launch_t<1>(p, st) : launch_t<8>(p, st);
}
'''

CPP_SRC = r'''
#include <torch/extension.h>
int muse_attn_launch(const void* qkv, const void* positions, const void* cache_loc, const void* seq_lens, void* kc,
                     void* vc, const void* page_table, const void* k_scale, const void* v_scale, const void* inv_freq,
                     void* out, void* opart, void* lse, void* counters, int B, int L, int Hq, int Hkv, int ps,
                     int max_pages, int window, int use_rope, int C, int NC, int chunk, int nmax, float scale_log2,
                     int64_t stream);
int muse_max_clusters(int L, int C);
int muse_func_info(int L, int which);
int64_t func_info(int64_t L, int64_t which) { return muse_func_info((int)L, (int)which); }

int64_t max_clusters(int64_t L, int64_t C) { return muse_max_clusters((int)L, (int)C); }

int64_t run(torch::Tensor qkv, torch::Tensor positions, torch::Tensor cache_loc, torch::Tensor seq_lens,
            torch::Tensor kc, torch::Tensor vc, torch::Tensor page_table, torch::Tensor k_scale, torch::Tensor v_scale,
            torch::Tensor inv_freq, torch::Tensor out, torch::Tensor opart, torch::Tensor lse, torch::Tensor counters,
            int64_t B, int64_t L, int64_t Hq, int64_t Hkv, int64_t ps, int64_t max_pages, int64_t window,
            int64_t use_rope, int64_t C, int64_t NC, int64_t chunk, int64_t nmax, double scale_log2, int64_t stream) {
  return muse_attn_launch(qkv.data_ptr(), positions.data_ptr(), cache_loc.data_ptr(), seq_lens.data_ptr(),
                          kc.data_ptr(), vc.data_ptr(), page_table.data_ptr(), k_scale.data_ptr(), v_scale.data_ptr(),
                          inv_freq.data_ptr(), out.data_ptr(), opart.data_ptr(), lse.data_ptr(), counters.data_ptr(),
                          (int)B, (int)L, (int)Hq, (int)Hkv, (int)ps, (int)max_pages, (int)window, (int)use_rope,
                          (int)C, (int)NC, (int)chunk, (int)nmax, (float)scale_log2, stream);
}
'''

_EXT = load_inline(
    name="muse_attn_decode_v2",
    cpp_sources=[CPP_SRC],
    cuda_sources=[CUDA_SRC],
    functions=["run", "max_clusters", "func_info"],
    extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"],
    extra_cflags=["-O3"],
    verbose=False,
)

NUM_SMS = 132
_INV_FREQ = {}
_COUNTERS = {}
_PLANS = {}


def _inv_freq(theta, device):
    key = (float(theta), str(device))
    t = _INV_FREQ.get(key)
    if t is None:  # exactly the reference's formula (a function of the config, not the inputs)
        t = (1.0 / (theta ** (torch.arange(0, 64, 2, device=device).float() / 64))).contiguous()
        _INV_FREQ[key] = t
    return t


def _counters(device):
    """Arrival counters for the cross-cluster merge; the last arriver resets its counter to 0."""
    key = str(device)
    t = _COUNTERS.get(key)
    if t is None:
        t = _COUNTERS[key] = torch.zeros(1 << 16, dtype=torch.int32, device=device)
    return t


def _plan(batch, length, hkv, nmax):
    key = (batch, length, hkv, nmax)
    if key in _PLANS:
        return _PLANS[key]
    pairs = batch * (hkv // 2)
    splits = max(1, (2 * NUM_SMS) // pairs)
    splits = min(splits, max(1, nmax // 128))
    c = 16 if splits >= 16 else 8 if splits >= 8 else 4 if splits >= 4 else 2
    nc = max(1, splits // c)
    while c > 2 and _EXT.max_clusters(length, c) < nc * pairs:
        c //= 2
        nc *= 2
    chunk = -(-nmax // (c * nc))
    chunk = ((chunk + 63) // 64) * 64
    _PLANS[key] = (c, nc, chunk)
    print(f"[plan] B={batch} L={length} hkv={hkv} nmax={nmax} -> C={c} NC={nc} chunk={chunk} "
          f"maxclusters(C)={_EXT.max_clusters(length, c)} regs={_EXT.func_info(length, 0)} "
          f"local={_EXT.func_info(length, 1)}", flush=True)
    return c, nc, chunk


def _rms(x, eps=1e-5):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _rope(x, pos, theta):
    d = x.shape[-1]
    inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, device=x.device).float() / d))
    ang = pos.float()[..., None] * inv_freq
    cos, sin = ang.cos().unsqueeze(-2), ang.sin().unsqueeze(-2)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack([even * cos - odd * sin, even * sin + odd * cos], -1).flatten(-2)


def _torch_path(data):
    """Plain PyTorch fallback for configurations the CUDA kernel does not cover."""
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d, page = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"], cfg["page_size"]
    q_in, k_in, v_in, gate = data["qkv"].split([hq * d, hkv * d, hkv * d, hq * d], -1)
    pos = data["positions"].view(batch, length)
    q = _rms(q_in.reshape(batch, length, hq, d)) * cfg["qk_scale_factor"]
    k = _rms(k_in.reshape(batch, length, hkv, d))
    if cfg["rope_theta"]:
        q, k = _rope(q, pos, cfg["rope_theta"]), _rope(k, pos, cfg["rope_theta"])
    q = q.to(torch.bfloat16)
    table = data["page_table"].long()
    slots = data["cache_loc"].view(batch, length)
    k_cache = data["k_cache"].view(-1, hkv, d)
    v_cache = data["v_cache"].view(-1, hkv, d)
    k_scale, v_scale = data["k_scale"], data["v_scale"]
    k_cache[slots] = (k / k_scale).to(torch.float8_e4m3fn)
    v_cache[slots] = (v_in.reshape(batch, length, hkv, d).float() / v_scale).to(torch.float8_e4m3fn)
    t = torch.arange(table.shape[1] * page, device=q.device)
    out = torch.empty((batch, length, hq, d), dtype=torch.bfloat16, device=q.device)
    for b in range(batch):
        rows = table[b].clamp(min=0)[t // page] * page + t % page
        keys = k_cache[rows].float() * k_scale
        vals = v_cache[rows].float() * v_scale
        visible = t[None, :] <= pos[b][:, None]
        if cfg["window"]:
            visible &= t[None, :] > pos[b][:, None] - cfg["window"]
        att = torch.einsum("lkgd,skd->lkgs", q[b].view(length, hkv, hq // hkv, d).float(), keys) * cfg["sm_scale"]
        att = att.masked_fill(~visible[:, None, None, :], float("-inf")).softmax(-1)
        out[b] = torch.einsum("lkgs,skd->lkgd", att, vals).reshape(length, hq, d)
    out = _rms(out) * torch.sigmoid(gate.float()).view(batch, length, hq, d)
    return out.reshape(batch, length, hq * d).to(torch.bfloat16)


def kernel_fn(data: Inputs) -> torch.Tensor:
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d, page = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"], cfg["page_size"]
    if d != 64 or hq != 8 * hkv or hkv % 2 or length not in (1, 8):
        return _torch_path(data)
    qkv = data["qkv"]
    device = qkv.device
    page_table = data["page_table"]
    max_pages = page_table.shape[1]
    window = cfg["window"]
    maxlen = max_pages * page
    nmax = min(window + length - 1, maxlen) if window else maxlen
    c, nc, chunk = _plan(batch, length, hkv, nmax)
    out = torch.empty((batch, length, hq * d), dtype=torch.bfloat16, device=device)
    rows = 16 * length
    if nc > 1:
        opart = torch.empty((batch * (hkv // 2) * nc * rows * 64,), dtype=torch.float32, device=device)
        lse = torch.empty((batch * (hkv // 2) * nc * rows,), dtype=torch.float32, device=device)
    else:
        opart = lse = out
    theta = cfg["rope_theta"]
    inv_freq = _inv_freq(theta, device) if theta else data["k_scale"]
    scale_log2 = cfg["qk_scale_factor"] * cfg["sm_scale"] * math.log2(math.e)
    err = _EXT.run(qkv, data["positions"], data["cache_loc"], data["seq_lens"], data["k_cache"], data["v_cache"],
                   page_table, data["k_scale"], data["v_scale"], inv_freq, out, opart, lse, _counters(device),
                   batch, length, hq, hkv, page, max_pages, window, 1 if theta else 0, c, nc, chunk, nmax,
                   scale_log2, torch.cuda.current_stream(device).cuda_stream)
    if err:
        raise RuntimeError(f"attention launch failed: cuda error {err} (C={c}, NC={nc}, chunk={chunk})")
    return out
