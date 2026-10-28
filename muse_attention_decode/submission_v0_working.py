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
// Fused gated-attention decode for Muse Spark Open (H200 / sm_90a).
// One launch does: QK-RMSNorm (+RoPE), FP8 cache write of the new K/V, split-KV attention over
// the paged FP8 cache with GQA packing (8 query heads x L tokens share each K/V tile), and either
// the final RMSNorm x sigmoid(gate) epilogue (1 split) or a partial (O, lse) for the combine kernel.
#include <cuda_runtime.h>
#include <stdint.h>

#define HD 64
#define TILE 64
#define NST 4
#define MAXC 4096
#define NTHR 128
#define RMS_EPS 1e-5f

struct Params {
  const uint16_t* qkv;        // bf16 [T, W]
  const int64_t* positions;   // [T]
  const int64_t* cache_loc;   // [T]
  const int* seq_lens;        // [B]
  uint8_t* kc;                // fp8 [rows, Hkv, 64]
  uint8_t* vc;
  const int* page_table;      // [B, max_pages]
  const float* k_scale;
  const float* v_scale;
  const float* inv_freq;      // [32]
  uint16_t* out;              // bf16 [T, Hq*64]
  float* opart;               // [B, NHP, S, 2*ROWS, 64]
  float* lse;                 // [B, NHP, S, 2*ROWS]
  int B, Hq, Hkv, W, ps, max_pages, window, use_rope, S, chunk, nmax;
  float scale_log2;           // qk_factor * sm_scale * log2(e)  (k_scale applied on device)
};

__device__ __forceinline__ float bf2f(uint16_t x) { return __uint_as_float(((uint32_t)x) << 16); }
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
// 4 e4m3 bytes -> two f16x2 (bytes 0,1 -> lo ; bytes 2,3 -> hi)
__device__ __forceinline__ void e4m3x4_to_f16(uint32_t x, uint32_t& lo, uint32_t& hi) {
  asm("{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, %2;\n\tcvt.rn.f16x2.e4m3x2 %0, l;\n\tcvt.rn.f16x2.e4m3x2 %1, h;\n\t}"
      : "=r"(lo), "=r"(hi) : "r"(x));
}
__device__ __forceinline__ float ex2(float x) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp16(void* sm, const void* g, bool valid) {
  uint32_t s = (uint32_t)__cvta_generic_to_shared(sm);
  int n = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(g), "r"(n));
}
__device__ __forceinline__ void cp4(void* sm, const void* g) {
  uint32_t s = (uint32_t)__cvta_generic_to_shared(sm);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(s), "l"(g));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// 16-byte chunk swizzle inside a 128-byte K/V row (both kv heads of the CTA).
__device__ __forceinline__ int swz(int key) { return ((key ^ (key >> 2)) & 1) << 2; }

template <int L>
__global__ void __launch_bounds__(NTHR, 2) attn_kernel(const Params p) {
  constexpr int ROWS = 8 * L;              // query rows per kv head (token-major, 8 heads each)
  constexpr int MT = (L == 1) ? 1 : 2;     // m16 tiles per warp
  constexpr int KW = (L == 1) ? 32 : 64;   // keys per warp per tile
  constexpr int NT = KW / 8;
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* sK = smem;
  uint8_t* sV = sK + NST * TILE * 128;
  int* sIdx = (int*)(sV + NST * TILE * 128);
  uint16_t* sQ = (uint16_t*)(sIdx + MAXC + 8);
  uint8_t* sNew = (uint8_t*)(sQ + 2 * ROWS * HD);
  uint16_t* sGate = (uint16_t*)(sNew + 2 * L * 128);
  float* sRed = (float*)smem;                       // aliases the K/V stages after the loop
  float* sML = sRed + 4 * 16 * MT * HD;

  const int s = blockIdx.x, hp = blockIdx.y, b = blockIdx.z;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  const int Hkv = p.Hkv, ps = p.ps;
  const int tok0 = b * L;
  const bool slide = p.window > 0;
  const int NHP = Hkv >> 1;

  // ---- 1. page-table prefetch (full layers: chunk is absolute, no need to wait for seq_lens)
  int pf_first = 0;
  if (!slide) {
    int kb0 = s * p.chunk, ke0 = min(kb0 + p.chunk, p.nmax);
    if (kb0 < ke0) {
      pf_first = kb0 / ps;
      int n = (ke0 - 1) / ps - pf_first + 1;
      const int* src = p.page_table + (size_t)b * p.max_pages + pf_first;
      for (int i = tid; i < n; i += NTHR) cp4(&sIdx[i], src + i);
    }
  }
  if (p.S == 1) {  // gate prefetch for the fused epilogue
    for (int i = tid; i < 2 * ROWS * 8; i += NTHR) {
      int rc = i >> 3, c = i & 7;
      int h = rc / ROWS, r = rc % ROWS, j = r >> 3, gg = r & 7;
      int H = (hp * 2 + h) * 8 + gg;
      const uint16_t* src = p.qkv + (size_t)(tok0 + j) * p.W + (p.Hq + 2 * Hkv) * HD + H * HD + c * 8;
      cp16(sGate + rc * HD + c * 8, src, true);
    }
  }
  cp_commit();

  const int seq = p.seq_lens[b];
  const int end = seq + L;
  const int base = slide ? max(0, seq - p.window + 1) : 0;
  const int kb = base + s * p.chunk;
  const int ke = min(kb + p.chunk, end);
  if (slide && kb < ke) {
    pf_first = kb / ps;
    int n = (ke - 1) / ps - pf_first + 1;
    const int* src = p.page_table + (size_t)b * p.max_pages + pf_first;
    for (int i = tid; i < n; i += NTHR) cp4(&sIdx[i], src + i);
  }
  cp_commit();

  // ---- 2. queries: RMSNorm (+RoPE) -> f16 in smem.  q_eff = q_f16 * qk_factor (folded in scale)
  if (tid < 2 * ROWS) {
    int rc = tid, h = rc / ROWS, r = rc % ROWS, j = r >> 3, gg = r & 7;
    int H = (hp * 2 + h) * 8 + gg;
    const uint4* src = (const uint4*)(p.qkv + (size_t)(tok0 + j) * p.W + H * HD);
    float x[HD];
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      uint4 u = src[c];
      uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        x[c * 8 + 2 * e] = bf2f((uint16_t)(w[e] & 0xffff));
        x[c * 8 + 2 * e + 1] = bf2f((uint16_t)(w[e] >> 16));
      }
    }
    float ss = 0.f;
#pragma unroll
    for (int d = 0; d < HD; ++d) ss += x[d] * x[d];
    float inv = rsqrtf(ss * (1.f / HD) + RMS_EPS);
#pragma unroll
    for (int d = 0; d < HD; ++d) x[d] *= inv;
    if (p.use_rope) {
      float pos = (float)p.positions[tok0 + j];
#pragma unroll
      for (int i = 0; i < HD / 2; ++i) {
        float sn, cs;
        sincosf(pos * p.inv_freq[i], &sn, &cs);
        float x0 = x[2 * i], x1 = x[2 * i + 1];
        x[2 * i] = x0 * cs - x1 * sn;
        x[2 * i + 1] = x0 * sn + x1 * cs;
      }
    }
    uint32_t* dst = (uint32_t*)(sQ + rc * HD);
#pragma unroll
    for (int i = 0; i < HD / 2; ++i) dst[i] = pack_f16(x[2 * i], x[2 * i + 1]);
  }

  // ---- 3. new K/V of the block (only CTAs whose range holds them): norm, rope, FP8, cache write
  const bool has_new = (kb < ke) && (ke > seq);
  if (has_new && tid < 4 * L) {
    const bool isv = tid >= 2 * L;
    int idx = isv ? tid - 2 * L : tid;
    int j = idx >> 1, h = idx & 1;
    int kvh = hp * 2 + h;
    const uint4* src = (const uint4*)(p.qkv + (size_t)(tok0 + j) * p.W + (p.Hq + (isv ? Hkv : 0) + kvh) * HD);
    float x[HD];
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      uint4 u = src[c];
      uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        x[c * 8 + 2 * e] = bf2f((uint16_t)(w[e] & 0xffff));
        x[c * 8 + 2 * e + 1] = bf2f((uint16_t)(w[e] >> 16));
      }
    }
    float qs;
    if (!isv) {
      float ss = 0.f;
#pragma unroll
      for (int d = 0; d < HD; ++d) ss += x[d] * x[d];
      float inv = rsqrtf(ss * (1.f / HD) + RMS_EPS);
#pragma unroll
      for (int d = 0; d < HD; ++d) x[d] *= inv;
      if (p.use_rope) {
        float pos = (float)p.positions[tok0 + j];
#pragma unroll
        for (int i = 0; i < HD / 2; ++i) {
          float sn, cs;
          sincosf(pos * p.inv_freq[i], &sn, &cs);
          float x0 = x[2 * i], x1 = x[2 * i + 1];
          x[2 * i] = x0 * cs - x1 * sn;
          x[2 * i + 1] = x0 * sn + x1 * cs;
        }
      }
      qs = 1.f / *p.k_scale;
    } else {
      qs = 1.f / *p.v_scale;
    }
    uint32_t w[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      uint32_t lo = f2e4m3x2(x[4 * i] * qs, x[4 * i + 1] * qs);
      uint32_t hi = f2e4m3x2(x[4 * i + 2] * qs, x[4 * i + 3] * qs);
      w[i] = lo | (hi << 16);
    }
    uint4* sd = (uint4*)(sNew + ((isv ? L : 0) + j) * 128 + h * HD);
#pragma unroll
    for (int c = 0; c < 4; ++c) sd[c] = make_uint4(w[4 * c], w[4 * c + 1], w[4 * c + 2], w[4 * c + 3]);
    int pos = seq + j;
    if (pos >= kb && pos < ke) {
      int64_t row = p.cache_loc[tok0 + j];
      uint4* gd = (uint4*)((isv ? p.vc : p.kc) + ((size_t)row * Hkv + kvh) * HD);
#pragma unroll
      for (int c = 0; c < 4; ++c) gd[c] = make_uint4(w[4 * c], w[4 * c + 1], w[4 * c + 2], w[4 * c + 3]);
    }
  }
  cp_wait<0>();
  __syncthreads();

  // ---- 4. Q A-fragments (d permuted so each lane reads 16 contiguous K bytes per key)
  const int h = (L == 1) ? (warp & 1) : (warp >> 1);
  const int keyoff = (L == 1) ? (warp >> 1) * 32 : 0;
  const int mtb = (L == 1) ? 0 : (warp & 1) * 2;
  uint32_t qa[MT][4][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    int r0 = (mtb + mt) * 16 + g, r1 = r0 + 8;
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      int d0 = 16 * t + 4 * ks;
      const uint32_t* q0 = (const uint32_t*)(sQ + (h * ROWS + r0) * HD + d0);
      const uint32_t* q1 = (const uint32_t*)(sQ + (h * ROWS + r1) * HD + d0);
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
    uint8_t* bk = sK + st * TILE * 128;
    uint8_t* bv = sV + st * TILE * 128;
#pragma unroll
    for (int r = 0; r < (TILE * 8) / NTHR; ++r) {
      int ci = tid + r * NTHR;
      int key = ci >> 3, c = ci & 7;
      int pos = t0 + key;
      int so = key * 128 + ((c ^ swz(key)) << 4);
      if (pos < ke && pos < seq) {
        int pi = pos / ps;
        int row = sIdx[pi - pf_first] * ps + (pos - pi * ps);
        size_t off = (size_t)row * Hkv * HD + headoff + c * 16;
        cp16(bk + so, p.kc + off, true);
        cp16(bv + so, p.vc + off, true);
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
      uint32_t kw[4] = {kr.x, kr.y, kr.z, kr.w};
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        uint32_t b0, b1;
        e4m3x4_to_f16(kw[ks], b0, b1);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) mma16816(sc[mt][nt], qa[mt][ks], b0, b1);
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

#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        if (L == 1 && hf == 1) {
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) sc[mt][nt][2] = sc[mt][nt][3] = 0.f;
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
        float ls = 0.f;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          float p0 = ex2(sc[mt][nt][2 * hf] - mu);
          float p1 = ex2(sc[mt][nt][2 * hf + 1] - mu);
          sc[mt][nt][2 * hf] = p0;
          sc[mt][nt][2 * hf + 1] = p1;
          ls += p0 + p1;
        }
        l[mt][hf] += ls;
      }
    }

#pragma unroll
    for (int ks = 0; ks < NT / 2; ++ks) {
      uint32_t pa[MT][4];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        pa[mt][0] = pack_f16(sc[mt][2 * ks][0], sc[mt][2 * ks][1]);
        pa[mt][1] = pack_f16(sc[mt][2 * ks][2], sc[mt][2 * ks][3]);
        pa[mt][2] = pack_f16(sc[mt][2 * ks + 1][0], sc[mt][2 * ks + 1][1]);
        pa[mt][3] = pack_f16(sc[mt][2 * ks + 1][2], sc[mt][2 * ks + 1][3]);
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
          uint32_t ab = __byte_perm(a, bb, sel);
          uint32_t cd = __byte_perm(c, dd, sel);
          uint32_t ab0, ab1, cd0, cd1;
          e4m3x4_to_f16(ab, ab0, ab1);
          e4m3x4_to_f16(cd, cd0, cd1);
          const int i = half * 4 + q2 * 2;
#pragma unroll
          for (int mt = 0; mt < MT; ++mt) {
            mma16816(o[mt][i], pa[mt], ab0, cd0);
            mma16816(o[mt][i + 1], pa[mt], ab1, cd1);
          }
        }
      }
    }
  }

  cp_wait<0>();
  __syncthreads();

  // ---- 5. per-warp state -> smem, then merge and finish per row
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      if (L == 1 && hf == 1) continue;
      float lv = l[mt][hf];
      lv += __shfl_xor_sync(0xffffffff, lv, 1);
      lv += __shfl_xor_sync(0xffffffff, lv, 2);
      int lr = mt * 16 + hf * 8 + g;
      float* dst = sRed + (warp * 16 * MT + lr) * HD;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        dst[16 * t + i] = o[mt][i][2 * hf];
        dst[16 * t + 8 + i] = o[mt][i][2 * hf + 1];
      }
      if (t == 0) {
        sML[(warp * 16 * MT + lr) * 2] = m[mt][hf];
        sML[(warp * 16 * MT + lr) * 2 + 1] = lv;
      }
    }
  __syncthreads();

  const float vscale = *p.v_scale;
  for (int rc = warp; rc < 2 * ROWS; rc += 4) {
    const int hh = rc / ROWS, r = rc % ROWS, j = r >> 3, gg = r & 7;
    int wi[2], li[2], nc;
    if (L == 1) {
      wi[0] = hh; wi[1] = hh + 2; li[0] = li[1] = r; nc = 2;
    } else {
      int mtg = r >> 4;
      wi[0] = hh * 2 + (mtg >> 1); li[0] = (mtg & 1) * 16 + (r & 15); nc = 1;
      wi[1] = wi[0]; li[1] = li[0];
    }
    float M = -INFINITY;
    for (int c = 0; c < nc; ++c) M = fmaxf(M, sML[(wi[c] * 16 * MT + li[c]) * 2]);
    float a0 = 0.f, a1 = 0.f, den = 0.f;
    if (M != -INFINITY) {
      for (int c = 0; c < nc; ++c) {
        int k = wi[c] * 16 * MT + li[c];
        float w = ex2(sML[k * 2] - M);
        den += w * sML[k * 2 + 1];
        a0 += w * sRed[k * HD + 2 * lane];
        a1 += w * sRed[k * HD + 2 * lane + 1];
      }
    }
    float inv = (den > 0.f) ? vscale / den : 0.f;
    a0 *= inv;
    a1 *= inv;
    const int H = (hp * 2 + hh) * 8 + gg;
    if (p.S == 1) {
      float ss = a0 * a0 + a1 * a1;
#pragma unroll
      for (int o2 = 16; o2 > 0; o2 >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o2);
      float rn = rsqrtf(ss * (1.f / HD) + RMS_EPS);
      uint32_t gw = *(const uint32_t*)(sGate + rc * HD + 2 * lane);
      float g0 = bf2f((uint16_t)(gw & 0xffff)), g1 = bf2f((uint16_t)(gw >> 16));
      float y0 = a0 * rn / (1.f + __expf(-g0));
      float y1 = a1 * rn / (1.f + __expf(-g1));
      uint32_t ov = (uint32_t)f2bf(y0) | ((uint32_t)f2bf(y1) << 16);
      *(uint32_t*)(p.out + (size_t)(tok0 + j) * p.Hq * HD + H * HD + 2 * lane) = ov;
    } else {
      size_t pi = ((size_t)(b * NHP + hp) * p.S + s) * (2 * ROWS) + rc;
      *(float2*)(p.opart + pi * HD + 2 * lane) = make_float2(a0, a1);
      if (lane == 0) p.lse[pi] = (den > 0.f) ? M + __log2f(den) : -INFINITY;
    }
  }
}

__global__ void __launch_bounds__(128) combine_kernel(const float* __restrict__ opart, const float* __restrict__ lse,
                                                      const uint16_t* __restrict__ qkv, uint16_t* __restrict__ out,
                                                      int B, int L, int Hq, int Hkv, int W, int S) {
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (row >= B * L * Hq) return;
  const int H = row % Hq, tk = row / Hq, b = tk / L, j = tk % L;
  const int kvh = H >> 3, hp = kvh >> 1, h = kvh & 1, gg = H & 7;
  const int ROWS = 8 * L, NHP = Hkv >> 1;
  const int rc = h * ROWS + j * 8 + gg;
  const size_t base = ((size_t)(b * NHP + hp) * S) * (2 * ROWS) + rc;
  const size_t stride = 2 * ROWS;
  float M = -INFINITY;
  for (int s = lane; s < S; s += 32) M = fmaxf(M, lse[base + s * stride]);
#pragma unroll
  for (int o2 = 16; o2 > 0; o2 >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffff, M, o2));
  float a0 = 0.f, a1 = 0.f, den = 0.f;
#pragma unroll 4
  for (int s = 0; s < S; ++s) {
    float ls = lse[base + s * stride];
    float w = ex2(ls - M);
    float2 v = *(const float2*)(opart + (base + s * stride) * HD + 2 * lane);
    den += w;
    a0 += w * v.x;
    a1 += w * v.y;
  }
  a0 /= den;
  a1 /= den;
  float ss = a0 * a0 + a1 * a1;
#pragma unroll
  for (int o2 = 16; o2 > 0; o2 >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o2);
  float rn = rsqrtf(ss * (1.f / HD) + RMS_EPS);
  uint32_t gw = *(const uint32_t*)(qkv + (size_t)tk * W + (Hq + 2 * Hkv) * HD + H * HD + 2 * lane);
  float g0 = bf2f((uint16_t)(gw & 0xffff)), g1 = bf2f((uint16_t)(gw >> 16));
  float y0 = a0 * rn / (1.f + __expf(-g0));
  float y1 = a1 * rn / (1.f + __expf(-g1));
  *(uint32_t*)(out + (size_t)tk * Hq * HD + H * HD + 2 * lane) = (uint32_t)f2bf(y0) | ((uint32_t)f2bf(y1) << 16);
}

static size_t smem_bytes(int L, int S) {
  int ROWS = 8 * L;
  size_t n = (size_t)2 * NST * TILE * 128 + (MAXC + 8) * 4 + 2 * ROWS * HD * 2 + 2 * L * 128;
  if (S == 1) n += 2 * ROWS * HD * 2;
  return n;
}

template <int L>
static void launch_t(const Params& p, cudaStream_t st) {
  static bool init = false;
  size_t sm = smem_bytes(L, p.S);
  if (!init) {
    cudaFuncSetAttribute(attn_kernel<L>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes(L, 1));
    init = true;
  }
  dim3 grid(p.S, p.Hkv / 2, p.B);
  attn_kernel<L><<<grid, NTHR, sm, st>>>(p);
}

void muse_attn_launch(const void* qkv, const void* positions, const void* cache_loc, const void* seq_lens, void* kc,
                      void* vc, const void* page_table, const void* k_scale, const void* v_scale, const void* inv_freq,
                      void* out, void* opart, void* lse, int B, int L, int Hq, int Hkv, int ps, int max_pages,
                      int window, int use_rope, int S, int chunk, int nmax, float scale_log2, int64_t stream) {
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
  p.B = B; p.Hq = Hq; p.Hkv = Hkv; p.W = (2 * Hq + 2 * Hkv) * HD; p.ps = ps; p.max_pages = max_pages;
  p.window = window; p.use_rope = use_rope; p.S = S; p.chunk = chunk; p.nmax = nmax; p.scale_log2 = scale_log2;
  cudaStream_t st = (cudaStream_t)stream;
  if (L == 1) launch_t<1>(p, st);
  else launch_t<8>(p, st);
  if (S > 1) {
    int rows = B * L * Hq;
    combine_kernel<<<(rows + 3) / 4, 128, 0, st>>>(p.opart, p.lse, p.qkv, p.out, B, L, Hq, Hkv, p.W, S);
  }
}
'''

CPP_SRC = r'''
#include <torch/extension.h>
void muse_attn_launch(const void* qkv, const void* positions, const void* cache_loc, const void* seq_lens, void* kc,
                      void* vc, const void* page_table, const void* k_scale, const void* v_scale, const void* inv_freq,
                      void* out, void* opart, void* lse, int B, int L, int Hq, int Hkv, int ps, int max_pages,
                      int window, int use_rope, int S, int chunk, int nmax, float scale_log2, int64_t stream);

void run(torch::Tensor qkv, torch::Tensor positions, torch::Tensor cache_loc, torch::Tensor seq_lens,
         torch::Tensor kc, torch::Tensor vc, torch::Tensor page_table, torch::Tensor k_scale, torch::Tensor v_scale,
         torch::Tensor inv_freq, torch::Tensor out, torch::Tensor opart, torch::Tensor lse,
         int64_t B, int64_t L, int64_t Hq, int64_t Hkv, int64_t ps, int64_t max_pages, int64_t window,
         int64_t use_rope, int64_t S, int64_t chunk, int64_t nmax, double scale_log2, int64_t stream) {
  muse_attn_launch(qkv.data_ptr(), positions.data_ptr(), cache_loc.data_ptr(), seq_lens.data_ptr(), kc.data_ptr(),
                   vc.data_ptr(), page_table.data_ptr(), k_scale.data_ptr(), v_scale.data_ptr(), inv_freq.data_ptr(),
                   out.data_ptr(), opart.data_ptr(), lse.data_ptr(), (int)B, (int)L, (int)Hq, (int)Hkv, (int)ps,
                   (int)max_pages, (int)window, (int)use_rope, (int)S, (int)chunk, (int)nmax, (float)scale_log2,
                   stream);
}
'''

_EXT = load_inline(
    name="muse_attn_decode_v0",
    cpp_sources=[CPP_SRC],
    cuda_sources=[CUDA_SRC],
    functions=["run"],
    extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"],
    extra_cflags=["-O3"],
    verbose=False,
)

MAXC = 4096          # keys per split, bounded by the page-index buffer in shared memory
NUM_SMS = 132
_INV_FREQ = {}


def _inv_freq(theta, device):
    key = (float(theta), str(device))
    t = _INV_FREQ.get(key)
    if t is None:  # exactly the reference's formula (a function of the config, not the inputs)
        t = 1.0 / (theta ** (torch.arange(0, 64, 2, device=device).float() / 64))
        t = t.contiguous()
        _INV_FREQ[key] = t
    return t


def _plan(batch, length, hkv, nmax):
    pairs = batch * (hkv // 2)
    target = 2 * NUM_SMS
    splits = max(1, target // pairs)
    chunk = -(-nmax // splits)
    chunk = max(128, ((chunk + 63) // 64) * 64)
    chunk = min(chunk, MAXC)
    splits = -(-nmax // chunk)
    return splits, chunk


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
    splits, chunk = _plan(batch, length, hkv, nmax)
    out = torch.empty((batch, length, hq * d), dtype=torch.bfloat16, device=device)
    rows = 16 * length
    if splits > 1:
        opart = torch.empty((batch * (hkv // 2) * splits * rows * 64,), dtype=torch.float32, device=device)
        lse = torch.empty((batch * (hkv // 2) * splits * rows,), dtype=torch.float32, device=device)
    else:
        opart = lse = out
    theta = cfg["rope_theta"]
    inv_freq = _inv_freq(theta if theta else 10000.0, device)
    scale_log2 = cfg["qk_scale_factor"] * cfg["sm_scale"] * math.log2(math.e)
    _EXT.run(qkv, data["positions"], data["cache_loc"], data["seq_lens"], data["k_cache"], data["v_cache"],
             page_table, data["k_scale"], data["v_scale"], inv_freq, out, opart, lse,
             batch, length, hq, hkv, page, max_pages, window, 1 if theta else 0, splits, chunk, nmax,
             scale_log2, torch.cuda.current_stream(device).cuda_stream)
    return out
