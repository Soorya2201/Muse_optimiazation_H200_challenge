"""One-rank Muse-Spark MoE harness: stock NVFP4 marlin path vs an fp32 reference.

Builds random NVFP4 experts shaped like one EP8 rank (32 of 256 experts,
hidden 4096, expert intermediate 4096, top-8), repacks them with SGLang's own
prepare_moe_nvfp4_layer_for_marlin, and runs a copy of
MuseSpark1xFp4Experts._marlin_expert_outputs. Candidate kernels plug in via
`candidates` and are checked against the same reference and timed the same way.

Run on the pod:  python harness.py [--tokens 1,8,64,...] [--check-only]
"""

import argparse
import json
import time
from types import SimpleNamespace

import torch

HIDDEN = 4096
INTER = 4096
N_EXPERTS = 256
LOCAL_EXPERTS = 32
LOCAL_START = 0
TOP_K = 8
GROUP = 16
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)  # muse_spark_fp4_moe.py

FP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
)


# ---------------------------------------------------------------- weights
def make_raw_weights(seed: int, device: str) -> dict:
    """HF-layout NVFP4 tensors, as create_weights allocates them."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n, h, i = LOCAL_EXPERTS, HIDDEN, INTER

    def fp4(shape):
        return torch.randint(0, 256, shape, generator=g, dtype=torch.uint8)

    def block_scale(shape):
        s = 0.5 + torch.rand(shape, generator=g)  # [0.5, 1.5)
        return s.to(torch.float8_e4m3fn)

    # Same global scale for gate and up columns: marlin keeps only the gate one,
    # and the post-expert norm cancels any ratio in the real model.
    s13 = 0.01 + 0.01 * torch.rand((n, 1), generator=g)
    raw = {
        "w13_weight": fp4((n, 2 * i, h // 2)),
        "w2_weight": fp4((n, h, i // 2)),
        "w13_weight_scale": block_scale((n, 2 * i, h // GROUP)),
        "w2_weight_scale": block_scale((n, h, i // GROUP)),
        "w13_weight_scale_2": s13.expand(n, 2).contiguous(),
        "w2_weight_scale_2": 0.01 + 0.01 * torch.rand((n,), generator=g),
    }
    return {k: v.to(device) for k, v in raw.items()}


def dequant(q: torch.Tensor, scale: torch.Tensor, scale_2: torch.Tensor) -> torch.Tensor:
    """[E, N, K/2] uint8 (low nibble = even k) -> [E, N, K] fp32."""
    table = FP4_VALUES.to(q.device)
    lo = table[(q & 0xF).long()]
    hi = table[(q >> 4).long()]
    w = torch.stack((lo, hi), dim=-1).flatten(-2)
    s = scale.float().repeat_interleave(GROUP, dim=-1)
    return w * s * scale_2.float().view(-1, 1, 1)


def build_marlin_layer(raw: dict):
    from sglang.srt.layers.quantization.marlin_utils_fp4 import (
        prepare_moe_nvfp4_layer_for_marlin,
    )

    layer = torch.nn.Module()
    for k, v in raw.items():
        t = v[:, 0].contiguous() if k == "w13_weight_scale_2" else v
        layer.register_parameter(k, torch.nn.Parameter(t.clone(), requires_grad=False))
    layer.quant_config = SimpleNamespace(group_size=GROUP)
    layer.moe_runner_config = SimpleNamespace(is_gated=True)
    layer.intermediate_size_per_partition = INTER
    layer.params_dtype = torch.bfloat16
    prepare_moe_nvfp4_layer_for_marlin(layer)
    expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32)
    expert_map[LOCAL_START:LOCAL_START + LOCAL_EXPERTS] = torch.arange(
        LOCAL_EXPERTS, dtype=torch.int32
    )
    layer.expert_map = expert_map.to(raw["w13_weight"].device)
    return layer


# ---------------------------------------------------------------- marlin
def marlin_expert_outputs(layer, hidden_states, topk_scores, topk_ids,
                          block_size_m=None, use_atomic_add=False, use_fp32_reduce=True):
    """Copy of MuseSpark1xFp4Experts._marlin_expert_outputs (EP path).

    The keyword overrides exist only for tuning sweeps; defaults are stock.
    """
    from sgl_kernel.scalar_type import scalar_types

    from sglang.kernels.ops.activation.activation import silu_and_mul
    from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace

    hidden_states = hidden_states.contiguous()
    num_tokens, size_k = hidden_states.shape
    topk = topk_ids.shape[1]
    size_n = INTER
    gemm1_n = 2 * size_n
    expert_map = layer.expert_map

    local_topk_ids = expert_map[topk_ids.to(torch.int32)].contiguous()
    topk_weights = topk_scores.to(torch.float32).contiguous()

    if block_size_m == "ep":
        # Proposed fix: under EP every rank sees all tokens, so a local expert
        # expects num_tokens * topk / global_num_experts routes.
        for block_size_m in BLOCK_SIZE_M_CANDIDATES:
            if num_tokens * topk / N_EXPERTS / block_size_m < 0.9:
                break
    elif block_size_m is None:
        for block_size_m in BLOCK_SIZE_M_CANDIDATES:
            if num_tokens * topk / LOCAL_EXPERTS / block_size_m < 0.9:
                break
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        local_topk_ids, block_size_m, N_EXPERTS
    )
    workspace = marlin_make_workspace(hidden_states.device, max_blocks_per_sm=4)
    cache2 = torch.empty((num_tokens * topk, size_n), device=hidden_states.device,
                         dtype=hidden_states.dtype)
    cache13 = torch.empty((num_tokens * topk * max(gemm1_n, size_k),),
                          device=hidden_states.device, dtype=hidden_states.dtype)
    cache1 = cache13[: num_tokens * topk * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * topk * size_k].view(-1, size_k)

    common = dict(
        mul_topk_weights=False, is_ep=True, b_q_type=scalar_types.float4_e2m1f,
        is_k_full=True, use_atomic_add=use_atomic_add, use_fp32_reduce=use_fp32_reduce,
        is_zp_float=False,
    )
    cache1 = moe_wna16_marlin_gemm(
        hidden_states, cache1, layer.w13_weight, None, layer.w13_weight_scale,
        layer.w13_weight_scale_2, None, None, None, workspace, sorted_token_ids,
        expert_ids, num_tokens_post_padded, topk_weights, moe_block_size=block_size_m,
        top_k=topk, size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common,
    )
    silu_and_mul(cache1.view(-1, gemm1_n), cache2,
                 expert_ids=local_topk_ids.view(-1), expert_step=1)
    out = moe_wna16_marlin_gemm(
        cache2, cache3, layer.w2_weight, None, layer.w2_weight_scale,
        layer.w2_weight_scale_2, None, None, None, workspace, sorted_token_ids,
        expert_ids, num_tokens_post_padded, topk_weights, moe_block_size=block_size_m,
        top_k=1, size_m=num_tokens * topk, size_n=size_k, size_k=size_n, **common,
    )
    return out.view(num_tokens, topk, size_k), local_topk_ids


# ---------------------------------------------------------------- reference
class Reference:
    def __init__(self, raw: dict):
        self.w13 = dequant(raw["w13_weight"], raw["w13_weight_scale"],
                           raw["w13_weight_scale_2"][:, 0])
        self.w2 = dequant(raw["w2_weight"], raw["w2_weight_scale"],
                          raw["w2_weight_scale_2"])

    def __call__(self, hidden_states, local_topk_ids):
        """[T, topk, H] fp32; non-local slots are left at zero."""
        x = hidden_states.float()
        t, k = local_topk_ids.shape
        out = torch.zeros((t, k, HIDDEN), device=x.device)
        for e in local_topk_ids.unique().tolist():
            if e < 0:
                continue
            tok, slot = (local_topk_ids == e).nonzero(as_tuple=True)
            h1 = x[tok] @ self.w13[e].T
            gate, up = h1[:, :INTER], h1[:, INTER:]
            # Round the intermediate to bf16 like the kernel path does.
            act = (torch.nn.functional.silu(gate) * up).bfloat16().float()
            out[tok, slot] = act @ self.w2[e].T
        return out


# ---------------------------------------------------------------- inputs
def make_inputs(num_tokens: int, seed: int, device: str):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((num_tokens, HIDDEN), generator=g).bfloat16()
    # Distinct experts per token over all 256 experts, like a real router.
    ids = torch.rand((num_tokens, N_EXPERTS), generator=g).argsort(dim=1)[:, :TOP_K]
    scores = torch.softmax(torch.randn((num_tokens, TOP_K), generator=g), dim=-1)
    return x.to(device), scores.to(device), ids.to(torch.int32).to(device)


def compare(got, ref, local_topk_ids):
    mask = local_topk_ids >= 0
    if not mask.any():
        return {"local_slots": 0}
    g, r = got.float()[mask], ref[mask]
    err = (g - r).abs()
    return {
        "local_slots": int(mask.sum()),
        "max_abs": err.max().item(),
        "rel_l2": (err.norm() / r.norm()).item(),
        "cos": torch.nn.functional.cosine_similarity(g.flatten(), r.flatten(), dim=0).item(),
        "ref_rms": r.pow(2).mean().sqrt().item(),
    }


def time_fn(fn, iters: int) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000.0  # us


def time_graph(fn, iters: int) -> float:
    """Replay a captured CUDA graph, as the server does for decode batches."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return time_fn(graph.replay, iters)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", default="1,2,4,8,16,32,64,128,288,512,1024,2048,4096,8192,16384")
    p.add_argument("--check-tokens", default="1,8,64,288,1024")
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--out", default="baseline.json")
    args = p.parse_args()
    device = "cuda"
    torch.manual_seed(0)

    t0 = time.time()
    raw = make_raw_weights(seed=0, device=device)
    layer = build_marlin_layer(raw)
    ref = Reference(raw)
    del raw
    torch.cuda.synchronize()
    print(f"weights ready in {time.time() - t0:.1f}s, "
          f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB allocated")

    results = {"check": {}, "timing_us": {}}
    for t in [int(s) for s in args.check_tokens.split(",")]:
        x, scores, ids = make_inputs(t, seed=t, device=device)
        got, local_ids = marlin_expert_outputs(layer, x, scores, ids)
        stats = compare(got, ref(x, local_ids), local_ids)
        results["check"][t] = stats
        print(f"check T={t:6d}: {json.dumps(stats)}")

    if not args.check_only:
        for t in [int(s) for s in args.tokens.split(",")]:
            x, scores, ids = make_inputs(t, seed=t, device=device)
            iters = 200 if t <= 1024 else 30
            fn = lambda: marlin_expert_outputs(layer, x, scores, ids)  # noqa: E731
            eager = min(time_fn(fn, iters) for _ in range(3))
            graph = min(time_graph(fn, iters) for _ in range(3)) if t <= 1024 else None
            results["timing_us"][t] = {"eager": eager, "graph": graph}
            g = f"{graph:10.1f}" if graph is not None else "         -"
            print(f"marlin T={t:6d}: eager {eager:10.1f} us   graph {g} us")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
