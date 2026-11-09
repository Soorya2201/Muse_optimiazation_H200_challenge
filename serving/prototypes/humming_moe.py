"""Development-only Muse projection adapter for humming-kernels==0.1.12.

Not integrated into the serving patch. Uses the installed Apache-2.0 Humming
projection API, retaining the caller's existing Muse post-expert finalization.
No Stream-K: the pinned backend reduces its partial outputs in BF16, whereas
stock Muse requests FP32 reduction. No extra canonical weight layout is kept.
"""
from __future__ import annotations

import importlib.metadata
import json

import torch


def stock_outer_scale(scale: torch.Tensor, experts: int, *, gate_up: bool):
    """Match stock's gate-column choice and BF16 rounding before dequantization."""
    if gate_up and scale.ndim == 2:
        assert scale.shape[0] == experts and scale.shape[1] in (1, 2)
        scale = scale[:, 0]
    assert scale.shape == (experts,)
    return scale.to(torch.bfloat16).float().contiguous()


def expected_local_routes(rows: int, topk: int, local: int, global_: int):
    assert rows > 0 and topk > 0 and 0 < local <= global_
    # Shape-only estimate, without copying counts from GPU to the host.
    return max(1, (rows * topk * local + global_ - 1) // global_)


def select_tactic(configs, valid_rows: int):
    for lower, upper, config in configs:
        if lower < valid_rows <= upper:
            assert config.get('use_stream_k') is False
            assert not config.get('use_f16_accum', False)
            return config
    raise ValueError(f'No Humming tactic for {valid_rows} expected local routes')


def prepare_humming_moe(layer):
    """Consume conventional raw w13/w2 NVFP4 parameters, replacing them in place.

    Call before Marlin repacking on a dedicated development fixture. This does
    not adapt native checkpoint aliases or provide a runtime Marlin fallback.
    """
    assert importlib.metadata.version('humming-kernels') == '0.1.12'
    assert layer.params_dtype == torch.bfloat16 and layer.w13_weight.is_cuda
    assert torch.cuda.get_device_capability(layer.w13_weight.device)[0] == 9
    from humming import dtypes
    from humming.config import GemmType
    from humming.layer import HummingMethod
    from humming.schema import HummingInputSchema, HummingWeightSchema

    hidden = layer.w13_weight.shape[-1] * 2
    intermediate = layer.intermediate_size_per_partition
    experts = layer.num_local_experts
    schema = HummingWeightSchema(
        b_dtype=dtypes.float4e2m1, bs_dtype=dtypes.float8e4m3,
        weight_scale_group_size=16, weight_scale_type='group',
        weight_scale_2_type='tensor',
    )
    layer._humming_compute = json.dumps(dict(
        gemm_type='indexed', use_f16_accum=False, use_batch_invariant=True))
    layer._humming_tactics = {}
    for prefix, n, k in [('w13', 2*intermediate, hidden), ('w2', hidden, intermediate)]:
        weight = getattr(layer, prefix+'_weight')
        scales = getattr(layer, prefix+'_weight_scale')
        outer = stock_outer_scale(getattr(layer, prefix+'_weight_scale_2'), experts,
                                  gate_up=prefix == 'w13')
        assert weight.dtype == torch.uint8 and weight.shape == (experts, n, k//2)
        assert scales.dtype == torch.float8_e4m3fn and scales.shape == (experts, n, k//16)
        assert n % 128 == 0 and k % 128 == 0
        # Keep FP8 scales: expanding them to BF16 adds one byte per 16 weights.
        setattr(layer, prefix+'_weight', torch.nn.Parameter(
            weight.contiguous().view(torch.int32), requires_grad=False))
        setattr(layer, prefix+'_weight_scale', torch.nn.Parameter(
            scales.contiguous(), requires_grad=False))
        setattr(layer, prefix+'_weight_scale_2', torch.nn.Parameter(outer, requires_grad=False))
        HummingMethod.prepare_layer_meta(
            layer=layer, shape_n=n, shape_k=k, weight_schema=schema,
            input_schema=HummingInputSchema(a_dtype=dtypes.bfloat16),
            num_experts=experts, torch_dtype=torch.bfloat16, sublayer_name=prefix,
        )
        HummingMethod.transform_humming_layer(layer, sublayer_name=prefix)
        configs = HummingMethod.get_default_tuning_configs(
            layer=layer, gemm_type=GemmType.INDEXED, sublayer_name=prefix,
            use_f16_accum=False, use_batch_invariant=True,
        )
        assert configs and all(config.get('use_stream_k') is False
                               and not config.get('use_f16_accum', False)
                               for _, _, config in configs)
        layer._humming_tactics[prefix] = configs
    return layer


def humming_expert_outputs(self, hidden_states, topk_scores, topk_ids):
    """Projection-only replacement for `_marlin_expert_outputs` in a fixture."""
    from humming.forward import humming_forward
    from sglang.kernels.ops.activation.activation import silu_and_mul
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
    from sglang.srt.layers.moe.token_dispatcher.standard import _map_topk_ids_to_local_experts

    assert hidden_states.dtype == torch.bfloat16 and hidden_states.is_cuda
    hidden_states = hidden_states.contiguous()
    rows, hidden = hidden_states.shape
    topk = topk_ids.shape[1]
    intermediate = self.intermediate_size_per_partition
    local_ids = topk_ids.to(torch.int32)
    if self._expert_map is not None:
        local_ids = _map_topk_ids_to_local_experts(local_ids, self._expert_map)
    local_ids = local_ids.contiguous()
    valid_rows = expected_local_routes(rows, topk, self.num_local_experts, self.global_num_experts)
    aligned = {}
    scratch = torch.empty(rows*topk*max(2*intermediate, hidden),
                          device=hidden_states.device, dtype=hidden_states.dtype)
    gate_up = scratch[:rows*topk*2*intermediate].view(-1, 2*intermediate)
    down = scratch[:rows*topk*hidden].view(-1, hidden)
    activated = torch.empty(rows*topk, intermediate, device=hidden_states.device,
                            dtype=hidden_states.dtype)
    for prefix, inputs, output, top_k in [('w13', hidden_states, gate_up, topk),
                                         ('w2', activated, down, 1)]:
        configs = self._humming_tactics[prefix]
        tactic = select_tactic(configs, valid_rows)
        block = tactic['block_shape'][0]
        if block not in aligned:
            aligned[block] = moe_align_block_size(
                local_ids, block, self.num_local_experts, ignore_invalid_expert=True)
        sorted_ids, expert_ids, padded = aligned[block]
        humming_forward(
            self.humming_metas[prefix], inputs=inputs, outputs=output,
            weight=getattr(self, prefix+'_weight'),
            weight_scale=getattr(self, prefix+'_weight_scale'),
            weight_scale_2=getattr(self, prefix+'_weight_scale_2'),
            sorted_ids=sorted_ids, expert_ids=expert_ids, num_tokens_padded=padded,
            top_k=top_k, valid_shape_m=valid_rows, locks=None,
            compute_config=self._humming_compute, tuning_config=json.dumps(configs),
        )
        if prefix == 'w13':
            silu_and_mul(gate_up, activated, expert_ids=local_ids.reshape(-1), expert_step=1)
    return down.view(rows, topk, hidden), local_ids
