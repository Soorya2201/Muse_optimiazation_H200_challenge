"""CPU adapter protocol doubles; never imports/executes Humming or CUDA code.

Checks exact pinned API signatures, parameter replacement, routing/table reuse,
precision guards, source scale rounding and projection call flow. Numerical
operations stand in for kernels and cannot establish NVFP4/GPU correctness.
"""
import argparse
import ast
import gc
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import weakref

import torch

ROOT = Path(__file__).resolve().parents[1]


def signature(path, function):
    # Preserve required/optional and positional/keyword binding, replacing
    # annotations/default values so no dependency expression is executed.
    tree = ast.parse(path.read_text())
    node = next(x for x in ast.walk(tree) if isinstance(x, ast.FunctionDef) and x.name == function)
    node.decorator_list, node.returns, node.body = [], None, [ast.Pass()]
    for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
        arg.annotation = None
    node.args.defaults = [ast.Constant(None) for _ in node.args.defaults]
    node.args.kw_defaults = [ast.Constant(None) if x is not None else None for x in node.args.kw_defaults]
    env = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), '<signature only>', 'exec'), env)
    return inspect.signature(env[function])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'research/public/humming_0_1_12')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(1)
    path = ROOT/'serving/prototypes/humming_moe.py'
    spec = importlib.util.spec_from_file_location('adapter_cpu_protocol', path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    recorded = {}
    for name, field in [('HUMMING_PINNED_PACKAGE_2026-11-09.json', 'selected'),
                        ('HUMMING_PINNED_API_2026-11-09.json', 'files')]:
        recorded.update({x['path']: x['sha256'] for x in json.loads(
            (ROOT/'research/serving'/name).read_text())[field]})
    for relative in ('humming/layer.py', 'humming/forward.py'):
        assert hashlib.sha256((args.source/relative).read_bytes()).hexdigest() == recorded[relative]
    api = {name: signature(args.source/'humming/layer.py', name) for name in (
        'prepare_layer_meta', 'transform_humming_layer', 'get_default_tuning_configs')}
    api['humming_forward'] = signature(args.source/'humming/forward.py', 'humming_forward')
    calls, tables = [], []
    active = {}

    class CpuTensorWithSimulatedCudaFlag(torch.Tensor):
        @property
        def is_cuda(self):
            return True

    class Method:
        @classmethod
        def prepare_layer_meta(cls, **kw):
            api['prepare_layer_meta'].bind(cls, **kw)
            layer, prefix = kw['layer'], kw['sublayer_name']
            assert kw['weight_schema']['bs_dtype'] == 'fp8' and kw['input_schema']['a_dtype'] == 'bf16'
            if not hasattr(layer, 'humming_metas'):
                layer.humming_metas = {}
            layer.humming_metas[prefix] = SimpleNamespace(prefix=prefix, n=kw['shape_n'])

        @classmethod
        def transform_humming_layer(cls, layer, **kw):
            api['transform_humming_layer'].bind(cls, layer, **kw)
            prefix = kw['sublayer_name']
            for suffix in ('weight', 'weight_scale', 'weight_scale_2'):
                value = getattr(layer, prefix+'_'+suffix)
                setattr(layer, prefix+'_'+suffix, torch.nn.Parameter(value.clone(), requires_grad=False))

        @classmethod
        def get_default_tuning_configs(cls, **kw):
            api['get_default_tuning_configs'].bind(cls, **kw)
            assert kw['use_f16_accum'] is False and kw['use_batch_invariant'] is True
            block = 16 if active['same_block'] or kw['sublayer_name'] == 'w13' else 32
            return [[0, 1 << 30, dict(block_shape=(block, 128, 32), use_stream_k=False,
                                      use_f16_accum=False)]]

    def align(ids, block, experts, *, ignore_invalid_expert):
        assert ignore_invalid_expert and experts == active['local']
        flat = ids.reshape(-1)
        tokens, blocks = [], []
        for expert in range(experts):
            rows = torch.where(flat == expert)[0].tolist()
            if rows:
                rows += [flat.numel()] * ((-len(rows)) % block)
                tokens.extend(rows); blocks.extend([expert]*(len(rows)//block))
        result = (torch.tensor(tokens, dtype=torch.int32), torch.tensor(blocks, dtype=torch.int32),
                  torch.tensor([len(tokens)], dtype=torch.int32))
        tables.append(block)
        return result

    def forward(meta, **kw):
        api['humming_forward'].bind(meta, **kw)
        compute = json.loads(kw['compute_config'])
        assert compute == dict(gemm_type='indexed', use_f16_accum=False, use_batch_invariant=True)
        assert kw['locks'] is None
        assert kw['inputs'].dtype == kw['outputs'].dtype == torch.bfloat16
        assert kw['weight_scale'].dtype == torch.float8_e4m3fn
        config = adapter.select_tactic(json.loads(kw['tuning_config']), kw['valid_shape_m'])
        block, indices = config['block_shape'][0], kw['sorted_ids']
        output, inputs = kw['outputs'], kw['inputs']
        for i, expert in enumerate(kw['expert_ids'].tolist()):
            rows = indices[i*block:(i+1)*block].long()
            rows = rows[rows < output.shape[0]]
            assert (active['local_ids'].reshape(-1)[rows] == expert).all()
            output[rows] = (inputs[rows//kw['top_k'], 0] + expert+1)[:, None]
        calls.append(meta.prefix)
        return output

    def activation(gate_up, out, *, expert_ids, expert_step):
        assert expert_step == 1
        gate, up = gate_up.chunk(2, dim=1)
        mask = expert_ids >= 0
        out[mask] = torch.nn.functional.silu(gate[mask])*up[mask]

    def map_ids(ids, mapping):
        return torch.where(ids < 0, -1, mapping[ids.clamp_min(0)])

    entries = {'humming': {'dtypes': SimpleNamespace(float4e2m1='fp4', bfloat16='bf16', float8e4m3='fp8')},
               'humming.config': {'GemmType': SimpleNamespace(INDEXED='indexed')},
               'humming.layer': {'HummingMethod': Method},
               'humming.schema': {'HummingWeightSchema': lambda **kw: kw, 'HummingInputSchema': lambda **kw: kw},
               'humming.forward': {'humming_forward': forward},
               'sglang.kernels.ops.activation.activation': {'silu_and_mul': activation},
               'sglang.srt.layers.moe.fused_moe_triton': {'moe_align_block_size': align},
               'sglang.srt.layers.moe.token_dispatcher.standard': {'_map_topk_ids_to_local_experts': map_ids}}
    modules = {}
    for name, values in entries.items():
        pieces = name.split('.')
        for i in range(1, len(pieces)+1):
            key = '.'.join(pieces[:i])
            if key not in modules:
                modules[key] = ModuleType(key); modules[key].__path__ = []
        vars(modules[name]).update(values)
    cases = 0
    torch.manual_seed(11912)
    with patch.dict(sys.modules, modules), patch('importlib.metadata.version', return_value='0.1.12'), \
         patch('torch.cuda.get_device_capability', return_value=(9, 0)):
        for ep in (1, 2, 4, 8):
            local, rank = 64//ep, ep-1
            for same_block in (False, True):
                active.update(local=local, same_block=same_block)
                layer = torch.nn.Module()
                layer.params_dtype = torch.bfloat16
                layer.num_local_experts, layer.global_num_experts = local, 64
                layer.intermediate_size_per_partition = 128
                old = []
                for prefix, n, k in [('w13', 256, 256), ('w2', 256, 128)]:
                    tensors = {'weight': torch.zeros(local, n, k//2, dtype=torch.uint8),
                               'weight_scale': torch.ones(local, n, k//16).to(torch.float8_e4m3fn),
                               'weight_scale_2': torch.full((local, 2) if prefix == 'w13' else (local,), .123456)}
                    if prefix == 'w13':
                        tensors['weight_scale_2'][:, 1] = 3.25
                    for suffix, tensor in tensors.items():
                        value = torch.nn.Parameter(tensor.as_subclass(CpuTensorWithSimulatedCudaFlag), requires_grad=False)
                        setattr(layer, prefix+'_'+suffix, value); old.append(weakref.ref(value))
                del value, tensors, tensor
                adapter.prepare_humming_moe(layer)
                gc.collect()
                assert all(ref() is None for ref in old), 'Original Parameter objects retained'
                assert len(list(layer.named_parameters())) == 6
                assert layer.w13_weight.dtype == torch.int32 and layer.w13_weight_scale.dtype == torch.float8_e4m3fn
                assert torch.all(layer.w13_weight_scale_2 == torch.tensor(.123456).bfloat16().float())
                mapping = torch.full((64,), -1, dtype=torch.int32)
                mapping[rank*local:(rank+1)*local] = torch.arange(local)
                layer._expert_map = mapping if ep > 1 else None
                for rows in (1, 4, 8, 32):
                    for pattern in ('random', 'local', 'dropped'):
                        ids = torch.randint(0, 64, (rows, 8), dtype=torch.int32)
                        if pattern == 'local':
                            ids[:] = torch.arange(8)+rank*local
                        elif pattern == 'dropped':
                            ids.fill_(-1)
                        local_ids = map_ids(ids, mapping)
                        active['local_ids'] = local_ids
                        x = (torch.randn(rows, 256)/20).bfloat16().as_subclass(CpuTensorWithSimulatedCudaFlag)
                        calls.clear(); tables.clear()
                        output, actual_ids = adapter.humming_expert_outputs(layer, x, torch.ones(rows, 8), ids)
                        assert torch.equal(actual_ids, local_ids) and calls == ['w13', 'w2']
                        assert len(tables) == (1 if same_block else 2)
                        mask = local_ids >= 0
                        gate = (x[:, :1] + local_ids + 1).bfloat16()
                        expected = ((torch.nn.functional.silu(gate)*gate).bfloat16()+local_ids+1).bfloat16()
                        assert torch.equal(output[mask], expected[mask, None].expand(-1, 256))
                        cases += 1
    for bad in (dict(block_shape=(16, 128, 32), use_stream_k=True),
                dict(block_shape=(16, 128, 32), use_stream_k=False, use_f16_accum=True)):
        try:
            adapter.select_tactic([[0, 100, bad]], 1)
        except AssertionError:
            pass
        else:
            raise AssertionError('Forbidden precision configuration accepted')
    report = dict(scope=__doc__, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  protocol_cases=cases, preparations=8, unsafe_tactic_rejections=2,
                  humming_imported=False, native_gpu_executed=False,
                  api_binding_shapes_defaults_and_annotations_omitted={k: str(v) for k, v in api.items()})
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
