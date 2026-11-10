"""Compile exact frozen Triton kernel bodies for Hopper without a GPU.

Uses an explicit CUDA GPUTarget and never launches a kernel or loads Torch.
Extracts only @triton.jit definitions from added source files, preserving their
text/AST. Host wrappers, numerical correctness, collectives, actual image
compatibility, CUDA graphs and serving performance are outside this check.
"""
from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
HELPERS = {
    'greedy': ('02_draft_single_collective.diff', 'PATCH_MANIFEST_2026-11-09.json',
               'python/sglang/kernels/ops/speculative/draft_head_collective.py'),
    'map': ('04_fused_expert_map.diff', 'PATCH_MANIFEST_EXPERT_MAP_2026-11-09.json',
            'python/sglang/kernels/ops/moe/map_marlin_expert_ids.py'),
    'selector': ('09_selector_single_collective.diff', 'PATCH_MANIFEST_SELECTOR_2026-11-10.json',
                 'python/sglang/kernels/ops/speculative/selector_collective.py'),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def new_file(patch, path):
    marker = 'diff --git a/'+path+' b/'+path+'\n'
    text = patch.decode()
    if text.count(marker) != 1:
        raise ValueError('Expected exactly one added source file: '+path)
    remaining = text.split(marker)[1]
    next_diff = re.search(r'(?m)^diff --git ', remaining)
    block = remaining[:next_diff.start()] if next_diff else remaining
    expected = '--- /dev/null\n+++ b/'+path+'\n'
    if expected not in block:
        raise ValueError('Only complete added-file sources are supported')
    body = block.split(expected, 1)[1].splitlines(keepends=True)
    match = re.fullmatch(r'@@ -0,0 \+1,(\d+) @@\n', body[0])
    if not match or not all(line.startswith('+') for line in body[1:]):
        raise ValueError('Unexpected added-file hunk structure')
    if len(body)-1 != int(match.group(1)):
        raise ValueError('Added-file line count differs from hunk header')
    return ''.join(line[1:] for line in body[1:]).encode()


def prepare(outdir):
    sources = {}
    for name, (patch_name, manifest_name, path) in HELPERS.items():
        patch = ROOT/'serving/candidates'/patch_name
        data = patch.read_bytes()
        entry = next(row for row in json.loads((ROOT/'serving'/manifest_name).read_text())['patches']
                     if row['patch'].endswith('/'+patch_name))
        assert digest(data) == entry['sha256'], 'Frozen patch hash mismatch'
        source = new_file(data, path)
        expected = next(row for row in entry['files'] if row['path'] == path)
        assert digest(source) == expected['sha256'], 'Source extraction differs from verified patch'
        tree = ast.parse(source.decode())
        kernels = [node for node in tree.body if isinstance(node, ast.FunctionDef)
            and any(isinstance(dec, ast.Attribute) and isinstance(dec.value, ast.Name)
                    and dec.value.id == 'triton' and dec.attr == 'jit' for dec in node.decorator_list)]
        assert kernels
        lines = source.decode().splitlines(keepends=True)
        segments = [''.join(lines[min([node.lineno]+[d.lineno for d in node.decorator_list])-1:node.end_lineno])
                    for node in kernels]
        # No import of the host Torch wrappers and no mocked compiler/operator.
        generated = 'import triton\nimport triton.language as tl\n\n'+'\n\n'.join(segments)
        generated_nodes = [node for node in ast.parse(generated).body if isinstance(node, ast.FunctionDef)]
        assert [ast.dump(node, include_attributes=False) for node in generated_nodes] == [
            ast.dump(node, include_attributes=False) for node in kernels]
        generated_path = outdir/(name+'_kernels.py')
        generated_path.write_text(generated)
        compile(generated, str(generated_path), 'exec')
        sources[name] = dict(patch=str(patch.relative_to(ROOT)), patch_sha256=digest(data),
            source_path=path, source_sha256=digest(source),
            extracted_module=str(generated_path), extracted_module_sha256=digest(generated.encode()),
            kernels={node.name: dict(arguments=[arg.arg for arg in node.args.args],
                ast_sha256=digest(ast.dump(node, include_attributes=False).encode())) for node in kernels})
    return sources


def cases():
    rows = []
    def add(helper, kernel, label, pointers, constants):
        rows.append(dict(helper=helper, kernel=kernel, label=label,
                         pointer_types=pointers, constexprs=constants))
    for dtype in ('fp16', 'bf16', 'fp32'):
        add('greedy', '_pack_greedy_pairs', 'greedy_pack_'+dtype,
            dict(values='*'+dtype, indices='*i64', pairs='*i64'),
            dict(n=224, vocab_start=1 << 54, BLOCK=256))
    for tp in (2, 3, 8):
        for n in (7, 224):
            add('greedy', '_select_greedy_pairs', f'greedy_select_tp{tp}_n{n}',
                dict(gathered='*i64', output='*i64'),
                dict(n=n, tp_size=tp, RANKS=1 << (tp-1).bit_length(), BLOCK=128))
    for dtype in ('i32', 'i64'):
        add('map', '_map_kernel', 'map_'+dtype,
            dict(ids='*'+dtype, mapping='*i32', out='*i32'), dict(N=2049, BLOCK=256))
    for dtype in ('fp16', 'bf16', 'fp32'):
        for id_type in ('i32', 'i64'):
            for strided in (False, True):
                add('selector', '_pack', f'selector_pack_{dtype}_{id_type}_strided{int(strided)}',
                    dict(values='*'+dtype, ids='*'+id_type, packed='*i64'),
                    dict(size=3584 if strided else 112, columns=16,
                         vs0=37 if strided else 16, vs1=2 if strided else 1,
                         is0=53 if strided else 16, is1=3 if strided else 1,
                         start=(1 << 31)-32 if strided else 176792, BLOCK=256))
    for strided in (False, True):
        columns = 48 if strided else 128
        add('selector', '_unpack', f'selector_unpack_strided{int(strided)}',
            dict(packed='*i64', values='*fp32', ids='*i64'),
            dict(size=224*columns if strided else 7*columns, columns=columns,
                 ps0=columns*2+1 if strided else columns, ps1=2 if strided else 1, BLOCK=256))
    return rows


def compile_all(sources, variants, report, outdir, save):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    assert triton.__version__ == '3.8.0', 'Use the pinned compiler environment'
    report.update(compiler_version=triton.__version__, native_compiler_executed=True,
                  target=dict(backend='cuda', arch=90, warp_size=32))
    ptxas = Path(triton.__file__).parent/'backends/nvidia/bin/ptxas'
    if ptxas.exists():
        result = subprocess.run([str(ptxas), '--version'], text=True, capture_output=True, check=True)
        report['ptxas'] = dict(sha256=digest(ptxas.read_bytes()), version=result.stdout+result.stderr)
    save()
    modules = {}
    for name, source in sources.items():
        spec = importlib.util.spec_from_file_location('muse_aot_'+name, source['extracted_module'])
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        modules[name] = mod
    for variant in variants:
        report.update(inflight_case=variant['label']); save()
        fn = getattr(modules[variant['helper']], variant['kernel'])
        provided = set(variant['pointer_types']) | set(variant['constexprs'])
        assert provided == set(fn.arg_names), 'Signature does not cover the actual kernel'
        signature = {name: variant['pointer_types'].get(name, 'constexpr') for name in fn.arg_names}
        kernel = triton.compile(ASTSource(fn, signature, constexprs=variant['constexprs']),
            target=GPUTarget('cuda', 90, 32), options=dict(num_warps=4))
        assert kernel.asm['cubin'].startswith(b'\x7fELF'), 'Expected assembled CUDA ELF'
        assert re.search(r'\.target\s+sm_90[a-z]?', kernel.asm['ptx']), 'Not Hopper PTX'
        files = {}
        for extension in ('ttir', 'ttgir', 'llir', 'ptx', 'cubin'):
            data = kernel.asm[extension]
            data = data.encode() if isinstance(data, str) else data
            path = outdir/(variant['label']+'.'+extension)
            path.write_bytes(data)
            files[extension] = dict(path=str(path), bytes=len(data), sha256=digest(data))
        # Never read lazy launch/driver properties such as n_regs or run.
        metadata = {name: getattr(kernel.metadata, name) for name in
            ('name', 'num_warps', 'num_ctas', 'shared', 'global_scratch_size', 'profile_scratch_size')
            if hasattr(kernel.metadata, name)}
        report['compiled_cases'].append(dict(variant, signature=signature, files=files,
                                              compile_metadata=metadata))
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    assert not args.output_dir.exists(), 'Choose a fresh artifact directory'
    args.output_dir.mkdir(parents=True)
    report_path = args.output_dir/'report.json'
    report = dict(scope=__doc__, requested_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        python=sys.version, platform=platform.platform(), state='preparing',
        native_compiler_executed=False, gpu_kernel_executed=False,
        tool_sha256=digest(Path(__file__).read_bytes()), compiled_cases=[],
        ci={name: os.environ.get(name) for name in ('GITHUB_SHA', 'GITHUB_RUN_ID', 'GITHUB_RUN_ATTEMPT',
                                                 'RUNNER_OS', 'RUNNER_ARCH')})
    def save():
        report_path.write_text(json.dumps(report, indent=2)+'\n')
    save()
    try:
        sources = prepare(args.output_dir)
        variants = cases()
        for helper, source in sources.items():
            assert set(source['kernels']) == {v['kernel'] for v in variants if v['helper'] == helper}
        report.update(sources=sources, planned_cases=variants)
        if args.plan_only:
            report['state'] = 'source_plan_only'
        else:
            compile_all(sources, variants, report, args.output_dir, save)
            report['state'] = 'hopper_compilation_passed_only'
    except BaseException:
        report.update(state='failed', error=traceback.format_exc())
        raise
    finally:
        save()
    print(json.dumps(dict(state=report['state'], planned=len(report.get('planned_cases', [])),
                         compiled=len(report['compiled_cases']), gpu_kernel_executed=False)))


if __name__ == '__main__':
    main()
