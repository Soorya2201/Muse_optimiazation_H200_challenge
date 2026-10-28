#!/usr/bin/env python3
"""Fail fast on structural drift in the reserved BF16 reset queue."""

from __future__ import annotations

import ast
import base64
import gzip
import hashlib
import subprocess
from pathlib import Path


TRACK = Path(__file__).resolve().parents[1]
PROBLEM_MARKER = b"# okc: problem=muse_bf16_proj\n"
V18_SHA256 = "111630ff363e5988ca57f28434a2e1a7e9c1d524079c64727b52c6aa287dcb44"

RESET_QUEUE = (
    "upstream_nathen_v18/submission.py",
    "submission_v30_all_small_qkv32.py",
    "submission_v31_pooler32_geometry.py",
    "submission_v20_cluster_dense_m128.py",
    "submission_v21_cluster_dense_m128_m256.py",
    "submission_v22_cluster_dense_autotune.py",
    "submission_v32_c12_cluster_pooler128.py",
)

PREFIXES = (
    ("submission_v19_nathen_route_autotune.py", "submission_v20_cluster_dense_m128.py"),
    ("submission_v19_nathen_route_autotune.py", "submission_v21_cluster_dense_m128_m256.py"),
    ("submission_v21_cluster_dense_m128_m256.py", "submission_v22_cluster_dense_autotune.py"),
    ("submission_v23_full_history_autotune.py", "submission_v25_history_cross_autotune.py"),
    ("submission_v25_history_cross_autotune.py", "submission_v28_history_cross_qkv16.py"),
    ("submission_v28_history_cross_qkv16.py", "submission_v29_all_small_autotune.py"),
    ("submission_v29_all_small_autotune.py", "submission_v30_all_small_qkv32.py"),
    ("submission_v19_nathen_route_autotune.py", "submission_v31_pooler32_geometry.py"),
    ("upstream_nathen_v18/submission.py", "submission_v32_c12_cluster_pooler128.py"),
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def assignment(tree: ast.AST, name: str):
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"missing assignment {name}")


def audit_queue() -> None:
    for relative in RESET_QUEUE:
        path = TRACK / relative
        data = path.read_bytes()
        assert data.startswith(PROBLEM_MARKER), f"wrong problem marker: {relative}"
        compile(data, str(path), "exec", ast.PyCF_ONLY_AST)
        print(f"syntax marker ok: {relative} ({len(data):,} bytes)")


def audit_prefixes() -> None:
    for base_relative, candidate_relative in PREFIXES:
        base = (TRACK / base_relative).read_bytes()
        candidate = (TRACK / candidate_relative).read_bytes()
        assert candidate[: len(base)] == base, f"prefix drift: {candidate_relative}"
        print(f"prefix ok: {candidate_relative} <- {base_relative} ({len(base):,} bytes)")


def audit_live_submission() -> None:
    live = (TRACK / "submission.py").read_bytes()
    archived = (TRACK / "submission_v3_cluster.py").read_bytes()
    assert live == archived, "live scored submission no longer matches archived v3"
    print(f"live v3 ok: {sha256(live)}")


def audit_v18() -> None:
    data = (TRACK / "upstream_nathen_v18/submission.py").read_bytes()
    assert sha256(data) == V18_SHA256, "vendored v18 hash drift"
    print(f"v18 hash ok: {V18_SHA256}")


def audit_v32_payload() -> None:
    candidate = TRACK / "submission_v32_c12_cluster_pooler128.py"
    tree = ast.parse(candidate.read_text())
    declared_sha = assignment(tree, "_V32_CUDA_SHA256")
    payload = assignment(tree, "_V32_CUDA_GZIP")
    embedded = gzip.decompress(base64.b64decode(payload))
    generated = subprocess.check_output(
        ["/usr/bin/python3", str(TRACK / "tools/c12_cluster_pooler128_source.py")]
    )
    assert embedded == generated, "v32 embedded CUDA differs from generator stdout"
    assert sha256(generated) == declared_sha, "v32 declared CUDA hash differs"
    print(f"v32 CUDA ok: {declared_sha} ({len(generated):,} bytes)")


def audit_pruned_geometries() -> None:
    expected = {
        "submission_v30_all_small_qkv32.py": (
            "_V30_GEOMETRIES",
            (("wr4", (4, 4, 8, 4)),),
        ),
        "submission_v31_pooler32_geometry.py": (
            "_V31_GEOMETRIES",
            (("wr2", (2, 8, 16, 1)),),
        ),
    }
    for relative, (name, value) in expected.items():
        tree = ast.parse((TRACK / relative).read_text())
        assert assignment(tree, name) == value, f"compile-known spill geometry returned: {relative}"
        print(f"pruned geometry ok: {relative} {value}")


def main() -> None:
    audit_queue()
    audit_prefixes()
    audit_live_submission()
    audit_v18()
    audit_v32_payload()
    audit_pruned_geometries()
    print("BF16 reset preflight passed")


if __name__ == "__main__":
    main()
