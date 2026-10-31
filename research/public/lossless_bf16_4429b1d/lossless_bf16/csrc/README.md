# CUDA sources

`zg.cu`, `zt.cu`, `zr.cu` and `zs.cu` are byte-identical to the kernel sources that were built and measured on the
H200 (runs R1–R3 in [`results/`](../../results/)). They are unmodified, including their comments, which refer to the
kernels by these short names:

| file | kernel in the Python API | SHA-256 |
|---|---|---|
| `zg.cu` | `fullk` | `142974b9b163b842ed972c281e3d0a910135b20c2bad7444cdd080f43214628f` |
| `zt.cu` | `fullk_tma` | `683f581d74362ed38cfb3c1110659cc7c726f32401ecf67cc0e3592014a4b544` |
| `zr.cu` | `rowpair` | `7b17f03e890db26e7af5dc08322b917bdcb16c5d42c4f7053c33d1e74d270f40` |
| `zs.cu` | `splitk` | `d9a75510f5be74ebccc564752b86b795d5b634f8b1704dd4bd0d2bbb905eff48` |

Some comments use names from outside these files: `zk` is the 12-bit format itself (its encoder is
`lossless_bf16/format.py`); `zs_block` and `zs_plan` are `packing.blocked_from_rowtile` and `configs.splitk_plan`
here; `sgx_fin_kernel` is the finalize kernel of an uncompressed stream-K GEMM that is not part of this work.

`zr.cu` also contains a `PAIR` mode, configurations 9–11: fp32 output, summing the two rows of a tile. It is
compiled but not exposed by the Python API.

`bindings.cpp` is new: it declares the four host entry points and registers them with pybind11. The measured build
used PyTorch's inline-extension loader and compiled `zg`+`zr` and `zt`+`zs` as two translation units, with
`-O3 --use_fast_math -gencode=arch=compute_90a,code=sm_90a` and a CUDA 13 toolchain. This package compiles each file
as its own translation unit with the same flags (`lossless_bf16/ops.py`). It was compile-checked with nvcc 12.8, not
run.
