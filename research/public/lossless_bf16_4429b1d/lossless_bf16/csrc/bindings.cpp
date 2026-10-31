// Python bindings for the 12-bit BF16 GEMV/GEMM kernels. The declarations match the host entry points in
// zg.cu, zt.cu, zr.cu and zs.cu.
#include <torch/extension.h>

void zg_gemm(torch::Tensor x, torch::Tensor wp, torch::Tensor rscale, torch::Tensor etab, int64_t emax,
             torch::Tensor out, int64_t m, int64_t k, int64_t nrt, int64_t nout, int64_t cfg);

int64_t zt_bps(int64_t cfg);
void zt_gemm(torch::Tensor x, torch::Tensor wp, torch::Tensor rscale, torch::Tensor etab, int64_t emax,
             torch::Tensor out, int64_t m, int64_t k, int64_t nrt, int64_t nout, int64_t cfg);

torch::Tensor zr_info();
void zr_gemv(torch::Tensor x, torch::Tensor wp, torch::Tensor rscale, torch::Tensor etab, int64_t emax,
             c10::optional<torch::Tensor> out, c10::optional<torch::Tensor> outf, int64_t k, int64_t ntiles,
             int64_t cfg);

torch::Tensor zs_info();
int64_t zs_gemm(torch::Tensor x, torch::Tensor wp, torch::Tensor rscale, torch::Tensor erow, int64_t emr,
                torch::Tensor out, torch::Tensor ws, int64_t m, int64_t k, int64_t n, int64_t cfg, int64_t G,
                int64_t jmax, int64_t tpp, int64_t npt, int64_t pdl, c10::optional<torch::Tensor> dbg);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("zg_gemm", &zg_gemm, "full-K GEMM on rowtile weights (m <= 32)");
    m.def("zt_bps", &zt_bps, "CTAs per SM of a fullk_tma config");
    m.def("zt_gemm", &zt_gemm, "full-K GEMM with a TMA bulk-copy ring (m <= 16)");
    m.def("zr_info", &zr_info, "registers / local bytes / CTAs per SM of each rowpair config");
    m.def("zr_gemv", &zr_gemv, "row-pair GEMV on rowpair weights (m = 1)");
    m.def("zs_info", &zs_info, "registers / local bytes / CTAs per SM / smem of each splitk config");
    m.def("zs_gemm", &zs_gemm, "split-K GEMM on blocked weights with a PDL finalize (m <= 32)");
}
