// One block scans each model over the flattened period/velocity grid.
// The normalized Dunkin evaluator is shared with the CPU implementation.

#ifndef MODISP_CUDA_COARSE_SCAN_CUH
#define MODISP_CUDA_COARSE_SCAN_CUH

#include <cuda_runtime.h>
#include <math_constants.h>

#include "common/dunkin.hpp"
#include "common/scan_grid.hpp"

namespace md {

constexpr float kTwoPiF = 6.283185307179586476925286766559f;

// Store signed FP32 secular values. Grid coordinates are computed in FP64 so
// refinement can reconstruct exactly the same candidate velocities.
__global__ void coarse_scan_kernel(const double *__restrict__ thickness,
                                   const double *__restrict__ vp,
                                   const double *__restrict__ vs,
                                   const double *__restrict__ rho,
                                   const float *__restrict__ periods, int L,
                                   int nf, int nc, double cmin, double cmax,
                                   float *__restrict__ out) {
  extern __shared__ float smem[];
  float *sh = smem;
  float *svp = sh + L;
  float *svs = svp + L;
  float *srho = svs + L;

  const int b = blockIdx.x;
  const size_t base = static_cast<size_t>(b) * static_cast<size_t>(L);
  for (int i = threadIdx.x; i < L; i += blockDim.x) {
    sh[i] = static_cast<float>(thickness[base + i]);
    svp[i] = static_cast<float>(vp[base + i]);
    svs[i] = static_cast<float>(vs[base + i]);
    srho[i] = static_cast<float>(rho[base + i]);
  }
  __syncthreads();

  const double dc =
      (nc > 1) ? (cmax - cmin) / static_cast<double>(nc - 1) : 0.0;
  const int total = nf * nc;
  float *outb = out + static_cast<size_t>(b) * static_cast<size_t>(total);

  // How far along the c axis anything will ever look.  Past this, Stage 2 does
  // not read -- see common/scan_grid.hpp for why that is a shared definition
  // and not an expression repeated in both kernels.
  //
  // Taken from the DOUBLE input rather than from the narrowed float shared
  // copy: Stage 2 computes the same bound from doubles, and reading it from a
  // float could land on a different integer at a boundary.
  const int last_needed =
      md::scan_last_needed<double>(vs[base + L - 1], cmin, cmax, nc);

  for (int t = threadIdx.x; t < total; t += blockDim.x) {
    const int f = t / nc;
    const int ic = t - f * nc;
    if (ic > last_needed) {
      // Cells beyond the shared scan bound are explicitly uncomputed.
      outb[t] = CUDART_NAN;
      continue;
    }
    const float omega = kTwoPiF / __ldg(periods + f);
    const float c =
        static_cast<float>(cmin + dc * static_cast<double>(ic));
    outb[t] = dunkin_secular<float, true>(sh, svp, svs, srho, L, omega / c, omega);
  }
}

// Shared memory the launch must reserve for one block.
inline size_t coarse_scan_smem(int L) {
  return static_cast<size_t>(4) * static_cast<size_t>(L) * sizeof(float);
}

}  // namespace md

#endif  // MODISP_CUDA_COARSE_SCAN_CUH
