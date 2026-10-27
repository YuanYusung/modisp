// C ABI for reusable coarse-scan device state.

#include "cuda/coarse_scan.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>

struct md_cuda_state {
  int B, L, nf, nc, block;

  // Launch and download require current geometry to match the allocation.
  int alloc_nf, alloc_nc;

  // Refinement consumes the range used by the most recent scan.
  double cmin, cmax;
  bool range_known;

  bool models_uploaded;

  double *d_thickness;
  double *d_vp;
  double *d_vs;
  double *d_rho;
  float *d_periods;
  float *d_out;
};

namespace {

char g_last_error[512] = {0};

int fail(const char *what, cudaError_t err) {
  std::snprintf(g_last_error, sizeof(g_last_error), "%s: %s", what,
                cudaGetErrorString(err));
  return 1;
}

}  // namespace

extern "C" {

// Forward declaration: md_cuda_create rolls back with this on a partial
// allocation, and it is defined further down.
void md_cuda_free(md_cuda_state *st);

const char *md_cuda_last_error(void) { return g_last_error; }

const char *md_cuda_device_name(void) {
  static char name[512] = {0};
  if (name[0] == 0) {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
      std::snprintf(name, sizeof(name), "%s sm_%d%d", prop.name, prop.major,
                    prop.minor);
    } else {
      std::snprintf(name, sizeof(name), "<no device>");
    }
  }
  return name;
}

md_cuda_state *md_cuda_create(int B, int L) {
  if (B <= 0 || L < 2) {
    std::snprintf(g_last_error, sizeof(g_last_error),
                  "md_cuda_create: bad shape B=%d L=%d (L must be >= 2: layers "
                  "plus half-space)",
                  B, L);
    return nullptr;
  }
  auto *st = static_cast<md_cuda_state *>(std::calloc(1, sizeof(md_cuda_state)));
  if (!st) return nullptr;
  st->B = B;
  st->L = L;
  st->block = 256;

  const size_t n = static_cast<size_t>(B) * static_cast<size_t>(L);
  if (cudaMalloc(&st->d_thickness, n * sizeof(double)) != cudaSuccess ||
      cudaMalloc(&st->d_vp, n * sizeof(double)) != cudaSuccess ||
      cudaMalloc(&st->d_vs, n * sizeof(double)) != cudaSuccess ||
      cudaMalloc(&st->d_rho, n * sizeof(double)) != cudaSuccess) {
    md_cuda_free(st);
    fail("md_cuda_create: cudaMalloc(model arrays)", cudaGetLastError());
    return nullptr;
  }
  return st;
}

int md_cuda_upload_models(md_cuda_state *st, const double *thickness,
                          const double *vp, const double *vs,
                          const double *rho) {
  if (!st) {
    std::snprintf(g_last_error, sizeof(g_last_error),
                  "md_cuda_upload_models: null state");
    return 1;
  }
  const size_t n = static_cast<size_t>(st->B) * static_cast<size_t>(st->L);
  const size_t bytes = n * sizeof(double);
  cudaError_t e;
  if ((e = cudaMemcpy(st->d_thickness, thickness, bytes,
                      cudaMemcpyHostToDevice)) != cudaSuccess ||
      (e = cudaMemcpy(st->d_vp, vp, bytes, cudaMemcpyHostToDevice)) != cudaSuccess ||
      (e = cudaMemcpy(st->d_vs, vs, bytes, cudaMemcpyHostToDevice)) != cudaSuccess ||
      (e = cudaMemcpy(st->d_rho, rho, bytes, cudaMemcpyHostToDevice)) != cudaSuccess) {
    return fail("md_cuda_upload_models", e);
  }
  st->models_uploaded = true;
  return 0;
}

int md_cuda_upload_periods(md_cuda_state *st, int nf, const double *periods) {
  if (!st) {
    std::snprintf(g_last_error, sizeof(g_last_error),
                  "md_cuda_upload_periods: null state");
    return 1;
  }
  if (nf <= 0) {
    std::snprintf(g_last_error, sizeof(g_last_error),
                  "md_cuda_upload_periods: nf must be positive, got %d", nf);
    return 1;
  }
  // A different length invalidates any output sized against the old one.
  // Dropping the allocation here means the next run() reports "not allocated"
  // instead of writing past the end of a stale buffer.
  if (st->d_out && nf != st->nf) {
    cudaFree(st->d_out);
    st->d_out = nullptr;
    st->alloc_nf = st->alloc_nc = 0;
  }
  if (st->d_periods) {
    cudaFree(st->d_periods);
    st->d_periods = nullptr;
  }
  if (cudaMalloc(&st->d_periods, static_cast<size_t>(nf) * sizeof(float)) !=
      cudaSuccess) {
    return fail("md_cuda_upload_periods: cudaMalloc", cudaGetLastError());
  }
  auto *tmp = static_cast<float *>(std::malloc(static_cast<size_t>(nf) * sizeof(float)));
  if (!tmp) return fail("md_cuda_upload_periods: host staging alloc", cudaErrorMemoryAllocation);
  for (int i = 0; i < nf; ++i) tmp[i] = static_cast<float>(periods[i]);
  const cudaError_t e = cudaMemcpy(st->d_periods, tmp,
                                   static_cast<size_t>(nf) * sizeof(float),
                                   cudaMemcpyHostToDevice);
  std::free(tmp);
  if (e != cudaSuccess) return fail("md_cuda_upload_periods: cudaMemcpy", e);
  st->nf = nf;
  return 0;
}

int md_cuda_set_output(md_cuda_state *st, int nc) {
  if (!st) {
    std::snprintf(g_last_error, sizeof(g_last_error),
                  "md_cuda_set_output: null state");
    return 1;
  }
  if (nc <= 0) {
    std::snprintf(g_last_error, sizeof(g_last_error),
                  "md_cuda_set_output: nc must be positive, got %d", nc);
    return 1;
  }
  if (st->nf <= 0) {
    std::snprintf(g_last_error, sizeof(g_last_error),
                  "md_cuda_set_output: upload_periods() first -- the output "
                  "size depends on the period count");
    return 1;
  }
  if (st->d_out) {
    cudaFree(st->d_out);
    st->d_out = nullptr;
  }
  const size_t n = static_cast<size_t>(st->B) * static_cast<size_t>(st->nf) *
                   static_cast<size_t>(nc);
  if (cudaMalloc(&st->d_out, n * sizeof(float)) != cudaSuccess) {
    return fail("md_cuda_set_output: cudaMalloc", cudaGetLastError());
  }
  st->nc = nc;
  st->alloc_nf = st->nf;
  st->alloc_nc = nc;
  return 0;
}

// Launch only -- no synchronisation, no copy.  Timing harnesses wrap this.
int md_cuda_run(md_cuda_state *st, double cmin, double cmax, int block) {
  if (!st || !st->d_periods) return fail("md_cuda_run: periods not uploaded", cudaErrorInvalidValue);
  if (!st->d_out) return fail("md_cuda_run: output not allocated", cudaErrorInvalidValue);
  if (!st->models_uploaded) return fail("md_cuda_run: models not uploaded", cudaErrorInvalidValue);
  if (st->nf != st->alloc_nf || st->nc != st->alloc_nc) {
    std::snprintf(g_last_error, sizeof(g_last_error),
                  "md_cuda_run: geometry changed since allocation "
                  "(allocated nf=%d nc=%d, now nf=%d nc=%d); call "
                  "set_output() again",
                  st->alloc_nf, st->alloc_nc, st->nf, st->nc);
    return 1;
  }
  if (block > 0) st->block = block;

  const size_t smem = md::coarse_scan_smem(st->L);
  if (smem > 48 * 1024) {
    const cudaError_t e = cudaFuncSetAttribute(
        md::coarse_scan_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem));
    if (e != cudaSuccess) return fail("md_cuda_run: cudaFuncSetAttribute", e);
  }

  md::coarse_scan_kernel<<<st->B, st->block, smem>>>(
      st->d_thickness, st->d_vp, st->d_vs, st->d_rho, st->d_periods, st->L,
      st->nf, st->nc, cmin, cmax, st->d_out);
  st->cmin = cmin;
  st->cmax = cmax;
  st->range_known = true;
  return 0;  // launch errors surface at the next sync
}

// The (cmin, cmax) the last scan ran with, so a caller can confirm that a
// refinement is about to read a row this scan actually filled.  Returns 0 and
// writes through the pointers on success, 1 if no scan has run.
int md_cuda_run_range(md_cuda_state *st, double *cmin, double *cmax, int *nc) {
  if (!st || !st->range_known) return 1;
  if (cmin) *cmin = st->cmin;
  if (cmax) *cmax = st->cmax;
  if (nc) *nc = st->nc;
  return 0;
}

int md_cuda_sync(void) {
  const cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) return fail("md_cuda_sync", e);
  return 0;
}

// Geometry of this state, so a caller (or the other stage) can check that two
// objects agree before one consumes the other's buffer.
void md_cuda_geometry(md_cuda_state *st, int *B, int *L, int *nf, int *nc) {
  if (!st) {
    if (B) *B = 0;
    if (L) *L = 0;
    if (nf) *nf = 0;
    if (nc) *nc = 0;
    return;
  }
  if (B) *B = st->B;
  if (L) *L = st->L;
  if (nf) *nf = st->nf;
  if (nc) *nc = st->nc;
}

// Expose the device scan buffer for in-place refinement.
const float *md_cuda_scan_ptr(md_cuda_state *st) {
  return st ? st->d_out : nullptr;
}

int md_cuda_download(md_cuda_state *st, float *h_out) {
  if (!st) {
    std::snprintf(g_last_error, sizeof(g_last_error),
                  "md_cuda_download: null state");
    return 1;
  }
  if (!st->d_out) {
    std::snprintf(g_last_error, sizeof(g_last_error),
                  "md_cuda_download: output not allocated");
    return 1;
  }
  if (st->nf != st->alloc_nf || st->nc != st->alloc_nc) {
    std::snprintf(g_last_error, sizeof(g_last_error),
                  "md_cuda_download: geometry changed since allocation "
                  "(allocated nf=%d nc=%d, now nf=%d nc=%d)",
                  st->alloc_nf, st->alloc_nc, st->nf, st->nc);
    return 1;
  }
  const size_t n = static_cast<size_t>(st->B) * static_cast<size_t>(st->nf) *
                   static_cast<size_t>(st->nc);
  const cudaError_t e =
      cudaMemcpy(h_out, st->d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
  if (e != cudaSuccess) return fail("md_cuda_download", e);
  return 0;
}

// Report registers, spills, shared memory and occupancy.
int md_cuda_kernel_report(int L, int block, char *buf, int buflen) {
  cudaFuncAttributes attr{};
  cudaError_t e = cudaFuncGetAttributes(&attr, md::coarse_scan_kernel);
  if (e != cudaSuccess) return fail("md_cuda_kernel_report: getAttributes", e);

  int max_blocks = 0;
  e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_blocks, md::coarse_scan_kernel, block, md::coarse_scan_smem(L));
  if (e != cudaSuccess) return fail("md_cuda_kernel_report: occupancy", e);

  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);

  std::snprintf(buf, buflen,
                "regs=%d spill_bytes=%d smem_static=%zu smem_dynamic=%zu "
                "max_blocks_per_sm=%d threads_per_sm=%d occupancy=%.1f%%",
                attr.numRegs, static_cast<int>(attr.localSizeBytes),
                attr.sharedSizeBytes, md::coarse_scan_smem(L), max_blocks,
                max_blocks * block,
                100.0 * max_blocks * block /
                    static_cast<double>(prop.maxThreadsPerMultiProcessor));
  return 0;
}

void md_cuda_free(md_cuda_state *st) {
  if (!st) return;
  cudaFree(st->d_thickness);
  cudaFree(st->d_vp);
  cudaFree(st->d_vs);
  cudaFree(st->d_rho);
  cudaFree(st->d_periods);
  cudaFree(st->d_out);
  std::free(st);
}

}  // extern "C"
