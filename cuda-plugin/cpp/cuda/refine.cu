// C ABI for refinement using the coarse scan directly on the device.

#include "cuda/refine.cuh"

#include <cstdio>
#include <cstdlib>

struct md_refine_state {
  int B, L, nf, nc, nmodes, block;

  // Launch and download require current geometry to match the allocation.
  int alloc_nf, alloc_nc, alloc_nmodes;

  double *d_thickness;
  double *d_vp;
  double *d_vs;
  double *d_rho;
  float *d_periods;
  double *d_out;
  // Per-(model, period) trace flags, one unsigned int each.  Sized and freed
  // with d_out because they share its geometry, and cleared at the start of
  // every run: a stale flag from the previous batch would be read as a failure
  // in the current one.
  unsigned int *d_flags;
  unsigned int *h_flags;
  unsigned long long *d_stats;
  unsigned long long h_stats[md::kStatCount];
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

void md_refine_free(md_refine_state *st);
int md_refine_flags(md_refine_state *, unsigned int *);

const char *md_refine_last_error(void) { return g_last_error; }

md_refine_state *md_refine_create(int B, int L) {
  if (B <= 0 || L < 2) {
    std::snprintf(g_last_error, sizeof(g_last_error),
                  "md_refine_create: bad shape B=%d L=%d", B, L);
    return nullptr;
  }
  auto *st = static_cast<md_refine_state *>(std::calloc(1, sizeof(md_refine_state)));
  if (!st) return nullptr;
  st->B = B;
  st->L = L;
  st->block = 256;

  const size_t n = static_cast<size_t>(B) * static_cast<size_t>(L);
  if (cudaMalloc(&st->d_thickness, n * sizeof(double)) != cudaSuccess ||
      cudaMalloc(&st->d_vp, n * sizeof(double)) != cudaSuccess ||
      cudaMalloc(&st->d_vs, n * sizeof(double)) != cudaSuccess ||
      cudaMalloc(&st->d_rho, n * sizeof(double)) != cudaSuccess ||
      cudaMalloc(&st->d_stats, md::kStatCount * sizeof(unsigned long long)) !=
          cudaSuccess) {
    md_refine_free(st);
    fail("md_refine_create: cudaMalloc", cudaGetLastError());
    return nullptr;
  }
  return st;
}

int md_refine_upload_models(md_refine_state *st, const double *thickness,
                            const double *vp, const double *vs,
                            const double *rho) {
  if (!st) {
    std::snprintf(g_last_error, sizeof(g_last_error),
                  "md_refine_upload_models: null state");
    return 1;
  }
  const size_t bytes =
      static_cast<size_t>(st->B) * static_cast<size_t>(st->L) * sizeof(double);
  cudaError_t e;
  if ((e = cudaMemcpy(st->d_thickness, thickness, bytes,
                      cudaMemcpyHostToDevice)) != cudaSuccess ||
      (e = cudaMemcpy(st->d_vp, vp, bytes, cudaMemcpyHostToDevice)) != cudaSuccess ||
      (e = cudaMemcpy(st->d_vs, vs, bytes, cudaMemcpyHostToDevice)) != cudaSuccess ||
      (e = cudaMemcpy(st->d_rho, rho, bytes, cudaMemcpyHostToDevice)) != cudaSuccess) {
    return fail("md_refine_upload_models", e);
  }
  return 0;
}

int md_refine_upload_periods(md_refine_state *st, int nf, const double *periods) {
  if (!st) {
    std::snprintf(g_last_error, sizeof(g_last_error),
                  "md_refine_upload_periods: null state");
    return 1;
  }
  if (nf <= 0) {
    std::snprintf(g_last_error, sizeof(g_last_error),
                  "md_refine_upload_periods: nf must be positive, got %d", nf);
    return 1;
  }
  // A different period count invalidates any output sized against the old one,
  // flags included -- they are one per (model, period).
  if (st->d_out && nf != st->nf) {
    cudaFree(st->d_out);
    st->d_out = nullptr;
    st->alloc_nf = st->alloc_nc = st->alloc_nmodes = 0;
  }
  if (st->d_flags && nf != st->nf) {
    cudaFree(st->d_flags);
    st->d_flags = nullptr;
    std::free(st->h_flags);
    st->h_flags = nullptr;
  }
  if (st->d_periods) {
    cudaFree(st->d_periods);
    st->d_periods = nullptr;
  }
  if (cudaMalloc(&st->d_periods, static_cast<size_t>(nf) * sizeof(float)) !=
      cudaSuccess) {
    return fail("md_refine_upload_periods: cudaMalloc", cudaGetLastError());
  }
  auto *tmp = static_cast<float *>(std::malloc(static_cast<size_t>(nf) * sizeof(float)));
  if (!tmp) return fail("md_refine_upload_periods: staging alloc", cudaErrorMemoryAllocation);
  for (int i = 0; i < nf; ++i) tmp[i] = static_cast<float>(periods[i]);
  const cudaError_t e =
      cudaMemcpy(st->d_periods, tmp, static_cast<size_t>(nf) * sizeof(float),
                 cudaMemcpyHostToDevice);
  std::free(tmp);
  if (e != cudaSuccess) return fail("md_refine_upload_periods: cudaMemcpy", e);
  st->nf = nf;
  return 0;
}

int md_refine_set_output(md_refine_state *st, int nc, int nmodes) {
  if (!st) {
    std::snprintf(g_last_error, sizeof(g_last_error),
                  "md_refine_set_output: null state");
    return 1;
  }
  if (nc < 2 || nmodes <= 0) {
    std::snprintf(g_last_error, sizeof(g_last_error),
                  "md_refine_set_output: need nc >= 2 and nmodes >= 1, got "
                  "nc=%d nmodes=%d",
                  nc, nmodes);
    return 1;
  }
  if (st->nf <= 0) {
    std::snprintf(g_last_error, sizeof(g_last_error),
                  "md_refine_set_output: upload_periods() first -- the output "
                  "size depends on the period count");
    return 1;
  }
  if (st->d_out) {
    cudaFree(st->d_out);
    st->d_out = nullptr;
  }
  const size_t n = static_cast<size_t>(st->B) * static_cast<size_t>(st->nf) *
                   static_cast<size_t>(nmodes);
  if (cudaMalloc(&st->d_out, n * sizeof(double)) != cudaSuccess) {
    return fail("md_refine_set_output: cudaMalloc", cudaGetLastError());
  }
  // One flag word per trace, independent of nmodes -- a trace is a (model,
  // period) pair and all of its mode threads share it.
  if (st->d_flags) {
    cudaFree(st->d_flags);
    st->d_flags = nullptr;
  }
  const size_t nflag = static_cast<size_t>(st->B) * static_cast<size_t>(st->nf);
  if (cudaMalloc(&st->d_flags, nflag * sizeof(unsigned int)) != cudaSuccess) {
    return fail("md_refine_set_output: cudaMalloc flags", cudaGetLastError());
  }
  if (!st->h_flags) {
    st->h_flags = static_cast<unsigned int *>(std::malloc(nflag * sizeof(unsigned int)));
    if (!st->h_flags) {
      return fail("md_refine_set_output: host flags alloc", cudaErrorMemoryAllocation);
    }
  }
  st->nc = nc;
  st->nmodes = nmodes;
  st->alloc_nf = st->nf;
  st->alloc_nc = nc;
  st->alloc_nmodes = nmodes;
  return 0;
}

// Launch only.  `scan_dev` is the Stage 1 device buffer, consumed in place.
int md_refine_run(md_refine_state *st, const float *scan_dev, double cmin,
                  double cmax, double c_tol_rel, double f_accept, int nmodes,
                  int block) {
  if (!st || !st->d_periods) return fail("md_refine_run: periods not uploaded", cudaErrorInvalidValue);
  if (!st->d_out) return fail("md_refine_run: output not allocated", cudaErrorInvalidValue);
  if (!st->d_flags) {
    return fail("md_refine_run: flags not allocated", cudaErrorInvalidValue);
  }
  if (!scan_dev) return fail("md_refine_run: null scan buffer", cudaErrorInvalidValue);
  if (st->nf != st->alloc_nf || st->nc != st->alloc_nc) {
    std::snprintf(g_last_error, sizeof(g_last_error),
                  "md_refine_run: geometry changed since allocation "
                  "(allocated nf=%d nc=%d, now nf=%d nc=%d); call "
                  "set_output() again",
                  st->alloc_nf, st->alloc_nc, st->nf, st->nc);
    return 1;
  }
  // A larger mode count requires reallocating the output buffer.
  if (nmodes > 0 && nmodes > st->alloc_nmodes) {
    std::snprintf(g_last_error, sizeof(g_last_error),
                  "md_refine_run: nmodes=%d exceeds the allocated %d; call "
                  "set_output(nc, nmodes) to resize",
                  nmodes, st->alloc_nmodes);
    return 1;
  }
  if (nmodes > 0) st->nmodes = nmodes;
  // Match the kernel's 256-thread launch bound.
  if (block > 256) block = 256;
  if (block > 0) {
    st->block = block;
  } else {
    const int n = (st->nf + 31) / 32 * 32;
    st->block = n < 32 ? 32 : (n > 256 ? 256 : n);
  }

  const double dc =
      (st->nc > 1) ? (cmax - cmin) / static_cast<double>(st->nc - 1) : 0.0;
  // cmin is the tightest c anywhere on the grid, so it needs the most
  // halvings anywhere.  One uniform count for every thread: no divergence.
  // See the note on c_tol_rel in refine.cuh -- this is DISBA's own stopping
  // rule and is independent of the |F| acceptance threshold.
  const int iters = md::refine_iters(dc, c_tol_rel * cmin);

  cudaMemset(st->d_stats, 0, md::kStatCount * sizeof(unsigned long long));
  // Flags are per run, not cumulative.  A flag left over from the previous
  // batch would be read as a failure in this one.
  cudaMemset(st->d_flags, 0,
             static_cast<size_t>(st->B) * static_cast<size_t>(st->nf) *
                 sizeof(unsigned int));

  const size_t smem = md::refine_smem(st->L);
  if (smem > 48 * 1024) {
    const cudaError_t e = cudaFuncSetAttribute(
        md::refine_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem));
    if (e != cudaSuccess) return fail("md_refine_run: cudaFuncSetAttribute", e);
  }

  md::refine_kernel<<<st->B, st->block, smem>>>(
      scan_dev, st->d_thickness, st->d_vp, st->d_vs, st->d_rho, st->d_periods,
      st->L, st->nf, st->nc, st->nmodes, iters, cmin, cmax, f_accept, st->d_out,
      st->d_flags, st->d_stats);
  return 0;  // launch errors surface at the next sync
}

int md_refine_sync(void) {
  const cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) return fail("md_refine_sync", e);
  return 0;
}

int md_refine_download(md_refine_state *st, double *h_out) {
  if (!st) {
    std::snprintf(g_last_error, sizeof(g_last_error),
                  "md_refine_download: null state");
    return 1;
  }
  if (!st->d_out) {
    std::snprintf(g_last_error, sizeof(g_last_error),
                  "md_refine_download: output not allocated");
    return 1;
  }
  if (st->nf != st->alloc_nf || st->nc != st->alloc_nc ||
      st->nmodes > st->alloc_nmodes) {
    std::snprintf(g_last_error, sizeof(g_last_error),
                  "md_refine_download: geometry changed since allocation "
                  "(allocated nf=%d nc=%d nmodes=%d, now nf=%d nc=%d nmodes=%d)",
                  st->alloc_nf, st->alloc_nc, st->alloc_nmodes, st->nf, st->nc,
                  st->nmodes);
    return 1;
  }
  const size_t n = static_cast<size_t>(st->B) * static_cast<size_t>(st->nf) *
                   static_cast<size_t>(st->nmodes);
  const cudaError_t e =
      cudaMemcpy(h_out, st->d_out, n * sizeof(double), cudaMemcpyDeviceToHost);
  if (e != cudaSuccess) return fail("md_refine_download", e);
  return 0;
}

// Per-run brackets and root-acceptance counters.
int md_refine_stats(md_refine_state *st, unsigned long long *out, int n) {
  if (!st || !out) {
    std::snprintf(g_last_error, sizeof(g_last_error),
                  "md_refine_stats: null argument");
    return 1;
  }
  if (n < md::kStatCount) {
    std::snprintf(g_last_error, sizeof(g_last_error),
                  "md_refine_stats: need room for %d counters, got %d",
                  md::kStatCount, n);
    return 1;
  }
  const cudaError_t e = cudaMemcpy(st->h_stats, st->d_stats,
                                   md::kStatCount * sizeof(unsigned long long),
                                   cudaMemcpyDeviceToHost);
  if (e != cudaSuccess) return fail("md_refine_stats", e);
  for (int i = 0; i < md::kStatCount; ++i) out[i] = st->h_stats[i];
  return 0;
}

// Number of counters written by md_refine_stats.
int md_refine_stat_count(void) { return md::kStatCount; }

// Geometry of this state, so a caller can check it against the Stage 1 object
// whose buffer it is about to consume.
void md_refine_geometry(md_refine_state *st, int *B, int *L, int *nf, int *nc,
                        int *nmodes) {
  if (!st) {
    if (B) *B = 0;
    if (L) *L = 0;
    if (nf) *nf = 0;
    if (nc) *nc = 0;
    if (nmodes) *nmodes = 0;
    return;
  }
  if (B) *B = st->B;
  if (L) *L = st->L;
  if (nf) *nf = st->nf;
  if (nc) *nc = st->nc;
  if (nmodes) *nmodes = st->nmodes;
}

int md_refine_iters_for(double dc, double rtol, double cmin) {
  return md::refine_iters(dc, rtol * cmin);
}

int md_refine_kernel_report(int L, int block, char *buf, int buflen) {
  cudaFuncAttributes attr{};
  cudaError_t e = cudaFuncGetAttributes(&attr, md::refine_kernel);
  if (e != cudaSuccess) return fail("md_refine_kernel_report: getAttributes", e);
  const size_t smem = md::refine_smem(L);
  int max_blocks = 0;
  e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks,
                                                    md::refine_kernel, block,
                                                    smem);
  if (e != cudaSuccess) return fail("md_refine_kernel_report: occupancy", e);
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);
  std::snprintf(buf, buflen,
                "regs=%d spill_bytes=%d smem_dynamic=%zu max_blocks_per_sm=%d "
                "threads_per_sm=%d occupancy=%.1f%%",
                attr.numRegs, static_cast<int>(attr.localSizeBytes), smem,
                max_blocks, max_blocks * block,
                100.0 * max_blocks * block /
                    static_cast<double>(prop.maxThreadsPerMultiProcessor));
  return 0;
}

void md_refine_free(md_refine_state *st) {
  if (!st) return;
  cudaFree(st->d_thickness);
  cudaFree(st->d_vp);
  cudaFree(st->d_vs);
  cudaFree(st->d_rho);
  cudaFree(st->d_periods);
  cudaFree(st->d_out);
  cudaFree(st->d_flags);
  std::free(st->h_flags);
  cudaFree(st->d_stats);
  std::free(st);
}

// Per-(model, period) flags from the last run, [B * nf] unsigned ints,
// row-major [model][period].  Zeroed at the start of every run by
// md_refine_run, so this always describes the most recent one.
int md_refine_flags(md_refine_state *st, unsigned int *out) {
  if (!st || !out) {
    std::snprintf(g_last_error, sizeof(g_last_error),
                  "md_refine_flags: null argument");
    return 1;
  }
  if (!st->d_flags || !st->h_flags) {
    std::snprintf(g_last_error, sizeof(g_last_error),
                  "md_refine_flags: no flags allocated -- call set_output() "
                  "and run() first");
    return 1;
  }
  const size_t n = static_cast<size_t>(st->B) * static_cast<size_t>(st->nf);
  const cudaError_t e =
      cudaMemcpy(st->h_flags, st->d_flags, n * sizeof(unsigned int),
                 cudaMemcpyDeviceToHost);
  if (e != cudaSuccess) return fail("md_refine_flags", e);
  for (size_t i = 0; i < n; ++i) out[i] = st->h_flags[i];
  return 0;
}

}  // extern "C"
