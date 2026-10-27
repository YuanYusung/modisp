// Locate sign-change brackets and strict low-|F| troughs on a rectangular
// (model, period, mode) grid, then refine candidates in FP64. Unoccupied mode
// slots remain NaN and per-trace flags record numerical diagnostics.

#ifndef MODISP_CUDA_REFINE_CUH
#define MODISP_CUDA_REFINE_CUH

#include <cmath>
#include <cuda_runtime.h>
#include <math_constants.h>

#include "common/dunkin.hpp"
#include "common/scan_grid.hpp"

namespace md {

// A strict trough triggers 64 interior samples over its two-cell window.
#ifndef MD_SUBSAMPLES
#define MD_SUBSAMPLES 64
#endif
constexpr int kSubSamples = MD_SUBSAMPLES;

// Absolute |F| a trough must fall below to be worth sub-sampling.  A hidden
// pair means F came close to zero TWICE, so the trough that shadows it is a
// small value; a generic |F| dip in a smooth region is not.  Set to 0 to
// disable trough subsampling.
#ifndef MD_TROUGH_MAX_ABS
#define MD_TROUGH_MAX_ABS 0.05f
#endif
constexpr float kTroughMaxAbs = MD_TROUGH_MAX_ABS;

// Reconstruct the FP64 grid coordinate used by the coarse scan.
__host__ __device__ __forceinline__ double scan_c(double cmin, double dc, int i) {
  return cmin + dc * static_cast<double>(i);
}

// Uniform bisection count for |c1-c2| <= c_tol_rel*c at the grid minimum.
// The secular-value acceptance threshold is an independent parameter.
inline int refine_iters(double dc, double tol) {
  if (!(tol > 0.0) || !(dc > tol)) return 1;
  int n = static_cast<int>(std::ceil(std::log2(dc / tol)));
  if (n < 1) n = 1;
  if (n > 60) n = 60;
  return n;
}

// Batch-wide refinement counters; trace-specific diagnostics use flags.
enum RefineStat {
  kStatBrackets = 0,       // sign-change brackets found and bisected
  kStatRejectedSign = 1,   // bracket failed the FP64 sign confirmation
  kStatRejectedF = 2,      // |F| at the root above f_accept (only when enabled)
  kStatTroughs = 3,        // point-troughs that TRIGGERED sub-sampling
  kStatHiddenRoots = 4,    // roots actually recovered by sub-sampling
  kStatRelocated = 5,      // brackets FP64 rejected, recovered from a neighbour
  kStatCount = 6,
};

// Per-(model, period) diagnostic bits accumulated across mode threads.
enum TraceFlag {
  // FP64 rejected a bracket and no neighbour supplied a replacement.
  kFlagUnresolvedBracket = 1u << 0,
  // FP64 recovered a rejected bracket from a neighbouring cell.
  kFlagRelocated = 1u << 1,
};

// Recheck an FP32 bracket in FP64 from an adjacent cell when necessary.
__device__ __forceinline__ bool retry_bracket(
    int cell, int dir, int i_last, double cmin, double dc, double omega,
    const double *sh, const double *svp, const double *svs, const double *srho,
    int L, double &lo, double &hi, double &flo) {
  const int c2 = cell + dir;
  if (c2 < 0 || c2 > i_last) return false;
  const double a = scan_c(cmin, dc, c2);
  const double b = scan_c(cmin, dc, c2 + 1);
  const double fa =
      dunkin_secular_omega<double, true>(sh, svp, svs, srho, L, omega, a);
  const double fb =
      dunkin_secular_omega<double, true>(sh, svp, svs, srho, L, omega, b);
  if ((fa < 0.0) == (fb < 0.0)) return false;
  lo = a;
  hi = b;
  flo = fa;
  return true;
}

// Bisect a bracket whose endpoint values are already known, and store the root
// if there is room for it.  `lo`/`hi` bound a sign change; `flo` is F(lo).
__device__ __forceinline__ void bisect_store(
    double *slot, int &mode, int nmodes, double lo, double hi, double flo,
    double omega, int iters, const double *sh, const double *svp,
    const double *svs, const double *srho, int L, double f_accept,
    unsigned long long *stats) {
  for (int k = 0; k < iters; ++k) {
    const double mid = 0.5 * (lo + hi);
    const double fm =
        dunkin_secular_omega<double, true>(sh, svp, svs, srho, L, omega, mid);
    if ((fm < 0.0) == (flo < 0.0)) {
      lo = mid;
      flo = fm;
    } else {
      hi = mid;
    }
  }
  const double root = 0.5 * (lo + hi);

  // Optional normalized secular-value acceptance; disabled at f_accept=1.
  if (f_accept < 1.0) {
    const double fr =
        dunkin_secular_omega<double, true>(sh, svp, svs, srho, L, omega, root);
    if (fabs(fr) > f_accept) {
      if (mode < nmodes) slot[mode] = CUDART_NAN;
      ++mode;
      atomicAdd(&stats[kStatRejectedF], 1ULL);
      return;
    }
  }
  if (mode < nmodes) slot[mode] = root;
  ++mode;
}

// Launch bounds cap blocks at 256 threads; the C ABI enforces the same limit.
#ifndef MD_REFINE_MINBLOCKS
#define MD_REFINE_MINBLOCKS 3
#endif
#if MD_REFINE_MINBLOCKS > 0
#define MD_REFINE_LB __launch_bounds__(256, MD_REFINE_MINBLOCKS)
#else
#define MD_REFINE_LB
#endif

__global__ void MD_REFINE_LB refine_kernel(const float *__restrict__ scan,
                              const double *__restrict__ thickness,
                              const double *__restrict__ vp,
                              const double *__restrict__ vs,
                              const double *__restrict__ rho,
                              const float *__restrict__ periods, int L, int nf,
                              int nc, int nmodes, int iters, double cmin,
                              double cmax, double f_accept,
                              double *__restrict__ out,
                              unsigned int *__restrict__ flags,
                              unsigned long long *__restrict__ stats) {
  extern __shared__ double smem[];
  double *sh = smem;
  double *svp = sh + L;
  double *svs = svp + L;
  double *srho = svs + L;

  const int b = blockIdx.x;
  const size_t base = static_cast<size_t>(b) * static_cast<size_t>(L);
  for (int i = threadIdx.x; i < L; i += blockDim.x) {
    sh[i] = thickness[base + i];
    svp[i] = vp[base + i];
    svs[i] = vs[base + i];
    srho[i] = rho[base + i];
  }
  __syncthreads();


  const double dc = md::scan_dc<double>(cmin, cmax, nc);
  const int total = nf * nmodes;
  const float *scanb = scan + static_cast<size_t>(b) * nf * nc;
  double *outb = out + static_cast<size_t>(b) * nf * nmodes;

  for (int t = threadIdx.x; t < total; t += blockDim.x) {
    const int f = t / nmodes;
    const int m = t - f * nmodes;
    const float *row = scanb + static_cast<size_t>(f) * nc;
    // Output layout is [model][mode][period], NOT [model][period][mode]: the
    // public contract is phase_velocity.shape == [B, nmodes, T] (spec section
    // 2), and writing it that way costs nothing here while a Python-side
    // transpose of a large batch would copy hundreds of MB per call.
    double *slot = outb + static_cast<size_t>(m) * nf + f;
    *slot = CUDART_NAN;
    const double period = static_cast<double>(__ldg(periods + f));
    // Reuse one angular frequency throughout this period's refinement.
    const double omega = two_pi<double>() / period;

    // Pass 1 locates candidates without bisecting. The half-space S velocity
    // caps the trapped-mode domain and is shared with the coarse scan.
    const double c_cap = svs[L - 1];
    // The same function Stage 1 uses to decide how much of the row to fill.
    // If the two ever disagreed, the reader would consume a cell that was never
    // written -- which is why neither side spells the bound out itself.
    const int i_last = md::scan_i_last<double>(c_cap, cmin, cmax, nc);

    int mode = 0;
    int cell = -1;                                 // sign-change cell for mode m
    double h_lo = 0.0, h_hi = 0.0, h_flo = 0.0;    // or the hidden bracket
    bool have_hidden = false;
    // Trigger count, accumulated in a register and flushed with one atomicAdd
    // in pass 2 -- the walk itself must stay free of atomics.
    int n_point = 0;
    for (int i = 0; i + 1 < nc && i <= i_last && cell < 0 && !have_hidden;) {
      if ((row[i] < 0.0f) != (row[i + 1] < 0.0f)) {
        if (mode == m) {
          cell = i;
          break;
        }
        ++mode;
        ++i;
        continue;
      }

      const int tc = i + 1;
      // A strict low-|F| point trough with no adjacent sign change defines a
      // two-cell window for hidden-pair subsampling.
      int wlo = -1, whi = -1;
      if (tc >= 1 && tc + 1 < nc && tc <= i_last) {
        const float am = fabsf(row[tc - 1]);       // |F[i]|
        const float bm = fabsf(row[tc]);           // |F[i+1]|
        const float cm = fabsf(row[tc + 1]);       // |F[i+2]|
        // Cell (tc-1, tc) == (i, i+1) already has no sign change; require cell
        // (tc, tc+1) to have none either, so the window [c_{tc-1}, c_{tc+1}]
        // holds no root the sign-change path can see.
        const bool no_sc_right = ((row[tc] < 0.0f) == (row[tc + 1] < 0.0f));
        if (no_sc_right && (am > bm) && (bm < cm) && (bm < kTroughMaxAbs)) {
          wlo = tc - 1;
          whi = tc + 1;
          ++n_point;
        }
      }
      if (wlo < 0) {
        ++i;
        continue;
      }

      const double c_a = scan_c(cmin, dc, wlo);
      const double c_b = scan_c(cmin, dc, whi);
      const double span = c_b - c_a;
      double prev_c = c_a;
      double prev_f =
          dunkin_secular_omega<double, true>(sh, svp, svs, srho, L, omega, prev_c);
      for (int j = 1; j <= kSubSamples + 1; ++j) {
        const double cj =
            c_a + span * static_cast<double>(j) /
                      static_cast<double>(kSubSamples + 1);
        const double fj =
            dunkin_secular_omega<double, true>(sh, svp, svs, srho, L, omega, cj);
        if ((prev_f < 0.0) != (fj < 0.0)) {
          if (mode == m) {
            h_lo = prev_c;
            h_hi = cj;
            h_flo = prev_f;
            have_hidden = true;
            break;
          }
          ++mode;
        }
        prev_c = cj;
        prev_f = fj;
      }
      // Resume after the window just swept, so no cell is examined twice.  The
      // point-trough window spans [i, i+2], so this resumes at cell (i+2, i+3).
      i = whi;
    }

    // ---- PASS 2: bisect whatever pass 1 located ----
    //
    // Record triggered troughs separately from recovered hidden roots.
    if (n_point) atomicAdd(&stats[kStatTroughs], (unsigned long long)n_point);

    int sink = 0;
    if (cell >= 0) {
      atomicAdd(&stats[kStatBrackets], 1ULL);
      const double lo = scan_c(cmin, dc, cell);
      const double hi = scan_c(cmin, dc, cell + 1);
      const double flo =
          dunkin_secular_omega<double, true>(sh, svp, svs, srho, L, omega, lo);
      const double fhi =
          dunkin_secular_omega<double, true>(sh, svp, svs, srho, L, omega, hi);
      // Confirm every FP32 bracket in FP64 before bisection.
      if ((flo < 0.0) == (fhi < 0.0)) {
        atomicAdd(&stats[kStatRejectedSign], 1ULL);
        // Probe adjacent cells, excluding cells already counted as sign changes
        // to avoid duplicate roots.
        const int c_left = cell - 1, c_right = cell + 1;
        const bool cand_left =
            c_left >= 0 && ((row[c_left] < 0.0f) == (row[c_left + 1] < 0.0f));
        const bool cand_right =
            c_right + 1 < nc && c_right <= i_last &&
            ((row[c_right] < 0.0f) == (row[c_right + 1] < 0.0f));

        double l_lo = 0.0, l_hi = 0.0, l_flo = 0.0;
        double r_lo = 0.0, r_hi = 0.0, r_flo = 0.0;
        const bool left =
            cand_left && retry_bracket(cell, -1, i_last, cmin, dc, omega, sh,
                                       svp, svs, srho, L, l_lo, l_hi, l_flo);
        const bool right =
            cand_right && retry_bracket(cell, +1, i_last, cmin, dc, omega, sh,
                                        svp, svs, srho, L, r_lo, r_hi, r_flo);
        if (left != right) {
          // Accept only an unambiguous single-sided recovery.
          atomicAdd(&stats[kStatRelocated], 1ULL);
          atomicOr(&flags[static_cast<size_t>(b) * nf + f], kFlagRelocated);
          bisect_store(slot, sink, nmodes, left ? l_lo : r_lo,
                       left ? l_hi : r_hi, left ? l_flo : r_flo, omega, iters,
                       sh, svp, svs, srho, L, f_accept, stats);
        } else {
          // Preserve NaN and flag an unresolved bracket.
          atomicOr(&flags[static_cast<size_t>(b) * nf + f],
                   kFlagUnresolvedBracket);
        }
      } else {
        bisect_store(slot, sink, nmodes, lo, hi, flo, omega, iters, sh, svp,
                     svs, srho, L, f_accept, stats);
      }
    } else if (have_hidden) {
      atomicAdd(&stats[kStatHiddenRoots], 1ULL);
      bisect_store(slot, sink, nmodes, h_lo, h_hi, h_flo, omega, iters, sh,
                   svp, svs, srho, L, f_accept, stats);
    }

    // A bracket can straddle the bound: its left edge is below the cap while
    // the root itself is above.  Drop those rather than renumbering.
    if (*slot == *slot && *slot >= c_cap) *slot = CUDART_NAN;
  }
}

// Shared memory for the four layer arrays.
inline size_t refine_smem(int L) {
  return static_cast<size_t>(4) * static_cast<size_t>(L) * sizeof(double);
}

}  // namespace md

#endif  // MODISP_CUDA_REFINE_CUH
