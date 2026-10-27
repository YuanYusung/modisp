// C ABI wrappers for Dunkin point and grid precision probes.

#include "common/dunkin.hpp"
#include <cstddef>
using std::size_t;

namespace {

// Shared body for the FP64 grid, differing only in the normalization switch.
template <bool NORMALIZE>
void grid_f64(const double *d, const double *vp, const double *vs,
              const double *rho, int nlayer, int nf, const double *periods,
              int nc, double cmin, double cmax, double *out) {
  const double dc = (nc > 1) ? (cmax - cmin) / static_cast<double>(nc - 1) : 0.0;
  for (int f = 0; f < nf; ++f) {
    const double period = periods[f];
    const double omega = 6.283185307179586476925286766559L / period;
    double *row = out + static_cast<size_t>(f) * nc;
    for (int ic = 0; ic < nc; ++ic) {
      const double c = cmin + dc * static_cast<double>(ic);
      row[ic] = md::dunkin_secular<double, NORMALIZE>(d, vp, vs, rho, nlayer,
                                                      omega / c, omega, 0, false);
    }
  }
}

}  // namespace

extern "C" {

// --- FP64, the validated reference (bit-comparable target vs DISBA) ---------

double md_dunkin_point(const double *d, const double *vp, const double *vs,
                       const double *rho, int nlayer, double period, double c,
                       int normalize) {
  const double omega=md::two_pi<double>()/period;
  if(normalize)return md::dunkin_secular<double,true>(d,vp,vs,rho,nlayer,omega/c,omega,0,false);
  return md::dunkin_secular<double,false>(d,vp,vs,rho,nlayer,omega/c,omega,0,false);
}

void md_dunkin_grid(const double *d, const double *vp, const double *vs,
                    const double *rho, int nlayer, int nf,
                    const double *periods, int nc, double cmin, double cmax,
                    int normalize, double *out) {
  if (normalize) {
    grid_f64<true>(d, vp, vs, rho, nlayer, nf, periods, nc, cmin, cmax, out);
  } else {
    grid_f64<false>(d, vp, vs, rho, nlayer, nf, periods, nc, cmin, cmax, out);
  }
}

// --- FP32, the candidate precision for the coarse scan ---------------------
//
// Model parameters are narrowed on entry (a real deployment would store them
// as FP32 for the scan); the results are widened to double on output so the
// caller can see exactly what the FP32 arithmetic produced.

void md_dunkin_grid_f32(const double *d, const double *vp, const double *vs,
                        const double *rho, int nlayer, int nf,
                        const double *periods, int nc, double cmin, double cmax,
                        int normalize, double *out) {
  const int n = nlayer;
  if (n > 64) return;
  float df[64], af[64], bf[64], rf[64];
  for (int i = 0; i < n; ++i) {
    df[i] = static_cast<float>(d[i]);
    af[i] = static_cast<float>(vp[i]);
    bf[i] = static_cast<float>(vs[i]);
    rf[i] = static_cast<float>(rho[i]);
  }
  const double dc = (nc > 1) ? (cmax - cmin) / static_cast<double>(nc - 1) : 0.0;
  for (int f = 0; f < nf; ++f) {
    const float period = static_cast<float>(periods[f]);
    const float omega = static_cast<float>(6.283185307179586476925286766559L) / period;
    double *row = out + static_cast<size_t>(f) * nc;
    for (int ic = 0; ic < nc; ++ic) {
      const float c = static_cast<float>(cmin + dc * static_cast<double>(ic));
      row[ic] = static_cast<double>(
          normalize ? md::dunkin_secular<float, true>(df, af, bf, rf, n, omega / c, omega, 0, false)
                    : md::dunkin_secular<float, false>(df, af, bf, rf, n, omega / c, omega, 0, false));
    }
  }
}

}  // extern "C"
