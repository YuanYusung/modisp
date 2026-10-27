// Numerical-validation adapter to unmodified QEDispInv ed1b5dd sources.
// The author's solver and its separate license remain in an external cache.
// This adapter contains no QEDispInv implementation and is BSD-3-Clause.
#include <Eigen/Dense>
#include "disp.hpp"
#include <cmath>
#include <limits>
#include <exception>
#include <cstdio>

namespace { thread_local char last_error[512] = {}; }
extern "C" const char *qed_reference_error() { return last_error; }
extern "C" int qed_reference_roots(int L, const double *h, const double *vp,
    const double *vs, const double *rho, int nt, const double *periods,
    int modes, double cmin, double cmax, double *out) {
  try {
    Eigen::ArrayXXd model(L,5);
    double z=0.0;
    for (int j=0;j<L;++j) {
      model(j,0)=j+1; model(j,1)=z; model(j,2)=rho[j];
      model(j,3)=vs[j]; model(j,4)=vp[j]; z+=h[j];
    }
    Dispersion solver(model,false);
    for (int t=0;t<nt;++t) {
      for (int m=0;m<modes;++m) out[t*modes+m]=std::numeric_limits<double>::quiet_NaN();
      // Request extra roots so an explicit cmin can be applied after the
      // author's physical lower bound, without silently shifting the slice.
      auto roots=solver.search(1.0/periods[t],modes+32);
      int m=0;
      for (double c:roots) if (std::isfinite(c) && c>=cmin && c<cmax && c<vs[L-1]) {
        out[t*modes+m++]=c; if (m==modes) break;
      }
    }
    return 0;
  } catch (const std::exception &e) {
    std::snprintf(last_error,sizeof(last_error),"%s",e.what()); return 1;
  }
}

extern "C" int qed_benchmark_roots(int L, const double *h, const double *vp,
    const double *vs, const double *rho, int nt, const double *periods,
    int modes, double cmin, double cmax, double *out) {
  try {
    Eigen::ArrayXXd model(L,5);
    double z=0.0;
    for (int j=0;j<L;++j) {
      model(j,0)=j+1; model(j,1)=z; model(j,2)=rho[j];
      model(j,3)=vs[j]; model(j,4)=vp[j]; z+=h[j];
    }
    Dispersion solver(model,false);
    for (int t=0;t<nt;++t) {
      for (int m=0;m<modes;++m) out[t*modes+m]=std::numeric_limits<double>::quiet_NaN();
      auto roots=solver.search(1.0/periods[t],modes);
      int m=0;
      for (double c:roots) if (std::isfinite(c) && c>=cmin && c<cmax && c<vs[L-1]) {
        out[t*modes+m++]=c; if (m==modes) break;
      }
    }
    return 0;
  } catch (const std::exception &e) {
    std::snprintf(last_error,sizeof(last_error),"%s",e.what()); return 1;
  }
}
