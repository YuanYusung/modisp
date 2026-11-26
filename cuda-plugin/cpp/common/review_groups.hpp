// Group output for a confirmed FP64 trace. The packed input contains one
// model (h, vp, vs, rho) followed by its ordered roots, including guards.
#pragma once
#include <vector>
#include <pybind11/numpy.h>
#include "common/group_velocity.hpp"

namespace md {
struct ReviewTrace {
  int layers, wave, secular, known;
  double omega, lower, upper, rtol, close_rtol;
  MD_HD GroupResult group(const double *data, int j) const {
    if (j >= known) {
      GroupResult missing;
      missing.flags = GROUP_MISSING_ROOT;
      return missing;
    }
    const double *roots = data + 4 * layers;
    const double c = roots[j];
    if ((j && c - roots[j-1] <= close_rtol*c) ||
        (j+1 < known && roots[j+1]-c <= close_rtol*roots[j+1])) {
      GroupResult close;
      close.flags = GROUP_CLOSE_ROOT;
      return close;
    }
    const double lo = j ? .5*(roots[j-1]+c) : lower;
    const double hi = j+1 < known ? .5*(c+roots[j+1]) : upper;
    Medium m{data, data+layers, data+2*layers, data+3*layers, nullptr, nullptr, layers, wave};
    m.secular = secular;
    return group_velocity(m, omega, c, lo, hi, rtol, close_rtol);
  }
};
} // namespace md

using ReviewInput = pybind11::array_t<double, pybind11::array::c_style>;
void evaluate_review_groups(const double *data, size_t size, md::ReviewTrace trace,
                            int modes, md::GroupResult *out);

inline void register_review_groups(pybind11::module_ &m) {
  namespace py = pybind11;
  m.def("_review_groups", [](ReviewInput packed, int layers, int modes, int wave, int secular,
                            double period, double lo, double hi, double rtol, double close_rtol) {
    if (packed.ndim()!=1 || layers<1 || layers>256 || modes<1 || modes>256 ||
        packed.size()<4*layers || packed.size()>4*layers+256 || !(period>0.))
      throw py::value_error("invalid review trace");
    md::ReviewTrace trace{layers, wave, secular, static_cast<int>(packed.size()-4*layers),
                          md::two_pi<double>()/period, lo, hi, rtol, close_rtol};
    std::vector<md::GroupResult> result(modes);
    {
      py::gil_scoped_release release;
      evaluate_review_groups(packed.data(), packed.size(), trace, modes, result.data());
    }
    py::array_t<double> velocities(modes);
    py::array_t<unsigned> flags(modes);
    int evaluations=0;
    for (int j=0; j<modes; ++j) {
      velocities.mutable_data()[j]=result[j].velocity;
      flags.mutable_data()[j]=result[j].flags;
      evaluations+=result[j].evaluations;
    }
    return py::make_tuple(velocities, flags, evaluations);
  });
}
