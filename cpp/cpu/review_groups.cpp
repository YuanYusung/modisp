#include "common/review_groups.hpp"

void evaluate_review_groups(const double *data, size_t, md::ReviewTrace trace,
                            int modes, md::GroupResult *out) {
  for (int j=0; j<modes; ++j) out[j]=trace.group(data,j);
}

void bind_review_groups(pybind11::module_ &m) { register_review_groups(m); }
