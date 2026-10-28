// pybind11 bindings over the CUDA C ABI.
// Kernel headers define external symbols and belong to their implementation
// translation units; this file uses only the C interface.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstring>
#include <string>

namespace py = pybind11;

void bind_qe(py::module_ &m);

struct md_cuda_state;
struct md_refine_state;

extern "C" {
const char *md_cuda_last_error(void);
const char *md_cuda_device_name(void);
md_cuda_state *md_cuda_create(int B, int L);
int md_cuda_upload_models(md_cuda_state *, const double *, const double *,
                          const double *, const double *);
int md_cuda_upload_periods(md_cuda_state *, int, const double *);
int md_cuda_set_output(md_cuda_state *, int);
int md_cuda_run(md_cuda_state *, double, double, int);
int md_cuda_sync(void);
const float *md_cuda_scan_ptr(md_cuda_state *);
int md_cuda_download(md_cuda_state *, float *);
int md_cuda_kernel_report(int, int, char *, int);
void md_cuda_free(md_cuda_state *);

const char *md_refine_last_error(void);
md_refine_state *md_refine_create(int B, int L);
int md_refine_upload_models(md_refine_state *, const double *, const double *,
                            const double *, const double *);
int md_refine_upload_periods(md_refine_state *, int, const double *);
int md_refine_set_output(md_refine_state *, int, int);
int md_refine_run(md_refine_state *, const float *, double, double, double,
                  double, int, int);
int md_refine_sync(void);
int md_refine_download(md_refine_state *, double *);
int md_refine_stats(md_refine_state *, unsigned long long *, int);
int md_refine_stat_count(void);
int md_refine_flags(md_refine_state *, unsigned int *);
int md_refine_kernel_report(int, int, char *, int);
void md_refine_free(md_refine_state *);

void md_cuda_geometry(md_cuda_state *, int *, int *, int *, int *);
int md_cuda_run_range(md_cuda_state *, double *, double *, int *);
void md_refine_geometry(md_refine_state *, int *, int *, int *, int *, int *);
}

namespace {

// Refinement counter names in C ABI slot order.
const char *kStatNames[] = {"brackets", "rejected_sign", "rejected_f",
                            "troughs",  "hidden_roots", "relocated"};
constexpr int kStatCount = 6;

[[noreturn]] void raise(const char *err) {
  throw std::runtime_error(err && *err ? err : "unknown CUDA error");
}

// py::array::shape() returns a raw pointer, not a tuple, so it cannot be
// handed to py::str directly.
std::string shape_str(const py::array &a) {
  std::string s = "(";
  for (py::ssize_t i = 0; i < a.ndim(); ++i) {
    if (i) s += ", ";
    s += std::to_string(a.shape(i));
  }
  if (a.ndim() == 1) s += ",";
  return s + ")";
}

// Convert array-like input to contiguous float64 and validate its shape.
py::array_t<double, py::array::c_style | py::array::forcecast> as_2d(
    py::handle obj, py::ssize_t rows, py::ssize_t cols, const char *name) {
  auto a = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(obj);
  if (!a) {
    throw py::type_error(std::string(name) +
                         " must be an array-like of real numbers");
  }
  if (a.ndim() != 2 || a.shape(0) != rows || a.shape(1) != cols) {
    throw py::value_error(std::string(name) + " must have shape (" +
                          std::to_string(rows) + ", " + std::to_string(cols) +
                          "), got " + shape_str(a));
  }
  return a;
}

// Each stage owns its error buffer.
std::string report_of(int (*fn)(int, int, char *, int),
                      const char *(*err)(void), int L, int block) {
  char buf[512] = {0};
  if (fn(L, block, buf, sizeof(buf))) raise(err());
  return std::string(buf);
}

struct Geom {
  int B = 0, L = 0, nf = 0, nc = 0, nmodes = 0;
};

Geom scan_geometry(md_cuda_state *st) {
  Geom g;
  md_cuda_geometry(st, &g.B, &g.L, &g.nf, &g.nc);
  return g;
}

Geom refine_geometry(md_refine_state *st) {
  Geom g;
  md_refine_geometry(st, &g.B, &g.L, &g.nf, &g.nc, &g.nmodes);
  return g;
}

// ---------------------------------------------------------------------------
// Stage 1
// ---------------------------------------------------------------------------
class CoarseScan {
 public:
  CoarseScan(int B, int L) : B_(B), L_(L) {
    st_ = md_cuda_create(B, L);
    if (!st_) raise(md_cuda_last_error());
  }
  ~CoarseScan() {
    if (st_) md_cuda_free(st_);
  }
  CoarseScan(const CoarseScan &) = delete;
  CoarseScan &operator=(const CoarseScan &) = delete;

  void upload_models(py::object thickness, py::object vp, py::object vs,
                     py::object rho) {
    auto t = as_2d(thickness, B_, L_, "thickness");
    auto a = as_2d(vp, B_, L_, "vp");
    auto b = as_2d(vs, B_, L_, "vs");
    auto r = as_2d(rho, B_, L_, "rho");
    if (md_cuda_upload_models(st_, t.data(), a.data(), b.data(), r.data())) {
      raise(md_cuda_last_error());
    }
    uploaded_ = true;
  }

  void upload_periods(py::object periods) {
    auto t = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(periods);
    if (!t || t.ndim() != 1 || t.shape(0) <= 0) {
      throw py::value_error("periods must be a non-empty 1-D array");
    }
    if (md_cuda_upload_periods(st_, static_cast<int>(t.shape(0)), t.data())) {
      raise(md_cuda_last_error());
    }
    nf_ = static_cast<int>(t.shape(0));
  }

  void allocate(int nc) {
    if (md_cuda_set_output(st_, nc)) raise(md_cuda_last_error());
    nc_ = nc;
  }

  void run(double cmin, double cmax, int block = 256, bool sync = false) {
    if (!uploaded_) throw std::runtime_error("upload_models() first");
    if (md_cuda_run(st_, cmin, cmax, block)) raise(md_cuda_last_error());
    if (sync) this->sync();
  }

  void sync() {
    if (md_cuda_sync()) raise(md_cuda_last_error());
  }

  py::array_t<float> download() {
    if (!nc_) throw std::runtime_error("allocate() first");
    py::array_t<float> out({B_, nf_, nc_});
    if (md_cuda_download(st_, out.mutable_data())) raise(md_cuda_last_error());
    return out;
  }

  py::array_t<float> scan(double cmin, double cmax, int block = 256) {
    run(cmin, cmax, block, true);
    return download();
  }

  // Internal device handle for geometry-checked refinement without a host copy.
  md_cuda_state *raw() const { return st_; }

  std::string kernel_report(int block = 256) const {
    return report_of(md_cuda_kernel_report, md_cuda_last_error, L_, block);
  }


 private:
  md_cuda_state *st_ = nullptr;
  int B_, L_, nf_ = 0, nc_ = 0;
  bool uploaded_ = false;
};

// ---------------------------------------------------------------------------
// Stage 2 + 3
// ---------------------------------------------------------------------------
class Refine {
 public:
  Refine(int B, int L) : B_(B), L_(L) {
    st_ = md_refine_create(B, L);
    if (!st_) raise(md_refine_last_error());
  }
  ~Refine() {
    if (st_) md_refine_free(st_);
  }
  Refine(const Refine &) = delete;
  Refine &operator=(const Refine &) = delete;

  void upload_models(py::object thickness, py::object vp, py::object vs,
                     py::object rho) {
    auto t = as_2d(thickness, B_, L_, "thickness");
    auto a = as_2d(vp, B_, L_, "vp");
    auto b = as_2d(vs, B_, L_, "vs");
    auto r = as_2d(rho, B_, L_, "rho");
    if (md_refine_upload_models(st_, t.data(), a.data(), b.data(), r.data())) {
      raise(md_refine_last_error());
    }
  }

  void upload_periods(py::object periods) {
    auto t = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(periods);
    if (!t || t.ndim() != 1 || t.shape(0) <= 0) {
      throw py::value_error("periods must be a non-empty 1-D array");
    }
    if (md_refine_upload_periods(st_, static_cast<int>(t.shape(0)), t.data())) {
      raise(md_refine_last_error());
    }
    nf_ = static_cast<int>(t.shape(0));
  }

  void allocate(int nc, int nmodes) {
    if (md_refine_set_output(st_, nc, nmodes)) raise(md_refine_last_error());
    nc_ = nc;
    nmodes_ = nmodes;
  }

  void run(CoarseScan &scan, double cmin, double cmax, double c_tol_rel,
           double f_accept, int nmodes, int block, bool sync) {
    if (!nf_) throw std::runtime_error("upload_periods() first");
    if (!nmodes_) throw std::runtime_error("allocate() first");

    // Commit the requested mode count only after geometry and range checks.
    const int want = (nmodes > 0) ? nmodes : nmodes_;

    const float *ptr = nullptr;
    {
      CoarseScan *cs = &scan;
      const Geom a = scan_geometry(cs->raw());
      const Geom b = refine_geometry(st_);
      if (a.B != b.B || a.L != b.L || a.nf != b.nf || a.nc != b.nc) {
        throw py::value_error(
            "scan and refine geometries differ: scan has B=" +
            std::to_string(a.B) + " L=" + std::to_string(a.L) +
            " nf=" + std::to_string(a.nf) + " nc=" + std::to_string(a.nc) +
            ", refine has B=" + std::to_string(b.B) +
            " L=" + std::to_string(b.L) + " nf=" + std::to_string(b.nf) +
            " nc=" + std::to_string(b.nc));
      }
      // Both stages must use identical range and grid spacing.
      double scan_cmin = 0.0, scan_cmax = 0.0;
      int scan_nc = 0;
      if (md_cuda_run_range(cs->raw(), &scan_cmin, &scan_cmax, &scan_nc) != 0) {
        throw py::value_error(
            "this scan has not been run yet, so its output row holds nothing "
            "to refine; call run() on it first");
      }
      if (scan_cmin != cmin || scan_cmax != cmax) {
        throw py::value_error(
            "scan and refine ranges differ: scan ran over (" +
            std::to_string(scan_cmin) + ", " + std::to_string(scan_cmax) +
            "), refine was given (" + std::to_string(cmin) + ", " +
            std::to_string(cmax) +
            "). The range sets the grid spacing, so a difference moves every "
            "grid point and the two stages would not be talking about the same "
            "velocities");
      }
      ptr = md_cuda_scan_ptr(cs->raw());
    }
    if (!ptr) throw std::runtime_error("scan buffer is null");

    if (md_refine_run(st_, ptr, cmin, cmax, c_tol_rel, f_accept, want, block)) {
      // Re-read device geometry before reporting the error.
      nmodes_ = refine_geometry(st_).nmodes;
      raise(md_refine_last_error());
    }
    nmodes_ = refine_geometry(st_).nmodes;
    if (sync) this->sync();
  }

  void sync() {
    if (md_refine_sync()) raise(md_refine_last_error());
  }

  // Return [B, nmodes, T], sized from current device geometry.
  py::array_t<double> download() {
    const Geom g = refine_geometry(st_);
    if (!g.nmodes) throw std::runtime_error("allocate() first");
    B_ = g.B;
    nf_ = g.nf;
    nmodes_ = g.nmodes;
    py::array_t<double> out({B_, nmodes_, nf_});
    if (md_refine_download(st_, out.mutable_data())) raise(md_refine_last_error());
    return out;
  }

  py::dict stats() {
    // Keep Python counter names synchronized with the C ABI.
    if (md_refine_stat_count() != kStatCount)
      raise("refine stats: kStatNames is out of step with md::RefineStat");
    unsigned long long buf[kStatCount] = {0};
    if (md_refine_stats(st_, buf, kStatCount)) raise(md_refine_last_error());
    py::dict d;
    for (int i = 0; i < kStatCount; ++i) d[kStatNames[i]] = buf[i];
    return d;
  }

  // [B, nf] of per-trace flag words, from the last run only.  Unlike stats(),
  // which is a batch-wide total, this says WHICH traces went wrong -- a caller
  // cannot recover that from a counter.
  py::array_t<unsigned int> flags() {
    const Geom g = refine_geometry(st_);
    if (!g.B || !g.nf) throw std::runtime_error("allocate() first");
    py::array_t<unsigned int> out({g.B, g.nf});
    if (md_refine_flags(st_, out.mutable_data())) raise(md_refine_last_error());
    return out;
  }

  std::string kernel_report(int block = 256) const {
    return report_of(md_refine_kernel_report, md_refine_last_error, L_, block);
  }

 private:
  md_refine_state *st_ = nullptr;
  int B_, L_, nf_ = 0, nc_ = 0, nmodes_ = 0;
};

}  // namespace

PYBIND11_MODULE(_cuda, m) {
  m.attr("ABI_VERSION")=2;
  m.attr("VERSION")="0.2.1";
  bind_qe(m);
  m.doc() = "CUDA kernels for GPU-accelerated surface-wave dispersion forward "
            "modeling. Low-level stage interface; the public API is in "
            "modisp.__init__.";
  py::tuple stat_names(kStatCount);
  for (int i = 0; i < kStatCount; ++i) stat_names[i] = py::str(kStatNames[i]);
  m.attr("STAT_NAMES") = stat_names;

  m.def("device_name", []() {
    const char *n = md_cuda_device_name();
    return std::string(n ? n : "<unknown>");
  });

  py::class_<CoarseScan>(m, "CoarseScan",
                         "Stage 1: coarse |F| scan over a (frequency, c) grid.")
      .def(py::init<int, int>(), py::arg("B"), py::arg("L"))
      .def("upload_models", &CoarseScan::upload_models, py::arg("thickness"),
           py::arg("vp"), py::arg("vs"), py::arg("rho"))
      .def("upload_periods", &CoarseScan::upload_periods, py::arg("periods"))
      .def("allocate", &CoarseScan::allocate, py::arg("nc"))
      .def("run", &CoarseScan::run, py::arg("cmin"), py::arg("cmax"),
           py::arg("block") = 256, py::arg("sync") = false)
      .def("sync", &CoarseScan::sync)
      .def("download", &CoarseScan::download)
      .def("scan", &CoarseScan::scan, py::arg("cmin"), py::arg("cmax"),
           py::arg("block") = 256)
      .def("__enter__",
           [](CoarseScan &s) -> CoarseScan & { return s; },
           py::return_value_policy::reference)
      .def("__exit__", [](CoarseScan &, py::args) {})
      .def("kernel_report", &CoarseScan::kernel_report, py::arg("block") = 256);

  py::class_<Refine>(m, "Refine",
                     "Stages 2-3: locate roots from the Stage 1 scan.")
      .def(py::init<int, int>(), py::arg("B"), py::arg("L"))
      .def("upload_models", &Refine::upload_models, py::arg("thickness"),
           py::arg("vp"), py::arg("vs"), py::arg("rho"))
      .def("upload_periods", &Refine::upload_periods, py::arg("periods"))
      .def("allocate", &Refine::allocate, py::arg("nc"), py::arg("nmodes"))
      .def("run", &Refine::run, py::arg("scan"), py::arg("cmin"),
           py::arg("cmax"), py::arg("c_tol_rel") = 1e-6,
           py::arg("f_accept") = 1.0, py::arg("nmodes") = 0,
           py::arg("block") = 256, py::arg("sync") = true)
      .def("sync", &Refine::sync)
      .def("download", &Refine::download)
      .def("stats", &Refine::stats)
      .def("flags", &Refine::flags)
      .def("kernel_report", &Refine::kernel_report, py::arg("block") = 256)
      .def("__enter__",
           [](Refine &r) -> Refine & { return r; },
           py::return_value_policy::reference)
      .def("__exit__", [](Refine &, py::args) {});
}
