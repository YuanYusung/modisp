// Accuracy presets shared by the CPU and CUDA backends.
//
// Evaluation/refinement remain FP64. High retains the full 0.2.1 schedule.
// Balanced/fast use logarithmic velocity scans at half/full close_root_rtol;
// their recovered ranks and possible close-cluster omissions are flagged.
#ifndef MODISP_COMMON_ACCURACY_HPP
#define MODISP_COMMON_ACCURACY_HPP

namespace md {

enum class Accuracy : int { High = 0, Balanced = 1, Fast = 2, Uniform = 3 };

struct AccuracyPolicy {
  Accuracy accuracy = Accuracy::High;
  // Highest early-accept level. 99 disables early acceptance at that level.
  int accept_level1 = 99;
  int accept_level2 = 99;
  double period_threshold = 0.0;  // level-1 zone; level 2 is always allowed
  double gap_factor = 0.0;        // adaptive root gap > gap_factor * local bracket width
  double dense_gap_factor = 0.0;  // dense root gap > dense_gap_factor * grid cell width
  double trough_threshold = 0.0;  // normalized |F| trough that signals a hidden pair
  int dense_floor4 = 32768;
  int dense_floor5 = 32768;
  int dense_factor4 = 1;
  int dense_factor5 = 2;
  bool verify_safe_skip = false;
  bool allow_dense_early = false;
  int cpu_dense_tile = 2048, cuda_dense_tile = 2048;
  bool reuse_refinement = false;
  double close_root_rtol=.005;
};

inline AccuracyPolicy make_accuracy_policy(int accuracy, double period_threshold, double close_root_rtol=.005) {
  AccuracyPolicy p;
  p.accuracy = static_cast<Accuracy>(accuracy);
  p.close_root_rtol=close_root_rtol;
  // Observation grids are selected by the solver; these legacy schedule
  // fields preserve the high path. period_threshold is compatibility metadata.
  if(p.accuracy==Accuracy::Balanced || p.accuracy==Accuracy::Fast){
    p.period_threshold=period_threshold>0.0?period_threshold:(p.accuracy==Accuracy::Balanced?1.0:0.63);
    p.cpu_dense_tile=p.accuracy==Accuracy::Balanced?64:32;
    // Smaller GPU tiles / larger caches were measured and rejected: no
    // material speed-up on A800, while memory and launch work increased.
    p.reuse_refinement=false;
  }
  return p;
}

inline const char *accuracy_name(Accuracy a) {
  switch (a) {
    case Accuracy::High: return "high";
    case Accuracy::Balanced: return "balanced";
    case Accuracy::Fast: return "fast";
    case Accuracy::Uniform: return "uniform";
  }
  return "unknown";
}

#ifdef __CUDACC__
__host__ __device__
#endif
inline bool has_close_roots(const double *r,int n,double tol){
  for(int i=1;i<n;++i)if(r[i]-r[i-1]<=tol*r[i])return true;
  return false;
}
}  // namespace md

#endif  // MODISP_COMMON_ACCURACY_HPP
