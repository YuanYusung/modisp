// Accuracy presets shared by the CPU and CUDA backends.
//
// The secular function, root refinement and diagnostics remain FP64 in every
// mode.  Presets only change how much sampling and independent verification is
// spent before a root set is accepted.  HIGH must reproduce the 0.2.1 schedule
// exactly; BALANCED and FAST add a spacing / hidden-pair safety test before
// accepting a coarser level and use a coarser first dense fallback.
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
};

inline AccuracyPolicy make_accuracy_policy(int accuracy, double period_threshold) {
  AccuracyPolicy p;
  p.accuracy = static_cast<Accuracy>(accuracy);
  switch (p.accuracy) {
    case Accuracy::High:
      p.accept_level1 = 99;
      p.accept_level2 = 99;
      p.period_threshold = 0.0;
      p.gap_factor = 0.0;
      p.dense_gap_factor = 0.0;
      p.trough_threshold = 0.0;
      p.dense_floor4 = 32768;
      p.dense_floor5 = 32768;
      p.dense_factor4 = 1;
      p.dense_factor5 = 2;
      p.verify_safe_skip = false;
      p.allow_dense_early = false;
      break;
    case Accuracy::Balanced:
      p.accept_level1 = 1;
      p.accept_level2 = 2;
      p.period_threshold = period_threshold > 0.0 ? period_threshold : 1.0;
      p.gap_factor = 2.0;
      p.dense_gap_factor = 4.0;
      p.trough_threshold = 0.05;
      p.dense_floor4 = 8192;
      p.dense_floor5 = 32768;
      p.dense_factor4 = 1;
      p.dense_factor5 = 1;
      p.verify_safe_skip = true;
      p.allow_dense_early = true;
      break;
    case Accuracy::Fast:
      p.accept_level1 = 1;
      p.accept_level2 = 2;
      p.period_threshold = period_threshold > 0.0 ? period_threshold : 0.63;
      p.gap_factor = 1.5;
      p.dense_gap_factor = 3.0;
      p.trough_threshold = 0.05;
      p.dense_floor4 = 4096;
      p.dense_floor5 = 16384;
      p.dense_factor4 = 1;
      p.dense_factor5 = 1;
      p.verify_safe_skip = true;
      p.allow_dense_early = true;
      break;
    case Accuracy::Uniform:
      p.accept_level1 = 99;
      p.accept_level2 = 99;
      p.period_threshold = 0.0;
      p.gap_factor = 0.0;
      p.dense_gap_factor = 0.0;
      p.trough_threshold = 0.0;
      p.dense_floor4 = 32768;
      p.dense_floor5 = 32768;
      p.dense_factor4 = 1;
      p.dense_factor5 = 2;
      p.verify_safe_skip = false;
      p.allow_dense_early = false;
      break;
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

}  // namespace md

#endif  // MODISP_COMMON_ACCURACY_HPP
