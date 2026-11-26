// BSD-3-Clause. CPU scheduling and storage for two-sided frequency continuation.
#pragma once
#include "common/continuation.hpp"
#include <algorithm>
#include <vector>

struct CpuContinuation {
  std::vector<int> anchors, targets, left_slot;
  std::vector<double> roots, slopes;
  std::vector<unsigned char> safe, differentiated;

  // The periods may change between calls with the same geometry.
  bool prepare(const double *periods,int nf,int stride,int models,int k){
    anchors.clear();targets.clear();left_slot.resize(nf);
    for(int left=0;;){
      const int slot=static_cast<int>(anchors.size());
      anchors.push_back(left);left_slot[left]=slot;
      if(left==nf-1)break;
      const int right=md::continuation_next_anchor(periods,nf,left,stride);
      for(int f=left+1;f<right;++f){left_slot[f]=slot;targets.push_back(f);}
      left=right;
    }
    if(targets.empty())return false;
    const size_t records=static_cast<size_t>(models)*anchors.size();
    roots.resize(records*k);slopes.resize(records*k);
    safe.resize(records);differentiated.assign(records,0);
    std::fill(slopes.begin(),slopes.end(),NAN);
    return true;
  }
};

enum CpuContinuationCounter {
  CONT_ANCHORS=29, CONT_CANDIDATES, CONT_ACCEPTED, CONT_FALLBACKS,
  CONT_AUDIT_FAILED, CONT_EVALS, GROUP_REUSED, CONT_DERIVATIVE_ROOTS,
  CONT_DERIVATIVE_EVALS, CONT_PRESCREEN_SKIPPED_ROOTS, CPU_COUNTERS
};
