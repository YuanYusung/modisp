// Run with: c++ -std=c++17 -O2 -Icpp tools/check_continuation_window.cpp -o /tmp/check-window && /tmp/check-window
// Manufactured secular functions isolate the window audit from geophysical model assumptions.
#include "common/continuation.hpp"
#include <iostream>

namespace witness {
struct Medium : md::Medium { bool triplet=false; };

double evaluate(const Medium &m,double,double c){
  // Both extra roots lie on the left of the polished root. All six
  // ordinary interior samples and both bracket edges miss the extra crossings.
  return m.triplet?(c-1.46)*(c-1.48)*(c-2.):(c-2.);
}
md::ShapeSample shape_evaluate(const Medium &m,double omega,double c){
  const double f=evaluate(m,omega,c);
  return {f,f};
}
}

int main(){
  // The Hermite centers are still ordered in these two examples. One tangent
  // ordering reverses in the first, and the two rank envelopes overlap in
  // the second. A wide but isolated prediction is an ordinary fallback.
  const double pair[]={1.,1.1},right[]={0.,0.},reversed[]={2.,0.},overlap[]={1.2,1.2};
  double predicted[2],difference[2];
  for(const double *left:{reversed,overlap}){
    const auto reason=md::continuation_predict(pair,pair,left,right,2,.1,-.1,.5,2.,1e-5,
                                               predicted,difference);
    if(reason!=(md::P_WINDOW|md::P_REVIEW))return 4;
  }
  if(md::continuation_predict(pair,pair,overlap,right,1,.1,-.1,.5,2.,1e-5,
                              predicted,difference)!=md::P_WINDOW)return 5;
  witness::Medium m{};
  for(bool triplet:{false,true}){
    m.triplet=triplet;
    unsigned long long calls=0;
    const double a=1., b=3., root=2., tol=1.e-6;
    for(int side=0;side<2;++side){
      const double edge=side?b:a, sign=side?1.:-1.;
      const double end=root+std::copysign(tol*root,edge-root);
      for(int k=1;k<=3;++k){
        if(sign*witness::evaluate(m,1.,edge+k*(end-edge)/4.)<=0.)return 1;
      }
    }
    const unsigned status=md::continuation_window(m,1.,a,b,witness::evaluate(m,1.,a),
        witness::evaluate(m,1.,b),root,tol,0.,calls);
    if(status!=(triplet?(md::P_AUDIT_FAILED|md::P_REVIEW):0u))return 2;
    if(triplet&&calls<=6)return 3;
    std::cout<<(triplet?"Three roots":"One root")<<": status="<<status<<", evaluations="<<calls<<"\n";
  }
}
