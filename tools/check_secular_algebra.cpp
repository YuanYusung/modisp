// Algebra-only review: compare direct contraction with all 25 matrix entries.
// Arbitrary VarOut coefficients test the contraction, not physical root completeness.
#include "../cpp/common/dunkin.hpp"
#include <random>
#include <iostream>
#include <limits>
int main() {
  std::mt19937_64 rng(20261029);
  std::uniform_real_distribution<double> u(-1.,1.);
  double worst=0.;
  for(int n=0;n<100000;++n) {
    md::VarOut<double> v{};
    v.a0=u(rng);v.cpcq=u(rng);v.cpy=u(rng);v.cpz=u(rng);
    v.cqw=u(rng);v.cqx=u(rng);v.xy=u(rng);v.xz=u(rng);
    v.wy=u(rng);v.wz=u(rng);
    double k2=std::exp(8.*u(rng)),gm=std::exp(8.*u(rng));
    double gam=gm*k2,rho=std::exp(5.*u(rng));
    double e[5],out[5],ca[5][5];
    for(double &z:e)z=u(rng);
    md::dunkin_dnka(k2,gam,gm,rho,v,ca);
    md::dunkin_apply_fast(k2,gam,gm,rho,v,1./rho,(1./rho)*(1./rho),e,out);
    for(int i=0;i<5;++i) {
      double ref=0.,scale=0.;
      for(int j=0;j<5;++j){ref+=e[j]*ca[j][i];scale+=std::abs(e[j]*ca[j][i]);}
      double err=std::abs(out[i]-ref)/std::max(scale,1e-300);
      worst=std::max(worst,err);
      if(!std::isfinite(out[i])||err>64.*std::numeric_limits<double>::epsilon())return 1;
    }
  }
  std::cout<<"100000 matrices / 500000 components passed; max error / sum(abs(terms)) = "<<worst<<"\n";
}
