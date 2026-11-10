// BSD-3-Clause. SH recurrence follows CPS/disba; see THIRD_PARTY_NOTICES.md.
#pragma once
#include <cfloat>
#include "common/dunkin.hpp"

namespace md {
#ifndef __CUDACC__
using std::isfinite; using std::fmin; using std::fmax; using std::fabs;
#endif
struct Medium {
  const double *h,*vp,*vs,*rho,*ivp,*ivs;
  int L,wave; // wave=0: coupled P-SV; wave=1: SH
  // Optional per-trace constants. Never cached by Python object address.
  const double *ka=nullptr,*kb=nullptr,*gammk=nullptr;
  // Experimental fast path: density reciprocals are model constants. The
  // `secular` selector is 0 for the released recurrence and 1 for the fast
  // recurrence.
  const double *irho=nullptr,*irho2=nullptr;
  int secular=0;
  int refinement=1; // 0: legacy, 1: Brent, 2: Chandrupatla (per Solver)
};

MD_HD bool sh_has_guide(const Medium &m) {
  const int top=m.vs[0]==0.?1:0;
  for(int j=top;j<m.L-1;++j)if(m.h[j]>0.&&m.vs[j]<m.vs[m.L-1])return true;
  return false;
}

MD_HD bool ambiguous_sh_cutoff(const Medium &m,double c,double f) {
  return m.wave&&f==0.&&fabs(c-m.vs[m.L-1])<=64.*2.220446049250313e-16*m.vs[m.L-1];
}

template<class T>
MD_HD T sh_secular_values(const T *h,const T *vs,const T *rho,const T *prepared_kb,
                          int layers,T omega,T c,T *shape) {
  const int top=vs[0]==T(0) ? 1:0, last=layers-1;
  const T k=omega/c, kb=prepared_kb ? prepared_kb[last] : omega/vs[last];
  T e1=rho[last]*MD_SQRT((k+kb)*MD_FABS(k-kb));
  T e2=T(1)/(vs[last]*vs[last]);
  for (int j=last-1;j>=top;--j) {
    const T mu=rho[j]*vs[j]*vs[j], ka=prepared_kb ? prepared_kb[j] : omega/vs[j];
    const T r=MD_SQRT((k+ka)*MD_FABS(k-ka)), q=r*h[j];
    T cq,y,z;
    if(k<ka) {T sn; MD_SINCOS(q,&sn,&cq);y=sn/r;z=-r*sn;}
    else if(k==ka) {cq=T(1);y=h[j];z=T(0);}
    else {const T f=q<T(16)?MD_EXP(-T(2)*q):T(0);cq=(T(1)+f)*T(.5);y=(T(1)-f)*T(.5)/r;z=(T(1)-f)*T(.5)*r;}
    T u=e1*cq+e2*mu*z, v=e1*y/mu+e2*cq;
    T s=MD_FMAX(MD_FABS(u),MD_FABS(v)); if(s<(sizeof(T)==sizeof(float)?T(1e-30):T(1e-40)))s=T(1);
    e1=u/s;e2=v/s;
  }
  if(shape)*shape=e1/MD_SQRT(e1*e1+e2*e2+shape_small<T>());
  return e1;
}

MD_HD double sh_secular(const Medium &m,double omega,double c,double *shape=nullptr) {
  return sh_secular_values(m.h,m.vs,m.rho,m.kb,m.L,omega,c,shape);
}

struct ShapeSample {double f,g;};
MD_HD ShapeSample shape_evaluate(const Medium &m,double omega,double c){
  ShapeSample s;
  if(m.wave){s.f=sh_secular(m,omega,c,&s.g);return s;}
  const bool water=m.vs[0]==0.;
  if(m.secular==1){
    if(m.ka&&m.kb&&m.gammk&&m.irho&&m.irho2)
      s.f=dunkin_secular_fast<double,true>(m.h,m.vp,m.vs,m.rho,m.L,omega/c,omega,water,false,m.ka,m.kb,m.gammk,m.irho,m.irho2,&s.g);
    else
      s.f=dunkin_secular_fast<double,false>(m.h,m.vp,m.vs,m.rho,m.L,omega/c,omega,water,false,nullptr,nullptr,nullptr,m.irho,m.irho2,&s.g);
  }else s.f=dunkin_secular<double,true>(m.h,m.vp,m.vs,m.rho,m.L,omega/c,omega,water,false,m.ka,m.kb,m.gammk,&s.g);
  return s;
}

MD_HD double evaluate(const Medium &m,double omega,double c) {
  if(m.wave==1)return sh_secular(m,omega,c);
  if(m.secular==1){
    const bool water=m.vs[0]==0.;
    if(m.ka&&m.kb&&m.gammk&&m.irho&&m.irho2)
      return dunkin_secular_fast<double,true>(m.h,m.vp,m.vs,m.rho,m.L,omega/c,omega,water,false,m.ka,m.kb,m.gammk,m.irho,m.irho2);
    return dunkin_secular_fast<double,false>(m.h,m.vp,m.vs,m.rho,m.L,omega/c,omega,water,false,nullptr,nullptr,nullptr,m.irho,m.irho2);
  }
  return dunkin_secular<double,true>(m.h,m.vp,m.vs,m.rho,m.L,omega/c,omega,m.vs[0]==0.,false,m.ka,m.kb,m.gammk);
}

#ifdef __CUDACC__
// Keep search geometry in FP64 so distinct brackets stay ordered. Only the
// secular and phase evaluators read the converted model and run in FP32.
struct FloatMedium : Medium {
  const float *fh,*fvp,*fvs,*frho,*firho,*firho2;
  float rayleigh_stable_below;
  const float *fka;
};

static __device__ __noinline__ float evaluate_evanescent(const FloatMedium &m,float omega,float c,float *shape) {
  return dunkin_secular_fast<float,false,true>(m.fh,m.fvp,m.fvs,m.frho,m.L,omega/c,omega,m.fvs[0]==0.f,
                                               false,nullptr,nullptr,nullptr,m.firho,m.firho2,shape);
}

__device__ __forceinline__ float evaluate_float(const FloatMedium &m,float omega,float c,float *shape) {
  const float *fkb=m.fka?m.fka+m.L:nullptr;
  if(m.wave)
    return sh_secular_values(m.fh,m.fvs,m.frho,fkb,m.L,omega,c,shape);
  if(c<m.rayleigh_stable_below)return evaluate_evanescent(m,omega,c,shape);
  const bool water=m.fvs[0]==0.f;
  const float *fgammk=fkb?fkb+m.L:nullptr;
  if(m.secular==1)
    return dunkin_secular_fast<float,false>(m.fh,m.fvp,m.fvs,m.frho,m.L,omega/c,omega,water,false,
                                          m.fka,fkb,fgammk,m.firho,m.firho2,shape);
  return dunkin_secular<float,true>(m.fh,m.fvp,m.fvs,m.frho,m.L,omega/c,omega,water,false,
                                   m.fka,fkb,fgammk,shape);
}

__device__ __forceinline__ ShapeSample shape_evaluate(const FloatMedium &m,double omega,double c) {
  float g;
  const float f=evaluate_float(m,static_cast<float>(omega),static_cast<float>(c),&g);
  return {f,g};
}

__device__ __forceinline__ double evaluate(const FloatMedium &m,double omega,double c) {
  return evaluate_float(m,static_cast<float>(omega),static_cast<float>(c),nullptr);
}

MD_HD bool ambiguous_sh_cutoff(const FloatMedium &m,double c,double f) {
  return m.wave&&f==0.&&static_cast<float>(c)==m.fvs[m.L-1];
}

MD_HD double coordinate_floor(const FloatMedium &) { return 4.*FLT_EPSILON; }
#endif
MD_HD double coordinate_floor(const Medium &) { return 16.*DBL_EPSILON; }

MD_HD double density_base(const Medium &m,double c) {
  const double ic=1.0/(c*c);double n=0.;
  for(int j=0;j<m.L-1;++j) {
    if(m.h[j]==0.)continue;
    if(m.ivs[j]>ic)n+=m.h[j]*MD_SQRT(m.ivs[j]-ic);
    if(m.wave==0 && m.ivp[j]>ic)n+=m.h[j]*MD_SQRT(m.ivp[j]-ic);
  }
  return 2.*n;
}

MD_HD double rayleigh_reference(double vp,double vs) {
  const double ratio=(vs/vp)*(vs/vp);double l=0.,h=1.;
  for(int j=0;j<56;++j){double x=(l+h)*.5;double f=((x-8.)*x+24.-16.*ratio)*x-16.*(1.-ratio);if(f<0.)l=x;else h=x;}
  return vs*MD_SQRT((l+h)*.5);
}

MD_HD double scholte_reference(double vf,double rf,double vp,double vs,double rs) {
  const double ratio=(vs/vp)*(vs/vp),qf=(vs/vf)*(vs/vf);
  double l=0.,h=MD_FMAX(0.,fmin(1.,vf/vs)*(1.-1e-12));
  for(int j=0;j<64;++j) {
    const double x=(l+h)*.5,x2=x*x,A=MD_SQRT(fmax(0.,1.-ratio*x2)),B=MD_SQRT(fmax(0.,1.-x2));
    // Rayleigh/Scholte secular equation divided by x^2, rationalized at x=0.
    const double f=4.*(1.+ratio-ratio*x2)/(1.+A*B)-4.+x2+(rf/rs)*x2*A/MD_SQRT(fmax(1e-30,1.-qf*x2));
    if(f<0.)l=x;else h=x;
  }
  return vs*(l+h)*.5;
}

MD_HD double automatic_lower(const Medium &m) {
  double seed=1e300;
  const int top=m.vs[0]==0. ? 1:0;
  for(int j=top;j<m.L;++j) {
    if(m.h[j]==0. && j<m.L-1)continue;
    const double vs=m.vs[j],vp=m.vp[j];
    if(!m.wave && !top && vs>0. && .68*vs>seed){
      // For (Vs/Vp)^2 <= 3/4, the Rayleigh cubic is negative throughout
      // [0, .68^2], so this layer cannot lower the current minimum.
      const double ratio=vs/vp;
      if(ratio>0. && ratio*ratio<=.75)continue;
    }
    double x=m.wave ? vs : rayleigh_reference(vp,vs);
    if(!m.wave && top)x=fmin(x,scholte_reference(m.vp[0],m.rho[0],vp,vs,m.rho[j]));
    seed=fmin(seed,x);
  }
  return .125*seed;
}

// A model-only monotone table brackets the inverse before any actual N(c)
// iterations. All frequency/density levels reuse it; the final inverse still
// evaluates the exact FP64 density, so this is not table interpolation physics.
MD_HD double inverse_density(const Medium &m,const double *tx,const double *tn,int nt,double target) {
  int l=0,h=nt-1;
  while(h-l>1){int q=(l+h)/2;if(tn[q]<target)l=q;else h=q;}
  double a=tx[l],b=tx[h],fa=tn[l],fb=tn[h];
  const double eps=1e-9*fmax(1.,fabs(target));
  for(int j=0;j<48;++j){
    double x=a+(b-a)*(target-fa)/(fb-fa);
    if(j%4==3 || !isfinite(x) || x<=a || x>=b)x=(a+b)*.5;
    const double f=density_base(m,x);
    if(fabs(f-target)<=eps || b-a<=8e-15*fmax(1.,fabs(x)))return x;
    if(f<target){a=x;fa=f;}else {b=x;fb=f;}
  }
  return (a+b)*.5;
}

MD_HD double dense_workload(const Medium &m,double period,double lo,double hi) {
  double need=32.*density_base(m,hi)/period;
  // The first propagating acoustic roots approach Vp_water quadratically in
  // water optical thickness. A linear N estimate alone cannot resolve them.
  if(!m.wave&&m.vs[0]==0.){
    const double optical=2.*m.h[0]/(period*m.vp[0]);
    need=fmax(need,2.*fmax(0.,hi-lo)*optical*optical/m.vp[0]);
  }
  return need;
}

MD_HD bool roots_agree(double current,double previous,double tol,double floor=0.) {
  return isfinite(current)&&isfinite(previous)&&fabs(current-previous)<=fmax(4e-12,fmax(floor,.25*tol)*fabs(current));
}

// Ordered, one-to-one witness matching. This never merges root candidates by
// an absolute velocity gap. Witnesses above a complete requested prefix are
// outside this call's scope; every witness inside it must remain represented.
MD_HD bool contains_witnesses(const double *roots,int n,const double *witness,int nw,int K,double tol,double floor=0.){
  int j=0;
  for(int i=0;i<nw;++i){
    if(n==K&&n>0&&witness[i]>roots[n-1]+fmax(4e-12,fmax(floor,.25*tol)*fabs(roots[n-1])))break;
    while(j<n&&!roots_agree(roots[j],witness[i],tol,floor)&&roots[j]<witness[i])++j;
    if(j>=n||!roots_agree(roots[j],witness[i],tol,floor))return false;
    ++j;
  }
  return true;
}

template<class M>
MD_HD double refine_root_legacy(const M &m,double omega,double lo,double hi,double fl,double fh,double tol,int &evals) {
  if(lo==hi)return lo;
  const double target=fmax(coordinate_floor(m)*fmax(fabs(lo),fabs(hi)),fmin((hi-lo)/64.,.125*tol*fmax(fabs(lo),fabs(hi))));
  for(int j=0;j<100 && hi-lo>target;++j){
    double x=lo+(hi-lo)*(fabs(fl)/(fabs(fl)+fabs(fh)));
    if(j%3==2 || !isfinite(x) || x<=lo+.1*(hi-lo) || x>=hi-.1*(hi-lo))x=(lo+hi)*.5;
    const double f=evaluate(m,omega,x);++evals;
    if(!isfinite(f))return NAN;
    if(f==0.)return x;
    if((f<0.)==(fl<0.)){lo=x;fl=f;}else{hi=x;fh=f;}
  }
  return hi-lo<=target ? (lo+hi)*.5:NAN;
}
} // namespace md

#include "common/refinement.hpp"
#include "common/love_phase.hpp"
