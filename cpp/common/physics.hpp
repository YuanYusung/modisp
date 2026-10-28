// BSD-3-Clause. SH recurrence follows CPS/disba; see THIRD_PARTY_NOTICES.md.
#pragma once
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
};

MD_HD bool sh_has_guide(const Medium &m) {
  const int top=m.vs[0]==0.?1:0;
  for(int j=top;j<m.L-1;++j)if(m.h[j]>0.&&m.vs[j]<m.vs[m.L-1])return true;
  return false;
}

MD_HD bool ambiguous_sh_cutoff(const Medium &m,double c,double f) {
  return m.wave&&f==0.&&fabs(c-m.vs[m.L-1])<=64.*2.220446049250313e-16*m.vs[m.L-1];
}

MD_HD double sh_secular(const Medium &m,double omega,double c) {
  const int top=m.vs[0]==0.0 ? 1:0, last=m.L-1;
  const double k=omega/c, kb=m.kb ? m.kb[last] : omega/m.vs[last];
  double e1=m.rho[last]*MD_SQRT((k+kb)*MD_FABS(k-kb));
  double e2=1.0/(m.vs[last]*m.vs[last]);
  for (int j=last-1;j>=top;--j) {
    const double mu=m.rho[j]*m.vs[j]*m.vs[j], ka=m.kb ? m.kb[j] : omega/m.vs[j];
    const double r=MD_SQRT((k+ka)*MD_FABS(k-ka)), q=r*m.h[j];
    double cq,y,z;
    if(k<ka) {double sn; MD_SINCOS(q,&sn,&cq);y=sn/r;z=-r*sn;}
    else if(k==ka) {cq=1.;y=m.h[j];z=0.;}
    else {const double f=q<16.?MD_EXP(-2.*q):0.;cq=(1.+f)*.5;y=(1.-f)*.5/r;z=(1.-f)*.5*r;}
    double u=e1*cq+e2*mu*z, v=e1*y/mu+e2*cq;
    double s=MD_FMAX(MD_FABS(u),MD_FABS(v)); if(s<1e-40)s=1.;
    e1=u/s;e2=v/s;
  }
  return e1;
}

MD_HD double evaluate(const Medium &m,double omega,double c) {
  if(m.wave==1)return sh_secular(m,omega,c);
  return dunkin_secular<double,true>(m.h,m.vp,m.vs,m.rho,m.L,omega/c,omega,m.vs[0]==0.,false,m.ka,m.kb,m.gammk);
}

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
    double x=m.wave ? m.vs[j] : rayleigh_reference(m.vp[j],m.vs[j]);
    if(!m.wave && top)x=fmin(x,scholte_reference(m.vp[0],m.rho[0],m.vp[j],m.vs[j],m.rho[j]));
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

MD_HD bool roots_agree(double current,double previous,double tol) {
  return isfinite(current)&&isfinite(previous)&&fabs(current-previous)<=fmax(4e-12,.25*tol*fabs(current));
}

// Ordered, one-to-one witness matching. This never merges root candidates by
// an absolute velocity gap. Witnesses above a complete requested prefix are
// outside this call's scope; every witness inside it must remain represented.
MD_HD bool contains_witnesses(const double *roots,int n,const double *witness,int nw,int K,double tol){
  int j=0;
  for(int i=0;i<nw;++i){
    if(n==K&&n>0&&witness[i]>roots[n-1]+fmax(4e-12,.25*tol*fabs(roots[n-1])))break;
    while(j<n&&!roots_agree(roots[j],witness[i],tol)&&roots[j]<witness[i])++j;
    if(j>=n||!roots_agree(roots[j],witness[i],tol))return false;
    ++j;
  }
  return true;
}

MD_HD double refine_root(const Medium &m,double omega,double lo,double hi,double fl,double fh,double tol,int &evals) {
  if(lo==hi)return lo;
  const double target=fmax(16.*2.220446049250313e-16*fmax(fabs(lo),fabs(hi)),fmin((hi-lo)/64.,.125*tol*fmax(fabs(lo),fabs(hi))));
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
