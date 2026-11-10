// BSD-3-Clause. Unwrapped SH phase with the decaying solid half-space boundary.
#pragma once
#include <type_traits>
namespace md {
template<class T>
MD_HD T phase_unwrap(T raw,T previous){
  return previous+atan2(sin(raw-previous),cos(raw-previous));
}
template<class T>
MD_HD T love_phase_values(const T *h,const T *vs,const T *rho,int layers,T omega,T c){
  const int last=layers-1,top=vs[0]==T(0)?1:0;
  const T k=omega/c,khalf=omega/vs[last];
  if(!(k>khalf))return NAN;
  T scale=rho[last]*vs[last]*vs[last]*sqrt((k+khalf)*(k-khalf));
  T theta=two_pi<T>()*T(.125);
  for(int j=last-1;j>=top;--j){
    if(h[j]==T(0))continue;
    const T mu=rho[j]*vs[j]*vs[j],kb=omega/vs[j];
    const T r=sqrt(fabs((k+kb)*(k-kb))),next=mu*(r>T(0)?r:T(1)/h[j]);
    if(!(scale>T(0))||!(next>T(0))||!refinement_finite(scale/next))return NAN;
    theta=phase_unwrap(atan2((scale/next)*sin(theta),cos(theta)),theta);scale=next;
    if(k<kb)theta-=r*h[j];
    else{
      const T x=sin(theta),y=cos(theta);T u,t;
      if(r==T(0)){t=x;u=y+x*h[j]*scale/mu;}
      else{const T e=exp(-T(2)*r*h[j]),cq=T(.5)*(T(1)+e),sn=T(.5)*(T(1)-e);t=x*cq+y*sn;u=y*cq+x*sn;}
      // An opaque layer can cancel both scaled components. Do not infer a
      // winding number from a lost state; retain the scan fallback instead.
      if(fmax(fabs(t),fabs(u))<(sizeof(T)==sizeof(float)?T(1e-6):T(1e-14)))return NAN;
      theta=phase_unwrap(atan2(t,u),theta);
    }
    if(!refinement_finite(theta)||fabs(theta)>T(1048576))return NAN;
  }
  return theta;
}

MD_HD double love_phase(const Medium &m,double omega,double c){
  return love_phase_values(m.h,m.vs,m.rho,m.L,omega,c);
}

#ifdef __CUDACC__
MD_HD double love_phase(const FloatMedium &m,double omega,double c){
  return love_phase_values(m.fh,m.fvs,m.frho,m.L,static_cast<float>(omega),static_cast<float>(c));
}
#endif

template<class M>
MD_HD double love_phase_root(const M &m,double omega,double low,double high,
                             double pl,double ph,double level,double phase_tol,int &evaluations){
    double a=low,b=high;
    if(!(pl>=level&&ph<level))return NAN;
    const double stop=refinement_target(m,a,b,phase_tol);
    for(int j=0;j<160&&b-a>stop;++j){
      const double mid=a+.5*(b-a),p=love_phase(m,omega,mid);++evaluations;
      if(!refinement_finite(p))return NAN;
      if(p>level)a=mid;else b=mid;
    }
    if(b-a>stop)return NAN;
    if constexpr (!std::is_same_v<M,Medium>) {
      // The independently evaluated phase and secular sign can cross at
      // different FP32 nodes. Give the sign check room for roundoff.
      const double padding=32.*FLT_EPSILON*fmax(fabs(a),fabs(b));
      a=fmax(low,a-padding);b=fmin(high,b+padding);
    }
    double fa=evaluate(m,omega,a),fb=evaluate(m,omega,b);evaluations+=2;
    if(!refinement_finite(fa)||!refinement_finite(fb)||
       (fa!=0.&&fb!=0.&&((fa<0.)==(fb<0.))))return NAN;
    const double root=refine_root(m,omega,a,b,fa,fb,phase_tol,evaluations);
    if(!refinement_finite(root)||root<low||root>high)return NAN;
  return root;
}

// Returns false on an invalid phase state/bracket or floating-point ambiguity.
// Then the caller reruns the existing scan, preserving its failure diagnostics.
MD_HD bool love_phase_roots(const Medium &m,double omega,double low,double high,
                           int K,double tol,double *roots,int &nr,int &evaluations){
  nr=0;
  if(!(high>low))return false;
  if(!sh_has_guide(m))return true;
  const double pi=two_pi<double>()*.5,pl=love_phase(m,omega,low),ph=love_phase(m,omega,high);
  evaluations+=2;
  if(!refinement_finite(pl)||!refinement_finite(ph)||ph>pl||fabs(pl)>1048576.||fabs(ph)>1048576.)return false;
  const double phase_tol=fmin(tol,1e-8);
  for(int n=static_cast<int>(floor(pl/pi));n*pi>ph&&nr<K;--n){
    const double root=love_phase_root(m,omega,low,high,pl,ph,n*pi,phase_tol,evaluations);
    if(!refinement_finite(root)||(nr&&root<=roots[nr-1]))return false;
    roots[nr++]=root;
  }
  return true;
}
} // namespace md
