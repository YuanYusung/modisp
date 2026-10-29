// BSD-3-Clause. Independently implemented from Zhu et al. (2026),
// doi:10.1785/0120260071, equations 13--14. No upstream code is included.
#pragma once
#include "common/physics.hpp"

namespace md {
struct ShapeNode {double a,b;ShapeSample s0,sm,s1;int depth;};
constexpr int shape_queue=32,shape_depth=48,shape_work=524288;
constexpr int shape_storage_doubles=(2*shape_queue*sizeof(ShapeNode)+2*shape_queue*sizeof(ShapeSample))/sizeof(double);

MD_HD bool shape_ratio(double a,double b,double scale,double eta){
  const double floor=64.*2.220446049250313e-16*scale;
  return fabs(a-b)<=eta*fmax(fabs(a+b),floor)+floor;
}

// The three integral comparisons are used on a signed, smoothly rescaled
// real secular function. This adapts the paper's magnitude-GRT criterion;
// cancellation gets a roundoff guard rather than an unchecked 0/0 ratio.
MD_HD bool shape_three(double a,double b,double c,double d,double e,double eta,double conditioning){
  const double q11=5.*a+8.*c-e,q12=-a+8.*c+5.*e;
  const double q21=2.*a+8.*b+2.*c,q22=2.*c+8.*d+2.*e;
  return shape_ratio(q11,q21,16.*conditioning,eta)&&shape_ratio(q12,q22,16.*conditioning,eta)&&shape_ratio(q11+q12,q21+q22,32.*conditioning,eta);
}
MD_HD bool shape_consistent(const ShapeSample *s,double eta,bool variation=true){
  double scale=0.,nearest=INFINITY,low=INFINITY,high=-INFINITY;
  for(int j=0;j<5;++j){if(!isfinite(s[j].f)||!isfinite(s[j].g))return false;scale=fmax(scale,fabs(s[j].g));nearest=fmin(nearest,fabs(s[j].g));low=fmin(low,s[j].g);high=fmax(high,s[j].g);}
  if(scale==0.)return false;
  const double a=s[0].g/scale,b=s[1].g/scale,c=s[2].g/scale,d=s[3].g/scale,e=s[4].g/scale;
  if(!shape_three(a,b,c,d,e,eta,1.))return false;
  if(!variation||(scale<.9&&nearest>.1*scale))return true;
  // A large DC component can hide a weak buried-mode feature in a relative
  // integral test. An apparently smooth approach to zero can also conceal
  // a pair just before the endpoint. Test centered variation in either case;
  // the conditioning factor carries the original FP64 roundoff into it.
  const double span=(high-low)/scale;
  return span<=1e-8||shape_three((a-c)/span,(b-c)/span,0.,(d-c)/span,(e-c)/span,eta,1./span);
}

MD_HD bool shape_finite(const ShapeSample &s){return isfinite(s.f)&&isfinite(s.g);}

// Locate a suspicious same-sign extremum. Only original-function brackets
// produced by this proposal can be accepted as roots; amplitude alone cannot.
MD_HD ShapeSample shape_minimum(const Medium &m,double omega,double a,double b,double sign,double tol,double &position,int &evals){
  constexpr double phi=.6180339887498948482;
  double x=b-phi*(b-a),y=a+phi*(b-a);
  ShapeSample sx=shape_evaluate(m,omega,x),sy=shape_evaluate(m,omega,y);evals+=2;
  for(int j=0;j<32;++j){
    if(!shape_finite(sx)||!shape_finite(sy))break;
    if(sign*sx.f<0.||sign*sy.f<0.||b-a<=.125*tol*fmax(fabs(a),fabs(b)))break;
    if(sign*sx.g<sign*sy.g){b=y;y=x;sy=sx;x=b-phi*(b-a);sx=shape_evaluate(m,omega,x);}
    else{a=x;x=y;sx=sy;y=a+phi*(b-a);sy=shape_evaluate(m,omega,y);}
    ++evals;
  }
  if(sign*sx.g<sign*sy.g){position=x;return sx;}position=y;return sy;
}
MD_HD double shape_seed(const double *x,const double *n,int nt,double target){
  int lo=0,hi=nt-1;while(hi-lo>1){int mid=(lo+hi)/2;if(n[mid]<target)lo=mid;else hi=mid;}
  const double den=n[hi]-n[lo];
  return den>0.?x[lo]+(x[hi]-x[lo])*(target-n[lo])/den:x[lo];
}

MD_HD double shape_exact_seed(const Medium &m,const double *x,const double *n,int nt,double target,double step){
  int lo=0,hi=nt-1;while(hi-lo>1){int mid=(lo+hi)/2;if(n[mid]<target)lo=mid;else hi=mid;}
  double a=x[lo],b=x[hi],ratio=(target-n[lo])/(n[hi]-n[lo]);
  double c=a+(b-a)*(n[lo]==0.?ratio*ratio:ratio);
  const double tolerance=fmin(1e-9*fmax(1.,fabs(target)),1e-3*step);
  for(int k=0;k<32;++k){
    const double ic=1./(c*c);double value=0.,slope=0.;
    for(int j=0;j<m.L-1;++j){
      if(m.h[j]==0.)continue;
      const double z=m.ivs[j]-ic;if(z>0.){const double r=MD_SQRT(z);value+=m.h[j]*r;slope+=m.h[j]/r;}
      if(!m.wave){const double z=m.ivp[j]-ic;if(z>0.){const double r=MD_SQRT(z);value+=m.h[j]*r;slope+=m.h[j]/r;}}
    }
    value*=2.;
    if(isfinite(value)&&fabs(value-target)<=tolerance)return c;
    if(!isfinite(value)||b-a<=8.*2.220446049250313e-16*fmax(fabs(a),fabs(b)))return NAN;
    if(value<target)a=c;else b=c;
    const double derivative=2.*slope/(c*c*c);
    double next=c-(value-target)/derivative;
    if(!isfinite(next)||next<=a||next>=b||k%8==7)next=a+.5*(b-a);
    c=next;
  }
  return NAN;
}
} // namespace md
