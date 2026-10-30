// BSD-3-Clause. Unwrapped SH phase with the decaying solid half-space boundary.
#pragma once
namespace md {
MD_HD double phase_unwrap(double raw,double previous){
  return previous+atan2(sin(raw-previous),cos(raw-previous));
}
MD_HD double love_phase(const Medium &m,double omega,double c){
  const int last=m.L-1,top=m.vs[0]==0.?1:0;
  const double k=omega/c,khalf=omega/m.vs[last];
  if(!(k>khalf))return NAN;
  double scale=m.rho[last]*m.vs[last]*m.vs[last]*sqrt((k+khalf)*(k-khalf));
  double theta=two_pi<double>()*.125;
  for(int j=last-1;j>=top;--j){
    if(m.h[j]==0.)continue;
    const double mu=m.rho[j]*m.vs[j]*m.vs[j],kb=omega/m.vs[j];
    const double r=sqrt(fabs((k+kb)*(k-kb))),next=mu*(r>0.?r:1./m.h[j]);
    if(!(scale>0.)||!(next>0.)||!refinement_finite(scale/next))return NAN;
    theta=phase_unwrap(atan2((scale/next)*sin(theta),cos(theta)),theta);scale=next;
    if(k<kb)theta-=r*m.h[j];
    else{
      const double x=sin(theta),y=cos(theta);double u,t;
      if(r==0.){t=x;u=y+x*m.h[j]*scale/mu;}
      else{const double e=exp(-2.*r*m.h[j]),cq=.5*(1.+e),sn=.5*(1.-e);t=x*cq+y*sn;u=y*cq+x*sn;}
      // An opaque layer can cancel both scaled components. Do not infer a
      // winding number from a lost state; retain the scan fallback instead.
      if(fmax(fabs(t),fabs(u))<1e-14)return NAN;
      theta=phase_unwrap(atan2(t,u),theta);
    }
    if(!refinement_finite(theta)||fabs(theta)>1048576.)return NAN;
  }
  return theta;
}

MD_HD double love_phase_root(const Medium &m,double omega,double low,double high,
                             double pl,double ph,double level,double phase_tol,int &evaluations){
    double a=low,b=high,pa=pl,pb=ph;
    if(!(pa>=level&&pb<level))return NAN;
    const double stop=refinement_target(a,b,phase_tol);
    for(int j=0;j<160&&b-a>stop;++j){
      const double mid=a+.5*(b-a),p=love_phase(m,omega,mid);++evaluations;
      if(!refinement_finite(p))return NAN;
      if(p>level){a=mid;pa=p;}else{b=mid;pb=p;}
    }
    if(b-a>stop)return NAN;
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
