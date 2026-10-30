// BSD-3-Clause. Safeguarded Brent-Dekker and Chandrupatla FP64 refinement.
// Chandrupatla interpolation test adapted from SciPy; full BSD notice is in
// THIRD_PARTY_NOTICES.md. Shared host/device code; each Medium carries its Solver's selection.
#pragma once
namespace md {
MD_HD bool refinement_finite(double x){return x==x&&fabs(x)!=INFINITY;}
MD_HD double refinement_target(double a,double b,double tol){
  return fmax(16.*2.220446049250313e-16*fmax(fabs(a),fabs(b)),fmin((b-a)/64.,.125*tol*fmax(fabs(a),fabs(b))));
}
MD_HD double refine_root(const Medium &m,double w,double a,double b,double fa,double fb,double tol,int &ev){
  if(m.refinement==0)return refine_root_legacy(m,w,a,b,fa,fb,tol,ev);
  if(a==b)return fa==0.?a:NAN;
  if(!refinement_finite(fa)||!refinement_finite(fb)||((fa<0.)==(fb<0.)&&fa!=0.&&fb!=0.))return NAN;
  if(fa==0.)return a;if(fb==0.)return b;
  const double stop=refinement_target(a,b,tol);
  if(m.refinement==2){
    // x1 and x2 bracket; x3 is the displaced endpoint. No residual-only exit.
    double x1=a,x2=b,x3=b,f1=fa,f2=fb,f3=fb,t=.5;
    for(int j=0;j<160;++j){
      if(fabs(x2-x1)<=stop)return .5*(x1+x2);
      double x=x1+t*(x2-x1);
      if(!(x>fmin(x1,x2)&&x<fmax(x1,x2)))x=.5*(x1+x2);
      double f=md::evaluate(m,w,x);++ev;if(!refinement_finite(f))return NAN;if(f==0.)return x;
      x3=x2;f3=f2;
      if((f<0.)==(f1<0.)){x3=x1;f3=f1;}else{x2=x1;f2=f1;}
      x1=x;f1=f;
      const double dx=fabs(x2-x1);if(dx<=stop)return .5*(x1+x2);
      const double xi=(x1-x2)/(x3-x2),phi=(f1-f2)/(f3-f2);
      t=.5;
      if(xi>0.&&xi<1.&&(1.-sqrt(1.-xi))<phi&&phi<sqrt(xi)){
        const double alpha=(x3-x1)/(x2-x1);
        const double z=f1/(f1-f2)*f3/(f3-f2)-alpha*f1/(f3-f1)*f2/(f2-f3);
        if(refinement_finite(z))t=z;
      }
      const double limit=fmin(.5,.5*stop/dx);t=fmax(limit,fmin(1.-limit,t));
    }
    return NAN;
  }
  if(m.refinement==1){
    // Brent-Dekker with b as the best point, and an enclosing opposite c.
    double c=b,fc=fb,d=b-a,e=d;
    for(int j=0;j<160;++j){
      if((fb<0.)==(fc<0.)){c=a;fc=fa;e=d=b-a;}
      if(fabs(fc)<fabs(fb)){a=b;fa=fb;b=c;fb=fc;c=a;fc=fa;}
      if(fb==0.)return b;
      const double mid=.5*(c-b),step=.5*stop;
      if(fabs(c-b)<=stop)return .5*(b+c);
      if(fabs(e)>=step&&fabs(fa)>fabs(fb)){
        double s=fb/fa,p,q;
        if(a==c){p=2.*mid*s;q=1.-s;}
        else{q=fa/fc;const double r=fb/fc;p=s*(2.*mid*q*(q-r)-(b-a)*(r-1.));q=(q-1.)*(r-1.)*(s-1.);}
        if(p>0.)q=-q;else p=-p;
        const double old=e;e=d;
        if(refinement_finite(p/q)&&2.*p<fmin(3.*mid*q-fabs(step*q),fabs(old*q)))d=p/q;
        else{d=mid;e=mid;}
      }else{d=mid;e=mid;}
      a=b;fa=fb;b+=fabs(d)>step?d:(mid>0.?step:-step);
      fb=md::evaluate(m,w,b);++ev;if(!refinement_finite(fb))return NAN;
    }
    return NAN;
  }
  return NAN;
}
} // namespace md
