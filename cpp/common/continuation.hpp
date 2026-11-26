// BSD-3-Clause. Shared CPU/CUDA prediction and signed-function validation.
#pragma once
#include "common/shape.hpp"

namespace md {
constexpr unsigned P_ANCHOR=1u, P_ACCEPTED=2u, P_FALLBACK=4u, P_BYPASSED=8u,
                   P_AUDIT=16u, P_AUDIT_FAILED=32u, P_SOURCE_UNCERTAIN=64u,
                   P_ROOT_COUNT=128u, P_ANCHOR_UNSAFE=256u, P_WINDOW=512u,
                   P_BRACKET=1024u, P_CORRECTION=2048u, P_REVIEW=4096u;
constexpr unsigned CONTINUATION_APPROXIMATE=65536u;

MD_HD double continuation_log_step(double target,double anchor){
  if(fabs(target-anchor)<.5*anchor)return log1p((target-anchor)/anchor);
  return log(target)-log(anchor);
}

MD_HD int continuation_next_anchor(const double *periods,int nf,int left,int stride){
  int right=left+(stride<nf-1-left?stride:nf-1-left);
  while(right>left+1&&continuation_log_step(periods[right],periods[left])>.25)--right;
  return right;
}

MD_HD bool continuation_source_safe(const double *roots,int nr,int K,unsigned flags,double close_rtol){
  constexpr unsigned unusable=4u|8u|16u|32u|64u|128u|512u|1024u;
  if(nr<1||nr>K||(flags&unusable))return false;
  for(int j=0;j<nr;++j){
    if(!isfinite(roots[j])||!(roots[j]>0.))return false;
    if(j&&roots[j]-roots[j-1]<=close_rtol*roots[j])return false;
  }
  return true;
}

MD_HD double continuation_cap(const double *predicted,int j,int nr,double lower,double upper){
  const double c=predicted[j];
  double gap=fmin(c-lower,upper-c);
  if(j)gap=fmin(gap,c-predicted[j-1]);
  if(j+1<nr)gap=fmin(gap,predicted[j+1]-c);
  return fmin(.02*c,.2*gap);
}

MD_HD unsigned continuation_predict(const double *left_roots,const double *right_roots,
    const double *left_slopes,const double *right_slopes,int nr,double dl,double dr,
    double lower,double upper,double tol,double *predicted,double *disagreement){
  if(!(dl>0.&&dr<0.&&upper>lower))return P_WINDOW;
  const double span=dl-dr,u=dl/span,u2=u*u,u3=u2*u;
  double last_left=0.,last_right=0.,last_upper=0.;
  for(int j=0;j<nr;++j){
    const double c0=left_roots[j],c1=right_roots[j],sl=left_slopes[j],sr=right_slopes[j];
    if(!isfinite(sl)||!isfinite(sr))return P_ANCHOR_UNSAFE;
    const double cl=c0+sl*dl,cr=c1+sr*dr;
    const double c=(2.*u3-3.*u2+1.)*c0+(u3-2.*u2+u)*span*sl+
                   (-2.*u3+3.*u2)*c1+(u3-u2)*span*sr;
    if(!isfinite(c)||!isfinite(cl)||!isfinite(cr)||!(c>lower&&c<upper))return P_WINDOW;
    // Both independent tangent projections must preserve the association of
    // neighboring ranks, even when their Hermite average remains ordered.
    const double projected_lower=fmin(cl,cr),projected_upper=fmax(cl,cr);
    if(j&&(c<=predicted[j-1]||cl<=last_left||cr<=last_right||projected_lower<=last_upper))
      return P_WINDOW|P_REVIEW;
    last_left=cl;last_right=cr;last_upper=projected_upper;
    predicted[j]=c;disagreement[j]=fmax(fabs(cl-cr),fabs(c-((1.-u)*c0+u*c1)));
  }
  for(int j=0;j<nr;++j){
    const double cap=continuation_cap(predicted,j,nr,lower,upper);
    if(!(cap>16.*tol*predicted[j])||disagreement[j]>.5*cap)return P_WINDOW;
  }
  return 0;
}

// Return REVIEW only on finite spectral evidence of a missing/extra crossing.
// Nonfinite or ambiguous evaluations still reject the local approximation.
template<class M>
MD_HD unsigned continuation_gap_sample(const M &m,double omega,double c,double sign,
                                       double endpoint_min,unsigned long long &evals){
  const auto s=shape_evaluate(m,omega,c);++evals;
  if(!shape_finite(s))return P_AUDIT_FAILED;
  if(s.f==0.||((s.f<0.)!=(sign<0.))||
     (fabs(s.g)<.05&&fabs(s.g)<.1*endpoint_min))return P_AUDIT_FAILED|P_REVIEW;
  return 0;
}

template<class M>
MD_HD unsigned continuation_gaps(const M &m,double omega,double lower,double upper,
    const double *lo,const double *hi,int nr,int K,const double *tx,const double *tn,
    int nt,const double *seeds,unsigned long long &evals){
  if(nr<1||nr>K)return P_AUDIT_FAILED;
  if(nr==K)upper=hi[nr-1];
  const double nlo=density_base(m,lower),nhi=density_base(m,upper);
  for(int gap=0;gap<nr+(nr<K);++gap){
    const double a=gap?hi[gap-1]:lower,b=gap<nr?lo[gap]:upper;
    if(!(b>a))continue;
    const auto sa=shape_evaluate(m,omega,a),sb=shape_evaluate(m,omega,b);evals+=2;
    if(!shape_finite(sa)||!shape_finite(sb))return P_AUDIT_FAILED;
    if(sa.f==0.||sb.f==0.){
      const bool interior=(sa.f==0.&&a>lower)||(sb.f==0.&&b<upper);
      return interior?P_AUDIT_FAILED|P_REVIEW:P_AUDIT_FAILED;
    }
    if((sa.f<0.)!=(sb.f<0.))return P_AUDIT_FAILED|P_REVIEW;
    const double endpoint=fmin(fabs(sa.g),fabs(sb.g));
    unsigned issue=continuation_gap_sample(m,omega,a+.5*(b-a),sa.f,endpoint,evals);
    if(issue)return issue;
    const double na=density_base(m,a),nb=density_base(m,b);
    for(int n=1;n<32;++n){
      double c;
      const double target=nlo+(nhi-nlo)*n/32.;
      if(nhi>nlo){
        if(!(target>na&&target<nb))continue;
        c=inverse_density(m,tx,tn,nt,target);
      }else c=lower+(upper-lower)*n/32.;
      if(c>a&&c<b&&(issue=continuation_gap_sample(m,omega,c,sa.f,endpoint,evals)))return issue;
    }
    for(int layer=0;layer<m.L;++layer)for(int body=0;body<2;++body){
      const double v=body?m.vs[layer]:m.vp[layer];
      for(int side=-1;side<=1;++side){
        const double c=v*(1.+side*1e-7);
        if(c>a&&c<b&&(issue=continuation_gap_sample(m,omega,c,sa.f,endpoint,evals)))return issue;
      }
    }
    for(int n=0;n<2;++n){
      const double c=seeds[n];
      if(c>a&&c<b&&(issue=continuation_gap_sample(m,omega,c,sa.f,endpoint,evals)))return issue;
    }
  }
  return 0;
}

template<class M>
MD_HD unsigned continuation_window(const M &m,double omega,double a,double b,double fa,
    double fb,double root,double tol,double phase_level,unsigned long long &evals){
  if(m.wave){
    const double pi=two_pi<double>()*.5;
    const double left=love_phase(m,omega,a),right=love_phase(m,omega,b);evals+=2;
    if(!isfinite(left)||!isfinite(right))return P_AUDIT_FAILED;
    if(floor(left/pi)!=phase_level||floor(right/pi)!=phase_level-1.)
      return P_AUDIT_FAILED|P_REVIEW;
  }else{
    // A sign-changing bracket can conceal a triplet. Sample both open sides
    // of the polished root, outside its numerical uncertainty. Even root
    // pairs can still escape finite sampling.
    const double margin=fmax(tol,coordinate_floor(m))*root;
    for(int side=0;side<2;++side){
      const double edge=side?b:a,sign=(side?fb:fa)<0.?-1.:1.;
      if(fabs(root-edge)<=2.*margin)continue;
      // Three equally spaced interior probes let a same-sign trough
      // trigger an independent minimization even if no probe crosses zero.
      const double end=root+copysign(margin,edge-root);
      const double step=(end-edge)*.25;
      double magnitude[3];
      for(int k=0;k<3;++k){
        const auto sample=shape_evaluate(m,omega,edge+(k+1)*step);++evals;
        if(!shape_finite(sample))return P_AUDIT_FAILED;
        if(sign*sample.f<=0.)return P_AUDIT_FAILED|P_REVIEW;
        magnitude[k]=sign*sample.g;
      }
      const double u=magnitude[0],v=magnitude[1],w=magnitude[2];
      if(v<u&&v<w&&(v<.5*fmin(u,w)||v<.05)){
        const double x=edge+step,y=edge+3.*step;
        double pos;int calls=0;
        const auto minimum=shape_minimum(m,omega,fmin(x,y),fmax(x,y),sign,tol,pos,calls);
        evals+=calls;
        if(!shape_finite(minimum))return P_AUDIT_FAILED;
        if(sign*minimum.f<=0.||(fabs(minimum.g)<.05&&fabs(minimum.g)<.1*fmin(u,w)))
          return P_AUDIT_FAILED|P_REVIEW;
      }
    }
  }
  return 0;
}

template<class M>
MD_HD unsigned continuation_roots(const M &m,double omega,double dl,double dr,
    double lower,double upper,const double *predicted,double *roots,int nr,int K,double tol,
    double accept,double close_rtol,const double *tx,const double *tn,int nt,const double *seeds,
    double *lo,double *hi,unsigned long long &evals,bool &audited){
  audited=false;
  double first_level=0.;
  if(m.wave){
    const double phase=love_phase(m,omega,lower);++evals;
    if(!isfinite(phase)){audited=true;return P_AUDIT_FAILED;}
    first_level=floor(phase/(two_pi<double>()*.5));
  }
  for(int j=0;j<nr;++j){
    const double c=predicted[j],cap=continuation_cap(predicted,j,nr,lower,upper);
    double width=fmin(cap,fmax(16.*tol*c,fmax(2.*roots[j],.05*c*fmin(fabs(dl),fabs(dr)))));
    double a=0.,b=0.,fa=0.,fb=0.;bool bracket=false;
    for(int attempt=0;attempt<4;++attempt){
      a=fmax(lower,c-width);b=fmin(upper,c+width);
      fa=evaluate(m,omega,a);fb=evaluate(m,omega,b);evals+=2;
      if(isfinite(fa)&&isfinite(fb)&&fa!=0.&&fb!=0.&&((fa<0.)!=(fb<0.))){bracket=true;break;}
      if(width>=cap)break;
      width=fmin(cap,2.*width);
    }
    if(!bracket)return P_BRACKET;
    int ev=0;const double root=refine_root(m,omega,a,b,fa,fb,tol,ev);evals+=ev;
    if(!isfinite(root)||root<a||root>b)return P_CORRECTION;
    if(j){
      const double previous=roots[j-1];
      if(root-previous<=close_rtol*fabs(root)||
         (!std::is_same_v<M,Medium> && static_cast<float>(root)<=static_cast<float>(previous)))
        return P_CORRECTION|P_REVIEW;
    }
    if(accept<1.){
      const double f=evaluate(m,omega,root);++evals;
      if(!isfinite(f)||fabs(f)>accept)return P_CORRECTION;
    }
    unsigned issue=continuation_window(m,omega,a,b,fa,fb,root,tol,first_level-j,evals);
    if(issue){audited=true;return issue;}
    lo[j]=a;hi[j]=b;roots[j]=root;
  }
  audited=true;
  return continuation_gaps(m,omega,lower,upper,lo,hi,nr,K,tx,tn,nt,seeds,evals);
}
} // namespace md
