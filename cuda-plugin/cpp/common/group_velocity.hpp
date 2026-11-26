// BSD-3-Clause. Local implicit dispersion derivatives, shared by CPU/CUDA.
// U = c A/(A+B), A = dD/d(log c), B = dD/d(log omega).
// The smooth, signed shape feature has the same simple zeros as D. Its
// positive normalization cancels from the derivative ratio at a polished root.
#pragma once
#include "common/physics.hpp"

namespace md {
enum GroupVelocityFlag : unsigned {
  GROUP_MISSING_ROOT=1, GROUP_POLISH_FAILED=2, GROUP_DERIVATIVE_UNSTABLE=4,
  GROUP_ILL_CONDITIONED=8, GROUP_CLOSE_ROOT=16, GROUP_NONFINITE=32
};
struct GroupResult {
  double velocity=NAN, slope=NAN;
  unsigned flags=0;
  int evaluations=0;
};

MD_HD double group_shape(const Medium &m,double omega,double c,int &evaluations) {
  ++evaluations;
  return shape_evaluate(m,omega,c).g;
}

// lower/upper delimit the root's local neighborhood (neighbor midpoints when
// available, clipped to the original search bounds). Never modify public roots.
MD_HD GroupResult group_velocity(const Medium &source,double omega,double root,
                                double lower,double upper,double rtol,
                                double close_root_rtol) {
  GroupResult result;
  (void)close_root_rtol; // Callers flag known close-root pairs explicitly.
  if(!isfinite(root)||!(root>0.)) {result.flags=GROUP_MISSING_ROOT;return result;}
  if(!isfinite(omega)||!(omega>0.)||!isfinite(lower)||!isfinite(upper)) {
    result.flags=GROUP_NONFINITE;return result;
  }
  if(!(lower>0.&&root>lower&&root<upper)) {result.flags=GROUP_POLISH_FAILED;return result;}
  Medium m=source;
  // Trace constants depend on omega. In particular the frequency stencil
  // must not silently reuse ka/kb/gammk from the unperturbed trace.
  m.ka=nullptr;m.kb=nullptr;m.gammk=nullptr;m.refinement=1;
  double polished=NAN;
  double radius=2e-6*root;
  const double limit=fmin(.002*root,.9*fmin(root-lower,upper-root));
  for(int attempt=0;attempt<12;++attempt) {
    const double span=fmin(radius,limit),lo=root-span,hi=root+span;
    if(!(span>0.&&lo<root&&hi>root))break;
    const double fl=evaluate(m,omega,lo),fh=evaluate(m,omega,hi);
    result.evaluations+=2;
    if(isfinite(fl)&&isfinite(fh)&&((fl<0.)!=(fh<0.)||fl==0.||fh==0.)) {
      polished=refine_root(m,omega,lo,hi,fl,fh,1e-11,result.evaluations);
      break;
    }
    if(span==limit)break;
    radius*=2.;
  }
  if(!isfinite(polished)||!(polished>lower&&polished<upper)) {
    result.flags=GROUP_POLISH_FAILED;return result;
  }
  // Stay well inside the isolating neighborhood, including after FP64 polish.
  // log1p retains very small but distinct distances from neighboring roots.
  double hc=fmin(1e-3,.2*fmin(log1p((polished-lower)/lower),
                                          log1p((upper-polished)/polished)));
  double hw=1e-3;
  if(!(hc>64.*DBL_EPSILON)) {result.flags=GROUP_DERIVATIVE_UNSTABLE;return result;}
  double cp=group_shape(m,omega,polished*MD_EXP(hc),result.evaluations);
  double cm=group_shape(m,omega,polished*MD_EXP(-hc),result.evaluations);
  double wp=group_shape(m,omega*MD_EXP(hw),polished,result.evaluations);
  double wm=group_shape(m,omega*MD_EXP(-hw),polished,result.evaluations);
  double previous=NAN;
  unsigned failure=GROUP_DERIVATIVE_UNSTABLE;
  for(int iteration=0;iteration<6;++iteration) {
    const double cp2=group_shape(m,omega,polished*MD_EXP(.5*hc),result.evaluations);
    const double cm2=group_shape(m,omega,polished*MD_EXP(-.5*hc),result.evaluations);
    const double wp2=group_shape(m,omega*MD_EXP(.5*hw),polished,result.evaluations);
    const double wm2=group_shape(m,omega*MD_EXP(-.5*hw),polished,result.evaluations);
    if(!isfinite(cp)||!isfinite(cm)||!isfinite(wp)||!isfinite(wm)||
       !isfinite(cp2)||!isfinite(cm2)||!isfinite(wp2)||!isfinite(wm2)) {
      result.flags=GROUP_NONFINITE;return result;
    }
    const double ac=(cp-cm)/(2.*hc),bc=(wp-wm)/(2.*hw);
    const double ah=(cp2-cm2)/hc,bh=(wp2-wm2)/hw;
    const double a=(4.*ah-ac)/3.,b=(4.*bh-bc)/3.,den=a+b;
    if(isfinite(a)&&isfinite(b)&&fabs(den)>64.*DBL_EPSILON*(fabs(a)+fabs(b))) {
      const double u=polished*(a/den);
      // Include a roundoff floor. A zero frequency derivative is legitimate
      // (e.g. a homogeneous Rayleigh half-space), so never require B != 0.
      const double ea=fabs(a-ah)+16.*DBL_EPSILON*(fabs(cp2)+fabs(cm2))/hc;
      const double eb=fabs(b-bh)+16.*DBL_EPSILON*(fabs(wp2)+fabs(wm2))/hw;
      const double error=polished*((fabs(b)*ea+fabs(a)*eb)/fabs(den))/fabs(den);
      const double target=.25*rtol*fmax(polished,fabs(u));
      if(isfinite(u)&&isfinite(error)&&isfinite(previous)&&error<=target&&fabs(u-previous)<=target) {
        result.velocity=u;
        // This slope is private prediction data. Zero A can yield U=0 while
        // the corresponding phase-velocity slope is undefined.
        result.slope=a!=0.?polished*(b/a):NAN;
        return result;
      }
      previous=u;failure=GROUP_DERIVATIVE_UNSTABLE;
    } else {failure=GROUP_ILL_CONDITIONED;previous=NAN;}
    cp=cp2;cm=cm2;wp=wp2;wm=wm2;hc*=.5;hw*=.5;
    if(!(hc>64.*DBL_EPSILON))break;
  }
  result.flags=failure;
  return result;
}
} // namespace md
