// Normalized Dunkin compound-matrix P-SV evaluator, shared by CPU and CUDA.
// Adapted from CPS surf96 and disba (Copyright 2020 Keurfon Luu, BSD-3-Clause).
// CPS: D. R. Russell / R. B. Herrmann, 1986/1991. See THIRD_PARTY_NOTICES.md.
// Dunkin (1965), BSSA 55(2), 335-358.
// Units: km, km/s, g/cm^3, rad/s, rad/km. Last-layer thickness is ignored.
// water=1 selects a genuine top acoustic layer with a pressure-free water
// surface and continuous normal traction/displacement at the fluid/solid base.
// The production solver uses normalized FP64. Other template instantiations
// are retained only for legacy uniform CUDA and diagnostic precision probes.

#ifndef MODISP_COMMON_DUNKIN_HPP
#define MODISP_COMMON_DUNKIN_HPP

#include <cmath>

#ifdef __CUDACC__
#define MD_HD __host__ __device__ __forceinline__
#else
#define MD_HD inline
#endif

namespace md {

template<class T> MD_HD T shape_small(){if constexpr(sizeof(T)==8)return T(1e-300);else return T(1e-30);}

// Route math calls through the host or device standard library.
#ifdef __CUDACC__
#define MD_FABS(x) fabs(x)
#define MD_FMAX(a, b) fmax((a), (b))
#define MD_SQRT(x) sqrt(x)
#define MD_SIN(x) sin(x)
#define MD_COS(x) cos(x)
#define MD_EXP(x) exp(x)
#define MD_SINCOS(x, sp, cp) sincos((x), (sp), (cp))
#else
#define MD_FABS(x) std::fabs(x)
#define MD_FMAX(a, b) std::fmax((a), (b))
#define MD_SQRT(x) std::sqrt(x)
#define MD_SIN(x) std::sin(x)
#define MD_COS(x) std::cos(x)
#define MD_EXP(x) std::exp(x)
#define MD_SINCOS(x, sp, cp) \
  do { *(sp) = std::sin(x); *(cp) = std::cos(x); } while (0)
#endif

// Eigenfunction products, following disba/_cps/_surf96.py.
template <typename T>
struct VarOut {
  T w, cosp;  // only the fluid-top branch needs these; kept for parity
  T a0;
  T cpcq, cpy, cpz, cqw, cqx, xy, xz, wy, wz;
};

template <typename T>
MD_HD VarOut<T> dunkin_var(T p, T q, T ra, T rb, T wvno, T xka, T xkb, T dpth) {
  VarOut<T> o;
  T cosp = T(0), w = T(0), x = T(0);
  T cosq = T(0), y = T(0), z = T(0);
  T pex = T(0), sex = T(0);

  // --- P-wave eigenfunctions: c < vp, c == vp, or c > vp ---
  if (wvno < xka) {
    T sinp, cosp_pair;
    MD_SINCOS(p, &sinp, &cosp_pair);
    w = sinp / ra;
    x = -ra * sinp;
    cosp = cosp_pair;
  } else if (wvno == xka) {
    cosp = T(1);  // l'Hopital limit; avoids 0/0
    w = dpth;
    x = T(0);
  } else {
    pex = p;
    const T fac = (p < T(16)) ? MD_EXP(-T(2) * p) : T(0);
    cosp = (T(1) + fac) * T(0.5);
    const T sinp = (T(1) - fac) * T(0.5);
    w = sinp / ra;
    x = ra * sinp;
  }

  // --- S-wave eigenfunctions ---
  if (wvno < xkb) {
    T sinq, cosq_pair;
    MD_SINCOS(q, &sinq, &cosq_pair);
    y = sinq / rb;
    z = -rb * sinq;
    cosq = cosq_pair;
  } else if (wvno == xkb) {
    cosq = T(1);
    y = dpth;
    z = T(0);
  } else {
    sex = q;
    const T fac = (q < T(16)) ? MD_EXP(-T(2) * q) : T(0);
    cosq = (T(1) + fac) * T(0.5);
    const T sinq = (T(1) - fac) * T(0.5);
    y = sinq / rb;
    z = rb * sinq;
  }

  // --- products for the compound matrix ---
  const T exa = pex + sex;
  // Both branches propagate when exa is zero, so the attenuation factor is one.
  o.a0 = (exa == T(0)) ? T(1) : ((exa < T(60)) ? MD_EXP(-exa) : T(0));
  o.cpcq = cosp * cosq;
  o.cpy = cosp * y;
  o.cpz = cosp * z;
  o.cqw = cosq * w;
  o.cqx = cosq * x;
  o.xy = x * y;
  o.xz = x * z;
  o.wy = w * y;
  o.wz = w * z;
  o.w = w;
  o.cosp = cosp;
  return o;
}

// ---------------------------------------------------------------------------
// Dunkin's matrix.
//
// Mirrors `dnka()` in disba/_cps/_surf96.py:37-84.  All 25 entries are written
// on every call (rows 0/1/3/4 explicitly, row 2 last since it reuses them), so
// `ca` carries no state across layers.
//
// `ca` is NOT symmetric, and not symmetric under diag(1,1,-1,1,1) either, so
// all 25 products in the mat-vec are genuinely required.
// ---------------------------------------------------------------------------
template <typename T>
MD_HD void dunkin_dnka(T wvno2, T gam, T gammk, T rho, const VarOut<T> &v,
                       T ca[5][5]) {
  const T a0 = v.a0;
  const T cpcq = v.cpcq, cpy = v.cpy, cpz = v.cpz;
  const T cqw = v.cqw, cqx = v.cqx;
  const T xy = v.xy, xz = v.xz, wy = v.wy, wz = v.wz;

  const T gamm1 = gam - T(1);
  const T twgm1 = gam + gamm1;
  const T gmgmk = gam * gammk;
  const T gmgm1 = gam * gamm1;
  const T gm1sq = gamm1 * gamm1;

  const T rho2 = rho * rho;
  const T a0pq = a0 - cpcq;
  const T t = -T(2) * wvno2;

  ca[0][0] = cpcq - T(2) * gmgm1 * a0pq - gmgmk * xz - wvno2 * gm1sq * wy;
  ca[0][1] = (wvno2 * cpy - cqx) / rho;
  ca[0][2] = -(twgm1 * a0pq + gammk * xz + wvno2 * gamm1 * wy) / rho;
  ca[0][3] = (cpz - wvno2 * cqw) / rho;
  ca[0][4] = -(T(2) * wvno2 * a0pq + xz + wvno2 * wvno2 * wy) / rho2;

  ca[1][0] = (gmgmk * cpz - gm1sq * cqw) * rho;
  ca[1][1] = cpcq;
  ca[1][2] = gammk * cpz - gamm1 * cqw;
  ca[1][3] = -wz;
  ca[1][4] = ca[0][3];

  ca[3][0] = (gm1sq * cpy - gmgmk * cqx) * rho;
  ca[3][1] = -xy;
  ca[3][2] = gamm1 * cpy - gammk * cqx;
  ca[3][3] = ca[1][1];
  ca[3][4] = ca[0][1];

  ca[4][0] = -(T(2) * gmgmk * gm1sq * a0pq + gmgmk * gmgmk * xz +
               gm1sq * gm1sq * wy) * rho2;
  ca[4][1] = ca[3][0];
  ca[4][2] = -(gammk * gamm1 * twgm1 * a0pq + gam * gammk * gammk * xz +
               gamm1 * gm1sq * wy) * rho;
  ca[4][3] = ca[1][0];
  ca[4][4] = ca[0][0];

  // Row 2 depends on rows 0/1/3/4, so it is filled last.
  ca[2][0] = t * ca[4][2];
  ca[2][1] = t * ca[3][2];
  ca[2][2] = a0 + T(2) * (cpcq - ca[0][0]);
  ca[2][3] = t * ca[1][2];
  ca[2][4] = t * ca[0][2];
}

// ---------------------------------------------------------------------------
// Rayleigh secular function for a solid layered model.
//
// Mirrors `dltar4()` in disba/_cps/_surf96.py:204-285 with llw == -1.
// Returns F = e[0] after propagating the 5-vector from the half-space up.
//
// NORMALIZE rescales the five-vector after every layer to bound |F| <= 1.
// ---------------------------------------------------------------------------
template <typename T, bool NORMALIZE>
MD_HD T dunkin_secular(const T *d, const T *a, const T *b, const T *rho,
                       int nlayer, T wvno, T omega, int water = 0, bool clamp_omega = true,
                       const T *prepared_ka = nullptr, const T *prepared_kb = nullptr,
                       const T *prepared_gammk = nullptr, T *shape = nullptr) {
  if (clamp_omega && omega < T(1.0e-4)) omega = T(1.0e-4);
  const T wvno2 = wvno * wvno;
  const int last = nlayer - 1;

  // --- E vector for the bottom half-space ---
  const T xka0 = prepared_ka ? prepared_ka[last] : omega / a[last];
  const T xkb0 = prepared_kb ? prepared_kb[last] : omega / b[last];
  T ra = MD_SQRT((wvno + xka0) * MD_FABS(wvno - xka0));
  T rb = MD_SQRT((wvno + xkb0) * MD_FABS(wvno - xkb0));
  T t = b[last] / omega;
  T gammk = prepared_gammk ? prepared_gammk[last] : T(2) * t * t;
  T gam = gammk * wvno2;
  const T gamm1 = gam - T(1);
  const T rho_hs = rho[last];

  T e[5];
  e[0] = rho_hs * rho_hs * (gamm1 * gamm1 - gam * gammk * ra * rb);
  e[1] = -rho_hs * ra;
  e[2] = rho_hs * (gamm1 - gammk * ra * rb);
  e[3] = rho_hs * rb;
  e[4] = wvno2 - ra * rb;

  // --- propagate upward, bottom layer first ---
  for (int m = nlayer - 2; m >= water; --m) {
    // Recompute layer coefficients for the current angular frequency.
    const T xka = prepared_ka ? prepared_ka[m] : omega / a[m];
    const T xkb = prepared_kb ? prepared_kb[m] : omega / b[m];
    t = b[m] / omega;
    gammk = prepared_gammk ? prepared_gammk[m] : T(2) * t * t;
    gam = gammk * wvno2;
    ra = MD_SQRT((wvno + xka) * MD_FABS(wvno - xka));
    rb = MD_SQRT((wvno + xkb) * MD_FABS(wvno - xkb));

    const T dpth = d[m];
    const T rhol = rho[m];
    const T p = ra * dpth;
    const T q = rb * dpth;

    const VarOut<T> v = dunkin_var<T>(p, q, ra, rb, wvno, xka, xkb, dpth);

    T ca[5][5];
    dunkin_dnka<T>(wvno2, gam, gammk, rhol, v, ca);

    // e <- e . ca   (row vector times matrix)
    T ee[5];
#pragma unroll
    for (int i = 0; i < 5; ++i) {
      T s = T(0);
#pragma unroll
      for (int j = 0; j < 5; ++j) s += e[j] * ca[j][i];
      ee[i] = s;
    }

    if (NORMALIZE) {
      T t1 = T(0);
#pragma unroll
      for (int i = 0; i < 5; ++i) t1 = MD_FMAX(t1, MD_FABS(ee[i]));
      if (t1 < T(1.0e-40)) t1 = T(1);
#pragma unroll
      for (int i = 0; i < 5; ++i) ee[i] /= t1;
    }

#pragma unroll
    for (int i = 0; i < 5; ++i) e[i] = ee[i];
  }

  if (water) {
    // Pressure-release acoustic surface. No shear velocity is introduced in
    // the fluid. Use scaled cos/sin directly: a tangent impedance would add
    // poles whose sign changes could be mistaken for dispersion roots.
    const T ka=omega/a[0];
    const T rf=MD_SQRT((wvno+ka)*MD_FABS(wvno-ka));
    const T p=rf*d[0];
    T cp, w;
    if (wvno<ka) { T sn; MD_SINCOS(p,&sn,&cp); w=sn/rf; }
    else if (wvno==ka) { cp=T(1); w=d[0]; }
    else { const T z=(p<T(16)) ? MD_EXP(-T(2)*p) : T(0); cp=(T(1)+z)*T(.5); w=(T(1)-z)*T(.5)/rf; }
    const T t0=cp*e[0], t1=rho[0]*w*e[1];
    const T scale=MD_FMAX(T(1),MD_FMAX(MD_FABS(t0),MD_FABS(t1)));
    if(shape){const T u=t0/scale,v=t1/scale;*shape=(u-v)/MD_SQRT(u*u+v*v+shape_small<T>());}
    return t0/scale-t1/scale;
  }
  // A search feature only. The original signed value and its normalization
  // remain unchanged. A smooth positive L2 scale retains variation when the maximum component
  // normalization would return an exactly saturated +/-1 plateau.
  if(shape){T other=shape_small<T>();for(int i=1;i<5;++i)other+=e[i]*e[i];*shape=e[0]/MD_SQRT(e[0]*e[0]+other);}
  return e[0];
}

// Use one precise 2*pi constant in every CPU and CUDA evaluation.
template <typename T>
MD_HD T two_pi() {
  return T(6.283185307179586476925286766559L);
}

// Evaluate at a fixed angular frequency during candidate-velocity searches.
template <typename T, bool NORMALIZE>
MD_HD T dunkin_secular_omega(const T *d, const T *vp, const T *vs, const T *rho,
                             int nlayer, T omega, T c) {
  return dunkin_secular<T, NORMALIZE>(d, vp, vs, rho, nlayer, omega / c, omega);
}

// Convenience wrapper taking physical inputs (period in s, phase velocity in
// km/s) rather than (wvno, omega).
template <typename T, bool NORMALIZE>
MD_HD T dunkin_secular_c(const T *d, const T *vp, const T *vs, const T *rho,
                         int nlayer, T period, T c) {
  return dunkin_secular_omega<T, NORMALIZE>(d, vp, vs, rho, nlayer,
                                            two_pi<T>() / period, c);
}

}  // namespace md

#endif  // MODISP_COMMON_DUNKIN_HPP
