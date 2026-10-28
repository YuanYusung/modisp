// Adaptive coupled P-SV / SH root search.
// Algorithmic reference: Pan et al. (2026), doi:10.1785/0120250207.
// Extrema are candidate brackets and are refined before acceptance.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <utility>
#include <stdexcept>
#include <string>
#include "common/accuracy.hpp"
#include "common/physics.hpp"
#ifndef MD_DENSITY_TABLE
#define MD_DENSITY_TABLE 1
#endif
#ifndef MD_VALUE_CACHE
#define MD_VALUE_CACHE 1
#endif
#ifndef MD_MERGE
#define MD_MERGE 1
#endif
#ifndef MD_ACTIVITY_QUEUE
#define MD_ACTIVITY_QUEUE 1
#endif
#ifndef MD_PREFIX_DENSE
#define MD_PREFIX_DENSE 1
#endif
#ifndef MD_WARP_STATS
#define MD_WARP_STATS 1
#endif
#ifndef MD_TRACE_CONSTANTS
#define MD_TRACE_CONSTANTS 1
#endif

namespace py = pybind11;
namespace qe {
constexpr int TABLE=1024, CACHE=256;
constexpr int CAP = 2048;  // power of two; x/y live in block shared memory
constexpr unsigned UNCONVERGED = 1u << 2, WORKSPACE = 1u << 3;
constexpr unsigned CUTOFF = 1u << 4, NONFINITE = 1u << 5;
constexpr unsigned REFINE_FAILED = 1u << 6, F_REJECTED = 1u << 7;
enum Stat { EVALS, EXTREMA, BRACKETS, ADAPT_OK, DENSE_OK, FAILED,
            CAPACITY_HITS, CUTOFFS, NONFINITES, REFINE_FAILURES, SEARCHES,
            ACCEPT_REJECTED, ANALYTIC_EMPTY, RANGE_EMPTY, VERIFICATIONS, CONFLICTS, CACHE_HITS, CACHE_MISSES,
            EARLY_ACCEPTS, VERIFICATION_SKIPPED, DENSE_COARSE_OK, DENSE_FINE_ESCALATED, TROUGH_RISKS, NSTATS };
const char *stat_names[NSTATS] = {
    "evaluations", "extrema_proposals", "brackets", "adaptive_converged",
    "dense_converged", "unconverged", "capacity_hits", "near_cutoff",
    "nonfinite", "refinement_failed", "searches", "rejected_f", "analytical_empty", "range_empty", "verification_traces", "family_conflicts", "cache_hits", "cache_misses",
    "early_accepts", "verification_skipped", "dense_coarse_accepted", "dense_fine_escalated", "trough_risks"};

struct Device {
  double *d, *a, *b, *r, *period, *bounds, *ia, *ib, *tx, *tn, *seeds, *cache_x, *cache_y;
  int *nt, *cache_n, *queue, *next_queue, *queue_n, *next_n, *cursor;
  int wave, automatic, guarded, profile;
  int accuracy, accept_level1, accept_level2, verify_safe_skip, allow_dense_early;
  double period_threshold, gap_factor, dense_gap_factor, trough_threshold;
  int dense_floor4, dense_floor5, dense_factor4, dense_factor5;
  double *sample_gap; int *risk, *escalated;
  double *full_roots,*guard_roots;
  int *verify,*guard_count,*full_count;
  double *lo, *hi, *flo, *fhi, *roots, *previous, *output;
  int *count, *prev_count, *status, *active, *level;
  unsigned *flags;
  unsigned long long *stats;
};

// All lanes reach this at the end of block-owned sampling work. Integer
// reduction preserves counters; divergent refinement retains its own atomics.
__device__ void add_sampling_stat(unsigned long long *address,unsigned long long value) {
  if(MD_WARP_STATS){for(int d=16;d;d/=2)value+=__shfl_down_sync(0xffffffffu,value,d);if((threadIdx.x&31)==0&&value)atomicAdd(address,value);}
  else if(value)atomicAdd(address,value);
}

__device__ md::Medium medium(Device st,int model,int L) {
  const size_t o=static_cast<size_t>(model)*L;
  return {st.d+o,st.a+o,st.b+o,st.r+o,st.ia+o,st.ib+o,L,st.wave};
}

__device__ void sort_pairs(double *x, double *y, int count) {
  int width=1;
  while (width<count) width*=2;
  for (int k=2; k<=width; k*=2) {
    for (int j=k/2; j>0; j/=2) {
      for (int i=threadIdx.x; i<width; i+=blockDim.x) {
        const int q=i^j;
        if (q>i && (((i&k)==0) ? x[i]>x[q] : x[i]<x[q])) {
          const double tx=x[i], ty=y[i];
          x[i]=x[q]; y[i]=y[q]; x[q]=tx; y[q]=ty;
        }
      }
      __syncthreads();
    }
  }
}

// Sort only the new extremum proposals. Parallel rank merge uses the unused
// shared-memory tail; large sequences fall back to the bounded bitonic sort.
__device__ void merge_pairs(double *x,double *y,int oldn,int extra) {
  if(!MD_MERGE){sort_pairs(x,y,oldn+extra);return;}
  const int n=oldn+extra;int width=1;while(width<extra)width*=2;
  if(n>CAP/2 || oldn+width>CAP){sort_pairs(x,y,n);return;}
  sort_pairs(x+oldn,y+oldn,extra);
  __syncthreads();
  for(int i=threadIdx.x;i<n;i+=blockDim.x){
    const double c=x[i];int lo=0,hi=i<oldn?extra:oldn;
    while(lo<hi){int q=(lo+hi)/2;double v=x[(i<oldn?oldn:0)+q];if(i<oldn?v<c:v<=c)lo=q+1;else hi=q;}
    const int dest=(i<oldn?i:i-oldn)+lo;
    x[CAP-n+dest]=c;y[CAP-n+dest]=y[i];
  }
  __syncthreads();
  for(int i=threadIdx.x;i<n;i+=blockDim.x){x[i]=x[CAP-n+i];y[i]=y[CAP-n+i];}
  __syncthreads();
  for(int i=n+threadIdx.x;i<CAP;i+=blockDim.x){x[i]=CUDART_INF;y[i]=0.;}
  __syncthreads();
}

__global__ void prepare_models(Device st,int B,int L) {
  const int model=blockIdx.x;const size_t off=static_cast<size_t>(model)*L;
  __shared__ double x[TABLE],y[TABLE];
  for(int j=threadIdx.x;j<L;j+=blockDim.x){st.ia[off+j]=1./(st.a[off+j]*st.a[off+j]);st.ib[off+j]=st.b[off+j]==0.?0.:1./(st.b[off+j]*st.b[off+j]);}
  __syncthreads();
  const double lo=st.bounds[2*model],hi=st.bounds[2*model+1];
  for(int j=threadIdx.x;j<TABLE;j+=blockDim.x){
    double c=CUDART_INF;
    if(j<129)c=lo+(hi-lo)*j/128.;
    else if(j<129+2*L){int k=(j-129)/2;c=(j-129)%2?st.b[off+k]:st.a[off+k];if(!(c>lo&&c<hi))c=CUDART_INF;}
    x[j]=c;y[j]=0.;
  }
  __syncthreads();sort_pairs(x,y,129+2*L);
  if(!threadIdx.x){int n=0;for(int j=0;j<TABLE&&isfinite(x[j]);++j)if(!n||x[j]>x[n-1])x[n++]=x[j];st.nt[model]=n;
    const int top=st.b[off]==0.?1:0;
    st.seeds[2*model]=st.wave?st.b[off+top]:md::rayleigh_reference(st.a[off+top],st.b[off+top]);
    st.seeds[2*model+1]=(!st.wave&&top)?md::scholte_reference(st.a[off],st.r[off],st.a[off+1],st.b[off+1],st.r[off+1]):CUDART_NAN;
  }
  __syncthreads();const md::Medium m=medium(st,model,L);
  for(int j=threadIdx.x;j<st.nt[model];j+=blockDim.x){st.tx[static_cast<size_t>(model)*TABLE+j]=x[j];st.tn[static_cast<size_t>(model)*TABLE+j]=md::density_base(m,x[j]);}
}

__device__ double cached_value(Device st,int trace,const md::Medium &m,double omega,double c,unsigned long long &evals,unsigned long long &hits) {
  if(!MD_VALUE_CACHE){++evals;return md::evaluate(m,omega,c);}
  int lo=0,hi=st.cache_n[trace];const size_t o=static_cast<size_t>(trace)*CACHE;
  while(lo<hi){int q=(lo+hi)/2;if(st.cache_x[o+q]<c)lo=q+1;else hi=q;}
  if(lo<st.cache_n[trace]&&st.cache_x[o+lo]==c){if(st.profile)++hits;return st.cache_y[o+lo];}
  ++evals;return md::evaluate(m,omega,c);
}

// Called by lane zero after sorting; all duplicate sampling coordinates are
// removed exactly. Distinct root brackets are never merged by an absolute gap.
__device__ int unique_pairs(double *x, double *y) {
  int n=0;
  for (int i=0; i<CAP && isfinite(x[i]); ++i) {
    if (n==0 || x[i]>x[n-1]) { x[n]=x[i]; y[n]=y[i]; ++n; }
  }
  for (int i=n; i<CAP; ++i) { x[i]=CUDART_INF; y[i]=0.0; }
  return n;
}

__device__ void append_bracket(Device st, int trace, int K, int &n,
                               double lo, double hi, double flo, double fhi) {
  if (n<K) {
    const size_t p=static_cast<size_t>(trace)*K+n;
    st.lo[p]=lo; st.hi[p]=hi; st.flo[p]=flo; st.fhi[p]=fhi;
  }
  ++n;
}

__device__ void collect(Device st, int trace, int K, double *x, double *y,
                        int n, int &nr, int &bad, bool first, bool last) {
  for (int i=0; i<n-1; ++i) {
    if (!isfinite(y[i]) || !isfinite(y[i+1])) { bad=1; continue; }
    if (y[i]==0.0) {
      // A run of floating-point zero is an unresolved numerical collapse.
      // Isolated exact nodes are counted once, independent of grid alignment.
      if (y[i+1]==0.0 || (i>0 && y[i-1]==0.0)) { bad=1; continue; }
      if ((i>0 && (signbit(y[i-1])!=signbit(y[i+1]))) || (first && i==0))
        append_bracket(st,trace,K,nr,x[i],x[i],0.0,0.0);
      else bad=1;
    } else if (y[i+1]!=0.0 && signbit(y[i])!=signbit(y[i+1])) {
      append_bracket(st,trace,K,nr,x[i],x[i+1],y[i],y[i+1]);
    }
  }
  if (last && n>1 && y[n-1]==0.0) {
    if (y[n-2]==0.0) bad=1;
    else append_bracket(st,trace,K,nr,x[n-1],x[n-1],0.0,0.0);
  }
}

__global__ void initialize(Device st, int traces, int K,int L,int nf,int modes) {
  const int t=blockIdx.x*blockDim.x+threadIdx.x;
  if (t>=traces) return;
  st.active[t]=1; st.prev_count[t]=-1; st.count[t]=0; st.status[t]=0;
  st.flags[t]=0; st.level[t]=-1; st.cache_n[t]=0;st.queue[t]=t;
  st.verify[t]=0;
  st.guard_count[t]=-1;
  st.full_count[t]=-1;
  st.sample_gap[t]=0.;st.risk[t]=1;st.escalated[t]=0;
  if(!t){*st.queue_n=traces;*st.next_n=0;*st.cursor=0;}
  for (int m=0; m<K; ++m){st.previous[static_cast<size_t>(t)*K+m]=CUDART_NAN;st.roots[static_cast<size_t>(t)*K+m]=CUDART_NAN;}
  if(st.bounds[2*(t/nf)+1]>st.bounds[2*(t/nf)]&&st.wave&&!md::sh_has_guide(medium(st,t/nf,L))){st.active[t]=0;st.level[t]=-3;for(int m=0;m<modes;++m)st.output[(static_cast<size_t>(t/nf)*modes+m)*nf+t%nf]=CUDART_NAN;atomicAdd(&st.stats[ANALYTIC_EMPTY],1ULL);}
  if(st.bounds[2*(t/nf)+1]<=st.bounds[2*(t/nf)]){st.active[t]=0;st.level[t]=-4;for(int m=0;m<modes;++m)st.output[(static_cast<size_t>(t/nf)*modes+m)*nf+t%nf]=CUDART_NAN;atomicAdd(&st.stats[RANGE_EMPTY],1ULL);}
}

// Spacing / hidden-pair safety for early acceptance. The bracket width at the
// roots is the local sampling resolution; a root gap smaller than a few bracket
// widths means a missed pair cannot be excluded.
__device__ bool roots_spaced(const double *r,int count,double sample_gap,double gap_factor){
  if(count<1||!(gap_factor>0.))return false;
  for(int m=1;m<count;++m){
    const double gap=r[m]-r[m-1];
    const double floor=16.*2.220446049250313e-16*fmax(1.,fabs(r[m]));
    if(gap<=gap_factor*fmax(sample_gap,floor))return false;
  }
  return true;
}

__device__ void spacing_risk(const double *x,const double *y,int n,double threshold,double &min_gap,int &risk){
  min_gap=0.;risk=0;
  for(int i=0;i+1<n;++i){
    const double fa=y[i],fb=y[i+1];
    if(!isfinite(fa)||!isfinite(fb))continue;
    if((fa<0.)!=(fb<0.)){
      const double g=x[i+1]-x[i];
      if(!(min_gap>0.)||g<min_gap)min_gap=g;
    }
  }
  if(!(threshold>0.))return;
  for(int i=1;i+1<n;++i){
    const double f0=y[i-1],f1=y[i],f2=y[i+1];
    if(!isfinite(f0)||!isfinite(f1)||!isfinite(f2))continue;
    if(((f0<0.)!=(f1<0.))||((f1<0.)!=(f2<0.)))continue;
    if(fabs(f1)<threshold&&fabs(f1)<fabs(f0)&&fabs(f1)<fabs(f2)){risk=1;return;}
  }
}

// A block owns one trace's sampling work. Every secular evaluation is parallel
// across candidate velocities; layers remain inside the register recurrence.
__device__ void adaptive_one(Device st,int L,int nf,int K,double delta,double tol,int lev,int t,double *sm) {
  double *x=sm, *y=x+CAP, *h=y+CAP, *a=h+L, *b=a+L, *r=b+L;
  double *ia=r+L, *ib=ia+L;
  double *ka=ib+L,*kb=ka+L,*gammk=kb+L;
  __shared__ int n, extra, bad, overflow;
  const int model=t/nf;
  const size_t off=static_cast<size_t>(model)*L;
  const double per=st.period[t%nf],omega=md::two_pi<double>()/per;
  for (int i=threadIdx.x; i<L; i+=blockDim.x) {
    h[i]=st.d[off+i]; a[i]=st.a[off+i]; b[i]=st.b[off+i]; r[i]=st.r[off+i];
    ia[i]=st.ia[off+i]; ib[i]=st.ib[off+i];
    if(MD_TRACE_CONSTANTS){ka[i]=omega/a[i];kb[i]=b[i]==0.?0.:omega/b[i];const double z=b[i]/omega;gammk[i]=2.*z*z;}
  }
  for (int i=threadIdx.x; i<CAP; i+=blockDim.x) { x[i]=CUDART_INF; y[i]=0.0; }
  if (threadIdx.x==0) { bad=0; overflow=0; st.status[t]=0; }
  __syncthreads();
  double upper=st.bounds[2*(t/nf)+1];
  // Refined guard roots can bound later adaptive work; the original global
  // grid/range is retained for both independent dense fallback resolutions.
  if(st.guarded&&lev>0&&st.prev_count[t]==K&&!(st.flags[t]&(WORKSPACE|NONFINITE|REFINE_FAILED|512u)))upper=fmin(upper,st.previous[static_cast<size_t>(t)*K+K-1]*(1.+4.*tol));
  const double cmin=(st.automatic && lev<2)?fmin((lev==0?4.:2.)*st.bounds[2*(t/nf)],(st.bounds[2*(t/nf)]+upper)*.5):st.bounds[2*(t/nf)];
  const md::Medium med{h,a,b,r,ia,ib,L,st.wave,MD_TRACE_CONSTANTS?ka:nullptr,MD_TRACE_CONSTANTS?kb:nullptr,MD_TRACE_CONSTANTS?gammk:nullptr};
  if (!(upper>cmin)) { if (!threadIdx.x) st.count[t]=0; return; }
  const double nlo=md::density_base(med,cmin)/per;
  const double nhi=md::density_base(med,upper)/per;
  // Four sub-intervals correspond to the reference's two initial fine passes.
  const double cells=ceil(fmax(0.0,nhi-nlo)/(delta*0.25));
  if (!isfinite(cells) || cells>CAP-64-6*L-2) {
    if (!threadIdx.x) {
      st.status[t]=1; st.count[t]=-1; st.flags[t]|=WORKSPACE;
      atomicAdd(&st.stats[CAPACITY_HITS],1ULL);
    }
    return;
  }
  const int nn=static_cast<int>(cells), initial=64+6*L+nn+2;
  if (!threadIdx.x) { n=initial; atomicAdd(&st.stats[SEARCHES],1ULL); }
  // N(c) can be flat below the slowest body velocity; a uniform guard and a
  // homogeneous surface Rayleigh seed cover that region without trusting N.
  for (int i=threadIdx.x; i<initial; i+=blockDim.x) {
    double c;
    if (i<64) c=cmin+(upper-cmin)*(st.guarded&&st.full_count[t]<K&&i>0&&i<63?(i-.5)/62.:static_cast<double>(i)/63.0);
    else if (i<64+6*L) {
      const int q=i-64, j=q/6, which=q%6;
      const double v=which<3 ? b[j] : a[j];
      const int side=(which%3)-1;
      c=v*(1.0+side*1.0e-7);
    } else if (i>=initial-2) c=st.seeds[2*model+i-(initial-2)];
    else {
      const int q=i-64-6*L+1;
      const double target=(nlo+(q-(st.guarded&&st.full_count[t]<K?.5:0.))*delta*.25)*per;
      const size_t tab=static_cast<size_t>(model)*TABLE;
      c=CUDART_NAN;
      if(target<st.tn[tab+st.nt[model]-1]){
        if(MD_DENSITY_TABLE)c=md::inverse_density(med,st.tx+tab,st.tn+tab,st.nt[model],target);
        else {const double xx[2]={cmin,upper},nn[2]={nlo*per,nhi*per};c=md::inverse_density(med,xx,nn,2,target);}
      }
      if(target<st.tn[tab+st.nt[model]-1]&&(!isfinite(c)||fabs((md::density_base(med,c)-target)/per)>delta*1e-3)){atomicOr(&bad,2);atomicOr(&st.flags[t],512u);}
    }
    if (c>=cmin && c<=upper) x[i]=c;
  }
  __syncthreads();
  sort_pairs(x,y,initial);
  if (!threadIdx.x) n=unique_pairs(x,y);
  __syncthreads();
  unsigned long long evals=0, extrema=0, hits=0;
  for (int i=threadIdx.x; i<n; i+=blockDim.x) {
    y[i]=cached_value(st,t,med,omega,x[i],evals,hits);
    if (!isfinite(y[i])) atomicExch(&bad,1);
  }
  __syncthreads();
  for (int round=0; round<3; ++round) {
    if (!threadIdx.x) extra=0;
    __syncthreads();
    const int oldn=n;
    for (int i=threadIdx.x; i+2<oldn; i+=blockDim.x) {
      const double x1=x[i], x2=x[i+1], x3=x[i+2];
      const double f1=y[i], f2=y[i+1], f3=y[i+2];
      if (!isfinite(f1) || !isfinite(f2) || !isfinite(f3)) continue;
      const double scale=fmax(fabs(f1),fmax(fabs(f2),fabs(f3)));
      if (scale==0.0) { atomicExch(&bad,1); continue; }
      const double s1=(f2/scale-f1/scale)/(x2-x1);
      const double s2=(f3/scale-f2/scale)/(x3-x2);
      const double curvature=(s2-s1)/(x3-x1);
      double xe=CUDART_NAN;
      if (isfinite(curvature) && fabs(curvature)>
          64.0*2.220446049250313e-16/((x3-x1)*(x3-x1)))
        xe=(x1+x2)*0.5-s1/(2.0*curvature);
      // Degenerate interpolation near a root gets a real midpoint probe.
      // Flat saturated +/-1 plateaus have no extremum evidence and are covered
      // by the density ladder / independent dense fallback instead.
      if (!isfinite(xe) && fabs(f2)<0.05) xe=(x1+x3)*0.5;
      const double gap=16.0*2.220446049250313e-16*fmax(1.0,fabs(x2));
      if (!(xe>x1+gap && xe<x3-gap) || fabs(xe-x2)<=gap) continue;
      const int q=atomicAdd(&extra,1), dest=oldn+q;
      ++extrema;
      if (dest<CAP) {
        x[dest]=xe;
        y[dest]=cached_value(st,t,med,omega,xe,evals,hits);
        if (!isfinite(y[dest])) atomicExch(&bad,1);
      } else atomicExch(&overflow,1);
    }
    __syncthreads();
    if (overflow || extra==0) break;
    merge_pairs(x,y,oldn,extra);
    if (!threadIdx.x) n=unique_pairs(x,y);
    __syncthreads();
  }
  __syncthreads();
  if(st.accuracy!=0&&!threadIdx.x){
    double gap=0.;int risk=0;spacing_risk(x,y,n,st.trough_threshold,gap,risk);
    st.sample_gap[t]=gap;st.risk[t]=risk;
    if(risk)atomicAdd(&st.stats[TROUGH_RISKS],1ULL);
  }
  __syncthreads();
  if(n<=CACHE && !overflow && !bad){for(int i=threadIdx.x;i<n;i+=blockDim.x){const size_t o=static_cast<size_t>(t)*CACHE+i;st.cache_x[o]=x[i];st.cache_y[o]=y[i];}}
  __syncthreads();if(!threadIdx.x)st.cache_n[t]=(n<=CACHE&&!overflow&&!bad)?n:0;
  add_sampling_stat(&st.stats[EVALS],evals);
  add_sampling_stat(&st.stats[EXTREMA],extrema);
  if(st.profile){add_sampling_stat(&st.stats[CACHE_HITS],hits);add_sampling_stat(&st.stats[CACHE_MISSES],evals);}
  if (!threadIdx.x) {
    int nr=0;
    if(!overflow&&!bad&&n>0&&md::ambiguous_sh_cutoff(med,x[n-1],y[n-1])){bad=4;st.flags[t]|=CUTOFF;}
    if (!overflow && !bad) collect(st,t,K,x,y,n,nr,bad,true,true);
    if (overflow) { st.flags[t]|=WORKSPACE; atomicAdd(&st.stats[CAPACITY_HITS],1ULL); }
    if (bad&1) { st.flags[t]|=NONFINITE; atomicAdd(&st.stats[NONFINITES],1ULL); }
    st.status[t]=(overflow || bad) ? 1 : 0;
    st.count[t]=st.status[t] ? -1 : min(nr,K);
    atomicAdd(&st.stats[BRACKETS],static_cast<unsigned long long>(nr));
  }
}

// Tiled dense fallback needs O(CAP) shared storage, independent of nc. Only
// unresolved traces execute it. No scan matrices or candidates visit the CPU.
__device__ void dense_one(Device st,int L,int nf,int K,int floor_cells,int factor,int t,double *sm) {
  double *x=sm, *y=x+CAP, *h=y+CAP, *a=h+L, *b=a+L, *r=b+L;
  double *ia=r+L, *ib=ia+L;
  double *ka=ib+L,*kb=ka+L,*gammk=kb+L;
  __shared__ int nr, bad;
  const size_t off=static_cast<size_t>(t/nf)*L;
  const double omega=md::two_pi<double>()/st.period[t%nf];
  for (int i=threadIdx.x; i<L; i+=blockDim.x) {
    h[i]=st.d[off+i]; a[i]=st.a[off+i]; b[i]=st.b[off+i]; r[i]=st.r[off+i];
    ia[i]=st.ia[off+i]; ib[i]=st.ib[off+i];
    if(MD_TRACE_CONSTANTS){ka[i]=omega/a[i];kb[i]=b[i]==0.?0.:omega/b[i];const double z=b[i]/omega;gammk[i]=2.*z*z;}
  }
  if (!threadIdx.x) { nr=0; bad=0; st.status[t]=0; atomicAdd(&st.stats[SEARCHES],1ULL); }
  __syncthreads();
  const double cmin=st.bounds[2*(t/nf)],upper=st.bounds[2*(t/nf)+1];
  const md::Medium med{h,a,b,r,ia,ib,L,st.wave,MD_TRACE_CONSTANTS?ka:nullptr,MD_TRACE_CONSTANTS?kb:nullptr,MD_TRACE_CONSTANTS?gammk:nullptr};
  if (!(upper>cmin)) { if (!threadIdx.x) st.count[t]=0; return; }
  // Scale dense fallback with the density estimate and water optical thickness,
  // with a 32768-cell floor and an explicit workspace limit.
  const double need=md::dense_workload(med,st.period[t%nf],cmin,upper);
  if (!isfinite(need) || need>524288.0) {
    if (!threadIdx.x) {
      st.status[t]=1; st.count[t]=-1; st.flags[t]|=WORKSPACE;
      atomicAdd(&st.stats[CAPACITY_HITS],1ULL);
    }
    return;
  }
  int cells=floor_cells;
  while (cells<need) cells*=2;
  const int nc=factor*cells+1;
  if(!threadIdx.x){st.sample_gap[t]=cells>0?(upper-cmin)/static_cast<double>(cells):0.;st.risk[t]=0;}
  __syncthreads();
  unsigned long long evals=0;
  for (int start=0; start<nc-1; start+=CAP-1) {
    const int n=min(CAP,nc-start);
    for (int i=threadIdx.x; i<n; i+=blockDim.x) {
      x[i]=cmin+(upper-cmin)*static_cast<double>(start+i)/(nc-1.0);
      y[i]=md::evaluate(med,omega,x[i]); ++evals;
    }
    __syncthreads();
    if(st.accuracy!=0&&!threadIdx.x){
      double gap=0.;int risk=0;spacing_risk(x,y,n,st.trough_threshold,gap,risk);
      // The dense grid is uniform: keep the actual cell width as the sampling
      // resolution. A narrow bracket next to a root is not evidence that a
      // neighbouring close pair is resolved.
      if(risk){st.risk[t]=1;atomicAdd(&st.stats[TROUGH_RISKS],1ULL);}
    }
    __syncthreads();
    if (!threadIdx.x){if(start+n==nc&&md::ambiguous_sh_cutoff(med,x[n-1],y[n-1])){bad=4;st.flags[t]|=CUTOFF;}if(!bad)collect(st,t,K,x,y,n,nr,bad,start==0,start+n==nc);}
    __syncthreads();
    // Grid spacing remains defined over the complete interval. Only the
    // guarded ascending prefix is accepted/returned by either implementation.
    if(MD_PREFIX_DENSE && nr>=K && !bad)break;
  }
  add_sampling_stat(&st.stats[EVALS],evals);
  if (!threadIdx.x) {
    if (bad&1) { st.flags[t]|=NONFINITE; atomicAdd(&st.stats[NONFINITES],1ULL); }
    st.status[t]=bad; st.count[t]=bad ? -1 : min(nr,K);
    atomicAdd(&st.stats[BRACKETS],static_cast<unsigned long long>(nr));
  }
}

// Persistent blocks consume only active traces, entirely on the device.
template<bool ADAPTIVE>
__global__ void search(Device st,int L,int nf,int K,double delta,double tol,int lev) {
  extern __shared__ double sm[];__shared__ int index;
  if(!MD_ACTIVITY_QUEUE){const int t=blockIdx.x;if(!st.active[t])return;if(ADAPTIVE)adaptive_one(st,L,nf,K,delta,tol,lev,t,sm);else dense_one(st,L,nf,K,lev==4?st.dense_floor4:st.dense_floor5,lev==4?st.dense_factor4:st.dense_factor5,t,sm);return;}
  for(;;){if(!threadIdx.x)index=atomicAdd(st.cursor,1);__syncthreads();if(index>=*st.queue_n)return;
    const int t=st.queue[index];
    if(st.active[t]){if(ADAPTIVE)adaptive_one(st,L,nf,K,delta,tol,lev,t,sm);
    else dense_one(st,L,nf,K,lev==4?st.dense_floor4:st.dense_floor5,lev==4?st.dense_factor4:st.dense_factor5,t,sm);}
    __syncthreads();
  }
}

__global__ void refine(Device st, int traces, int L, int nf, int K, double tol) {
  const int q=blockIdx.x*blockDim.x+threadIdx.x,index=q/K,m=q%K;
  if(index>=*st.queue_n)return;const int t=st.queue[index];
  if(!st.active[t])return;
  if(st.status[t]||m>=st.count[t])return;
  const size_t p=static_cast<size_t>(t)*K+m;
  double lo=st.lo[p], hi=st.hi[p], fl=st.flo[p], fh=st.fhi[p];
  const double omega=md::two_pi<double>()/st.period[t%nf];
  int eval_count=0;
  bool good=(lo==hi)||(isfinite(fl)&&isfinite(fh)&&signbit(fl)!=signbit(fh));
  const md::Medium med=medium(st,t/nf,L);
  st.roots[p]=good?md::refine_root(med,omega,lo,hi,fl,fh,tol,eval_count):CUDART_NAN;
  good=isfinite(st.roots[p]);const unsigned long long evals=eval_count;
  if (!good) {
    atomicOr(&st.flags[t],REFINE_FAILED); atomicExch(&st.status[t],1);
    atomicAdd(&st.stats[REFINE_FAILURES],1ULL);
  }
  if (evals) atomicAdd(&st.stats[EVALS],evals);
}

__global__ void compare(Device st, int traces, int L, int nf, int K, int modes,
                        double tol, double f_accept, int level, bool force,
                        bool finish) {
  const int index=blockIdx.x*blockDim.x+threadIdx.x;
  if(index>=*st.queue_n)return;const int t=st.queue[index];
  if(!st.active[t])return;
  const size_t p=static_cast<size_t>(t)*K, off=static_cast<size_t>(t/nf)*L;
  const int count=st.count[t];
  if(level==0&&!st.status[t]){
    double *destination=st.guarded?st.guard_roots:st.full_roots;
    for(int m=0;m<count;++m)destination[p+m]=st.roots[p+m];
    st.guard_count[t]=count;
  }
  bool same=!st.status[t] && count>=0 && count==st.prev_count[t];
  for (int m=0; m<count && same; ++m) {
    const double now=st.roots[p+m], prev=st.previous[p+m];
    same=md::roots_agree(now,prev,tol);
    if (m>0 && !(now>st.roots[p+m-1])) same=false;
  }
  if(level==3&&(st.flags[t]&(WORKSPACE|NONFINITE|REFINE_FAILED|512u)))same=false;
  if(same&&!force&&st.verify[t]!=2&&st.guard_count[t]>=0){
    const double *witness=st.guarded?st.guard_roots+p:st.full_roots+p;
    if(!md::contains_witnesses(st.roots+p,count,witness,st.guard_count[t],K,tol)){
      same=false;if(!(st.flags[t]&1024u))atomicAdd(&st.stats[CONFLICTS],1ULL);st.flags[t]|=1024u;
    }
  }
  if(same&&level==5&&st.verify[t]==2){
    same=md::contains_witnesses(st.roots+p,count,st.full_roots+p,st.full_count[t],K,tol)&&md::contains_witnesses(st.roots+p,count,st.guard_roots+p,st.guard_count[t],K,tol);
  }
  const bool adaptive_bad=(st.flags[t]&(WORKSPACE|NONFINITE|REFINE_FAILED|512u|CUTOFF))!=0;
  const bool spaced=count>0&&st.risk[t]==0&&roots_spaced(st.roots+p,count,st.sample_gap[t],
                                                        level>=4?st.dense_gap_factor:st.gap_factor);
  const double per=st.period[t%nf];
  const int early=st.accuracy==0?3:(per>=st.period_threshold?1:2);
  const bool early_accept=st.accuracy!=0&&level>=1&&level<=2&&level>=early&&same&&!adaptive_bad&&spaced;
  const bool dense_bad=(st.flags[t]&(CUTOFF|NONFINITE|1024u))!=0;
  const bool dense_accept=st.allow_dense_early&&st.accuracy!=0&&st.verify[t]==0&&!dense_bad&&!st.status[t]&&spaced&&
                          ((level==4&&!finish)||(level==5&&finish&&st.escalated[t]));
  const bool accept=(same&&!force)||early_accept||dense_accept;
  if (accept) {
    if(early_accept)atomicAdd(&st.stats[EARLY_ACCEPTS],1ULL);
    if(dense_accept)atomicAdd(&st.stats[level==4?DENSE_COARSE_OK:DENSE_FINE_ESCALATED],1ULL);
    st.active[t]=0; st.level[t]=level;
    bool near=false;
    for (int m=0; m<modes; ++m) {
      double root=m<count ? st.roots[p+m] : CUDART_NAN;
      if (isfinite(root) && root>=0.99*st.b[off+L-1]) near=true;
      if (isfinite(root) && f_accept<1.0) {
        const double omega=md::two_pi<double>()/st.period[t%nf];
        const double fr=md::evaluate(medium(st,t/nf,L),omega,root);
        atomicAdd(&st.stats[EVALS],1ULL);
        if (!isfinite(fr) || fabs(fr)>f_accept) {
          root=CUDART_NAN; st.flags[t]|=F_REJECTED;
          atomicAdd(&st.stats[ACCEPT_REJECTED],1ULL);
        }
      }
      st.output[(static_cast<size_t>(t/nf)*modes+m)*nf+t%nf]=root;
    }
    if (near) st.flags[t]|=CUTOFF;
  } else {
    st.prev_count[t]=st.status[t] ? -1 : count;
    if(!st.status[t])for (int m=0; m<count; ++m) st.previous[p+m]=st.roots[p+m];
    if(!finish)st.next_queue[atomicAdd(st.next_n,1)]=t;
    if(level==4&&!finish)st.escalated[t]=1;
    if (finish) {
      st.active[t]=0; st.flags[t]|=UNCONVERGED;
      for (int m=0; m<modes; ++m)
        st.output[(static_cast<size_t>(t/nf)*modes+m)*nf+t%nf]=CUDART_NAN;
    }
  }
}

// A second sampling family is a verification, never a replacement for the
// full-window density ladder. Only rich, successfully sampled prefixes need
// the additional check. Disagreements retain both root sets as witnesses.
__global__ void prepare_verification(Device st,int traces,int K){
  const int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=traces)return;
  st.active[t]=0;st.verify[t]=0;
  if(st.accuracy!=0&&(st.level[t]==1||st.level[t]==2)){atomicAdd(&st.stats[VERIFICATION_SKIPPED],1ULL);return;}
  if(st.level[t]!=3||(st.flags[t]&(UNCONVERGED|WORKSPACE|NONFINITE|REFINE_FAILED|512u)))return;
  const size_t p=static_cast<size_t>(t)*K;
  if(st.accuracy!=0&&st.verify_safe_skip){
    const int count=st.count[t];
    if(count>0&&st.risk[t]==0&&roots_spaced(st.roots+p,count,st.sample_gap[t],st.gap_factor)){
      atomicAdd(&st.stats[VERIFICATION_SKIPPED],1ULL);return;
    }
  }
  st.full_count[t]=st.count[t];
  for(int m=0;m<K;++m){st.full_roots[p+m]=m<st.count[t]?st.roots[p+m]:CUDART_NAN;st.previous[p+m]=CUDART_NAN;}
  st.verify[t]=1;st.active[t]=1;st.prev_count[t]=-1;st.count[t]=0;st.status[t]=0;st.level[t]=-1;st.guard_count[t]=-1;
  st.queue[atomicAdd(st.queue_n,1)]=t;atomicAdd(&st.stats[VERIFICATIONS],1ULL);
}

__global__ void reconcile(Device st,int traces,int K,double tol){
  const int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=traces||st.verify[t]!=1)return;
  const size_t p=static_cast<size_t>(t)*K;
  bool same=!(st.flags[t]&UNCONVERGED)&&st.count[t]==st.full_count[t];
  for(int m=0;m<st.count[t]&&same;++m)same=md::roots_agree(st.roots[p+m],st.full_roots[p+m],tol);
  if(same){st.verify[t]=0;return;}
  if(!(st.flags[t]&1024u))atomicAdd(&st.stats[CONFLICTS],1ULL);st.flags[t]|=1024u;
  if(st.flags[t]&UNCONVERGED){st.verify[t]=0;return;}
  st.guard_count[t]=st.count[t];
  for(int m=0;m<st.count[t];++m)st.guard_roots[p+m]=st.roots[p+m];
  st.verify[t]=2;st.active[t]=1;st.prev_count[t]=-1;st.count[t]=0;st.status[t]=0;st.level[t]=-1;
  st.queue[atomicAdd(st.queue_n,1)]=t;
}

__global__ void final_statistics(Device st,int traces){
  const int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=traces)return;
  if(st.flags[t]&UNCONVERGED)atomicAdd(&st.stats[FAILED],1ULL);
  else if(st.level[t]>=1&&st.level[t]<=3)atomicAdd(&st.stats[ADAPT_OK],1ULL);
  else if(st.level[t]==4||st.level[t]==5)atomicAdd(&st.stats[DENSE_OK],1ULL);
  if(st.flags[t]&CUTOFF)atomicAdd(&st.stats[CUTOFFS],1ULL);
}

inline void check(cudaError_t e) {
  if (e!=cudaSuccess) throw std::runtime_error(std::string("QE CUDA: ")+cudaGetErrorString(e));
}
template<class T> void alloc(T *&p, size_t n) { check(cudaMalloc(reinterpret_cast<void **>(&p),n*sizeof(T))); }
template<class T> void upload(T *p,const T *h,size_t n) { check(cudaMemcpy(p,h,n*sizeof(T),cudaMemcpyHostToDevice)); }
template<class T> void download(T *h,const T *p,size_t n) { check(cudaMemcpy(h,p,n*sizeof(T),cudaMemcpyDeviceToHost)); }

using HostClock=std::chrono::steady_clock;
double host_seconds(HostClock::time_point t){return std::chrono::duration<double>(HostClock::now()-t).count();}
int env_integer(const char *name,int fallback,int low,int high){const char *v=std::getenv(name);if(!v)return fallback;char *end=nullptr;long n=std::strtol(v,&end,10);if(!end||*end||n<low||n>high)throw std::invalid_argument(std::string("Invalid ")+name);return static_cast<int>(n);}

class Solver {
  Device st_{};
  int B_,L_,nf_,modes_,K_,blocks_,device_=0,block_=256;
  cudaStream_t stream_=nullptr;
  void *host_input_=nullptr,*host_output_=nullptr;
  double *hd_=nullptr,*ha_=nullptr,*hb_=nullptr,*hr_=nullptr,*hp_=nullptr,*hbound_=nullptr,*hv_=nullptr;
  unsigned *hf_=nullptr;int *hl_=nullptr,*hc_=nullptr,*hactivity_=nullptr,*activity_=nullptr;
  unsigned long long *hs_=nullptr;
  size_t host_bytes_=0,device_bytes_=0;
  struct Event {std::string name;cudaEvent_t begin=nullptr,end=nullptr;};
  std::vector<Event> events_;
  size_t event_index_=0;

  template<class F> void measured(const char *name,bool profile,F f){
    if(!profile){f();return;}
    if(event_index_==events_.size()){
      Event e;e.name=name;check(cudaEventCreate(&e.begin));
      try{check(cudaEventCreate(&e.end));}catch(...){cudaEventDestroy(e.begin);throw;}
      events_.push_back(e);
    }
    auto &e=events_[event_index_++];e.name=name;
    check(cudaEventRecord(e.begin,stream_));f();check(cudaEventRecord(e.end,stream_));
  }
  template<class T> void up(T *dst,const T *src,size_t n){check(cudaMemcpyAsync(dst,src,n*sizeof(T),cudaMemcpyHostToDevice,stream_));}
  template<class T> void down(T *dst,const T *src,size_t n){check(cudaMemcpyAsync(dst,src,n*sizeof(T),cudaMemcpyDeviceToHost,stream_));}
  void zero(void *dst,size_t n){check(cudaMemsetAsync(dst,0,n,stream_));}

  void enqueue(int B,double tol,double accept,bool profile){
    const size_t n=static_cast<size_t>(B)*L_;const int traces=B*nf_;
    measured("h2d",profile,[&]{up(st_.d,hd_,n);up(st_.a,ha_,n);up(st_.b,hb_,n);up(st_.r,hr_,n);up(st_.period,hp_,nf_);up(st_.bounds,hbound_,2*B);});
    measured("model_prepare",profile,[&]{prepare_models<<<B,256,0,stream_>>>(st_,B,L_);});
    zero(st_.stats,NSTATS*sizeof(unsigned long long));
    measured("initialize",profile,[&]{initialize<<<(traces+255)/256,256,0,stream_>>>(st_,traces,K_,L_,nf_,modes_);});
    const size_t sm=(2*CAP+(MD_TRACE_CONSTANTS?9:6)*L_)*sizeof(double);
    const double delta[4]={0.5,0.25,0.125,0.0625};int stage=0;
    auto stages=[&](int begin,bool finish_last,bool record){for(int lev=begin;lev<6;++lev){
      if(profile&&record)check(cudaMemcpyAsync(activity_+stage++,st_.queue_n,sizeof(int),cudaMemcpyDeviceToDevice,stream_));
      zero(st_.cursor,sizeof(int));zero(st_.next_n,sizeof(int));
      const int blocks=MD_ACTIVITY_QUEUE?std::min(traces,blocks_):traces;
      measured(lev<4?"search_adaptive":"search_dense",profile,[&]{
        if(lev<4)search<true><<<blocks,block_,sm,stream_>>>(st_,L_,nf_,K_,delta[lev],tol,lev);
        else search<false><<<blocks,block_,sm,stream_>>>(st_,L_,nf_,K_,0.,tol,lev);
      });
      measured("refinement",profile,[&]{refine<<<(traces*K_+255)/256,256,0,stream_>>>(st_,traces,L_,nf_,K_,tol);});
      measured("compare",profile,[&]{compare<<<(traces+255)/256,256,0,stream_>>>(st_,traces,L_,nf_,K_,modes_,tol,accept,lev,lev<3||lev==4,lev==5&&finish_last);});
      check(cudaGetLastError());std::swap(st_.queue,st_.next_queue);std::swap(st_.queue_n,st_.next_n);
    }};
    const bool high_mode=st_.accuracy==0;
    const int saved_f4=st_.dense_floor4,saved_f5=st_.dense_floor5,saved_g4=st_.dense_factor4,saved_g5=st_.dense_factor5,saved_early=st_.allow_dense_early;
    stages(0,high_mode,true);
    if(!high_mode){
      // Traces that could not prove spacing/agreement in the coarse dense
      // pass already sit in st_.queue. Give exactly those traces the HIGH
      // dense schedule so an extreme close-root case never becomes newly
      // unresolved merely because the coarse floor was cheaper.
      st_.dense_floor4=32768;st_.dense_floor5=32768;st_.dense_factor4=1;st_.dense_factor5=2;st_.allow_dense_early=0;
      stages(4,true,false);
      st_.dense_floor4=saved_f4;st_.dense_floor5=saved_f5;st_.dense_factor4=saved_g4;st_.dense_factor5=saved_g5;st_.allow_dense_early=saved_early;
    }
    zero(st_.queue_n,sizeof(int));
    measured("verification_prepare",profile,[&]{prepare_verification<<<(traces+255)/256,256,0,stream_>>>(st_,traces,K_);});
    st_.guarded=1;
    if(!high_mode){
      // The independent family is the conflict-resolution path; never give it
      // the cheaper dense floor. Only verified/conflicted traces pay this cost.
      st_.dense_floor4=32768;st_.dense_floor5=32768;st_.dense_factor4=1;st_.dense_factor5=2;st_.allow_dense_early=0;
    }
    stages(0,true,true);
    if(!high_mode){
      st_.dense_floor4=saved_f4;st_.dense_floor5=saved_f5;st_.dense_factor4=saved_g4;st_.dense_factor5=saved_g5;st_.allow_dense_early=saved_early;
    }
    zero(st_.queue_n,sizeof(int));
    measured("reconcile",profile,[&]{reconcile<<<(traces+255)/256,256,0,stream_>>>(st_,traces,K_,tol);});
    st_.guarded=0;
    if(high_mode)stages(4,true,true);
    else{
      // Reconcile disagreements are rare and numerically delicate: use the
      // exact HIGH dense schedule instead of the coarse preset.
      st_.dense_floor4=32768;st_.dense_floor5=32768;st_.dense_factor4=1;st_.dense_factor5=2;st_.allow_dense_early=0;
      stages(4,true,true);
      st_.dense_floor4=saved_f4;st_.dense_floor5=saved_f5;st_.dense_factor4=saved_g4;st_.dense_factor5=saved_g5;st_.allow_dense_early=saved_early;
    }
    measured("final_statistics",profile,[&]{final_statistics<<<(traces+255)/256,256,0,stream_>>>(st_,traces);});
    measured("d2h",profile,[&]{down(hv_,st_.output,static_cast<size_t>(B)*modes_*nf_);down(hf_,st_.flags,traces);down(hl_,st_.level,traces);down(hc_,st_.count,traces);down(hs_,st_.stats,NSTATS);if(profile)down(hactivity_,activity_,14);});
    check(cudaGetLastError());
  }
 public:
  Solver(int B,int L,int nf,int modes,int accuracy,double period_threshold):B_(B),L_(L),nf_(nf),modes_(modes),K_(modes+2){
    if(B<1||L<1||nf<1||modes<1||L>256||modes>256)throw py::value_error("QE geometry requires B,nf,modes>=1, 1<=L<=256, modes<=256");
    const md::AccuracyPolicy policy=md::make_accuracy_policy(accuracy,period_threshold);
    st_.accuracy=static_cast<int>(policy.accuracy);st_.accept_level1=policy.accept_level1;st_.accept_level2=policy.accept_level2;
    st_.period_threshold=policy.period_threshold;st_.gap_factor=policy.gap_factor;st_.dense_gap_factor=policy.dense_gap_factor;st_.trough_threshold=policy.trough_threshold;
    st_.dense_floor4=policy.dense_floor4;st_.dense_floor5=policy.dense_floor5;st_.dense_factor4=policy.dense_factor4;st_.dense_factor5=policy.dense_factor5;
    st_.verify_safe_skip=policy.verify_safe_skip?1:0;st_.allow_dense_early=policy.allow_dense_early?1:0;
    const size_t models=static_cast<size_t>(B)*L,tr=static_cast<size_t>(B)*nf,roots=tr*K_;
    check(cudaGetDevice(&device_));
    try{
      check(cudaStreamCreateWithFlags(&stream_,cudaStreamNonBlocking));
      block_=env_integer("MODISP_CUDA_BLOCK",256,128,256);if(block_!=128&&block_!=256)throw std::invalid_argument("MODISP_CUDA_BLOCK must be 128 or 256");
      cudaDeviceProp prop;check(cudaGetDeviceProperties(&prop,device_));blocks_=env_integer("MODISP_CUDA_BLOCKS_PER_SM",2,1,8)*prop.multiProcessorCount;
      const int shared=(2*CAP+(MD_TRACE_CONSTANTS?9:6)*L_)*sizeof(double);
      // Use one maximum for all supported geometries: another concurrent
      // solver must never lower this per-function attribute below our need.
      if(shared>static_cast<int>(prop.sharedMemPerBlock)){const int maximum=(2*CAP+(MD_TRACE_CONSTANTS?9:6)*256)*sizeof(double);check(cudaFuncSetAttribute(search<true>,cudaFuncAttributeMaxDynamicSharedMemorySize,maximum));check(cudaFuncSetAttribute(search<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,maximum));}
      auto allocate=[&](auto *&ptr,size_t count){alloc(ptr,count);device_bytes_+=count*sizeof(*ptr);};
      allocate(st_.d,models);allocate(st_.a,models);allocate(st_.b,models);allocate(st_.r,models);
      allocate(st_.period,nf);allocate(st_.bounds,2*B);allocate(st_.ia,models);allocate(st_.ib,models);
      allocate(st_.tx,static_cast<size_t>(B)*TABLE);allocate(st_.tn,static_cast<size_t>(B)*TABLE);allocate(st_.nt,B);allocate(st_.seeds,2*B);
      allocate(st_.cache_x,tr*CACHE);allocate(st_.cache_y,tr*CACHE);allocate(st_.cache_n,tr);
      allocate(st_.queue,tr);allocate(st_.next_queue,tr);allocate(st_.queue_n,1);allocate(st_.next_n,1);allocate(st_.cursor,1);
      allocate(st_.lo,roots);allocate(st_.hi,roots);allocate(st_.flo,roots);allocate(st_.fhi,roots);
      allocate(st_.roots,roots);allocate(st_.previous,roots);allocate(st_.output,tr*modes);
      allocate(st_.count,tr);allocate(st_.prev_count,tr);allocate(st_.status,tr);allocate(st_.active,tr);allocate(st_.level,tr);
      allocate(st_.flags,tr);allocate(st_.stats,NSTATS);allocate(st_.full_roots,roots);allocate(st_.guard_roots,roots);allocate(st_.verify,tr);allocate(st_.guard_count,tr);allocate(st_.full_count,tr);allocate(activity_,14);
      allocate(st_.sample_gap,tr);allocate(st_.risk,tr);allocate(st_.escalated,tr);
      const size_t in_bytes=(4*models+nf+2*B)*sizeof(double);
      const size_t output_base=tr*modes*sizeof(double)+3*tr*sizeof(int);
      const size_t stats_offset=(output_base+7)&~size_t(7);
      const size_t out_bytes=stats_offset+NSTATS*sizeof(unsigned long long)+14*sizeof(int);
      check(cudaHostAlloc(&host_input_,in_bytes,cudaHostAllocDefault));
      check(cudaHostAlloc(&host_output_,out_bytes,cudaHostAllocDefault));host_bytes_=in_bytes+out_bytes;
      hd_=static_cast<double*>(host_input_);ha_=hd_+models;hb_=ha_+models;hr_=hb_+models;hp_=hr_+models;hbound_=hp_+nf;
      hv_=static_cast<double*>(host_output_);hf_=reinterpret_cast<unsigned*>(hv_+tr*modes);hl_=reinterpret_cast<int*>(hf_+tr);hc_=hl_+tr;
      hs_=reinterpret_cast<unsigned long long*>(static_cast<char*>(host_output_)+stats_offset);hactivity_=reinterpret_cast<int*>(hs_+NSTATS);
    }catch(...){close();throw;}
  }
  Solver(const Solver&)=delete;
  ~Solver(){close();}
  void close(){
    int current=device_;cudaGetDevice(&current);cudaSetDevice(device_);
    if(stream_)cudaStreamSynchronize(stream_);
    for(auto &e:events_){if(e.begin)cudaEventDestroy(e.begin);if(e.end)cudaEventDestroy(e.end);}events_.clear();
#define QE_FREE(name) if(st_.name){cudaFree(st_.name);st_.name=nullptr;}
    QE_FREE(d) QE_FREE(a) QE_FREE(b) QE_FREE(r) QE_FREE(period) QE_FREE(bounds) QE_FREE(ia) QE_FREE(ib)
    QE_FREE(tx) QE_FREE(tn) QE_FREE(nt) QE_FREE(seeds) QE_FREE(cache_x) QE_FREE(cache_y) QE_FREE(cache_n)
    QE_FREE(queue) QE_FREE(next_queue) QE_FREE(queue_n) QE_FREE(next_n) QE_FREE(cursor)
    QE_FREE(lo) QE_FREE(hi) QE_FREE(flo) QE_FREE(fhi) QE_FREE(roots) QE_FREE(previous) QE_FREE(output)
    QE_FREE(count) QE_FREE(prev_count) QE_FREE(status) QE_FREE(active) QE_FREE(level) QE_FREE(flags) QE_FREE(stats)
    QE_FREE(full_roots) QE_FREE(guard_roots) QE_FREE(verify) QE_FREE(guard_count) QE_FREE(full_count)
    QE_FREE(sample_gap) QE_FREE(risk) QE_FREE(escalated)
#undef QE_FREE
    if(activity_){cudaFree(activity_);activity_=nullptr;}
    if(host_input_){cudaFreeHost(host_input_);host_input_=nullptr;}if(host_output_){cudaFreeHost(host_output_);host_output_=nullptr;}
    if(stream_){cudaStreamDestroy(stream_);stream_=nullptr;}cudaSetDevice(current);
  }
  py::tuple run(py::array_t<double,py::array::c_style|py::array::forcecast> d,
                py::array_t<double,py::array::c_style|py::array::forcecast> a,
                py::array_t<double,py::array::c_style|py::array::forcecast> b,
                py::array_t<double,py::array::c_style|py::array::forcecast> r,
                py::array_t<double,py::array::c_style|py::array::forcecast> periods,
                py::array_t<double,py::array::c_style|py::array::forcecast> bounds,double tol,double accept,int wave,bool automatic,bool profile=false){
    if(!st_.d)throw std::runtime_error("QE solver is closed");check(cudaSetDevice(device_));
    if(d.ndim()!=2||d.shape(0)<1||d.shape(0)>B_||d.shape(1)!=L_)throw py::value_error("QE model shape exceeds allocated geometry");
    for(auto *v:{&a,&b,&r})if(v->ndim()!=2||v->shape(0)!=d.shape(0)||v->shape(1)!=L_)throw py::value_error("QE model shapes must agree");
    if(periods.ndim()!=1||periods.size()!=nf_||bounds.ndim()!=2||bounds.shape(0)!=d.shape(0)||bounds.shape(1)!=2||!(tol>0))throw py::value_error("QE invalid periods, range or tolerance");
    const int B=static_cast<int>(d.shape(0)),traces=B*nf_;const size_t n=static_cast<size_t>(B)*L_;
    py::array_t<double> out({B,modes_,nf_});py::array_t<unsigned> flags({B,nf_});py::array_t<int> levels({B,nf_}),counts({B,nf_});
    double pack_s=0.,unpack_s=0.;
    {
      py::gil_scoped_release release;auto t=profile?HostClock::now():HostClock::time_point{};
      std::memcpy(hd_,d.data(),n*8);std::memcpy(ha_,a.data(),n*8);std::memcpy(hb_,b.data(),n*8);std::memcpy(hr_,r.data(),n*8);std::memcpy(hp_,periods.data(),nf_*8);std::memcpy(hbound_,bounds.data(),2*B*8);
      if(profile)pack_s=host_seconds(t);
      st_.wave=wave;st_.automatic=automatic;st_.guarded=0;st_.profile=profile;event_index_=0;
      enqueue(B,tol,accept,profile);
      check(cudaStreamSynchronize(stream_));
      t=profile?HostClock::now():HostClock::time_point{};
      std::memcpy(out.mutable_data(),hv_,static_cast<size_t>(B)*modes_*nf_*8);std::memcpy(flags.mutable_data(),hf_,traces*4);std::memcpy(levels.mutable_data(),hl_,traces*4);std::memcpy(counts.mutable_data(),hc_,traces*4);
      if(profile)unpack_s=host_seconds(t);
    }
    py::dict stats;
    for(int i=0;i<NSTATS;++i)if(i<CACHE_HITS||profile||i>=EARLY_ACCEPTS)stats[stat_names[i]]=hs_[i];
    stats["strategy"]="qe_adaptive";stats["capacity"]=CAP;stats["density_table_knots"]=129;stats["cache_capacity"]=CACHE;
    stats["accuracy"]=md::accuracy_name(static_cast<md::Accuracy>(st_.accuracy));
    stats["period_threshold"]=st_.period_threshold;stats["sampling_capacity"]=CAP;
    stats["dense_floor4"]=st_.dense_floor4;stats["dense_floor5"]=st_.dense_floor5;
    stats["pinned_host_bytes"]=host_bytes_;stats["device_workspace_bytes"]=device_bytes_;
    if(profile){py::dict timing;for(size_t i=0;i<event_index_;++i){float ms=0.;check(cudaEventElapsedTime(&ms,events_[i].begin,events_[i].end));py::str key(events_[i].name);double prev=timing.contains(key)?timing[key].cast<double>():0.;timing[key]=prev+ms*.001;}
      timing["host_pack"]=pack_s;timing["host_unpack"]=unpack_s;stats["timings_s"]=timing;py::list activity;for(int j=0;j<14;++j)activity.append(hactivity_[j]);stats["active_traces_by_stage"]=activity;}
    return py::make_tuple(out,flags,stats,levels,counts);
  }
};

} // namespace qe

void bind_qe(py::module_ &m) {
  m.def("available",[](int device){int n=0;return cudaGetDeviceCount(&n)==cudaSuccess&&device>=0&&device<n;},py::arg("device")=0);
  m.def("set_device",[](int device){qe::check(cudaSetDevice(device));});
  m.def("memory_info",[](){size_t free,total;qe::check(cudaMemGetInfo(&free,&total));return py::make_tuple(free,total);});
  m.def("qe_kernel_report",[](int L){cudaFuncAttributes a;qe::check(cudaFuncGetAttributes(&a,qe::search<true>));py::dict d;d["registers_per_thread"]=a.numRegs;d["static_shared_bytes"]=a.sharedSizeBytes;d["local_bytes_per_thread"]=a.localSizeBytes;int block=qe::env_integer("MODISP_CUDA_BLOCK",256,128,256),active=0;qe::check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,qe::search<true>,block,(2*qe::CAP+(MD_TRACE_CONSTANTS?9:6)*L)*sizeof(double)));d["search_block_threads"]=block;d["max_active_blocks_per_sm"]=active;d["configured_blocks_per_sm"]=qe::env_integer("MODISP_CUDA_BLOCKS_PER_SM",2,1,8);d["dynamic_shared_bytes"]=(2*qe::CAP+(MD_TRACE_CONSTANTS?9:6)*L)*sizeof(double);d["cache_points_per_trace"]=qe::CACHE;d["table_capacity_per_model"]=qe::TABLE;return d;});
  py::class_<qe::Solver>(m,"QeSolver")
    .def(py::init<int,int,int,int,int,double>(),py::arg("B"),py::arg("L"),py::arg("nf"),py::arg("nmodes"),py::arg("accuracy")=0,py::arg("period_threshold")=0.)
    .def("run",&qe::Solver::run,py::arg("thickness"),py::arg("vp"),py::arg("vs"),py::arg("rho"),
         py::arg("periods"),py::arg("bounds"),py::arg("c_tol_rel")=1e-6,py::arg("f_accept")=1.0,py::arg("wave")=0,py::arg("automatic")=false,py::arg("profile")=false)
    .def("close",&qe::Solver::close);
}
