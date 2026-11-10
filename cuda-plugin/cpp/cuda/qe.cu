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
#include "common/shape.hpp"
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
constexpr unsigned FP64_REPAIR = 1u << 15;
enum Stat { EVALS, EXTREMA, BRACKETS, ADAPT_OK, DENSE_OK, FAILED,
            CAPACITY_HITS, CUTOFFS, NONFINITES, REFINE_FAILURES, SEARCHES,
            ACCEPT_REJECTED, ANALYTIC_EMPTY, RANGE_EMPTY, VERIFICATIONS, CONFLICTS, CACHE_HITS, CACHE_MISSES,
            EARLY_ACCEPTS, VERIFICATION_SKIPPED, DENSE_COARSE_OK, DENSE_FINE_ESCALATED, TROUGH_RISKS,
            SHAPE_ACCEPTED,SHAPE_SUBDIVISIONS,SHAPE_REUSED,SHAPE_DEPTH_LIMIT,SHAPE_WORK_LIMIT,LEGACY_FALLBACKS,LOVE_PHASE_OK,LOVE_PHASE_FALLBACK,FP64_REPAIRS,NSTATS };
const char *stat_names[NSTATS] = {
    "evaluations", "extrema_proposals", "brackets", "adaptive_converged",
    "dense_converged", "unconverged", "capacity_hits", "near_cutoff",
    "nonfinite", "refinement_failed", "searches", "rejected_f", "analytical_empty", "range_empty", "verification_traces", "family_conflicts", "cache_hits", "cache_misses",
    "early_accepts", "verification_skipped", "dense_coarse_accepted", "dense_fine_escalated", "trough_risks",
    "shape_accepted","shape_subdivisions","shape_reused","shape_depth_limit","shape_work_limit","legacy_fallbacks","love_phase_accepted","love_phase_fallbacks","fp64_repair_traces"};

struct Device {
  double *d, *a, *b, *r, *period, *bounds, *ia, *ib, *irho, *irho2, *tx, *tn, *seeds, *cache_x, *cache_y;
  float *float_models,*float_limits;
  int *nt, *cache_n, *queue, *next_queue, *queue_n, *next_n, *cursor;
  int wave, automatic, canonical_lower, guarded, profile, secular, refinement, love_phase;
  int accuracy, accept_level1, accept_level2, verify_safe_skip, allow_dense_early;
  int shape;double sampling_rtol;double *shape_x;int *shape_n;
  double close_root_rtol;
  double period_threshold, gap_factor, dense_gap_factor, trough_threshold;
  int dense_floor4, dense_floor5, dense_factor4, dense_factor5, dense_tile;
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
  return {st.d+o,st.a+o,st.b+o,st.r+o,st.ia+o,st.ib+o,L,st.wave,
          nullptr,nullptr,nullptr,st.irho+o,st.irho2+o,st.secular,st.refinement};
}

template<bool FP32>
__device__ auto secular_medium(Device st,md::Medium base,size_t offset) {
  if constexpr (FP32) {
    const int L=base.L;
    const float *h=st.float_models+6*offset,*a=h+L,*b=a+L,*r=b+L;
    const auto *constants=reinterpret_cast<const float *>(base.ka);
    return md::FloatMedium{base,h,a,b,r,r+L,r+2*L,st.float_limits[offset/L],constants};
  } else {
    return base;
  }
}

template<bool FP32>
__device__ void prepare_trace_layer(int i,int L,double omega,double vp,double vs,
                                    double *ka,double *kb,double *gammk) {
  if constexpr (MD_TRACE_CONSTANTS) {
    if constexpr (FP32) {
      // The three FP32 arrays fit in the existing dynamic shared tail.
      auto *values=reinterpret_cast<float *>(ka);
      const float w=static_cast<float>(omega),b=static_cast<float>(vs),z=b/w;
      values[i]=w/static_cast<float>(vp);
      values[i+L]=b==0.f?0.f:w/b;
      values[i+2*L]=2.f*z*z;
    } else {
      ka[i]=omega/vp;kb[i]=vs==0.?0.:omega/vs;
      const double z=vs/omega;gammk[i]=2.*z*z;
    }
  }
}

template<bool FP32>
__device__ bool root_after(double root,double previous) {
  if constexpr (FP32) return static_cast<float>(root)>static_cast<float>(previous);
  return root>previous;
}

__device__ bool fp32_sign_cluster(double root,double previous) {
  return root-previous<=8.*FLT_EPSILON*fabs(root);
}

__device__ int sort_width(int count) {
  int width=1;
  while (width<count) width*=2;
  return width;
}

template<bool WithValues = true>
__device__ void sort_pairs(double *x, double *y, int count) {
  const int width=sort_width(count);
  for (int k=2; k<=width; k*=2) {
    for (int j=k/2; j>0; j/=2) {
      for (int i=threadIdx.x; i<width; i+=blockDim.x) {
        const int q=i^j;
        if (q>i && (((i&k)==0) ? x[i]>x[q] : x[i]<x[q])) {
          const double tx=x[i];
          x[i]=x[q]; x[q]=tx;
          if constexpr (WithValues) {
            const double ty=y[i];
            y[i]=y[q]; y[q]=ty;
          }
        }
      }
      __syncthreads();
    }
  }
}

__device__ int unique_keys(double *x, int width) {
  int count=0;
  for (int i=0; i<width && isfinite(x[i]); ++i) {
    if (!count || x[i]>x[count-1]) x[count++]=x[i];
  }
  return count;
}

// Sort only the new extremum proposals. Parallel rank merge uses the unused
// shared-memory tail; large sequences fall back to the bounded bitonic sort.
__device__ void merge_pairs(double *x,double *y,int oldn,int extra) {
  if(!MD_MERGE){sort_pairs(x,y,oldn+extra);return;}
  const int n=oldn+extra, width=sort_width(extra);
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

template<bool FP32>
__global__ void prepare_models(Device st,int B,int L) {
  const int model=blockIdx.x;const size_t off=static_cast<size_t>(model)*L;
  __shared__ double x[TABLE];
  for(int j=threadIdx.x;j<L;j+=blockDim.x){
    st.ia[off+j]=1./(st.a[off+j]*st.a[off+j]);
    st.ib[off+j]=st.b[off+j]==0.?0.:1./(st.b[off+j]*st.b[off+j]);
    if constexpr (FP32) {
      float *values=st.float_models+6*off+j;
      values[0]=static_cast<float>(st.d[off+j]);
      values[L]=static_cast<float>(st.a[off+j]);
      values[2*L]=static_cast<float>(st.b[off+j]);
      values[3*L]=static_cast<float>(st.r[off+j]);
      values[4*L]=1.f/values[3*L];
      values[5*L]=values[4*L]*values[4*L];
    }
    st.irho[off+j]=1./st.r[off+j];
    st.irho2[off+j]=st.irho[off+j]*st.irho[off+j];
  }
  const double lo=st.bounds[2*model],hi=st.bounds[2*model+1];
  const int count=129+2*L, width=sort_width(count);
  for(int j=threadIdx.x;j<width;j+=blockDim.x){
    double c=CUDART_INF;
    if(j<129)c=lo+(hi-lo)*j/128.;
    else if(j<count){int k=(j-129)/2;c=(j-129)%2?st.b[off+k]:st.a[off+k];if(!(c>lo&&c<hi))c=CUDART_INF;}
    x[j]=c;
  }
  __syncthreads();sort_pairs<false>(x,nullptr,count);
  if(!threadIdx.x){st.nt[model]=unique_keys(x,width);
    if constexpr (FP32) {
      // Select the factored FP32 recurrence where the standard compound
      // terms can cancel as powers of Vs/c.
      double max_vs=0.;
      for(int j=0;j<L;++j)max_vs=fmax(max_vs,st.b[off+j]);
      st.float_limits[model]=static_cast<float>(max_vs/16.);
    }
    const int top=st.b[off]==0.?1:0;
    st.seeds[2*model]=st.wave?st.b[off+top]:md::rayleigh_reference(st.a[off+top],st.b[off+top]);
    st.seeds[2*model+1]=(!st.wave&&top)?md::scholte_reference(st.a[off],st.r[off],st.a[off+1],st.b[off+1],st.r[off+1]):CUDART_NAN;
  }
  __syncthreads();const md::Medium m=medium(st,model,L);
  for(int j=threadIdx.x;j<st.nt[model];j+=blockDim.x){st.tx[static_cast<size_t>(model)*TABLE+j]=x[j];st.tn[static_cast<size_t>(model)*TABLE+j]=md::density_base(m,x[j]);}
}

template<class M>
__device__ double cached_value(Device st,int trace,const M &m,double omega,double c,unsigned long long &evals,unsigned long long &hits) {
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

// One warp owns one trace, with roots refined concurrently across its lanes.
template<bool FP32>
__global__ void love_phase_search(Device st,int traces,int L,int nf,int K,int modes,double tol,double accept){
  extern __shared__ double data[];
  __shared__ int bad,nr;__shared__ double pl,ph;
  for(int t=blockIdx.x;t<traces;t+=gridDim.x){
    if(!st.active[t])continue;
    const int model=t/nf;const size_t offset=static_cast<size_t>(model)*L,p=static_cast<size_t>(t)*K;
    const double omega=md::two_pi<double>()/st.period[t%nf],low=st.bounds[2*model],upper=st.bounds[2*model+1];
    double high=upper;
    auto base=medium(st,model,L);
    if constexpr (!FP32) {
      double *h=data,*vs=h+L,*rho=vs+L,*kb=rho+L;
      for(int j=threadIdx.x;j<L;j+=blockDim.x){
        h[j]=base.h[j];vs[j]=base.vs[j];rho[j]=base.rho[j];
        kb[j]=vs[j]>0.?omega/vs[j]:0.;
      }
      __syncthreads();
      base.h=h;base.vs=vs;base.rho=rho;base.kb=kb;
    }
    const auto med=secular_medium<FP32>(st,base,offset);
    if constexpr (FP32) {
      // The FP64 upper bound can round to the half-space velocity in FP32,
      // where the decaying phase boundary cannot be initialized.
      const float inner=nextafterf(nextafterf(med.fvs[L-1],0.f),0.f);
      high=fmin(high,static_cast<double>(inner));
    }
    const double phase_tol=FP32?fmax(8.*FLT_EPSILON,fmin(tol,1e-8)):fmin(tol,1e-8);
    const double pi=md::two_pi<double>()*.5;unsigned long long evals=0;
    if(!threadIdx.x){
      pl=md::love_phase(med,omega,low);ph=md::love_phase(med,omega,high);evals+=2;nr=0;
      bad=!(high>low)||!isfinite(pl)||!isfinite(ph)||ph>pl||fabs(pl)>1048576.||fabs(ph)>1048576.;
      if constexpr (FP32) {
        if(!bad&&high<upper){
          const double inner_f=md::evaluate(med,omega,high),upper_f=md::evaluate(med,omega,upper);
          evals+=2;
          if(!isfinite(inner_f)||!isfinite(upper_f)||inner_f==0.||upper_f==0.||signbit(inner_f)!=signbit(upper_f)){
            bad=1;
            st.flags[t]|=CUTOFF;
          }
        }
      }
      if(!bad)for(int n=static_cast<int>(floor(pl/pi));n*pi>ph&&nr<K;--n)++nr;
    }
    __syncthreads();const int initial_bad=bad;
    if(!initial_bad)for(int j=threadIdx.x;j<nr;j+=blockDim.x){int ev=0;const double level=(floor(pl/pi)-j)*pi;
      const double root=md::love_phase_root(med,omega,low,high,pl,ph,level,phase_tol,ev);evals+=ev;st.roots[p+j]=root;
      if(!isfinite(root))atomicExch(&bad,1);
    }
    __syncthreads();
    if(!threadIdx.x&&!bad){for(int j=1;j<nr;++j)if(!root_after<FP32>(st.roots[p+j],st.roots[p+j-1]))bad=1;
      if(!bad){st.count[t]=nr;st.level[t]=11;st.active[t]=0;if(md::has_close_roots(st.roots+p,nr,st.close_root_rtol))st.flags[t]|=4096u;}}
    __syncthreads();
    if(!bad)for(int j=threadIdx.x;j<modes;j+=blockDim.x){double root=j<nr?st.roots[p+j]:CUDART_NAN;
      if(isfinite(root)&&root>=.99*med.vs[L-1])atomicOr(&st.flags[t],CUTOFF);
      if(isfinite(root)&&accept<1.){double f=md::evaluate(med,omega,root);++evals;if(!isfinite(f)||fabs(f)>accept){root=CUDART_NAN;atomicOr(&st.flags[t],F_REJECTED);atomicAdd(&st.stats[ACCEPT_REJECTED],1ULL);}}
      st.output[(static_cast<size_t>(model)*modes+j)*nf+t%nf]=root;
    }
    __syncthreads();add_sampling_stat(&st.stats[EVALS],evals);
    if(!threadIdx.x){if(bad){st.flags[t]|=16384u;atomicAdd(&st.stats[LOVE_PHASE_FALLBACK],1ULL);}else{atomicAdd(&st.stats[LOVE_PHASE_OK],1ULL);atomicAdd(&st.stats[BRACKETS],static_cast<unsigned long long>(nr));}}
    __syncthreads();
  }
}

// A block owns one trace's sampling work. Every secular evaluation is parallel
// across candidate velocities; layers remain inside the register recurrence.
template<bool FP32>
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
    prepare_trace_layer<FP32>(i,L,omega,a[i],b[i],ka,kb,gammk);
  }
  if (threadIdx.x==0) { bad=0; overflow=0; st.status[t]=0; }
  __syncthreads();
  double upper=st.bounds[2*(t/nf)+1];
  // Refined guard roots can bound later adaptive work; the original global
  // grid/range is retained for both independent dense fallback resolutions.
  if(st.guarded&&lev>0&&st.prev_count[t]==K&&!(st.flags[t]&(WORKSPACE|NONFINITE|REFINE_FAILED|512u)))upper=fmin(upper,st.previous[static_cast<size_t>(t)*K+K-1]*(1.+4.*tol));
  const double cmin=(st.automatic && lev<2)?fmin((lev==0?4.:2.)*st.bounds[2*(t/nf)],(st.bounds[2*(t/nf)]+upper)*.5):st.bounds[2*(t/nf)];
  const auto med=secular_medium<FP32>(st,
      {h,a,b,r,ia,ib,L,st.wave,MD_TRACE_CONSTANTS?ka:nullptr,MD_TRACE_CONSTANTS?kb:nullptr,
       MD_TRACE_CONSTANTS?gammk:nullptr,st.irho+off,st.irho2+off,st.secular,st.refinement},off);
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
  const int nn=static_cast<int>(cells), initial=64+6*L+nn+2, width=sort_width(initial);
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
    x[i]=(c>=cmin && c<=upper)?c:CUDART_INF;
  }
  for (int i=initial+threadIdx.x; i<width; i+=blockDim.x) x[i]=CUDART_INF;
  __syncthreads();
  sort_pairs<false>(x,nullptr,initial);
  if (!threadIdx.x) n=unique_keys(x,width);
  __syncthreads();
  unsigned long long evals=0, extrema=0, hits=0;
  for (int i=threadIdx.x; i<n; i+=blockDim.x) {
    y[i]=cached_value(st,t,med,omega,x[i],evals,hits);
    if (!isfinite(y[i])) atomicExch(&bad,1);
  }
  for (int i=n+threadIdx.x; i<CAP; i+=blockDim.x) { x[i]=CUDART_INF; y[i]=0.0; }
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
template<bool FP32>
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
    prepare_trace_layer<FP32>(i,L,omega,a[i],b[i],ka,kb,gammk);
  }
  if (!threadIdx.x) { nr=0; bad=0; st.status[t]=0; atomicAdd(&st.stats[SEARCHES],1ULL); }
  __syncthreads();
  const double cmin=st.bounds[2*(t/nf)],upper=st.bounds[2*(t/nf)+1];
  const auto med=secular_medium<FP32>(st,
      {h,a,b,r,ia,ib,L,st.wave,MD_TRACE_CONSTANTS?ka:nullptr,MD_TRACE_CONSTANTS?kb:nullptr,
       MD_TRACE_CONSTANTS?gammk:nullptr,st.irho+off,st.irho2+off,st.secular,st.refinement},off);
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
  for (int start=0; start<nc-1; start+=st.dense_tile-1) {
    const int n=min(st.dense_tile,nc-start);
    for (int i=threadIdx.x; i<n; i+=blockDim.x) {
      x[i]=cmin+(upper-cmin)*static_cast<double>(start+i)/(nc-1.0);
      y[i]=md::evaluate(med,omega,x[i]); ++evals;
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

// Observation-scale logarithmic scan. The relative grid spacing is <= the
// configured close-root tolerance (half of it in balanced). Missing even pairs
// inside a cell are explicitly approximate, never certified mode labels.
template<bool FP32>
__device__ void observation_one(Device st,int L,int nf,int K,int modes,double tol,double accept,int t,double *sm){
  double *x=sm,*y=x+CAP,*h=y+CAP,*a=h+L,*b=a+L,*r=b+L;
  double *ia=r+L,*ib=ia+L,*ka=ib+L,*kb=ka+L,*gammk=kb+L;
  __shared__ int nr,bad,n,end;
  const size_t off=static_cast<size_t>(t/nf)*L;
  const double period=st.period[t%nf],omega=md::two_pi<double>()/period;
  for(int i=threadIdx.x;i<L;i+=blockDim.x){
    h[i]=st.d[off+i];a[i]=st.a[off+i];b[i]=st.b[off+i];r[i]=st.r[off+i];
    ia[i]=st.ia[off+i];ib[i]=st.ib[off+i];
    prepare_trace_layer<FP32>(i,L,omega,a[i],b[i],ka,kb,gammk);
  }
  if(!threadIdx.x){nr=0;bad=0;st.flags[t]|=2048u;atomicAdd(&st.stats[SEARCHES],1ULL);}
  __syncthreads();
  const auto med=secular_medium<FP32>(st,
      {h,a,b,r,ia,ib,L,st.wave,MD_TRACE_CONSTANTS?ka:nullptr,MD_TRACE_CONSTANTS?kb:nullptr,
       MD_TRACE_CONSTANTS?gammk:nullptr,st.irho+off,st.irho2+off,st.secular,st.refinement},off);
  const double lower=st.bounds[2*(t/nf)],upper=st.bounds[2*(t/nf)+1];
  const double target=st.close_root_rtol*(st.accuracy==1?.5:1.);
  const double origin=log(lower),span=log(upper)-origin,need=ceil(span/log1p(target));
  if(!isfinite(need)||need>524288.){
    if(!threadIdx.x){st.flags[t]|=WORKSPACE|UNCONVERGED;st.level[t]=-1;st.count[t]=-1;atomicAdd(&st.stats[CAPACITY_HITS],1ULL);}
    for(int m=threadIdx.x;m<modes;m+=blockDim.x)st.output[(static_cast<size_t>(t/nf)*modes+m)*nf+t%nf]=CUDART_NAN;
    return;
  }
  const int cells=max(1,static_cast<int>(need));const double step=span/cells;
  unsigned long long evals=0;
  for(int start=0;start<cells;){
    // sort_pairs pads to a power of two and unique_pairs scans until infinity;
    // clear the entire shared tail on every tile and reused persistent block.
    for(int j=threadIdx.x;j<CAP;j+=blockDim.x){x[j]=CUDART_INF;y[j]=0.;}
    __syncthreads();
    const int last=min(cells,start+255),initial=last-start+1;
    if(!threadIdx.x){n=initial;end=last;}
    for(int j=threadIdx.x;j<initial;j+=blockDim.x){
      const int index=start+j;const double c=index==0?lower:(index==cells?upper:exp(origin+step*index));
      x[j]=c;y[j]=md::evaluate(med,omega,c);++evals;
    }
    __syncthreads();
    if(!threadIdx.x&&last<cells&&y[initial-1]==0.){
      ++end;const double c=end==cells?upper:exp(origin+step*end);
      x[n]=c;y[n]=md::evaluate(med,omega,c);++n;++evals;
    }
    __syncthreads();
    const int grid_n=n;const double left=x[0],right=x[grid_n-1];
    // Freeze each lane's grid_n/right before any lane appends to shared n.
    __syncthreads();
    for(int j=threadIdx.x;j<6*L+2;j+=blockDim.x){
      double c;
      if(j<6*L){const int layer=j/6,which=j%6;if(st.wave&&b[0]==0.&&layer==0)continue;const double v=which<3?b[layer]:a[layer];c=v*(1.+((which%3)-1)*1e-7);}
      else c=st.seeds[2*(t/nf)+j-6*L];
      if(c>left&&c<right){const int dest=atomicAdd(&n,1);x[dest]=c;y[dest]=md::evaluate(med,omega,c);++evals;}
    }
    __syncthreads();
    if(n!=grid_n){sort_pairs(x,y,n);if(!threadIdx.x)n=unique_pairs(x,y);}
    __syncthreads();
    if(!threadIdx.x){
      for(int j=1;j+1<n;++j){const double f0=y[j-1],f1=y[j],f2=y[j+1];
        if((f0<0.)==(f1<0.)&&(f1<0.)==(f2<0.)&&fabs(f1)<.05&&fabs(f1)<fabs(f0)&&fabs(f1)<fabs(f2)){st.flags[t]|=4096u;break;}}
      if(end==cells&&md::ambiguous_sh_cutoff(med,x[n-1],y[n-1])){bad=4;st.flags[t]|=CUTOFF;}
      if(!bad)collect(st,t,K,x,y,n,nr,bad,start==0,end==cells);
    }
    __syncthreads();
    if(bad||nr>=K)break;
    start=end;
  }
  add_sampling_stat(&st.stats[EVALS],evals);
  const size_t roots=static_cast<size_t>(t)*K;
  if(!bad){
    for(int m=threadIdx.x;m<min(nr,K);m+=blockDim.x){
      const size_t i=roots+m;int count=0;
      const double root=md::refine_root(med,omega,st.lo[i],st.hi[i],st.flo[i],st.fhi[i],tol,count);
      st.roots[i]=root;
      if(!isfinite(root)){atomicExch(&bad,2);atomicOr(&st.flags[t],REFINE_FAILED);}
      if(count)atomicAdd(&st.stats[EVALS],static_cast<unsigned long long>(count));
    }
  }
  __syncthreads();
  if(!threadIdx.x){
    if constexpr (FP32) {
      for(int m=1;m<min(nr,K)&&!bad;++m)
        if(!root_after<true>(st.roots[roots+m],st.roots[roots+m-1]))bad=2;
      if(bad==2)st.flags[t]|=REFINE_FAILED;
    }
    if(bad){st.flags[t]|=UNCONVERGED;if(bad&1)st.flags[t]|=NONFINITE;st.level[t]=-1;st.count[t]=-1;}
    else{
      st.level[t]=st.accuracy==1?6:7;st.count[t]=min(nr,K);
      if(md::has_close_roots(st.roots+roots,min(nr,K),st.close_root_rtol))st.flags[t]|=4096u;
      atomicAdd(&st.stats[VERIFICATION_SKIPPED],1ULL);
    }
    atomicAdd(&st.stats[BRACKETS],static_cast<unsigned long long>(nr));
  }
  __syncthreads();
  for(int m=threadIdx.x;m<modes;m+=blockDim.x){
    double root=!bad&&m<min(nr,K)?st.roots[roots+m]:CUDART_NAN;
    if(isfinite(root)&&root>=.99*b[L-1])atomicOr(&st.flags[t],CUTOFF);
    if(isfinite(root)&&accept<1.){
      const double f=md::evaluate(med,omega,root);atomicAdd(&st.stats[EVALS],1ULL);
      if(!isfinite(f)||fabs(f)>accept){root=CUDART_NAN;atomicOr(&st.flags[t],F_REJECTED);atomicAdd(&st.stats[ACCEPT_REJECTED],1ULL);}
    }
    st.output[(static_cast<size_t>(t/nf)*modes+m)*nf+t%nf]=root;
  }
  __syncthreads();
}

template<bool FP32>
__global__ void observation_search(Device st,int L,int nf,int K,int modes,double tol,double accept){
  extern __shared__ double sm[];__shared__ int index;
  for(;;){
    if(!threadIdx.x)index=atomicAdd(st.cursor,1);
    __syncthreads();if(index>=*st.queue_n)return;
    const int t=st.queue[index];
    if(st.active[t])observation_one<FP32>(st,L,nf,K,modes,tol,accept,t,sm);
    __syncthreads();
  }
}

// Persistent blocks consume only active traces, entirely on the device.
#include "cuda/shape.cuh"

template<bool ADAPTIVE,bool FP32>
__global__ void search(Device st,int L,int nf,int K,double delta,double tol,int lev) {
  extern __shared__ double sm[];__shared__ int index;
  if(!MD_ACTIVITY_QUEUE){const int t=blockIdx.x;if(!st.active[t])return;if(ADAPTIVE)adaptive_one<FP32>(st,L,nf,K,delta,tol,lev,t,sm);else dense_one<FP32>(st,L,nf,K,lev==4?st.dense_floor4:st.dense_floor5,lev==4?st.dense_factor4:st.dense_factor5,t,sm);return;}
  for(;;){if(!threadIdx.x)index=atomicAdd(st.cursor,1);__syncthreads();if(index>=*st.queue_n)return;
    const int t=st.queue[index];
    if(st.active[t]){if(ADAPTIVE)adaptive_one<FP32>(st,L,nf,K,delta,tol,lev,t,sm);
    else dense_one<FP32>(st,L,nf,K,lev==4?st.dense_floor4:st.dense_floor5,lev==4?st.dense_factor4:st.dense_factor5,t,sm);}
    __syncthreads();
  }
}

template<bool FP32>
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
  const auto med=secular_medium<FP32>(st,medium(st,t/nf,L),static_cast<size_t>(t/nf)*L);
  st.roots[p]=good?md::refine_root(med,omega,lo,hi,fl,fh,tol,eval_count):CUDART_NAN;
  good=isfinite(st.roots[p]);const unsigned long long evals=eval_count;
  if (!good) {
    atomicOr(&st.flags[t],REFINE_FAILED); atomicExch(&st.status[t],1);
    atomicAdd(&st.stats[REFINE_FAILURES],1ULL);
  }
  if (evals) atomicAdd(&st.stats[EVALS],evals);
}

template<bool FP32>
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
    same=md::roots_agree(now,prev,tol,FP32?8.*FLT_EPSILON:0.);
    if (m>0 && !root_after<FP32>(now,st.roots[p+m-1])) same=false;
    if constexpr (FP32) {
      if(m>0&&fp32_sign_cluster(now,st.roots[p+m-1])){same=false;st.flags[t]|=4096u;}
    }
  }
  if(level==3&&(st.flags[t]&(WORKSPACE|NONFINITE|REFINE_FAILED|512u)))same=false;
  if(same&&!force&&st.verify[t]!=2&&st.guard_count[t]>=0){
    const double *witness=st.guarded?st.guard_roots+p:st.full_roots+p;
    if(!md::contains_witnesses(st.roots+p,count,witness,st.guard_count[t],K,tol,FP32?8.*FLT_EPSILON:0.)){
      same=false;if(!(st.flags[t]&1024u))atomicAdd(&st.stats[CONFLICTS],1ULL);st.flags[t]|=1024u;
    }
  }
  if(same&&level==5&&st.verify[t]==2){
    same=md::contains_witnesses(st.roots+p,count,st.full_roots+p,st.full_count[t],K,tol,FP32?8.*FLT_EPSILON:0.)&&md::contains_witnesses(st.roots+p,count,st.guard_roots+p,st.guard_count[t],K,tol,FP32?8.*FLT_EPSILON:0.);
  }
  // All presets retain the high density ladder. Nested sparse/coarse grids
  // can agree while missing a pair; observed-root spacing cannot bypass it.
  if (same && !force) {
    st.active[t]=0; st.level[t]=level;
    bool near=false;
    for (int m=0; m<modes; ++m) {
      double root=m<count ? st.roots[p+m] : CUDART_NAN;
      if (isfinite(root) && root>=0.99*st.b[off+L-1]) near=true;
      if (isfinite(root) && f_accept<1.0) {
        const double omega=md::two_pi<double>()/st.period[t%nf];
        const double fr=md::evaluate(secular_medium<FP32>(st,medium(st,t/nf,L),off),omega,root);
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
  if(st.level[t]!=3||(st.flags[t]&(UNCONVERGED|WORKSPACE|NONFINITE|REFINE_FAILED|512u)))return;
  const size_t p=static_cast<size_t>(t)*K;
  st.full_count[t]=st.count[t];
  for(int m=0;m<K;++m){st.full_roots[p+m]=m<st.count[t]?st.roots[p+m]:CUDART_NAN;st.previous[p+m]=CUDART_NAN;}
  st.verify[t]=1;st.active[t]=1;st.prev_count[t]=-1;st.count[t]=0;st.status[t]=0;st.level[t]=-1;st.guard_count[t]=-1;
  st.queue[atomicAdd(st.queue_n,1)]=t;atomicAdd(&st.stats[VERIFICATIONS],1ULL);
}

template<bool FP32>
__global__ void reconcile(Device st,int traces,int K,double tol){
  const int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=traces||st.verify[t]!=1)return;
  const size_t p=static_cast<size_t>(t)*K;
  bool same=!(st.flags[t]&UNCONVERGED)&&st.count[t]==st.full_count[t];
  for(int m=0;m<st.count[t]&&same;++m)same=md::roots_agree(st.roots[p+m],st.full_roots[p+m],tol,FP32?8.*FLT_EPSILON:0.);
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
  else if((st.level[t]>=1&&st.level[t]<=3)||st.level[t]>=8)atomicAdd(&st.stats[ADAPT_OK],1ULL);
  else if(st.level[t]>=4)atomicAdd(&st.stats[DENSE_OK],1ULL);
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
  bool single_precision_;
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

  template<bool FP32>
  void enqueue(int B,double tol,double accept,bool profile){
    const size_t n=static_cast<size_t>(B)*L_;const int traces=B*nf_;
    measured("h2d",profile,[&]{up(st_.d,hd_,n);up(st_.a,ha_,n);up(st_.b,hb_,n);up(st_.r,hr_,n);up(st_.period,hp_,nf_);up(st_.bounds,hbound_,2*B);});
    measured("model_prepare",profile,[&]{prepare_models<FP32><<<B,256,0,stream_>>>(st_,B,L_);});
    if(st_.shape)measured("shape_model_prepare",profile,[&]{shape_prepare_models<<<B,256,0,stream_>>>(st_,B,L_);});
    zero(st_.stats,NSTATS*sizeof(unsigned long long));
    measured("initialize",profile,[&]{initialize<<<(traces+255)/256,256,0,stream_>>>(st_,traces,K_,L_,nf_,modes_);});
    if(st_.wave&&st_.love_phase)measured("love_phase_search",profile,[&]{love_phase_search<FP32><<<std::min(traces,blocks_),32,FP32?0:4*L_*sizeof(double),stream_>>>(st_,traces,L_,nf_,K_,modes_,tol,accept);});
    const size_t sm=(2*CAP+(MD_TRACE_CONSTANTS?9:6)*L_)*sizeof(double);
    if(profile)zero(activity_,14*sizeof(int));
    const double delta[4]={0.5,0.25,0.125,0.0625};int stage=0;
    auto stages=[&](int begin,bool finish_last,bool record){for(int lev=begin;lev<6;++lev){
      if(profile&&record)check(cudaMemcpyAsync(activity_+stage++,st_.queue_n,sizeof(int),cudaMemcpyDeviceToDevice,stream_));
      zero(st_.cursor,sizeof(int));zero(st_.next_n,sizeof(int));
      const int blocks=MD_ACTIVITY_QUEUE?std::min(traces,blocks_):traces;
      measured(lev<4?"search_adaptive":"search_dense",profile,[&]{
        if(lev<4)search<true,FP32><<<blocks,block_,sm,stream_>>>(st_,L_,nf_,K_,delta[lev],tol,lev);
        else search<false,FP32><<<blocks,block_,sm,stream_>>>(st_,L_,nf_,K_,0.,tol,lev);
      });
      measured("refinement",profile,[&]{refine<FP32><<<(traces*K_+255)/256,256,0,stream_>>>(st_,traces,L_,nf_,K_,tol);});
      measured("compare",profile,[&]{compare<FP32><<<(traces+255)/256,256,0,stream_>>>(st_,traces,L_,nf_,K_,modes_,tol,accept,lev,lev<3||lev==4,lev==5&&finish_last);});
      check(cudaGetLastError());std::swap(st_.queue,st_.next_queue);std::swap(st_.queue_n,st_.next_n);
    }};
    if(st_.shape){
      const size_t shape_shared=(md::shape_storage_doubles+(MD_TRACE_CONSTANTS?9:6)*L_)*sizeof(double);
      const int blocks=std::min(traces,blocks_);
      measured("shape_search",profile,[&]{
        if(FP32&&st_.accuracy!=0)
          shape_search<true,false,false><<<blocks,block_,shape_shared,stream_>>>(st_,L_,nf_,K_,modes_,tol,accept);
        else
          shape_search<FP32><<<blocks,block_,shape_shared,stream_>>>(st_,L_,nf_,K_,modes_,tol,accept);
      });
      if constexpr (FP32) {
        zero(st_.queue_n,sizeof(int));zero(st_.cursor,sizeof(int));
        if(st_.accuracy==0){
          prepare_shape_retry<<<(traces+255)/256,256,0,stream_>>>(st_,traces,shape_sign_uncertain);
          measured("shape_precision_repair",profile,[&]{
            const int limit=std::min(32,std::max(1,traces/32));
            shape_search<false,true><<<blocks,block_,shape_shared,stream_>>>(st_,L_,nf_,K_,modes_,tol,accept,limit);
          });
        }else{
          prepare_shape_retry<<<(traces+255)/256,256,0,stream_>>>(st_,traces,shape_frontier_full);
          measured("shape_frontier_retry",profile,[&]{
            shape_search<true><<<blocks,block_,shape_shared,stream_>>>(st_,L_,nf_,K_,modes_,tol,accept);
          });
        }
      }
      zero(st_.queue_n,sizeof(int));
      measured("shape_fallback_prepare",profile,[&]{prepare_shape_fallback<<<(traces+255)/256,256,0,stream_>>>(st_,traces);});
      zero(st_.cursor,sizeof(int));
    }
    if(st_.accuracy!=0){
      if(profile)check(cudaMemcpyAsync(activity_,st_.queue_n,sizeof(int),cudaMemcpyDeviceToDevice,stream_));
      measured("observation_search",profile,[&]{observation_search<FP32><<<std::min(traces,blocks_),block_,sm,stream_>>>(st_,L_,nf_,K_,modes_,tol,accept);});
    }else{
    stages(0,true,true);zero(st_.queue_n,sizeof(int));
    measured("verification_prepare",profile,[&]{prepare_verification<<<(traces+255)/256,256,0,stream_>>>(st_,traces,K_);});
    st_.guarded=1;stages(0,true,true);zero(st_.queue_n,sizeof(int));
    measured("reconcile",profile,[&]{reconcile<FP32><<<(traces+255)/256,256,0,stream_>>>(st_,traces,K_,tol);});
    st_.guarded=0;stages(4,true,true);
    }
    measured("final_statistics",profile,[&]{final_statistics<<<(traces+255)/256,256,0,stream_>>>(st_,traces);});
    measured("d2h",profile,[&]{down(hv_,st_.output,static_cast<size_t>(B)*modes_*nf_);down(hf_,st_.flags,traces);down(hl_,st_.level,traces);down(hc_,st_.count,traces);down(hs_,st_.stats,NSTATS);if(profile)down(hactivity_,activity_,14);});
    check(cudaGetLastError());
  }
  template<bool FP32>
  void set_shared_limits(const cudaDeviceProp &prop,bool shape){
    const int shared=(2*CAP+(MD_TRACE_CONSTANTS?9:6)*L_)*sizeof(double);
    const int maximum=(2*CAP+(MD_TRACE_CONSTANTS?9:6)*256)*sizeof(double);
    // Concurrent solvers can change this per-function attribute, so always
    // set the same maximum for a given kernel.
    if(shared>static_cast<int>(prop.sharedMemPerBlock)){
      check(cudaFuncSetAttribute(search<true,FP32>,cudaFuncAttributeMaxDynamicSharedMemorySize,maximum));
      check(cudaFuncSetAttribute(search<false,FP32>,cudaFuncAttributeMaxDynamicSharedMemorySize,maximum));
      check(cudaFuncSetAttribute(observation_search<FP32>,cudaFuncAttributeMaxDynamicSharedMemorySize,maximum));
    }
    if(shape){
      check(cudaFuncSetAttribute(shape_search<FP32>,cudaFuncAttributeMaxDynamicSharedMemorySize,maximum));
      if constexpr (FP32){
        check(cudaFuncSetAttribute(shape_search<false,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,maximum));
        check(cudaFuncSetAttribute(shape_search<true,false,false>,cudaFuncAttributeMaxDynamicSharedMemorySize,maximum));
      }
    }
  }
 public:
  Solver(int B,int L,int nf,int modes,int accuracy,double period_threshold,double close_root_rtol,bool shape=false,double sampling_rtol=.001,int secular=0,int refinement=1,int love_phase=1,bool single_precision=false)
      :B_(B),L_(L),nf_(nf),modes_(modes),K_(modes+2),single_precision_(single_precision){
    if(B<1||L<1||nf<1||modes<1||L>256||modes>256)throw py::value_error("QE geometry requires B,nf,modes>=1, 1<=L<=256, modes<=256");
    if(secular<0||secular>1)throw py::value_error("secular must be 0 (dunkin) or 1 (fast)");
    st_.secular=secular;st_.refinement=refinement;st_.love_phase=love_phase;
    if(refinement<0||refinement>2||love_phase<0||love_phase>1)throw py::value_error("invalid refinement or Love strategy");
    const md::AccuracyPolicy policy=md::make_accuracy_policy(accuracy,period_threshold,close_root_rtol);
    st_.close_root_rtol=policy.close_root_rtol;
    st_.shape=shape;st_.sampling_rtol=sampling_rtol;
    st_.accuracy=static_cast<int>(policy.accuracy);st_.accept_level1=policy.accept_level1;st_.accept_level2=policy.accept_level2;
    st_.period_threshold=policy.period_threshold;st_.gap_factor=policy.gap_factor;st_.dense_gap_factor=policy.dense_gap_factor;st_.trough_threshold=policy.trough_threshold;
    st_.dense_floor4=policy.dense_floor4;st_.dense_floor5=policy.dense_floor5;st_.dense_factor4=policy.dense_factor4;st_.dense_factor5=policy.dense_factor5;
    st_.dense_tile=policy.cuda_dense_tile;
    st_.verify_safe_skip=policy.verify_safe_skip?1:0;st_.allow_dense_early=policy.allow_dense_early?1:0;
    const size_t models=static_cast<size_t>(B)*L,tr=static_cast<size_t>(B)*nf,roots=tr*K_;
    check(cudaGetDevice(&device_));
    try{
      check(cudaStreamCreateWithFlags(&stream_,cudaStreamNonBlocking));
      block_=env_integer("MODISP_CUDA_BLOCK",shape?32:256,shape?32:128,256);if((shape?(block_!=32&&block_!=64&&block_!=128&&block_!=256):(block_!=128&&block_!=256)))throw std::invalid_argument(shape?"MODISP_CUDA_BLOCK must be 32, 64, 128 or 256":"MODISP_CUDA_BLOCK must be 128 or 256");
      cudaDeviceProp prop;check(cudaGetDeviceProperties(&prop,device_));blocks_=env_integer("MODISP_CUDA_BLOCKS_PER_SM",shape?12:2,1,shape?16:8)*prop.multiProcessorCount;
      if(single_precision_)set_shared_limits<true>(prop,shape);
      else set_shared_limits<false>(prop,shape);
      auto allocate=[&](auto *&ptr,size_t count){alloc(ptr,count);device_bytes_+=count*sizeof(*ptr);};
      allocate(st_.d,models);allocate(st_.a,models);allocate(st_.b,models);allocate(st_.r,models);
      allocate(st_.period,nf);allocate(st_.bounds,2*B);allocate(st_.ia,models);allocate(st_.ib,models);
      allocate(st_.irho,models);allocate(st_.irho2,models);
      if(single_precision_){allocate(st_.float_models,6*models);allocate(st_.float_limits,B);}
      allocate(st_.tx,static_cast<size_t>(B)*TABLE);allocate(st_.tn,static_cast<size_t>(B)*TABLE);allocate(st_.nt,B);allocate(st_.seeds,2*B);
      if(shape){allocate(st_.shape_x,static_cast<size_t>(B)*CAP);allocate(st_.shape_n,B);}
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
    QE_FREE(d) QE_FREE(a) QE_FREE(b) QE_FREE(r) QE_FREE(period) QE_FREE(bounds) QE_FREE(ia) QE_FREE(ib) QE_FREE(irho) QE_FREE(irho2)
    QE_FREE(float_models) QE_FREE(float_limits)
    QE_FREE(tx) QE_FREE(tn) QE_FREE(nt) QE_FREE(seeds) QE_FREE(cache_x) QE_FREE(cache_y) QE_FREE(cache_n)
    QE_FREE(queue) QE_FREE(next_queue) QE_FREE(queue_n) QE_FREE(next_n) QE_FREE(cursor)
    QE_FREE(lo) QE_FREE(hi) QE_FREE(flo) QE_FREE(fhi) QE_FREE(roots) QE_FREE(previous) QE_FREE(output)
    QE_FREE(count) QE_FREE(prev_count) QE_FREE(status) QE_FREE(active) QE_FREE(level) QE_FREE(flags) QE_FREE(stats)
    QE_FREE(full_roots) QE_FREE(guard_roots) QE_FREE(verify) QE_FREE(guard_count) QE_FREE(full_count)
    QE_FREE(sample_gap) QE_FREE(risk) QE_FREE(escalated)
    QE_FREE(shape_x) QE_FREE(shape_n)
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
                py::array_t<double,py::array::c_style|py::array::forcecast> bounds,double tol,double accept,int wave,bool automatic,bool profile=false,bool canonical_lower=false){
    if(!st_.d)throw std::runtime_error("QE solver is closed");check(cudaSetDevice(device_));
    if(d.ndim()!=2||d.shape(0)<1||d.shape(0)>B_||d.shape(1)!=L_)throw py::value_error("QE model shape exceeds allocated geometry");
    for(auto *v:{&a,&b,&r})if(v->ndim()!=2||v->shape(0)!=d.shape(0)||v->shape(1)!=L_)throw py::value_error("QE model shapes must agree");
    if(periods.ndim()!=1||periods.size()!=nf_||bounds.ndim()!=2||bounds.shape(0)!=d.shape(0)||bounds.shape(1)!=2||!(tol>0))throw py::value_error("QE invalid periods, range or tolerance");
    const int B=static_cast<int>(d.shape(0)),traces=B*nf_;const size_t n=static_cast<size_t>(B)*L_;
    const py::array::ShapeContainer output_shape{B,modes_,nf_};
    py::array out(single_precision_?py::dtype::of<float>():py::dtype::of<double>(),output_shape);
    py::array_t<unsigned> flags({B,nf_});py::array_t<int> levels({B,nf_}),counts({B,nf_});
    void *velocity_out=out.mutable_data();
    auto *flags_out=flags.mutable_data();
    auto *levels_out=levels.mutable_data(),*counts_out=counts.mutable_data();
    const size_t velocities=static_cast<size_t>(B)*modes_*nf_;
    double pack_s=0.,unpack_s=0.;
    {
      py::gil_scoped_release release;auto t=profile?HostClock::now():HostClock::time_point{};
      std::memcpy(hd_,d.data(),n*8);std::memcpy(ha_,a.data(),n*8);std::memcpy(hb_,b.data(),n*8);std::memcpy(hr_,r.data(),n*8);std::memcpy(hp_,periods.data(),nf_*8);std::memcpy(hbound_,bounds.data(),2*B*8);
      if(profile)pack_s=host_seconds(t);
      st_.wave=wave;st_.automatic=automatic;st_.canonical_lower=single_precision_&&automatic&&canonical_lower;st_.guarded=0;st_.profile=profile;event_index_=0;
      if(single_precision_)enqueue<true>(B,tol,accept,profile);
      else enqueue<false>(B,tol,accept,profile);
      check(cudaStreamSynchronize(stream_));
      t=profile?HostClock::now():HostClock::time_point{};
      if(single_precision_)std::copy_n(hv_,velocities,static_cast<float *>(velocity_out));
      else std::memcpy(velocity_out,hv_,velocities*sizeof(double));
      std::memcpy(flags_out,hf_,traces*sizeof(*flags_out));
      std::memcpy(levels_out,hl_,traces*sizeof(*levels_out));
      std::memcpy(counts_out,hc_,traces*sizeof(*counts_out));
      if(profile)unpack_s=host_seconds(t);
    }
    py::dict stats;
    for(int i=0;i<NSTATS;++i)if(i<CACHE_HITS||profile||i>=EARLY_ACCEPTS)stats[stat_names[i]]=hs_[i];
    stats["strategy"]="qe_adaptive";stats["capacity"]=CAP;stats["density_table_knots"]=129;stats["cache_capacity"]=CACHE;
    stats["dense_tile"]=st_.accuracy==0?st_.dense_tile:256;
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
  m.attr("SUPPORTS_FP32")=true;
  m.attr("SUPPORTS_CANONICAL_LOWER")=true;
  m.def("love_phase_kernel_report",[](int L,bool single_precision){
    if(L<1||L>256)throw py::value_error("L must be 1..256");
    cudaFuncAttributes a;
    const auto kernel=single_precision?qe::love_phase_search<true>:qe::love_phase_search<false>;
    qe::check(cudaFuncGetAttributes(&a,kernel));
    py::dict d;d["registers_per_thread"]=a.numRegs;d["local_bytes_per_thread"]=a.localSizeBytes;
    d["static_shared_bytes"]=a.sharedSizeBytes;d["dynamic_shared_bytes"]=single_precision?0:4*L*sizeof(double);
    d["search_block_threads"]=32;return d;
  },py::arg("L"),py::arg("single_precision")=false);
  m.def("available",[](int device){int n=0;return cudaGetDeviceCount(&n)==cudaSuccess&&device>=0&&device<n;},py::arg("device")=0);
  m.def("set_device",[](int device){qe::check(cudaSetDevice(device));});
  m.def("memory_info",[](){size_t free,total;qe::check(cudaMemGetInfo(&free,&total));return py::make_tuple(free,total);});
  m.def("qe_kernel_report",[](int L,int accuracy,bool shape,bool single_precision){
    const void *kernel;
    if(shape){
      if(!single_precision)kernel=reinterpret_cast<const void*>(qe::shape_search<false>);
      else if(accuracy==0)kernel=reinterpret_cast<const void*>(qe::shape_search<true>);
      else kernel=reinterpret_cast<const void*>(qe::shape_search<true,false,false>);
    }
    else if(accuracy==0)kernel=reinterpret_cast<const void*>(single_precision?qe::search<true,true>:qe::search<true,false>);
    else kernel=reinterpret_cast<const void*>(single_precision?qe::observation_search<true>:qe::observation_search<false>);
    cudaFuncAttributes a;qe::check(cudaFuncGetAttributes(&a,kernel));
    const size_t sm=((shape?md::shape_storage_doubles:2*qe::CAP)+(MD_TRACE_CONSTANTS?9:6)*L)*sizeof(double);
    int block=qe::env_integer("MODISP_CUDA_BLOCK",shape?32:256,shape?32:128,256),active=0;
    qe::check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,kernel,block,sm));
    py::dict d;
    d["registers_per_thread"]=a.numRegs;d["static_shared_bytes"]=a.sharedSizeBytes;
    d["local_bytes_per_thread"]=a.localSizeBytes;d["search_block_threads"]=block;
    d["max_active_blocks_per_sm"]=active;
    d["configured_blocks_per_sm"]=qe::env_integer("MODISP_CUDA_BLOCKS_PER_SM",shape?12:2,1,shape?16:8);
    d["dynamic_shared_bytes"]=sm;d["cache_points_per_trace"]=qe::CACHE;d["table_capacity_per_model"]=qe::TABLE;
    d["search_kernel"]=shape?"shape_warp":(accuracy==0?"qe_adaptive":"observation_log");
    return d;
  },py::arg("L"),py::arg("accuracy")=0,py::arg("shape")=false,py::arg("single_precision")=false);
  py::class_<qe::Solver>(m,"QeSolver")
    .def(py::init<int,int,int,int,int,double,double,bool,double,int,int,int,bool>(),py::arg("B"),py::arg("L"),py::arg("nf"),py::arg("nmodes"),py::arg("accuracy")=0,py::arg("period_threshold")=0.,py::arg("close_root_rtol")=.005,py::arg("shape")=false,py::arg("sampling_rtol")=.001,py::arg("secular")=0,py::arg("refinement")=1,py::arg("love_phase")=1,py::arg("single_precision")=false)
    .def("run",&qe::Solver::run,py::arg("thickness"),py::arg("vp"),py::arg("vs"),py::arg("rho"),
         py::arg("periods"),py::arg("bounds"),py::arg("c_tol_rel")=1e-6,py::arg("f_accept")=1.0,py::arg("wave")=0,py::arg("automatic")=false,py::arg("profile")=false,py::arg("canonical_lower")=false)
    .def("close",&qe::Solver::close);
}
