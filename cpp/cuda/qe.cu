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
#include <stdexcept>
#include <string>
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

namespace py = pybind11;
namespace qe {
constexpr int TABLE=1024, CACHE=256;
constexpr int CAP = 2048;  // power of two; x/y live in block shared memory
constexpr unsigned UNCONVERGED = 1u << 2, WORKSPACE = 1u << 3;
constexpr unsigned CUTOFF = 1u << 4, NONFINITE = 1u << 5;
constexpr unsigned REFINE_FAILED = 1u << 6, F_REJECTED = 1u << 7;
enum Stat { EVALS, EXTREMA, BRACKETS, ADAPT_OK, DENSE_OK, FAILED,
            CAPACITY_HITS, CUTOFFS, NONFINITES, REFINE_FAILURES, SEARCHES,
            ACCEPT_REJECTED, ANALYTIC_EMPTY, RANGE_EMPTY, VERIFICATIONS, CONFLICTS, NSTATS };
const char *stat_names[NSTATS] = {
    "evaluations", "extrema_proposals", "brackets", "adaptive_converged",
    "dense_converged", "unconverged", "capacity_hits", "near_cutoff",
    "nonfinite", "refinement_failed", "searches", "rejected_f", "analytical_empty", "range_empty", "verification_traces", "family_conflicts"};

struct Device {
  double *d, *a, *b, *r, *period, *bounds, *ia, *ib, *tx, *tn, *seeds, *cache_x, *cache_y;
  int *nt, *cache_n, *queue, *next_queue, *queue_n, *next_n, *cursor;
  int wave, automatic, guarded;
  double *full_roots,*guard_roots;
  int *verify,*guard_count,*full_count;
  double *lo, *hi, *flo, *fhi, *roots, *previous, *output;
  int *count, *prev_count, *status, *active, *level;
  unsigned *flags;
  unsigned long long *stats;
};

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

__device__ double cached_value(Device st,int trace,const md::Medium &m,double omega,double c,unsigned long long &evals) {
  if(!MD_VALUE_CACHE){++evals;return md::evaluate(m,omega,c);}
  int lo=0,hi=st.cache_n[trace];const size_t o=static_cast<size_t>(trace)*CACHE;
  while(lo<hi){int q=(lo+hi)/2;if(st.cache_x[o+q]<c)lo=q+1;else hi=q;}
  if(lo<st.cache_n[trace]&&st.cache_x[o+lo]==c)return st.cache_y[o+lo];
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
  if(!t){*st.queue_n=traces;*st.next_n=0;*st.cursor=0;}
  for (int m=0; m<K; ++m){st.previous[static_cast<size_t>(t)*K+m]=CUDART_NAN;st.roots[static_cast<size_t>(t)*K+m]=CUDART_NAN;}
  if(st.bounds[2*(t/nf)+1]>st.bounds[2*(t/nf)]&&st.wave&&!md::sh_has_guide(medium(st,t/nf,L))){st.active[t]=0;st.level[t]=-3;for(int m=0;m<modes;++m)st.output[(static_cast<size_t>(t/nf)*modes+m)*nf+t%nf]=CUDART_NAN;atomicAdd(&st.stats[ANALYTIC_EMPTY],1ULL);}
  if(st.bounds[2*(t/nf)+1]<=st.bounds[2*(t/nf)]){st.active[t]=0;st.level[t]=-4;for(int m=0;m<modes;++m)st.output[(static_cast<size_t>(t/nf)*modes+m)*nf+t%nf]=CUDART_NAN;atomicAdd(&st.stats[RANGE_EMPTY],1ULL);}
}

// A block owns one trace's sampling work. Every secular evaluation is parallel
// across candidate velocities; layers remain inside the register recurrence.
__device__ void adaptive_one(Device st,int L,int nf,int K,double delta,double tol,int lev,int t,double *sm) {
  double *x=sm, *y=x+CAP, *h=y+CAP, *a=h+L, *b=a+L, *r=b+L;
  double *ia=r+L, *ib=ia+L;
  __shared__ int n, extra, bad, overflow;
  const int model=t/nf;
  const size_t off=static_cast<size_t>(model)*L;
  for (int i=threadIdx.x; i<L; i+=blockDim.x) {
    h[i]=st.d[off+i]; a[i]=st.a[off+i]; b[i]=st.b[off+i]; r[i]=st.r[off+i];
    ia[i]=st.ia[off+i]; ib[i]=st.ib[off+i];
  }
  for (int i=threadIdx.x; i<CAP; i+=blockDim.x) { x[i]=CUDART_INF; y[i]=0.0; }
  if (threadIdx.x==0) { bad=0; overflow=0; st.status[t]=0; }
  __syncthreads();
  double upper=st.bounds[2*(t/nf)+1];
  // Refined guard roots can bound later adaptive work; the original global
  // grid/range is retained for both independent dense fallback resolutions.
  if(st.guarded&&lev>0&&st.prev_count[t]==K&&!(st.flags[t]&(WORKSPACE|NONFINITE|REFINE_FAILED|512u)))upper=fmin(upper,st.previous[static_cast<size_t>(t)*K+K-1]*(1.+4.*tol));
  const double cmin=(st.automatic && lev<2)?fmin((lev==0?4.:2.)*st.bounds[2*(t/nf)],(st.bounds[2*(t/nf)]+upper)*.5):st.bounds[2*(t/nf)];
  const md::Medium med{h,a,b,r,ia,ib,L,st.wave};
  const double per=st.period[t%nf], omega=md::two_pi<double>()/per;
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
  unsigned long long evals=0, extrema=0;
  for (int i=threadIdx.x; i<n; i+=blockDim.x) {
    y[i]=cached_value(st,t,med,omega,x[i],evals);
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
        y[dest]=cached_value(st,t,med,omega,xe,evals);
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
  if (evals) atomicAdd(&st.stats[EVALS],evals);
  if (extrema) atomicAdd(&st.stats[EXTREMA],extrema);
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
__device__ void dense_one(Device st,int L,int nf,int K,int grid_factor,int t,double *sm) {
  double *x=sm, *y=x+CAP, *h=y+CAP, *a=h+L, *b=a+L, *r=b+L;
  double *ia=r+L, *ib=ia+L;
  __shared__ int nr, bad;
  const size_t off=static_cast<size_t>(t/nf)*L;
  for (int i=threadIdx.x; i<L; i+=blockDim.x) {
    h[i]=st.d[off+i]; a[i]=st.a[off+i]; b[i]=st.b[off+i]; r[i]=st.r[off+i];
    ia[i]=st.ia[off+i]; ib[i]=st.ib[off+i];
  }
  if (!threadIdx.x) { nr=0; bad=0; st.status[t]=0; atomicAdd(&st.stats[SEARCHES],1ULL); }
  __syncthreads();
  const double cmin=st.bounds[2*(t/nf)],upper=st.bounds[2*(t/nf)+1];
  const md::Medium med{h,a,b,r,ia,ib,L,st.wave};
  if (!(upper>cmin)) { if (!threadIdx.x) st.count[t]=0; return; }
  const double omega=md::two_pi<double>()/st.period[t%nf];
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
  int cells=32768;
  while (cells<need) cells*=2;
  const int nc=grid_factor*cells+1;
  unsigned long long evals=0;
  for (int start=0; start<nc-1; start+=CAP-1) {
    const int n=min(CAP,nc-start);
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
  if (evals) atomicAdd(&st.stats[EVALS],evals);
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
  if(!MD_ACTIVITY_QUEUE){const int t=blockIdx.x;if(!st.active[t])return;if(ADAPTIVE)adaptive_one(st,L,nf,K,delta,tol,lev,t,sm);else dense_one(st,L,nf,K,lev==4?1:2,t,sm);return;}
  for(;;){if(!threadIdx.x)index=atomicAdd(st.cursor,1);__syncthreads();if(index>=*st.queue_n)return;
    const int t=st.queue[index];
    if(st.active[t]){if(ADAPTIVE)adaptive_one(st,L,nf,K,delta,tol,lev,t,sm);
    else dense_one(st,L,nf,K,lev==4?1:2,t,sm);}
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
  if (same && !force) {
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
  else if(st.level[t]==3)atomicAdd(&st.stats[ADAPT_OK],1ULL);
  else if(st.level[t]==5)atomicAdd(&st.stats[DENSE_OK],1ULL);
  if(st.flags[t]&CUTOFF)atomicAdd(&st.stats[CUTOFFS],1ULL);
}

inline void check(cudaError_t e) {
  if (e!=cudaSuccess) throw std::runtime_error(std::string("QE CUDA: ")+cudaGetErrorString(e));
}
template<class T> void alloc(T *&p, size_t n) { check(cudaMalloc(reinterpret_cast<void **>(&p),n*sizeof(T))); }
template<class T> void upload(T *p,const T *h,size_t n) { check(cudaMemcpy(p,h,n*sizeof(T),cudaMemcpyHostToDevice)); }
template<class T> void download(T *h,const T *p,size_t n) { check(cudaMemcpy(h,p,n*sizeof(T),cudaMemcpyDeviceToHost)); }

class Solver {
  Device st_{};
  int B_,L_,nf_,modes_,K_,blocks_,device_=0;
 public:
  Solver(int B,int L,int nf,int modes):B_(B),L_(L),nf_(nf),modes_(modes),K_(modes+2) {
    if (B<1 || L<1 || nf<1 || modes<1 || L>256 || modes>256)
      throw py::value_error("QE geometry requires B,nf,modes>=1, 1<=L<=256, modes<=256");
    const size_t models=static_cast<size_t>(B)*L, tr=static_cast<size_t>(B)*nf, roots=tr*K_;
    check(cudaGetDevice(&device_));
    try {
      alloc(st_.d,models); alloc(st_.a,models); alloc(st_.b,models); alloc(st_.r,models);
      alloc(st_.period,nf);alloc(st_.bounds,2*B);alloc(st_.ia,models);alloc(st_.ib,models);
      alloc(st_.tx,static_cast<size_t>(B)*TABLE);alloc(st_.tn,static_cast<size_t>(B)*TABLE);alloc(st_.nt,B);alloc(st_.seeds,2*B);
      alloc(st_.cache_x,tr*CACHE);alloc(st_.cache_y,tr*CACHE);alloc(st_.cache_n,tr);
      alloc(st_.queue,tr);alloc(st_.next_queue,tr);alloc(st_.queue_n,1);alloc(st_.next_n,1);alloc(st_.cursor,1);
      int device;check(cudaGetDevice(&device));cudaDeviceProp prop;check(cudaGetDeviceProperties(&prop,device));blocks_=2*prop.multiProcessorCount;
      alloc(st_.lo,roots); alloc(st_.hi,roots); alloc(st_.flo,roots); alloc(st_.fhi,roots);
      alloc(st_.roots,roots); alloc(st_.previous,roots); alloc(st_.output,tr*modes);
      alloc(st_.count,tr); alloc(st_.prev_count,tr); alloc(st_.status,tr); alloc(st_.active,tr); alloc(st_.level,tr);
      alloc(st_.flags,tr); alloc(st_.stats,NSTATS);
      alloc(st_.full_roots,roots);alloc(st_.guard_roots,roots);alloc(st_.verify,tr);alloc(st_.guard_count,tr);
      alloc(st_.full_count,tr);
    } catch (...) { close(); throw; }
  }
  Solver(const Solver &)=delete;
  ~Solver() { close(); }
  void close() {
    int current=device_;cudaGetDevice(&current);cudaSetDevice(device_);
#define QE_FREE(name) if (st_.name) { cudaFree(st_.name); st_.name=nullptr; }
    QE_FREE(d) QE_FREE(a) QE_FREE(b) QE_FREE(r) QE_FREE(period) QE_FREE(bounds) QE_FREE(ia) QE_FREE(ib)
    QE_FREE(tx) QE_FREE(tn) QE_FREE(nt) QE_FREE(seeds) QE_FREE(cache_x) QE_FREE(cache_y) QE_FREE(cache_n)
    QE_FREE(queue) QE_FREE(next_queue) QE_FREE(queue_n) QE_FREE(next_n) QE_FREE(cursor)
    QE_FREE(lo) QE_FREE(hi) QE_FREE(flo) QE_FREE(fhi) QE_FREE(roots) QE_FREE(previous) QE_FREE(output)
    QE_FREE(count) QE_FREE(prev_count) QE_FREE(status) QE_FREE(active) QE_FREE(level) QE_FREE(flags) QE_FREE(stats)
    QE_FREE(full_roots) QE_FREE(guard_roots) QE_FREE(verify) QE_FREE(guard_count) QE_FREE(full_count)
#undef QE_FREE
    cudaSetDevice(current);
  }
  py::tuple run(py::array_t<double,py::array::c_style|py::array::forcecast> d,
                py::array_t<double,py::array::c_style|py::array::forcecast> a,
                py::array_t<double,py::array::c_style|py::array::forcecast> b,
                py::array_t<double,py::array::c_style|py::array::forcecast> r,
                py::array_t<double,py::array::c_style|py::array::forcecast> periods,
                py::array_t<double,py::array::c_style|py::array::forcecast> bounds,double tol,double f_accept,int wave,bool automatic) {
    if (!st_.d) throw std::runtime_error("QE solver is closed");
    check(cudaSetDevice(device_));
    if (d.ndim()!=2 || d.shape(0)<1 || d.shape(0)>B_ || d.shape(1)!=L_)
      throw py::value_error("QE model shape exceeds allocated geometry");
    for (auto *v:{&a,&b,&r})
      if (v->ndim()!=2 || v->shape(0)!=d.shape(0) || v->shape(1)!=L_)
        throw py::value_error("QE model shapes must agree");
    if (periods.ndim()!=1 || periods.size()!=nf_ || bounds.ndim()!=2 || bounds.shape(0)!=d.shape(0) || bounds.shape(1)!=2 || !(tol>0))
      throw py::value_error("QE invalid periods, range or tolerance");
    const int B=static_cast<int>(d.shape(0)), traces=B*nf_;
    py::array_t<double> out({B,modes_,nf_});
    py::array_t<unsigned> flags({B,nf_});
    py::array_t<int> levels({B,nf_}), counts({B,nf_});
    unsigned long long counters[NSTATS]{};
    {
      py::gil_scoped_release release;
      const size_t n=static_cast<size_t>(B)*L_;
      upload(st_.d,d.data(),n); upload(st_.a,a.data(),n); upload(st_.b,b.data(),n); upload(st_.r,r.data(),n);
      upload(st_.period,periods.data(),nf_);upload(st_.bounds,bounds.data(),2*B);st_.wave=wave;st_.automatic=automatic;st_.guarded=0;
      prepare_models<<<B,256>>>(st_,B,L_);
      check(cudaMemset(st_.stats,0,NSTATS*sizeof(unsigned long long)));
      initialize<<<(traces+255)/256,256>>>(st_,traces,K_,L_,nf_,modes_);
      const size_t sm=(2*CAP+6*L_)*sizeof(double);
      const double delta[4]={0.5,0.25,0.125,0.0625};
      auto stages=[&](int begin){for (int lev=begin; lev<6; ++lev) {
        check(cudaMemset(st_.cursor,0,sizeof(int)));check(cudaMemset(st_.next_n,0,sizeof(int)));
        const int blocks=MD_ACTIVITY_QUEUE?std::min(traces,blocks_):traces;
        if(lev<4)search<true><<<blocks,256,sm>>>(st_,L_,nf_,K_,delta[lev],tol,lev);
        else search<false><<<blocks,256,sm>>>(st_,L_,nf_,K_,0.,tol,lev);
        refine<<<(traces*K_+255)/256,256>>>(st_,traces,L_,nf_,K_,tol);
        // Never compare an adaptive set directly with the first dense set:
        // the fallback must agree with its own doubled-resolution scan.
        compare<<<(traces+255)/256,256>>>(st_,traces,L_,nf_,K_,modes_,tol,f_accept,lev,lev<3 || lev==4,lev==5);
        check(cudaGetLastError());std::swap(st_.queue,st_.next_queue);std::swap(st_.queue_n,st_.next_n);
      }};
      stages(0);
      check(cudaMemset(st_.queue_n,0,sizeof(int)));
      prepare_verification<<<(traces+255)/256,256>>>(st_,traces,K_);
      st_.guarded=1;stages(0);
      check(cudaMemset(st_.queue_n,0,sizeof(int)));
      reconcile<<<(traces+255)/256,256>>>(st_,traces,K_,tol);
      st_.guarded=0;stages(4);
      final_statistics<<<(traces+255)/256,256>>>(st_,traces);
      check(cudaDeviceSynchronize());
      download(out.mutable_data(),st_.output,static_cast<size_t>(B)*modes_*nf_);
      download(flags.mutable_data(),st_.flags,traces); download(levels.mutable_data(),st_.level,traces);
      download(counts.mutable_data(),st_.count,traces); download(counters,st_.stats,NSTATS);
    }
    py::dict stats;
    for (int i=0;i<NSTATS;++i) stats[stat_names[i]]=counters[i];
    stats["strategy"]="qe_adaptive";
    stats["capacity"]=CAP;stats["density_table_knots"]=129;stats["cache_capacity"]=CACHE;
    return py::make_tuple(out,flags,stats,levels,counts);
  }
};
} // namespace qe

void bind_qe(py::module_ &m) {
  m.def("available",[](int device){int n=0;return cudaGetDeviceCount(&n)==cudaSuccess&&device>=0&&device<n;},py::arg("device")=0);
  m.def("set_device",[](int device){qe::check(cudaSetDevice(device));});
  m.def("memory_info",[](){size_t free,total;qe::check(cudaMemGetInfo(&free,&total));return py::make_tuple(free,total);});
  m.def("qe_kernel_report",[](int L){cudaFuncAttributes a;qe::check(cudaFuncGetAttributes(&a,qe::search<true>));py::dict d;d["registers_per_thread"]=a.numRegs;d["static_shared_bytes"]=a.sharedSizeBytes;d["dynamic_shared_bytes"]=(2*qe::CAP+6*L)*sizeof(double);d["cache_points_per_trace"]=qe::CACHE;d["table_capacity_per_model"]=qe::TABLE;return d;});
  py::class_<qe::Solver>(m,"QeSolver")
    .def(py::init<int,int,int,int>(),py::arg("B"),py::arg("L"),py::arg("nf"),py::arg("nmodes"))
    .def("run",&qe::Solver::run,py::arg("thickness"),py::arg("vp"),py::arg("vs"),py::arg("rho"),
         py::arg("periods"),py::arg("bounds"),py::arg("c_tol_rel")=1e-6,py::arg("f_accept")=1.0,py::arg("wave")=0,py::arg("automatic")=false)
    .def("close",&qe::Solver::close);
}
