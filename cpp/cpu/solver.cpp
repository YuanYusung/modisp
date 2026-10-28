// BSD-3-Clause. Complete FP64 multimode CPU backend, no CUDA dependency.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <exception>
#include <map>
#include <mutex>
#include <thread>
#include <vector>
#include "common/accuracy.hpp"
#include "common/physics.hpp"

namespace py=pybind11;
using Array=py::array_t<double,py::array::c_style|py::array::forcecast>;
constexpr int CAP=2048;
using Point=std::pair<double,double>;
using Counters=std::array<unsigned long long,21>;
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
const char *counter_names[]={"evaluations","extrema_proposals","brackets","adaptive_converged","dense_converged","unconverged","capacity_hits","near_cutoff","nonfinite","refinement_failed","searches","rejected_f","analytical_empty","range_empty","verification_traces","family_conflicts","early_accepts","verification_skipped","dense_coarse_accepted","dense_fine_escalated","trough_risks"};
extern "C" void md_dunkin_grid(const double*,const double*,const double*,const double*,int,int,const double*,int,double,double,int,double*);
extern "C" void md_dunkin_grid_f32(const double*,const double*,const double*,const double*,int,int,const double*,int,double,double,int,double*);
extern "C" double md_dunkin_point(const double*,const double*,const double*,const double*,int,double,double,int);

class Pool {
  std::vector<std::thread> workers;
  std::mutex mutex;
  std::condition_variable ready,done;
  std::function<void(size_t,int)> job;
  std::atomic<size_t> next{0};
  size_t count=0,generation=0;int finished=0;bool stop=false;
  std::exception_ptr failure;
 public:
  explicit Pool(int n){if(n<1||n>256)throw std::invalid_argument("threads must be 1..256");try{for(int id=0;id<n;++id)workers.emplace_back([this,id]{
    size_t seen=0;
    for(;;){
      {std::unique_lock<std::mutex> lock(mutex);ready.wait(lock,[&]{return stop||generation!=seen;});if(stop)return;seen=generation;}
      try{for(size_t i;(i=next.fetch_add(1))<count;)job(i,id);}
      catch(...){std::lock_guard<std::mutex> lock(mutex);if(!failure)failure=std::current_exception();next=count;}
      {std::lock_guard<std::mutex> lock(mutex);if(++finished==static_cast<int>(workers.size()))done.notify_one();}
    }
  });}catch(...){ {std::lock_guard<std::mutex> lock(mutex);stop=true;}ready.notify_all();for(auto&t:workers)t.join();throw;}}
  ~Pool(){ {std::lock_guard<std::mutex> lock(mutex);stop=true;}ready.notify_all();for(auto &t:workers)t.join(); }
  void run(size_t n,std::function<void(size_t,int)> f){
    {std::lock_guard<std::mutex> lock(mutex);job=std::move(f);count=n;next=0;finished=0;failure=nullptr;++generation;}
    ready.notify_all();std::unique_lock<std::mutex> lock(mutex);done.wait(lock,[&]{return finished==static_cast<int>(workers.size());});job=nullptr;if(failure)std::rethrow_exception(failure);
  }
};

struct Model {
  md::Medium view;
  std::vector<double> ia,ib,tx,tn;
  double seed,water_seed;
};
struct BracketRoot { double lo,hi,root; };
struct Scratch {
  std::vector<BracketRoot> refined;
  std::vector<double> dense_values;
  int dense_cells=0;
  unsigned long long dense_cache_hits=0;
  unsigned long long refinement_cache_hits=0;
  int refinement_cache_limit=0;
  std::vector<Point> points,extra,cache,merged;
  std::vector<double> roots,previous,coarse,full,guarded,ka,kb,gammk;
  md::Medium medium{};
  bool profile=false;
  double sample_s=0.,inverse_s=0.,sort_s=0.,refine_s=0.,adaptive_s=0.,dense_s=0.;
  unsigned long long cache_hits=0,cache_misses=0;
  double sample_gap=0.;int risk=0;bool escalated=false;
  md::AccuracyPolicy policy{};
  Counters stats{};
  Scratch(){points.reserve(CAP);extra.reserve(CAP);cache.reserve(CAP);merged.reserve(CAP);}
};

double evaluate(const md::Medium &m,double omega,double c,Scratch &w){
  ++w.stats[0];if(!w.profile)return md::evaluate(m,omega,c);
  const auto t=Clock::now();double y=md::evaluate(m,omega,c);w.sample_s+=seconds(t);return y;
}
void unique_points(std::vector<Point> &p){
  std::sort(p.begin(),p.end(),[](const Point&a,const Point&b){return a.first<b.first;});
  p.erase(std::unique(p.begin(),p.end(),[](const Point&a,const Point&b){return a.first==b.first;}),p.end());
}

bool collect_roots(const md::Medium &m,double omega,Scratch &w,int K,double tol,bool first=true,bool last=true){
  auto &p=w.points;auto &roots=w.roots;
  for(size_t i=0;i+1<p.size();++i){
    double a=p[i].first,b=p[i+1].first,fa=p[i].second,fb=p[i+1].second;
    if(!std::isfinite(fa)||!std::isfinite(fb))return false;
    if(fa==0.){
      if(fb==0.||(i>0&&p[i-1].second==0.))return false;
      if((i>0&&std::signbit(p[i-1].second)!=std::signbit(fb))||(first&&i==0)){
        if(static_cast<int>(roots.size())<K)roots.push_back(a);++w.stats[2];
      }else return false;
    }else if(fb!=0.&&std::signbit(fa)!=std::signbit(fb)){
      ++w.stats[2];
      if(static_cast<int>(roots.size())<K){
        double r=NAN;
        for(const auto &saved:w.refined)if(saved.lo==a&&saved.hi==b){r=saved.root;++w.refinement_cache_hits;break;}
        if(!std::isfinite(r)){
          int ev=0;const auto t=w.profile?Clock::now():Clock::time_point{};
          r=md::refine_root(m,omega,a,b,fa,fb,tol,ev);
          if(w.profile)w.refine_s+=seconds(t);w.stats[0]+=ev;
          if(!std::isfinite(r)){++w.stats[9];return false;}
          if(static_cast<int>(w.refined.size())<w.refinement_cache_limit)w.refined.push_back({a,b,r});
        }
        roots.push_back(r);
      }
    }
  }
  if(last&&p.size()>1&&p.back().second==0.){
    if(p[p.size()-2].second==0.)return false;
    if(static_cast<int>(roots.size())<K)roots.push_back(p.back().first);
  }
  return true;
}

bool search_adaptive(const Model &model,double period,double lo,double hi,double delta,int K,double tol,Scratch &w,unsigned &flags,bool shifted=false){
  const auto &m=w.medium;const double freq=1./period,omega=md::two_pi<double>()/period;
  const double nlo=freq*md::density_base(m,lo),nhi=freq*md::density_base(m,hi),step=delta*.25;
  const double count=std::ceil(std::fmax(0.,nhi-nlo)/step);
  w.roots.clear();w.points.clear();
  if(!std::isfinite(count)||count>CAP-64-6*m.L-1){flags|=8;++w.stats[6];return false;}
  ++w.stats[10];
  auto point=[&](double c){
    if(c<lo||c>hi||!std::isfinite(c))return;
    auto it=std::lower_bound(w.cache.begin(),w.cache.end(),Point(c,-INFINITY));
    const bool hit=it!=w.cache.end()&&it->first==c;
    if(w.profile){if(hit)++w.cache_hits;else ++w.cache_misses;}
    const double f=hit?it->second:evaluate(m,omega,c,w);
    w.points.emplace_back(c,f);
  };
  for(int j=0;j<64;++j)point(lo+(hi-lo)*(shifted&&j>0&&j<63?(j-.5)/62.:j/63.));
  for(int j=0;j<m.L;++j)for(double v:{m.vp[j],m.vs[j]})for(int side=-1;side<=1;++side)point(v*(1.+side*1e-7));
  point(model.seed);
  if(!m.wave&&m.vs[0]==0.)point(model.water_seed);
  for(int j=1;j<=static_cast<int>(count);++j){
    const double target=(nlo+(j-(shifted?.5:0.))*step)/freq;
    if(target<model.tn.back()){
      const auto t=w.profile?Clock::now():Clock::time_point{};
      const double c=md::inverse_density(m,model.tx.data(),model.tn.data(),model.tx.size(),target);
      if(w.profile)w.inverse_s+=seconds(t);
      if(!std::isfinite(c)||std::fabs(freq*(md::density_base(m,c)-target))>delta*1e-3){flags|=512;return false;}
      point(c);
    }
  }
  auto sort_start=w.profile?Clock::now():Clock::time_point{};
  unique_points(w.points);if(w.profile)w.sort_s+=seconds(sort_start);
  if(w.points.size()>CAP){flags|=8;++w.stats[6];return false;}
  for(int round=0;round<3;++round){
    w.extra.clear();
    for(size_t i=0;i+2<w.points.size();++i){
      double x1=w.points[i].first,x2=w.points[i+1].first,x3=w.points[i+2].first;
      double f1=w.points[i].second,f2=w.points[i+1].second,f3=w.points[i+2].second;
      if(!std::isfinite(f1)||!std::isfinite(f2)||!std::isfinite(f3)){flags|=32;++w.stats[8];return false;}
      double scale=std::fmax(std::fabs(f1),std::fmax(std::fabs(f2),std::fabs(f3)));
      if(scale==0.){flags|=32;++w.stats[8];return false;}
      double s1=(f2/scale-f1/scale)/(x2-x1),s2=(f3/scale-f2/scale)/(x3-x2);
      double a=(s2-s1)/(x3-x1),x=NAN;
      if(std::isfinite(a)&&std::fabs(a)>64.*2.220446049250313e-16/((x3-x1)*(x3-x1)))x=(x1+x2)*.5-s1/(2.*a);
      if(!std::isfinite(x)&&std::fabs(f2)<.05)x=(x1+x3)*.5;
      double gap=16.*2.220446049250313e-16*std::fmax(1.,std::fabs(x2));
      if(!(x>x1+gap&&x<x3-gap)||std::fabs(x-x2)<=gap)continue;
      ++w.stats[1];w.extra.emplace_back(x,evaluate(m,omega,x,w));
    }
    if(w.extra.empty())break;
    if(w.points.size()+w.extra.size()>CAP){flags|=8;++w.stats[6];return false;}
    sort_start=w.profile?Clock::now():Clock::time_point{};
    std::sort(w.extra.begin(),w.extra.end());
    w.merged.clear();
    std::merge(w.points.begin(),w.points.end(),w.extra.begin(),w.extra.end(),std::back_inserter(w.merged));
    w.points.swap(w.merged);
    w.points.erase(std::unique(w.points.begin(),w.points.end(),[](const Point&a,const Point&b){return a.first==b.first;}),w.points.end());
    if(w.profile)w.sort_s+=seconds(sort_start);
  }
  if(w.policy.reuse_refinement){
    // Bounded union across density levels/families. Only exact FP64 velocity
    // keys are reused, scoped to this model-period trace.
    w.extra.clear();size_t i=0,j=0;
    while((i<w.cache.size()||j<w.points.size())&&w.extra.size()<CAP){
      if(j==w.points.size()||(i<w.cache.size()&&w.cache[i].first<w.points[j].first))w.extra.push_back(w.cache[i++]);
      else if(i==w.cache.size()||w.points[j].first<w.cache[i].first)w.extra.push_back(w.points[j++]);
      else{w.extra.push_back(w.points[j++]);++i;}
    }
    if(i==w.cache.size()&&j==w.points.size())w.cache.swap(w.extra);
    else w.cache=w.points;
  }else w.cache=w.points;
  if(!w.points.empty()&&md::ambiguous_sh_cutoff(m,w.points.back().first,w.points.back().second)){flags|=16;return false;}
  if(!collect_roots(m,omega,w,K,tol)){flags|=32;++w.stats[8];return false;}
  return true;
}

bool search_dense(const Model &model,double period,double lo,double hi,int factor,int K,double tol,Scratch &w,unsigned &flags){
  const auto &m=w.medium;double need=md::dense_workload(m,period,lo,hi);
  w.roots.clear();++w.stats[10];
  if(!std::isfinite(need)||need>524288.){flags|=8;++w.stats[6];return false;}
  int cells=32768;while(cells<need)cells*=2;cells*=factor;
  w.sample_gap=cells>0?(hi-lo)/cells:0.;w.risk=0;
  const double omega=md::two_pi<double>()/period;
  const int tile=w.policy.cpu_dense_tile;
  for(int start=0;start<cells;){
    w.points.clear();int n=std::min(tile,cells-start+1);
    for(int j=0;j<n;++j){
      const int index=start+j;double c=lo+(hi-lo)*index/cells,value;
      const int cached=factor==1?index:index/2;
      const bool reusable=w.policy.reuse_refinement&&w.dense_cells>0
        &&(cells==w.dense_cells||(cells==2*w.dense_cells&&index%2==0))
        &&cached<static_cast<int>(w.dense_values.size())
        &&lo+(hi-lo)*cached/w.dense_cells==c;
      if(reusable){value=w.dense_values[cached];++w.dense_cache_hits;}
      else value=evaluate(m,omega,c,w);
      if(w.policy.reuse_refinement&&factor==1&&index<32769){
        if(w.dense_cells!=cells){w.dense_values.clear();w.dense_cells=cells;}
        if(index==static_cast<int>(w.dense_values.size()))w.dense_values.push_back(value);
      }
      w.points.emplace_back(c,value);
    }
    // An exact zero at an internal tile boundary needs its right neighbour
    // for the sign test. Extend an optimized tile instead of treating that
    // zero as an ambiguous left endpoint on the next tile.
    if(n<CAP&&start+n<cells+1&&w.points.back().second==0.){
      const double c=lo+(hi-lo)*(start+n)/cells;
      w.points.emplace_back(c,evaluate(m,omega,c,w));++n;
    }
    if(start+n==cells+1&&md::ambiguous_sh_cutoff(m,w.points.back().first,w.points.back().second)){flags|=16;return false;}
    if(!collect_roots(m,omega,w,K,tol,start==0,start+n==cells+1)){flags|=32;++w.stats[8];return false;}
    // Both dense levels retain their full-window spacing. Higher roots cannot
    // change an already collected ascending K-root prefix; root_count is
    // explicitly capped at K. No test of the requested prefix is removed.
    if(static_cast<int>(w.roots.size())>=K)break;
    start+=n-1;
  }
  return true;
}


// The logarithmic grid's maximum relative cell width is tied directly to the
// observational close-root threshold. A missed even root pair inside one cell
// is below that scale; every approximate trace remains explicitly marked.
bool search_observation(const Model &model,double period,double lo,double hi,int K,double tol,Scratch &w,unsigned &flags){
  const auto &m=w.medium;const double omega=md::two_pi<double>()/period;
  const double target=w.policy.close_root_rtol*(w.policy.accuracy==md::Accuracy::Balanced?.5:1.);
  const double span=std::log(hi)-std::log(lo),need=std::ceil(span/std::log1p(target));
  flags|=2048;w.roots.clear();++w.stats[10];
  if(!std::isfinite(need)||need>524288.){flags|=8;++w.stats[6];return false;}
  const int cells=std::max(1,static_cast<int>(need));const double step=span/cells,origin=std::log(lo);
  auto coordinate=[&](int i){return i==0?lo:(i==cells?hi:std::exp(origin+step*i));};
  for(int start=0;start<cells;){
    w.points.clear();int end=std::min(cells,start+63);
    for(int j=start;j<=end;++j){double c=coordinate(j);w.points.emplace_back(c,evaluate(m,omega,c,w));}
    if(end<cells&&w.points.back().second==0.){++end;double c=coordinate(end);w.points.emplace_back(c,evaluate(m,omega,c,w));}
    const double left=w.points.front().first,right=w.points.back().first;
    auto add=[&](double c){if(c>left&&c<right)w.points.emplace_back(c,evaluate(m,omega,c,w));};
    const size_t original=w.points.size();
    const int top=m.wave&&m.vs[0]==0.?1:0;
    for(int j=top;j<m.L;++j)for(double v:{m.vp[j],m.vs[j]})for(int side=-1;side<=1;++side)add(v*(1.+side*1e-7));
    add(model.seed);if(m.vs[0]==0.&&!m.wave)add(model.water_seed);
    if(w.points.size()!=original)unique_points(w.points);
    for(size_t j=1;j+1<w.points.size();++j){
      const double a=w.points[j-1].second,b=w.points[j].second,c=w.points[j+1].second;
      if((a<0.)==(b<0.)&&(b<0.)==(c<0.)&&std::fabs(b)<.05&&std::fabs(b)<std::fabs(a)&&std::fabs(b)<std::fabs(c)){flags|=4096;break;}
    }
    if(end==cells&&md::ambiguous_sh_cutoff(m,w.points.back().first,w.points.back().second)){flags|=16;return false;}
    if(!collect_roots(m,omega,w,K,tol,start==0,end==cells)){flags|=32;++w.stats[8];return false;}
    if(static_cast<int>(w.roots.size())>=K)break;
    start=end;
  }
  if(md::has_close_roots(w.roots.data(),w.roots.size(),w.policy.close_root_rtol))flags|=4096;
  ++w.stats[17];return true;
}

class CpuSolver {
  int threads_;Pool pool;std::vector<Scratch> work;
  std::vector<Model> models;md::AccuracyPolicy policy_;
 public:
  CpuSolver(int threads,int accuracy,double period_threshold,double close_root_rtol):threads_(threads),pool(threads),work(threads),
    policy_(md::make_accuracy_policy(accuracy,period_threshold,close_root_rtol)){if(threads<1||threads>256)throw py::value_error("threads must be 1..256");}
  py::tuple run(Array h,Array vp,Array vs,Array rho,Array periods,Array bounds,int modes,int wave,bool automatic,double tol,double accept,bool profile=false){
    if(h.ndim()!=2||h.shape(0)<1||h.shape(1)<1)throw py::value_error("model must be [B,L]");
    int B=h.shape(0),L=h.shape(1),nf=periods.size(),K=modes+2;
    for(auto *a:{&vp,&vs,&rho})if(a->ndim()!=2||a->shape(0)!=B||a->shape(1)!=L)throw py::value_error("model shapes differ");
    if(periods.ndim()!=1||nf<1||bounds.ndim()!=2||bounds.shape(0)!=B||bounds.shape(1)!=2||modes<1||L>256)throw py::value_error("invalid CPU geometry");
    py::array_t<double> output({B,modes,nf});py::array_t<unsigned> flags({B,nf});
    py::array_t<int> levels({B,nf}),counts({B,nf});
    if(models.size()<static_cast<size_t>(B))models.resize(B);
    double prepare_s=0.,solve_s=0.;
    {
      py::gil_scoped_release release;
      for(auto &w:work){w.stats.fill(0);w.profile=profile;w.policy=policy_;w.sample_gap=0.;w.risk=0;w.escalated=false;w.sample_s=w.inverse_s=w.sort_s=w.refine_s=w.adaptive_s=w.dense_s=0.;w.cache_hits=w.cache_misses=w.refinement_cache_hits=w.dense_cache_hits=0;
        if(policy_.reuse_refinement)w.dense_values.reserve(32769);
        w.refinement_cache_limit=policy_.reuse_refinement?16*K:0;
        w.refined.reserve(w.refinement_cache_limit);
        for(auto *v:{&w.roots,&w.previous,&w.coarse,&w.full,&w.guarded})v->reserve(K);
        w.ka.resize(L);w.kb.resize(L);w.gammk.resize(L);}
      auto stage_start=profile?Clock::now():Clock::time_point{};
      pool.run(B,[&](size_t i,int){
        auto &q=models[i];q.ia.resize(L);q.ib.resize(L);const size_t off=i*L;
        q.tx.clear();q.tn.clear();q.tx.reserve(129+2*L);q.tn.reserve(129+2*L);
        for(int j=0;j<L;++j){q.ia[j]=1./(vp.data()[off+j]*vp.data()[off+j]);q.ib[j]=vs.data()[off+j]==0.?0.:1./(vs.data()[off+j]*vs.data()[off+j]);}
        q.view={h.data()+off,vp.data()+off,vs.data()+off,rho.data()+off,q.ia.data(),q.ib.data(),L,wave};
        const int top=q.view.vs[0]==0.?1:0;
        q.seed=wave?q.view.vs[top]:md::rayleigh_reference(q.view.vp[top],q.view.vs[top]);
        q.water_seed=(!wave&&top)?md::scholte_reference(q.view.vp[0],q.view.rho[0],q.view.vp[1],q.view.vs[1],q.view.rho[1]):NAN;
        double lo=bounds.data()[2*i],hi=bounds.data()[2*i+1];
        for(int j=0;j<129;++j)q.tx.push_back(lo+(hi-lo)*j/128.);
        for(int j=0;j<L;++j)for(double c:{q.view.vp[j],q.view.vs[j]})if(c>lo&&c<hi)q.tx.push_back(c);
        std::sort(q.tx.begin(),q.tx.end());q.tx.erase(std::unique(q.tx.begin(),q.tx.end()),q.tx.end());
        for(double c:q.tx)q.tn.push_back(md::density_base(q.view,c));
      });
      if(profile){prepare_s=seconds(stage_start);stage_start=Clock::now();}
      pool.run(static_cast<size_t>(B)*nf,[&](size_t t,int id){
        auto &w=work[id];const int i=t/nf,f=t%nf;auto &model=models[i];
        unsigned word=0;bool ready=false,previous_ok=false;int level=-1;
        w.cache.clear();w.previous.clear();w.refined.clear();w.dense_values.clear();w.dense_cells=0;
        double lo=bounds.data()[2*i],hi=bounds.data()[2*i+1],period=periods.data()[f];
        const double omega=md::two_pi<double>()/period;
        for(int j=0;j<L;++j){w.ka[j]=omega/model.view.vp[j];w.kb[j]=model.view.vs[j]==0.?0.:omega/model.view.vs[j];const double z=model.view.vs[j]/omega;w.gammk[j]=2.*z*z;}
        w.medium=model.view;w.medium.ka=w.ka.data();w.medium.kb=w.kb.data();w.medium.gammk=w.gammk.data();
        if(hi>lo&&model.view.wave&&!md::sh_has_guide(model.view)){for(int m=0;m<modes;++m)output.mutable_data()[(static_cast<size_t>(i)*modes+m)*nf+f]=NAN;flags.mutable_data()[t]=0;levels.mutable_data()[t]=-3;counts.mutable_data()[t]=0;++w.stats[12];return;}
        if(hi<=lo){for(int m=0;m<modes;++m)output.mutable_data()[(static_cast<size_t>(i)*modes+m)*nf+f]=NAN;flags.mutable_data()[t]=0;levels.mutable_data()[t]=-4;counts.mutable_data()[t]=0;++w.stats[13];return;}
        bool shifted_companion=false;
        auto search=[&](bool guarded,int begin=0){
        previous_ok=false;w.previous.clear();w.escalated=false;
        auto &coarse=w.coarse;coarse.clear();bool coarse_ok=false;
        for(int lev=begin;lev<6;++lev){
          const double low=automatic&&lev<2?fmin((lev==0?4.:2.)*lo,(lo+hi)*.5):lo;
          double adaptive_hi=hi;
          // A valid Kth bracket bounds the ascending prefix. Retain a margin
          // much wider than refinement error, and never narrow after a failed
          // sample/capacity check. Dense fallbacks keep the original window.
          if(guarded&&lev>0&&lev<4&&previous_ok&&static_cast<int>(w.previous.size())==K&&!(word&(8|32|64|512)))adaptive_hi=std::fmin(hi,w.previous.back()*(1.+4.*tol));
          const auto t=profile?Clock::now():Clock::time_point{};
          bool ok=lev<4?search_adaptive(model,period,low,adaptive_hi,.5/(1<<lev),K,tol,w,word,guarded&&shifted_companion)
                       :search_dense(model,period,lo,hi,lev==4?1:2,K,tol,w,word);
          if(profile){if(lev<4)w.adaptive_s+=seconds(t);else w.dense_s+=seconds(t);}
          bool same=ok&&previous_ok&&w.roots.size()==w.previous.size();
          for(size_t m=0;m<w.roots.size()&&same;++m){same=md::roots_agree(w.roots[m],w.previous[m],tol);if(m&&!(w.roots[m]>w.roots[m-1]))same=false;}
          if(lev==0&&ok){coarse=w.roots;coarse_ok=true;}
          if(lev==3&&(word&(8|32|64|512)))same=false;
          if(same&&(lev==3||lev==5)&&coarse_ok&&!md::contains_witnesses(w.roots.data(),w.roots.size(),coarse.data(),coarse.size(),K,tol)){
            same=false;if(!(word&1024))++w.stats[15];word|=1024;
          }
          // Every preset retains all densities and the high dense pair.
          if(same&&(lev==3||lev==5))return lev;
          if(lev==4)w.escalated=true;
          previous_ok=ok;w.previous=w.roots;
        }return -1;};
        if(w.policy.accuracy!=md::Accuracy::High){
          const auto begin=profile?Clock::now():Clock::time_point{};
          level=search_observation(model,period,lo,hi,K,tol,w,word)?(w.policy.accuracy==md::Accuracy::Balanced?6:7):-1;
          if(profile)w.dense_s+=seconds(begin);
        }else level=search(false);
        if(level>=1&&level<=3&&!(word&(8|32|64|512))){
          {
            ++w.stats[14];auto &full=w.full;full=w.roots;
            shifted_companion=static_cast<int>(full.size())<K;
            int check_level=search(true);bool same=check_level>=0&&full.size()==w.roots.size();
            for(size_t m=0;m<full.size()&&same;++m)same=md::roots_agree(full[m],w.roots[m],tol);
            if(same)level=std::max(level,check_level);
            else{
              if(!(word&1024))++w.stats[15];word|=1024;
              if(check_level<0)level=-1;
              else{auto &guarded=w.guarded;guarded=w.roots;level=search(false,4);
                if(level>=0&&(!md::contains_witnesses(w.roots.data(),w.roots.size(),full.data(),full.size(),K,tol)||!md::contains_witnesses(w.roots.data(),w.roots.size(),guarded.data(),guarded.size(),K,tol)))level=-1;
              }
            }
          }
        }
        ready=level>=0;
        if(ready)++w.stats[(level>=1&&level<=3)?3:4];
        if(!ready){word|=4;++w.stats[5];}
        bool cutoff=false;
        for(int m=0;m<modes;++m){
          double root=ready&&m<static_cast<int>(w.roots.size())?w.roots[m]:NAN;
          if(std::isfinite(root)&&root>=.99*model.view.vs[L-1])cutoff=true;
          if(std::isfinite(root)&&accept<1.){double value=evaluate(w.medium,omega,root,w);if(!std::isfinite(value)||std::fabs(value)>accept){root=NAN;word|=128;++w.stats[11];}}
          output.mutable_data()[(static_cast<size_t>(i)*modes+m)*nf+f]=root;
        }
        if(cutoff)word|=16;if(word&16)++w.stats[7];flags.mutable_data()[t]=word;levels.mutable_data()[t]=level;counts.mutable_data()[t]=ready?w.roots.size():-1;
      });
      if(profile)solve_s=seconds(stage_start);
    }
    py::dict stats;for(int j=0;j<21;++j){unsigned long long v=0;for(auto&w:work)v+=w.stats[j];stats[counter_names[j]]=v;}
    stats["threads"]=threads_;stats["strategy"]="qe_adaptive";
    stats["accuracy"]=md::accuracy_name(policy_.accuracy);
    stats["period_threshold"]=policy_.period_threshold;
    stats["sampling_capacity"]=CAP;
    stats["dense_tile"]=policy_.accuracy==md::Accuracy::High?policy_.cpu_dense_tile:64;
    unsigned long long reuse=0;for(auto &w:work)reuse+=w.refinement_cache_hits;
    stats["refinement_cache_hits"]=reuse;
    unsigned long long dense_reuse=0;for(auto &w:work)dense_reuse+=w.dense_cache_hits;
    stats["dense_cache_hits"]=dense_reuse;
    if(profile){py::dict timing;timing["model_prepare"]=prepare_s;timing["search_wall"]=solve_s;
      double sample=0.,inverse=0.,sort=0.,refine=0.,adaptive=0.,dense=0.;unsigned long long hits=0,misses=0;
      for(auto &w:work){sample+=w.sample_s;inverse+=w.inverse_s;sort+=w.sort_s;refine+=w.refine_s;adaptive+=w.adaptive_s;dense+=w.dense_s;hits+=w.cache_hits;misses+=w.cache_misses;}
      timing["sample_evaluation_worker"]=sample;timing["density_inverse_worker"]=inverse;timing["sort_merge_worker"]=sort;timing["refinement_worker"]=refine;timing["adaptive_worker"]=adaptive;timing["dense_worker"]=dense;
      stats["timings_s"]=timing;stats["cache_hits"]=hits;stats["cache_misses"]=misses;}
    return py::make_tuple(output,flags,stats,levels,counts);
  }
};

PYBIND11_MODULE(_cpu,m){
  m.attr("ABI_VERSION")=4;m.attr("VERSION")="0.3.0";
  py::class_<CpuSolver>(m,"CpuSolver").def(py::init<int,int,double,double>(),py::arg("threads"),py::arg("accuracy")=0,py::arg("period_threshold")=0.,py::arg("close_root_rtol")=.005).def("run",&CpuSolver::run,py::arg("thickness"),py::arg("vp"),py::arg("vs"),py::arg("rho"),py::arg("periods"),py::arg("bounds"),py::arg("nmodes"),py::arg("wave"),py::arg("automatic"),py::arg("tol"),py::arg("accept"),py::arg("profile")=false);
  m.def("bounds",[](Array h,Array vp,Array vs,Array rho,int wave,py::object low,py::object high){
    int B=h.shape(0),L=h.shape(1);py::array_t<double> result({B,2});
    const bool auto_low=low.is_none(),auto_hi=high.is_none();double given_low=auto_low?0.:low.cast<double>(),given_hi=auto_hi?0.:high.cast<double>();
    py::gil_scoped_release release;
    for(int i=0;i<B;++i){size_t p=static_cast<size_t>(i)*L;md::Medium v{h.data()+p,vp.data()+p,vs.data()+p,rho.data()+p,nullptr,nullptr,L,wave};
      result.mutable_data()[2*i]=auto_low?md::automatic_lower(v):given_low;
      result.mutable_data()[2*i+1]=std::fmin(auto_hi?vs.data()[p+L-1]:given_hi,std::nextafter(vs.data()[p+L-1],0.));
    }
    return result;
  });
  m.def("secular",[](Array h,Array vp,Array vs,Array rho,double period,Array velocity,int wave){
    py::array_t<double> out(velocity.size());md::Medium v{h.data(),vp.data(),vs.data(),rho.data(),nullptr,nullptr,static_cast<int>(h.size()),wave};
    py::gil_scoped_release release;for(py::ssize_t j=0;j<velocity.size();++j)out.mutable_data()[j]=md::evaluate(v,md::two_pi<double>()/period,velocity.data()[j]);return out;
  });
  m.def("md_dunkin_point",[](Array h,Array vp,Array vs,Array rho,int L,double period,double c,int normalize){if(L<1||h.size()!=L||vp.size()!=L||vs.size()!=L||rho.size()!=L||!(period>0)||!(c>0))throw py::value_error("invalid point geometry");return md_dunkin_point(h.data(),vp.data(),vs.data(),rho.data(),L,period,c,normalize);});
  m.def("probe_grid",[](Array h,Array vp,Array vs,Array rho,Array periods,double lo,double hi,int nc,bool normalize,bool fp32){
    int L=h.size();if(h.ndim()!=1||L<1||L>256||vp.size()!=L||vs.size()!=L||rho.size()!=L||periods.ndim()!=1||periods.size()<1||nc<1||hi<lo||!(lo>0)||(fp32&&L>64))throw py::value_error("invalid probe geometry");
    for(int j=0;j<L;++j)if(vs.data()[j]<=0.)throw py::value_error("raw precision/normalization probe supports solid models only");
    py::array_t<double> out({periods.size(),static_cast<py::ssize_t>(nc)});py::gil_scoped_release release;
    auto fn=fp32?md_dunkin_grid_f32:md_dunkin_grid;fn(h.data(),vp.data(),vs.data(),rho.data(),L,periods.size(),periods.data(),nc,lo,hi,normalize,out.mutable_data());return out;
  });
}
