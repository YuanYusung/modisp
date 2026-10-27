// BSD-3-Clause. Complete FP64 multimode CPU backend, no CUDA dependency.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <exception>
#include <map>
#include <mutex>
#include <thread>
#include <vector>
#include "common/physics.hpp"

namespace py=pybind11;
using Array=py::array_t<double,py::array::c_style|py::array::forcecast>;
constexpr int CAP=2048;
using Point=std::pair<double,double>;
using Counters=std::array<unsigned long long,16>;
const char *counter_names[]={"evaluations","extrema_proposals","brackets","adaptive_converged","dense_converged","unconverged","capacity_hits","near_cutoff","nonfinite","refinement_failed","searches","rejected_f","analytical_empty","range_empty","verification_traces","family_conflicts"};
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
};
struct Scratch {
  std::vector<Point> points,extra,cache;
  std::vector<double> roots,previous;
  Counters stats{};
  Scratch(){points.reserve(CAP);extra.reserve(CAP);cache.reserve(CAP);}
};

double evaluate(const md::Medium &m,double omega,double c,Scratch &w){++w.stats[0];return md::evaluate(m,omega,c);}
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
      if(static_cast<int>(roots.size())<K){int ev=0;double r=md::refine_root(m,omega,a,b,fa,fb,tol,ev);w.stats[0]+=ev;if(!std::isfinite(r)){++w.stats[9];return false;}roots.push_back(r);}
    }
  }
  if(last&&p.size()>1&&p.back().second==0.){
    if(p[p.size()-2].second==0.)return false;
    if(static_cast<int>(roots.size())<K)roots.push_back(p.back().first);
  }
  return true;
}

bool search_adaptive(const Model &model,double period,double lo,double hi,double delta,int K,double tol,Scratch &w,unsigned &flags,bool shifted=false){
  const auto &m=model.view;const double freq=1./period,omega=md::two_pi<double>()/period;
  const double nlo=freq*md::density_base(m,lo),nhi=freq*md::density_base(m,hi),step=delta*.25;
  const double count=std::ceil(std::fmax(0.,nhi-nlo)/step);
  w.roots.clear();w.points.clear();
  if(!std::isfinite(count)||count>CAP-64-6*m.L-1){flags|=8;++w.stats[6];return false;}
  ++w.stats[10];
  auto point=[&](double c){
    if(c<lo||c>hi||!std::isfinite(c))return;
    auto it=std::lower_bound(w.cache.begin(),w.cache.end(),Point(c,-INFINITY));
    const double f=(it!=w.cache.end()&&it->first==c)?it->second:evaluate(m,omega,c,w);
    w.points.emplace_back(c,f);
  };
  for(int j=0;j<64;++j)point(lo+(hi-lo)*(shifted&&j>0&&j<63?(j-.5)/62.:j/63.));
  for(int j=0;j<m.L;++j)for(double v:{m.vp[j],m.vs[j]})for(int side=-1;side<=1;++side)point(v*(1.+side*1e-7));
  const int top=m.vs[0]==0.?1:0;
  point(m.wave ? m.vs[top] : md::rayleigh_reference(m.vp[top],m.vs[top]));
  if(!m.wave&&top)point(md::scholte_reference(m.vp[0],m.rho[0],m.vp[1],m.vs[1],m.rho[1]));
  for(int j=1;j<=static_cast<int>(count);++j){
    const double target=(nlo+(j-(shifted?.5:0.))*step)/freq;
    if(target<model.tn.back()){
      const double c=md::inverse_density(m,model.tx.data(),model.tn.data(),model.tx.size(),target);
      if(!std::isfinite(c)||std::fabs(freq*(md::density_base(m,c)-target))>delta*1e-3){flags|=512;return false;}
      point(c);
    }
  }
  unique_points(w.points);
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
    std::sort(w.extra.begin(),w.extra.end());
    const auto old=w.points.size();w.points.insert(w.points.end(),w.extra.begin(),w.extra.end());
    std::inplace_merge(w.points.begin(),w.points.begin()+old,w.points.end());
    w.points.erase(std::unique(w.points.begin(),w.points.end(),[](const Point&a,const Point&b){return a.first==b.first;}),w.points.end());
  }
  w.cache=w.points;
  if(!w.points.empty()&&md::ambiguous_sh_cutoff(m,w.points.back().first,w.points.back().second)){flags|=16;return false;}
  if(!collect_roots(m,omega,w,K,tol)){flags|=32;++w.stats[8];return false;}
  return true;
}

bool search_dense(const Model &model,double period,double lo,double hi,int factor,int K,double tol,Scratch &w,unsigned &flags){
  const auto &m=model.view;double need=md::dense_workload(m,period,lo,hi);
  w.roots.clear();++w.stats[10];
  if(!std::isfinite(need)||need>524288.){flags|=8;++w.stats[6];return false;}
  int cells=32768;while(cells<need)cells*=2;cells*=factor;
  const double omega=md::two_pi<double>()/period;
  for(int start=0;start<cells;start+=CAP-1){
    w.points.clear();int n=std::min(CAP,cells-start+1);
    for(int j=0;j<n;++j){double c=lo+(hi-lo)*(start+j)/cells;w.points.emplace_back(c,evaluate(m,omega,c,w));}
    if(start+n==cells+1&&md::ambiguous_sh_cutoff(m,w.points.back().first,w.points.back().second)){flags|=16;return false;}
    if(!collect_roots(m,omega,w,K,tol,start==0,start+n==cells+1)){flags|=32;++w.stats[8];return false;}
    // Both dense levels retain their full-window spacing. Higher roots cannot
    // change an already collected ascending K-root prefix; root_count is
    // explicitly capped at K. No test of the requested prefix is removed.
    if(static_cast<int>(w.roots.size())>=K)break;
  }
  return true;
}

class CpuSolver {
  int threads_;Pool pool;std::vector<Scratch> work;
 public:
  explicit CpuSolver(int threads):threads_(threads),pool(threads),work(threads){if(threads<1||threads>256)throw py::value_error("threads must be 1..256");}
  py::tuple run(Array h,Array vp,Array vs,Array rho,Array periods,Array bounds,int modes,int wave,bool automatic,double tol,double accept){
    if(h.ndim()!=2||h.shape(0)<1||h.shape(1)<1)throw py::value_error("model must be [B,L]");
    int B=h.shape(0),L=h.shape(1),nf=periods.size(),K=modes+2;
    for(auto *a:{&vp,&vs,&rho})if(a->ndim()!=2||a->shape(0)!=B||a->shape(1)!=L)throw py::value_error("model shapes differ");
    if(periods.ndim()!=1||nf<1||bounds.ndim()!=2||bounds.shape(0)!=B||bounds.shape(1)!=2||modes<1||L>256)throw py::value_error("invalid CPU geometry");
    py::array_t<double> output({B,modes,nf});py::array_t<unsigned> flags({B,nf});
    py::array_t<int> levels({B,nf}),counts({B,nf});
    std::vector<Model> models(B);
    {
      py::gil_scoped_release release;
      for(auto &w:work)w.stats.fill(0);
      pool.run(B,[&](size_t i,int){
        auto &q=models[i];q.ia.resize(L);q.ib.resize(L);const size_t off=i*L;
        for(int j=0;j<L;++j){q.ia[j]=1./(vp.data()[off+j]*vp.data()[off+j]);q.ib[j]=vs.data()[off+j]==0.?0.:1./(vs.data()[off+j]*vs.data()[off+j]);}
        q.view={h.data()+off,vp.data()+off,vs.data()+off,rho.data()+off,q.ia.data(),q.ib.data(),L,wave};
        double lo=bounds.data()[2*i],hi=bounds.data()[2*i+1];
        for(int j=0;j<129;++j)q.tx.push_back(lo+(hi-lo)*j/128.);
        for(int j=0;j<L;++j)for(double c:{q.view.vp[j],q.view.vs[j]})if(c>lo&&c<hi)q.tx.push_back(c);
        std::sort(q.tx.begin(),q.tx.end());q.tx.erase(std::unique(q.tx.begin(),q.tx.end()),q.tx.end());
        for(double c:q.tx)q.tn.push_back(md::density_base(q.view,c));
      });
      pool.run(static_cast<size_t>(B)*nf,[&](size_t t,int id){
        auto &w=work[id];const int i=t/nf,f=t%nf;auto &model=models[i];
        unsigned word=0;bool ready=false,previous_ok=false;int level=-1;
        w.cache.clear();w.previous.clear();
        double lo=bounds.data()[2*i],hi=bounds.data()[2*i+1],period=periods.data()[f];
        if(hi>lo&&model.view.wave&&!md::sh_has_guide(model.view)){for(int m=0;m<modes;++m)output.mutable_data()[(static_cast<size_t>(i)*modes+m)*nf+f]=NAN;flags.mutable_data()[t]=0;levels.mutable_data()[t]=-3;counts.mutable_data()[t]=0;++w.stats[12];return;}
        if(hi<=lo){for(int m=0;m<modes;++m)output.mutable_data()[(static_cast<size_t>(i)*modes+m)*nf+f]=NAN;flags.mutable_data()[t]=0;levels.mutable_data()[t]=-4;counts.mutable_data()[t]=0;++w.stats[13];return;}
        bool shifted_companion=false;
        auto search=[&](bool guarded,int begin=0){
        previous_ok=false;w.previous.clear();
        std::vector<double> coarse;bool coarse_ok=false;
        for(int lev=begin;lev<6;++lev){
          const double low=automatic&&lev<2?fmin((lev==0?4.:2.)*lo,(lo+hi)*.5):lo;
          double adaptive_hi=hi;
          // A valid Kth bracket bounds the ascending prefix. Retain a margin
          // much wider than refinement error, and never narrow after a failed
          // sample/capacity check. Dense fallbacks keep the original window.
          if(guarded&&lev>0&&lev<4&&previous_ok&&static_cast<int>(w.previous.size())==K&&!(word&(8|32|64|512)))adaptive_hi=std::fmin(hi,w.previous.back()*(1.+4.*tol));
          bool ok=lev<4?search_adaptive(model,period,low,adaptive_hi,.5/(1<<lev),K,tol,w,word,guarded&&shifted_companion):search_dense(model,period,lo,hi,lev==4?1:2,K,tol,w,word);
          bool same=ok&&previous_ok&&w.roots.size()==w.previous.size();
          for(size_t m=0;m<w.roots.size()&&same;++m){same=md::roots_agree(w.roots[m],w.previous[m],tol);if(m&&!(w.roots[m]>w.roots[m-1]))same=false;}
          if(lev==0&&ok){coarse=w.roots;coarse_ok=true;}
          if(lev==3&&(word&(8|32|64|512)))same=false;
          if(same&&(lev==3||lev==5)&&coarse_ok&&!md::contains_witnesses(w.roots.data(),w.roots.size(),coarse.data(),coarse.size(),K,tol)){
            same=false;if(!(word&1024))++w.stats[15];word|=1024;
          }
          // Nested density points can share a missed pair. Always visit all
          // four prescribed adaptive densities before accepting agreement.
          if(same&&(lev==3||lev==5))return lev;
          previous_ok=ok;w.previous=w.roots;
        }return -1;};
        level=search(false);
        if(level==3&&!(word&(8|32|64|512))){
          ++w.stats[14];auto full=w.roots;
          shifted_companion=static_cast<int>(full.size())<K;
          int check_level=search(true);bool same=check_level>=0&&full.size()==w.roots.size();
          for(size_t m=0;m<full.size()&&same;++m)same=md::roots_agree(full[m],w.roots[m],tol);
          if(same)level=std::max(level,check_level);
          else{
            if(!(word&1024))++w.stats[15];word|=1024;
            if(check_level<0)level=-1;
            else{auto guarded=w.roots;level=search(false,4);
              if(level>=0&&(!md::contains_witnesses(w.roots.data(),w.roots.size(),full.data(),full.size(),K,tol)||!md::contains_witnesses(w.roots.data(),w.roots.size(),guarded.data(),guarded.size(),K,tol)))level=-1;
            }
          }
        }
        ready=level>=0;
        if(ready)++w.stats[level==3?3:4];
        if(!ready){word|=4;++w.stats[5];}
        bool cutoff=false;
        for(int m=0;m<modes;++m){
          double root=ready&&m<static_cast<int>(w.roots.size())?w.roots[m]:NAN;
          if(std::isfinite(root)&&root>=.99*model.view.vs[L-1])cutoff=true;
          if(std::isfinite(root)&&accept<1.){double value=evaluate(model.view,md::two_pi<double>()/period,root,w);if(!std::isfinite(value)||std::fabs(value)>accept){root=NAN;word|=128;++w.stats[11];}}
          output.mutable_data()[(static_cast<size_t>(i)*modes+m)*nf+f]=root;
        }
        if(cutoff)word|=16;if(word&16)++w.stats[7];flags.mutable_data()[t]=word;levels.mutable_data()[t]=level;counts.mutable_data()[t]=ready?w.roots.size():-1;
      });
    }
    py::dict stats;for(int j=0;j<16;++j){unsigned long long v=0;for(auto&w:work)v+=w.stats[j];stats[counter_names[j]]=v;}
    stats["threads"]=threads_;stats["strategy"]="qe_adaptive";
    return py::make_tuple(output,flags,stats,levels,counts);
  }
};

PYBIND11_MODULE(_cpu,m){
  m.attr("ABI_VERSION")=1;m.attr("VERSION")="0.2.0";
  py::class_<CpuSolver>(m,"CpuSolver").def(py::init<int>()).def("run",&CpuSolver::run);
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
