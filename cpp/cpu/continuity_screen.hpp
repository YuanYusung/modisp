// Host-side screening of completed CPU or CUDA phases in caller grid order.
#pragma once
#include <cstdint>
#include <cstring>
#include <numeric>

template<class T>
py::tuple continuity_screen(py::array_t<double, py::array::c_style> periods,
                            py::array_t<T, py::array::c_style> phase,
                            py::object prediction, bool bilateral) {
  if (periods.ndim()!=1 || phase.ndim()!=3 || phase.shape(2)!=periods.size())
    throw py::value_error("continuity screening requires periods[T] and phases[B,K,T]");
  const auto B=phase.shape(0), K=phase.shape(1), Tn=periods.size();
  py::array_t<std::uint32_t, py::array::c_style> words;
  const std::uint32_t *pred=nullptr;
  if (!prediction.is_none()) {
    words=decltype(words)::ensure(prediction);
    if (!words || words.ndim()!=2 || words.shape(0)!=B || words.shape(1)!=Tn)
      throw py::value_error("prediction flags must have shape [B,T]");
    pred=words.data();
  }
  py::array_t<std::uint8_t> result({B,Tn});
  auto *out=result.mutable_data();
  const auto *p=periods.data();
  const auto *v=phase.data();
  bool suspect=false;
  {
    py::gil_scoped_release release;
    std::vector<py::ssize_t> order;
    if (bilateral && Tn>2 && !std::is_sorted(p,p+Tn)) {
      order.resize(Tn);
      std::iota(order.begin(),order.end(),0);
      std::sort(order.begin(),order.end(),[&](auto i,auto j){return p[i]<p[j];});
    }
    auto original=[&](auto rank){return order.empty()?rank:order[rank];};
    for (py::ssize_t j=0;j<Tn;++j) {
      const auto f=original(j);
      const auto l=original(j>0?j-1:j), r=original(j+1<Tn?j+1:j);
      const bool interior=bilateral && j>0 && j+1<Tn && md::continuation_log_step(p[r],p[l])<=.25;
      for (py::ssize_t b=0;b<B;++b) {
        std::uint8_t flag=bilateral?8:0; // NO_BILATERAL until a mode has two finite neighbors.
        if (interior || (bilateral && std::is_same_v<T,float> && K>1)) {
          for (py::ssize_t k=0;k<K;++k) {
            const T *mode=v+(b*K+k)*Tn;
            if (interior && std::isfinite(mode[l]) && std::isfinite(mode[r])) {
              flag &= ~8u;
              if (!std::isfinite(mode[f]) || mode[f]<.95*std::min(mode[l],mode[r])
                                          || mode[f]>1.05*std::max(mode[l],mode[r])) flag |= 16;
            }
            if constexpr (std::is_same_v<T,float>) {
              if (k && std::isfinite(mode[f]) && std::isfinite(mode[f-Tn])) {
                std::uint32_t a,c;
                std::memcpy(&a,mode+f,sizeof(a));
                std::memcpy(&c,mode+f-Tn,sizeof(c));
                if (std::max(a,c)-std::min(a,c)<=16) flag |= 32;
              }
            }
          }
        }
        if (pred && (pred[b*Tn+f]&md::P_REVIEW)) flag |= 64;
        out[b*Tn+f]=flag;
        suspect |= (flag & (16|32|64))!=0;
      }
    }
  }
  return py::make_tuple(result,suspect);
}
