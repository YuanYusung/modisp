// Optional two-batch continuation. Included inside namespace qe.
// Complete anchors provide both predictions; predicted traces never seed others.
using md::P_ANCHOR; using md::P_ACCEPTED; using md::P_FALLBACK; using md::P_BYPASSED;
using md::P_AUDIT; using md::P_AUDIT_FAILED; using md::P_SOURCE_UNCERTAIN;
using md::P_ROOT_COUNT; using md::CONTINUATION_APPROXIMATE;

__global__ void continuation_plan(Device st,int nf,int stride){
  if(blockIdx.x||threadIdx.x)return;
  int left=0;st.anchor_left[0]=st.anchor_right[0]=0;
  while(left<nf-1){
    const int right=md::continuation_next_anchor(st.period,nf,left,stride);
    for(int f=left+1;f<right;++f){st.anchor_left[f]=left;st.anchor_right[f]=right;}
    st.anchor_left[right]=st.anchor_right[right]=right;left=right;
  }
}

__global__ void continuation_hold(Device st,int traces,int nf,bool enabled){
  const int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=traces)return;
  if(!enabled){st.prediction_flags[t]=P_BYPASSED;}
  else if(st.anchor_left[t%nf]==st.anchor_right[t%nf]){
    st.prediction_flags[t]=P_ANCHOR;atomicAdd(&st.stats[CONT_ANCHORS],1ULL);
  }else{st.prediction_flags[t]=0;st.active[t]=0;}
  if(st.active[t])st.queue[atomicAdd(st.queue_n,1)]=t;
}

__device__ bool group_close(Device st,int t,int j,int K,double root){
  const size_t p=static_cast<size_t>(t)*K;
  return (j&&root-st.roots[p+j-1]<=st.close_root_rtol*fabs(root))||
         (j+1<st.count[t]&&st.roots[p+j+1]-root<=st.close_root_rtol*fabs(st.roots[p+j+1]));
}

__device__ md::GroupResult root_group(Device st,int t,int j,int L,int nf,int K,double rtol){
  const size_t p=static_cast<size_t>(t)*K;const int model=t/nf;
  const double root=st.roots[p+j];
  if(group_close(st,t,j,K,root)){
    md::GroupResult result;result.flags=md::GROUP_CLOSE_ROOT;return result;
  }
  const double lo=j?.5*(st.roots[p+j-1]+root):st.bounds[2*model];
  const double hi=j+1<st.count[t]?.5*(root+st.roots[p+j+1]):st.bounds[2*model+1];
  return md::group_velocity(medium(st,model,L),md::two_pi<double>()/st.trace_period(t,nf),root,lo,hi,rtol,st.close_root_rtol);
}

// verify is only used by the independent high-accuracy path. In this
// balanced/fast stage it holds one immutable eligibility bit per anchor.
__global__ void continuation_sources(Device st,int traces,int K){
  const int t=blockIdx.x*blockDim.x+threadIdx.x;
  if(t<traces)st.verify[t]=st.prediction_flags[t]==P_ANCHOR&&
      md::continuation_source_safe(st.roots+static_cast<size_t>(t)*K,st.count[t],K,st.flags[t],st.close_root_rtol);
}

__global__ void anchor_slopes(Device st,int traces,int L,int nf,int K,double rtol){
  const int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=traces*K)return;
  st.slopes[index]=CUDART_NAN;
  const int t=index/K,j=index%K;
  if(st.prediction_flags[t]!=P_ANCHOR||j>=st.count[t])return;
  const int f=t%nf,m=t-f;
  const int left=f?st.anchor_left[f-1]:f;
  const int right=f+1<nf?st.anchor_right[f+1]:f;
  const bool needed=(left<f-1&&st.verify[m+left]&&st.count[m+left]==st.count[t])||
                    (right>f+1&&st.verify[m+right]&&st.count[m+right]==st.count[t]);
  if(!st.verify[t]||!needed){
    if(j==0)atomicAdd(&st.stats[CONT_PRESCREEN_SKIPPED_ROOTS],static_cast<unsigned long long>(st.count[t]));
    return;
  }
  if(j==0)atomicAdd(&st.stats[CONT_DERIVATIVE_ROOTS],static_cast<unsigned long long>(st.count[t]));
  const auto result=root_group(st,t,j,L,nf,K,rtol);
  atomicAdd(&st.stats[CONT_EVALS],static_cast<unsigned long long>(result.evaluations));
  atomicAdd(&st.stats[CONT_DERIVATIVE_EVALS],static_cast<unsigned long long>(result.evaluations));
  if(!result.flags&&isfinite(result.slope)){
    st.slopes[index]=result.slope;
    if(st.group_output&&j<K-2){
      const size_t out=(static_cast<size_t>(t/nf)*(K-2)+j)*nf+t%nf;
      st.group_output[out]=result.velocity;st.group_flags[out]=0;
    }
  }
}

__device__ void continuation_fallback(Device st,int t,int K){
  const size_t p=static_cast<size_t>(t)*K;
  st.prediction_flags[t]|=P_FALLBACK;st.active[t]=1;st.count[t]=0;st.prev_count[t]=-1;st.status[t]=0;st.level[t]=-1;st.flags[t]=0;
  st.cache_n[t]=0;st.verify[t]=0;st.guard_count[t]=-1;st.full_count[t]=-1;
  for(int j=0;j<K;++j){st.previous[p+j]=CUDART_NAN;st.roots[p+j]=CUDART_NAN;}
  st.queue[atomicAdd(st.queue_n,1)]=t;atomicAdd(&st.stats[CONT_FALLBACKS],1ULL);
}

// Reject cheap geometry failures before running any secular functions. Pack
// only eligible traces so neighboring lanes do useful correction work rather
// than waiting for a small scattered subset of an otherwise inactive warp.
__global__ void continuation_prepare(Device st,int traces,int nf,int K,double tol){
  const int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=traces)return;
  // Observation fallback leaves completed traces active in the legacy path.
  // Hold every completed anchor before a second round can inspect activity.
  if(st.prediction_flags[t]==P_ANCHOR){st.active[t]=0;return;}
  if(st.level[t]==-3||st.level[t]==-4){st.prediction_flags[t]=P_BYPASSED;return;}
  const int f=t%nf,model=t/nf,left=model*nf+st.anchor_left[f],right=model*nf+st.anchor_right[f];
  if((st.flags[left]|st.flags[right])&4096u)st.prediction_flags[t]|=P_SOURCE_UNCERTAIN;
  const size_t p=static_cast<size_t>(t)*K,pl=static_cast<size_t>(left)*K,pr=static_cast<size_t>(right)*K;
  const double dl=md::continuation_log_step(st.period[f],st.period[left%nf]);
  const double dr=md::continuation_log_step(st.period[f],st.period[right%nf]);
  const double lower=st.bounds[2*model],upper=st.bounds[2*model+1];
  const int nr=st.count[left];
  unsigned reason=P_ROOT_COUNT;
  if(nr>0&&nr<=K&&st.count[right]==nr&&upper>lower){
    reason=md::continuation_predict(st.roots+pl,st.roots+pr,st.slopes+pl,st.slopes+pr,nr,
                                    dl,dr,lower,upper,tol,st.previous+p,st.full_roots+p);
  }
  if(!reason){
    st.count[t]=nr;
    st.next_queue[atomicAdd(st.next_n,1)]=t;
    atomicAdd(&st.stats[CONT_CANDIDATES],1ULL);
  }else{st.prediction_flags[t]|=reason;continuation_fallback(st,t,K);}
}

// Match the existing global search's prepared evaluator. These constants
// depend on frequency but not on the local root's velocity. Derivative
// stencils deliberately do not reuse them when perturbing frequency.
template<bool FP32>
__global__ void continuation_constants(Device st,int L,int nf){
  for(int index=blockIdx.x;index<*st.next_n;index+=gridDim.x){
    const int t=st.next_queue[index],model=t/nf;
    const double omega=md::two_pi<double>()/st.trace_period(t,nf);
    double *ka=st.continuation_constants+static_cast<size_t>(t)*3*L;
    for(int j=threadIdx.x;j<L;j+=blockDim.x)
      prepare_trace_layer<FP32>(j,L,omega,st.a[static_cast<size_t>(model)*L+j],
                               st.b[static_cast<size_t>(model)*L+j],ka,ka+L,ka+2*L);
  }
}

template<bool FP32>
__global__ void continuation_correct(Device st,int traces,int L,int nf,int K,int modes,double tol,double accept){
  const int index=blockIdx.x*blockDim.x+threadIdx.x;
  if(index>=traces||index>=*st.next_n)return;
  const int t=st.next_queue[index],f=t%nf,model=t/nf;
  const int nr=st.count[t];
  const size_t p=static_cast<size_t>(t)*K;
  const double omega=md::two_pi<double>()/st.period[f];
  const double dl=md::continuation_log_step(st.period[f],st.period[st.anchor_left[f]]);
  const double dr=md::continuation_log_step(st.period[f],st.period[st.anchor_right[f]]);
  const double lower=st.bounds[2*model],upper=st.bounds[2*model+1];
  auto base=medium(st,model,L);
  if(st.continuation_constants){
    base.ka=st.continuation_constants+static_cast<size_t>(t)*3*L;
    base.kb=base.ka+L;base.gammk=base.ka+2*L;
  }
  const auto med=secular_medium<FP32>(st,base,static_cast<size_t>(model)*L);
  unsigned long long evals=0;
  bool audited=false;
  const size_t tab=static_cast<size_t>(model)*TABLE;
  const unsigned issue=md::continuation_roots(med,omega,dl,dr,lower,upper,st.previous+p,st.full_roots+p,
      nr,K,tol,accept,st.close_root_rtol,st.tx+tab,st.tn+tab,st.nt[model],st.seeds+2*model,
      st.lo+p,st.hi+p,evals,audited);
  st.prediction_flags[t]|=issue|(audited?P_AUDIT:0u);
  if(issue&P_AUDIT_FAILED)atomicAdd(&st.stats[CONT_AUDIT_FAILED],1ULL);
  atomicAdd(&st.stats[CONT_EVALS],evals);
  if(!issue){
    for(int j=0;j<K;++j)st.roots[p+j]=j<nr?st.full_roots[p+j]:CUDART_NAN;
    st.count[t]=nr;st.level[t]=12;st.active[t]=0;st.flags[t]|=CONTINUATION_APPROXIMATE|2048u;
    st.prediction_flags[t]|=P_ACCEPTED;
    for(int j=0;j<modes;++j){
      const double root=j<nr?st.roots[p+j]:CUDART_NAN;
      st.output[(static_cast<size_t>(model)*modes+j)*nf+f]=root;
      if(root>=.99*med.vs[L-1])st.flags[t]|=CUTOFF;
    }
    atomicAdd(&st.stats[CONT_ACCEPTED],1ULL);
  }else{
    // Restore a clean trace before the unchanged complete search fallback.
    continuation_fallback(st,t,K);
  }
}

__global__ void group_output(Device st,int traces,int L,int nf,int K,int modes,double rtol){
  const int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=traces*modes)return;
  const int t=index/modes,j=index%modes,model=t/nf;
  const size_t out=(static_cast<size_t>(model)*modes+j)*nf+t%nf;
  if(st.prediction_flags&&st.prediction_flags[t]==P_ANCHOR&&
     isfinite(st.slopes[static_cast<size_t>(t)*K+j])){
    atomicAdd(&st.stats[GROUP_REUSED],1ULL);return;
  }
  st.group_output[out]=CUDART_NAN;st.group_flags[out]=1u;
  if(j>=st.count[t]||!isfinite(st.output[out])){atomicAdd(&st.stats[GROUP_INVALID],1ULL);return;}
  const auto result=root_group(st,t,j,L,nf,K,rtol);
  st.group_flags[out]=result.flags;st.group_output[out]=result.flags?CUDART_NAN:result.velocity;
  atomicAdd(&st.stats[GROUP_EVALS],static_cast<unsigned long long>(result.evaluations));
  if(result.flags)atomicAdd(&st.stats[GROUP_INVALID],1ULL);
}
