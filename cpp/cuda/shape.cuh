// BSD-3-Clause. Included inside namespace qe after the common CUDA helpers.
// A persistent CTA owns one trace. Ordered initial knots are merged lazily;
// local interval queues reuse parent samples without sorting full trace grids.
__global__ void shape_prepare_models(Device st,int B,int L){
  const int model=blockIdx.x;const size_t off=static_cast<size_t>(model)*L;
  __shared__ double x[CAP];
  __shared__ double log_anchor,log_span;
  const double lo=st.bounds[2*model],hi=st.bounds[2*model+1];
  if(!threadIdx.x&&hi>lo){
    const double c=st.canonical_lower?lo:md::automatic_lower(medium(st,model,L));
    log_anchor=log(c>0.&&isfinite(c)?fmin(lo,c):lo);
    log_span=log(nextafter(st.b[off+L-1],0.))-log_anchor;
  }
  __syncthreads();
  const int count=129+6*L+4+(st.accuracy==0?20:0), width=sort_width(count);
  for(int j=threadIdx.x;j<width;j+=blockDim.x){
    double c=CUDART_INF;
    if(j==0)c=lo;else if(j==128)c=hi;
    else if(j<128&&hi>lo)c=exp(log_anchor+log_span*j/128.);
    else if(j>=129&&j<129+6*L){
      const int layer=(j-129)/6,which=(j-129)%6;
      if(!(st.wave&&st.b[off]==0.&&layer==0)){
        const double v=which<3?st.a[off+layer]:st.b[off+layer];
        c=v*(1.+(which%3-1)*1e-7);
      }
    }else if(j>=129+6*L&&j<129+6*L+4){const int q=j-129-6*L;c=st.seeds[2*model+q/2]*(q%2?.9:1.);}
    else if(j>=129+6*L+4&&j<count){const int q=j-129-6*L-4;double gap=1e-7;for(int k=0;k<(q%10)/2;++k)gap*=10.;c=st.seeds[2*model+q/10]*(1.+(q%2?gap:-gap));}
    if(!isfinite(c)||c<lo||c>hi)c=CUDART_INF;
    x[j]=c;
  }
  __syncthreads();sort_pairs<false>(x,nullptr,count);
  if(!threadIdx.x)st.shape_n[model]=unique_keys(x,width);
  __syncthreads();
  for(int j=threadIdx.x;j<st.shape_n[model];j+=blockDim.x)st.shape_x[static_cast<size_t>(model)*CAP+j]=x[j];
}

template<bool FP32>
__device__ bool same_shape_coordinate(double a,double b){
  if constexpr (FP32)return static_cast<float>(a)==static_cast<float>(b);
  return a==b;
}

template<bool FP32>
__device__ void shape_insert(Device st,int t,int K,double a,double b,double fa,double fb,int &nr){
  const size_t p=static_cast<size_t>(t)*K;int j=0;
  while(j<nr&&st.lo[p+j]<a&&!same_shape_coordinate<FP32>(st.lo[p+j],a))++j;
  if(j<nr&&same_shape_coordinate<FP32>(st.lo[p+j],a)&&same_shape_coordinate<FP32>(st.hi[p+j],b))return;
  if(j>=K)return;
  const int last=min(nr,K-1);
  for(int k=last;k>j;--k){st.lo[p+k]=st.lo[p+k-1];st.hi[p+k]=st.hi[p+k-1];st.flo[p+k]=st.flo[p+k-1];st.fhi[p+k]=st.fhi[p+k-1];}
  st.lo[p+j]=a;st.hi[p+j]=b;st.flo[p+j]=fa;st.fhi[p+j]=fb;
  nr=min(nr+1,K);
}

template<bool FP32>
__device__ bool shape_collect(Device st,int t,int K,double *x,double *f,int n,bool last,int &nr){
  for(int j=1;j<n;++j){double a=x[j],b=f[j];int k=j;while(k&&x[k-1]>a){x[k]=x[k-1];f[k]=f[k-1];--k;}x[k]=a;f[k]=b;}
  int count=0;for(int j=0;j<n;++j)if(!count||!same_shape_coordinate<FP32>(x[j],x[count-1])){x[count]=x[j];f[count++]=f[j];}
  for(int j=0;j+1<count;++j){
    if(f[j]==0.){
      if(f[j+1]==0.||(j&&f[j-1]==0.))return false;
      if(!j||(signbit(f[j-1])!=signbit(f[j+1])))shape_insert<FP32>(st,t,K,x[j],x[j],0.,0.,nr);else return false;
    }else if(f[j+1]!=0.&&signbit(f[j])!=signbit(f[j+1]))shape_insert<FP32>(st,t,K,x[j],x[j+1],f[j],f[j+1],nr);
  }
  if(last&&count>1&&f[count-1]==0.){
    if(f[count-2]==0.)return false;
    shape_insert<FP32>(st,t,K,x[count-1],x[count-1],0.,0.,nr);
  }
  return true;
}

struct FloatShapeSample {
  float f,g;
  __device__ FloatShapeSample()=default;
  __device__ FloatShapeSample(md::ShapeSample s):f(static_cast<float>(s.f)),g(static_cast<float>(s.g)){}
  __device__ operator md::ShapeSample()const{return {f,g};}
};
struct FloatShapeNode {double a,b;FloatShapeSample s0,sm,s1;int depth;};
constexpr int float_shape_batch=16;
constexpr int float_shape_stack=2*float_shape_batch+md::shape_depth;
constexpr int shape_sign_uncertain=2;
constexpr int shape_frontier_full=3;
static_assert((float_shape_stack+2*float_shape_batch)*sizeof(FloatShapeNode)
              +2*float_shape_batch*sizeof(FloatShapeSample)
              <= md::shape_storage_doubles*sizeof(double));

// The caller bounds the interval to a handful of float ULPs. Check every
// representable velocity instead of subdividing the same sampled step.
__device__ bool shape_float_leaf(Device st,const md::FloatMedium &med,double omega,
                                 int t,int K,double a,double b,bool last,int &nr,
                                 unsigned long long &evals,int &work){
  double x[12],f[12];
  int n=0;
  const float stop=static_cast<float>(b);
  for(float c=static_cast<float>(a);;c=nextafterf(c,CUDART_INF_F)){
    if(n==12)return false;
    x[n]=fmax(a,fmin(b,static_cast<double>(c)));
    f[n]=md::evaluate(med,omega,x[n]);++evals;++work;
    if(!isfinite(f[n]))return false;
    ++n;
    if(c==stop)break;
  }
  if(n==1&&f[0]==0.)return false;
  if(last&&md::ambiguous_sh_cutoff(med,b,f[n-1]))return false;
  return shape_collect<true>(st,t,K,x,f,n,last,nr);
}

template<bool FP32,bool REPAIR,bool STREAMED>
__device__ void shape_one(Device st,int L,int nf,int K,int modes,double tol,double accept,int t,double *sm){
  constexpr int stack_size=STREAMED?float_shape_stack:md::shape_queue;
  constexpr int next_size=STREAMED?2*float_shape_batch:stack_size;
  using Node=std::conditional_t<STREAMED,FloatShapeNode,md::ShapeNode>;
  using Sample=std::conditional_t<STREAMED,FloatShapeSample,md::ShapeSample>;
  auto *nodes=reinterpret_cast<Node*>(sm),*next=nodes+stack_size;
  auto *probe=reinterpret_cast<Sample*>(next+next_size);
  double *h=sm+md::shape_storage_doubles,*a=h+L,*b=a+L,*r=b+L,*ia=r+L,*ib=ia+L,*ka=ib+L,*kb=ka+L,*gammk=kb+L;
  __shared__ double coords[17];__shared__ Sample initial[33];
  __shared__ int bad,nr,work,nseg,nq,nnext,pending,bulk_limit,seed_index,finished,density_ready;
  __shared__ double density_next,previous,covered,density_point;
  __shared__ unsigned long long accepted_nodes,subdivided_nodes,reused_nodes;
  const int model=t/nf;const size_t off=static_cast<size_t>(model)*L,tab=static_cast<size_t>(model)*TABLE;
  const double period=st.period[t%nf],omega=md::two_pi<double>()/period;
  const double lower=st.bounds[2*model],upper=st.bounds[2*model+1];
  const bool exact=(!FP32||STREAMED)&&st.accuracy==0;
  const double epsilon=exact?.015625:.25;
  const double min_relative=exact?0.:st.close_root_rtol*(st.accuracy==1?.5:1.);
  const double nlo=st.tn[tab]/period,nhi=st.tn[tab+st.nt[model]-1]/period;
  const double start=floor(nlo/epsilon)+1.,stop=nhi;
  for(int j=threadIdx.x;j<L;j+=blockDim.x){
    h[j]=st.d[off+j];a[j]=st.a[off+j];b[j]=st.b[off+j];r[j]=st.r[off+j];ia[j]=st.ia[off+j];ib[j]=st.ib[off+j];
    prepare_trace_layer<FP32>(j,L,omega,a[j],b[j],ka,kb,gammk);
  }
  if(!threadIdx.x){
    bad=0;nr=0;work=0;seed_index=0;finished=0;density_ready=0;density_point=CUDART_INF;accepted_nodes=subdivided_nodes=reused_nodes=0;previous=lower;density_next=start*epsilon;
    covered=upper;
    if(!isfinite(nhi)){bad=1;atomicAdd(&st.stats[SHAPE_WORK_LIMIT],1ULL);}
    atomicAdd(&st.stats[SEARCHES],1ULL);if(!exact)st.flags[t]|=2048u;
  }
  __syncthreads();
  const auto med=secular_medium<FP32>(st,
      {h,a,b,r,ia,ib,L,st.wave,MD_TRACE_CONSTANTS?ka:nullptr,MD_TRACE_CONSTANTS?kb:nullptr,
       MD_TRACE_CONSTANTS?gammk:nullptr,st.irho+off,st.irho2+off,st.secular,st.refinement},off);
  unsigned long long evals=0;
  while(!bad&&!finished){
    // All lanes must finish the loop-condition read before the leader can
    // set bad while producing density coordinates (independent warp lanes).
    __syncthreads();
    if(!threadIdx.x){
      coords[0]=previous;nseg=0;
      const double *seeds=st.shape_x+static_cast<size_t>(model)*CAP;
      while(nseg<16&&previous<upper){
        while(seed_index<st.shape_n[model]&&seeds[seed_index]<=previous)++seed_index;
        const double c0=seed_index<st.shape_n[model]?seeds[seed_index]:CUDART_INF;
        double c1=CUDART_INF;
        while(density_next<stop){
          if(!density_ready){
            if(++work>md::shape_work){bad=1;atomicAdd(&st.stats[SHAPE_WORK_LIMIT],1ULL);break;}
            const double target=density_next*period;
            density_point=exact?md::shape_exact_seed(med,st.tx+tab,st.tn+tab,st.nt[model],target,epsilon*period)
                               :md::shape_seed(st.tx+tab,st.tn+tab,st.nt[model],target);
            density_ready=1;
          }
          c1=density_point;
          if(!isfinite(c1)||density_next+epsilon==density_next){bad=1;atomicAdd(&st.stats[SHAPE_WORK_LIMIT],1ULL);break;}
          if(c1>previous)break;density_next+=epsilon;density_ready=0;
        }
        if(bad)break;
        if(!(density_next<stop))c1=CUDART_INF;
        const double c=fmin(c0,c1);
        if(c>covered||!isfinite(c)){bad=1;atomicAdd(&st.stats[SHAPE_WORK_LIMIT],1ULL);break;}
        coords[++nseg]=c;previous=c;
      }
      work+=2*nseg+1;nq=nseg;
      if constexpr (STREAMED) {
        pending=nseg;
        const double span=coords[nseg]-coords[0];
        const double leaf=(exact?md::coordinate_floor(med):min_relative)*coords[0];
        int depth=md::shape_depth;
        if(leaf>0.)depth=span>leaf?min(depth,ilogb(span)-ilogb(leaf)+2):0;
        bulk_limit=stack_size-float_shape_batch-depth;
      }
    }
    __syncthreads();if(bad)break;
    for(int j=threadIdx.x;j<2*nseg+1;j+=blockDim.x){
      const double c=j%2?coords[j/2]+.5*(coords[j/2+1]-coords[j/2]):coords[j/2];
      initial[j]=md::shape_evaluate(med,omega,c);++evals;
    }
    __syncthreads();
    for(int j=threadIdx.x;j<nq;j+=blockDim.x)
      nodes[STREAMED?pending-1-j:j]={coords[j],coords[j+1],initial[2*j],initial[2*j+1],initial[2*j+2],0};
    __syncthreads();
    while(nq&&!bad){
      for(int j=threadIdx.x;j<2*nq;j+=blockDim.x){const auto node=nodes[STREAMED?pending-1-j/2:j/2];const double c=node.a+(j%2?.75:.25)*(node.b-node.a);probe[j]=md::shape_evaluate(med,omega,c);++evals;}
      __syncthreads();
      if(!threadIdx.x){
        nnext=0;work+=2*nq;
        if(work>md::shape_work){bad=1;atomicAdd(&st.stats[SHAPE_WORK_LIMIT],1ULL);}
        for(int j=0;j<nq&&!bad;++j){
          const auto node=nodes[STREAMED?pending-1-j:j];const double width=node.b-node.a;
          if(nr==K&&node.a>st.hi[static_cast<size_t>(t)*K+K-1])continue;
          md::ShapeSample s[5]={node.s0,probe[2*j],node.sm,probe[2*j+1],node.s1};
          reused_nodes+=3;
          for(int k=0;k<5;++k)if(!md::shape_finite(s[k]))bad=1;
          if(bad)break;
          bool good=md::shape_consistent(s,st.sampling_rtol,!exact,FP32?8.*FLT_EPSILON:64.*DBL_EPSILON),bracket=false;
          for(int k=0;k<4;++k)if((s[k].f<0.)!=(s[k+1].f<0.)||s[k].f==0.)bracket=true;
          if constexpr (FP32) {
            if(!good&&exact&&width<=md::coordinate_floor(med)*node.b){
              if(!shape_float_leaf(st,med,omega,t,K,node.a,node.b,node.b==upper,nr,evals,work)){
                st.status[t]=shape_sign_uncertain;bad=1;break;
              }
              ++accepted_nodes;
              if(!bracket)st.flags[t]|=4096u;
              continue;
            }
            // Use an absolute FP32 noise floor only after a high-search
            // interval has been narrowed to a few parts per million.
            if(!good&&exact&&width<=64.*FLT_EPSILON*node.b){
              good=md::shape_consistent(s,st.sampling_rtol,false,8.*FLT_EPSILON,8.*FLT_EPSILON);
              if(good&&!bracket)st.flags[t]|=4096u;
            }
            if(STREAMED&&!good&&!exact&&!bracket){
              double nearest=INFINITY;
              for(const auto &sample:s)nearest=fmin(nearest,fabs(sample.g));
              if(nearest>.125)good=md::shape_consistent(s,st.sampling_rtol,true,32.*FLT_EPSILON);
            }
          }
          if(!good&&exact&&bracket&&width<=.125*tol*node.b)good=true;
          if(!good&&min_relative>0.&&width<=min_relative*node.b){good=true;st.flags[t]|=4096u;}
          if(!good){
            if(node.depth>=md::shape_depth||node.a+.25*width<=node.a){bad=1;atomicAdd(&st.stats[SHAPE_DEPTH_LIMIT],1ULL);break;}
            if(nnext+2>next_size){
              if constexpr (FP32&&!STREAMED)st.status[t]=shape_frontier_full;
              bad=1;atomicAdd(&st.stats[SHAPE_WORK_LIMIT],1ULL);break;
            }
            const double mid=node.a+.5*width;
            next[nnext++]={node.a,mid,s[0],s[1],s[2],node.depth+1};next[nnext++]={mid,node.b,s[2],s[3],s[4],node.depth+1};
            ++subdivided_nodes;continue;
          }
          ++accepted_nodes;
          double x[11],f[11];int n=5;
          for(int k=0;k<5;++k){x[k]=node.a+k*.25*width;f[k]=s[k].f;}
          if constexpr (STREAMED)x[4]=node.b;
          for(int k=1;k<4;++k){
            if((s[k-1].f<0.)!=(s[k].f<0.)||(s[k].f<0.)!=(s[k+1].f<0.))continue;
            const double sign=s[k].f<0.?-1.:1.,u=sign*s[k-1].g,v=sign*s[k].g,z=sign*s[k+1].g;
            const double curvature=.5*(u+z)-v;if(!(curvature>0.&&v<u&&v<z))continue;
            const double offset=(u-z)/(4.*curvature),predicted=v-(z-u)*(z-u)/(16.*curvature);
            if(fabs(offset)>=1.||!(predicted<=0.||v<.1*fmax(u,z)))continue;
            const double c=node.a+(k+offset)*.25*width;const auto q=md::shape_evaluate(med,omega,c);++evals;++work;
            if(!md::shape_finite(q)){bad=1;break;}x[n]=c;f[n++]=q.f;
            atomicAdd(&st.stats[EXTREMA],1ULL);if(q.f*sign<0.)st.flags[t]|=4096u;
            else{int e=0;double candidate=0.;const auto minimum=md::shape_minimum(med,omega,node.a+(k-1)*.25*width,node.a+(k+1)*.25*width,sign,tol,candidate,e);evals+=e;work+=e;if(!md::shape_finite(minimum)){bad=1;break;}x[n]=candidate;f[n++]=minimum.f;if(minimum.f*sign<0.)st.flags[t]|=4096u;}
          }
          if(bad)break;
          if(node.b==upper&&md::ambiguous_sh_cutoff(med,node.b,s[4].f)){bad=1;break;}
          if(!shape_collect<STREAMED>(st,t,K,x,f,n,node.b==upper,nr)){
            if constexpr (FP32)st.status[t]=shape_sign_uncertain;
            bad=1;break;
          }
        }
      }
      __syncthreads();
      for(int j=threadIdx.x;j<nnext;j+=blockDim.x)
        nodes[STREAMED?pending-nq+nnext-1-j:j]=next[j];
      __syncthreads();
      if(!threadIdx.x){
        if constexpr (STREAMED) {
          pending+=nnext-nq;
          // Reserve space for a depth-first branch beyond the last bulk
          // batch. Both the geometric leaf width and the depth cap bound it.
          nq=pending>bulk_limit?1:min(pending,float_shape_batch);
        } else nq=nnext;
      }
      __syncthreads();
    }
    if(!threadIdx.x)finished=nr>=K||previous>=upper;
    __syncthreads();
  }
  const size_t p=static_cast<size_t>(t)*K;
  const int refine_bad=bad;__syncthreads();
  if(!refine_bad)for(int j=threadIdx.x;j<nr;j+=blockDim.x){int count=0;const size_t i=p+j;const double root=md::refine_root(med,omega,st.lo[i],st.hi[i],st.flo[i],st.fhi[i],tol,count);evals+=count;st.roots[i]=root;if(!isfinite(root))atomicExch(&bad,1);}
  __syncthreads();
  if(!threadIdx.x&&!bad){
    for(int j=1;j<nr;++j){
      const double root=st.roots[p+j],previous=st.roots[p+j-1];
      if constexpr (FP32){
        if(exact&&fp32_sign_cluster(root,previous)){
          st.status[t]=shape_sign_uncertain;st.flags[t]|=4096u;bad=1;break;
        }
      }
      if(!root_after<FP32||REPAIR>(root,previous)){bad=1;break;}
    }
    if(!bad){st.level[t]=8+st.accuracy;st.count[t]=nr;st.active[t]=0;if(md::has_close_roots(st.roots+p,nr,st.close_root_rtol))st.flags[t]|=4096u;}
  }
  __syncthreads();
  if(!bad)for(int j=threadIdx.x;j<modes;j+=blockDim.x){
    double root=j<nr?st.roots[p+j]:CUDART_NAN;
    if(isfinite(root)&&root>=.99*b[L-1])atomicOr(&st.flags[t],CUTOFF);
    if(isfinite(root)&&accept<1.){const double f=md::evaluate(med,omega,root);++evals;if(!isfinite(f)||fabs(f)>accept){root=CUDART_NAN;atomicOr(&st.flags[t],F_REJECTED);atomicAdd(&st.stats[ACCEPT_REJECTED],1ULL);}}
    st.output[(static_cast<size_t>(model)*modes+j)*nf+t%nf]=root;
  }
  __syncthreads();add_sampling_stat(&st.stats[EVALS],evals);
  if(!threadIdx.x){if(!bad){atomicAdd(&st.stats[BRACKETS],static_cast<unsigned long long>(nr));if(!exact)atomicAdd(&st.stats[VERIFICATION_SKIPPED],1ULL);}if(accepted_nodes)atomicAdd(&st.stats[SHAPE_ACCEPTED],accepted_nodes);if(subdivided_nodes)atomicAdd(&st.stats[SHAPE_SUBDIVISIONS],subdivided_nodes);if(reused_nodes)atomicAdd(&st.stats[SHAPE_REUSED],reused_nodes);}
}

template<bool FP32,bool REPAIR=false,bool STREAMED=FP32>
__global__ void shape_search(Device st,int L,int nf,int K,int modes,double tol,double accept,int repair_limit=0){
  extern __shared__ double sm[];__shared__ int index;
  if constexpr (REPAIR){
    // Apply a common cap to the entire candidate queue.
    if(*st.queue_n>repair_limit)return;
  }
  for(;;){if(!threadIdx.x)index=atomicAdd(st.cursor,1);__syncthreads();if(index>=*st.queue_n)return;
    const int t=st.queue[index];
    if(st.active[t]){
      if constexpr (REPAIR){
        if(!threadIdx.x){st.flags[t]|=FP64_REPAIR;atomicAdd(&st.stats[FP64_REPAIRS],1ULL);}
      }
      shape_one<FP32,REPAIR,STREAMED>(st,L,nf,K,modes,tol,accept,t,sm);
    }
    __syncthreads();
  }
}

__global__ void prepare_shape_retry(Device st,int traces,int reason){
  const int t=blockIdx.x*blockDim.x+threadIdx.x;
  if(t<traces&&st.active[t]&&st.status[t]==reason)
    st.queue[atomicAdd(st.queue_n,1)]=t;
}

__global__ void prepare_shape_fallback(Device st,int traces){
  const int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=traces||!st.active[t])return;
  st.flags[t]|=8192u;st.count[t]=0;st.prev_count[t]=-1;st.status[t]=0;
  st.queue[atomicAdd(st.queue_n,1)]=t;atomicAdd(&st.stats[LEGACY_FALLBACKS],1ULL);
}
