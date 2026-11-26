#include <cuda_runtime.h>
#include <math_constants.h>
#include "common/review_groups.hpp"

namespace {
__global__ void group_trace(const double *data, md::ReviewTrace trace, int modes,
                            md::GroupResult *out) {
  const int j=blockIdx.x*blockDim.x+threadIdx.x;
  if (j>=modes) return;
  auto result=trace.group(data,j);
  if (result.flags) result.velocity=CUDART_NAN;
  out[j]=result;
}

void check(cudaError_t error) {
  if (error==cudaErrorMemoryAllocation) throw std::bad_alloc();
  if (error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
struct Buffer {
  void *data=nullptr;
  ~Buffer() { if (data) cudaFree(data); }
};
}

void evaluate_review_groups(const double *data, size_t size, md::ReviewTrace trace,
                            int modes, md::GroupResult *out) {
  Buffer buffer;
  check(cudaMalloc(&buffer.data, size*sizeof(double)+modes*sizeof(*out)));
  auto *packed=static_cast<double *>(buffer.data);
  auto *result=reinterpret_cast<md::GroupResult *>(packed+size);
  check(cudaMemcpy(packed,data,size*sizeof(double),cudaMemcpyHostToDevice));
  group_trace<<<(modes+31)/32,32>>>(packed,trace,modes,result);
  check(cudaGetLastError());
  check(cudaMemcpy(out,result,modes*sizeof(*out),cudaMemcpyDeviceToHost));
}

void bind_review_groups(pybind11::module_ &m) { register_review_groups(m); }
