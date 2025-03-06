#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA_KERNEL_ERRORS(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}

#define CHECK_CUDA_ERRORS() CHECK_CUDA_KERNEL_ERRORS(cudaPeekAtLastError());

#ifdef NDEBUG
  #define SYNCH_AND_CHECK_CUDA_ERROR(kern_name)               \
    {                                                         \
      CHECK_CUDA_KERNEL_ERRORS(cudaPeekAtLastError());               \
      CHECK_CUDA_KERNEL_ERRORS(cudaDeviceSynchronize());             \
      printf("running the peek and synch " #kern_name " \n"); \
    }
#else
  #define SYNCH_AND_CHECK_CUDA_ERROR(kern_name) \
    {}
#endif