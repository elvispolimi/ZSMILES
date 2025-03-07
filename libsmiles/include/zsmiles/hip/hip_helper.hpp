#pragma once

#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>

#define CHECK_HIP_KERNEL_ERRORS(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(hipError_t code, const char* file, int line, bool abort = true) {
  if (code != hipSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}

#define CHECK_HIP_ERRORS() CHECK_HIP_KERNEL_ERRORS(hipPeekAtLastError());

#ifdef NDEBUG
  #define SYNCH_AND_CHECK_HIP_ERROR(kern_name)                \
    {                                                         \
      CHECK_HIP_KERNEL_ERRORS(hipPeekAtLastError());          \
      CHECK_HIP_KERNEL_ERRORS(hipDeviceSynchronize());        \
      printf("running the peek and synch " #kern_name " \n"); \
    }
#else
  #define SYNCH_AND_CHECK_HIP_ERROR(kern_name) \
    {}
#endif

#ifdef __HIP_PLATFORM_NVIDIA__
  #define SYNC                         __syncwarp()
  #define SHFL_UP(mask, value, offset) __shfl_up_sync(mask, value, offset)
  #define SHFL(mask, value, offset)    __shfl_sync(mask, value, offset)
#else
  #define SYNC                         _syncthreads()
  #define SHFL_UP(mask, value, offset) __shfl_up(mask, value, offset)
  #define SHFL(mask, value, offset)    __shfl(mask, value, offset)
#endif
