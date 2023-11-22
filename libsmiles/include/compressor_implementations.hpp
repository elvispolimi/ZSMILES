#pragma once

#include <cpu/compressor.hpp>
#ifdef ENABLE_CUDA_IMPLEMENTATION
  #include <cuda/compressor.cuh>
#endif
#ifdef ENABLE_HIP_IMPLEMENTATION
  #include <hip/compressor.hpp>
#endif