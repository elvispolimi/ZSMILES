#pragma once

// Order of ring preprocessing:
// - 0 deepest ring lowest ID
// - 1 external ring lower ID
#define PREPROCESS_ORDER 0

#include <cpu/compressor.hpp>
#ifdef ENABLE_CUDA_IMPLEMENTATION
  #include <cuda/compressor.cuh>
#endif
#ifdef ENABLE_HIP_IMPLEMENTATION
  #include <hip/compressor.hpp>
#endif