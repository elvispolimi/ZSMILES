#pragma once

#include <compression_dictionary.hpp>
// Both:
// - NVIDIA split warp-scheduler for 16 threads
// - AMD work in locksteps of 16 thread (64/16=4-SIMD unit)
// Problem if the other one is empty
#define WORK_GROUP_SIZE 32
#ifdef __HIP_PLATFORM_HCC__
  #define WARP_SIZE 64
#else
  #define FULL_MASK 0xFFFFFFFF
  // #define FULL_MASK 0x00FF00FF
  #define WARP_SIZE 32
#endif
static_assert(WARP_SIZE>LONGEST_PATTERN, "WARP_SIZE has to be greater than the LONGEST_PATTERN");
static_assert(WARP_SIZE%WORK_GROUP_SIZE==0, "WARP_SIZE should be a multiple of WORK_GROUP_SIZE");
#define BLOCK_SIZE WARP_SIZE
#define GRID_SIZE  1024
#define NUM_WORK_GROUP GRID_SIZE * WARP_SIZE / WORK_GROUP_SIZE 

#define MAX_SMILES_LEN   96
#define SMILES_PER_BLOCK 64

#define SMILES_PER_DEVICE (std::size_t) GRID_SIZE* BLOCK_SIZE* SMILES_PER_BLOCK
#define CHAR_PER_DEVICE   (std::size_t) SMILES_PER_DEVICE* MAX_SMILES_LEN
