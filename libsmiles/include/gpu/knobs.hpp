#pragma once

// #define WARP_SIZE  32
// #define BLOCK_SIZE WARP_SIZE
// #define GRID_SIZE  2056

// #define MAX_SMILES_LEN   96
// #define SMILES_PER_BLOCK 1024

// #define SMILES_PER_DEVICE (std::size_t)GRID_SIZE* BLOCK_SIZE* SMILES_PER_BLOCK
// #define CHAR_PER_DEVICE   (std::size_t)SMILES_PER_DEVICE* MAX_SMILES_LEN

#ifdef __HIP_PLATFORM_HCC__
  #define WARP_SIZE 64
#else
  #define WARP_SIZE 32
#endif
#define BLOCK_SIZE WARP_SIZE
#define GRID_SIZE  512

#define MAX_SMILES_LEN   96
#define SMILES_PER_BLOCK 128

#define SMILES_PER_DEVICE (std::size_t) GRID_SIZE* BLOCK_SIZE* SMILES_PER_BLOCK
#define CHAR_PER_DEVICE   (std::size_t) SMILES_PER_DEVICE* MAX_SMILES_LEN