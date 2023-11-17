#include "compression_dictionary.hpp"
#include "cpu/compressor.hpp"
#include "cuda/compressor.cuh"
#include "cuda/dictionary.cuh"
#include "cuda/nvidia_helper.cuh"
#include "utils.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace smiles {
  namespace cuda {
    const __device__ __constant__ node dictionary_tree_gpu[GPU_DICT_SIZE];
    const __device__ __constant__ smiles_dictionary_entry_gpu smiles_dictionary_gpu[256];

    base_compressor::base_compressor() {
      smiles_host.reserve(CHAR_PER_DEVICE);
      CHECK_CUDA_KERNEL_ERRORS(cudaMalloc(&smiles_dev, CHAR_PER_DEVICE * sizeof(smiles_type)));
      CHECK_CUDA_KERNEL_ERRORS(cudaMalloc(&smiles_len_dev, SMILES_PER_DEVICE * sizeof(index_type)));
      CHECK_CUDA_KERNEL_ERRORS(cudaMalloc(&smiles_index_dev, SMILES_PER_DEVICE * sizeof(index_type)));
      CHECK_CUDA_KERNEL_ERRORS(cudaMalloc(&smiles_output_dev, CHAR_PER_DEVICE * sizeof(smiles_type)));
      smiles_output_host.resize(CHAR_PER_DEVICE);
    };

    base_compressor::~base_compressor() {
      if (smiles_dev != nullptr)
        CHECK_CUDA_KERNEL_ERRORS(cudaFree(smiles_dev));
      if (smiles_len_dev != nullptr)
        CHECK_CUDA_KERNEL_ERRORS(cudaFree(smiles_len_dev));
      if (smiles_output_dev != nullptr)
        CHECK_CUDA_KERNEL_ERRORS(cudaFree(smiles_output_dev));
      if (smiles_index_dev != nullptr)
        CHECK_CUDA_KERNEL_ERRORS(cudaFree(smiles_index_dev));
    }

    smiles_compressor::smiles_compressor() {
      CHECK_CUDA_KERNEL_ERRORS(
          cudaMalloc(&match_matrix_dev,
                     MAX_SMILES_LEN * GRID_SIZE * LONGEST_PATTERN * sizeof(pattern_index_type)));
      CHECK_CUDA_KERNEL_ERRORS(
          cudaMalloc(&dijkstra_matrix_dev,
                     MAX_SMILES_LEN * GRID_SIZE * LONGEST_PATTERN * sizeof(pattern_index_type)));
      CHECK_CUDA_KERNEL_ERRORS(cudaMemcpyToSymbol(dictionary_tree_gpu,
                                                  build_gpu_smiles_dictionary().data(),
                                                  sizeof(node) * GPU_DICT_SIZE,
                                                  0,
                                                  cudaMemcpyHostToDevice));
    };

    smiles_compressor::~smiles_compressor() {
      if (match_matrix_dev != nullptr)
        CHECK_CUDA_KERNEL_ERRORS(cudaFree(match_matrix_dev));
      if (dijkstra_matrix_dev != nullptr)
        CHECK_CUDA_KERNEL_ERRORS(cudaFree(dijkstra_matrix_dev));
    }

    smiles_decompressor::smiles_decompressor() {
      CHECK_CUDA_KERNEL_ERRORS(cudaMemcpyToSymbol(smiles_dictionary_gpu,
                                                  build_gpu_smiles_dictionary_entries().data(),
                                                  sizeof(smiles_dictionary_entry_gpu) * 256,
                                                  0,
                                                  cudaMemcpyHostToDevice));
    }

    __global__ void compress_gpu(const base_compressor::smiles_type* __restrict__ smiles_in,
                                 const base_compressor::index_type* __restrict__ smiles_in_index,
                                 base_compressor::smiles_type* __restrict__ smiles_out,
                                 const base_compressor::index_type* __restrict__ smiles_len,
                                 const int num_smiles,
                                 base_compressor::pattern_index_type* __restrict__ match_matrix,
                                 base_compressor::pattern_index_type* __restrict__ dijkstra_matrix) {
      const int threadId      = threadIdx.x;
      const int blockId       = blockIdx.x;
      const int stride_smile  = gridDim.x;
      const int stride        = blockDim.x;
      const int matrix_offset = MAX_SMILES_LEN * LONGEST_PATTERN * blockId;

      // TODO check if it causes a problem with a bigger number of blocks
      // __shared__ base_compressor::smiles_type smiles_s[MAX_SMILES_LEN];

      const int* smiles_len_l                                = smiles_len + blockId;
      base_compressor::pattern_index_type* match_matrix_l    = match_matrix + matrix_offset;
      base_compressor::pattern_index_type* dijkstra_matrix_l = dijkstra_matrix + matrix_offset;
      for (int id = blockId; id < num_smiles; id += stride_smile, smiles_len_l += stride_smile) {
        const int smile_len                             = *smiles_len_l;
        const base_compressor::smiles_type* smiles_in_l = smiles_in + smiles_in_index[id];
        base_compressor::smiles_type* smiles_out_l      = smiles_out + smiles_in_index[id];

        assert(smile_len < MAX_SMILES_LEN);
        // for (int i = threadId; i < smile_len; i += stride) smiles_s[i] = smiles_in_l[i];
        const base_compressor::smiles_type* smiles_s = smiles_in_l;
        __syncwarp();
#pragma unroll 8
        for (int i = 0; i < LONGEST_PATTERN; i++)
          for (int j = threadId; j <= smile_len; j += stride) match_matrix_l[LONGEST_PATTERN * j + i] = 0;
        __syncwarp();
        // For each position in the input string

        for (int i = threadId; i < smile_len; i += stride) {
          const node* curr = dictionary_tree_gpu;
          int curr_id      = 0;
#pragma unroll 8
          for (int j = 0; j < LONGEST_PATTERN && curr && j < (smile_len - i); j++) {
            const int next_i = curr->neighbor[smiles_s[i + j] - NOT_PRINTABLE];
            if (next_i) {
              curr    = &dictionary_tree_gpu[next_i + curr_id];
              curr_id = next_i + curr_id;
              if (curr->pattern != 0)
                match_matrix_l[LONGEST_PATTERN * (i + j + 1) + j] = curr->pattern;
            } else {
              curr = nullptr;
            }
          }
        }
#pragma unroll 8
        for (int i = 0; i < LONGEST_PATTERN; i++)
          for (int j = threadId; j <= smile_len; j += stride) {
            dijkstra_matrix_l[i * MAX_SMILES_LEN + j] =
                std::numeric_limits<base_compressor::pattern_index_type>().max();
          }
        __syncwarp();
        if (threadId % WARP_SIZE == 0) {
          dijkstra_matrix_l[smile_len]                      = 0;
          dijkstra_matrix_l[MAX_SMILES_LEN + smile_len]     = 0;
          dijkstra_matrix_l[MAX_SMILES_LEN * 2 + smile_len] = 0;

          // Skip the first one which is trivial to select the smallest value
          for (int l = smile_len; l > 0; l--) {
            // Save the index of the prev first element of tot_cost into global memory
            base_compressor::pattern_index_type* costs_index_temp = match_matrix_l + l * LONGEST_PATTERN;
            base_compressor::cost_type best_costs                 = dijkstra_matrix_l[l] + 2;
            base_compressor::cost_type best_index                 = 0;
// Compute the best for the next one
#pragma unroll 8
            for (int t = 0; t < LONGEST_PATTERN; t++) {
              if (*costs_index_temp) {
                dijkstra_matrix_l[MAX_SMILES_LEN * t + l - (t + 1)] = dijkstra_matrix_l[l] + 1;
              }
              costs_index_temp += 1;
              if (best_costs > dijkstra_matrix_l[MAX_SMILES_LEN * t + l - 1]) {
                best_index = t;
                best_costs = dijkstra_matrix_l[MAX_SMILES_LEN * t + l - 1];
              }
            }
            dijkstra_matrix_l[l - 1]                  = best_costs;
            dijkstra_matrix_l[MAX_SMILES_LEN + l - 1] = best_index;
            dijkstra_matrix_l[MAX_SMILES_LEN * 2 + l - 1] =
                match_matrix_l[best_index + (l + best_index) * LONGEST_PATTERN];
          }
        }
        __syncwarp();
        // TODO you can parallelize and then make a reduction performed only by threadID 0
        if (threadId % WARP_SIZE == 0) {
          int o = 0;
          for (int l = 0; l < smile_len; l++) {
            if (!dijkstra_matrix_l[MAX_SMILES_LEN * 2 + l] && !dijkstra_matrix_l[MAX_SMILES_LEN + l]) {
              smiles_out_l[o] = smiles_dictionary_escape_char;
              o++;
              smiles_out_l[o] = smiles_s[l];
              o++;
            } else {
              smiles_out_l[o] =
                  static_cast<base_compressor::smiles_type>(dijkstra_matrix_l[MAX_SMILES_LEN * 2 + l]);
              o++;
              l += dijkstra_matrix_l[MAX_SMILES_LEN + l];
            }
          }
          smiles_out_l[o] = '\0';
        }
        __syncwarp();
      }
    }

    void smiles_compressor::compress(std::ofstream& out_s) {
      if (need_clean_up)
        copy_out(out_s);
      // TODO allocate for smiles_len worst case but then pass where the smiles begin to reduce the copied amount of data
      CHECK_CUDA_KERNEL_ERRORS(cudaMemcpy(smiles_dev,
                                          smiles_host.data(),
                                          (smiles_len.back() + smiles_index.back()) * sizeof(smiles_type),
                                          cudaMemcpyHostToDevice));
      CHECK_CUDA_KERNEL_ERRORS(cudaMemcpy(smiles_len_dev,
                                          smiles_len.data(),
                                          smiles_len.size() * sizeof(index_type),
                                          cudaMemcpyHostToDevice));
      CHECK_CUDA_KERNEL_ERRORS(cudaMemcpy(smiles_index_dev,
                                          smiles_index.data(),
                                          smiles_index.size() * sizeof(index_type),
                                          cudaMemcpyHostToDevice));

      const dim3 block_dimension{BLOCK_SIZE};
      const dim3 grid_dimension{GRID_SIZE};
      need_clean_up = true;
      compress_gpu<<<grid_dimension, block_dimension>>>(smiles_dev,
                                                        smiles_index_dev,
                                                        smiles_output_dev,
                                                        smiles_len_dev,
                                                        smiles_len.size(),
                                                        match_matrix_dev,
                                                        dijkstra_matrix_dev);
      // Clean up
      temp_len   = smiles_len;
      temp_index = smiles_index;
      smiles_len.clear();
      smiles_index.clear();
      smiles_host.clear();

      return;
    }

    void smiles_compressor::copy_out(std::ofstream& out_s) {
      cudaDeviceSynchronize();
      CHECK_CUDA_ERRORS();
      // The copy back is SYNC
      CHECK_CUDA_KERNEL_ERRORS(cudaMemcpy((void*) smiles_output_host.data(),
                                          smiles_output_dev,
                                          (temp_len.back() + temp_index.back()) * sizeof(smiles_type),
                                          cudaMemcpyDeviceToHost));
      temp_out.clear();
      temp_out.reserve(temp_len.back() + temp_index.back());
      // Print output
      for (int i = 0; i < temp_len.size(); i++) {
        temp_out.append(&smiles_output_host.data()[temp_index[i]]);
        temp_out += '\n';
      }

      out_s << temp_out;

      temp_out.clear();
      smiles_output_host.clear();
      need_clean_up = false;
      return;
    }

    void smiles_decompressor::copy_out(std::ofstream& out_s) {
      cudaDeviceSynchronize();
      CHECK_CUDA_ERRORS();
      // The copy back is SYNC
      CHECK_CUDA_KERNEL_ERRORS(cudaMemcpy((void*) smiles_output_host.data(),
                                          smiles_output_dev,
                                          temp_len.size() * MAX_SMILES_LEN * sizeof(smiles_type),
                                          cudaMemcpyDeviceToHost));
      temp_out.clear();
      temp_out.reserve(temp_len.size() * MAX_SMILES_LEN);
      // Print output
      for (int i = 0; i < temp_len.size(); i++) {
        temp_out.append(&smiles_output_host.data()[i * MAX_SMILES_LEN]);
        temp_out += '\n';
      }

      out_s << temp_out;

      temp_out.clear();
      smiles_output_host.clear();
      need_clean_up = false;
      return;
    }

    void smiles_compressor::clean_up(std::ofstream& out_s) {
      compress(out_s);
      if (need_clean_up)
        copy_out(out_s);
      return;
    }

    void smiles_decompressor::clean_up(std::ofstream& out_s) {
      decompress(out_s);
      if (need_clean_up)
        copy_out(out_s);
      return;
    }

    __global__ void decompress_gpu(const base_compressor::smiles_type* __restrict__ smiles_in,
                                   const base_compressor::index_type* __restrict__ smiles_in_index,
                                   base_compressor::smiles_type* __restrict__ smiles_out,
                                   const base_compressor::index_type* __restrict__ smiles_len,
                                   const int num_smiles) {
      const int threadId     = threadIdx.x;
      const int blockId      = blockIdx.x;
      const int stride_smile = gridDim.x;
      const int stride       = blockDim.x;

      const int* smiles_len_l = smiles_len + blockId;
      for (int id = blockId; id < num_smiles; id += stride_smile, smiles_len_l += stride_smile) {
        const int smile_len                             = *smiles_len_l;
        unsigned long last_index                        = 0;
        const base_compressor::smiles_type* smiles_in_l = smiles_in + smiles_in_index[id];
        base_compressor::smiles_type* smiles_out_l      = smiles_out + id * MAX_SMILES_LEN;

        int is_previous_escape = 0;
        for (int i = threadId; i < smile_len; i += stride) {
          unsigned int mask   = __activemask();
          const base_compressor::smiles_type smile_c = smiles_in_l[i];
          const int is_escape = smile_c == smiles_dictionary_escape_char;
          // Maybe for 206 there is a smarter way to do this
          const auto is_previous_escape_t = __shfl_up_sync(mask, is_escape, 1);
          if (threadId % WARP_SIZE != 0)
            is_previous_escape = is_previous_escape_t;
          const auto find_index = static_cast<unsigned char>(smile_c);
          const auto find_pattern_length =
              is_previous_escape ? 1 : (is_escape ? 0 : smiles_dictionary_gpu[find_index].size);
          const auto find_pattern = is_previous_escape
                                        ? &smile_c
                                        : (is_escape ? "" : smiles_dictionary_gpu[find_index].pattern);

          auto index = find_pattern_length;
          for (int offset = 1; offset < warpSize; offset *= 2) {
            int tmp = __shfl_up_sync(mask, index, offset);
            if (threadIdx.x % warpSize >= offset) {
              index += tmp;
            }
          }
          index = __shfl_up_sync(mask, index, 1);
          if (threadId % WARP_SIZE == 0)
            index = 0;
          index += last_index;
          // Expand in the destination array
          for (unsigned long t = 0; t < find_pattern_length; t++) {
            smiles_out_l[t + index] = find_pattern[t];
          }
          unsigned int v = mask;
          unsigned int last_thread;
          for (last_thread = 0; v; last_thread++) {
            v &= v - 1; // clear the least significant bit set
          }
          last_index         = __shfl_sync(mask, index + find_pattern_length, last_thread - 1);
          is_previous_escape = __shfl_sync(mask, is_escape, last_thread - 1);
        }
        if (threadId % WARP_SIZE == 0)
          smiles_out_l[last_index] = '\0';
        __syncwarp();
      }
    }

    void smiles_decompressor::decompress(std::ofstream& out_s) {
      if (need_clean_up)
        copy_out(out_s);
      // TODO allocate for the worst case but then pass where the smiles begin to reduce the copied amount of data
      CHECK_CUDA_KERNEL_ERRORS(cudaMemcpy(smiles_dev,
                                          smiles_host.data(),
                                          (smiles_len.back() + smiles_index.back()) * sizeof(smiles_type),
                                          cudaMemcpyHostToDevice));
      CHECK_CUDA_KERNEL_ERRORS(cudaMemcpy(smiles_len_dev,
                                          smiles_len.data(),
                                          smiles_len.size() * sizeof(index_type),
                                          cudaMemcpyHostToDevice));
      CHECK_CUDA_KERNEL_ERRORS(cudaMemcpy(smiles_index_dev,
                                          smiles_index.data(),
                                          smiles_index.size() * sizeof(index_type),
                                          cudaMemcpyHostToDevice));

      const dim3 block_dimension{BLOCK_SIZE};
      const dim3 grid_dimension{GRID_SIZE};
      need_clean_up = true;
      decompress_gpu<<<grid_dimension, block_dimension>>>(smiles_dev,
                                                          smiles_index_dev,
                                                          smiles_output_dev,
                                                          smiles_len_dev,
                                                          smiles_len.size());
      // Clean up
      temp_len   = smiles_len;
      temp_index = smiles_index;
      smiles_len.clear();
      smiles_index.clear();
      smiles_host.clear();

      return;
    }
  } // namespace cuda
} // namespace smiles