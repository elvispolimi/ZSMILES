#include "compression_dictionary.hpp"
#include "cpu/compressor.hpp"
#include "cuda/compressor.cuh"
#include "cuda/nvidia_helper.cuh"
#include "utils.hpp"

#include <cassert>
#include <cstdint>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace smiles {
  namespace cuda {
    __device__ __constant__ node dictionary_tree_gpu[425];

    smiles_compressor::smiles_compressor() {
      smiles_host.reserve(CHAR_PER_DEVICE);
      CHECK_CUDA_KERNEL_ERRORS(cudaMalloc(&smiles_dev, CHAR_PER_DEVICE * sizeof(smiles_type)));
      CHECK_CUDA_KERNEL_ERRORS(cudaMalloc(&smiles_len_dev, SMILES_PER_DEVICE * sizeof(index_type)));
      CHECK_CUDA_KERNEL_ERRORS(
          cudaMalloc(&match_matrix_dev,
                     MAX_SMILES_LEN * GRID_SIZE * LONGEST_PATTERN * sizeof(pattern_index_type)));
      CHECK_CUDA_KERNEL_ERRORS(
          cudaMalloc(&dijkstra_matrix_dev,
                     MAX_SMILES_LEN * GRID_SIZE * LONGEST_PATTERN * sizeof(pattern_index_type)));
      CHECK_CUDA_KERNEL_ERRORS(cudaMalloc(&smiles_output_dev, CHAR_PER_DEVICE * sizeof(smiles_type)));
      smiles_output_host.resize(CHAR_PER_DEVICE);
      CHECK_CUDA_KERNEL_ERRORS(cudaMemcpyToSymbol(dictionary_tree_gpu,
                                                  build_gpu_smiles_dictionary().data(),
                                                  sizeof(node) * 425,
                                                  0,
                                                  cudaMemcpyHostToDevice));
    };

    smiles_compressor::~smiles_compressor() {
      if (smiles_dev != nullptr)
        CHECK_CUDA_KERNEL_ERRORS(cudaFree(smiles_dev));
      if (smiles_len_dev != nullptr)
        CHECK_CUDA_KERNEL_ERRORS(cudaFree(smiles_len_dev));
      if (match_matrix_dev != nullptr)
        CHECK_CUDA_KERNEL_ERRORS(cudaFree(match_matrix_dev));
      if (dijkstra_matrix_dev != nullptr)
        CHECK_CUDA_KERNEL_ERRORS(cudaFree(dijkstra_matrix_dev));
      if (smiles_output_dev != nullptr)
        CHECK_CUDA_KERNEL_ERRORS(cudaFree(smiles_output_dev));
    }

    __global__ void compress_gpu(const smiles_compressor::smiles_type* __restrict__ smiles_in,
                                 smiles_compressor::smiles_type* __restrict__ smiles_out,
                                 const smiles_compressor::index_type* __restrict__ smiles_len,
                                 const int num_smiles,
                                 smiles_compressor::pattern_index_type* __restrict__ match_matrix,
                                 smiles_compressor::pattern_index_type* __restrict__ dijkstra_matrix) {
      const int threadId      = threadIdx.x;
      const int blockId       = blockIdx.x;
      const int stride_smile  = gridDim.x;
      const int stride        = blockDim.x;
      const int char_offset   = MAX_SMILES_LEN * blockId;
      const int matrix_offset = MAX_SMILES_LEN * LONGEST_PATTERN * blockId;

      __shared__ smiles_compressor::smiles_type smiles_s[MAX_SMILES_LEN];

      const int* smiles_len_l                                  = smiles_len + blockId;
      const smiles_compressor::smiles_type* smiles_in_l        = smiles_in + char_offset;
      smiles_compressor::smiles_type* smiles_out_l             = smiles_out + char_offset;
      smiles_compressor::pattern_index_type* match_matrix_l    = match_matrix + matrix_offset;
      smiles_compressor::pattern_index_type* dijkstra_matrix_l = dijkstra_matrix + matrix_offset;
      for (int id = blockId; id < num_smiles; id += stride_smile,
               smiles_len_l += stride_smile,
               smiles_in_l += MAX_SMILES_LEN,
               smiles_out_l += MAX_SMILES_LEN) {
        const int smile_len = *smiles_len_l;
#pragma unroll 8
        for (int i = threadId; i < smile_len; i += stride) smiles_s[i] = smiles_in_l[i];
        __syncwarp();
#pragma unroll 8
        for (int i = 0; i < LONGEST_PATTERN; i++)
          for (int j = threadId; j < smile_len; j += stride) match_matrix_l[LONGEST_PATTERN * j + i] = 0;
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
              if (curr->pattern != -1)
                match_matrix_l[LONGEST_PATTERN * (i + j + 1) + j] = curr->pattern;
            } else {
              curr = nullptr;
            }
          }
        }
#pragma unroll 8
        for (int i = 0; i < LONGEST_PATTERN; i++)
          for (int j = threadId; j < smile_len; j += stride) {
            dijkstra_matrix_l[i * MAX_SMILES_LEN + j] =
                std::numeric_limits<smiles_compressor::pattern_index_type>().max();
          }
        __syncwarp();
        if (threadId % stride == 0) {
          dijkstra_matrix_l[smile_len]                      = 0;
          dijkstra_matrix_l[MAX_SMILES_LEN + smile_len]     = 0;
          dijkstra_matrix_l[MAX_SMILES_LEN * 2 + smile_len] = 0;

          // Skip the first one which is trivial to select the smallest value
          for (int l = smile_len; l > 0; l--) {
            // Save the index of the prev first element of tot_cost into global memory
            smiles_compressor::pattern_index_type* costs_index_temp = match_matrix_l + l * LONGEST_PATTERN;
            smiles_compressor::cost_type best_costs                 = dijkstra_matrix_l[l] + 2;
            smiles_compressor::cost_type best_index                 = 0;
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
            dijkstra_matrix_l[l - 1] = dijkstra_matrix_l[MAX_SMILES_LEN * best_index + l - 1];
            dijkstra_matrix_l[MAX_SMILES_LEN + l - 1] = best_index;
            dijkstra_matrix_l[MAX_SMILES_LEN * 2 + l - 1] =
                match_matrix_l[best_index + (l + best_index) * LONGEST_PATTERN];
          }
        }
        __syncwarp();
        // TODO you can parallelize and then make a reduction performed only by threadID 0
        if (threadId % stride == 0) {
          int o = 0;
          for (int l = 0; l < smile_len; l++) {
            if (!dijkstra_matrix_l[MAX_SMILES_LEN * 2 + l] && !dijkstra_matrix_l[MAX_SMILES_LEN + l]) {
              smiles_out_l[o] = '\\';
              o++;
              smiles_out_l[o] = smiles_s[l];
              o++;
            } else {
              smiles_out_l[o] =
                  static_cast<smiles_compressor::smiles_type>(dijkstra_matrix_l[MAX_SMILES_LEN * 2 + l]);
              o++;
              l += dijkstra_matrix_l[MAX_SMILES_LEN + l];
            }
          }
          smiles_out_l[o] = '\0';
        }
        __syncwarp();
      }
    }

    void smiles_compressor::compute_host(std::ofstream& out_s) {
      if (need_clean_up)
        copy_out(out_s);
      // TODO allocate for the worst case but then pass where the smiles begin to reduce the copied amount of data
      CHECK_CUDA_KERNEL_ERRORS(cudaMemcpy(smiles_dev,
                                          smiles_host.data(),
                                          smiles_len.size() * MAX_SMILES_LEN * sizeof(smiles_type),
                                          cudaMemcpyHostToDevice));
      CHECK_CUDA_KERNEL_ERRORS(cudaMemcpy(smiles_len_dev,
                                          smiles_len.data(),
                                          smiles_len.size() * sizeof(int),
                                          cudaMemcpyHostToDevice));

      const dim3 block_dimension{BLOCK_SIZE};
      const dim3 grid_dimension{GRID_SIZE};
      need_clean_up = true;
      compress_gpu<<<grid_dimension, block_dimension>>>(smiles_dev,
                                                        smiles_output_dev,
                                                        smiles_len_dev,
                                                        smiles_len.size(),
                                                        match_matrix_dev,
                                                        dijkstra_matrix_dev);
      // Clean up
      temp_len = smiles_len;
      smiles_len.clear();
      smiles_host.clear();

      return;
    }

    void smiles_compressor::copy_out(std::ofstream& out_s) {
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
        for(int j=0; smiles_output_host.data()[i * MAX_SMILES_LEN+j]!='\0'; j++)
          temp_out+=smiles_output_host.data()[i * MAX_SMILES_LEN+j];
        temp_out+='\n';
      }

      out_s << temp_out << std::endl;

      smiles_output_host.clear();
      need_clean_up = false;
      return;
    }

    void smiles_compressor::clean_up(std::ofstream& out_s) {
      compute_host(out_s);
      if(need_clean_up) copy_out(out_s);
      return;
    }

    std::string_view smiles_decompressor::operator()(const std::string_view& compressed_description) {
      // decompressing a SMILES is really just a look up on the SMILES_DICTIONARY. We just need to pay attention
      // when the compressed SMILES has excaped something
      // NOTE: we need to start from a clean string
      // TODO
      return {};
    }
  } // namespace cuda
} // namespace smiles