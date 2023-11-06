#include "compression_dictionary.hpp"
#include "cpu/compressor.hpp"
#include "cuda/compressor.cuh"
#include "cuda/nvidia_helper.cuh"
#include "utils.hpp"

#include <cstdint>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <string_view>
#include <thrust/fill.h>

// TODO maybe you can create a script to compute which is the maximum number of overlapping pattern
// for something like BCAL and dict{B, BCA} maximum 2 overlapping starting from B
#define WARP_SIZE          32
#define THREADS_PER_BLOCK  WARP_SIZE
#define THREADS_PER_SM     128
#define SHARED_BYTE_PER_SM 65 * 1024
// #define SMILES_CHAR_PER_SM (SHARED_BYTE_PER_SM + LONGEST_PATTERN) / 8
#define SMILES_CHAR_PER_SM 2056 * 8

namespace smiles {
  namespace cuda {
    smiles_compressor::smiles_compressor() {
      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, 0);
      smiles_host.reserve(SMILES_CHAR_PER_SM * deviceProp.multiProcessorCount);
      CHECK_CUDA_KERNEL_ERRORS(
          cudaMalloc(&smiles_dev, SMILES_CHAR_PER_SM * deviceProp.multiProcessorCount * sizeof(smiles_type)));
      CHECK_CUDA_KERNEL_ERRORS(
          cudaMalloc(&shortest_path_dev,
                     SMILES_CHAR_PER_SM * deviceProp.multiProcessorCount * sizeof(cost_type)));
      CHECK_CUDA_KERNEL_ERRORS(
          cudaMalloc(&shortest_path_index_dev,
                     SMILES_CHAR_PER_SM * deviceProp.multiProcessorCount * sizeof(pattern_index_type)));
      CHECK_CUDA_KERNEL_ERRORS(cudaMalloc(&costs_index_dev,
                                          SMILES_CHAR_PER_SM * LONGEST_PATTERN *
                                              deviceProp.multiProcessorCount * sizeof(pattern_index_type)));
      CHECK_CUDA_KERNEL_ERRORS(cudaMalloc(&costs_dev,
                                          SMILES_CHAR_PER_SM * LONGEST_PATTERN *
                                              deviceProp.multiProcessorCount * sizeof(cost_type)));
      CHECK_CUDA_KERNEL_ERRORS(
          cudaMalloc(&smiles_start_dev,
                     SMILES_CHAR_PER_SM * deviceProp.multiProcessorCount * sizeof(index_type)));
      // TODO can we assume that on average the size is reduced?
      // Potentially the size can double if no match have been found
      // Problem when writing in output for sizes of smiles index
      CHECK_CUDA_KERNEL_ERRORS(
          cudaMalloc(&smiles_output_dev,
                     SMILES_CHAR_PER_SM * deviceProp.multiProcessorCount * sizeof(smiles_type)));
      smiles_output_host.resize(SMILES_CHAR_PER_SM * deviceProp.multiProcessorCount);
    };

    smiles_compressor::~smiles_compressor() {
      if (smiles_dev != nullptr)
        CHECK_CUDA_KERNEL_ERRORS(cudaFree(smiles_dev));
      if (smiles_output_dev != nullptr)
        CHECK_CUDA_KERNEL_ERRORS(cudaFree(smiles_output_dev));
      if (costs_dev != nullptr)
        CHECK_CUDA_KERNEL_ERRORS(cudaFree(costs_dev));
      if (smiles_start_dev != nullptr)
        CHECK_CUDA_KERNEL_ERRORS(cudaFree(smiles_start_dev));
      if (shortest_path_dev != nullptr)
        CHECK_CUDA_KERNEL_ERRORS(cudaFree(shortest_path_dev));
      if (costs_index_dev != nullptr)
        CHECK_CUDA_KERNEL_ERRORS(cudaFree(costs_index_dev));
    }

    __global__ void match_pattern(const smiles_compressor::smiles_type* __restrict__ smiles,
                                  const smiles_compressor::index_type last_smiles_char,
                                  smiles_compressor::cost_type* __restrict__ costs,
                                  smiles_compressor::pattern_index_type* __restrict__ costs_index,
                                  const smiles_compressor::index_type costs_stride) {
      const int stride   = blockDim.x * blockDim.y;
      const int threadId = threadIdx.x + blockDim.x * threadIdx.y;
      const int offset   = blockIdx.x * SMILES_CHAR_PER_SM;
      smiles += offset;
      costs_index += offset;
      costs += offset;

      __shared__ smiles_compressor::smiles_type smiles_shared[SMILES_CHAR_PER_SM + LONGEST_PATTERN];
// Cooperative load the string in shared memory
#pragma unroll
      for (int i = threadId; i < SMILES_CHAR_PER_SM && (i + offset) < last_smiles_char; i += stride) {
        smiles_shared[i] = smiles[i];
      }
      __syncwarp();

      constexpr_for<SMILES_DICT_NOT_PRINT, SMILES_DICT_SIZE, 1>([&](const auto index) {
        auto costs_t       = costs + costs_stride * (SMILES_DICTIONARY[index].size - 1);
        auto costs_index_t = costs_index + costs_stride * (SMILES_DICTIONARY[index].size - 1);
#pragma unroll 8
        for (int i = threadId; i < SMILES_CHAR_PER_SM && (i + offset) < last_smiles_char; i += stride) {
          int equal;
#pragma unroll
          for (int t = 0, equal = 1; t < SMILES_DICTIONARY[index].size; t++)
            equal &= (SMILES_DICTIONARY[index].pattern[t] == smiles_shared[i + t]);

          if (equal) {
            costs_t[i + SMILES_DICTIONARY[index].size] = 1;
            costs_index_t[i + SMILES_DICTIONARY[index].size] =
                static_cast<smiles_compressor::pattern_index_type>(index);
          }
        }
        __syncwarp();
      });
    }

    __global__ void compute_dijkstra(const smiles_compressor::smiles_type* __restrict__ smiles,
                                     smiles_compressor::smiles_type* __restrict__ smiles_output,
                                     smiles_compressor::cost_type* const __restrict__ costs,
                                     smiles_compressor::pattern_index_type* const __restrict__ costs_index,
                                     const smiles_compressor::index_type* __restrict__ smiles_start,
                                     const int num_of_smiles,
                                     smiles_compressor::cost_type* __restrict__ shortest_path,
                                     smiles_compressor::pattern_index_type* __restrict__ shortest_path_index,
                                     const int costs_stride) {
      // const int threadId = threadIdx.x + blockDim.x * threadIdx.y;
      const int threadId = threadIdx.x;
      const int stride   = gridDim.x * blockDim.y;

      const int smiles_index = blockIdx.x * blockDim.y + threadIdx.y;

      if (threadId % WARP_SIZE == 0) {
        smiles_compressor::cost_type* costs_temp;
        smiles_compressor::pattern_index_type* costs_index_temp;
        smiles_compressor::cost_type best_costs;
        smiles_compressor::cost_type best_index;
        smiles_compressor::pattern_index_type best_costs_index;
        int o, l, t, t1, i;
        __shared__ smiles_compressor::cost_type cost_l[LONGEST_PATTERN * LONGEST_PATTERN];
        __shared__ smiles_compressor::pattern_index_type cost_index_l[LONGEST_PATTERN * LONGEST_PATTERN];
        for (int index = smiles_index; index < num_of_smiles; index += stride) {
          // TODO Maybe move instantiation before the loop if the compiler is not able to do so
          // amd manually SET THE MEMORY TO 0
          // You need only one and then you move to the left every iteration
#pragma unroll 8
          for (i = 0; i < LONGEST_PATTERN * LONGEST_PATTERN; i++)
            cost_l[i] = std::numeric_limits<smiles_compressor::cost_type>().max();
          cost_l[0]                   = 0;
          cost_l[LONGEST_PATTERN]     = 0;
          cost_l[LONGEST_PATTERN * 2] = 0;

          const int smile_start = smiles_start[index];
          const int smile_end   = smiles_start[index + 1];
          const int smile_len   = smile_end - smile_start;
          // Skip the first one which is trivial to select the smallest value
          for (l = 0; l < smile_len; l++) {
            // Save the index of the prev first element of tot_cost into global memory
            shortest_path[smile_end - l]       = cost_l[LONGEST_PATTERN];
            shortest_path_index[smile_end - l] = cost_l[LONGEST_PATTERN * 2];
            // Reset the next memory
            // #pragma unroll
            //             for (int i = 0; i < LONGEST_PATTERN * LONGEST_PATTERN; i++)
            //               next_cost_l[i] = std::numeric_limits<smiles_compressor::cost_type>().max();
            costs_temp       = costs + smile_end - l;
            costs_index_temp = costs_index + smile_end - l;
            best_costs       = cost_l[0] + 2;
            // 0 in shortest path means that you should escape the character
            best_index       = 0;
            best_costs_index = 0;

// Compute the best for the next one
#pragma unroll 8
            for (t = 0; t < LONGEST_PATTERN; t++) {
              if (*costs_temp) {
                cost_l[LONGEST_PATTERN * t + t + 1]       = cost_l[0] + 1;
                cost_index_l[LONGEST_PATTERN * t + t + 1] = *costs_index_temp;
              }
              costs_temp += costs_stride;
              costs_index_temp += costs_stride;
              if (best_costs > cost_l[LONGEST_PATTERN * t + 1]) {
                best_costs       = cost_l[LONGEST_PATTERN * t + 1];
                best_costs_index = cost_index_l[LONGEST_PATTERN * t + 1];
                best_index       = t;
              }
            }
#pragma unroll 8
            for (t = 0; t < LONGEST_PATTERN; t++) {
#pragma unroll 8
              for (t1 = 2; t1 < LONGEST_PATTERN; t1++) {
                cost_l[LONGEST_PATTERN * t + t1 - 1]       = cost_l[LONGEST_PATTERN * t + t1];
                cost_index_l[LONGEST_PATTERN * t + t1 - 1] = cost_index_l[LONGEST_PATTERN * t + t1];
              }
              cost_l[LONGEST_PATTERN * t + LONGEST_PATTERN - 1] =
                  std::numeric_limits<smiles_compressor::cost_type>().max();
            }
            cost_l[0]                   = best_costs;
            cost_l[LONGEST_PATTERN]     = best_index;
            cost_l[LONGEST_PATTERN * 2] = best_costs_index;
          }
          // Save the result of the first one
          shortest_path[smile_start] = cost_l[LONGEST_PATTERN];
          // TODO potential cast if pattern_index_type!=index_type
          shortest_path_index[smile_start] = cost_l[LONGEST_PATTERN * 2];
          // Now compress the string by doing pattern matching and replacing
          o = 0;
          for (l = 0; l < smile_len - 1; l++) {
            if (!shortest_path[smile_start + l] && !shortest_path_index[smile_start + l]) {
              smiles_output[smile_start + o] = '\\';
              o++;
              smiles_output[smile_start + o] = smiles[smile_start + l];
              o++;
            } else {
              // TODO Check casting problem
              smiles_output[smile_start + o] =
                  static_cast<smiles_compressor::smiles_type>(shortest_path_index[smile_start + l]);
              o++;
              l += shortest_path[smile_start + l];
            }
          }
          smiles_output[smile_start + o] = '\0';
          // printf("%s\n", &smiles_output[smile_start]);
        }
      }
      __syncwarp();
    }

    void smiles_compressor::compute_host(std::ofstream& out_s) {
      // Push back the length of the last one
      smiles_start.push_back(smiles_host.size() - 1);
      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, 0);
      CHECK_CUDA_KERNEL_ERRORS(cudaMemcpy(smiles_dev,
                                          smiles_host.data(),
                                          smiles_host.size() * sizeof(smiles_type),
                                          cudaMemcpyHostToDevice));
      CHECK_CUDA_KERNEL_ERRORS(
          cudaMemset(smiles_start_dev,
                     0,
                     SMILES_CHAR_PER_SM * deviceProp.multiProcessorCount * sizeof(index_type)));
      CHECK_CUDA_KERNEL_ERRORS(
          cudaMemset(shortest_path_dev,
                     0,
                     SMILES_CHAR_PER_SM * deviceProp.multiProcessorCount * sizeof(cost_type)));
      CHECK_CUDA_KERNEL_ERRORS(cudaMemcpy(smiles_start_dev,
                                          smiles_start.data(),
                                          smiles_start.size() * sizeof(index_type),
                                          cudaMemcpyHostToDevice));

      const dim3 block_dimension_1{THREADS_PER_BLOCK, THREADS_PER_SM / THREADS_PER_BLOCK};
      const dim3 grid_dimension_1{2056 * 2};
      match_pattern<<<grid_dimension_1, block_dimension_1>>>(smiles_dev,
                                                             smiles_host.size() - 1,
                                                             costs_dev,
                                                             costs_index_dev,
                                                             SMILES_CHAR_PER_SM *
                                                                 deviceProp.multiProcessorCount);
      const dim3 block_dimension_2{WARP_SIZE, THREADS_PER_SM / WARP_SIZE};
      const dim3 grid_dimension_2{2056};
      compute_dijkstra<<<grid_dimension_2, block_dimension_2>>>(smiles_dev,
                                                                smiles_output_dev,
                                                                costs_dev,
                                                                costs_index_dev,
                                                                smiles_start_dev,
                                                                smiles_start.size() - 1,
                                                                shortest_path_dev,
                                                                shortest_path_index_dev,
                                                                SMILES_CHAR_PER_SM *
                                                                    deviceProp.multiProcessorCount);
      CHECK_CUDA_ERRORS();
      cudaDeviceSynchronize();

      // The copy back is SYNC
      CHECK_CUDA_KERNEL_ERRORS(cudaMemcpy((void*) smiles_output_host.data(),
                                          smiles_output_dev,
                                          smiles_output_host.size() * sizeof(smiles_type),
                                          cudaMemcpyDeviceToHost));

      // Print output
      // -1 because we've also added the last one
      for (int i = 0; i < smiles_start.size() - 1; i++) {
        out_s << &smiles_output_host.data()[smiles_start[i]] << std::endl;
      }
      // Clean up
      smiles_start.clear();
      smiles_host.clear();
      smiles_output_host.clear();

      return;
    }

    void smiles_compressor::clean_up(std::ofstream& out_s) {
      compute_host(out_s);
      return;
    }

    // we model the problem of SMILES compression as choosing the minimum path between the first character and the
    // last one. The cost of each path is the number of character that we need to produce in the output. We solve
    // this problem using Dijkstra and we use a support tree to perform pattern matching.
    void smiles_compressor::operator()(const std::string_view& plain_description, std::ofstream& out_s) {
      if (smiles_host.size() + plain_description.size() >= smiles_host.capacity()) {
        return compute_host(out_s);
      }
      smiles_start.push_back(smiles_host.size());
      smiles_host.append(plain_description);
      smiles_host.append(1, '\0');
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