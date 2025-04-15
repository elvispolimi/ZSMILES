#include "zsmiles/gpu/knobs.hpp"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <pthread.h>
#include <zsmiles/compression_dictionary.hpp>
#include <zsmiles/cuda/compressor.cuh>
#include <zsmiles/cuda/nvidia_helper.cuh>
#include <zsmiles/gpu/dictionary.hpp>
#include <zsmiles/likwid_utils.hpp>
#include <zsmiles/utils.hpp>

namespace smiles {
  namespace cuda {
    const __device__ __constant__ gpu::node dictionary_tree_gpu[GPU_DICT_SIZE];
    const __device__ __constant__ gpu::smiles_dictionary_entry_gpu smiles_dictionary_gpu[DICT_SIZE];

    base_compressor::base_compressor() {
      CHECK_CUDA_KERNEL_ERRORS(cudaMalloc(&smiles_len_dev, SMILES_PER_DEVICE * sizeof(index_type)));
      CHECK_CUDA_KERNEL_ERRORS(cudaMalloc(&smiles_index_dev, SMILES_PER_DEVICE * sizeof(index_type)));
      CHECK_CUDA_KERNEL_ERRORS(cudaMalloc(&smiles_index_out_dev, SMILES_PER_DEVICE * sizeof(index_type)));
    };

    base_compressor::~base_compressor() {
      if (smiles_len_dev != nullptr)
        CHECK_CUDA_KERNEL_ERRORS(cudaFree(smiles_len_dev));
      if (smiles_index_dev != nullptr)
        CHECK_CUDA_KERNEL_ERRORS(cudaFree(smiles_index_dev));
      if (smiles_index_out_dev != nullptr)
        CHECK_CUDA_KERNEL_ERRORS(cudaFree(smiles_index_out_dev));
    }

    smiles_compressor::smiles_compressor() {
      smiles_host.reserve(CHAR_PER_DEVICE);
      smiles_output_host.resize(CHAR_PER_DEVICE);
      CHECK_CUDA_KERNEL_ERRORS(cudaMalloc(&smiles_dev, CHAR_PER_DEVICE / 2 * sizeof(smiles_type)));
      CHECK_CUDA_KERNEL_ERRORS(
          cudaMalloc(&score_matrix_dev,
                     MAX_SMILES_LEN * GRID_SIZE * BLOCK_SIZE * sizeof(pattern_index_type)));
      CHECK_CUDA_KERNEL_ERRORS(
          cudaMalloc(&pattern_matrix_dev,
                     MAX_SMILES_LEN * BLOCK_SIZE * GRID_SIZE * sizeof(pattern_index_type)));
      CHECK_CUDA_KERNEL_ERRORS(
          cudaMalloc(&length_matrix_dev,
                     MAX_SMILES_LEN * BLOCK_SIZE * GRID_SIZE * sizeof(pattern_index_type)));
      CHECK_CUDA_KERNEL_ERRORS(cudaMemcpyToSymbol(dictionary_tree_gpu,
                                                  gpu::build_gpu_smiles_dictionary().data(),
                                                  sizeof(gpu::node) * GPU_DICT_SIZE,
                                                  0,
                                                  cudaMemcpyHostToDevice));
      // We assume that the output smiles len does not exceed the length of the input one
      CHECK_CUDA_KERNEL_ERRORS(cudaMalloc(&smiles_output_dev, CHAR_PER_DEVICE * sizeof(smiles_type)));
    };

    smiles_compressor::~smiles_compressor() {
      if (smiles_dev != nullptr)
        CHECK_CUDA_KERNEL_ERRORS(cudaFree(smiles_dev));
      if (score_matrix_dev != nullptr)
        CHECK_CUDA_KERNEL_ERRORS(cudaFree(score_matrix_dev));
      if (pattern_matrix_dev != nullptr)
        CHECK_CUDA_KERNEL_ERRORS(cudaFree(pattern_matrix_dev));
      if (length_matrix_dev != nullptr)
        CHECK_CUDA_KERNEL_ERRORS(cudaFree(length_matrix_dev));
      if (smiles_output_dev != nullptr)
        CHECK_CUDA_KERNEL_ERRORS(cudaFree(smiles_output_dev));
    }

    smiles_decompressor::smiles_decompressor() {
      smiles_host.reserve(CHAR_PER_DEVICE / LONGEST_PATTERN);
      smiles_output_host.resize(CHAR_PER_DEVICE);
      CHECK_CUDA_KERNEL_ERRORS(
          cudaMalloc(&smiles_dev, CHAR_PER_DEVICE / LONGEST_PATTERN * sizeof(smiles_type)));
      CHECK_CUDA_KERNEL_ERRORS(cudaMalloc(&smiles_output_dev, CHAR_PER_DEVICE * sizeof(smiles_type)));
      CHECK_CUDA_KERNEL_ERRORS(cudaMemcpyToSymbol(smiles_dictionary_gpu,
                                                  gpu::build_gpu_smiles_dictionary_entries().data(),
                                                  sizeof(gpu::smiles_dictionary_entry_gpu) * DICT_SIZE,
                                                  0,
                                                  cudaMemcpyHostToDevice));
    }

    smiles_decompressor::~smiles_decompressor() {
      if (smiles_dev != nullptr)
        CHECK_CUDA_KERNEL_ERRORS(cudaFree(smiles_dev));
      if (smiles_output_dev != nullptr)
        CHECK_CUDA_KERNEL_ERRORS(cudaFree(smiles_output_dev));
    }

    __global__ void compress_gpu(const base_compressor::smiles_type* __restrict__ smiles_in,
                                 const base_compressor::index_type* __restrict__ smiles_in_index,
                                 const base_compressor::index_type* __restrict__ smiles_out_index,
                                 base_compressor::smiles_type* __restrict__ smiles_out,
                                 const base_compressor::index_type* __restrict__ smiles_len,
                                 const int num_smiles,
                                 base_compressor::pattern_index_type* __restrict__ pattern_matrix,
                                 base_compressor::pattern_index_type* __restrict__ length_matrix,
                                 base_compressor::pattern_index_type* __restrict__ score_matrix) {
      const auto threadId = threadIdx.x + blockDim.x * blockIdx.x;

      const auto stride                                     = gridDim.x * blockDim.x;
      const int score_stride                                = MAX_SMILES_LEN;
      base_compressor::pattern_index_type* score_matrix_l   = score_matrix + score_stride * threadId;
      base_compressor::pattern_index_type* pattern_matrix_l = pattern_matrix + score_stride * threadId;
      base_compressor::pattern_index_type* length_matrix_l  = length_matrix + score_stride * threadId;

      for (auto i = threadId; i < num_smiles; i += stride) {
        const base_compressor::index_type smile_len = *(smiles_len + i);
        assert(MAX_SMILES_LEN > smile_len);
        const base_compressor::smiles_type* smiles_in_l = &smiles_in[smiles_in_index[i]];
        base_compressor::smiles_type* smiles_out_l      = &smiles_out[smiles_out_index[i]];

        score_matrix_l[smile_len] = 0;

        for (auto index = static_cast<int>(smile_len - 1); index >= 0; index--) {
          score_matrix_l[index]   = score_matrix_l[index + 1] + 2;
          pattern_matrix_l[index] = 0;
          length_matrix_l[index]  = 1;
          const gpu::node* curr   = dictionary_tree_gpu;
          int curr_id             = 0;
          for (int j = 0; j < LONGEST_PATTERN && curr && j < (smile_len - index); j++) {
            const int next_i = curr->neighbor[smiles_in_l[index + j] - NOT_PRINTABLE];
            if (next_i) {
              curr    = &dictionary_tree_gpu[next_i + curr_id];
              curr_id = next_i + curr_id;
              if (curr->pattern != 0) {
                const auto next = index + j + 1;
                const auto w    = score_matrix_l[next] + 1;
                if (w < score_matrix_l[index]) {
                  score_matrix_l[index]   = w;
                  pattern_matrix_l[index] = curr->pattern;
                  length_matrix_l[index]  = j + 1;
                }
              }
            } else {
              curr = nullptr;
            }
          }
        }

        int index     = 0;
        int out_index = 0;
        while (index < smile_len) {
          if (pattern_matrix_l[index] != 0) {
            smiles_out_l[out_index] = pattern_matrix_l[index];
            ++out_index;
            index += length_matrix_l[index];
          } else {
            smiles_out_l[out_index] = smiles_dictionary_escape_char;
            ++out_index;
            smiles_out_l[out_index] = smiles_in_l[index];
            ++out_index;
            index += 1;
          }
        }
        smiles_out_l[out_index] = '\0';
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
      CHECK_CUDA_KERNEL_ERRORS(cudaMemcpy(smiles_index_out_dev,
                                          smiles_index_out.data(),
                                          smiles_index_out.size() * sizeof(index_type),
                                          cudaMemcpyHostToDevice));

      const dim3 block_dimension{BLOCK_SIZE};
      const dim3 grid_dimension{GRID_SIZE};
      need_clean_up = true;
      GPUMON_MARKER_START("Compress_CUDA");
      compress_gpu<<<grid_dimension, block_dimension>>>(smiles_dev,
                                                        smiles_index_dev,
                                                        smiles_index_out_dev,
                                                        smiles_output_dev,
                                                        smiles_len_dev,
                                                        smiles_len.size(),
                                                        pattern_matrix_dev,
                                                        length_matrix_dev,
                                                        score_matrix_dev);
      GPUMON_MARKER_STOP("Compress_CUDA");
      // Clean up
      temp_len       = smiles_len;
      temp_index_out = smiles_index_out;
      smiles_len.clear();
      smiles_index.clear();
      smiles_index_out.clear();
      smiles_host.clear();

      return;
    }

    void smiles_compressor::copy_out(std::ofstream& out_s) {
      CHECK_CUDA_ERRORS();
      CHECK_CUDA_KERNEL_ERRORS(cudaDeviceSynchronize());

      // The copy back is SYNC
      smiles_output_host.clear();
      smiles_output_host.resize((temp_index_out.back() + temp_len.back() * 2 + 1));
      CHECK_CUDA_KERNEL_ERRORS(cudaMemcpy((void*) smiles_output_host.data(),
                                          smiles_output_dev,
                                          smiles_output_host.size() * sizeof(smiles_type),
                                          cudaMemcpyDeviceToHost));
      temp_out.clear();
      temp_out.reserve(temp_len.back() * 2 + 1 + temp_index_out.back());
      // Print output
      for (int i = 0; i < temp_len.size(); i++) {
        temp_out.append(&smiles_output_host.data()[temp_index_out[i]]);
        temp_out += '\n';
      }

      out_s << temp_out;

      need_clean_up = false;

      return;
    }

    void smiles_decompressor::copy_out(std::ofstream& out_s) {
      CHECK_CUDA_ERRORS();
      CHECK_CUDA_KERNEL_ERRORS(cudaDeviceSynchronize());

      // The copy back is SYNC
      smiles_output_host.clear();
      smiles_output_host.resize((temp_index_out.back() + temp_len.back() * LONGEST_PATTERN + 1));
      CHECK_CUDA_KERNEL_ERRORS(cudaMemcpy((void*) smiles_output_host.data(),
                                          smiles_output_dev,
                                          smiles_output_host.size() * sizeof(smiles_type),
                                          cudaMemcpyDeviceToHost));
      // Print output
      temp_out.clear();
      temp_out.reserve((temp_index_out.back() + temp_len.back() * LONGEST_PATTERN + 1));
      for (int i = 0; i < temp_len.size(); i++) {
        temp_out.append(&smiles_output_host[temp_index_out[i]]);
        temp_out += '\n';
      }
      out_s << temp_out;

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
                                   const base_compressor::index_type* __restrict__ smiles_out_index,
                                   base_compressor::smiles_type* __restrict__ smiles_out,
                                   const base_compressor::index_type* __restrict__ smiles_len,
                                   const int num_smiles) {
      const int threadId     = threadIdx.x;
      const int blockId      = blockIdx.x;
      const int stride_smile = gridDim.x;
      const int stride       = blockDim.x;

      const base_compressor::index_type* smiles_len_l = smiles_len + blockId;
      for (int id = blockId; id < num_smiles; id += stride_smile, smiles_len_l += stride_smile) {
        const base_compressor::index_type smile_len     = *smiles_len_l;
        unsigned long last_index                        = 0;
        const base_compressor::smiles_type* smiles_in_l = smiles_in + smiles_in_index[id];
        base_compressor::smiles_type* smiles_out_l      = smiles_out + smiles_out_index[id];

        int is_previous_escape = 0;
        for (int i = threadId; i < smile_len; i += stride) {
          unsigned int mask                          = __activemask();
          const base_compressor::smiles_type smile_c = smiles_in_l[i];
          const int is_escape                        = smile_c == smiles_dictionary_escape_char;
          // Maybe for 206 there is a smarter way to do this
          const auto is_previous_escape_t = __shfl_up_sync(mask, is_escape, 1);
          if (threadId % WARP_SIZE != 0)
            is_previous_escape = is_previous_escape_t;
          const auto find_index = static_cast<unsigned char>(smile_c);
          const auto find_pattern_length =
              is_previous_escape ? 1 : (is_escape ? 0 : smiles_dictionary_gpu[find_index].size);
          const auto find_pattern =
              is_previous_escape ? &smile_c : (is_escape ? "" : smiles_dictionary_gpu[find_index].pattern);

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

          memcpy(&smiles_out_l[index], find_pattern, find_pattern_length);

          unsigned int v = mask;
          unsigned int last_thread;
          for (last_thread = 0; v; last_thread++) {
            v &= v - 1; // clear the least significant bit set
          }
          last_index         = __shfl_sync(mask, index + find_pattern_length, last_thread - 1);
          is_previous_escape = __shfl_sync(mask, is_escape, last_thread - 1);
        }
        assert(last_index < MAX_SMILES_LEN);
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
      CHECK_CUDA_KERNEL_ERRORS(cudaMemcpy(smiles_index_out_dev,
                                          smiles_index_out.data(),
                                          smiles_index_out.size() * sizeof(index_type),
                                          cudaMemcpyHostToDevice));

      const dim3 block_dimension{BLOCK_SIZE};
      const dim3 grid_dimension{GRID_SIZE};
      need_clean_up = true;
      GPUMON_MARKER_START("Decompress_CUDA");
      decompress_gpu<<<grid_dimension, block_dimension>>>(smiles_dev,
                                                          smiles_index_dev,
                                                          smiles_index_out_dev,
                                                          smiles_output_dev,
                                                          smiles_len_dev,
                                                          smiles_len.size());
      GPUMON_MARKER_STOP("Decompress_CUDA");

      // Clean up
      temp_len       = smiles_len;
      temp_index_out = smiles_index_out;
      smiles_len.clear();
      smiles_index.clear();
      smiles_index_out.clear();
      smiles_host.clear();

      return;
    }
  } // namespace cuda
} // namespace smiles
