#include "likwid-marker.h"

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
          cudaMalloc(&match_matrix_dev,
                     MAX_SMILES_LEN * NUM_WORK_GROUP * LONGEST_PATTERN * sizeof(pattern_index_type)));
      CHECK_CUDA_KERNEL_ERRORS(
          cudaMalloc(&dijkstra_matrix_dev,
                     MAX_SMILES_LEN * NUM_WORK_GROUP * LONGEST_PATTERN * sizeof(pattern_index_type)));
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
      if (match_matrix_dev != nullptr)
        CHECK_CUDA_KERNEL_ERRORS(cudaFree(match_matrix_dev));
      if (dijkstra_matrix_dev != nullptr)
        CHECK_CUDA_KERNEL_ERRORS(cudaFree(dijkstra_matrix_dev));
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
                                 base_compressor::pattern_index_type* __restrict__ match_matrix,
                                 base_compressor::pattern_index_type* __restrict__ dijkstra_matrix) {
      const int threadId     = threadIdx.x % WORK_GROUP_SIZE;
      const int intraBlockId = threadIdx.x / WORK_GROUP_SIZE;
      const int blockId      = blockIdx.x * NUM_WORK_GROUP_BLOCK + intraBlockId;
      const int stride_smile = gridDim.x * NUM_WORK_GROUP_BLOCK;

      const int stride        = WORK_GROUP_SIZE;
      const int matrix_offset = MAX_SMILES_LEN * LONGEST_PATTERN * blockId;

      const int pattern_id = threadId;

      const base_compressor::index_type* smiles_len_l        = smiles_len + blockId;
      base_compressor::pattern_index_type* match_matrix_l    = match_matrix + matrix_offset;
      base_compressor::pattern_index_type* dijkstra_matrix_l = dijkstra_matrix + matrix_offset;
      for (int id = blockId; id < num_smiles; id += stride_smile, smiles_len_l += stride_smile) {
        const base_compressor::index_type smile_len     = *smiles_len_l;
        const base_compressor::smiles_type* smiles_in_l = &smiles_in[smiles_in_index[id]];
        base_compressor::smiles_type* smiles_out_l      = &smiles_out[smiles_out_index[id]];
        // if(threadId==0) printf("Id %d with output id %lu\n",id,smiles_out_index[id]);

        assert(smile_len < MAX_SMILES_LEN);
        // for (int i = threadId; i < smile_len; i += stride) smiles_s[i] = smiles_in_l[i];
        const base_compressor::smiles_type* smiles_s = smiles_in_l;
        __syncwarp();
        for (int j = 0; j <= smile_len; j += 1)
          if (pattern_id < LONGEST_PATTERN)
            match_matrix_l[LONGEST_PATTERN * j + pattern_id] = 0;
        __syncwarp();
        // For each position in the input string
        for (int i = threadId; i < smile_len; i += stride) {
          const gpu::node* curr = dictionary_tree_gpu;
          int curr_id           = 0;
#pragma unroll 8
          for (int j = 0; j < LONGEST_PATTERN && curr && j < (smile_len - i); j++) {
            const int next_i = curr->neighbor[smiles_s[i + j] - NOT_PRINTABLE];
            if (next_i) {
              curr    = &dictionary_tree_gpu[next_i + curr_id];
              curr_id = next_i + curr_id;
              if (curr->pattern != 0) {
                // Between braches to get the legth of the match
                match_matrix_l[LONGEST_PATTERN * (i + (j + 1)) + j] = curr->pattern;
                // if ((LONGEST_PATTERN * (i + (j + 1)) + j + matrix_offset) ==
                //     (MAX_SMILES_LEN * LONGEST_PATTERN * 220 + 392))
                //   printf("(i,j) %d %d-> (%d,%d) with value %d on %d\n",
                //          threadId,blockId,i,
                //          j,
                //          match_matrix[matrix_offset+LONGEST_PATTERN * (i + (j + 1)) + j],
                //          LONGEST_PATTERN * (i + (j + 1)) + j);
              }
            } else {
              curr = nullptr;
            }
          }
        }

        __syncwarp();
        // if(id==220)printf("Check %d (%d,%d)\n",match_matrix[matrix_offset+392],threadId,blockId);

        for (int j = 0; j <= smile_len; j += 1)
          if (pattern_id < LONGEST_PATTERN)
            dijkstra_matrix_l[j * LONGEST_PATTERN + pattern_id] =
                std::numeric_limits<base_compressor::pattern_index_type>().max();

        base_compressor::pattern_index_type const * match_matrix_tmp =
            &match_matrix_l[LONGEST_PATTERN * smile_len];
        base_compressor::pattern_index_type* costs_matrix_tmp =
            &dijkstra_matrix_l[LONGEST_PATTERN * smile_len];

        // Set the termination stage since later we skip it with l<smile_len check
        if (threadId % stride == 0) {
          costs_matrix_tmp[0] = 0;
          costs_matrix_tmp[1] = 0;
          costs_matrix_tmp[2] = 0;
        }

        __syncwarp();

        // TODO change the index of match_matrix

        // Skip the first one which is trivial to select the smallest value
        for (int l = smile_len; l >= 0;
             l--, match_matrix_tmp -= LONGEST_PATTERN, costs_matrix_tmp -= LONGEST_PATTERN) {
          if (l < smile_len) {
            // Compute the best for the next one
            base_compressor::pattern_index_type best_cost =
                pattern_id < LONGEST_PATTERN
                    ? costs_matrix_tmp[pattern_id]
                    : std::numeric_limits<base_compressor::pattern_index_type>().max();
            // printf("New %d and I have %d\n", id, best_cost);
            // Reduce
            // for (int offset = stride / 2; offset > 0; offset /= 2) {
            //   // Get the value and tid from the higher lane
            //   const base_compressor::pattern_index_type next_value =
            //       __shfl_down_sync(FULL_MASK, best_cost, offset);
            //   const int next_tid = __shfl_down_sync(FULL_MASK, threadId, offset);

            //   // Update the value and tid if the next value is larger
            //   if (best_cost > next_value) {
            //     best_cost  = next_value;
            //     best_index = next_tid;
            //   }
            // }
            // TODO check what this doesssss
            constexpr auto Everyone = -1u; // a mask that includes all threads in a warp
            const auto minval =
                __reduce_min_sync(Everyone, best_cost); // value is the local variable of each thread
            // TODO maybe a problem if they are equal????
            const auto minmask = __ballot_sync(
                Everyone,
                best_cost == minval); // a mask that indicates which threads have the minimum value
            const auto minpos = __ffs(minmask) - 1; // the position of the first thread with the minimum value
            if (threadId % stride == 0) {
              if (minval == std::numeric_limits<base_compressor::pattern_index_type>().max()) {
                costs_matrix_tmp[0] = costs_matrix_tmp[LONGEST_PATTERN] + 2;
                costs_matrix_tmp[1] = LONGEST_PATTERN;
              } else {
                costs_matrix_tmp[0] = minval;
                costs_matrix_tmp[1] = minpos;
                costs_matrix_tmp[2] = match_matrix_tmp[LONGEST_PATTERN * (minpos + 1) + minpos];
              }
              // printf("BEST %d and I have cost %d (mine %d) index %d pattern %d mask %d \n",
              //        id,
              //        minval,
              //        best_cost,
              //        minpos,
              //        costs_matrix_tmp[2],
              //        minmask);
            }
            __syncwarp();
          }
          // TODO check maybe l>0 is not needed
          if (l > 0 && pattern_id < LONGEST_PATTERN) {
            // Then update the next values
            // if ((LONGEST_PATTERN * l + pattern_id + matrix_offset) ==
            //     (MAX_SMILES_LEN * LONGEST_PATTERN * 220 + 392))
            //   printf("(pattern_id) -> (%d) with value %d\n", l, match_matrix[LONGEST_PATTERN * l + pattern_id + matrix_offset]);
            if (*(match_matrix_tmp + pattern_id)) {
              costs_matrix_tmp[-(LONGEST_PATTERN * (pattern_id + 1)) + pattern_id] = costs_matrix_tmp[0] + 1;
            }
          }
          __syncwarp();
        }
        // TODO add first iteration -> it should be fine with previous loop condition
        __syncwarp();
        if (threadId % stride == 0) {
          // printf("I'm blockId %d this is id %d -> %s\n",blockId,id,smiles_out_l[oo]);
          int o            = 0;
          costs_matrix_tmp = dijkstra_matrix_l;
          for (int l = 0; l < smile_len; l++) {
            // TODO check this condition
            if (costs_matrix_tmp[1] == LONGEST_PATTERN) {
              smiles_out_l[o] = smiles_dictionary_escape_char;
              o++;
              smiles_out_l[o] = smiles_s[l];
              o++;
              costs_matrix_tmp += LONGEST_PATTERN;
            } else {
              smiles_out_l[o] = static_cast<base_compressor::smiles_type>(costs_matrix_tmp[2]);
              o++;
              l += costs_matrix_tmp[1];
              costs_matrix_tmp += LONGEST_PATTERN * (costs_matrix_tmp[1] + 1);
            }
          }
          // TODO to verify if it is required to have also the +1 for the terminator
          assert(o < smile_len * 2 + 1);
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
      CHECK_CUDA_KERNEL_ERRORS(cudaMemcpy(smiles_index_out_dev,
                                          smiles_index_out.data(),
                                          smiles_index_out.size() * sizeof(index_type),
                                          cudaMemcpyHostToDevice));

      const dim3 block_dimension{BLOCK_SIZE};
      const dim3 grid_dimension{GRID_SIZE};
      need_clean_up = true;
      NVMON_MARKER_START("Compress_CUDA");
      compress_gpu<<<grid_dimension, block_dimension>>>(smiles_dev,
                                                        smiles_index_dev,
                                                        smiles_index_out_dev,
                                                        smiles_output_dev,
                                                        smiles_len_dev,
                                                        smiles_len.size(),
                                                        match_matrix_dev,
                                                        dijkstra_matrix_dev);
      NVMON_MARKER_STOP("Compress_CUDA");
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
      NVMON_MARKER_START("Decompress_CUDA");
      decompress_gpu<<<grid_dimension, block_dimension>>>(smiles_dev,
                                                          smiles_index_dev,
                                                          smiles_index_out_dev,
                                                          smiles_output_dev,
                                                          smiles_len_dev,
                                                          smiles_len.size());
      NVMON_MARKER_STOP("Decompress_CUDA");

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
