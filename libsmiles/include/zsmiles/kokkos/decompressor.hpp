#pragma once
#include <string>
#include <vector>
#include <ostream>
#include <fstream>
#include <iostream>
#include <Kokkos_Core.hpp>
#include <zsmiles/gpu/knobs.hpp>
#include <zsmiles/compression_dictionary.hpp>

namespace smiles {
namespace kokkos {

class smiles_decompressor {
public:
  using smiles_type = char;
  using index_type  = size_t;

  Kokkos::View<index_type*, Kokkos::DefaultExecutionSpace::memory_space> d_smiles_index_out;
  Kokkos::View<index_type*, Kokkos::DefaultExecutionSpace::memory_space> d_smiles_len;
  Kokkos::View<index_type*, Kokkos::DefaultExecutionSpace::memory_space> d_smiles_index;
  Kokkos::View<smiles_type*, Kokkos::DefaultExecutionSpace::memory_space> d_smiles_host;
  Kokkos::View<smiles_type*, Kokkos::DefaultExecutionSpace::memory_space> d_smiles_out;

  Kokkos::View<index_type*, Kokkos::HostSpace> h_smiles_index_out;
  Kokkos::View<index_type*, Kokkos::HostSpace> h_smiles_len;
  Kokkos::View<index_type*, Kokkos::HostSpace> h_smiles_index;
  Kokkos::View<smiles_type*, Kokkos::HostSpace> h_smiles_host;
  Kokkos::View<smiles_type*, Kokkos::HostSpace> h_smiles_out;

  // Contatori per simulare push_back/append
  size_t smiles_count = 0;      // quante SMILES abbiamo aggiunto
  size_t smiles_host_index  = 0;      // prossimo indice disponibile in smiles_host


  smiles_decompressor(size_t max_chars, size_t max_smiles)
    : d_smiles_index_out(Kokkos::view_alloc(Kokkos::WithoutInitializing, "smiles_index_out"), max_smiles),
      d_smiles_len(Kokkos::view_alloc(Kokkos::WithoutInitializing, "smiles_len"), max_smiles),
      d_smiles_index(Kokkos::view_alloc(Kokkos::WithoutInitializing, "smiles_index"), max_smiles),
      d_smiles_host(Kokkos::view_alloc(Kokkos::WithoutInitializing, "smiles_host"), max_chars),
      d_smiles_out(Kokkos::view_alloc(Kokkos::WithoutInitializing, "smiles_out"), max_chars),
      h_smiles_index_out(Kokkos::create_mirror_view(d_smiles_index_out)),
      h_smiles_len(Kokkos::create_mirror_view(d_smiles_len)),
      h_smiles_index(Kokkos::create_mirror_view(d_smiles_index)),
      h_smiles_host(Kokkos::create_mirror_view(d_smiles_host)),
      h_smiles_out(Kokkos::create_mirror_view(d_smiles_out)) {
      std::cout << "[DEBUG] SMILES decompressor initialized with max_chars: " 
            << max_chars << " and max_smiles: " << max_smiles << std::endl;
  }

  //void add_smiles(const std::string& smiles);
  void decompress(std::ofstream& out_s);
  void copy_out(std::ofstream& out_s);
  void clean_up(std::ofstream& out_s);
};


class smiles_compressor {
public:
  using smiles_type = char;
  using index_type  = size_t;

  Kokkos::View<index_type*, Kokkos::DefaultExecutionSpace::memory_space> d_smiles_index_out;
  Kokkos::View<index_type*, Kokkos::DefaultExecutionSpace::memory_space> d_smiles_len;
  Kokkos::View<index_type*, Kokkos::DefaultExecutionSpace::memory_space> d_smiles_index;
  Kokkos::View<smiles_type*, Kokkos::DefaultExecutionSpace::memory_space> d_smiles_host;
  Kokkos::View<smiles_type*, Kokkos::DefaultExecutionSpace::memory_space> d_smiles_out;
  Kokkos::View<index_type*, Kokkos::DefaultExecutionSpace::memory_space> d_smiles_out_len;

  Kokkos::View<index_type*, Kokkos::HostSpace> h_smiles_index;
  Kokkos::View<index_type*, Kokkos::HostSpace> h_smiles_index_out;
  Kokkos::View<index_type*, Kokkos::HostSpace> h_smiles_len;
  Kokkos::View<smiles_type*, Kokkos::HostSpace> h_smiles_host;
  Kokkos::View<smiles_type*, Kokkos::HostSpace> h_smiles_out;
  Kokkos::View<index_type*, Kokkos::HostSpace> h_smiles_out_len;

  // Contatori per simulare push_back/append
  size_t smiles_count = 0;      // quante SMILES abbiamo aggiunto
  size_t smiles_host_index  = 0;      // prossimo indice disponibile in smiles_host

  smiles_compressor()
    : d_smiles_index_out(Kokkos::view_alloc(Kokkos::WithoutInitializing, "smiles_index_out"), SMILES_PER_DEVICE),
      d_smiles_len(Kokkos::view_alloc(Kokkos::WithoutInitializing, "smiles_len"), SMILES_PER_DEVICE),
      d_smiles_index(Kokkos::view_alloc(Kokkos::WithoutInitializing, "smiles_index"), SMILES_PER_DEVICE),
      d_smiles_host(Kokkos::view_alloc(Kokkos::WithoutInitializing, "smiles_host"), CHAR_PER_DEVICE),
      d_smiles_out(Kokkos::view_alloc(Kokkos::WithoutInitializing, "smiles_out"), CHAR_PER_DEVICE),
      d_smiles_out_len(Kokkos::view_alloc(Kokkos::WithoutInitializing, "smiles_out_len"), SMILES_PER_DEVICE),
      h_smiles_index_out(Kokkos::create_mirror_view(d_smiles_index_out)),
      h_smiles_len(Kokkos::create_mirror_view(d_smiles_len)),
      h_smiles_index(Kokkos::create_mirror_view(d_smiles_index)),
      h_smiles_host(Kokkos::create_mirror_view(d_smiles_host)),
      h_smiles_out(Kokkos::create_mirror_view(d_smiles_out)),
      h_smiles_out_len(Kokkos::create_mirror_view(d_smiles_out_len)),
      pattern_matrix(Kokkos::view_alloc(Kokkos::WithoutInitializing, "pattern_matrix_dev"), SMILES_PER_DEVICE, 100),
      length_matrix(Kokkos::view_alloc(Kokkos::WithoutInitializing, "length_matrix_dev"), SMILES_PER_DEVICE, 100),
      score_matrix(Kokkos::view_alloc(Kokkos::WithoutInitializing, "score_matrix_dev"), SMILES_PER_DEVICE, 100) {
  std::cout << "[DEBUG] SMILES compressor initialized with max_chars: " 
            << CHAR_PER_DEVICE << " and max_smiles: " << SMILES_PER_DEVICE << std::endl;
}

  void compress(std::ofstream& out_s);
  void clean_up(std::ofstream& out_s);
  void test();

private:
  Kokkos::View<uint_fast8_t**, Kokkos::DefaultExecutionSpace::memory_space> pattern_matrix;
  Kokkos::View<uint_fast8_t**, Kokkos::DefaultExecutionSpace::memory_space> length_matrix;
  Kokkos::View<uint_fast8_t**, Kokkos::DefaultExecutionSpace::memory_space> score_matrix;
};

} // namespace kokkos
} // namespace smiles
