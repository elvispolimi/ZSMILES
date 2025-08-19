#pragma once
#include <string>
#include <vector>
#include <ostream>
#include <fstream>
#include <iostream>
#include <zsmiles/kokkos/kokkos_smiles_dictionary.hpp>
#include <Kokkos_Core.hpp>
#include <zsmiles/gpu/knobs.hpp>
#include <zsmiles/compression_dictionary.hpp>

namespace smiles {
namespace kokkos {

class smiles_decompressor {
public:
  using smiles_type = char;
  using index_type  = size_t;

  Kokkos::View<index_type*, Kokkos::CudaUVMSpace> smiles_index_out;
  Kokkos::View<index_type*, Kokkos::CudaUVMSpace> smiles_len;
  Kokkos::View<index_type*, Kokkos::CudaUVMSpace> smiles_index;
  Kokkos::View<smiles_type*, Kokkos::CudaUVMSpace> smiles_host;
  Kokkos::View<smiles_type*, Kokkos::CudaUVMSpace> smiles_out;

  // Contatori per simulare push_back/append
  size_t smiles_count = 0;      // quante SMILES abbiamo aggiunto
  size_t smiles_host_index  = 0;      // prossimo indice disponibile in smiles_host

  smiles_decompressor(size_t max_chars, size_t max_smiles)
    : smiles_index_out("smiles_index_out", max_smiles),
      smiles_len("smiles_len", max_smiles),
      smiles_index("smiles_index", max_smiles),
      smiles_host("smiles_host", max_chars),
      smiles_out("smiles_out", max_chars) {
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
  std::string smiles_host;
  std::vector<size_t> smiles_index;
  std::vector<size_t> smiles_index_out;
  std::vector<size_t> smiles_len;

  smiles_compressor();
  ~smiles_compressor();

  void compress(std::ofstream& out_s);
  void clean_up(std::ofstream& out_s);
  void test();
};

} // namespace kokkos
} // namespace smiles
