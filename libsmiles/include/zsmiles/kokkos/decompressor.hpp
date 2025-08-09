#pragma once
#include <string>
#include <vector>
#include <ostream>
#include <fstream>

namespace smiles {
namespace kokkos {

class smiles_decompressor {
public:
  std::string smiles_host;
  std::vector<size_t> smiles_index;
  std::vector<size_t> smiles_index_out;
  std::vector<size_t> smiles_len;
  

  smiles_decompressor();
  ~smiles_decompressor();

  void decompress(std::ofstream& out_s);
  void clean_up(std::ofstream& out_s);
  void copy_out(std::ofstream& out_s);
private:
  Dictionary dictionary;
};

} // namespace kokkos
} // namespace smiles
