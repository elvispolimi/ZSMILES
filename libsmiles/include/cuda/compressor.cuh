#pragma once

#include "cuda/dictionary.cuh"
#include "cuda/node.cuh"

#include <string>
#include <string_view>
#include <vector>

namespace smiles {
  namespace cuda {
    class smiles_compressor {
    public:
      using cost_type          = uint_fast16_t;
      using index_type         = int;
      using pattern_index_type = uint_fast8_t;
      using smiles_type        = std::string::value_type;
      // pre-allocate memory to avoid useless memory operations
      smiles_compressor();
      ~smiles_compressor();
      void operator()(const std::string_view& plain_description, std::ofstream& out_s);
      void clean_up(std::ofstream& out_s);

    private:
      std::string smiles_host;
      smiles_type* smiles_dev;
      std::vector<index_type> smiles_len;
      index_type* smiles_len_dev;

      pattern_index_type* match_matrix_dev;
      pattern_index_type* dijkstra_matrix_dev;

      std::string smiles_output_host;
      smiles_type* smiles_output_dev;

      void compute_host(std::ofstream& out_s);
    };

    class smiles_decompressor {
    public:
      smiles_decompressor() {} // pre-allocate
      std::string_view operator()(const std::string_view& compressed_description);
    };
  } // namespace cuda
} // namespace smiles