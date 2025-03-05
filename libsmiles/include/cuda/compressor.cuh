#pragma once

#include "gpu/dictionary.hpp"
#include "gpu/node.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <gpu/knobs.hpp>

namespace smiles {
  namespace cuda {
    class base_compressor {
    public:
      using cost_type          = uint_fast16_t;
      using index_type         = size_t;
      using pattern_index_type = uint_fast8_t;
      using smiles_type        = std::string::value_type;
      // pre-allocate memory to avoid useless memory operations
      base_compressor();
      ~base_compressor();

      std::string smiles_host;
      std::vector<index_type> smiles_index;
      std::vector<index_type> smiles_index_out;
      std::vector<index_type> smiles_len;

      smiles_type* smiles_dev;
      index_type* smiles_index_dev;
      index_type* smiles_index_out_dev;
      index_type* smiles_len_dev;

      std::string smiles_output_host;
      smiles_type* smiles_output_dev;
      bool need_clean_up = false;

      std::string temp_out;
      std::vector<index_type> temp_len;
      std::vector<index_type> temp_index_out;
    };
    class smiles_compressor: public base_compressor {
    public:
      void compress(std::ofstream& out_s);
      void clean_up(std::ofstream& out_s);
      void copy_out(std::ofstream& out_s);

      smiles_compressor();
      ~smiles_compressor();

    private:
      pattern_index_type* match_matrix_dev;
      pattern_index_type* dijkstra_matrix_dev;
    };

    class smiles_decompressor: public base_compressor {
    public:
      smiles_decompressor();
      ~smiles_decompressor();
      void decompress(std::ofstream& out_s);
      void clean_up(std::ofstream& out_s);
      void copy_out(std::ofstream& out_s);
    };
  } // namespace cuda
} // namespace smiles
