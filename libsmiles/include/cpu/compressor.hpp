#pragma once

#include <fstream>
#include <string>
#include <string_view>
#include <vector>

namespace smiles {
  namespace cpu {
    class smiles_compressor {
      using char_type = std::string::value_type;

      std::string output_string; // we need this buffer to minimize memory allocation/deallocation
      std::vector<std::size_t> min_path_index;
      std::vector<std::string::size_type> min_path_score;

    public:
      // pre-allocate memory to avoid useless memory operations
      smiles_compressor(void) {
        output_string.reserve(2000);
        min_path_index.reserve(2000);
        min_path_score.reserve(2000);
      }
      void operator()(const std::string_view& plain_description, std::ofstream&);
    };

    class smiles_decompressor {
      using char_type = std::string::value_type;

      std::string scratchpad; // we need this buffer to minimize memory allocation/deallocation

    public:
      smiles_decompressor(void) { scratchpad.reserve(2000); } // pre-allocate
      void operator()(const std::string_view& compressed_description, std::ofstream&);
    };
  } // namespace cpu
} // namespace smiles
