#pragma once

#include "compressor_interface.hpp"

#include <string>
#include <vector>

namespace smiles {
  namespace cuda {
    class smiles_compressor: public compressor_interface {
      using char_type = std::string::value_type;

    public:
      // pre-allocate memory to avoid useless memory operations
      smiles_compressor() {}
      std::string_view operator()(const std::string_view& plain_description);
    };

    class smiles_decompressor: public decompressor_interface {
      using char_type = std::string::value_type;

    public:
      smiles_decompressor() {} // pre-allocate
      std::string_view operator()(const std::string_view& compressed_description);
    };
  } // namespace cuda
} // namespace smiles