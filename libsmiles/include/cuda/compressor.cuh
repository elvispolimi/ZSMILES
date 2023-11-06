#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace smiles {
  namespace cuda {
    class smiles_compressor {
    public:
      using cost_type = uint_fast16_t;
      using index_type = uint_fast16_t;
      using pattern_index_type = uint_fast8_t;
      using smiles_type = std::string::value_type;
      // pre-allocate memory to avoid useless memory operations
      smiles_compressor();
      ~smiles_compressor();
      void operator()(const std::string_view& plain_description, std::ofstream& out_s);
      void clean_up(std::ofstream& out_s);

    private:
      std::string smiles_host;
      smiles_type* smiles_dev;
      std::vector<index_type> smiles_start;
      index_type* smiles_start_dev;
      // TODO maybe compact the next two into a single data structure, with the same data structure of the next two, maybe placing it inside the dictionary
      cost_type* shortest_path_dev;
      pattern_index_type* shortest_path_index_dev;
      // TODO maybe compact the next two into a single data structure
      cost_type* costs_dev;
      pattern_index_type* costs_index_dev;

      std::string smiles_output_host;
      smiles_type* smiles_output_dev;

      void compute_host(std::ofstream& out_s);
    };

    class smiles_decompressor{
    public:
      smiles_decompressor() {} // pre-allocate
      std::string_view operator()(const std::string_view& compressed_description);
    };
  } // namespace cuda
} // namespace smiles