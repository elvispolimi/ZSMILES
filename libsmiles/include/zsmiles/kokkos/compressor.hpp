#ifndef ZSMILES_KOKKOS_COMPRESSOR_HPP
#define ZSMILES_KOKKOS_COMPRESSOR_HPP

#include <zsmiles/compression_dictionary.hpp>
#include <fstream>

namespace smiles { 
    namespace kokkos {

        class smiles_compressor {
        public:
            void compress(std::ifstream& o_file);
            void clean_up(std::ofstream& o_file);
        };

        class smiles_decompressor {
        public:
            void decompress(std::ifstream& i_file, std::ofstream& o_file);
            void clean_up(std::ofstream& o_file);
        };

    } // namespace kokkos
} // namespace smiles

#endif