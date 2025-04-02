#ifndef ZSMILES_KOKKOS_COMPRESSOR_HPP
#define ZSMILES_KOKKOS_COMPRESSOR_HPP

#include <vector>
#include <string>
#include <fstream>

namespace smiles { 
    namespace kokkos {  // Sintassi C++11/C++17 corretta

        class smiles_compressor {
        public:
            void compress(std::ifstream& o_file);
            void clean_up(std::ofstream& o_file);
        };

        class smiles_decompressor {
        public:
            void decompress(std::ifstream&, std::ofstream& o_file);
            void clean_up(std::ofstream& o_file);
        };

    } // namespace kokkos
} // namespace smiles

#endif