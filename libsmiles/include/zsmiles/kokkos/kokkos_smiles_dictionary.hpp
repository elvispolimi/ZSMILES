#pragma once
#include <Kokkos_Core.hpp>
#include <zsmiles/kokkos/kokkos_decompression_node.hpp>
#include <zsmiles/compression_dictionary.hpp>
#include <array>
#include <string>

namespace smiles {
  namespace kokkos {

    Kokkos::View<decompression_node*> build_decompression_dictionary() {
      Kokkos::View<decompression_node*> dict("decompression_dictionary", DICT_SIZE);
      auto n = Kokkos::create_mirror_view(dict);

      for(int i = 0; i < DICT_SIZE; ++i) {
          n(i).length = SMILES_DICTIONARY[i].size;
          for(int j = 0; j < SMILES_DICTIONARY[i].size; ++j) {
              n(i).pattern[j] = SMILES_DICTIONARY[i].pattern[j];
          }
      }
      Kokkos::deep_copy(dict, n);
      return dict;
    }
  }
} // namespace smiles
