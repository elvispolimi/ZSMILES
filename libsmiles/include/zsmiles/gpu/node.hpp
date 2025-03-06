#pragma once

#include <cstring>
#include <zsmiles/compression_dictionary.hpp>
#include <zsmiles/utils.hpp>

namespace smiles {
  namespace gpu {
    class node {
    public:
      using node_letter   = char;
      using node_neighbor = unsigned short;

      node_letter pattern = 0;
      node_letter letter;
      // Sometimes it could be necessary to use short
      char neighbor[PRINTABLE_CHAR];

      constexpr node(node_letter letter): letter(letter), neighbor{} {
        constexpr_for<0, PRINTABLE_CHAR, 1>([&](auto index) { neighbor[index] = 0; });
      };
      constexpr node(node_letter letter, node_letter pattern): pattern(pattern), letter(letter), neighbor{} {
        constexpr_for<0, PRINTABLE_CHAR, 1>([&](auto index) { neighbor[index] = 0; });
      };
      constexpr node(): letter(), neighbor{} {
        constexpr_for<0, PRINTABLE_CHAR, 1>([&](auto index) { neighbor[index] = 0; });
      };
    };
  } // namespace gpu
} // namespace smiles
