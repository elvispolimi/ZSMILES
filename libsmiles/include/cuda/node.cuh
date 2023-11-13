#pragma once

#include "compression_dictionary.hpp"

#include <cstdint>

namespace smiles {
  namespace cuda {
    class node {
    public:
      using node_letter   = char;
      using node_neighbor = unsigned short;

      node_letter pattern = -1;
      node_letter letter;
      char neighbor[PRINTABLE_CHAR];

      constexpr node(node_letter letter): letter(letter), neighbor{0} {};
      constexpr node(node_letter letter, node_letter pattern)
          : pattern(pattern), letter(letter), neighbor{0} {};
      constexpr node(): letter(), neighbor{0} {};
    };
  } // namespace cuda
} // namespace smiles