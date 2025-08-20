#pragma once

#include <Kokkos_Core.hpp>
#include <zsmiles/compression_dictionary.hpp>

namespace smiles {
  namespace kokkos {
    struct node {
      using node_letter   = char;
      using node_neighbor = unsigned short;

      node_letter pattern = 0;
      node_letter letter;
      char neighbor[PRINTABLE_CHAR];

      // Costruttore predefinito
      KOKKOS_INLINE_FUNCTION
      node() : pattern(0), letter(' ') {
        for (int i = 0; i < PRINTABLE_CHAR; i++) {
          neighbor[i] = 0;   // inizializza tutti i vicini a -1
        }
      }

      // Costruttore con un solo parametro (letter)
      KOKKOS_INLINE_FUNCTION
      node(node_letter letter) : pattern(0), letter(letter) {
        for (int i = 0; i < PRINTABLE_CHAR; i++) {
          neighbor[i] = 0;
        }
      }

      // Costruttore con due parametri (letter e pattern)
      KOKKOS_INLINE_FUNCTION
      node(node_letter letter, node_letter pattern) : pattern(pattern), letter(letter) {
        for (int i = 0; i < PRINTABLE_CHAR; i++) {
          neighbor[i] = 0;
        }
      }
    };
  } // namespace kokkos
} // namespace smiles