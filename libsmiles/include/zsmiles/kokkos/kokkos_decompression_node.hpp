#pragma once

#include <Kokkos_Core.hpp>
#include <zsmiles/compression_dictionary.hpp>

namespace smiles {
namespace kokkos {

struct decompression_node {
    size_t length;
    char pattern[LONGEST_PATTERN];
};

}  // namespace kokkos
}  // namespace smiles