#pragma once

#include <array>
#include "dictionary_graph.hpp"
#include <string>

namespace smiles {

  struct smiles_dictionary_entry {
    const char* pattern;
    std::string::size_type size;
  };

  extern const std::array<smiles_dictionary_entry, 256> SMILES_DICTIONARY;

  static constexpr auto smiles_dictionary_escape_char = std::string::value_type{' '};

  dictionary_tree_type build_current_smiles_dictionary(void);

} // namespace smiles
