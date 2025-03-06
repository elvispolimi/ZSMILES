#pragma once

#include <string>

namespace smiles {
  class ring{
    std::string ring_smile;
    int next_available = -1;
  };
}