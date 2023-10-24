#include "cuda/compressor.cuh"

namespace smiles{
  namespace cuda{
    // we model the problem of SMILES compression as choosing the minimum path between the first character and the
    // last one. The cost of each path is the number of character that we need to produce in the output. We solve
    // this problem using dijkstra and we use a suppor tree to perform pattern matching.
    std::string_view smiles_compressor::operator()(const std::string_view& plain_description) {
      return {};
    }

    std::string_view smiles_decompressor::operator()(const std::string_view& compressed_description) {
      // decompressing a SMILES is really just a look up on the SMILES_DICTIONARY. We just need to pay attention
      // when the compressed SMILES has excaped something
      // NOTE: we need to start from a clean string
      // TODO
      return {};
    }
  }
}