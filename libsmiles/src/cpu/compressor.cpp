#include "cpu/compressor.hpp"

#include "compression_dictionary.hpp"
#include "cpu/dictionary.hpp"
#include <algorithm>
#include <boost/graph/adjacency_list.hpp>
#include <fstream>
#include <limits>

namespace smiles {
  namespace cpu {

    // we model the problem of SMILES compression as choosing the minimum path between the first character and the
    // last one. The cost of each path is the number of character that we need to produce in the output. We solve
    // this problem using dijkstra and we use a suppor tree to perform pattern matching.
    void smiles_compressor::operator()(const std::string_view& plain_description, std::ofstream& out_s) {
      using index_type = std::string_view::size_type;

      // once in the lifetime of the application build a SMILES dictionary. We use this tree to perform pattern
      // matching between the input SMILES and the words in the dictionary
      static const auto dictionary = build_current_smiles_dictionary();

      // before starting, we need to re-initialize the memory
      output_string.clear();
      min_path_index.resize(plain_description.size() + 1); // the +1 is for represeting the past-to-end case
      min_path_score.resize(plain_description.size() + 1); // also here
      const auto input_size = plain_description.size();

      // starting from the end of the path, we need to fill the support vector with information about the best to
      // reach the end, re-using the optimal path. The past-to-end case shall have zero cost.
      // NOTE: the condition on the for is only to prevent undefiend behaviour with zero-lenght SMILES
      min_path_index.back()     = std::numeric_limits<std::size_t>::max();
      min_path_score.back()     = index_type{0};
      const auto starting_index = std::max(index_type{0}, input_size - index_type{1}); // if size_t is signed
      for (index_type input_index{starting_index}; input_index < input_size; --input_index) {
        min_path_score[input_index] = std::numeric_limits<index_type>::max();

        // then we find all the possible matches that we can use to encode the string and that we will lower the
        // cost of this position
        bool find_mismatch  = false;
        auto current_node   = dictionary.root;
        auto matching_index = input_index;
        while (!find_mismatch && matching_index < plain_description.size()) {
          auto find_match                   = false;
          const auto [edge_begin, edge_end] = boost::out_edges(current_node, dictionary.graph);
          for (auto edge_it = edge_begin; edge_it != edge_end; ++edge_it) {
            if (dictionary.graph[*edge_it].character == plain_description[matching_index]) {
              current_node = boost::target(*edge_it, dictionary.graph);
              matching_index++;
              find_match = true;
              break;
            }
          }
          if (find_match) {
            const auto& graph_entry = dictionary.graph[current_node];
            if (graph_entry.with_output) {
              const auto& dictionary_entry = SMILES_DICTIONARY[graph_entry.index];

              // the cost of taking this pattern is equal to 1 (we need to encode this pattern using a
              // character) plus the cost of the index that we would reach encoding this pattern
              const auto pattern_lenght = dictionary_entry.size;
              const auto path_cost      = index_type{1} + min_path_score[input_index + pattern_lenght];

              // check if this path is the best. In this case we select it as the best one
              if (path_cost < min_path_score[input_index]) {
                min_path_index[input_index] = graph_entry.index;
                min_path_score[input_index] = path_cost;
              }
            }
          } else {
            find_mismatch = true;
          }
        }

        // if we didn't find any match we need to escape the current character with a cost of 2 unit because we
        // also need to insert the escaping character in the string
        if (min_path_score[input_index] == std::numeric_limits<index_type>::max()) {
          min_path_index[input_index] = std::numeric_limits<std::size_t>::max();
          min_path_score[input_index] = index_type{2} + min_path_score[input_index + 1];
        }

        // we need to bail out when we completed the input string
        if (input_index == index_type{0}) {
          break;
        }
      }

      // when we reached this point, we can traverse the path from left to right and we are sure that we select
      // the best coverage of patterns that can complete the string
      for (index_type input_index{0}; input_index < input_size; /* internally handled */) {
        const auto word_index = min_path_index[input_index];
        if (word_index < std::numeric_limits<std::size_t>::max()) {
          output_string.push_back(static_cast<char_type>(word_index));
          input_index += SMILES_DICTIONARY[word_index].size;
        } else {
          output_string.push_back(smiles_dictionary_escape_char);
          output_string.push_back(plain_description[input_index]);
          input_index += 1;
        }
      }

// when we reached this point we have correctly compressed the SMILES, we can return it
#pragma omp critical
      { out_s << output_string << std::endl; }
    }

    std::string_view smiles_decompressor::operator()(const std::string_view& compressed_description) {
      // decompressing a SMILES is really just a look up on the SMILES_DICTIONARY. We just need to pay attention
      // when the compressed SMILES has excaped something
      // NOTE: we need to start from a clean string
      scratchpad.clear();

      // now it is time to add the SMILES components
      const auto compressed_smiles_len = compressed_description.size();
      for (std::string_view::size_type i{0}; i < compressed_smiles_len; ++i) {
        const auto character = compressed_description[i];
        if (character != smiles_dictionary_escape_char) {
          const auto& dictionary_entry = SMILES_DICTIONARY[static_cast<uint8_t>(character)];
          scratchpad.append(dictionary_entry.pattern, dictionary_entry.size);
        } else {
          ++i;
          scratchpad.push_back(compressed_description[i]);
        }
      }

      // construct a view string using the memory owned by our buffer
      return {scratchpad};
    }
  } // namespace cpu
} // namespace smiles