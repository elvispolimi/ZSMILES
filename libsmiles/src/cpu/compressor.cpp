#include "cpu/compressor.hpp"

#include "compression_dictionary.hpp"
#include "cpu/dictionary.hpp"

#include <algorithm>
#include <boost/graph/adjacency_list.hpp>
#include <cstddef>
#include <fstream>
#include <limits>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace smiles {
  namespace cpu {
    template<typename char_type>
    static inline auto char2int(char_type&& c) {
      return static_cast<int>(c - char_type{'0'}); // hacky conversion from string to int
    }

    static inline auto int2char(const int num) {
      return static_cast<std::string_view::value_type>(num + int{'0'}); // again, hacky conversion
    }

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
      min_path_index.back() = std::numeric_limits<std::size_t>::max();
      min_path_score.back() = index_type{0};

      for (index_type input_index_safe{input_size}; input_index_safe > 0; --input_index_safe) {
        const auto input_index      = std::max(index_type{0}, input_index_safe - index_type{1});
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
      out_s << output_string << std::endl;
    }

    // we define the FSM states
    enum class smiles_ring_enumerator_states {
      in_smiles,
      in_symbol,
      in_second_digit_ring,
    };

    struct interval {
      size_t begin, end;
    };

    std::string smiles_compressor::preprocess(const std::string_view& smile) {
      std::vector<int> ids;
      std::vector<interval> indexes;
      auto state            = smiles_ring_enumerator_states::in_smiles;
      auto decimal_id       = int{0};
      const auto smile_size = smile.size();

      indexes.emplace_back();
      indexes.back().begin = 0;
      bool is_double_digit = false;
      for (std::size_t i = 0; i < smile_size; i++) {
        const auto character = smile[i];
        switch (state) {
          case smiles_ring_enumerator_states::in_smiles:
            if (character == '[' || character == '{') {
              state = smiles_ring_enumerator_states::in_symbol;
            } else if (std::isdigit(static_cast<unsigned char>(character))) {
              ids.push_back(char2int(character) + decimal_id);
              if (is_double_digit)
                indexes.back().end = i - 2;
              else
                indexes.back().end = i - 1;
              if (indexes.back().begin > indexes.back().end)
                indexes.back().end = indexes.back().end;
              indexes.emplace_back();
              indexes.back().begin = i + 1;
              decimal_id           = int{0};
              is_double_digit      = false;
            } else if (character == '%') {
              state = smiles_ring_enumerator_states::in_second_digit_ring;
            }
            break;

          case smiles_ring_enumerator_states::in_symbol:
            if (character == ']' || character == '}') {
              state = smiles_ring_enumerator_states::in_smiles;
            } else if (character == '[' || character == '{') {
              throw std::runtime_error("Nested symbols in SMILES \"" + std::string{smile} + "\"");
            }
            break;

          case smiles_ring_enumerator_states::in_second_digit_ring:
            if (std::isdigit(static_cast<unsigned char>(character))) {
              decimal_id      = char2int(character) * int{10};
              is_double_digit = true;
              state           = smiles_ring_enumerator_states::in_smiles;
            } else {
              throw std::runtime_error("Spurious % in SMILES \"" + std::string{smile} + "\"");
            }
            break;

          default:
            throw std::runtime_error("Unknow state while parsing SMILES \"" + std::string{smile} + "\"");
        }
      }
      indexes.back().end = smile_size - 1;

      std::vector<int> out_ids;
      const auto num_ids = ids.size();
      out_ids.resize(ids.size());
      std::fill(out_ids.begin(), out_ids.end(), -1);
      // for (size_t dd = 1; dd <= num_ids; dd++) {
      //   for (size_t i = 0; (i + dd) < num_ids; i++) {
      //     if (ids[i] == ids[i + dd]) {
      //       const auto min_id = *std::max_element(out_ids.begin() + i, out_ids.begin() + i + dd);
      //       out_ids[i]        = min_id + 1;
      //       out_ids[i + dd]   = min_id + 1;
      //     }
      //   }
      // }
      // Size_t here does not work because with -1 can do overflow and we will not capture it
      for (signed long t = 1; t < num_ids; t++) {
        for (signed long b = t - 1; b >= 0 && out_ids[t] == -1; b--) {
          if (out_ids[b] == -1 && ids[t] == ids[b]) {
            const auto min_id = *std::max_element(out_ids.begin() + b, out_ids.begin() + t);
            out_ids[b]        = min_id + 1;
            out_ids[t]        = min_id + 1;
          }
        }
      }

      std::string smile_out;
      for (auto i = 0; i < indexes.size(); i++) {
        smile_out.append(smile.begin() + indexes[i].begin, smile.begin() + indexes[i].end + 1);
        if (i < (indexes.size() - 1)) {
          smile_out.push_back(int2char(out_ids[i]));
        }
      }

      return smile_out;
      // auto state      = smiles_ring_enumerator_states::in_smiles;
      // auto decimal_id = int{0};
      // std::string smile_o;
      // int actual_depth = initial_depth;

      // // loop over the input SMILES to perform the substitution
      // for (std::size_t i=start_index; i<smile.size() ; i++) {
      //   const auto character = smile[i];
      //   switch (state) {
      //     case smiles_ring_enumerator_states::in_smiles:
      //       if (character == '[' || character == '{') {
      //         state = smiles_ring_enumerator_states::in_symbol;
      //         smile_o.push_back(character);
      //       } else if (std::isdigit(static_cast<unsigned char>(character))) {
      //         const auto find_id = char2int(character) + decimal_id;
      //         if (find_id == id) {
      //           return {std::to_string(actual_depth - initial_depth) + smile_o + std::to_string(actual_depth - initial_depth), actual_depth, i};
      //         } else {
      //           const auto t = preprocess(smile,find_id,initial_depth+1, i+1);
      //           actual_depth = std::max(actual_depth, std::get<1>(t));
      //           smile_o.append(std::get<0>(t));
      //           i=std::get<2>(t);
      //         }
      //         decimal_id = int{0};
      //       } else if (character == '%') {
      //         state = smiles_ring_enumerator_states::in_second_digit_ring;
      //       } else {
      //         smile_o.push_back(character);
      //       }
      //       break;

      //     case smiles_ring_enumerator_states::in_symbol:
      //       if (character == ']' || character == '}') {
      //         state = smiles_ring_enumerator_states::in_smiles;
      //       } else if (character == '[' || character == '{') {
      //         throw std::runtime_error("Nested symbols in SMILES \"" + std::string{smile} + "\"");
      //       }
      //       smile_o.push_back(character);
      //       break;

      //     case smiles_ring_enumerator_states::in_second_digit_ring:
      //       if (std::isdigit(static_cast<unsigned char>(character))) {
      //         decimal_id = char2int(character) * int{10};
      //         state      = smiles_ring_enumerator_states::in_smiles;
      //       } else {
      //         throw std::runtime_error("Spurious % in SMILES \"" + std::string{smile} + "\"");
      //       }
      //       break;

      //     default:
      //       throw std::runtime_error("Unknow state while parsing SMILES \"" + std::string{smile} + "\"");
      //   }
      // }

      // return {smile_o, actual_depth, smile.size()};
    }

    void smiles_decompressor::operator()(const std::string_view& compressed_description,
                                         std::ofstream& out_s) {
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
      out_s << scratchpad << std::endl;
    }
  } // namespace cpu
} // namespace smiles
