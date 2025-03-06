#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <cstdint>
#include <string>

namespace smiles {

  // these are the type alias that define a dictionary graph. The idea is that it should build a finite state
  // machine to recognized a pattern in the text to be replaced by another pattern. This graph should be
  // agnostic with respect to the dictionary that we are going to use to perform the substition.
  struct dictionary_vertex_type {
    bool with_output;  // state whether this not is associated to an output word
    std::size_t index; // the index of the word that will replace the pattern (if it has an output)
  };
  struct dictionary_edge_type {
    std::string::value_type character; // the character that we need to read to change the FSM state
  };
  using dictionary_graph_type              = boost::adjacency_list<boost::vecS,
                                                      boost::vecS,
                                                      boost::directedS,
                                                      dictionary_vertex_type,
                                                      dictionary_edge_type>;
  using dictionary_vertex_description_type = boost::graph_traits<dictionary_graph_type>::vertex_descriptor;
  using dictionary_edge_description_type   = boost::graph_traits<dictionary_graph_type>::edge_descriptor;

  // however, to actually use a dictionary graph for pattern matching, we would build some trees, most probably
  // trie. Therefore, it is useful to declare a support structure to also hold the tree root
  struct dictionary_tree_type {
    dictionary_graph_type graph;
    dictionary_vertex_description_type root;
  };

} // namespace smiles
