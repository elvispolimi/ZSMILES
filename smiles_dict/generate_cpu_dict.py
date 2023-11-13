import argparse
import pathlib
from sqlite3 import InternalError
import igraph


class dictionary_fsm:
    character_id = (0,)
    pattern = (1,)
    huffman = (2,)
    empty_line = 3


if __name__ == "__main__":
    # parse the program options
    parser = argparse.ArgumentParser(
        description="Generate the content of the cpu dictionary"
    )
    parser.add_argument(
        "--dictionary",
        type=pathlib.Path,
        default="dictionary.dct",
        help="Path to the dictionary",
    )
    args = parser.parse_args()

    # open and load the dictionary (make sure that we can use a single character for each word)
    dictionary = []
    state = dictionary_fsm.character_id
    with open(args.dictionary, "r") as dictionary_file:
        for line in dictionary_file:
            dictionary.append(line.strip())
    if len(dictionary) > 256 - 33:
        raise ValueError(
            "Too many words for the single character representation: {}".format(
                len(dictionary)
            )
        )

    # build the the graph that can recognize the dictionary
    g = igraph.Graph(directed="directed")
    root = g.add_vertex(with_output=False, index=0)  # add the root node
    for word_index, word in enumerate(dictionary):
        vertex = root
        for letter in word:
            matching_vertexes = [
                g.vs[g.es[e].target]
                for e in g.incident(vertex, mode="out")
                if g.es[e]["character"] == letter
            ]
            if matching_vertexes:
                if len(matching_vertexes) > 1:
                    raise InternalError("Failed to construct the graph")
                else:
                    vertex = matching_vertexes[0]
            else:
                new_vertex = g.add_vertex(with_output=False, index=0)
                g.add_edge(vertex, new_vertex, character=letter)
                vertex = new_vertex
        vertex["with_output"] = True
        vertex["index"] = word_index + 33

    # generate the function that initialize the boost graph
    for vertex in g.vs:
        vertex_index = vertex.index
        word_index = vertex["index"]
        print("  const auto v{} = boost::add_vertex(g);".format(vertex_index))
        print("  g[v{}].index = {};".format(vertex_index, word_index))
        if vertex["with_output"]:
            print("  g[v{}].with_output = true;".format(vertex_index))
        else:
            print("  g[v{}].with_output = false;".format(vertex_index))
    for edge in g.es:
        ei = edge.index
        es = edge.source
        et = edge.target
        print(
            "  const auto [e{0}, is_inserted{0}] = boost::add_edge(v{1},v{2},g);".format(
                ei, es, et
            )
        )
        if edge["character"] != "\\":
            print("  g[e{}].character = '{}';".format(ei, edge["character"]))
        else:
            print("  g[e{}].character = '\\{}';".format(ei, edge["character"]))
        print("  assert(is_inserted{});".format(ei))
    print("  dictionary.root = v0;")
