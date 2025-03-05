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

    # generate the array with he pattern dictionary
    print("const std::array<smiles_dictionary_entry, 256> SMILES_DICTIONARY = {{")
    for i in range(256):
        if i < 32:
            print(
                '    {{"ERROR", 5}}, // to avoid non-printable character no {}'.format(
                    i
                )
            )
        elif i == 32:
            print('    {"ERROR", 5}, // to avoid the space character')
        elif i < len(dictionary) + 33:
            word = dictionary[i - 33]
            print('    {{R"({})",{}}},'.format(word, len(word)))
        else:
            print('   {{"ERROR", 5}}, // to fill character {}'.format(i))
        print
    print("}};")
    print()
