import argparse
import pathlib
import typing
from typing import Dict, List


class dictionary_fsm:
    character_id = (0,)
    pattern = (1,)
    huffman = (2,)
    empty_line = 3


MAX_CHAR_VALUE = 256
NO_PRINTABLE = 33


class node:
    def __init__(self, letter) -> None:
        self.id: int = -1
        self.pattern = -1
        self.letter: str = letter
        self.neighbors: Dict[str, "node"] = {}
        pass


def insert_node(root: node, pattern: str, pattern_index: int):
    assert len(pattern) != 0
    letter = pattern[0]
    if not letter in root.neighbors:
        n = node(letter)
        root.neighbors[letter] = n
    else:
        n = root.neighbors[letter]
    if len(pattern)>1:
        insert_node(n, pattern[1:], pattern_index)
    else:
        n.pattern = pattern_index


def flatten_tree(flat: List[node], level_nodes: List[node]):
    next_level = []
    for l_node in level_nodes:
        l_node.id = len(flat)
        flat.append(l_node)
        next_level.extend(list(l_node.neighbors.values()))
    if next_level:
        flatten_tree(flat, next_level)


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
    dictionary: List[str] = []
    state = dictionary_fsm.character_id
    with open(args.dictionary, "r") as dictionary_file:
        for line in dictionary_file:
            dictionary.append(line.strip())
    if len(dictionary) > MAX_CHAR_VALUE - NO_PRINTABLE:
        raise ValueError(
            "Too many words for the single character representation: {}".format(
                len(dictionary)
            )
        )

    root = node(" ")
    for entry, index in zip(dictionary, range(len(dictionary))):
        insert_node(root, entry, index+NO_PRINTABLE)

    flat: List[node] = []
    flatten_tree(flat, [root])

    print(f"std::array<node,{len(flat)}> n" + "{")
    for n in flat:
        if n.pattern==-1:
            print(f"     node{{'{n.letter}'}},")
        else:
            print(f"     node{{'{n.letter}', static_cast<node::node_letter>({n.pattern})}},")
    print("};")
    for n in flat:
        for neigh_l, neigh in n.neighbors.items():
            print(f"n[{n.id}].neighbor[{ord(neigh_l)-33}]={neigh.id-n.id};")
    print("return n;")
