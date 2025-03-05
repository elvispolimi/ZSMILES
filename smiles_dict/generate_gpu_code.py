import argparse
import pathlib
from typing import List

class dictionary_fsm:
    character_id = (0,)
    pattern = (1,)
    huffman = (2,)
    empty_line = 3


MAX_CHAR_VALUE = 256
NO_PRINTABLE = 33


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
    for entry, index in zip(dictionary, range(len(dictionary))):
        print(f"else if (find_index == {index+33})")
        print(f"    memcpy(&smiles_out_l[index], \"{entry}\" , {len(entry)});")
