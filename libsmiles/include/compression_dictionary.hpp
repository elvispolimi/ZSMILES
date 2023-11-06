#pragma once

#include "dictionary_graph.hpp"

#include <array>
#include <string>

#define SMILES_DICT_SIZE 256
#define SMILES_DICT_NOT_PRINT 33
#define LONGEST_PATTERN 14

namespace smiles {

  struct smiles_dictionary_entry {
    const char* pattern;
    std::string::size_type size;
  };

  static constexpr std::array<smiles_dictionary_entry, 256> SMILES_DICTIONARY = {{
      {"ERROR", 5}, // to avoid non-printable character no 0
      {"ERROR", 5}, // to avoid non-printable character no 1
      {"ERROR", 5}, // to avoid non-printable character no 2
      {"ERROR", 5}, // to avoid non-printable character no 3
      {"ERROR", 5}, // to avoid non-printable character no 4
      {"ERROR", 5}, // to avoid non-printable character no 5
      {"ERROR", 5}, // to avoid non-printable character no 6
      {"ERROR", 5}, // to avoid non-printable character no 7
      {"ERROR", 5}, // to avoid non-printable character no 8
      {"ERROR", 5}, // to avoid non-printable character no 9
      {"ERROR", 5}, // to avoid non-printable character no 10
      {"ERROR", 5}, // to avoid non-printable character no 11
      {"ERROR", 5}, // to avoid non-printable character no 12
      {"ERROR", 5}, // to avoid non-printable character no 13
      {"ERROR", 5}, // to avoid non-printable character no 14
      {"ERROR", 5}, // to avoid non-printable character no 15
      {"ERROR", 5}, // to avoid non-printable character no 16
      {"ERROR", 5}, // to avoid non-printable character no 17
      {"ERROR", 5}, // to avoid non-printable character no 18
      {"ERROR", 5}, // to avoid non-printable character no 19
      {"ERROR", 5}, // to avoid non-printable character no 20
      {"ERROR", 5}, // to avoid non-printable character no 21
      {"ERROR", 5}, // to avoid non-printable character no 22
      {"ERROR", 5}, // to avoid non-printable character no 23
      {"ERROR", 5}, // to avoid non-printable character no 24
      {"ERROR", 5}, // to avoid non-printable character no 25
      {"ERROR", 5}, // to avoid non-printable character no 26
      {"ERROR", 5}, // to avoid non-printable character no 27
      {"ERROR", 5}, // to avoid non-printable character no 28
      {"ERROR", 5}, // to avoid non-printable character no 29
      {"ERROR", 5}, // to avoid non-printable character no 30
      {"ERROR", 5}, // to avoid non-printable character no 31
      {"ERROR", 5}, // to avoid the space character
      {R"(C(=O))", 5},
      {R"(c0ccc)", 5},
      {R"(#)", 1},
      {R"(CC)", 2},
      {R"(c0)", 2},
      {R"((C)", 2},
      {R"(cc)", 2},
      {R"(()", 1},
      {R"())", 1},
      {R"(0))", 2},
      {R"(+)", 1},
      {R"(S(=O)(=O))", 9},
      {R"(-)", 1},
      {R"((C))", 3},
      {R"(/)", 1},
      {R"(0)", 1},
      {R"(1)", 1},
      {R"(2)", 1},
      {R"(3)", 1},
      {R"(4)", 1},
      {R"(5)", 1},
      {R"(6)", 1},
      {R"(c1)", 2},
      {R"(0CC)", 3},
      {R"(c()", 2},
      {R"([C@@H])", 6},
      {R"(cc0)", 3},
      {R"([nH])", 4},
      {R"(=)", 1},
      {R"(O))", 2},
      {R"([C@H])", 5},
      {R"(@)", 1},
      {R"((CC)", 3},
      {R"(B)", 1},
      {R"(C)", 1},
      {R"((F))", 3},
      {R"(C0)", 2},
      {R"(F)", 1},
      {R"(C(=O)N)", 6},
      {R"(cn)", 2},
      {R"(I)", 1},
      {R"(c2)", 2},
      {R"(CN)", 2},
      {R"(c0ccccc0)", 8},
      {R"(C()", 2},
      {R"(N)", 1},
      {R"(O)", 1},
      {R"(P)", 1},
      {R"(c0n)", 3},
      {R"(c0ccc(cc0))", 10},
      {R"(S)", 1},
      {R"(C0CC)", 4},
      {R"(c1cc)", 4},
      {R"(c(n0))", 5},
      {R"(C1)", 2},
      {R"(C(C)(C))", 7},
      {R"(CO)", 2},
      {R"(Z)", 1},
      {R"([)", 1},
      {R"(\)", 1},
      {R"(])", 1},
      {R"(C))", 2},
      {R"([N+](=O)[O-])", 12},
      {R"(CC0)", 3},
      {R"(c0cc)", 4},
      {R"(NC(=O))", 6},
      {R"(c)", 1},
      {R"(n0)", 2},
      {R"(c2ccccc12)", 9},
      {R"(=C)", 2},
      {R"(c0))", 3},
      {R"((Cl))", 4},
      {R"(i)", 1},
      {R"(CC0))", 4},
      {R"(n1)", 2},
      {R"(l)", 1},
      {R"(N0CC)", 4},
      {R"(n)", 1},
      {R"(o)", 1},
      {R"(2))", 2},
      {R"((C0))", 4},
      {R"(r)", 1},
      {R"(s)", 1},
      {R"(nn)", 2},
      {R"(C0))", 3},
      {R"(c0cn)", 4},
      {R"((O))", 3},
      {R"(=O))", 3},
      {R"(OCC)", 3},
      {R"(c3ccccc23))", 10},
      {R"(c0ccccc0))", 9},
      {R"(N))", 2},
      {R"(cc()", 3},
      {R"(C(F)(F))", 7},
      {R"(c(C)", 3},
      {R"(c(C))", 4},
      {R"((CC0))", 5},
      {R"(CCN)", 3},
      {R"(c0o)", 3},
      {R"(C(C)", 3},
      {R"(C(=O)O)", 6},
      {R"(nc0)", 3},
      {R"(C(=C)", 4},
      {R"(C=C)", 3},
      {R"(s0))", 3},
      {R"(CC1)", 3},
      {R"(Br))", 3},
      {R"(cc1)", 3},
      {R"(c1n)", 3},
      {R"(c0nc(n[nH]0))", 12},
      {R"(c0cccc(c0))", 10},
      {R"(c0cs)", 4},
      {R"(c2cc)", 4},
      {R"(OC)", 2},
      {R"(CC(C))", 5},
      {R"(CC()", 3},
      {R"(cn0))", 4},
      {R"(c0ccc()", 6},
      {R"(c3)", 2},
      {R"(CC0CC)", 5},
      {R"(cc0))", 4},
      {R"(ccc0)", 4},
      {R"(C2)", 2},
      {R"(c0cc()", 5},
      {R"(C(=O)O))", 7},
      {R"(c1ccc()", 6},
      {R"(C(=O)N0CC)", 9},
      {R"(nc()", 3},
      {R"(CCO)", 3},
      {R"(C(C))", 4},
      {R"(c0onc(n0))", 9},
      {R"(C(O))", 4},
      {R"(c0ncc)", 5},
      {R"(CN(C)", 4},
      {R"(1))", 2},
      {R"(c0noc(n0))", 9},
      {R"(CS)", 2},
      {R"(CCC0)", 4},
      {R"(Cl))", 3},
      {R"(CC(C)(C)O)", 9},
      {R"(1CC)", 3},
      {R"(F))", 2},
      {R"(C(=O)OC(C)(C)C)", 14},
      {R"(NC)", 2},
      {R"(c0ccn)", 5},
      {R"(C0(CC)", 5},
      {R"(CC(C)", 4},
      {R"(n0))", 3},
      {R"(c1))", 3},
      {R"(3))", 2},
      {R"(=N)", 2},
      {R"(CN(CC)", 5},
      {R"(N(C)", 3},
      {R"(CCC)", 3},
      {R"([O-][N+](=O))", 12},
      {R"(c0n[nH]c(n0))", 12},
      {R"(N(CC0))", 6},
      {R"(C1=O)", 4},
      {R"(N(C))", 4},
      {R"(0C)", 2},
      {R"(c1cn)", 4},
      {R"(OCC1c0ccccc0)", 12},
      {R"(n0nn)", 4},
      {R"([C@@])", 5},
      {R"(c4ccccc34))", 10},
      {R"(\C=C\)", 5},
      {R"(c(Cl))", 5},
      {R"(C(N)", 3},
      {R"(C(=O)C)", 6},
      {R"(CCC0))", 5},
      {R"(CCCC)", 4},
      {R"(C#N))", 4},
      {R"(C(=S))", 5},
      {R"(c0ccsc0)", 7},
      {R"(CN(C0))", 6},
      {R"(Cc1cc)", 5},
      {R"(c0cccnc0)", 8},
      {R"(c0nnnn0C)", 8},
      {R"(OC(=O))", 6},
      {R"(c12))", 4},
      {R"(n1))", 3},
      {R"(nn0))", 4},
      {R"(CN0)", 3},
      {R"(s0)", 2},
      {R"(cc2)", 3},
      {R"(C0=)", 3},
      {R"(C0CC0))", 6},
      {R"(c0nc()", 5},
      {R"((CC0CC)", 6},
      {R"(c0occc0))", 8},
      {R"(N1)", 2},
      {R"(c0ccc(F)cc0)", 11},
      {R"(c2))", 3},
      {R"(c0s)", 3},
      {R"(C[C@H])", 6},
      {R"(CN0CC)", 5},
      {R"(C[C@@H])", 7},
      {R"(c(N)", 3},
      {R"(C0=O))", 5},
      {R"(OCC0))", 5},
      {R"((CC0)", 4},
      {R"(cn0)", 3},
      {R"(c12)", 3},
      {R"(Cl)", 2},
      {R"((N)", 2},
      {R"(O=C()", 4},
      {R"([C@])", 4},
      {R"(c1ccc0)", 6},
      {R"(N(C(=O))", 7},
      {R"(c(O))", 4},
      {R"(c0cnn)", 5},
      {R"(c(F))", 4},
      {R"(c1o)", 3},
      {R"(c1cccc()", 7},
      {R"(n0cc)", 4},
      {R"(C0CC0)", 5},
      {R"(c0ccc(Cl))", 9},
      {R"(C(CC0))", 6},
      {R"(N0)", 2},
      {R"(CC(=O)N)", 7},
      {R"(c2ccccc12))", 10},
      {R"(c0co)", 4},
      {R"(c[nH])", 5},
  }};

  static constexpr auto smiles_dictionary_escape_char = std::string::value_type{' '};

  dictionary_tree_type build_current_smiles_dictionary(void);

} // namespace smiles
