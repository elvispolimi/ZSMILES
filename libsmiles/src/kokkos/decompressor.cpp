#include "zsmiles/kokkos/decompressor.hpp"
#include "zsmiles/kokkos/dictionary.hpp"
#include <Kokkos_Core.hpp>
#include <iostream>
#include <cassert>

namespace smiles {
namespace kokkos {

smiles_decompressor::smiles_decompressor() {
  Kokkos::initialize();
}

smiles_decompressor::~smiles_decompressor() {
  Kokkos::finalize();
}

void smiles_decompressor::decompress(std::ofstream& out_s) {
  const size_t M = smiles_len.size();
  Dictionary dict;

  auto h_comp = Kokkos::create_mirror_view(Kokkos::View<char*>("comp", smiles_host.size()));
  std::memcpy(h_comp.data(), smiles_host.data(), smiles_host.size());
  Kokkos::View<char*> d_comp("comp", smiles_host.size());
  Kokkos::deep_copy(d_comp, h_comp);

  auto h_idx     = Kokkos::create_mirror_view(Kokkos::View<size_t*>("idx", M));
  auto h_idx_out = Kokkos::create_mirror_view(Kokkos::View<size_t*>("idx_out", M));
  auto h_len     = Kokkos::create_mirror_view(Kokkos::View<size_t*>("len", M));

  for (size_t j = 0; j < M; ++j) {
    h_idx(j)     = smiles_index[j];
    h_len(j)     = smiles_len[j];
    h_idx_out(j) = smiles_index_out[j];
  }

  Kokkos::View<size_t*> d_idx("idx", M);
  Kokkos::View<size_t*> d_len("len", M);
  Kokkos::View<size_t*> d_idx_out("idx_out", M);
  Kokkos::deep_copy(d_idx, h_idx);
  Kokkos::deep_copy(d_len, h_len);
  Kokkos::deep_copy(d_idx_out, h_idx_out);

  size_t output_buffer_size = smiles_index_out.back() + smiles_len.back() * 14 + 1;
  Kokkos::View<char*> d_out("out", output_buffer_size);
  Kokkos::parallel_for("DecompressLines", M, KOKKOS_LAMBDA(int j) {
    size_t start = d_idx(j);
    size_t length = d_len(j);
    size_t wp = d_idx_out(j);

    for (size_t k = 0; k < length; ++k) {
      char c = d_comp(start + k);
      if (c != ' ') {
        uint8_t entry = static_cast<uint8_t>(c);
        size_t off = dict.offsets(entry);
        size_t sz  = dict.sizes(entry);
        for (size_t t = 0; t < sz; ++t) {
          d_out(wp++) = dict.patterns(off + t);
        }
      } else {
        char literal = d_comp(start + (++k));
        d_out(wp++) = literal;
      }
    }
    d_out(wp++) = '\n';
  });

  auto h_out = Kokkos::create_mirror_view(d_out);
  Kokkos::deep_copy(h_out, d_out);

  for (size_t pos = 0, j = 0; j < M; ++j) {
    size_t end = pos;
    while (h_out(end) != '\n') ++end;
    out_s.write(&h_out(pos), end - pos + 1);
    pos = end + 1;
  }

  smiles_index.clear();
  smiles_index_out.clear();
  smiles_len.clear();
  smiles_host.clear();
}

void smiles_decompressor::clean_up(std::ofstream& out_s) {
  if (!smiles_len.empty()) {
    decompress(out_s);
  }
}

smiles_compressor::smiles_compressor() {
  Kokkos::initialize();
}

smiles_compressor::~smiles_compressor() {
  Kokkos::finalize();
}

void smiles_compressor::compress(std::ofstream& out_s) {
  // TODO: implement
}

void smiles_compressor::clean_up(std::ofstream& out_s) {
  // TODO: implement
}

void smiles_compressor::test() {
  // Test the dictionary traversal functionality
  std::cout << "Testing smiles_compressor pattern matching in trie..." << std::endl;

  // Costruisci il dizionario
  auto nodes = smiles::kokkos::build_gpu_smiles_dictionary();

  // Crea un mirror sul lato host per verificare i dati
  auto host_nodes = Kokkos::create_mirror_view(nodes);
  Kokkos::deep_copy(host_nodes, nodes);

  // Pattern da cercare (esempio)
  std::string pattern = "C0c";

  // Esegui un parallel_for per i primi 5 nodi
  Kokkos::parallel_for("TestPatternMatching", 5, KOKKOS_LAMBDA(const int i) {
    printf("Starting pattern matching from Node %d: letter = %c\n", i, nodes(i).letter);

    // Partenza dal nodo corrente
    const node* current_node = &nodes(i);
    bool pattern_found = true;

    // Traversare il trie per il pattern
    for (size_t j = 0; j < pattern.size(); ++j) {
      char c = pattern[j];
      int next_index = current_node->neighbor[static_cast<uint8_t>(c)];
      if (next_index == -1) {
        pattern_found = false;
        break; // Pattern non trovato
      }
      current_node = &nodes(next_index);
    }

    // Stampa il risultato del pattern matching
    if (pattern_found) {
      printf("Pattern '%s' found starting from Node %d\n", pattern.c_str(), i);
    } else {
      printf("Pattern '%s' not found starting from Node %d\n", pattern.c_str(), i);
    }
  });

  // Sincronizza per assicurarsi che l'output sia completo
  Kokkos::fence();
}

} // namespace kokkos
} // namespace smiles

