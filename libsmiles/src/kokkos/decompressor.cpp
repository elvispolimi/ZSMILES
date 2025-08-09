#include "zsmiles/kokkos/decompressor.hpp"
#include "zsmiles/kokkos/kokkos_smiles_dictionary.hpp"
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
  const size_t M = smiles_len.size();  // Number of SMILES in this batch
  
  // 1. Copy the compressed batch into a Kokkos device view
  auto h_comp = Kokkos::create_mirror_view(Kokkos::View<const char*>("comp", smiles_host.size()));
  std::memcpy(h_comp.data(), smiles_host.data(), smiles_host.size());
  Kokkos::View<const char*> d_comp("comp", smiles_host.size());
  Kokkos::deep_copy(d_comp, h_comp);

  // 2. Prepare host views for SMILES metadata (index, length, output position)
  auto h_idx = Kokkos::create_mirror_view(Kokkos::View<const size_t*>("idx", M));
  auto h_idx_out = Kokkos::create_mirror_view(Kokkos::View<const size_t*>("idx_out", M));
  auto h_len = Kokkos::create_mirror_view(Kokkos::View<const size_t*>("len", M));

  for (size_t j = 0; j < M; ++j) {
    h_idx(j) = smiles_index[j];
    h_len(j) = smiles_len[j];
    h_idx_out(j) = smiles_index_out[j]; // Not used in this context but retained
  }

  // 3. Create device views and transfer metadata to device
  Kokkos::View<const size_t*> d_idx("idx", M);
  Kokkos::View<const size_t*> d_len("len", M);
  Kokkos::View<const size_t*> d_idx_out("idx_out", M);
  Kokkos::deep_copy(d_idx, h_idx);
  Kokkos::deep_copy(d_len, h_len);
  Kokkos::deep_copy(d_idx_out, h_idx_out);

  // 4. Allocate output buffer on device (max size determined by caller)
  size_t output_buffer_size = smiles_index_out.back() + smiles_len.back() * LONGEST_PATTERN + 1;
  Kokkos::View<char*> d_out("out", output_buffer_size);

  // 5. Decompression kernel: one thread per SMILES
  Kokkos::parallel_for("DecompressLines", M, KOKKOS_LAMBDA(int j) {
    size_t start = d_idx(j);      // Start of compressed SMILES j in d_comp
    size_t length = d_len(j);     // Length of compressed SMILES j
    size_t wp = d_idx_out(j);     // Output write position for SMILES j

    for (size_t k = 0; k < length; ++k) {
      char c = d_comp(start + k);
      if (c != smiles_dictionary_escape_char) {
        // Dictionary entry: expand using dict table
        uint8_t entry = static_cast<uint8_t>(c);
        size_t off = dict.offsets(entry);
        size_t sz  = dict.sizes(entry);
        for (size_t t = 0; t < sz; ++t) {
          d_out(wp++) = dict.patterns(off + t);
        }
      } else {
        // Escape character: copy literal next character
        char literal = d_comp(start + (++k));
        d_out(wp++) = literal;
      }
    }
    d_out(wp++) = '\n'; // Add newline to terminate the SMILES line
  });

  // 6. Copy decompressed output back to host
  auto h_out = Kokkos::create_mirror_view(d_out);
  Kokkos::deep_copy(h_out, d_out);

  // 7. Write decompressed SMILES to output stream line-by-line
  for (size_t pos = 0, j = 0; j < M; ++j) {
    size_t end = pos;
    while (h_out(end) != '\n') ++end;
    output.write(&h_out(pos), end - pos + 1); // include '\n'
    pos = end + 1;
  }

  // 8. Clear host buffers for next batch
  smiles_index.clear();
  smiles_index_out.clear();
  smiles_len.clear();
  smiles_host.clear();
}

void smiles_decompressor::clean_up(std::ofstream& output) {
  if (!smiles_len.empty()) {
    decompress(output);
  }
}

} // namespace kokkos
} // namespace smiles
