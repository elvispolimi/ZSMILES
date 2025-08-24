#include "zsmiles/kokkos/decompressor.hpp"
#include "zsmiles/kokkos/dictionary.hpp"
#include <zsmiles/kokkos/kokkos_smiles_dictionary.hpp>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <cassert>

namespace smiles {
namespace kokkos {

void smiles::kokkos::smiles_decompressor::decompress(std::ofstream& out_s) {
  deep_copy(d_smiles_len, h_smiles_len);
  deep_copy(d_smiles_index, h_smiles_index);
  deep_copy(d_smiles_index_out, h_smiles_index_out);
  deep_copy(d_smiles_host, h_smiles_host);
  deep_copy(d_smiles_out, h_smiles_out);
  auto dict = build_decompression_dictionary();
  auto k_smiles_len = d_smiles_len;
  auto k_smiles_index = d_smiles_index;
  auto k_smiles_index_out = d_smiles_index_out;
  auto k_smiles_host = d_smiles_host;
  auto k_smiles_out = d_smiles_out;

  Kokkos::printf("Start decompressing");
  Kokkos::parallel_for("Decompress", Kokkos::RangePolicy<>(0, smiles_count),
    KOKKOS_LAMBDA(const int id) {
        const size_t length = k_smiles_len(id);
        const size_t in_idx = k_smiles_index(id);
        const size_t out_idx = k_smiles_index_out(id);

        size_t wp = out_idx; // Write pointer per l'output

        for (size_t k = 0; k < length; ++k) {
            char c = k_smiles_host(in_idx + k); // Legge il carattere corrente
            if (c != ' ') {
                // Espande l'entry del dizionario
                uint8_t entry = static_cast<uint8_t>(c);
                for (size_t t = 0; t < dict(entry).length; ++t) {
                    k_smiles_out(wp++) = dict(entry).pattern[t];
                }
            } else {
                // Carattere di escape: copia il carattere letterale successivo
                unsigned char literal = k_smiles_host(in_idx + (++k));
                k_smiles_out(wp++) = literal;
            }
        }
        k_smiles_out(wp++) = '\n'; // Aggiunge un terminatore di stringa per la linea SMILES
    });

  // Sincronizzazione
  Kokkos::fence();

  // Copia i dati di output su host e scrivi su file
  Kokkos::deep_copy(h_smiles_out, d_smiles_out);
  auto host_idx_out = Kokkos::create_mirror_view(d_smiles_index_out);
  Kokkos::deep_copy(host_idx_out, d_smiles_index_out);

  for(int i = 0; i < smiles_count; i++) {
      size_t start_idx = host_idx_out(i);
      size_t next_index = (i + 1 < smiles_count) ? host_idx_out(i + 1) : (h_smiles_len(i) * LONGEST_PATTERN + 1 + start_idx);
      size_t length = next_index - start_idx; // assume che smiles_index_out abbia un elemento in più alla fine
      std::string decompressed_smiles(&h_smiles_out(start_idx), length);
      // Trova il terminatore '\n' per determinare la lunghezza effettiva
      auto null_pos = decompressed_smiles.find('\n');
      if (null_pos != std::string::npos) {
          decompressed_smiles = decompressed_smiles.substr(0, null_pos);
      }

      // Scrivi lo SMILES nel file di output
      out_s << decompressed_smiles << '\n';
  }

  // Reset contatori
  smiles_count = 0;
  smiles_host_index = 0;
}

void smiles_decompressor::copy_out(std::ofstream& out_s) {
  
}


void smiles_decompressor::clean_up(std::ofstream& out_s) {
  
}

void smiles_compressor::compress(std::ofstream& out_s) {
  auto gpu_dict = build_gpu_smiles_dictionary();
  deep_copy(d_smiles_len, h_smiles_len);
  deep_copy(d_smiles_index, h_smiles_index);
  deep_copy(d_smiles_host, h_smiles_host);
  deep_copy(d_smiles_out, h_smiles_out);
  auto smiles_len_k = d_smiles_len;
  auto smiles_index_k = d_smiles_index;
  auto smiles_host_k = d_smiles_host;
  auto score_matrix_k = score_matrix;
  auto pattern_matrix_k = pattern_matrix;
  auto length_matrix_k = length_matrix;
  auto smiles_out_k = d_smiles_out;
  auto smiles_out_len_k = d_smiles_out_len;
  auto smiles_index_out_k = d_smiles_index_out;


  // 1. Calcola le lunghezze di output per ogni stringa

  Kokkos::parallel_for("CompressLength", Kokkos::RangePolicy<>(0, smiles_count),
    KOKKOS_LAMBDA(const int id) {
        auto smile_length = smiles_len_k(id);
        assert(MAX_SMILES_LEN > smile_length);
        auto smile_index = smiles_index_k(id);
        score_matrix_k(id, smile_length) = 0;

        for (auto index = static_cast<int>(smile_length - 1); index >= 0; index--) {
          score_matrix_k(id, index) = score_matrix_k(id, index + 1) + 2;
          pattern_matrix_k(id, index) = 0;
          length_matrix_k(id, index) = 1;
          auto curr = gpu_dict(0);
          auto curr_id = 0;
          auto valid_curr = true;

          for(int j = 0; j < LONGEST_PATTERN && valid_curr && j < (smile_length - index); j++){
            const int next_i = curr.neighbor[smiles_host_k(smile_index + index + j) - NOT_PRINTABLE];
            if (next_i) {
              curr = gpu_dict(next_i + curr_id);
              curr_id = next_i + curr_id;
              if (curr.pattern != 0) {
                const auto next = index + j + 1;
                const auto w = score_matrix_k(id, next) + 1;
                if (w < score_matrix_k(id, index)) {
                  score_matrix_k(id, index) = w;
                  pattern_matrix_k(id, index) = curr.pattern;
                  length_matrix_k(id, index) = j + 1;
                }
              }
            } else {
              valid_curr = false;
            }
          }
        }
          int idx = 0;
          int out_idx = 0;
          while (idx < smile_length) {
            if (pattern_matrix_k(id, idx) != 0) {
              out_idx++;
              idx += length_matrix_k(id, idx);
            } else {
              out_idx += 2;
              idx += 1;
          }
          }
          smiles_out_len_k(id) = out_idx + 1;
        }
      );
      
      Kokkos::fence();

      // 2. Calcola gli indici cumulativi per il buffer di output
  Kokkos::View<int*, Kokkos::DefaultExecutionSpace::memory_space> dev_smiles_index_out("dev_smiles_index_out", smiles_count + 1);

  Kokkos::parallel_scan("CalculateOutputIndices", Kokkos::RangePolicy<>(0, smiles_count),
    KOKKOS_LAMBDA(const int i, int& update, const bool final_pass) {
      update += smiles_out_len_k(i);
      if (final_pass) {
          smiles_index_out_k(i + 1) = update;
      }
    });

  Kokkos::fence();

  // 3. Esegui la compressione finale scrivendo sul buffer di output
  Kokkos::parallel_for("CompressWrite", Kokkos::RangePolicy<>(0, smiles_count),
    KOKKOS_LAMBDA(const int id) {
      auto smile_length = smiles_len_k(id);
      auto smile_index = smiles_index_k(id);
      auto out_start_idx = smiles_index_out_k(id);
      int idx = 0;
      int out_idx = 0;
      
      while (idx < smile_length) {
          if (pattern_matrix_k(id, idx) != 0) {
              smiles_out_k(out_start_idx + out_idx) = pattern_matrix_k(id, idx);
              ++out_idx;
              idx += length_matrix_k(id, idx);
          } else {
              smiles_out_k(out_start_idx + out_idx) = smiles_dictionary_escape_char;
              ++out_idx;
              smiles_out_k(out_start_idx + out_idx) = smiles_host_k(smile_index + idx);
              ++out_idx;
              idx += 1;
          }
      }
      smiles_out_k(out_start_idx + out_idx) = '\n';
  });

  Kokkos::fence();

      // 4. Copia i dati e scrivi su file
  Kokkos::deep_copy(h_smiles_out, d_smiles_out);
  Kokkos::deep_copy(h_smiles_index_out, smiles_index_out_k);

  std::string temp_out;
  auto total_size = h_smiles_index_out(smiles_count);
  temp_out.reserve(total_size);

  for (int i = 0; i < smiles_count; i++) {
    const char* ptr = &h_smiles_out(h_smiles_index_out(i));
    auto length = h_smiles_index_out(i + 1) - h_smiles_index_out(i);
    temp_out.append(ptr, length);
  }

  out_s << temp_out;
}

void smiles_compressor::clean_up(std::ofstream& out_s) {
  // TODO: implement
}



void smiles_compressor::test() {
  auto gpu_dict = build_gpu_smiles_dictionary();
  Kokkos::parallel_for("Test", Kokkos::RangePolicy<>(0, gpu_dict.extent(0)),
    KOKKOS_LAMBDA(const int id) {
        auto curr = gpu_dict(id);
        Kokkos::printf("[DEBUG] Node %d: letter: %c, pattern: %c\n", 
               id, curr.letter, curr.pattern);
        for (int i = 0; i < PRINTABLE_CHAR; i++) {
          if (curr.neighbor[i] != 0 && id == 0) {
            Kokkos::printf("[DEBUG] Neighbor %d: %d\n", id, static_cast<int>(curr.neighbor[i]));
          }
        }
  });
  std::cout << "[DEBUG] Test completed." << std::endl;
}

} // namespace kokkos
} // namespace smiles

