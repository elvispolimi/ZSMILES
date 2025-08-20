#include "zsmiles/kokkos/decompressor.hpp"
#include "zsmiles/kokkos/dictionary.hpp"
#include <zsmiles/kokkos/kokkos_smiles_dictionary.hpp>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <cassert>

namespace smiles {
namespace kokkos {

void smiles::kokkos::smiles_decompressor::decompress(std::ofstream& out_s) {
  std::cout << "[DEBUG] Starting decompression..." << std::endl;

  // Verifica il numero di SMILES da decomprimere
  std::cout << "[DEBUG] smiles_count: " << smiles_count << std::endl;

  // Copia i dati dall'host al dispositivo
  std::cout << "[DEBUG] Copying data from host to device..." << std::endl;
  auto host_smiles = Kokkos::create_mirror_view(smiles_host);
  auto host_index = Kokkos::create_mirror_view(smiles_index);
  auto host_index_out = Kokkos::create_mirror_view(smiles_index_out);
  auto host_len = Kokkos::create_mirror_view(smiles_len);
  auto host_smiles_out = Kokkos::create_mirror_view(smiles_out);

  for (size_t i = 0; i < smiles_count; ++i) {
    host_smiles(i) = smiles_host(i);
    host_index(i) = smiles_index(i);
    host_index_out(i) = smiles_index_out(i);
    host_len(i) = smiles_len(i);
    host_smiles_out(i) = smiles_out(i);
  }

  Kokkos::deep_copy(smiles_host, host_smiles);
  Kokkos::deep_copy(smiles_index, host_index);
  Kokkos::deep_copy(smiles_index_out, host_index_out);
  Kokkos::deep_copy(smiles_len, host_len);
  Kokkos::deep_copy(smiles_out, host_smiles_out);

  std::cout << "[DEBUG] Data copied to device." << std::endl;

  // Parallel decompression
  std::cout << "[DEBUG] Launching parallel_for for decompression..." << std::endl;

  auto d_smiles_len      = smiles_len;
  auto d_smiles_index    = smiles_index;
  auto d_smiles_index_out= smiles_index_out;
  auto d_smiles_host     = smiles_host;
  auto d_smiles_out      = smiles_out;
  auto dict = smiles::Dictionary();

  Kokkos::parallel_for("Decompress", Kokkos::RangePolicy<>(0, smiles_count),
    KOKKOS_LAMBDA(const int id) {
        const size_t length = smiles_len(id);
        const size_t in_idx = smiles_index(id);
        const size_t out_idx = smiles_index_out(id);

        size_t wp = out_idx; // Write pointer per l'output

        for (size_t k = 0; k < length; ++k) {
            char c = smiles_host(in_idx + k); // Legge il carattere corrente
            if (c != smiles::smiles_dictionary_escape_char) {
                // Espande l'entry del dizionario
                uint8_t entry = static_cast<uint8_t>(c);
                size_t off = dict.offset(entry);
                size_t sz  = dict.size(entry);
                for (size_t t = 0; t < sz; ++t) {
                    d_smiles_out(wp++) = dict.pattern_at(off + t);
                }
            } else {
                // Carattere di escape: copia il carattere letterale successivo
                char literal = d_smiles_host(in_idx + (++k));
                d_smiles_out(wp++) = literal;
            }
        }
        d_smiles_out(wp++) = '\n'; // Aggiunge una nuova riga per terminare la linea SMILES
    });

  std::cout << "[DEBUG] Decompression kernel completed." << std::endl;

  // Sincronizzazione
  Kokkos::fence();
  std::cout << "[DEBUG] Device synchronized." << std::endl;

  // Copia i dati decompressi dal dispositivo all'host
  auto host_output = Kokkos::create_mirror_view(smiles_out);
  Kokkos::deep_copy(host_output, smiles_out);

  std::cout << "[DEBUG] Copying decompressed data to output file..." << std::endl;
for (size_t i = 0; i < smiles_count; ++i) {
    size_t start_idx = smiles_index_out(i);
    size_t length = smiles_len(i) * LONGEST_PATTERN + 1; // Calcola la lunghezza massima decompressa
    std::string decompressed_smiles(&host_output(start_idx), length);

    // Trova il terminatore '\n' per determinare la lunghezza effettiva
    auto newline_pos = decompressed_smiles.find('\n');
    if (newline_pos != std::string::npos) {
        decompressed_smiles = decompressed_smiles.substr(0, newline_pos);
    }

    // Scrivi lo SMILES nel file di output
    out_s << decompressed_smiles << '\n';
    std::cout << "[DEBUG] Decompressed SMILES #" << i + 1 << ": " << decompressed_smiles << std::endl;
}

  std::cout << "[DEBUG] Decompression completed successfully." << std::endl;

  // Reset contatori
  smiles_count = 0;
  smiles_host_index = 0;
}

void smiles_decompressor::copy_out(std::ofstream& out_s) {
  
}

void add_smiles(const std::string& line) {
    
}

void smiles_decompressor::clean_up(std::ofstream& out_s) {
  
}

void smiles_compressor::compress(std::ofstream& out_s) {
  auto gpu_dict = build_gpu_smiles_dictionary();
  std::cout << "[DEBUG] Starting compression..." << std::endl;
  auto smiles_len_dev = smiles_len; //lunghezza degli smiles
  auto smiles_index_dev = smiles_index; //indice di ogni smile
  auto smiles_out_dev = smiles_out; //buffer continuo di smile in uscita
  auto smiles_host_dev = smiles_host; // buffer di SMILES in ingresso
  auto smiles_index_out_dev = smiles_index_out;
  auto score_matrix = score_matrix_dev;
  auto pattern_matrix = pattern_matrix_dev;
  auto length_matrix = length_matrix_dev;

  Kokkos::parallel_for("Compress", Kokkos::RangePolicy<>(0, smiles_count),
    KOKKOS_LAMBDA(const int id) {
        auto smile_length = smiles_len_dev(id);
        assert(MAX_SMILES_LEN > smile_length);
        auto smile_index = smiles_index_dev(id);
        score_matrix(id, smile_length) = 0;
        
        for (auto index = static_cast<int>(smile_length - 1); index >= 0; index--) { //for loop sullo smile
          score_matrix(id, index) = score_matrix(id, index + 1) + 2;
          pattern_matrix(id, index) = 0;
          length_matrix(id, index) = 1;
          auto curr = gpu_dict(0);
          auto curr_id = 0;
          auto valid_curr = true;

          for(int j = 0; j < LONGEST_PATTERN && valid_curr && j < (smile_length - index); j++){ //for loop per pattern matching
            const int next_i = curr.neighbor[smiles_host_dev(smile_index + index + j) - NOT_PRINTABLE];
            if (next_i) {
              curr = gpu_dict(next_i + curr_id);
              curr_id = next_i + curr_id;
              if (curr.pattern != 0) {
                const auto next = index + j + 1;
                const auto w = score_matrix(id, next) + 1;
                if (w < score_matrix(id, index)) {
                  score_matrix(id, index) = w;
                  pattern_matrix(id, index) = curr.pattern;
                  length_matrix(id, index) = j + 1;
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
            if (pattern_matrix(id, idx) != 0) {
             smiles_out_dev(smile_index + out_idx) = pattern_matrix(id, idx);
             ++out_idx;
             idx += length_matrix(id, idx);
            } else {
              smiles_out_dev(smile_index + out_idx) = smiles_dictionary_escape_char;
              ++out_idx;
              smiles_out_dev(smile_index + out_idx) = smiles_host_dev(smile_index + idx);
              ++out_idx;
              idx += 1;
            }
          }
          smiles_out_dev(smile_index + out_idx) = '\0'; // Terminate the SMILES string
          // fai una kokkos print della stringa
          //stampa le matrici di score, pattern e length // Solo il thread 0 stampa
          Kokkos::printf("[DEBUG] Compressed SMILES #%d: ", id + 1);
          for (int j = 0; j < out_idx; ++j) {
            Kokkos::printf("%c", smiles_out_dev(smile_index + j));
          }
          Kokkos::printf("\n");
        }
      );
      Kokkos::fence();

  // Reset contatori
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

