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
        const size_t length = d_smiles_len(id);
        const size_t in_idx = d_smiles_index(id);
        const size_t out_idx = d_smiles_index_out(id);

        size_t wp = out_idx; // Write pointer per l'output

        for (size_t k = 0; k < length; ++k) {
            char c = d_smiles_host(in_idx + k); // Legge il carattere corrente
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

smiles_compressor::smiles_compressor() {
  std::cout << "Initializing SMILES compressor..." << std::endl;
}

smiles_compressor::~smiles_compressor() {
  std::cout << "Finalizing SMILES compressor..." << std::endl;
}

void smiles_compressor::compress(std::ofstream& out_s) {
  // TODO: implement
}

void smiles_compressor::clean_up(std::ofstream& out_s) {
  // TODO: implement
}

void smiles_compressor::test() {
  auto dict_entries = smiles::kokkos::build_gpu_smiles_dictionary_entries();

  // Crea un mirror sul lato host per verificare i dati
  auto host_dict_entries = Kokkos::create_mirror_view(dict_entries);
  Kokkos::deep_copy(host_dict_entries, dict_entries);

  // Stampa le entry del dizionario
  for (size_t i = 0; i < host_dict_entries.extent(0); ++i) {
    std::cout << "Entry " << i << ": size = " << host_dict_entries(i).size
              << ", pattern = " << host_dict_entries(i).pattern << std::endl;
  }
  size_t total_size = 0;
  Kokkos::parallel_reduce("SumPatternSizes", host_dict_entries.extent(0),
                          KOKKOS_LAMBDA(const int i, size_t& local_sum) {
                            local_sum += dict_entries(i).size;
                          },
                          total_size);

  // Stampa il risultato della somma
  std::cout << "Total size of all patterns: " << total_size << std::endl;

}

} // namespace kokkos
} // namespace smiles

