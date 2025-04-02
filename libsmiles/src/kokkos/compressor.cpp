#include "zsmiles/kokkos/compressor.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <Kokkos_Core.hpp>
#include <cstdlib>
#include <zsmiles/compression_dictionary.hpp>

using namespace smiles::kokkos;

// Implementazioni minimali
void smiles_compressor::compress(std::ifstream& o_file) {
}

void smiles_compressor::clean_up(std::ofstream& o_file) {
    o_file << "[Kokkos] Cleanup completato\n";
}


// Funzione device per decomprimere una SMILE usando il dizionario.
// compressed: puntatore all'input compresso
// compressed_len: lunghezza dell'input
// decompressed: buffer di output (massimo max_decompressed caratteri)
// Restituisce la lunghezza effettiva dell'output scritto.
KOKKOS_INLINE_FUNCTION
size_t decompress_smile_using_dict(const char* compressed,
                                   const size_t compressed_len,
                                   char* decompressed,
                                   const size_t max_decompressed) {
  size_t out_index = 0;
  for (size_t i = 0; i < compressed_len && out_index < max_decompressed; ++i) {
    char c = compressed[i];
    if (c != smiles_dictionary_escape_char) {
      // Usa il dizionario: l'indice è il valore numerico del carattere
      const auto& entry = smiles::SMILES_DICTIONARY[static_cast<unsigned char>(c)];
      // Copia il pattern dell'entry nel buffer di output, se c'è spazio
      for (size_t j = 0; j < entry.size && out_index < max_decompressed; ++j) {
        decompressed[out_index++] = entry.pattern[j];
      }
    } else {
      // Se è il carattere di escape, copia il carattere successivo letteralmente
      if (i + 1 < compressed_len && out_index < max_decompressed) {
        ++i;
        decompressed[out_index++] = compressed[i];
      }
    }
  }
  return out_index;
}

namespace smiles {

  struct smiles_decompressor {
    void decompress(std::ifstream& i_file, std::ofstream& o_file);
  };

} // namespace smiles

// Implementazione della decompressione usando Kokkos
void smiles::smiles_decompressor::decompress(std::ifstream& i_file, std::ofstream& o_file) {

  Kokkos::initialize();
  {
    // Stima massima di caratteri decompressi per ogni SMILE
    const size_t MAX_DECOMPRESSED_SIZE = 128;
    
    // Lettura del file su host: ogni riga rappresenta una SMILE compressa
    std::vector<char> charArray;
    std::vector<size_t> offsets;
    size_t current_offset = 0;
    std::string line;
    while (std::getline(i_file, line)) {
      offsets.push_back(current_offset);
      charArray.insert(charArray.end(), line.begin(), line.end());
      current_offset = charArray.size();
    }
    // Aggiungi l'offset finale per definire la fine dell'ultima SMILE
    offsets.push_back(current_offset);

    // Creazione delle views per l'input compresso
    Kokkos::View<size_t*> smiles_offsets("smiles_offsets", offsets.size());
    Kokkos::View<char*> smiles_data("smiles_data", charArray.size());
    {
      auto host_offsets = Kokkos::View<size_t*, Kokkos::HostSpace>(offsets.data(), offsets.size());
      auto host_data    = Kokkos::View<char*, Kokkos::HostSpace>(charArray.data(), charArray.size());
      Kokkos::deep_copy(smiles_offsets, host_offsets);
      Kokkos::deep_copy(smiles_data, host_data);
    }
    
    // Numero di SMILES (righe)
    const size_t num_smiles = offsets.size() - 1;

    // Allocazione di una view 2D per i risultati decompressi:
    // ogni riga ha spazio per MAX_DECOMPRESSED_SIZE caratteri
    Kokkos::View<char**> smiles_decompressed("smiles_decompressed", num_smiles, MAX_DECOMPRESSED_SIZE);
    // View per memorizzare la lunghezza effettiva di ciascuna decompressione
    Kokkos::View<size_t*> smiles_decompressed_lengths("smiles_decompressed_lengths", num_smiles);

    // Parallel_for: decompressione per ogni SMILE in parallelo
    Kokkos::parallel_for("Smiles decompression", num_smiles, KOKKOS_LAMBDA(const size_t idx) {
      size_t input_start = smiles_offsets(idx);
      size_t input_end   = smiles_offsets(idx+1);
      size_t compressed_len = input_end - input_start;
      const char* compressed_ptr = &smiles_data(input_start);
      // Puntatore alla riga di output per questa SMILE
      char* out_ptr = &smiles_decompressed(idx, 0);
      // Decomprimi usando il dizionario; la funzione restituisce la lunghezza scritta
      size_t decompressed_len = decompress_smile_using_dict(compressed_ptr, compressed_len, out_ptr, MAX_DECOMPRESSED_SIZE);
      smiles_decompressed_lengths(idx) = decompressed_len;
    });

    // Fase di scrittura sequenziale: copia i risultati su host e scrivi nel file
    {
      // Copia le lunghezze su host
      std::vector<size_t> host_decompressed_lengths(num_smiles);
      {
        auto h_lengths = Kokkos::create_mirror_view(smiles_decompressed_lengths);
        Kokkos::deep_copy(h_lengths, smiles_decompressed_lengths);
        for (size_t i = 0; i < num_smiles; ++i) {
          host_decompressed_lengths[i] = h_lengths(i);
        }
      }
      
      // Costruisci un vettore di stringhe per il risultato
      std::vector<std::string> decompressed_smiles(num_smiles);
      for (size_t i = 0; i < num_smiles; ++i) {
        // Copia la riga i-esima dalla view 2D su un vettore host temporaneo
        std::vector<char> host_row(MAX_DECOMPRESSED_SIZE);
        for (size_t j = 0; j < MAX_DECOMPRESSED_SIZE; ++j) {
          host_row[j] = smiles_decompressed(i, j);
        }
        // Costruisci la stringa usando solo i caratteri validi (secondo la lunghezza salvata)
        decompressed_smiles[i] = std::string(host_row.data(), host_decompressed_lengths[i]);
      }
      
      // Scrive le SMILES decompressi, una per riga, nel file di output
      for (const auto& smile : decompressed_smiles) {
        o_file << smile << "\n";
      }
    }
  }
  Kokkos::finalize();
}



void smiles_decompressor::clean_up(std::ofstream& o_file) {
    o_file << "[Kokkos] Decompress cleanup\n";
}