#include "zsmiles/kokkos/compressor.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <Kokkos_Core.hpp>
#include <zsmiles/compression_dictionary.hpp>

namespace smiles {
  namespace kokkos {

    struct kokkos_entry{
      int size;
      char string[15];
    };



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
          const auto& entry = SMILES_DICTIONARY[static_cast<unsigned char>(c)];
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



    // Implementazione della decompressione usando Kokkos
    void smiles_decompressor::decompress(std::ifstream& i_file, std::ofstream& o_file) {

      
      Kokkos::initialize();
      {

        // Alloca una Kokkos::View in memoria Device, eventualmente in memoria costante se supportato,
        // oppure in una memory space apposita, e copia i dati.
        Kokkos::View<kokkos_entry[DICT_SIZE], Kokkos::CudaSpace> d_dictionary("d_dictionary");

        auto dict_host = Kokkos::create_mirror_view(d_dictionary);

        for(int i = 0; i < DICT_SIZE; i++) {
          dict_host(i).size = SMILES_DICTIONARY[i].size;
          for (int j = 0; j < SMILES_DICTIONARY[i].size; j++){
            dict_host(i).string[j] = SMILES_DICTIONARY[i].pattern[j];
          }
          dict_host(i).string[SMILES_DICTIONARY[i].size] = '\0';
        }

        Kokkos::deep_copy(d_dictionary, dict_host);

        
      

        Kokkos::parallel_for(DICT_SIZE, KOKKOS_LAMBDA(const int i){
          //Kokkos::printf("size: %d\n", d_dictionary(i).size);
          Kokkos::printf("p: %s\n", d_dictionary(i).string);
        });
      }
      Kokkos::finalize();
      
    }

    void smiles_decompressor::clean_up(std::ofstream& o_file) {
        o_file << "[Kokkos] Cleanup completato\n";
    }

  } // namespace kokkos
} // namespace smiles