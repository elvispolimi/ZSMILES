#include <Kokkos_Core.hpp>

using namespace Kokkos;

void decompress_kokkos(
    const base_compressor::smiles_type* smiles_in,
    const base_compressor::index_type* smiles_in_index,
    const base_compressor::index_type* smiles_out_index,
    base_compressor::smiles_type* smiles_out,
    const base_compressor::index_type* smiles_len,
    const int num_smiles) {

    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;

    // AVVIO DELLA PARALLELIZZAZIONE: ogni team gestisce una stringa
    Kokkos::parallel_for("Decompress SMILES", team_policy(num_smiles, Kokkos::AUTO), KOKKOS_LAMBDA(const member_type& team) {
        const int id = team.league_rank(); // ID del team (equivalente al blocco CUDA)
        const base_compressor::index_type smile_len = smiles_len[id]; // Lunghezza della stringa
        unsigned long last_index = 0; // Ultimo carattere scritto nell'output

        // Puntatori agli array di input e output
        const base_compressor::smiles_type* smiles_in_l = smiles_in + smiles_in_index[id];
        base_compressor::smiles_type* smiles_out_l = smiles_out + smiles_out_index[id];
				
        int is_previous_escape = 0; // Booleano per tenere traccia dello stato di escape

        // GESTIONE LAVORO DEI THREAD NEL TEAM => Ogni thread lavora su "pezzi" alterni della stessa stringa
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, smile_len), [&](const int i) {
            // Legge il carattere compresso
            const base_compressor::smiles_type smile_c = smiles_in_l[i];
            // Controlla se è un carattere di escape
            const int is_escape = (smile_c == smiles_dictionary_escape_char);
            
            // DETERMINA IL PATTERN DA SOSTITUIRE => Converte il carattere compresso in testo leggibile
            const auto find_index = static_cast<unsigned char>(smile_c);
            const auto find_pattern_length = is_previous_escape ? 1 : (is_escape ? 0 : smiles_dictionary_gpu[find_index].size);
            const auto find_pattern = is_previous_escape ? &smile_c : (is_escape ? "" : smiles_dictionary_gpu[find_index].pattern);
            
            // CALCOLO DELL'INDICE DI SCRITTURA DELL'OUTPUT
            size_t index = find_pattern_length;
            Kokkos::single(Kokkos::PerThread(team), [&]() {
                memcpy(&smiles_out_l[last_index], find_pattern, find_pattern_length); // Scrittura dell'output
                last_index += find_pattern_length; // Aggiornamento dell'ultimo indice scritto
            });
            is_previous_escape = is_escape; // Aggiorna lo stato di escape
        });
        
        // AGGIUNTA TERMINATORE fatta dal primo thread del team
        Kokkos::single(Kokkos::PerTeam(team), [&]() {
            smiles_out_l[last_index] = '\0';
        });
    });
}
