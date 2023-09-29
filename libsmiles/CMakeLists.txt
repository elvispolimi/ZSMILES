#########################################################################
#### Define the source files that compose the library
#########################################################################
set(header_path "${CMAKE_CURRENT_SOURCE_DIR}/include")
list(APPEND header_files
  "${header_path}/compression_dictionary.hpp"
  "${header_path}/compressor.hpp"
  "${header_path}/dictionary_graph.hpp"
)

set(source_path "${CMAKE_CURRENT_SOURCE_DIR}/src")
list(APPEND source_files
  "${source_path}/compression_dictionary.cpp"
  "${source_path}/compressor.cpp"
)

#########################################################################
#### Define how we should compile the library
#########################################################################

# define the library compilation steps
add_library(libsmiles STATIC ${source_files} ${header_files})
add_library(smiles::libsmiles ALIAS libsmiles)
set_target_properties(libsmiles PROPERTIES PUBLIC_HEADER "${headers}")
target_include_directories(libsmiles PUBLIC "${header_path}")
set_target_properties(libsmiles PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(libsmiles PUBLIC Boost::graph)

# set its name to avoid the double lib
set_target_properties(libsmiles PROPERTIES PREFIX "" OUTPUT_NAME "libsmiles")