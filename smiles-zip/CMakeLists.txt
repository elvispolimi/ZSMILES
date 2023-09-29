#########################################################################
#### Define the source files
#########################################################################
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}/src")
list(APPEND source_files
  "${source_path}/main.cpp"
)



#########################################################################
#### Define how we should compile the main executable
#########################################################################

add_executable(smiles-zip ${header_files} ${source_files})
add_executable(smiles::smiles-zip ALIAS smiles-zip)
target_include_directories(smiles-zip PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(smiles-zip PRIVATE Boost::program_options smiles::libsmiles)