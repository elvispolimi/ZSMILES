# ########################################################################
# ### Define the source files
# ########################################################################
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}/src")
list(APPEND source_files
  "${source_path}/main.cpp"
)

# ########################################################################
# ### Define how we should compile the main executable
# ########################################################################
add_executable(smiles-zip ${header_files} ${source_files})
add_executable(smiles::smiles-zip ALIAS smiles-zip)
target_include_directories(smiles-zip PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(smiles-zip PRIVATE Boost::program_options smiles::libsmiles)
set_target_properties(smiles-zip
    PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
  )

if(ENABLE_LIKWID)
  target_include_directories(smiles-zip PRIVATE ${LIKWID_INCLUDE_DIR})
  target_link_libraries(smiles-zip PRIVATE likwid::likwid)
  target_compile_definitions(libsmiles PUBLIC -DLIKWID_PERFMON)
endif()
