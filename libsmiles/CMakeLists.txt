#########################################################################
#### Define the source files that compose the library
#########################################################################
set(header_path "${CMAKE_CURRENT_SOURCE_DIR}/include")
list(APPEND header_files
  "${header_path}/cpu/compressor.hpp"
  "${header_path}/cpu/dictionary.hpp"
  "${header_path}/compressor_implementations.hpp"
  "${header_path}/dictionary_graph.hpp"
  "${header_path}/utils.hpp"
)

set(source_path "${CMAKE_CURRENT_SOURCE_DIR}/src")
list(APPEND source_files
  "${source_path}/cpu/compressor.cpp"
  "${source_path}/cpu/dictionary.cpp"
)

list(APPEND gpu_header_files
  "${header_path}/gpu/compressor.hpp"
  "${header_path}/gpu/dictionary.hpp"
  "${header_path}/gpu/node.hpp"
)

list(APPEND cuda_header_files
  "${header_path}/cuda/nvidia_helper.cuh"
)

list(APPEND hip_header_files
  "${header_path}/hip/nvidia_helper.hpp"
)

list(APPEND hip_source_files
  "${source_path}/hip/compressor.cpp"
)

if(ENABLE_CUDA OR ENABLE_HIP)
  list(APPEND header_files ${gpu_header_files})
  if(ENABLE_CUDA)
    list(APPEND source_files ${cuda_source_files})
    list(APPEND header_files ${cuda_header_files})
  endif()
  if(ENABLE_HIP)
    list(APPEND source_files ${hip_source_files})
    list(APPEND header_files ${hip_header_files})
  endif()
endif()

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
add_CUDA_dependency(libliglibsmilesen)
set_cxx_flags(libsmiles)
if(ENABLE_CUDA)
  set_cuda_flags(libsmiles)
  target_compile_definitions(libsmiles PUBLIC ENABLE_CUDA_IMPLEMENTATION)
  target_compile_options(libsmiles PUBLIC $<$<AND:$<CONFIG:DEBUG>,$<COMPILE_LANGUAGE:CUDA>>:-G>)
endif()
if(ENABLE_HIP)
  target_link_libraries(libsmiles PRIVATE OpenMP::OpenMP_CXX)
endif()


# set its name to avoid the double lib
set_target_properties(libsmiles PROPERTIES PREFIX "" OUTPUT_NAME "libsmiles")