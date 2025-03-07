# ########################################################################
# ### Define the source files that compose the library
# ########################################################################
set(header_path "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(header_prefix "zsmiles")
list(APPEND header_files
  "${header_path}/${header_prefix}/cpu/compressor.hpp"
  "${header_path}/${header_prefix}/cpu/dictionary.hpp"
  "${header_path}/${header_prefix}/compressor_implementations.hpp"
  "${header_path}/${header_prefix}/dictionary_graph.hpp"
  "${header_path}/${header_prefix}/utils.hpp"
)

set(source_path "${CMAKE_CURRENT_SOURCE_DIR}/src")
list(APPEND source_files
  "${source_path}/cpu/compressor.cpp"
  "${source_path}/cpu/dictionary.cpp"
)

list(APPEND gpu_header_files
  "${header_path}/${header_prefix}/gpu/dictionary.hpp"
  "${header_path}/${header_prefix}/gpu/node.hpp"
  "${header_path}/${header_prefix}/gpu/knobs.hpp"
)

list(APPEND cuda_header_files
  "${header_path}/${header_prefix}/cuda/compressor.cuh"
  "${header_path}/${header_prefix}/cuda/nvidia_helper.cuh"
)

list(APPEND cuda_source_files
  "${source_path}/cuda/compressor.cu"
)

list(APPEND hip_header_files
  "${header_path}/${header_prefix}/hip/compressor.hpp"
  "${header_path}/${header_prefix}/hip/hip_helper.hpp"
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
    set_source_files_properties(${hip_source_files} PROPERTIES LANGUAGE HIP)
    list(APPEND header_files ${hip_header_files})
  endif()
endif()

# ########################################################################
# ### Define how we should compile the library
# ########################################################################

# define the library compilation steps
add_library(libsmiles STATIC ${source_files} ${header_files})
add_library(smiles::libsmiles ALIAS libsmiles)
set_target_properties(libsmiles PROPERTIES PUBLIC_HEADER "${headers}")
target_include_directories(libsmiles PUBLIC "${header_path}")
target_link_libraries(libsmiles PUBLIC Boost::graph)
set_target_properties(libsmiles
    PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
  )

if(ENABLE_CUDA)
  # Compile commands for CUDA includes https://discourse.cmake.org/t/cmake-target-include-directories-doesnt-export-to-compile-commands-json-when-using-cuda/10072/9
  set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
  set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
  set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)

  set_target_properties(libsmiles
      PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
      CUDA_EXTENSIONS OFF
      CUDA_SEPARABLE_COMPILATION ON
  )
  target_compile_definitions(libsmiles PUBLIC -DENABLE_CUDA_IMPLEMENTATION)
endif()

if(ENABLE_HIP)
  set(CMAKE_HIP_USE_RESPONSE_FILE_FOR_INCLUDES 0)
  set(CMAKE_HIP_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
  set(CMAKE_HIP_USE_RESPONSE_FILE_FOR_OBJECTS 0)
  # TODO check on AMD
  set_property(TARGET libsmiles PROPERTY HIP_ARCHITECTURES ${GPU_ARCHITECTURE})
  target_compile_definitions(libsmiles PUBLIC -DENABLE_HIP_IMPLEMENTATION)
  target_include_directories(libsmiles PRIVATE ${hip_INCLUDE_DIRS})
  # TODO check on AMD
  set_source_files_properties(${hip_source_files} PROPERTIES COMPILE_OPTIONS "--expt-relaxed-constexpr")
endif()

if(ENABLE_LIKWID)
  target_include_directories(libsmiles PRIVATE ${LIKWID_INCLUDE_DIR})
  target_compile_definitions(libsmiles PUBLIC -DLIKWID_PERFMON)
  if(ENABLE_CUDA)
    target_compile_definitions(libsmiles PUBLIC -DLIKWID_NVMON)
  endif()
endif()

# set its name to avoid the double lib
set_target_properties(libsmiles PROPERTIES PREFIX "" OUTPUT_NAME "libsmiles")
