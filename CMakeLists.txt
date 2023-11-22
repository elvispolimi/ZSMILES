cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(smiles-zip VERSION 0.1)

# export also compile commands to have a better integration with the ide
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ENABLE_CUDA "Enable CUDA implementations" OFF)
option(ENABLE_HIP "Enable HIP implementations" OFF)
set(GPU_ARCHITECTURE "none" CACHE STRING "Optimize for the target GPU architecture (none to disable)")
set_property(CACHE GPU_ARCHITECTURE PROPERTY STRINGS ${gpu_architectures})

function(remove_version_from_filename input_filename version output_filename)
    # Extract the directory path and filename without extension
    get_filename_component(directory_path ${input_filename} DIRECTORY)
    get_filename_component(filename_without_extension ${input_filename} NAME_WE)

    # Reconstruct the modified filename with the directory path
    set(${output_filename} "${directory_path}/${filename_without_extension}.so" PARENT_SCOPE)
endfunction()


# we need to enable the language as soon as possible
enable_language(CXX)

if(ENABLE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "${GPU_ARCHITECTURE}")
  endif()

  enable_language(CUDA)
  find_package(CUDAToolkit)
endif()

if(ENABLE_HIP)
# TODO test if this work and if it is needed
  set(CMAKE_CXX_COMPILER hipcc)
  find_package(hip)
endif()

# define the function to handle the CUDA implementation
function(add_CUDA_dependency target_name)
  if(WITH_CUDA)
    set_target_properties(${target_name}
      PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
      CUDA_EXTENSIONS OFF
      CUDA_SEPARABLE_COMPILATION ON)
  endif()
endfunction()

# define the function to handle compilation flags custom compilation flags
function(set_cxx_flags target)
  set_target_properties(${target}
    PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
  )
endfunction(set_cxx_flags)

# function(set_cxx_sources_flags sources)
#   list(APPEND cxx_release_flags "-march=native")

#   # define the flag to set more warning message during the compilation
#   if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
#     list(APPEND cxx_flags "-Wall" "-Wextra" "-pedantic")
#   elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
#     list(APPEND cxx_flags
#       "-Weverything"
#       "-pedantic"
#       "-Wno-c++98-compat"
#       "-Wno-pre-c++14-compat"
#       "-Wno-c++98-compat-pedantic"
#       "-Wno-pre-c++14-compat-pedantic")
#   endif()
#   set_source_files_properties(my_file.cpp PROPERTIES COMPILE_OPTIONS "-Wall;-Wextra;-pedantic")
#   target_compile_options(${target} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>)
#   target_compile_options(${target} PUBLIC $<$<AND:$<CONFIG:RELEASE>,$<COMPILE_LANGUAGE:CXX>>:${cxx_release_flags}>)
# endfunction(set_cxx_sources_flags)

# define the function to handle the CUDA implementation
function(set_cuda_flags target_name)
  if(ENABLE_CUDA)
    set_target_properties(${target_name}
      PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
      CUDA_EXTENSIONS OFF
      CUDA_SEPARABLE_COMPILATION ON)
    list(APPEND cuda_flags
      "--expt-relaxed-constexpr"
      "--default-stream" "per-thread"
      "--use_fast_math"
      "-expt-relaxed-constexpr"
      "-Wno-pedantic"
      "-ftemplate-depth=256")
    list(APPEND cuda_debug_flags
      "-G")
    target_compile_options(${target_name} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${cuda_flags}>)
    target_compile_options(${target_name} PUBLIC $<$<AND:$<CONFIG:DEBUG>,$<COMPILE_LANGUAGE:CUDA>>:${cuda_debug_flags}>)
  endif()
endfunction()

# ########################################################################
# ### Define the configuration knobs to drive the building process
# ########################################################################

# define the global options that drive the compilation process
option(SMILES_TESTS "Enable the smiles tests" OFF)

# ########################################################################
# ### Figure out dependencies and custom compilation flags
# ########################################################################
# find the external packages
find_package(Boost REQUIRED COMPONENTS graph program_options)
remove_version_from_filename(${Boost_GRAPH_LIBRARY_RELEASE} ${boost_graph_VERSION} Boost_GRAPH_LIBRARY_RELEASE)
remove_version_from_filename(${Boost_PROGRAM_OPTIONS_LIBRARY_RELEASE} ${boost_program_options_VERSION} Boost_PROGRAM_OPTIONS_LIBRARY_RELEASE)
add_subdirectory(libsmiles)
add_subdirectory(smiles-zip)

if(SMILES_TESTS)
  enable_testing()
  find_package(Python COMPONENTS Interpreter REQUIRED)
  set(TEST_DATA "${CMAKE_CURRENT_LIST_DIR}/data")
  add_subdirectory(test) # end-to-end test
endif()

# ########################################################################
# ### Print a small recap of the smiles zipper configuration
# ########################################################################
message(STATUS "")
message(STATUS "//===---------------------------===//")
message(STATUS "//     SMILES ZIP CONFIGURATION    //")
message(STATUS "//===---------------------------===//")
