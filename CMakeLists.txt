cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(smiles-zip VERSION 0.1)

# export also compile commands to have a better integration with the ide
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ENABLE_CUDA "Enable CUDA implementations" OFF)
option(ENABLE_HIP "Enable HIP implementations" OFF)
set(GPU_ARCHITECTURE "none" CACHE STRING "Optimize for the target GPU architecture (none to disable)")
set(CPU_ARCHITECTURE "native" CACHE STRING "Optimize for the target CPU architecture (none to disable)")

# build Release version unless the user specify something else
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "The type of build" FORCE)
  message(STATUS "Setting build type to '${CMAKE_BUILD_TYPE}' as none was specified")
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=${CPU_ARCHITECTURE}")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=${CPU_ARCHITECTURE}")
endif()

# we need to enable the language as soon as possible
enable_language(CXX)

if(ENABLE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "${GPU_ARCHITECTURE}")
  endif()

  set(CMAKE_CUDA_FLAGS "--expt-relaxed-constexpr")
  set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -use_fast_math --expt-relaxed-constexpr")
  set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-G -g -Xptxas -v --expt-relaxed-constexpr")
  set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -Xptxas -v --expt-relaxed-constexpr")
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

if(ENABLE_HIP)
  # TODO test if this work and if it is needed
  find_package(HIP REQUIRED)
  enable_language(HIP)

  # Set HIP target architectures for both AMD and NVIDIA
  if(HIP_PLATFORM STREQUAL "amd")
    set(CMAKE_HIP_ARCHITECTURES "gfx906;gfx908;gfx90a")  # Adjust for your AMD GPU
  elseif(HIP_PLATFORM STREQUAL "nvidia")
    set(CMAKE_HIP_ARCHITECTURES "70;75;80")  # Adjust for your NVIDIA GPU
  endif()
endif()

# ########################################################################
# ### Define the configuration knobs to drive the building process
# ########################################################################

# define the global options that drive the compilation process
option(ENABLE_TEST "Enable the smiles tests" OFF)

# ########################################################################
# ### Figure out dependencies and custom compilation flags
# ########################################################################
# find the external packages
find_package(Boost REQUIRED COMPONENTS graph program_options)

# ########################################################################
# ### Profiling
# ########################################################################
option(ENABLE_LIKWID "Enable profiling using likwid" OFF)
if(ENABLE_LIKWID)
  find_package(likwid REQUIRED)
endif()

# ########################################################################
# ### Add source files
# ########################################################################
add_subdirectory(libsmiles)
add_subdirectory(smiles-zip)

if(ENABLE_TEST)
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
message(STATUS "CXX compiler........................: ${CMAKE_CXX_COMPILER} (CMAKE_CXX_COMPILER)")
message(STATUS "Build type..........................: ${CMAKE_BUILD_TYPE} (CMAKE_BUILD_TYPE)")
message(STATUS "Install prefix......................: ${CMAKE_INSTALL_PREFIX} (CMAKE_INSTALL_PREFIX)")
message(STATUS "Enable CUDA.........................: ${ENABLE_CUDA} (ENABLE_HIP)")
message(STATUS "Enable HIP..........................: ${ENABLE_HIP} (ENABLE_HIP)")
if(ENABLE_CUDA OR ENABLE_HIP)
  message(STATUS "Target GPU architecture.............: ${GPU_ARCHITECTURE} (GPU_ARCHITECTURE)")
endif()
message(STATUS "Target CPU architecture.............: ${CPU_ARCHITECTURE} (CPU_ARCHITECTURE)")
message(STATUS "Enable TEST.........................: ${ENABLE_TEST} (ENABLE_TEST)")
message(STATUS "Enable LIKWID.......................: ${ENABLE_LIKWID} (ENABLE_LIKWID)")


