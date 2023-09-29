cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(smiles-zip VERSION 0.1)

# export also compile commands to have a better integration with the ide
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


#########################################################################
#### Define the configuration knobs to drive the building process
#########################################################################

# define the global options that drive the compilation process
option(SMILES_TESTS "Enable the smiles tests" OFF)


#########################################################################
#### Figure out dependencies and custom compilation flags
#########################################################################
# find the external packages
find_package(Boost REQUIRED COMPONENTS graph program_options)

add_subdirectory(libsmiles)
add_subdirectory(smiles-zip)
if (SMILES_TESTS)
  enable_testing()
  find_package(Python COMPONENTS Interpreter REQUIRED)
  set(TEST_DATA "${CMAKE_CURRENT_LIST_DIR}/data")
  add_subdirectory(test)      # end-to-end test for ligen
endif()

#########################################################################
#### Print a small recap of the smiles zipper configuration
#########################################################################
message(STATUS "" )
message(STATUS "//===---------------------------===//" )
message(STATUS "//     SMILES ZIP CONFIGURATION    //" )
message(STATUS "//===---------------------------===//" )
