cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(smiles-zip VERSION 0.1)

# export also compile commands to have a better integration with the ide
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#########################################################################
#### Figure out dependencies and custom compilation flags
#########################################################################
# find the external packages
find_package(Boost REQUIRED COMPONENTS graph program_options)

add_subdirectory(libsmiles)
add_subdirectory(smiles-zip)

#########################################################################
#### Print a small recap of the smiles zipper configuration
#########################################################################
message(STATUS "" )
message(STATUS "//===---------------------------===//" )
message(STATUS "//     SMILES ZIP CONFIGURATION    //" )
message(STATUS "//===---------------------------===//" )
