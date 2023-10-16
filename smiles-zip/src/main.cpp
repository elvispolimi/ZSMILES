#include "cpu/compressor.hpp"
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
namespace po = boost::program_options;

#include "compressor_implementations.hpp"

int main(int argc, char* argv[]) {
  std::string input_file;
  std::string output_file;

  // parse the command line arguments
  po::variables_map vm;
  po::options_description app_description("Global Options of the executable");
  app_description.add_options()("help,h", "Print this help message");
  app_description.add_options()("input-file,i", po::value<std::string>(&input_file), "Input file");
  app_description.add_options()("output-file,o", po::value<std::string>(&output_file), "Output file");
  app_description.add_options()("compress,c", "Compress the input file");
  // TODO at the moment single GPU
  app_description.add_options()("cuda", "Enable cuda implementation");
  app_description.add_options()("decompress,d", "Decompress the input file");
  po::store(po::command_line_parser(argc, argv).options(app_description).run(), vm);
  if (vm.count("help") > 0) {
    std::cout << "This application prints in the standard output the compresses or decompress SMILES of each one that it "
                 "reads from the standard input."
              << std::endl;
    std::cout << std::endl;
    std::cout << "USAGE: " << argv[0] << "-i input_file -o output_file -c/-d" << std::endl;
    std::cout << std::endl;
    std::cout << app_description << std::endl;
    std::cout << std::endl;
    return EXIT_SUCCESS;
  }
  po::notify(vm);

  // Check if either compress or decompress is specified
  if (vm.count("compress")) {
    // declare the functor that performs the conversion
    smiles::compressor_container compress_cont;

    if(vm.count("cuda")){
      // TODO
    } else
      compress_cont.create<smiles::cpu::smiles_compressor>();

    // perform the translation
    std::ifstream i_file(input_file); // Open the file
    std::ofstream o_file(output_file); // Open the file
    for (std::string line; std::getline(i_file, line); /* automatically handled */) {
      o_file << compress_cont(line) << std::endl;
    }
    i_file.close();
    o_file.close();
  } else if (vm.count("decompress")) {
    // declare the functor that performs the conversion
    smiles::decompressor_container decompress_cont;

    if(vm.count("cuda")){
      // TODO
    } else
      decompress_cont.create<smiles::cpu::smiles_decompressor>();

    // perform the translation
    std::ifstream i_file(input_file); // Open the file
    std::ofstream o_file(output_file); // Open the file
    for (std::string line; std::getline(i_file, line); /* automatically handled */) {
      o_file << decompress_cont(line) << std::endl;
    }
    i_file.close();
    o_file.close();
  } else {
    std::cerr << "Error: You must specify either --compress or --decompress." << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}