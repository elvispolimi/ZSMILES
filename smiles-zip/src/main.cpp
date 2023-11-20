#include "compression_dictionary.hpp"
#include "cpu/compressor.hpp"
#include "cuda/compressor.cuh"

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <cstdlib>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iostream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
namespace po = boost::program_options;

#include "compressor_implementations.hpp"

#define BUFFER_SIZE 1024 * 16

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
  app_description.add_options()("verbose,v", "Verbose output and stats");
  // TODO at the moment single GPU
  app_description.add_options()("cuda", "Enable cuda implementation");
  app_description.add_options()("decompress,d", "Decompress the input file");
  po::store(po::command_line_parser(argc, argv).options(app_description).run(), vm);
  if (vm.count("help") > 0) {
    std::cout
        << "This application prints in the standard output the compresses or decompress SMILES of each one that it "
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

  // Add tier measurement for verbose output
  std::chrono::_V2::system_clock::time_point start_time, end_time;
  if (vm.count("verbose"))
    start_time = std::chrono::high_resolution_clock::now(); // start timer

  // Check if either compress or decompress is specified
  if (vm.count("compress")) {
    // Open files for input and output
    std::ifstream i_file(input_file);  // Open the file
    std::ofstream o_file(output_file); // Open the file
    if (vm.count("cuda")) {
#ifdef ENABLE_CUDA_IMPLEMENTATION
      // declare the functor that performs the conversion
      smiles::cuda::smiles_compressor compress_cont;
      std::string line;
      while (std::getline(i_file, line)) {
        if (compress_cont.smiles_len.size() >= SMILES_PER_DEVICE) {
          compress_cont.compress(o_file);
        }
        assert(line.size() < MAX_SMILES_LEN);
        compress_cont.smiles_index.push_back(compress_cont.smiles_host.size());
        compress_cont.smiles_len.push_back(line.size());
        compress_cont.smiles_host.append(line);
      }
      compress_cont.clean_up(o_file);
#else
      throw std::runtime_error("CUDA implementation required but not available");
#endif
    } else {
#pragma omp parallel num_threads(10)
      {
        std::string line;
        // declare the functor that performs the conversion
#pragma omp master
        while (std::getline(i_file, line))
#pragma omp task
        {
          smiles::cpu::smiles_compressor compress_cont;
          compress_cont(line, o_file);
        }
#pragma omp taskwait
      }
    }
    i_file.close();
    o_file.close();
  } else if (vm.count("decompress")) {
    // Open files for input and output
    std::ifstream i_file(input_file);  // Open the file
    std::ofstream o_file(output_file); // Open the file
    if (vm.count("cuda")) {
#ifdef ENABLE_CUDA_IMPLEMENTATION
      // declare the functor that performs the conversion
      smiles::cuda::smiles_decompressor decompress_cont;
      std::string line;

      int fd = open(input_file.c_str(), O_RDONLY);
      if (fd == -1) {
        std::cerr << "Failed to open file: " << input_file << std::endl;
        return EXIT_FAILURE;
      }

      struct stat sb;
      if (fstat(fd, &sb) == -1) {
        std::cerr << "Failed to get file size: " << input_file << std::endl;
        close(fd);
        return EXIT_FAILURE;
      }

      char* file_data = static_cast<char*>(mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0));
      if (file_data == MAP_FAILED) {
        std::cerr << "Failed to map file: " << input_file << std::endl;
        close(fd);
        return EXIT_FAILURE;
      }

      const char* start = file_data;
      const char* end   = file_data + sb.st_size;
      while (start < end) {
        const char* next = static_cast<const char*>(memchr(start, '\n', end - start));
        if (next == nullptr) {
          next = end;
        }
        const auto str_v = std::string_view(start, next - start);

        auto prev_end = decompress_cont.smiles_index_out.size() == 0
                            ? 0
                            : decompress_cont.smiles_index_out.back() +
                                  (decompress_cont.smiles_len.back()) * LONGEST_PATTERN + 1;
        if ((decompress_cont.smiles_index.size() > 0 &&
             (prev_end + str_v.size() * LONGEST_PATTERN + 1) >= CHAR_PER_DEVICE) ||
            decompress_cont.smiles_len.size() >= SMILES_PER_DEVICE) {
          decompress_cont.decompress(o_file);
        }

        prev_end = decompress_cont.smiles_index_out.size() == 0
                       ? 0
                       : decompress_cont.smiles_index_out.back() +
                             (decompress_cont.smiles_len.back()) * LONGEST_PATTERN + 1;

        decompress_cont.smiles_index.push_back(decompress_cont.smiles_host.size());
        // TODO check if the +1 is needed for the terminator char
        decompress_cont.smiles_index_out.push_back(prev_end);
        decompress_cont.smiles_len.push_back(str_v.size());
        decompress_cont.smiles_host.append(str_v);

        start = next + 1;
      }

      munmap(file_data, sb.st_size);
      close(fd);

      // char buffer[BUFFER_SIZE];
      // while (i_file.read(buffer, sizeof(buffer))) {
      //   long pos = 0;
      //   while (pos < i_file.gcount()) {
      //     long end = pos;
      //     int i    = 0;
      //     while (end < i_file.gcount() && buffer[end] != '\n') {
      //       ++end;
      //       i++;
      //     }

      //     // TODO check if the value SMILES_PER_DEVICE should be increased
      //     if ((decompress_cont.smiles_index.size() > 0 &&
      //          (decompress_cont.smiles_index.back() + decompress_cont.smiles_len.back() + i) >=
      //              CHAR_PER_DEVICE / LONGEST_PATTERN) ||
      //         decompress_cont.smiles_len.size() >= SMILES_PER_DEVICE) {
      //       decompress_cont.decompress(o_file);
      //     }
      //     decompress_cont.smiles_index.push_back(decompress_cont.smiles_host.size());
      //     // TODO check if the +1 is needed for the terminator char
      //     const auto prev_end = decompress_cont.smiles_index_out.size() == 0
      //                               ? 0
      //                               : decompress_cont.smiles_index_out.back() +
      //                                     (decompress_cont.smiles_len.back()) * LONGEST_PATTERN;
      //     decompress_cont.smiles_index_out.push_back(prev_end);
      //     decompress_cont.smiles_len.push_back(i);
      //     decompress_cont.smiles_host.append(buffer + pos, i);
      //     pos = end + 1;
      //   }
      // }

      // TODO check fastest reading method
      // while (std::getline(i_file, line)) {
      //   if ((decompress_cont.smiles_index.size() > 0 &&
      //        (decompress_cont.smiles_index.back() + decompress_cont.smiles_len.back()) * LONGEST_PATTERN >=
      //            CHAR_PER_DEVICE) ||
      //       decompress_cont.smiles_len.size() >= SMILES_PER_DEVICE) {
      //     decompress_cont.decompress(o_file);
      //   }
      //   decompress_cont.smiles_index.push_back(decompress_cont.smiles_host.size());
      //   decompress_cont.smiles_len.push_back(line.size());
      //   decompress_cont.smiles_host.append(line);
      // }

      decompress_cont.clean_up(o_file);
#else
      throw std::runtime_error("CUDA implementation required but not available");
#endif
    } else {
      std::string line;
      // Rework this version is not efficient, remove openmp spawn threads
#pragma omp parallel num_threads(10)
      {
        // declare the functor that performs the conversion
        smiles::cpu::smiles_decompressor decompress_cont;
#pragma omp master
        while (std::getline(i_file, line))
#pragma omp task
        {
          decompress_cont(line, o_file);
        }
#pragma omp taskwait
      }
    }
    i_file.close();
    o_file.close();
  } else {
    std::cerr << "Error: You must specify either --compress or --decompress." << std::endl;
    return EXIT_FAILURE;
  }

  if (vm.count("verbose")) {
    end_time = std::chrono::high_resolution_clock::now(); // end timer
    // print time taken and throughput
    auto input_size  = std::filesystem::file_size(input_file);
    auto output_size = std::filesystem::file_size(output_file);
    auto time_ms     = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    float input_size_mega = (float) input_size / (1000000.0);
    float time_s          = (float) time_ms / 1000.0;
    std::cout << "Compression ratio: " << (float) output_size / (float) input_size << '\n';
    std::cout << "Time taken: " << time_ms << " ms" << '\n';
    std::cout << "Throughput: " << input_size_mega / time_s << " MB/s" << '\n';
  }

  return EXIT_SUCCESS;
}