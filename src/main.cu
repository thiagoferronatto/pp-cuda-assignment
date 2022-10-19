/**
 * COMPLETE IMPLEMENTATION (MULTIPLE BLOCKS IN GRID)
 *
 * Please use the makefile, works perfectly both in our machines and on Colab.
 *
 * @file main.cu
 * @authors Diego F. S. Souza (diego.f.s.souza@ufms.br) and Thiago Ferronatto
 * (thiago.ferronatto@ufms.br)
 * @brief Reads an input file, calls functions from the other files and outputs
 * the desired results to stdout.
 * @version 1.0
 * @date 2022-10-18
 *
 * @copyright Copyright (c) 2022
 */

#include <cstdio>
#include <ctime>

#include "../include/genetic_distance.cuh"

int main(int argc, char** argv, char** envp) {
  namespace ca = cuda_assignment;

  // Checking number of command-line arguments, exiting on failure
  if (argc != 2) exit(fputs("Syntax: ./program <input file path>\n", stderr));

  // Trying to open input file, exiting on failure
  FILE* input_file_header = fopen(argv[1], "r");
  if (!input_file_header) exit(fputs("Input file not found\n", stderr));

  // Reading lengths of both DNA sequences
  ca::types::U16 s_length, r_length;
  fscanf(input_file_header, "%hu %hu", &s_length, &r_length);
  fgetc(input_file_header);

  // Reading shorter DNA sequence
  ca::types::Base* sequence_s = new ca::types::Base[s_length];
  for (ca::types::U16 i = 0; i < s_length; i++)
    sequence_s[i] = (ca::types::Base)fgetc(input_file_header);
  fgetc(input_file_header);

  // Reading longer DNA sequence
  ca::types::Base* sequence_r = new ca::types::Base[r_length];
  for (ca::types::U16 i = 0; i < r_length; i++)
    sequence_r[i] = (ca::types::Base)fgetc(input_file_header);

  // Cleaning up
  fclose(input_file_header);

  // Using C's internal clock for perf measuring, fight me
  clock_t start = clock();

  // All the boilerplate for launching the kernel
  const ca::types::U16 result =
      ca::algorithm::LaunchKernel(sequence_r, sequence_s, r_length, s_length);

  clock_t end = clock();

  // Finally printing the results
  printf("%hu\n%ld\n", result, end - start);

  return 0;
}
