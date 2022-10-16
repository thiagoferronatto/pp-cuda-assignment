#include <cstdio>

#include "genetic_distance.cuh"

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
  (void)fgetc(input_file_header);

  // Reading shorter DNA sequence
  ca::types::Base* sequence_s = new ca::types::Base[s_length];
  for (ca::types::U16 i = 0; i < s_length; i++)
    sequence_s[i] = (ca::types::Base)fgetc(input_file_header);
  (void)fgetc(input_file_header);

  // Reading longer DNA sequence
  ca::types::Base* sequence_r = new ca::types::Base[r_length];
  for (ca::types::U16 i = 0; i < r_length; i++)
    sequence_r[i] = (ca::types::Base)fgetc(input_file_header);

  // Cleaning up
  fclose(input_file_header);

  // All the boilerplate for launching the kernel
  const ca::types::U16* result =
      ca::algorithm::LaunchKernel(sequence_r, sequence_s, r_length, s_length);

  // TODO: remove this, prints the whole DP matrix
  // for (auto i = 0U; i <= s_length; i++) {
  //   for (auto j = 0U; j <= r_length; j++)
  //     printf("%u%s", result[i * (r_length + 1) + j], j == r_length ? "" :
  //     ",");
  //   puts("");
  // }

  // TODO: measure and print exec time
  printf("%hu\n", result[s_length * (r_length + 1) + r_length]);

  return 0;
}
