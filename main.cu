#include <cmath>
#include <cstdint>
#include <cstdio>

#include "genetic_distance.cuh"

int main(int argc, char** argv, char** envp) {
  namespace ca = cuda_assignment;

  // TODO: take exactly one command-line argument as the path of an input file,
  // read necessary data from it and output the result to screen along with
  // execution times

  constexpr auto A = ca::types::kA;
  constexpr auto C = ca::types::kC;
  constexpr auto G = ca::types::kG;
  constexpr auto T = ca::types::kT;

  constexpr ca::types::Base sequence_r[]{G, A, A, A, A, A, A, A, A, A, A,
                                         A, A, A, A, A, A, A, A, A, A, A,
                                         A, A, A, A, A, A, A, A, A, A, A};
  constexpr ca::types::Base sequence_s[]{A, A, A, A, A, A, A, A, A, A, A,
                                         A, A, A, A, A, C, A, A, A, A, A,
                                         A, A, A, A, A, A, A, A, A, T};

  ca::types::U16 r_length = sizeof(sequence_r) / sizeof(ca::types::Base);
  ca::types::U16 s_length = sizeof(sequence_s) / sizeof(ca::types::Base);

  const ca::types::U16* result =
      ca::algorithm::LaunchKernel(sequence_r, sequence_s, r_length, s_length);

  for (auto i = 0U; i <= s_length; i++) {
    for (auto j = 0U; j <= r_length; j++)
      printf("%u%s", result[i * (r_length + 1) + j], j == r_length ? "" : ",");
    puts("");
  }

  return 0;
}
