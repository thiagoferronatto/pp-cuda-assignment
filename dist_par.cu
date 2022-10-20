/**
 * COMPLETE IMPLEMENTATION (MULTIPLE BLOCKS IN GRID)
 *
 * Go ahead and compile with `nvcc dist_par.cu -o dist_par`
 *
 * @file dist_par.cu
 * @authors Diego F. S. Souza and Thiago Ferronatto
 *
 * @copyright Copyright (c) 2022
 */

#include <cooperative_groups.h>

#include <cstdint>
#include <cstdio>
#include <ctime>

/**
 * @brief Simple macro to check a CUDA function call for errors.
 */
#define CHECK(ans) \
  cuda_assignment::algorithm::CheckError(ans, __FILE__, __LINE__)

namespace cuda_assignment {
namespace types {

using U16 = uint16_t;
using U32 = uint32_t;

/// @brief These should make it easier to read the sequences from a file.
enum Base { kA = 'A', kC = 'C', kG = 'G', kT = 'T' };

}  // namespace types

namespace algorithm {

/**
 * @brief Function for use inside a macro to check CUDA function calls for
 * errors.
 *
 * @param error CUDA error code.
 * @param file Name of the source file where the error code was returned.
 * @param line Line of the source file where the error code was returned.
 */
static inline void CheckError(cudaError_t error, const char* file, int line) {
  const auto& GetErr = cudaGetErrorString;
  if (error != cudaSuccess) {
    fprintf(stderr, "CUDA Exception: %s @ %s:%d.\n", GetErr(error), file, line);
    exit(-1);
  }
}

/**
 * @brief Calculates the value of cell (i, j) of the DP table based on the
 * recurrence relation described inside.
 *
 * @param sequence_r Input DNA sequence, presumed longer than the second.
 * @param sequence_s Input DNA sequence, presumed shorter than the first.
 * @param i Row of the cell to be populated.
 * @param j Column of the cell to be populated.
 * @param r_length Length of the first DNA sequence.
 * @param dp_table DP table to be populated.
 */
__device__ inline void CalcCell(const types::Base* sequence_r,
                                const types::Base* sequence_s, types::U16 i,
                                types::U16 j, types::U16 r_length,
                                types::U16* dp_table) {
  // Recurrence relation for the DP algorithm is
  // d(i, j) = min{
  //   d(i, j - 1) + 1,
  //   d(i - 1, j - 1) + {0 if s[i] = r[j], 1 otherwise},
  //   d(i - 1, j) + 1
  // }
  const types::U16 left = dp_table[i * (r_length + 1) + j - 1] + 1;
  const types::U16 top_left = dp_table[(i - 1) * (r_length + 1) + j - 1] +
                              (sequence_s[i - 1] != sequence_r[j - 1]);
  const types::U16 top = dp_table[(i - 1) * (r_length + 1) + j] + 1;

  // Actually applying the relation and storing the result in (i, j)
  dp_table[i * (r_length + 1) + j] = umin(umin(left, top), top_left);
}

/**
 * @brief Goes through every anti-diagonal of the DP table and populates cells
 * based on the previous two anti-diagonals according to the DP algorithm used.
 *
 * @param sequence_r Input DNA sequence, presumed longer than the second.
 * @param sequence_s Input DNA sequence, presumed shorter than the first.
 * @param r_length Length of the first DNA sequence.
 * @param s_length Length of the second DNA sequence.
 * @param dp_table DP table to be populated. Final result will be at position
 * (s_length, r_length).
 */
__global__ void SolveDPP(const types::Base* sequence_r,
                         const types::Base* sequence_s, types::U16 r_length,
                         types::U16 s_length, types::U16* dp_table) {
  const types::U16 global_id = blockIdx.x * blockDim.x + threadIdx.x;
  const types::U16 grid_size = blockDim.x * gridDim.x;

  // Initializing first row and column
  types::U16 grids_per_row = (r_length + grid_size - 1) / grid_size;
  for (types::U16 k = 0; k < grids_per_row; k++) {
    types::U16 curr_id = global_id + k * grid_size;
    if (curr_id <= r_length) dp_table[curr_id] = curr_id;
    if (curr_id <= s_length) dp_table[curr_id * (r_length + 1)] = curr_id;
  }

  // Getting access to the whole grid for synchronization purposes
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();

  // Goes through every anti-diagonal in the DP table
  for (types::U16 d = 2, i, j; d < s_length + r_length + 1; d++) {
    types::U16 grids_per_diag = (s_length + grid_size - 1) / grid_size;

    // Slides the grid along anti-diagonal d
    for (types::U16 k = 0; k < grids_per_diag; k++) {
      types::U16 curr_id = global_id + k * grid_size;

      i = d - curr_id;
      j = curr_id;

      if (0 < i && i <= s_length && 0 < j && j <= r_length)
        CalcCell(sequence_r, sequence_s, i, j, r_length, dp_table);
    }

    grid.sync();
  }
}

/**
 * @brief Allocates device memory, transfers input data into the device,
 * launches the kernel, transfers output data back to host and cleans up.
 *
 * @param h_sequence_r Input DNA sequence, presumed longer than the second.
 * @param h_sequence_s Input DNA sequence, presumed shorter than the first.
 * @param r_length Length of the first DNA sequence.
 * @param s_length Length of the second DNA sequence.
 * @return U16 Shortest genetic distance between the two sequences.
 */
__host__ types::U16 LaunchKernel(const types::Base* h_sequence_r,
                                 const types::Base* h_sequence_s,
                                 types::U16 r_length, types::U16 s_length) {
  const types::U32 size = (s_length + 1) * (r_length + 1);

  // Allocating global device memory for the table and the two sequences
  types::U16* d_dp_table;
  types::Base* d_sequence_r;
  types::Base* d_sequence_s;
  CHECK(cudaMalloc(&d_dp_table, size * sizeof(types::U16)));
  CHECK(cudaMalloc(&d_sequence_r, r_length * sizeof(types::Base)));
  CHECK(cudaMalloc(&d_sequence_s, s_length * sizeof(types::Base)));

  // Transferring input data from host to device
  CHECK(cudaMemcpy(d_sequence_r, h_sequence_r, r_length * sizeof(types::Base),
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_sequence_s, h_sequence_s, s_length * sizeof(types::Base),
                   cudaMemcpyHostToDevice));

  // This seems like a generally good number of threads per block
  constexpr types::U16 block_size = 256;

  // Accessing device properties
  cudaDeviceProp device_prop;
  CHECK(cudaGetDeviceProperties(&device_prop, 0));

  // Getting exact number of blocks that can be in an SM simultaneously
  int blocks_per_sm;
  CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, SolveDPP,
                                                      block_size, 0));

  // Optimal grid size for the job (hopefully?)
  const types::U16 grid_size = device_prop.multiProcessorCount * blocks_per_sm;

  // Preparing arguments for cooperative kernel launch
  void** args = new void*[5];
  args[0] = &d_sequence_r;
  args[1] = &d_sequence_s;
  args[2] = &r_length;
  args[3] = &s_length;
  args[4] = &d_dp_table;

  // Launching cooperative kernel
  CHECK(cudaLaunchCooperativeKernel((const void*)SolveDPP, (dim3)grid_size,
                                    (dim3)block_size, args, (size_t)0,
                                    (cudaStream_t)0));

  // Transferring output data from device to host
  types::U16 h_result;
  CHECK(cudaMemcpy(&h_result, &d_dp_table[s_length * (r_length + 1) + r_length],
                   sizeof(types::U16), cudaMemcpyDeviceToHost));

  // Freeing device global memory
  CHECK(cudaFree(d_dp_table));
  CHECK(cudaFree(d_sequence_r));
  CHECK(cudaFree(d_sequence_s));

  // Resetting the device
  CHECK(cudaDeviceReset());

  return h_result;
}

}  // namespace algorithm
}  // namespace cuda_assignment

/**
 * @brief Reads input from a file, launches the kernel and outputs result and
 * execution time into stdout.
 *
 * @param argc Number of command-line arguments.
 * @param argv Array with command-line arguments.
 * @param envp Unused. Array with environment variables.
 * @return int Program termination status. Zero if successful.
 */
int main(int argc, char** argv, char** envp) {
  namespace ca = cuda_assignment;

  // Checking number of command-line arguments, exiting on failure
  if (argc != 2) exit(fputs("Syntax: ./program <input file path>\n", stderr));

  // Trying to open input file, exiting on failure
  FILE* input_file_handle = fopen(argv[1], "r");
  if (!input_file_handle) exit(fputs("Input file not found\n", stderr));

  // Reading lengths of both DNA sequences
  ca::types::U16 s_length, r_length;
  fscanf(input_file_handle, "%hu %hu", &s_length, &r_length);
  fgetc(input_file_handle);

  // Reading shorter DNA sequence
  ca::types::Base* sequence_s = new ca::types::Base[s_length];
  for (ca::types::U16 i = 0; i < s_length; i++)
    sequence_s[i] = (ca::types::Base)fgetc(input_file_handle);
  fgetc(input_file_handle);

  // Reading longer DNA sequence
  ca::types::Base* sequence_r = new ca::types::Base[r_length];
  for (ca::types::U16 i = 0; i < r_length; i++)
    sequence_r[i] = (ca::types::Base)fgetc(input_file_handle);

  // Cleaning up
  fclose(input_file_handle);

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
