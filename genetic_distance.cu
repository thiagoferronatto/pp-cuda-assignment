#include <cooperative_groups.h>

#include <cstdio>

#include "genetic_distance.cuh"

#define CHECK(ans) \
  cuda_assignment::algorithm::CheckError(ans, __FILE__, __LINE__)

namespace cuda_assignment {
namespace algorithm {

static inline void CheckError(cudaError_t error, const char* file, int line) {
  const auto& GetErr = cudaGetErrorString;
  if (error != cudaSuccess) {
    fprintf(stderr, "CUDA Exception: %s @ %s:%d.\n", GetErr(error), file, line);
    exit(-1);
  }
}

__device__ inline void CalcCell(const types::Base* sequence_r,
                                const types::Base* sequence_s, types::U16 i,
                                types::U16 j, types::U16 r_length,
                                types::U16* dp_table) {
  // Recurrence relation for the DP algorithm is
  // d(i, j) = min{
  //   d(i, j - 1) + 1,
  //   d(i - 1, j - 1) + {0 if s[j] = r[i], 1 otherwise},
  //   d(i - 1, j) + 1
  // }
  const types::U16 left = dp_table[i * (r_length + 1) + j - 1] + 1;
  const types::U16 top_left = dp_table[(i - 1) * (r_length + 1) + j - 1] +
                              (sequence_s[i - 1] != sequence_r[j - 1]);
  const types::U16 top = dp_table[(i - 1) * (r_length + 1) + j] + 1;

  // Actually applying the relation and storing the result in (i, j)
  dp_table[i * (r_length + 1) + j] = umin(umin(left, top), top_left);
}

__global__ void SolveDPP(const types::Base* sequence_r,
                         const types::Base* sequence_s, types::U16 r_length,
                         types::U16 s_length, types::U16* dp_table) {
  const types::U16 global_id = blockIdx.x * blockDim.x + threadIdx.x;

  // Initializing first row and column
  if (global_id <= s_length) dp_table[global_id * (r_length + 1)] = global_id;
  if (global_id <= r_length) dp_table[global_id] = global_id;

  // Getting access to the whole grid for synchronization purposes
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();

  // Goes through every anti-diagonal in the DP table
  for (types::U16 d = 2, i, j; d < s_length + r_length + 1; d++) {
    // Position of the current thread in anti-diagonal d
    i = d - global_id;
    j = global_id;

    // If thread is inside the DP table
    if (0 < i && i <= s_length && 0 < j && j <= r_length)
      CalcCell(sequence_r, sequence_s, i, j, r_length, dp_table);

    // This if statement doesn't really need to be here, since if the grid is
    // not valid, then there is no sync and therefore the program just won't
    // work. Still, we left it for completeness' sake.
    if (grid.is_valid()) {
      grid.sync();
    } else {
      // There really isn't much we can do if somehow this point is reached.
      printf("Something went terribly wrong\n");
    }
  }
}

__host__ types::U16* LaunchKernel(const types::Base* h_sequence_r,
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

  // TODO: try different values and test for performance
  constexpr types::U16 kBlockSize = 1024;
  const types::U16 grid_size = (r_length + kBlockSize) / kBlockSize;

  printf("block size = %u; grid size = %u\n", kBlockSize, grid_size);

  // Preparing arguments for cooperative kernel launch
  void** args = new void*[5];
  args[0] = &d_sequence_r;
  args[1] = &d_sequence_s;
  args[2] = &r_length;
  args[3] = &s_length;
  args[4] = &d_dp_table;

  // Launching cooperative kernel
  // FIXME: CUDA Error: too many blocks in cooperative launch
  CHECK(cudaLaunchCooperativeKernel((const void*)SolveDPP, (dim3)grid_size,
                                    (dim3)kBlockSize, args, (size_t)0,
                                    (cudaStream_t)0));

  // Transferring output data from device to host
  // TODO: currently copying the whole matrix; should transfer only the last
  // element, fix before submitting
  types::U16* h_result = new types::U16[size];
  CHECK(cudaMemcpy(h_result, d_dp_table, size * sizeof(types::U16),
                   cudaMemcpyDeviceToHost));

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
