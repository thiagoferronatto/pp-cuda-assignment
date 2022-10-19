#include <cooperative_groups.h>

#include <cstdio>

#include "../include/genetic_distance.cuh"

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
