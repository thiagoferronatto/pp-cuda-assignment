#ifndef CUDA_ASSIGNMENT_GENETIC_DISTANCE_CUH_
#define CUDA_ASSIGNMENT_GENETIC_DISTANCE_CUH_

#include <cooperative_groups.h>

#include "basic_types.cuh"

namespace cuda_assignment {
namespace algorithm {

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
__device__ inline void CalcCell(const types::Base*, const types::Base*,
                                types::U16, types::U16, types::U16,
                                types::U16*);

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
__global__ void SolveDPP(const types::Base*, const types::Base*, types::U16,
                         types::U16, types::U16*);

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
__host__ types::U16* LaunchKernel(const types::Base*, const types::Base*,
                                  types::U16, types::U16);

}  // namespace algorithm
}  // namespace cuda_assignment

#endif  // CUDA_ASSIGNMENT_GENETIC_DISTANCE_CUH_
