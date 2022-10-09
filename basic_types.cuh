#ifndef CUDA_ASSIGNMENT_BASIC_TYPES_CUH_
#define CUDA_ASSIGNMENT_BASIC_TYPES_CUH_

#include <cstdint>

namespace cuda_assignment {
namespace types {

using U16 = uint16_t;
using U32 = uint32_t;

/// @brief These should make it easier to read the sequences from a file.
enum Base { kA = 'A', kC = 'C', kG = 'G', kT = 'T' };

}  // namespace types
}  // namespace cuda_assignment

#endif  // CUDA_ASSIGNMENT_BASIC_TYPES_CUH_
