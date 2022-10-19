/**
 * COMPLETE IMPLEMENTATION (MULTIPLE BLOCKS IN GRID)
 *
 * Please use the makefile, works perfectly both in our machines and on Colab.
 *
 * @file basic_types.cu
 * @authors Diego F. S. Souza (diego.f.s.souza@ufms.br) and Thiago Ferronatto
 * (thiago.ferronatto@ufms.br)
 * @brief Defines some uint aliases and an enum. Super useful stuff.
 * @version 1.0
 * @date 2022-10-18
 *
 * @copyright Copyright (c) 2022
 */

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
