/**
 * Copyright 2017 Brendan Duke.
 *
 * This file is part of ROT ML Library.
 *
 * ROT ML Library is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * ROT ML Library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * ROT ML Library. If not, see <http://www.gnu.org/licenses/>.
 */
#include "rot_math.h"     /* for rot_tensor_t */
#include "rot_arena.h"    /* for rot_arena_t */

#include "TH/THTensor.h"  /* for THFloatTensor */
#include <stddef.h>       /* for size_t */
#include <stdint.h>       /* for uint32_t, uint8_t */

/**
 * struct matmul_dims - Dimensions describing a matrix multiply of an NxM
 * matrix by an MxK matrix.
 */
struct matmul_dims {
        size_t n;
        size_t m;
        size_t k;
};

/**
 * struct tensor_data - Convenience wrapper to store a tensor and a raw pointer
 * to its data.
 * @tensor: A ROT tensor.
 * @data: A pointer to the data in `tensor`.
 */
struct tensor_data {
        rot_tensor_t tensor;
        float *data;
};

/**
 * struct matmul_test_state - All of the test state needed for ROT_matmul
 * tests.
 * @a, @b, @c: ROT matrices.
 * @th_a, @th_b, @th_c: TH matrices.
 * @arena: Memory arena used to allocate the ROT matrices.
 */
struct matmul_test_state {
        struct tensor_data a;
        struct tensor_data b;
        struct tensor_data c;
        THFloatTensor *th_a;
        THFloatTensor *th_b;
        THFloatTensor *th_c;
        rot_arena_t arena;
};

/**
 * check_state_matches() - Verifies that the test state contained in `state` is
 * consistent, i.e., that the computed result C is correct.
 *
 * @state: The test state struct, which already holds a result in `state->c`.
 * `state->th_c` is updated by this function.
 * @dims: Holds the dimensions m, n, k of A, B and C.
 * @epsilon: The maximum value by which the computed result C is allowed to
 * differ from the ground truth and still be considered a PASS.
 *
 * TODO(brendan): Should epsilon be a fraction of the magnitude of the ground
 * truth, rather than an absolute value?
 *
 * The result C = A*B computed by ROT, is checked against the ground truth
 * result computed by TH.
 */
void check_state_matches(struct matmul_test_state *state,
                         const struct matmul_dims *dims,
                         float epsilon);

/**
 * rand_dim() - Returns a random dimension in [1, `max_dim`).
 */
size_t rand_dim(uint32_t max_dim);

/**
 * setup_matmul_test_state() - Setup before all tests of `ROT_matmul`.
 *
 * @state: The test state struct, holding all allocated memory and other state
 * that needs to be setup before each matmul test.
 * @mem: A contiguous buffer `mem_bytes` in size.
 * @mem_bytes: Size in bytes of `mem`.
 * @dims: Dimensions N, M and K of the matrices.
 */
void setup_matmul_test_state(struct matmul_test_state *state,
                             uint8_t *mem,
                             size_t mem_bytes,
                             const struct matmul_dims *dims);
