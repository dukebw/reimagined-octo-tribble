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
#ifndef ROT_MATH_H
#define ROT_MATH_H

#include "rot_arena.h"     /* for rot_arena_t */
#include "rot_platform.h"  /* for rot_backend */
#include <stddef.h>        /* for size_t */
#include <stdint.h>        /* for uint32_t */

/**
 * TODO(brendan): Only ROT_matmul belongs in rot_math; put the rest in
 * rot_types.
 */
typedef struct rot_tensor *rot_tensor_t;

/**
 * ROT_create_tensor() - Allocates and initializes a tensor with `num_dims`
 * dimensions given by `dims`.
 * @arena: Memory arena to allocate tensor from.
 * @num_dims: Number of dimensions of the tensor to allocate.
 * @dims: Size of each dimension of the allocated tensor.
 * @backend: The backend used for the tensor.
 *
 * Returns NULL on error.
 */
rot_tensor_t ROT_create_tensor(rot_arena_t arena,
                               uint32_t num_dims,
                               const size_t *dims,
                               enum rot_backend backend);

/**
 * ROT_matmul()
 *
 * Requirements
 *
 *
 * Inputs:
 *
 * Two tensors of dimension == 2.
 *
 * The inner dimension (number of columns) of the first argument must match the
 * outer dimension (number of rows) of the second argument.
 *
 * The memory of the result should not overlap with the memory of either of the
 * inputs.
 *
 *
 * Output:
 *
 * For input tensors of dimension mxn and m'xn', the output tensor is a single
 * tensor of mxn'.
 *
 * If any input requirements are not satisfied, a and b are not touched and
 * NULL is returned.
 */
rot_tensor_t ROT_matmul(rot_tensor_t result,
                        const rot_tensor_t a,
                        const rot_tensor_t b);

/**
 * ROT_tensor_get_data() - Returns a pointer to the float data in `tensor`.
 * @tensor: A tensor.
 */
float *ROT_tensor_get_data(rot_tensor_t tensor);

/**
 * ROT_tensor_get_dims() - Returns a pointer to the dimensions in `tensor`.
 * @tensor: A tensor.
 */
const size_t *ROT_tensor_get_dims(rot_tensor_t tensor);

/**
 * ROT_tensor_get_size() - Returns the size in bytes of the data pointed to by
 * `tensor.data`.
 * @tensor: A tensor.
 */
size_t ROT_tensor_get_size(rot_tensor_t tensor);

#endif /* ROT_MATH_H */
