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

#include "rot_arena.h"
#include <stdint.h>

typedef struct rot_tensor *rot_tensor_t;

/**
 * ROT_create_tensor() - Allocates and initializes a tensor with `num_dims`
 * dimensions given by `dims`.
 * @arena: Memory arena to allocate tensor from.
 * @num_dims: Number of dimensions of the tensor to allocate.
 * @dims: Size of each dimension of the allocated tensor.
 *
 * Returns NULL on error.
 */
rot_tensor_t ROT_create_tensor(rot_arena_t arena,
                               uint32_t num_dims,
                               size_t *dims);

/**
 * ROT_matmul()
 *
 * Requirements
 *
 * Inputs:
 * Two tensors of dimension == 2.
 * The inner dimension (number of columns) of the first argument must match the
 * outer dimension (number of rows) of the second argument.
 *
 * Output:
 * For input tensors of dimension mxn and m'xn', the output tensor is a single
 * tensor of mxn'.
 *
 * If any input requirements are not satisfied, a and b are not touched and
 * NULL is returned.
 */
rot_tensor_t ROT_matmul(rot_arena_t arena, rot_tensor_t a, rot_tensor_t b);

/**
 * ROT_tensor_get_data() - Returns a pointer to the float data in tensor `a`.
 * @tensor: A tensor.
 */
float *ROT_tensor_get_data(rot_tensor_t tensor);

#endif /* ROT_MATH_H */
