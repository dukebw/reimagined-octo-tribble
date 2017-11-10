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
#include "rot_math.h"
#include <stdlib.h>

/**
 * rot_tensor: Container for tensor data.
 *
 * NOTE(brendan): tensors are represented by contiguous memory. Viewing a
 * tensor's data starting from offset 0 and increasing, dims[0] represents the
 * slowest changing dimension, and dims[num_dims - 1] is the quickest changing
 * dimension.
 *
 * E.g. for a matrix, dims[0] would be the dimension of the
 * rows, and dims[1] would be the dimension for the columns.
 *
 * TODO(brendan): Supported dimensions? Different types?
 */
struct rot_tensor {
        uint32_t num_dims;
        size_t *dims;
        /**
         * NOTE(brendan): Use of the flexible array member here is to allow
         * variable length tensors to be allocated.
         */
        float data[];
};

rot_tensor_t ROT_create_tensor(rot_arena_t arena,
                               uint32_t num_dims,
                               size_t *dims)
{
        if ((dims == NULL) || (arena == NULL))
                return NULL;

        if (num_dims == 0)
                return NULL;

        size_t required_bytes = dims[0];
        for (uint32_t dim = 1;
             dim < num_dims;
             ++dim) {
                required_bytes *= dims[dim];
        }

        required_bytes += sizeof(struct rot_tensor);
        /**
         * NOTE(brendan): Storage for the dimensions' respective sizes must
         * also be allocated. These dimension sizes are placed _after_ all the
         * space for the data.
         *
         * So, the memory layout of a tensor is:
         * | num_dims | *dims | data | dims |
         * where *dims is a pointer to dims.
         */
        size_t dim_sizes_bytes = sizeof(size_t)*num_dims;
        required_bytes += dim_sizes_bytes;

        if (!ROT_arena_can_alloc(arena, required_bytes))
                return NULL;

        struct rot_tensor *result = ROT_arena_malloc(arena, required_bytes);
        result->num_dims = num_dims;
        result->dims = (size_t *)((char *)result +
                                  (required_bytes - dim_sizes_bytes));

        for (uint32_t dim = 0;
             dim < num_dims;
             ++dim) {
                result->dims[dim] = dims[dim];
        }

        return result;
}

rot_tensor_t ROT_matmul(rot_arena_t arena, rot_tensor_t a, rot_tensor_t b)
{
        if ((a == NULL) || (b == NULL))
            return NULL;

        return NULL;
}

float *ROT_tensor_get_data(rot_tensor_t tensor)
{
        return tensor->data;
}
