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
#include "error/log_error.h"
#include "platform/math.h"

#include "cblas.h"
#include <stdlib.h>

struct rot_cpu_tensor {
        /**
         * NOTE(brendan): Use of the flexible array member here is to allow
         * variable length tensors to be allocated.
         */
        float data[];
};

struct rot_roc_tensor {
        void *data;
};

/**
 * rot_tensor: Container for tensor data.
 *
 * NOTE(brendan): tensors are represented by contiguous memory. Viewing a
 * tensor's data starting from offset 0 and increasing, dims[0] represents the
 * slowest changing dimension, and dims[num_dims - 1] is the quickest changing
 * dimension.
 *
 * E.g. for a matrix, dims[0] would be the dimension of the rows, and dims[1]
 * would be the dimension for the columns.
 *
 * TODO(brendan): Supported dimensions? Different types?
 */
struct rot_tensor {
        enum rot_backend backend;
        size_t *dims;
        uint32_t num_dims;
        union {
                struct rot_cpu_tensor cpu;
                struct rot_roc_tensor roc;
        };
};

rot_tensor_t ROT_create_tensor(rot_arena_t arena,
                               uint32_t num_dims,
                               const size_t *dims,
                               enum rot_backend backend)
{
        if ((dims == NULL) || (arena == NULL)) {
                LOG_NULL();
                return NULL;
        }

        if (num_dims == 0) {
                LOG_ERROR("Tensors must have a non-zero number of "
                          "dimensions.");
                return NULL;
        }

        if ((backend != ROT_BACKEND_CPU) && (backend != ROT_BACKEND_ROC)) {
                LOG_UNSUPPORTED();
                return NULL;
        }

        size_t required_bytes = sizeof(struct rot_tensor);
        /**
         * NOTE(brendan): Storage for the dimensions' respective sizes must
         * also be allocated. These dimension sizes are placed _after_ all the
         * space for the data.
         *
         * So, the memory layout of a tensor is:
         * | backend | *dims | num_dims | data | dims |
         * where *dims is a pointer to dims.
         */
        size_t dim_sizes_bytes = sizeof(size_t)*num_dims;
        required_bytes += dim_sizes_bytes;

        size_t data_bytes = dims[0]*sizeof(float);
        for (uint32_t dim = 1;
             dim < num_dims;
             ++dim) {
                data_bytes *= dims[dim];
        }

        /**
         * NOTE(brendan): In the case of CPU tensors, data is stored in memory
         * directly contiguous with the tensor struct's metadata, and it can be
         * checked here that there is enough memory to allocate struct + data.
         */
        if (backend == ROT_BACKEND_CPU)
                required_bytes += data_bytes;

        struct rot_tensor *result =
                (struct rot_tensor *)ROT_arena_malloc(arena,
                                                      required_bytes,
                                                      ROT_BACKEND_CPU);
        if (result == NULL)
                return NULL;

        result->backend = backend;
        result->dims = (size_t *)((char *)result +
                                  (required_bytes - dim_sizes_bytes));

        for (uint32_t dim = 0;
             dim < num_dims;
             ++dim) {
                result->dims[dim] = dims[dim];
        }

        result->num_dims = num_dims;

        if (backend == ROT_BACKEND_ROC) {
                result->roc.data = ROT_arena_malloc(arena,
                                                    data_bytes,
                                                    ROT_BACKEND_ROC);
                if (result->roc.data == NULL)
                        return NULL;
        }

        return result;
}

rot_tensor_t ROT_matmul(rot_tensor_t result,
                        const rot_tensor_t a,
                        const rot_tensor_t b)
{
        if ((result == NULL) || (a == NULL) || (b == NULL)) {
                LOG_ERROR("Null input.");
                return NULL;
        }

        if ((a->num_dims != 2) || (b->num_dims != 2)) {
                LOG_ERROR("Matrix dimensions must be 2.");
                return NULL;
        }

        if (a->dims[1] != b->dims[0]) {
                LOG_ERROR("Matrix dimensions incompatible for "
                          "multiplication.");
                return NULL;
        }

        if ((result == a) || (result == b)) {
                LOG_ERROR("Result tensor of matmul must be different from "
                          "either operand tensor.");
                return NULL;
        }

        cblas_sgemm(CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    a->dims[0],
                    b->dims[1],
                    a->dims[1],
                    1.0f,
                    a->cpu.data,
                    a->dims[1],
                    b->cpu.data,
                    b->dims[1],
                    0.0f,
                    result->cpu.data,
                    b->dims[1]);

        return result;
}

float *ROT_tensor_get_data(rot_tensor_t tensor)
{
        return tensor->cpu.data;
}

size_t ROT_tensor_get_size(rot_tensor_t tensor)
{
        if (tensor->num_dims == 0)
                return 0;

        size_t size = sizeof(float);
        for (uint32_t i = 0;
             i < tensor->num_dims;
             ++i) {
                size *= tensor->dims[i];
        }

        return size;
}
