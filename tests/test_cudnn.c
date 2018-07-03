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
#include "tests/test_cudnn.h"
#include "rot_arena.h"         /* for rot_arena_t */
#include "rot_math.h"          /* for ROT_create_tensor, ROT_tensor_get_data */
#include "rot_platform.h"      /* for ROT_BACKEND_CUDA */
#include "tests/test_math.h"   /* for rand_dim, matmul_dims, tensor_data */

#include "cuda_runtime_api.h"  /* for cudaMemcpy, cudaDeviceGetLimit, ... */
#include "driver_types.h"      /* for cudaSuccess, cudaDeviceProp, ... */

#include <assert.h>            /* for assert */
#include <float.h>             /* for FLT_EPSILON */
#include <math.h>              /* for sqrt */
#include <stdint.h>
#include <stdio.h>             /* for size_t, print, NULL */
#include <stdlib.h>            /* for free, malloc */

static rot_tensor_t
init_cuda_tensor(rot_arena_t arena_gpu,
                 const struct tensor_data t,
                 cudaStream_t stream)
{
        const size_t *dims = ROT_tensor_get_dims(t.tensor);
        assert(dims != NULL);

        rot_tensor_t a_tens = ROT_create_tensor(arena_gpu,
                                                2,
                                                dims,
                                                ROT_BACKEND_CUDA);
        assert(a_tens != NULL);

        float *a_data = ROT_tensor_get_data(a_tens);
        assert(a_data != NULL);

        size_t t_size = ROT_tensor_get_size(t.tensor);
        cudaError_t cuda_err = cudaMemcpyAsync(a_data,
                                               t.data,
                                               t_size,
                                               cudaMemcpyHostToDevice,
                                               stream);
        assert(cuda_err == cudaSuccess);

        return a_tens;
}

MIN_UNIT_TEST_FUNC(test_matmul_small_cudnn)
{
        const size_t memory_size = 1024*1024*1024;
        uint8_t *memory = (uint8_t *)malloc(memory_size);
        assert(memory != NULL);

        int32_t device_id;
        cudaError_t cuda_err = cudaGetDevice(&device_id);
        assert(cuda_err == cudaSuccess);

        struct cudaDeviceProp device_prop;
        cuda_err = cudaGetDeviceProperties(&device_prop, 0);
        assert(cuda_err == cudaSuccess);

        printf("Device id in use: %d\n%s: Compute capability: %d.%d\n\n",
               device_id,
               device_prop.name,
               device_prop.major,
               device_prop.minor);

        size_t limit;
        cuda_err = cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize);
        assert(cuda_err == cudaSuccess);

        const uint32_t max_dim = sqrt(limit/sizeof(float));
        struct matmul_dims dims = {.n = rand_dim(max_dim),
                                   .m = rand_dim(max_dim),
                                   .k = rand_dim(max_dim)};

        struct matmul_test_state state;
        setup_matmul_test_state(&state, memory, memory_size, &dims);

        cudaStream_t stream;
        cuda_err = cudaStreamCreate(&stream);
        assert(cuda_err == cudaSuccess);

        printf("CUDA GPU alloc limit: %lu\n", limit);

        /* NOTE(brendan): major version 2 corresponds to Fermi. */
        assert(device_prop.major >= 2);

        constexpr uint32_t num_blocks = 3;
        void *gpu_mem_blocks[num_blocks];
        for (uint32_t block_i = 0;
             block_i < num_blocks;
             ++block_i) {
                cuda_err = cudaMalloc(gpu_mem_blocks + block_i, limit);
                assert(cuda_err == cudaSuccess);
        }

        rot_arena_t arena_gpu = ROT_arena_gpu_new(state.arena,
                                                  gpu_mem_blocks,
                                                  limit,
                                                  num_blocks);
        assert(arena_gpu != NULL);

        rot_tensor_t a_tens = init_cuda_tensor(arena_gpu, state.a, stream);
        rot_tensor_t b_tens = init_cuda_tensor(arena_gpu, state.b, stream);

        const size_t mn_dims[] = {dims.m, dims.n};
        rot_tensor_t c_tens = ROT_create_tensor(arena_gpu,
                                                2,
                                                mn_dims,
                                                ROT_BACKEND_CUDA);
        assert(c_tens != NULL);

        c_tens = ROT_matmul(c_tens, a_tens, b_tens);
        assert(c_tens != NULL);

        float *c_dev = (float *)ROT_tensor_get_data(c_tens);
        assert(c_dev != NULL);

        cuda_err = cudaMemcpy(state.c.data,
                              c_dev,
                              ROT_tensor_get_size(state.c.tensor),
                              cudaMemcpyDeviceToHost);
        assert(cuda_err == cudaSuccess);

        check_state_matches(&state, &dims, 1024*FLT_EPSILON);

        for (uint32_t block_i = 0;
             block_i < num_blocks;
             ++block_i) {
                cuda_err = cudaFree(gpu_mem_blocks[block_i]);
                assert(cuda_err == cudaSuccess);
        }

        cuda_err = cudaStreamDestroy(stream);
        assert(cuda_err == cudaSuccess);

        free(memory);
}
