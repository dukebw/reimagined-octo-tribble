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
#include "tests/test_miopen.h"

#include "hip/hip_runtime_api.h"

static rot_tensor_t
init_roc_tensor(rot_arena_t arena_roc,
                const struct tensor_data t,
                hipStream_t stream,
                size_t limit)
{
        const size_t *dims = ROT_tensor_get_dims(t.tensor);
        assert(dims != NULL);

        rot_tensor_t a_tens = ROT_create_tensor(arena_roc,
                                                2,
                                                dims,
                                                ROT_BACKEND_ROC);
        assert(a_tens != NULL);

        float *a_data = ROT_tensor_get_data(a_tens);
        assert(a_data != NULL);

        size_t t_size = ROT_tensor_get_size(t.tensor);
        hipError_t hip_err = hipMemcpyHtoDAsync(a_data,
                                                t.data,
                                                t_size,
                                                stream);
        assert(hip_err == hipSuccess);

        return a_tens;
}

MIN_UNIT_TEST_FUNC(test_matmul_small_miopen)
{
        const size_t memory_size = 1024*1024*1024;
        uint8_t *memory = (uint8_t *)malloc(memory_size);
        assert(memory != NULL);

        size_t limit;
        hipError_t hip_err = hipDeviceGetLimit(&limit, hipLimitMallocHeapSize);
        assert(hip_err == hipSuccess);

        const uint32_t max_dim = sqrt(limit/sizeof(float));
        struct matmul_dims dims = {.n = rand_dim(max_dim),
                                   .m = rand_dim(max_dim),
                                   .k = rand_dim(max_dim)};

        struct matmul_test_state state;
        setup_matmul_test_state(&state, memory, memory_size, &dims);

        hipStream_t stream;
        hip_err = hipStreamCreate(&stream);
        assert(hip_err == hipSuccess);

        constexpr uint32_t num_blocks = 3;
        void *roc_mem_blocks[num_blocks];
        for (uint32_t block_i = 0;
             block_i < num_blocks;
             ++block_i) {
                hipError_t hip_err = hipMalloc(roc_mem_blocks + block_i,
                                               limit);
                assert(hip_err == hipSuccess);
        }

        rot_arena_t arena_roc = ROT_arena_roc_new(state.arena,
                                                  roc_mem_blocks,
                                                  limit,
                                                  num_blocks);
        assert(arena_roc != NULL);

        rot_tensor_t a_tens = init_roc_tensor(arena_roc,
                                              state.a,
                                              stream,
                                              limit);

        rot_tensor_t b_tens = init_roc_tensor(arena_roc,
                                              state.b,
                                              stream,
                                              limit);

        const size_t mn_dims[] = {dims.m, dims.n};
        rot_tensor_t c_tens = ROT_create_tensor(arena_roc,
                                                2,
                                                mn_dims,
                                                ROT_BACKEND_ROC);
        assert(c_tens != NULL);

        hip_err = hipStreamSynchronize(stream);
        assert(hip_err == hipSuccess);

        c_tens = ROT_matmul(c_tens, a_tens, b_tens);
        assert(c_tens != NULL);

        float *c_dev = (float *)ROT_tensor_get_data(c_tens);
        assert(c_dev != NULL);

        hip_err = hipMemcpyDtoH(state.c.data,
                                c_dev,
                                ROT_tensor_get_size(state.c.tensor));
        assert(hip_err == hipSuccess);

        check_state_matches(&state, &dims, 1024*FLT_EPSILON);

        for (uint32_t block_i = 0;
             block_i < num_blocks;
             ++block_i) {
                hip_err = hipFree(roc_mem_blocks[block_i]);
                assert(hip_err == hipSuccess);
        }

        hip_err = hipStreamDestroy(stream);
        assert(hip_err == hipSuccess);

        free(memory);
}

