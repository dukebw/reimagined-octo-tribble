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
#include "platform/cudnn.h"
#include "tests/test_math.h"

#include "cuda_runtime.h"
#include "cublas_v2.h"
#include <assert.h>
#include <stdint.h>

MIN_UNIT_TEST_FUNC(test_matmul_small_cudnn)
{
        const size_t memory_size = 1024*1024*1024;
        uint8_t *memory = (uint8_t *)malloc(memory_size);
        struct matmul_test_state state;

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

        setup_matmul_test_state(&state, memory, memory_size, &dims);

        printf("CUDA GPU alloc limit: %lu\n", limit);

        /* NOTE(brendan): major version 2 corresponds to Fermi. */
        assert(device_prop.major >= 2);

        const size_t CUDA_MEM_BYTES = 1024*1024*1024;
        size_t a_bytes = sizeof(float)*dims.m*dims.k;
        size_t b_bytes = sizeof(float)*dims.k*dims.n;
        size_t c_bytes = sizeof(float)*dims.m*dims.n;
        size_t required_bytes = a_bytes + b_bytes + c_bytes;
        assert(CUDA_MEM_BYTES >= required_bytes);

        void *cuda_memory;
        cuda_err = cudaMalloc(&cuda_memory, CUDA_MEM_BYTES);
        assert(cuda_err == cudaSuccess);

        float *d_A = (float *)cuda_memory;
        float *d_B = (float *)((uint8_t *)cuda_memory + a_bytes);
        float *d_C = (float *)((uint8_t *)d_B + b_bytes);

        cuda_err = cudaMemcpy(d_A,
                              state.a.data,
                              a_bytes,
                              cudaMemcpyHostToDevice);
        assert(cuda_err == cudaSuccess);

        cuda_err = cudaMemcpy(d_B,
                              state.b.data,
                              b_bytes,
                              cudaMemcpyHostToDevice);
        assert(cuda_err == cudaSuccess);

        cublasHandle_t handle;
        cublasStatus_t cublas_status = cublasCreate(&handle);
        assert(cublas_status == CUBLAS_STATUS_SUCCESS);

        /* TODO(brendan): Turn this into a ROT_matmul call. */
        const float alpha = 1.0;
        const float beta = 0.0;
        cublas_status = cublasSgemm(handle,
                                    CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    dims.n,
                                    dims.m,
                                    dims.k,
                                    &alpha,
                                    d_B,
                                    dims.n,
                                    d_A,
                                    dims.k,
                                    &beta,
                                    d_C,
                                    dims.n);
        assert(cublas_status == CUBLAS_STATUS_SUCCESS);

        cuda_err = cudaMemcpy(state.c.data,
                              d_C,
                              c_bytes,
                              cudaMemcpyDeviceToHost);
        assert(cuda_err == cudaSuccess);

        check_state_matches(&state, &dims, 1024*FLT_EPSILON);

        cuda_err = cudaFree(cuda_memory);
        assert(cuda_err == cudaSuccess);

        free(memory);
}
