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
#include "platform/cudnn.h"
#include "tests/min_unit.h"

/* TODO(brendan): move the cuda includes to platform. */
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"
#include "TH/TH.h"

#include <assert.h>
#include <float.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

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
 * create_th_tensor() - Allocates a a TH matrix with dimensions given by
 * `dims`, and initializes said matrix with `data`.
 * @data: Data to initialize the TH matrix with.
 * @dims: [rows, columns] of the matrix.
 * @flat_size: Should be rows*columns.
 *
 * Returns a pointer to the allocated TH matrix.
 */
static THFloatTensor *
create_th_tensor(float *data, const size_t *dims, size_t flat_size)
{
        THFloatTensor *th_tensor = THFloatTensor_newWithSize2d(dims[0],
                                                               dims[1]);
        assert(th_tensor != NULL);

        memcpy(th_tensor->storage->data, data, flat_size*sizeof(float));

        return th_tensor;
}

/**
 * get_and_check_time_of_day() - Get time of day and store it in `tv`, checking
 * for errors.
 * @tv: Time of day output.
 */
static void
get_and_check_time_of_day(struct timeval *tv)
{
        int32_t status = gettimeofday(tv, NULL);
        assert(status == 0);
}

/**
 * get_seed_from_time_of_day() - Convenience function to return the 64-bit
 * micro-second part of the time of day.
 */
static uint64_t
get_seed_from_time_of_day(void)
{
        struct timeval seed;
        get_and_check_time_of_day(&seed);

        return seed.tv_usec;
}

/**
 * get_gsl_rng() - Allocates a GSL RNG and seeds it with the time in
 * microseconds.
 *
 * The caller owns the returned RNG.
 */
static gsl_rng *
get_gsl_rng(void)
{
        /**
         * NOTE(brendan): This must be done; passing the `gsl_rng_taus` pointer
         * directly to `gsl_rng_alloc` results in a SIGSEGV.
         */
        const gsl_rng_type *rng_type = gsl_rng_taus;
        gsl_rng *rng = gsl_rng_alloc(rng_type);
        assert(rng != NULL);

        gsl_rng_set(rng, get_seed_from_time_of_day());

        return rng;
}

/**
 * get_tensor_data() - Allocates a matrix from `arena` with size given by
 * `dims`, and fills in the resulting allocated data in `td`.
 * @td: Output tensor data goes here.
 * @arena: Memory arena to allocate the matrix from.
 * @dims: Dimensions of the matrix to allocate.
 */
static void
get_tensor_data(struct tensor_data *td, rot_arena_t arena, const size_t *dims)
{
        td->tensor = ROT_create_tensor(arena, 2, dims);
        assert(td->tensor != NULL);

        td->data = ROT_tensor_get_data(td->tensor);
        assert(td->data != NULL);
}

/**
 * setup() - Does test setup to be done before each ROT_math test.
 */
static void
setup_test(void)
{
        srand(get_seed_from_time_of_day());
}

/**
 * init_data_uniform() - Initializes `data`, assumed to represent a matrix,
 * from a uniform distribution with range [-1, 1].
 * @data: The buffer to be initialized.
 * @rng: GSL RNG state to used to draw the random samples.
 * @dims: The two dimensions of the `data` matrix.
 *
 * Returns the size in bytes of the entire `data` buffer initialized.
 */
static size_t
init_data_uniform(float *data, gsl_rng *rng, const size_t *dims)
{
        size_t num_elems = dims[0]*dims[1];
        for (uint32_t i = 0;
             i < num_elems;
             ++i) {
                data[i] = gsl_ran_flat(rng, -1, 1);
        }

        return num_elems;
}

/**
 * setup_matmul_test_state() - A setup function that runs before all tests of
 * `ROT_matmul`.
 * @state: The test state struct, holding all allocated memory and other state
 * that needs to be setup before each matmul test.
 * @mem: A contiguous buffer `mem_bytes` in size.
 * @mem_bytes: Size in bytes of `mem`.
 * @dims: Dimensions N, M and K of the matrices.
 */
static void
setup_matmul_test_state(struct matmul_test_state *state,
                        uint8_t *mem,
                        size_t mem_bytes,
                        const struct matmul_dims *dims)
{
        state->arena = ROT_arena_new(mem, mem_bytes);
        assert(state->arena != NULL);

        const size_t mk_dims[] = {dims->m, dims->k};
        get_tensor_data(&state->a, state->arena, mk_dims);

        const size_t kn_dims[] = {dims->k, dims->n};
        get_tensor_data(&state->b, state->arena, kn_dims);

        const size_t mn_dims[] = {dims->m, dims->n};
        get_tensor_data(&state->c, state->arena, mn_dims);

        for (uint32_t i = 0;
             i < dims->n*dims->m;
             ++i) {
                state->c.data[i] = 1.0 + i;
        }

        gsl_rng *rng = get_gsl_rng();

        size_t a_num_elems = init_data_uniform(state->a.data, rng, mk_dims);
        size_t b_num_elems = init_data_uniform(state->b.data, rng, kn_dims);
        size_t c_num_elems = dims->n*dims->m;

        size_t c_bytes = c_num_elems*sizeof(float);
        float *temp_c_data = ROT_arena_malloc(state->arena, c_bytes);
        assert(temp_c_data != NULL);

        state->th_a = create_th_tensor(state->a.data, mk_dims, a_num_elems);
        state->th_b = create_th_tensor(state->b.data, kn_dims, b_num_elems);
        state->th_c = create_th_tensor(temp_c_data, mn_dims, c_num_elems);

        gsl_rng_free(rng);
}

/**
 * rand_dim() - Returns a random dimension in [1, `max_dim`).
 */
static size_t
rand_dim(uint32_t max_dim)
{
        return (rand() % max_dim) + 1;
}

static struct matmul_dims
setup_matmul_test_state_small(struct matmul_test_state *state,
                              uint8_t *memory,
                              size_t memory_bytes)
{
        struct matmul_dims dims = {.n = rand_dim(128),
                                   .m = rand_dim(128),
                                   .k = rand_dim(128)};

        setup_matmul_test_state(state, memory, memory_bytes, &dims);

        return dims;
}

/**
 * get_elapsed_sec() - Returns the elapsed time in seconds since `start`, until
 * `end`.
 */
static double
get_elapsed_sec(struct timeval start, struct timeval end)
{
        return ((end.tv_sec - start.tv_sec) +
                (end.tv_usec - start.tv_usec)/1e6);
}

static void
check_result_against_th(struct matmul_test_state *state,
                        const struct matmul_dims *dims,
                        float epsilon)
{
        THFloatTensor_addmm(state->th_c,
                            0.0,
                            state->th_c,
                            1.0,
                            state->th_a,
                            state->th_b);

        for (uint32_t i = 0;
             i < dims->m*dims->n;
             ++i) {
                /* printf("%.5f\n", state->th_c->storage->data[i]); */
                /* printf("%.5f\n", state->c.data[i]); */
                /* printf("\n"); */
                float diff = state->th_c->storage->data[i] - state->c.data[i];
                MIN_UNIT_ASSERT(fabs(diff) < epsilon,
                                "ROT_matmul mismatches TH_addmm at index %d\n",
                                i);
        }
}

/**
 * test_matmul_small() - Simple test for correctness for small matrix
 * multiplication.
 *
 * Pass criteria: the result of multiplying two randomly initialized matrices,
 * of random (small and valid) dimensions, must match a reference
 * implementation to within the floating point precision of the system.
 */
static MIN_UNIT_TEST_FUNC(test_matmul_small)
{
        uint8_t memory[512*1024];
        struct matmul_test_state state;
        struct matmul_dims dims =
                setup_matmul_test_state_small(&state, memory, sizeof(memory));

        state.c.tensor = ROT_matmul(state.c.tensor,
                                    state.a.tensor,
                                    state.b.tensor);
        MIN_UNIT_ASSERT(state.c.tensor != NULL,
                        "NULL returned from ROT_matmul, expected "
                        "rot_tensor\n");

        check_result_against_th(&state, &dims, FLT_EPSILON);

        THFloatTensor_free(state.th_a);
        THFloatTensor_free(state.th_b);
        THFloatTensor_free(state.th_c);
}

static MIN_UNIT_TEST_FUNC(test_matmul_small_cudnn)
{
        const size_t memory_size = 1024*1024*1024;
        uint8_t *memory = malloc(memory_size);
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

        check_result_against_th(&state, &dims, 1024*FLT_EPSILON);

        cuda_err = cudaFree(cuda_memory);
        assert(cuda_err == cudaSuccess);

        free(memory);
}

static MIN_UNIT_TEST_FUNC(test_matmul_small_miopen)
{
}

/**
 * test_matmul_small_perf() - Test for speed for small matrix multiplication.
 *
 * Pass criteria: the ROT_matmul() implementation should be faster than
 * third-party ML library matrix-multiply implementations for small matrices
 * (with each dimension < 128) of arbitrary sizes.
 */
static MIN_UNIT_TEST_FUNC(test_matmul_small_perf)
{
        uint8_t memory[512*1024];
        struct matmul_test_state state;
        setup_matmul_test_state_small(&state, memory, sizeof(memory));

        struct timeval start;
        get_and_check_time_of_day(&start);

        for (uint32_t i = 0;
             i < 256;
             ++i) {
                THFloatTensor_addmm(state.th_c,
                                    0.0,
                                    state.th_c,
                                    1.0,
                                    state.th_a,
                                    state.th_b);
        }

        struct timeval end;
        get_and_check_time_of_day(&end);

        double th_elapsed_sec = get_elapsed_sec(start, end);

        get_and_check_time_of_day(&start);

        for (uint32_t i = 0;
             i < 256;
             ++i) {
                state.c.tensor = ROT_matmul(state.c.tensor,
                                            state.a.tensor,
                                            state.b.tensor);
                MIN_UNIT_ASSERT(state.c.tensor != NULL,
                                "NULL returned from ROT_matmul, expected "
                                "rot_tensor\n");
        }

        get_and_check_time_of_day(&end);

        double rot_elapsed_sec = get_elapsed_sec(start, end);

        MIN_UNIT_ASSERT(rot_elapsed_sec < th_elapsed_sec,
                        "ROT performance (%.5f) below that of TH (%.5f)\n",
                        rot_elapsed_sec,
                        th_elapsed_sec);
}

/**
 * run_test() - Sets up, runs and tears down the unit test function `test`.
 */
static void
run_test(min_unit_test_func test)
{
        setup_test();
        min_unit_run_test(test);
}

int main(void)
{
        run_test(test_matmul_small);
        run_test(test_matmul_small_cudnn);
        run_test(test_matmul_small_miopen);
        run_test(test_matmul_small_perf);

        printf("All tests passed!\n");

        return EXIT_SUCCESS;
}
