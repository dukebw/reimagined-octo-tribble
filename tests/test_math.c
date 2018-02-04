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
#include "tests/min_unit.h"
#include "TH/TH.h"
#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"
#include "cblas.h"
/*
 * TODO(brendan): There is a compile error if rocblas.h is included after
 * hip_runtime_api.h...
 */
#include "rocblas.h"
#include "hip/hcc_detail/hip_runtime_api.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>

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
        td->tensor = ROT_create_tensor(arena, 2, dims, ROT_BACKEND_CPU);
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

        gsl_rng *rng = get_gsl_rng();

        size_t a_num_elems = init_data_uniform(state->a.data, rng, mk_dims);
        size_t b_num_elems = init_data_uniform(state->b.data, rng, kn_dims);
        size_t c_num_elems = dims->m*dims->n;

        size_t c_bytes = c_num_elems*sizeof(float);
        float *temp_c_data = (float *)ROT_arena_malloc(state->arena,
                                                       c_bytes,
                                                       ROT_BACKEND_CPU);
        assert(temp_c_data != NULL);

        state->th_a = create_th_tensor(state->a.data, mk_dims, a_num_elems);
        state->th_b = create_th_tensor(state->b.data, kn_dims, b_num_elems);
        state->th_c = create_th_tensor(temp_c_data, mn_dims, c_num_elems);

        gsl_rng_free(rng);
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

/**
 * rand_dim() - Returns a random dimension in [1, `max_dim`).
 */
static size_t
rand_dim(uint32_t max_dim)
{
        return (rand() % max_dim) + 1;
}

/**
 * check_state_matches() - Verifies that the test state contained in `state` is
 * consistent, i.e. that the computed result C is correct.
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
static void
check_state_matches(struct matmul_test_state *state,
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
                const float diff = (state->th_c->storage->data[i] -
                                    state->c.data[i]);
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
        struct matmul_dims dims = {.n = rand_dim(128),
                                   .m = rand_dim(128),
                                   .k = rand_dim(128)};

        struct matmul_test_state state;
        setup_matmul_test_state(&state, memory, sizeof(memory), &dims);

        state.c.tensor = ROT_matmul(state.c.tensor,
                                    state.a.tensor,
                                    state.b.tensor);
        MIN_UNIT_ASSERT(state.c.tensor != NULL,
                        "NULL returned from ROT_matmul, expected "
                        "rot_tensor\n");

        check_state_matches(&state, &dims, FLT_EPSILON);

        THFloatTensor_free(state.th_a);
        THFloatTensor_free(state.th_b);
        THFloatTensor_free(state.th_c);
}

static MIN_UNIT_TEST_FUNC(test_matmul_small_cudnn)
{
}

static hipDeviceptr_t
init_tensor_on_dev(rot_arena_t arena_roc,
                   const struct tensor_data t,
                   hipStream_t stream,
                   size_t limit)
{
        hipDeviceptr_t t_device = ROT_arena_malloc(arena_roc,
                                                   limit,
                                                   ROT_BACKEND_ROC);
        assert(t_device != NULL);

        size_t t_size = ROT_tensor_get_size(t.tensor);

        hipError_t hip_err = hipMemcpyHtoDAsync(t_device,
                                                t.data,
                                                t_size,
                                                stream);
        assert(hip_err == hipSuccess);

        return t_device;
}

static MIN_UNIT_TEST_FUNC(test_matmul_small_miopen)
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

        hipDeviceptr_t a_dev = init_tensor_on_dev(arena_roc,
                                                  state.a,
                                                  stream,
                                                  limit);
        hipDeviceptr_t b_dev = init_tensor_on_dev(arena_roc,
                                                  state.b,
                                                  stream,
                                                  limit);

        hipDeviceptr_t c_dev = ROT_arena_malloc(arena_roc,
                                                limit,
                                                ROT_BACKEND_ROC);
        assert(hip_err == hipSuccess);

        hip_err = hipStreamSynchronize(stream);
        assert(hip_err == hipSuccess);

        rocblas_handle handle;
        rocblas_status rblas_err = rocblas_create_handle(&handle);
        assert(rblas_err == rocblas_status_success);

        const float alpha = 1.0f;
        const float beta = 0.0f;
        rblas_err = rocblas_sgemm(handle,
                                  rocblas_operation_none,
                                  rocblas_operation_none,
                                  dims.n,
                                  dims.m,
                                  dims.k,
                                  &alpha,
                                  (const float *)b_dev,
                                  dims.n,
                                  (const float *)a_dev,
                                  dims.k,
                                  &beta,
                                  (float *)c_dev,
                                  dims.n);
        assert(rblas_err == rocblas_status_success);

        rblas_err = rocblas_destroy_handle(handle);
        assert(rblas_err == rocblas_status_success);

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

/**
 * test_matmul_small_perf() - Test for speed for small matrix multiplication.
 *
 * Pass criteria: the ROT_matmul() implementation should be faster than
 * third-party ML library matrix-multiply implementations for small matrices
 * (with each dimension < 4096) of arbitrary sizes.
 */
static MIN_UNIT_TEST_FUNC(test_matmul_small_perf)
{
        const size_t memory_size = 1024*1024*1024;
        uint8_t *memory = (uint8_t *)malloc(memory_size);
        assert(memory != NULL);
        struct matmul_dims dims = {.n = rand_dim(4096),
                                   .m = rand_dim(4096),
                                   .k = rand_dim(4096)};

        struct timeval start;
        get_and_check_time_of_day(&start);

        struct matmul_test_state state;
        setup_matmul_test_state(&state, memory, memory_size, &dims);

        for (uint32_t i = 0;
             i < 16;
             ++i) {
                THFloatTensor_addmm(state.th_c,
                                    0.0,
                                    state.th_c,
                                    1.0,
                                    state.th_a,
                                    state.th_b);
        }
        printf("Done TH!\n");

        struct timeval end;
        get_and_check_time_of_day(&end);

        double th_elapsed_sec = get_elapsed_sec(start, end);

        get_and_check_time_of_day(&start);

        for (uint32_t i = 0;
             i < 16;
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

        free(memory);
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
