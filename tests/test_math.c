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
#include "tests/test_math.h"
#include "tests/test_cudnn.h" /* for test_matmul_small_cudnn */
#include "tests/min_unit.h"   /* for MIN_UNIT_ASSERT, min_unit_run_test */
#include "rot_math.h"         /* for ROT_matmul, ROT_create_tensor, ... */
#include "rot_nn.h"           /* for ROT_relu */
#include "rot_platform.h"     /* for ROT_BACKEND_CPU */

#include "gsl/gsl_rng.h"      /* for gsl_rng, gsl_rng_alloc, gsl_rng_free */
#include "gsl/gsl_randist.h"  /* for gsl_ran_flat */
#include "TH/THStorage.h"     /* for THFloatStorage_data, THFloatStorage */

#include <assert.h>           /* for assert */
#include <float.h>            /* for FLT_EPSILON */
#include <math.h>             /* for fabs */
#include <stdio.h>            /* for printf */
#include <stdlib.h>           /* for size_t, NULL, free, malloc, rand, srand */
#include <string.h>           /* for memcpy */
#include <sys/time.h>         /* for timeval, gettimeofday */

/**
 * array_size() - get the number of elements in array @arr.
 * @arr: array to be sized
 */
template<typename T, size_t N>
constexpr size_t
array_size(T (&)[N])
{
        return N;
}

struct beta_datum {
        double x;
        double alpha;
        double beta;
        double y;
};
template<size_t N>
struct sim_beta {
        struct beta_datum data[N];
};

/**
 * struct linear_layer - Contains weights and activations for a linear layer.
 */
struct linear_layer {
        rot_tensor_t w;
        rot_tensor_t a;
};

/**
 * get_th_tensor_data() - Get float pointer to data in th_tensor.
 *
 * @th_tensor: TH tensor to get data for.
 */
static float *
get_th_tensor_data(const THFloatTensor *th_tensor)
{
        THFloatStorage* storage = THFloatTensor_storage(th_tensor);

        return THFloatStorage_data(storage);
}

/**
 * create_th_tensor() - Allocate a TH matrix with dimensions given by `dims`,
 * and initialize said matrix with `data`.
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

        float *tensor_data = get_th_tensor_data(th_tensor);
        memcpy(tensor_data, data, flat_size*sizeof(float));

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
 * from a uniform distribution with range [-a, a].
 * @data: The buffer to be initialized.
 * @rng: GSL RNG state to used to draw the random samples.
 * @dims: The two dimensions of the `data` matrix.
 * @a: Half-width of the uniform distribution.
 *
 * Returns the size in bytes of the entire `data` buffer initialized.
 */
static size_t
init_data_uniform(float *data, gsl_rng *rng, const size_t *dims, const float a)
{
        size_t num_elems = dims[0]*dims[1];
        for (uint32_t i = 0;
             i < num_elems;
             ++i) {
                data[i] = gsl_ran_flat(rng, -a, a);
        }

        return num_elems;
}

void setup_matmul_test_state(struct matmul_test_state *state,
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

        size_t a_num_elems = init_data_uniform(state->a.data, rng, mk_dims, 1);
        size_t b_num_elems = init_data_uniform(state->b.data, rng, kn_dims, 1);
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

size_t rand_dim(uint32_t max_dim)
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

void
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
                float *th_c_data = get_th_tensor_data(state->th_c);
                const float diff = (th_c_data[i] - state->c.data[i]);
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
        struct matmul_dims dims;
                setup_matmul_test_state_small(&state, memory, sizeof(memory));

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

template<size_t N>
static void
init_layer(struct linear_layer *layer,
           rot_arena_t arena,
           gsl_rng *rng,
           const size_t (&dims)[N])
{
        layer->w = ROT_create_tensor(arena, N, dims, ROT_BACKEND_CPU);
        assert(layer->w != NULL);

        float *data = ROT_tensor_get_data(layer->w);
        float stddev = 1.0f/sqrt(dims[1]);
        init_data_uniform(data, rng, dims, stddev);

        const size_t a_dims[] = {dims[0], 1};
        layer->a = ROT_create_tensor(arena,
                                     array_size(a_dims),
                                     a_dims,
                                     ROT_BACKEND_CPU);
        assert(layer->a != NULL);
}

/**
 * test_feedforward_backward() - A test to run training for simulated data,
 * using SGD + momentum and a feedforward neural network.
 *
 * This is meant to be a toy example, to ensure that backprop is working for a
 * simple feedforward neural network.
 *
 * Inputs x, alpha and beta are sampled from a uniform distribution, and the
 * neural network must output the pdf f(x; alpha, beta) of a Beta distribution
 * at x, parametrized by alpha and beta.
 */
static MIN_UNIT_TEST_FUNC(test_feedforward_backward)
{
        constexpr size_t memory_size = 1024;
        uint8_t memory[memory_size];
        rot_arena_t arena = ROT_arena_new(memory, memory_size);

        constexpr uint32_t num_train = 1024;
        struct sim_beta<num_train> train_dataset;

        gsl_rng *rng = get_gsl_rng();

        for (uint32_t i = 0;
             i < num_train;
             ++i) {
                struct beta_datum *d = train_dataset.data + i;
                d->x = gsl_ran_flat(rng, -1, 1);
                d->alpha = gsl_ran_flat(rng, 0, 5);
                d->beta = gsl_ran_flat(rng, 0, 5);
                d->y = gsl_ran_beta_pdf(d->x, d->alpha, d->beta);
        }

        /* NOTE(brendan): The 3 is for the 3 inputs: (x, a, b). */
        constexpr size_t input_dims[] = {3, 1};
        rot_tensor_t input_tensor = ROT_create_tensor(arena,
                                                      array_size(input_dims),
                                                      input_dims,
                                                      ROT_BACKEND_CPU);
        assert(input_tensor != NULL);

        constexpr uint32_t num_hidden_units = 16;

        constexpr size_t layer0_dims[] = {num_hidden_units, input_dims[0]};
        struct linear_layer layer0;
        init_layer(&layer0, arena, rng, layer0_dims);

        struct linear_layer out_layer;
        constexpr size_t output_layer_dims[] = {1, num_hidden_units};
        init_layer(&out_layer, arena, rng, output_layer_dims);

        for (uint32_t iter_i = 0;
             iter_i < num_train;
             ++iter_i) {
                float *input_data = ROT_tensor_get_data(input_tensor);
                struct beta_datum *datum = train_dataset.data + iter_i;
                input_data[0] = datum->x;
                input_data[1] = datum->alpha;
                input_data[2] = datum->beta;

                ROT_matmul(layer0.a, layer0.w, input_tensor);

                ROT_relu(layer0.a);

                ROT_matmul(out_layer.a, out_layer.w, layer0.a);
                /* TODO(brendan): find faster sigmoid and benchmark */
                float *pred_data = ROT_tensor_get_data(out_layer.a);
                pred_data[0] = 1.0f / (1.0f + exp(-pred_data[0]));

                float error = pred_data[0] - datum->y;
                error *= error;

                /* TODO(brendan): Chain rule! */
        }
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
#ifdef PLATFORM_CUDNN
        run_test(test_matmul_small_cudnn);
#endif /* PLATFORM_CUDNN */
#ifdef PLATFORM_MIOPEN
        run_test(test_matmul_small_miopen);
#endif /* PLATFORM_MIOPEN */
        run_test(test_matmul_small_perf);
        run_test(test_feedforward_backward);

        printf("All tests passed!\n");

        return EXIT_SUCCESS;
}
