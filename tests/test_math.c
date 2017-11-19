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

#include <assert.h>
#include <float.h>
#include <stdlib.h>
#include <sys/time.h>

static THFloatTensor *
create_th_tensor(float *data, size_t *dims, size_t flat_size)
{
        THFloatTensor *th_tensor = THFloatTensor_newWithSize2d(dims[0],
                                                               dims[1]);
        assert(th_tensor != NULL);

        memcpy(th_tensor->storage->data, data, flat_size*sizeof(float));

        return th_tensor;
}

/**
 * Allocates a GSL RNG and seeds it with the time in microseconds.
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

        struct timeval seed;
        int32_t status = gettimeofday(&seed, NULL);
        assert(status == 0);

        gsl_rng_set(rng, seed.tv_usec);

        return rng;
}

static MIN_UNIT_TEST_FUNC(test_matmul)
{
        uint8_t memory[128];

        rot_arena_t arena = ROT_arena_new(memory, sizeof(memory));
        min_unit_assert(arena != NULL, "ROT_arena_new failed");

        /**
         * TODO(brendan):
         * 1. Allocate memory for tensors a and b.
         * 2. Fill the tensors with random values. Uniform random from GSL?
         * 3. A*B. Check result (functional req.) against known BLAS library?
         * 4. Check performance against known BLAS library.
         */
        size_t dims[2] = {4, 4};
        rot_tensor_t a = ROT_create_tensor(arena, 2, dims);
        assert(a != NULL);

        float *a_data = ROT_tensor_get_data(a);
        assert(a_data != NULL);

        rot_tensor_t b = ROT_create_tensor(arena, 2, dims);
        assert(b != NULL);

        float *b_data = ROT_tensor_get_data(b);
        assert(b_data != NULL);

        gsl_rng *rng = get_gsl_rng();

        size_t flat_size = dims[0]*dims[1];
        for (uint32_t i = 0;
             i < flat_size;
             ++i) {
                a_data[i] = gsl_ran_flat(rng, -1, 1);
                b_data[i] = gsl_ran_flat(rng, -1, 1);
        }

        THFloatTensor *th_tensor_a = create_th_tensor(a_data, dims, flat_size);
        THFloatTensor *th_tensor_b = create_th_tensor(b_data, dims, flat_size);

        THFloatTensor_addmm(th_tensor_a,
                            0.0,
                            th_tensor_a,
                            1.0,
                            th_tensor_a,
                            th_tensor_b);

        a = ROT_matmul(a, a, b);
        min_unit_assert(a != NULL,
                        "NULL returned from ROT_matmul, expected rot_tensor");

        for (uint32_t i = 0;
             i < flat_size;
             ++i) {
                float diff = th_tensor_a->storage->data[i] - a_data[i];
                min_unit_assert(fabs(diff) < FLT_EPSILON,
                                "ROT_matmul mismatches TH_addmm at index %d",
                                i);
        }

        THFloatTensor_free(th_tensor_a);
        gsl_rng_free(rng);
}

int main(void)
{
        test_matmul();

        printf("All tests passed!\n");

        return EXIT_SUCCESS;
}
