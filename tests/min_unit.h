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
#ifndef TEST_MIN_UNIT_H
#define TEST_MIN_UNIT_H

#include <stdint.h>  /* for int32_t */

#define MIN_UNIT_TEST_FUNC(name) void name(void)
typedef MIN_UNIT_TEST_FUNC(min_unit_test_func);

#define MIN_UNIT_ASSERT(b, s, ...)  \
        min_unit_assert(b, __func__, __FILE__, __LINE__, s, ##__VA_ARGS__)

/**
 * min_unit_assert() - Asserts that a given test passed, printing the number of
 * tests run and an error message upon failure.
 * @did_test_pass: Did the test pass?
 * @msg_format_str: Format string to be printed on error.
 *
 * This function should not be called directly. Test asserts should go through
 * `MIN_UNIT_ASSERT`.
 */
void min_unit_assert(bool did_test_pass,
                     const char *func_name,
                     const char *filename,
                     int32_t line_number,
                     const char *msg_format_str,
                     ...);

/**
 * min_unit_run_test() - Runs `test_fn` and keeps track of the total number of
 * tests run.
 * @test_fn: A test function with some min_unit_assert calls in it.
 */
void min_unit_run_test(min_unit_test_func *test_fn);

#endif /* TEST_MIN_UNIT_H */
