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
#ifndef _MIN_UNIT_H_
#define _MIN_UNIT_H_

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define MIN_UNIT_TEST_FUNC(name) void name(void)
typedef MIN_UNIT_TEST_FUNC(min_unit_test_func);

/**
 * min_unit_assert() - Asserts that a given test passed, printing the number of
 * tests run and an error message upon failure.
 * @did_test_pass: Did the test pass?
 * @msg_format_str: Format string to be printed on error.
 */
void min_unit_assert(bool did_test_pass, char *msg_format_str, ...);

/**
 * min_unit_run_test() - Runs `test_fn` and keeps track of the total number of
 * tests run.
 * @test_fn: A test function with some min_unit_assert calls in it.
 */
void min_unit_run_test(min_unit_test_func *test_fn);

#endif /* _MIN_UNIT_H_ */
