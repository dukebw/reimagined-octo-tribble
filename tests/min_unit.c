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
#include "stopif.h"

static uint32_t min_unit_num_tests_run;

void min_unit_assert(bool did_test_pass, char *msg_format_str, ...)
{
        if (!did_test_pass) {
                va_list args;
                va_start(args, msg_format_str);

                vprintf(msg_format_str, args);
                fprintf(stderr,
                        "Test failed!\nTests run: %d\n",
                        min_unit_num_tests_run);
                abort();

                va_end(args);
        }
}

void min_unit_run_test(min_unit_test_func *test_fn)
{
        stopif(test_fn == NULL, "Null input to min_unit_run_test");
        ++min_unit_num_tests_run;
        test_fn();
}
