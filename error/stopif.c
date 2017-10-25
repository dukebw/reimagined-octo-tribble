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
#include <stdarg.h>
#include <stdlib.h>

static FILE *stopif_error_log;

void stopif(bool assertion, char *msg_format_str, ...)
{
        if (assertion) {
                va_list args;
                va_start(args, msg_format_str);

                vfprintf(stopif_error_log ? stopif_error_log : stderr,
                         msg_format_str,
                         args);
                fprintf(stopif_error_log ? stopif_error_log : stderr, "\n");
                abort();

                va_end(args);
        }
}

void stopif_set_error_log(FILE *new_error_log)
{
        stopif_error_log = new_error_log;
}
