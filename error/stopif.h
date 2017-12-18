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
#ifndef ERROR_STOPIF_H
#define ERROR_STOPIF_H

#include <stdbool.h>
#include <stdio.h>

/**
 * stopif() - Stops the program and prints `msg_format_str` if `assertion` is
 * true, else does nothing.
 * @assertion: Statement that must be true, otherwise the program stops.
 * @msg_format_str: printf format string to be printed if `assertion` is
 * triggered.
 */
void stopif(bool assertion, char *msg_format_str, ...);

/**
 * stopif_set_error_log() - Set the error log file to be printed to upon
 * program halt.
 * @error_log: Log file.
 */
void stopif_set_error_log(FILE *error_log);

#endif /* ERROR_STOPIF_H */
