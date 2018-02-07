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
#ifndef ERROR_LOG_ERROR_H
#define ERROR_LOG_ERROR_H

#include <stdint.h>

#define LOG_ERROR(s) log_error(s, __func__, __FILE__, __LINE__)
#define LOG_NULL() LOG_ERROR("Null input.")
#define LOG_UNSUPPORTED() LOG_ERROR("Unsupported backend.")

/**
 * log_error.h - The purpose of this module is to provide internal interfaces
 * for error logging.
 */

/**
 * log_error() - Logs an error message, the function name, and current line.
 */
void log_error(const char *message,
               const char *func_name,
               const char *filename,
               int32_t line_number);

#endif // ERROR_LOG_ERROR_H
