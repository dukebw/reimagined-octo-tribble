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
#ifndef ROT_ARENA_H
#define ROT_ARENA_H

#include <stdlib.h>

typedef struct rot_arena *rot_arena_t;

/**
 * ROT_arena_min_bytes() - Returns the minimum number of bytes in an arena.
 */
size_t ROT_arena_min_bytes(void);

/**
 * ROT_arena_new() - Initializes a memory arena from memory allocated by the
 * caller.
 * @memory: Non-NULL pointer to an allocated contiguous buffer of mem_bytes
 * many bytes.
 * @mem_bytes: Size of memory in bytes.
 */
rot_arena_t ROT_arena_new(void *memory, size_t mem_bytes);

#endif /* ROT_ARENA_H */
