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

#include "rot_arena.h"
#include <stdint.h>

#define ROT_ARENA_MIN_BYTES (sizeof(struct rot_arena) + 8)

struct rot_arena {
        size_t mem_bytes;
        size_t used_bytes;
};

size_t ROT_arena_min_bytes(void)
{
        return ROT_ARENA_MIN_BYTES;
}

rot_arena_t ROT_arena_new(void *memory, size_t mem_bytes)
{
        if (memory == NULL)
                return NULL;

        if (mem_bytes < ROT_ARENA_MIN_BYTES)
                return NULL;

        struct rot_arena *arena = memory;
        arena->mem_bytes = mem_bytes;
        arena->used_bytes = sizeof(struct rot_arena);

        return (struct rot_arena *)memory;
}
