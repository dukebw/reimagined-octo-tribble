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
#include "error/log_error.h"

#include <stdint.h>

#define ROT_ARENA_MIN_BYTES (sizeof(struct rot_arena) + 8)

struct rot_arena {
        size_t mem_bytes;
        size_t used_bytes;
};

bool ROT_arena_can_alloc(const rot_arena_t arena, size_t request_bytes)
{
        return request_bytes <= (arena->mem_bytes - arena->used_bytes);
}

void *ROT_arena_malloc(rot_arena_t arena, size_t malloc_bytes)
{
        if (arena == NULL) {
                LOG_NULL();
                return NULL;
        }

        if (!ROT_arena_can_alloc(arena, malloc_bytes)) {
                LOG_ERROR("Not enough space in arena to malloc.");
                return NULL;
        }

        void *result = (char *)arena + arena->used_bytes;

        arena->used_bytes += malloc_bytes;

        return result;
}

size_t ROT_arena_min_bytes(void)
{
        return ROT_ARENA_MIN_BYTES;
}

rot_arena_t ROT_arena_new(void *memory, size_t mem_bytes)
{
        if (memory == NULL) {
                LOG_NULL();
                return NULL;
        }

        if (mem_bytes < ROT_ARENA_MIN_BYTES) {
                LOG_ERROR("Provided memory size is less than minimal arena "
                          "size.");
                return NULL;
        }

        struct rot_arena *arena = memory;
        arena->mem_bytes = mem_bytes;
        arena->used_bytes = sizeof(struct rot_arena);

        return (struct rot_arena *)memory;
}
