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

struct rot_arena_cpu {
        size_t mem_bytes;
        size_t used_bytes;
};

struct rot_arena_roc {
        void **mem_blocks;
        size_t block_bytes;
        uint32_t num_blocks;
        size_t used_bytes[];
};

struct rot_arena {
        enum rot_backend backend;
        union {
                struct rot_arena_cpu cpu;
                struct rot_arena_roc roc;
        };
};

static bool
arena_cpu_can_alloc(const struct rot_arena_cpu *arena_cpu,
                    size_t request_bytes)
{
        return request_bytes <= (arena_cpu->mem_bytes - arena_cpu->used_bytes);
}

/**
 * arena_roc_can_alloc() - Checks whether the ROC arena `arena_roc` is able to
 * allocate `request_bytes`.
 * @arena_roc: The ROC memory arena from which bytes are being requested.
 * @request_bytes: Number of bytes requested.
 */
static bool
arena_roc_can_alloc(const struct rot_arena_roc *arena_roc,
                    size_t request_bytes)
{
        if (request_bytes > arena_roc->block_bytes)
                return false;

        for (uint32_t block_i = 0;
             block_i < arena_roc->num_blocks;
             ++block_i) {
                size_t avail_bytes = (arena_roc->block_bytes -
                                      arena_roc->used_bytes[block_i]);
                if (avail_bytes >= request_bytes)
                        return true;
        }

        return false;
}

bool ROT_arena_can_alloc(const rot_arena_t arena, size_t request_bytes)
{
        if (arena == NULL) {
                LOG_NULL();
                return false;
        }

        if (arena->backend == ROT_BACKEND_CPU) {
                return arena_cpu_can_alloc(&arena->cpu, request_bytes);
        } else if (arena->backend == ROT_BACKEND_ROC) {
                return arena_roc_can_alloc(&arena_roc->roc, request_bytes);
        } else {
                LOG_UNSUPPORTED();
                return false;
        }
}

static void *
arena_cpu_malloc(struct rot_arena_cpu *arena_cpu, size_t malloc_bytes)
{
        void *result = (char *)arena_cpu + arena_cpu->used_bytes;

        arena_cpu->used_bytes += malloc_bytes;

        return result;
}

/**
 * arena_roc_malloc() - Attempts to allocate `malloc_bytes` from `arena_roc`.
 * @arena_roc: ROC arena from which to allocate memory.
 * @malloc_bytes: Number of bytes to allocate.
 *
 * NULL is returned on error, e.g. if there is no memory block in `arena_roc`
 * that can be used to satisfy the request.
 *
 * `arena_roc` must be checked for NULL by the caller.
 */
static void *
arena_roc_malloc(struct rot_arena_roc *arena_roc, size_t malloc_bytes)
{
        for (uint32_t block_i = 0;
             block_i < arena_roc->num_blocks;
             ++block_i) {
                size_t avail_bytes = (arena_roc->block_bytes -
                                      arena_roc->used_bytes[block_i]);
                if (avail_bytes >= malloc_bytes) {
                        void *result =
                                ((char *)arena_roc->mem_blocks[block_i] +
                                 arena_roc->used_bytes[block_i]);
                        arena_roc->used_bytes[block_i] += malloc_bytes;

                        return result;
                }
        }

        return NULL;
}

void *ROT_arena_malloc(rot_arena_t arena, size_t malloc_bytes)
{
        /**
         * NOTE(brendan): `arena is checked for NULL in `ROT_arena_can_alloc`.
         */
        if (!ROT_arena_can_alloc(arena, malloc_bytes)) {
                LOG_ERROR("Not enough space in arena to malloc.");
                return NULL;
        }

        if (arena->backend == ROT_BACKEND_CPU) {
                return arena_cpu_malloc(&arena->cpu, malloc_bytes)
        } else if (arena->backend == ROT_BACKEND_ROC) {
                return arena_roc_malloc(&arena->roc, malloc_bytes);
        } else {
                LOG_UNSUPPORTED();
                return NULL;
        }
}

size_t ROT_arena_min_bytes(void)
{
        return ROT_ARENA_MIN_BYTES;
}

/**
 * arena_roc_new() - Initializes a memory arena for the Radeon Open Compute
 * platform, i.e. AMD GPUs.
 * @arena: `ROT_arena` from which to allocate the ROC memory arena. Memory
 * on CPU is used to store the metadata for the GPU memory arena.
 * @memory: Pointer to `num_blocks` allocated memory pointers. Each memory
 * pointer must point to a contiguous block of `mem_bytes` of memory.
 * @block_bytes: Amount of memory allocated in each block.
 * @num_blocks: Number of block pointers pointed to by `memory`. I.e. the
 * number of blocks of memory allocated on the GPU, for use by the memory
 * arena.
 */
static struct rot_arena *
arena_roc_new(struct rot_arena *arena,
              void **memory,
              size_t block_bytes,
              uint32_t num_blocks)
{
        /**
         * TODO(brendan): This is no longer quite right, since
         * `struct rot_arena_roc` is now a member of `struct rot_arena` (via
         * the union).
         */
        /**
         * NOTE(brendan): `ROT_arena_malloc` checks that `arena` has enough
         * space to allocate, so we just test the return value here.
         */
        size_t required_bytes = (sizeof(struct rot_arena_roc) +
                                 num_blocks*sizeof(size_t));
        struct rot_arena_roc *arena_roc =
                (struct rot_arena_roc *)ROT_arena_malloc(arena,
                                                         required_bytes);
        if (arena_roc == NULL)
                return NULL;

        arena_roc->block_bytes = block_bytes;
        arena_roc->mem_blocks = memory;
        arena_roc->num_blocks = num_blocks;

        for (uint32_t block_i = 0;
             block_i < num_blocks;
             ++block_i) {
                arena_roc->used_bytes[block_i] = 0;
        }

        return arena;
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

        struct rot_arena *arena = (struct rot_arena *)memory;
        arena->mem_bytes = mem_bytes;
        arena->used_bytes = sizeof(struct rot_arena);

        if (arena->backend == ROT_BACKEND_CPU) {
                return (struct rot_arena *)memory;
        } else if (arena->backend == ROT_BACKEND_ROC) {
        }
}
