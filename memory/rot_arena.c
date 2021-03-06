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

struct rot_arena_gpu {
        size_t block_bytes;
        void **mem_blocks;
        uint32_t num_blocks;
        size_t *used_bytes;
};

struct rot_arena {
        struct rot_arena_cpu cpu;
        struct rot_arena_gpu gpu;
};

static bool
arena_cpu_can_alloc(const struct rot_arena_cpu *arena_cpu,
                    size_t request_bytes)
{
        return request_bytes <= (arena_cpu->mem_bytes - arena_cpu->used_bytes);
}

/**
 * arena_gpu_can_alloc() - Checks whether the ROC arena `arena_gpu` is able to
 * allocate `request_bytes`.
 * @arena_gpu: The GPU memory arena from which bytes are being requested.
 * @request_bytes: Number of bytes requested.
 */
static bool
arena_gpu_can_alloc(const struct rot_arena_gpu *arena_gpu,
                    size_t request_bytes)
{
        if (request_bytes > arena_gpu->block_bytes)
                return false;

        for (uint32_t block_i = 0;
             block_i < arena_gpu->num_blocks;
             ++block_i) {
                size_t avail_bytes = (arena_gpu->block_bytes -
                                      arena_gpu->used_bytes[block_i]);
                if (avail_bytes >= request_bytes)
                        return true;
        }

        return false;
}

bool ROT_arena_can_alloc(const rot_arena_t arena,
                         size_t request_bytes,
                         enum rot_backend backend)
{
        if (arena == NULL) {
                LOG_NULL();
                return false;
        }

        if (backend == ROT_BACKEND_CPU) {
                return arena_cpu_can_alloc(&arena->cpu, request_bytes);
        } else if ((backend == ROT_BACKEND_ROC) ||
                   (backend == ROT_BACKEND_CUDA)) {
                return arena_gpu_can_alloc(&arena->gpu, request_bytes);
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
 * arena_gpu_malloc() - Attempts to allocate `malloc_bytes` from `arena_gpu`.
 * @arena_gpu: GPU arena from which to allocate memory.
 * @malloc_bytes: Number of bytes to allocate.
 *
 * NULL is returned on error, e.g. if there is no memory block in `arena_gpu`
 * that can be used to satisfy the request.
 *
 * `arena_gpu` must be checked for NULL by the caller.
 */
static void *
arena_gpu_malloc(struct rot_arena_gpu *arena_gpu, size_t malloc_bytes)
{
        for (uint32_t block_i = 0;
             block_i < arena_gpu->num_blocks;
             ++block_i) {
                size_t avail_bytes = (arena_gpu->block_bytes -
                                      arena_gpu->used_bytes[block_i]);
                if (avail_bytes >= malloc_bytes) {
                        void *result =
                                ((char *)arena_gpu->mem_blocks[block_i] +
                                 arena_gpu->used_bytes[block_i]);
                        arena_gpu->used_bytes[block_i] += malloc_bytes;

                        return result;
                }
        }

        return NULL;
}

void *ROT_arena_malloc(rot_arena_t arena,
                       size_t malloc_bytes,
                       enum rot_backend backend)
{
        /**
         * NOTE(brendan): `arena is checked for NULL in `ROT_arena_can_alloc`.
         */
        if (!ROT_arena_can_alloc(arena, malloc_bytes, backend)) {
                LOG_ERROR("Not enough space in arena to malloc.");
                return NULL;
        }

        switch (backend) {
        case ROT_BACKEND_CPU:
                return arena_cpu_malloc(&arena->cpu, malloc_bytes);
        case ROT_BACKEND_CUDA:
        case ROT_BACKEND_ROC:
                return arena_gpu_malloc(&arena->gpu, malloc_bytes);
        default:
                LOG_UNSUPPORTED();
                return NULL;
        }
}

size_t ROT_arena_min_bytes(void)
{
        return ROT_ARENA_MIN_BYTES;
}

struct rot_arena *
ROT_arena_gpu_new(struct rot_arena *arena,
                  void **memory,
                  size_t block_bytes,
                  uint32_t num_blocks)
{
        if ((arena == NULL) || (memory == NULL)) {
                LOG_NULL();
                return NULL;
        }

        for (uint32_t block_i = 0;
             block_i < num_blocks;
             ++block_i) {
                if (memory[block_i] == NULL) {
                        LOG_NULL();
                        return NULL;
                }
        }

        /**
         * NOTE(brendan): `ROT_arena_malloc` checks that `arena` has enough
         * space to allocate, so we just test the return value here.
         */
        size_t required_bytes = num_blocks*sizeof(size_t);
        arena->gpu.used_bytes = (size_t *)ROT_arena_malloc(arena,
                                                           required_bytes,
                                                           ROT_BACKEND_CPU);
        if (arena->gpu.used_bytes == NULL)
                return NULL;

        arena->gpu.block_bytes = block_bytes;
        arena->gpu.mem_blocks = memory;
        arena->gpu.num_blocks = num_blocks;

        for (uint32_t block_i = 0;
             block_i < num_blocks;
             ++block_i) {
                arena->gpu.used_bytes[block_i] = 0;
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
        arena->cpu.mem_bytes = mem_bytes;
        arena->cpu.used_bytes = sizeof(struct rot_arena);
        arena->gpu.block_bytes = 0;
        arena->gpu.mem_blocks = NULL;
        arena->gpu.num_blocks = 0;
        arena->gpu.used_bytes = NULL;

        return (struct rot_arena *)memory;
}
