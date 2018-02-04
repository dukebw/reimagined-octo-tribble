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

#include "rot_platform.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct rot_arena *rot_arena_t;

/**
 * ROT_arena_can_alloc() - Can `arena` satisfy a request to allocate
 * `request_bytes` bytes?
 * @arena: Memory arena to request bytes from.
 * @request_bytes: Number of bytes requested.
 * @backend: Backend for which to check whether memory allocation is possible.
 */
bool ROT_arena_can_alloc(const rot_arena_t arena,
                         size_t request_bytes,
                         enum rot_backend backend);

/**
 * ROT_arena_malloc() - Allocate memory from arena.
 * @arena: Memory arena to allocate memory from.
 * @malloc_bytes: Number of bytes to allocate.
 * @backend: Backend to allocate for.
 *
 * NULL is returned on error, e.g. if there is not enough memory in `arena` to
 * allocate `malloc_bytes`.
 */
void *ROT_arena_malloc(rot_arena_t arena,
                       size_t malloc_bytes,
                       enum rot_backend backend);

/**
 * ROT_arena_min_bytes() - Returns the minimum number of bytes in an arena.
 */
size_t ROT_arena_min_bytes(void);

/**
 * ROT_arena_new() - Initializes a memory arena from memory allocated by the
 * caller.
 * @memory: Non-NULL pointer to an array of allocated contiguous buffer of
 * mem_bytes many bytes.
 * @mem_bytes: Size of memory in bytes.
 */
rot_arena_t ROT_arena_new(void *memory, size_t mem_bytes);

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
 *
 * When creating a GPU arena the following steps should be followed:
 *
 * 1. Find the number of bytes needed CPU side as the value returned from
 * `ROT_arena_min_bytes` plus `sizeof(size_t)` bytes for every memory block.
 *
 * 2. Allocate a `rot_arena` using `ROT_arena_new`, with at least as much
 * memory as computed in step 1.
 *
 * 3. Call `arena_roc_new` with the `rot_arena` from step 2, pointers to the
 * allocated GPU memory blocks, and the other arguments above.
 *
 * `arena` will be filled in with the passed arguments and will be returned on
 * success.
 *
 * NULL is returned on failure.
 */
struct rot_arena *
ROT_arena_roc_new(struct rot_arena *arena,
                  void **memory,
                  size_t block_bytes,
                  uint32_t num_blocks);

#endif /* ROT_ARENA_H */
