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
#ifndef MI_OPEN_H
#define MI_OPEN_H

#include "rot_math.h"

/**
 * matmul_roc() - Matrix multiplication using ROCm backend.
 */
rot_tensor_t matmul_roc(rot_tensor_t result,
                        const rot_tensor_t a,
                        const rot_tensor_t b);

#endif /* MI_OPEN_H */
