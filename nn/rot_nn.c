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
#include "rot_nn.h"
#include <stdint.h>

rot_tensor_t ROT_relu(rot_tensor_t tensor)
{
        if (tensor == NULL)
                return NULL;

        /* TODO(brendan): speed test vs. NNPACK */
        /* TODO(brendan): GPU implementations... */
        float *data = ROT_tensor_get_data(tensor);
        const size_t *dims = ROT_tensor_get_dims(tensor);
        for (uint32_t i = 0;
             i < dims[0];
             ++i) {
                float val = data[i];
                if (__builtin_signbit(val))
                        data[i] = 0.0f;
        }

        return tensor;
}

/**
 * TODO(brendan): Replace this forward-propagated derivative with reverse mode.
 */
rot_tensor_t ROT_relu_grad(rot_tensor_t out_grad,
                           rot_tensor_t in_grad,
                           rot_tensor_t activations)
{
        if ((out_grad == NULL) || (in_grad == NULL) || (activations == NULL)) {
                LOG_NULL();
                return NULL;
        }

        /* TODO(brendan): What if activations are zero-length? */
        size_t flattened_size = ROT_tensor_get_size(activations);
        float *act_data = ROT_tensor_get_data(activations);
        /* TODO(brendan): GPU implementation. */
        bool is_zero = true;
        for (uint32_t i = 0;
             i < flattened_size;
             ++i) {
                if (act_data[i] != 0.0f) {
                        is_zero = false;
                        break;
                }
        }

        if (is_zero) {
                if (ROT_set_dims(out_grad,
                                 in_grad->num_dims,
                                 in_grad->dims) == NULL) {
                        LOG_ERROR("Set dims failed.");
                        return NULL;
                }

                /**
                 * TODO(brendan): presumably the dimensions should be the same
                 * as the activations dimensions?
                 *
                 * Check reverse mode in autograd.
                 */
        }

        return out_grad;
}
