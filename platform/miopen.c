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
#include "platform/miopen.h"

#include "hip/hip_runtime_api.h"
#include "rocblas.h"

rot_tensor_t
matmul_roc(rot_tensor_t result, const rot_tensor_t a, const rot_tensor_t b)
{
        const float *a_dev = (const float *)ROT_tensor_get_data(a);
        const float *b_dev = (const float *)ROT_tensor_get_data(b);
        float *result_dev = (float *)ROT_tensor_get_data(result);
        if ((a_dev == NULL) || (b_dev == NULL) || (result_dev == NULL)) {
                LOG_ERROR("ROC tensor argument has uninitialized memory.");
                return NULL;
        }

        rocblas_handle handle;
        rocblas_status rblas_err = rocblas_create_handle(&handle);
        if (rblas_err != rocblas_status_success) {
                LOG_ERROR("ROC error creating handle.");
                return NULL;
        }

        const float alpha = 1.0f;
        const float beta = 0.0f;
        rblas_err = rocblas_sgemm(handle,
                                  rocblas_operation_none,
                                  rocblas_operation_none,
                                  b->dims[1],
                                  a->dims[0],
                                  a->dims[1],
                                  &alpha,
                                  b_dev,
                                  b->dims[1],
                                  a_dev,
                                  a->dims[1],
                                  &beta,
                                  result_dev,
                                  b->dims[1]);
        if (rblas_err != rocblas_status_success) {
                LOG_ERROR("ROC sgemm error.");
                return NULL;
        }

        rblas_err = rocblas_destroy_handle(handle);
        if (rblas_err != rocblas_status_success) {
                LOG_ERROR("ROC error destroying handle.");
                return NULL;
        }

        return result;
}
