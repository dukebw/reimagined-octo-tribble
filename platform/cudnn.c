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
#include "platform/cudnn.h"
#include "error/log_error.h"  /* for LOG_ERROR */

#include "cublas_v2.h"  /* for CUBLAS_OP_N, CUBLAS_STATUS_SUCCESS */

#include <stddef.h>  /* for NULL, size_t */

rot_tensor_t
matmul_cuda(rot_tensor_t result, const rot_tensor_t a, const rot_tensor_t b)
{
        const float *a_dev = (const float *)ROT_tensor_get_data(a);
        const float *b_dev = (const float *)ROT_tensor_get_data(b);
        float *result_dev = (float *)ROT_tensor_get_data(result);
        if ((a_dev == NULL) || (b_dev == NULL) || (result_dev == NULL)) {
                LOG_ERROR("CUDA tensor argument has uninitialized memory.");
                return NULL;
        }

        cublasHandle_t handle;
        cublasStatus_t cublas_status = cublasCreate(&handle);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                LOG_ERROR("cublasCreate error.");
                return NULL;
        }

        const size_t *a_dims = ROT_tensor_get_dims(a);
        const size_t *b_dims = ROT_tensor_get_dims(b);
        if ((a_dims == NULL) || (b_dims == NULL)) {
                LOG_ERROR("a or b dims uninitialized.");
                return NULL;
        }

        const float alpha = 1.0;
        const float beta = 0.0;
        cublas_status = cublasSgemm(handle,
                                    CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    b_dims[1],
                                    a_dims[0],
                                    a_dims[1],
                                    &alpha,
                                    b_dev,
                                    b_dims[1],
                                    a_dev,
                                    a_dims[1],
                                    &beta,
                                    result_dev,
                                    b_dims[1]);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                LOG_ERROR("cuBLAS sgemm error.");
                return NULL;
        }

        cublas_status = cublasDestroy(handle);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                LOG_ERROR("cuBLAS error destroying handle.");
                return NULL;
        }

        return result;
}
