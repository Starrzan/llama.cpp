//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_GETROWS_HPP
#define GGML_SYCL_GETROWS_HPP

#include "common.hpp"

void ggml_sycl_op_get_rows(ggml_backend_sycl_context & ctx, ggml_tensor *dst);

// TurboQuant get_rows: dequantize turbo blocks on-the-fly during gather
void ggml_sycl_get_rows_turbo3(
    ggml_backend_sycl_context &ctx,
    const void *src0, const int32_t *src1, float *dst,
    int64_t ne00, int64_t ne10, int64_t ne12,
    size_t nb01, size_t nb02, size_t nb03,
    size_t s1, size_t s2, size_t s3,
    size_t s10, size_t s11, size_t s12,
    queue_ptr stream);

void ggml_sycl_get_rows_turbo2(
    ggml_backend_sycl_context &ctx,
    const void *src0, const int32_t *src1, float *dst,
    int64_t ne00, int64_t ne10, int64_t ne12,
    size_t nb01, size_t nb02, size_t nb03,
    size_t s1, size_t s2, size_t s3,
    size_t s10, size_t s11, size_t s12,
    queue_ptr stream);

#endif // GGML_SYCL_GETROWS_HPP
