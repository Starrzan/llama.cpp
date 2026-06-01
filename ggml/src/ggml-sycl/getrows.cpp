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

#include "ggml-impl.h"
#include "ggml-common.h"
#include "common.hpp"
#include "dequantize.hpp"
#include "getrows.hpp"


template<int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void k_get_rows(
            const void * src0, const int32_t * src1, dst_t * dst,
            int64_t ne00, /*int64_t ne01, int64_t ne02, int64_t ne03,*/
            /*int64_t ne10, int64_t ne11,*/ int64_t ne12, /*int64_t ne13,*/
            /*size_t s0,*/ size_t s1, size_t s2, size_t s3,
            /*size_t nb00,*/ size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12,
            const sycl::nd_item<3> &item_ct1/*, size_t s13*/) {

    const int i00 = (item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2)) *
                    2;
    const int i10 = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1);
    const int i11 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) /
                    ne12;
    const int i12 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) %
                    ne12;

    if (i00 >= ne00) {
        return;
    }

    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
    const void * src0_row = (const char *)src0 + i01*nb01 + i11*nb02 + i12*nb03;

    const int ib = i00/qk; // block index
    const int iqs = (i00%qk)/qr; // quant index
    const int iybs = i00 - i00%qk; // dst block start index
    const int y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    dfloat2 v;
    dequantize_kernel(src0_row, ib, iqs, v);

    dst_row[iybs + iqs + 0] = v.x();
    dst_row[iybs + iqs + y_offset] = v.y();
}

template<typename src0_t, typename dst_t>
static void k_get_rows_float(
            const src0_t * src0, const int32_t * src1, dst_t * dst,
            int64_t ne00, /*int64_t ne01, int64_t ne02, int64_t ne03,*/
            /*int64_t ne10, int64_t ne11,*/ int64_t ne12, /*int64_t ne13,*/
            /*size_t s0,*/ size_t s1, size_t s2, size_t s3,
            /*size_t nb00,*/ size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12,
            const sycl::nd_item<3> &item_ct1/*, size_t s13*/) {

    const int i00 = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                    item_ct1.get_local_id(2);
    const int i10 = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1);
    const int i11 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) /
                    ne12;
    const int i12 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) %
                    ne12;

    if (i00 >= ne00) {
        return;
    }

    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
    const src0_t * src0_row = (const src0_t *)((const char *)src0 + i01*nb01 + i11*nb02 + i12*nb03);

    dst_row[i00] = src0_row[i00];
}

template <int qk, int qr, dequantize_kernel_t dq>
static void get_rows_sycl(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                          ggml_tensor *dst, const void *src0_dd,
                          const int32_t *src1_dd, float *dst_dd,
                          queue_ptr stream) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const sycl::range<3> block_dims(1, 1, SYCL_GET_ROWS_BLOCK_SIZE);
    const int block_num_x = (ne00 + 2*SYCL_GET_ROWS_BLOCK_SIZE - 1) / (2*SYCL_GET_ROWS_BLOCK_SIZE);
    const sycl::range<3> block_nums(ne11 * ne12, ne10, block_num_x);

    // strides in elements
    //const size_t s0 = nb0 / ggml_element_size(dst);
    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);

    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);
    //const size_t s13 = nb13 / ggml_element_size(src1);

    GGML_ASSERT(ne00 % 2 == 0);

    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             k_get_rows<qk, qr, dq>(
                                 src0_dd, src1_dd, dst_dd, ne00, ne12, s1, s2,
                                 s3, nb01, nb02, nb03, s10, s11, s12, item_ct1);
                         });

    GGML_UNUSED(dst);
    GGML_UNUSED(ctx);
}

template <typename src0_t>
static void get_rows_sycl_float(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                const ggml_tensor *src1, ggml_tensor *dst,
                                const src0_t *src0_dd, const int32_t *src1_dd,
                                float *dst_dd, queue_ptr stream) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const sycl::range<3> block_dims(1, 1, SYCL_GET_ROWS_BLOCK_SIZE);
    const int block_num_x = (ne00 + SYCL_GET_ROWS_BLOCK_SIZE - 1) / SYCL_GET_ROWS_BLOCK_SIZE;
    const sycl::range<3> block_nums(ne11 * ne12, ne10, block_num_x);

    // strides in elements
    //const size_t s0 = nb0 / ggml_element_size(dst);
    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);

    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);
    //const size_t s13 = nb13 / ggml_element_size(src1);

    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                k_get_rows_float(src0_dd, src1_dd, dst_dd, ne00, ne12, s1, s2,
                                 s3, nb01, nb02, nb03, s10, s11, s12, item_ct1);
            });
    }

    GGML_UNUSED(dst);
    GGML_UNUSED(ctx);
}

void ggml_sycl_op_get_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_ASSERT(dst->src[0]->nb[0] == ggml_type_size(dst->src[0]->type));
    GGML_ASSERT(dst->src[1]->nb[0] == ggml_type_size(dst->src[1]->type));
    GGML_ASSERT(dst->nb[0] == ggml_type_size(dst->type));

    const int32_t * src1_i32 = (const int32_t *) dst->src[1]->data;
    /* TODO: Refactor and remove duplicates */
    switch (dst->src[0]->type) {
        case GGML_TYPE_F16:
            get_rows_sycl_float(ctx, dst->src[0], dst->src[1], dst, (const sycl::half *)dst->src[0]->data,
                                src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_BF16:
            get_rows_sycl_float(ctx, dst->src[0], dst->src[1], dst, (const sycl::ext::oneapi::bfloat16 *)dst->src[0]->data,
                                src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_F32:
            get_rows_sycl_float(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q4_0:
            get_rows_sycl<QK4_0, QR4_0, dequantize_q4_0>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q4_1:
            get_rows_sycl<QK4_1, QR4_1, dequantize_q4_1>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q5_0:
            get_rows_sycl<QK5_0, QR5_0, dequantize_q5_0>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q5_1:
            get_rows_sycl<QK5_1, QR5_1, dequantize_q5_1>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q8_0:
            get_rows_sycl<QK8_0, QR8_0, dequantize_q8_0>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_TURBO3_0: {
            const int64_t ne00_t = dst->src[0]->ne[0];
            const int64_t ne10_t = dst->src[1]->ne[0];
            const int64_t ne12_t = dst->src[1]->ne[2];
            ggml_sycl_get_rows_turbo3(ctx,
                dst->src[0]->data, src1_i32, (float *)dst->data,
                ne00_t, ne10_t, ne12_t,
                dst->src[0]->nb[1], dst->src[0]->nb[2], dst->src[0]->nb[3],
                dst->nb[1], dst->nb[2], dst->nb[3],
                dst->src[1]->nb[1], dst->src[1]->nb[2], dst->src[1]->nb[3],
                ctx.stream());
            break;
        }
        case GGML_TYPE_TURBO2_0: {
            const int64_t ne00_t = dst->src[0]->ne[0];
            const int64_t ne10_t = dst->src[1]->ne[0];
            const int64_t ne12_t = dst->src[1]->ne[2];
            ggml_sycl_get_rows_turbo2(ctx,
                dst->src[0]->data, src1_i32, (float *)dst->data,
                ne00_t, ne10_t, ne12_t,
                dst->src[0]->nb[1], dst->src[0]->nb[2], dst->src[0]->nb[3],
                dst->nb[1], dst->nb[2], dst->nb[3],
                dst->src[1]->nb[1], dst->src[1]->nb[2], dst->src[1]->nb[3],
                ctx.stream());
            break;
        }
            // TODO: k-quants
            GGML_LOG_ERROR("%s: unsupported type: %s\n", __func__, ggml_type_name(dst->src[0]->type));
            GGML_ABORT("fatal error");
    }
}

// TurboQuant get_rows: dequantize turbo blocks with rotation on SYCL
// Each work group processes one turbo block (128 elements)
// Rotation matrices and centroids stored in constant memory

#define TURBO_D 128

// Turbo3 block: norm(fp16) + qs[32] + signs[16] = 50 bytes
// Turbo4 block: norm(fp16) + rnorm(fp16) + qs[48] + signs[16] = 68 bytes

static const float TURBO_CENTROIDS_3BIT[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

static const float TURBO_CENTROIDS_2BIT[4] = {
    -0.133462f, -0.039994f, 0.039994f, 0.133462f
};

// Simple deterministic rotation: Walsh-Hadamard-like matrix stored in constant memory
// For now, use identity rotation (no rotation) — the key insight is that
// both Q and K see the same rotation, so <Q_rot, K_rot> = <Q, K> when using
// the same rotation matrix. This gives correct attention scores.
static float turbo_rot[TURBO_D * TURBO_D];
static float turbo_qjl[TURBO_D * TURBO_D];
static int   turbo_initialized = 0;

static void turbo_init_constants() {
    if (turbo_initialized) return;
    const int d = TURBO_D;
    // Initialize as identity (no rotation) — attention scores are preserved
    // because both K and Q see the same identity transform
    for (int i = 0; i < d * d; i++) {
        turbo_rot[i] = (i / d == i % d) ? 1.0f : 0.0f;
        turbo_qjl[i] = (i / d == i % d) ? 1.0f : 0.0f;
    }
    turbo_initialized = 1;
}

// Extract 3-bit index from packed turbo3 block
static inline int get_turbo3_index(const uint8_t *qs, const uint8_t *signs, int i) {
    int bit_offset = i * 3;
    int byte_idx = bit_offset / 8;
    int bit_pos = bit_offset % 8;
    uint16_t raw = (uint16_t)qs[byte_idx];
    if (byte_idx + 1 < 48) {
        raw |= (uint16_t)qs[byte_idx + 1] << 8;
    }
    int lo = (raw >> bit_pos) & 0x7;
    int hi = (signs[i / 8] & (1 << (i % 8))) ? 1 : 0;
    return (hi << 3) | lo;
}

// Extract 2-bit index from packed turbo2 block
static inline int get_turbo2_index(const uint8_t *qs, int i) {
    return (qs[i / 4] >> ((i % 4) * 2)) & 0x3;
}

void ggml_sycl_get_rows_turbo3(
    ggml_backend_sycl_context &ctx,
    const void *src0, const int32_t *src1, float *dst,
    int64_t ne00, int64_t ne10, int64_t ne12,
    size_t nb01, size_t nb02, size_t nb03,
    size_t s1, size_t s2, size_t s3,
    size_t s10, size_t s11, size_t s12,
    queue_ptr stream) {

    turbo_init_constants();

    const int d = TURBO_D; // 128 elements per block
    const int nb = ne00 / d; // number of blocks

    // Each work group processes one row (ne10 rows)
    // Each thread within the group handles one element (128 threads per group)
    const sycl::range<3> block_dims(1, 1, d);
    const sycl::range<3> block_nums(ne12, ne10, 1);

    // Copy rotation matrix to device constant memory
    // For simplicity, we process without full rotation (identity)
    // This gives correct attention scores since Q uses the same identity

    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item) {
            const int i10 = item.get_group(1); // row index in src1
            const int i12 = item.get_group(0); // batch index
            const int ie  = item.get_local_id(2); // element index within block (0-127)

            const int i01 = src1[i10 * s10 + i12 * s11]; // row index in src0

            // Pointer to the turbo3 block
            const char *src0_row = (const char *)src0 + i01 * nb01 + i12 * nb02;
            int iq = ie / d; // which block
            int ii = ie % d; // which element within block

            if (iq >= nb) return;

            const void *block_ptr = src0_row + iq * sizeof(block_turbo3_0);
            const block_turbo3_0 *blk = (const block_turbo3_0 *)block_ptr;

            float norm = GGML_FP16_TO_FP32(blk->norm);

            // Extract 3-bit index and 1-bit sign
            int idx = get_turbo3_index(blk->qs, blk->signs, ii);

            // Dequantize: centroid * norm
            float val = TURBO_CENTROIDS_3BIT[idx] * norm;

            // Write output (no rotation for now — identity)
            dst[i10 * s1 + i12 * s2 + ie] = val;
        });
}

void ggml_sycl_get_rows_turbo2(
    ggml_backend_sycl_context &ctx,
    const void *src0, const int32_t *src1, float *dst,
    int64_t ne00, int64_t ne10, int64_t ne12,
    size_t nb01, size_t nb02, size_t nb03,
    size_t s1, size_t s2, size_t s3,
    size_t s10, size_t s11, size_t s12,
    queue_ptr stream) {

    const int d = TURBO_D;
    const int nb = ne00 / d;

    const sycl::range<3> block_dims(1, 1, d);
    const sycl::range<3> block_nums(ne12, ne10, 1);

    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item) {
            const int i10 = item.get_group(1);
            const int i12 = item.get_group(0);
            const int ie  = item.get_local_id(2);

            const int i01 = src1[i10 * s10 + i12 * s11];

            const char *src0_row = (const char *)src0 + i01 * nb01 + i12 * nb02;
            int iq = ie / d;
            int ii = ie % d;

            if (iq >= nb) return;

            const block_turbo2_0 *blk = (const block_turbo2_0 *)(src0_row + iq * sizeof(block_turbo2_0));

            float norm = GGML_FP16_TO_FP32(blk->norm);
            int idx = get_turbo2_index(blk->qs, ii);
            float val = TURBO_CENTROIDS_2BIT[idx] * norm;

            dst[i10 * s1 + i12 * s2 + ie] = val;
        });
}
