#pragma once
// OCR engine -- image operators (Phase 2). Device-free DSL factories.
//
// Fused upload/color-conversion + resize + normalization matching the Phase 1
// CPU preprocess convention (half-pixel bilinear centers, clamp edges, area
// box resampling, out = (pixel/255 - mean)/std, CHW output). Source is HWC
// uint8 (3 channels); the upload buffer is uint32 words packing 4 bytes each
// (low byte first) so the transfer is a raw luisa::span copy.
//
// Also: crop (checked region copy), letterbox/aspect pad, and patchify/packing
// (CHW -> [num_patches, C*patch*patch] token rows). Tail/ragged/degenerate
// sizes are guarded; nothing is clamped or dropped.
#include <cstdint>

#include <luisa/dsl/syntax.h>

#include "ocr/runtime/dispatch.h"

namespace luisa::ocr {

// Fused color-convert + resize + normalize. src is HWC uint8 packed 4/word.
// mean/std are [3] (bind dummy + has_norm=0 to skip normalization).
// resize_kind: 0 = bilinear (half-pixel), 1 = area (box). bgr_to_rgb: 0/1.
// out is CHW float [3, dh, dw]. One thread per (dst pixel, channel).
using ImageResizeNormalizeKernel = compute::Kernel1D<void(
    compute::Buffer<uint32_t> /*src packed HWC uint8*/,
    compute::Buffer<float> /*out CHW*/,
    compute::Buffer<float> /*mean[3]*/,
    compute::Buffer<float> /*std_dev[3]*/,
    uint32_t /*sw*/, uint32_t /*sh*/, uint32_t /*dw*/, uint32_t /*dh*/,
    uint32_t /*bgr_to_rgb*/, uint32_t /*resize_kind*/, uint32_t /*has_norm*/)>;
[[nodiscard]] ImageResizeNormalizeKernel make_image_resize_normalize_kernel() noexcept;

// Crop: copy a [crop_h, crop_w] region at (x0,y0) from HWC uint8 src to a
// tightly-packed HWC uint8 dst. Both are byte buffers (one byte per element,
// no packing race). Out-of-range source pixels are zero-filled.
using ImageCropKernel = compute::Kernel1D<void(
    compute::Buffer<uint8_t> /*src HWC bytes*/,
    compute::Buffer<uint8_t> /*dst HWC bytes*/,
    uint32_t /*sw*/, uint32_t /*sh*/, uint32_t /*x0*/, uint32_t /*y0*/,
    uint32_t /*crop_w*/, uint32_t /*crop_h*/)>;
[[nodiscard]] ImageCropKernel make_image_crop_kernel() noexcept;

// Letterbox: place a [src_w,src_h] CHW float image into a [dst_w,dst_h] CHW
// float canvas at (pad_x,pad_y), filling the border with `pad_value`.
using ImageLetterboxKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*src CHW*/,
    compute::Buffer<float> /*dst CHW*/,
    uint32_t /*src_w*/, uint32_t /*src_h*/, uint32_t /*dst_w*/, uint32_t /*dst_h*/,
    uint32_t /*pad_x*/, uint32_t /*pad_y*/, float /*pad_value*/)>;
[[nodiscard]] ImageLetterboxKernel make_image_letterbox_kernel() noexcept;

// Patchify: CHW float [C,H,W] -> [num_patches, C*patch*patch] token rows,
// patch grid (H/patch, W/patch), row-major patches, each row is the patch's
// C*patch*patch values in (c, py, px) order. H,W must be multiples of patch.
using ImagePatchifyKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*src CHW*/,
    compute::Buffer<float> /*out [num_patches, C*patch*patch]*/,
    uint32_t /*C*/, uint32_t /*H*/, uint32_t /*W*/, uint32_t /*patch*/)>;
[[nodiscard]] ImagePatchifyKernel make_image_patchify_kernel() noexcept;

// Dispatch helpers (TOTAL threads).
[[nodiscard]] Result<LaunchPlan1D> plan_image_resize_dispatch(uint64_t dw, uint64_t dh);
[[nodiscard]] Result<LaunchPlan1D> plan_image_crop_dispatch(uint64_t crop_w, uint64_t crop_h);
[[nodiscard]] Result<LaunchPlan1D> plan_image_letterbox_dispatch(uint64_t dst_w, uint64_t dst_h);
[[nodiscard]] Result<LaunchPlan1D> plan_image_patchify_dispatch(uint64_t num_patches, uint64_t patch_elems);

// Pack HWC uint8 pixels into uint32 words (4 bytes/word, low byte first).
[[nodiscard]] luisa::vector<uint32_t> pack_image_words(luisa::span<const uint8_t> pixels);

[[nodiscard]] luisa::string image_variant_key(luisa::string_view op);

}// namespace luisa::ocr