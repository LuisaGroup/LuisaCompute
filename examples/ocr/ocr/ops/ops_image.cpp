// OCR engine -- image operator implementations (Phase 2).
#include "ocr/ops/ops_image.h"

#include <luisa/dsl/sugar.h>

#include "ocr/ops/ops_common.h"
#include "ocr/quantization.h"
#include "ocr/runtime/shader_cache.h"

namespace luisa::ocr {

using namespace luisa::compute;

[[nodiscard]] luisa::vector<uint32_t> pack_image_words(luisa::span<const uint8_t> pixels) {
    luisa::vector<uint32_t> words((pixels.size() + 3u) / 4u, 0u);
    for (size_t i = 0; i < pixels.size(); i++) {
        words[i / 4u] |= static_cast<uint32_t>(pixels[i]) << (8u * (i % 4u));
    }
    return words;
}

namespace {

// Read one HWC uint8 pixel channel from the packed source (4 bytes/word).
[[nodiscard]] Var<uint> read_pixel_byte(const BufferUInt &src, Var<uint> byte_index) noexcept {
    const auto word = src.read(byte_index >> 2u);
    return (word >> ((byte_index & 3u) * 8u)) & 0xFFu;
}

// Bilinear sample (half-pixel centers, clamp edges) matching Phase 1.
[[nodiscard]] Var<float> sample_bilinear(const BufferUInt &src, Var<uint> sw, Var<uint> sh, Var<uint> sc,
                                         Var<float> x, Var<float> y) noexcept {
    const auto x0f = floor(x);
    const auto y0f = floor(y);
    const auto x0 = cast<int>(x0f);
    const auto y0 = cast<int>(y0f);
    const auto fx = x - x0f;
    const auto fy = y - y0f;
    const auto at = [&](Var<int> px, Var<int> py) noexcept {
        const auto cx = clamp(px, 0, cast<int>(sw) - 1);
        const auto cy = clamp(py, 0, cast<int>(sh) - 1);
        const auto idx = (cast<uint>(cy) * sw + cast<uint>(cx)) * 3u + sc;
        return cast<float>(read_pixel_byte(src, idx));
    };
    const auto v00 = at(x0, y0);
    const auto v10 = at(x0 + 1, y0);
    const auto v01 = at(x0, y0 + 1);
    const auto v11 = at(x0 + 1, y0 + 1);
    const auto top = v00 + (v10 - v00) * fx;
    const auto bottom = v01 + (v11 - v01) * fx;
    return top + (bottom - top) * fy;
}

// Area (box) sample matching Phase 1: average source pixels in the footprint.
[[nodiscard]] Var<float> sample_area(const BufferUInt &src, Var<uint> sw, Var<uint> sh, Var<uint> sc,
                                     Var<uint> dx, Var<uint> dy, Var<uint> dw, Var<uint> dh) noexcept {
    const auto sx = cast<float>(sw) / cast<float>(dw);
    const auto sy = cast<float>(sh) / cast<float>(dh);
    const auto x_begin = cast<float>(dx) * sx;
    const auto x_end = cast<float>(dx + 1u) * sx;
    const auto y_begin = cast<float>(dy) * sy;
    const auto y_end = cast<float>(dy + 1u) * sy;
    const auto x0 = cast<uint>(floor(x_begin));
    const auto x1 = min(sw, cast<uint>(ceil(x_end)));
    const auto y0 = cast<uint>(floor(y_begin));
    const auto y1 = min(sh, cast<uint>(ceil(y_end)));
    Var<float> sum = def(0.0f);
    Var<float> weight = def(0.0f);
    $for (y, y0, y1) {
        const auto wy = min(y_end, cast<float>(y + 1u)) - max(y_begin, cast<float>(y));
        $for (x, x0, x1) {
            const auto wx = min(x_end, cast<float>(x + 1u)) - max(x_begin, cast<float>(x));
            const auto w = wx * wy;
            const auto idx = (y * sw + x) * 3u + sc;
            sum += cast<float>(read_pixel_byte(src, idx)) * w;
            weight += w;
        };
    };
    Var<float> result = def(0.0f);
    $if (weight > 0.0f) { result = sum / weight; };
    return result;
}

}// namespace

[[nodiscard]] ImageResizeNormalizeKernel make_image_resize_normalize_kernel() noexcept {
    ImageResizeNormalizeKernel kernel = [](BufferUInt src, BufferFloat out, BufferFloat mean, BufferFloat std_dev,
                                           Var<uint> sw, Var<uint> sh, Var<uint> dw, Var<uint> dh,
                                           Var<uint> bgr_to_rgb, Var<uint> resize_kind, Var<uint> has_norm) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_image_resize_normalize");
        const auto id = ocr_linear_id();
        const auto plane = dw * dh;
        $if (id < plane * 3u) {
            const auto c = id / plane;
            const auto pix = id % plane;
            const auto x = pix % dw;
            const auto y = pix / dw;
            // Channel swap on read when converting BGR->RGB.
            const auto sc = select(c, 2u - c, bgr_to_rgb != 0u);
            const auto scale_x = cast<float>(sw) / cast<float>(dw);
            const auto scale_y = cast<float>(sh) / cast<float>(dh);
            Var<float> value = def(0.0f);
            $if (sw == dw & sh == dh) {
                value = cast<float>(read_pixel_byte(src, (y * sw + x) * 3u + sc));
            } $else {
                $if (resize_kind == 0u) {
                    value = sample_bilinear(src, sw, sh, sc,
                                            (cast<float>(x) + 0.5f) * scale_x - 0.5f,
                                            (cast<float>(y) + 0.5f) * scale_y - 0.5f);
                } $else {
                    value = sample_area(src, sw, sh, sc, x, y, dw, dh);
                };
            };
            Var<float> normalized = value / 255.0f;
            $if (has_norm != 0u) {
                normalized = (normalized - mean.read(c)) / std_dev.read(c);
            };
            out.write(c * plane + pix, normalized);
        };
    };
    return kernel;
}

[[nodiscard]] ImageCropKernel make_image_crop_kernel() noexcept {
    ImageCropKernel kernel = [](BufferVar<uint8_t> src, BufferVar<uint8_t> dst, Var<uint> sw, Var<uint> sh,
                                Var<uint> x0, Var<uint> y0, Var<uint> crop_w, Var<uint> crop_h) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_image_crop");
        const auto id = ocr_linear_id();
        $if (id < crop_w * crop_h * 3u) {
            const auto c = id % 3u;
            const auto pix = id / 3u;
            const auto x = pix % crop_w;
            const auto y = pix / crop_w;
            const auto sx = x0 + x;
            const auto sy = y0 + y;
            Var<uint8_t> value = def<uint8_t>(0u);
            $if (sx < sw & sy < sh) {
                value = src.read((sy * sw + sx) * 3u + c);
            };
            dst.write(id, value);
        };
    };
    return kernel;
}

[[nodiscard]] ImageLetterboxKernel make_image_letterbox_kernel() noexcept {
    ImageLetterboxKernel kernel = [](BufferFloat src, BufferFloat dst, Var<uint> src_w, Var<uint> src_h,
                                     Var<uint> dst_w, Var<uint> dst_h, Var<uint> pad_x, Var<uint> pad_y,
                                     Var<float> pad_value) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_image_letterbox");
        const auto id = ocr_linear_id();
        const auto plane = dst_w * dst_h;
        $if (id < plane * 3u) {
            const auto c = id / plane;
            const auto pix = id % plane;
            const auto x = pix % dst_w;
            const auto y = pix / dst_w;
            Var<float> value = def(pad_value);
            $if (x >= pad_x & y >= pad_y) {
                const auto sx = x - pad_x;
                const auto sy = y - pad_y;
                $if (sx < src_w & sy < src_h) {
                    value = src.read(c * src_w * src_h + sy * src_w + sx);
                };
            };
            dst.write(id, value);
        };
    };
    return kernel;
}

[[nodiscard]] ImagePatchifyKernel make_image_patchify_kernel() noexcept {
    ImagePatchifyKernel kernel = [](BufferFloat src, BufferFloat out, Var<uint> C, Var<uint> H, Var<uint> W,
                                    Var<uint> patch) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_image_patchify");
        const auto id = ocr_linear_id();
        const auto patches_w = W / patch;
        const auto patches_h = H / patch;
        const auto patch_elems = C * patch * patch;
        $if (id < patches_w * patches_h * patch_elems) {
            const auto p = id / patch_elems;
            const auto e = id % patch_elems;
            const auto px = p % patches_w;
            const auto py = p / patches_w;
            const auto c = e / (patch * patch);
            const auto rem = e % (patch * patch);
            const auto iy = rem / patch;
            const auto ix = rem % patch;
            const auto sy = py * patch + iy;
            const auto sx = px * patch + ix;
            out.write(id, src.read(c * H * W + sy * W + sx));
        };
    };
    return kernel;
}

[[nodiscard]] Result<LaunchPlan1D> plan_image_resize_dispatch(uint64_t dw, uint64_t dh) {
    uint64_t count = 0;
    if (!checked_mul(dw, dh, count) || !checked_mul(count, 3u, count)) {
        return Result<LaunchPlan1D>::failure(StatusCode::invalid_argument, "image element count overflow", "dw*dh*3");
    }
    return plan_threads(count, 256u);
}

[[nodiscard]] Result<LaunchPlan1D> plan_image_crop_dispatch(uint64_t crop_w, uint64_t crop_h) {
    uint64_t count = 0;
    if (!checked_mul(crop_w, crop_h, count) || !checked_mul(count, 3u, count)) {
        return Result<LaunchPlan1D>::failure(StatusCode::invalid_argument, "crop element count overflow", "crop_w*crop_h*3");
    }
    return plan_threads(count, 256u);
}

[[nodiscard]] Result<LaunchPlan1D> plan_image_letterbox_dispatch(uint64_t dst_w, uint64_t dst_h) {
    uint64_t count = 0;
    if (!checked_mul(dst_w, dst_h, count) || !checked_mul(count, 3u, count)) {
        return Result<LaunchPlan1D>::failure(StatusCode::invalid_argument, "letterbox element count overflow", "dst_w*dst_h*3");
    }
    return plan_threads(count, 256u);
}

[[nodiscard]] Result<LaunchPlan1D> plan_image_patchify_dispatch(uint64_t num_patches, uint64_t patch_elems) {
    uint64_t count = 0;
    if (!checked_mul(num_patches, patch_elems, count)) {
        return Result<LaunchPlan1D>::failure(StatusCode::invalid_argument, "patchify element count overflow", "num_patches*patch_elems");
    }
    return plan_threads(count, 256u);
}

[[nodiscard]] luisa::string image_variant_key(luisa::string_view op) {
    return make_shader_key("image", op);
}

}// namespace luisa::ocr