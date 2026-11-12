// OCR engine -- production preprocessing reference (Phase 1).
// Checked counts, explicit resize conventions, source-coordinate provenance.
#include "ocr/preprocess.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace luisa::ocr {

[[nodiscard]] luisa::string_view resize_convention_name(ResizeConvention convention) noexcept {
    switch (convention) {
        case ResizeConvention::bilinear: return "bilinear";
        case ResizeConvention::area: return "area";
        default: return "unknown";
    }
}

namespace {

[[nodiscard]] uint32_t round_up_multiple(uint32_t value, uint32_t multiple) noexcept {
    if (multiple <= 1u) { return value; }
    const uint64_t v = value;
    const uint64_t m = multiple;
    return static_cast<uint32_t>(((v + m - 1u) / m) * m);
}

// Sample the source at fractional coordinates with bilinear interpolation
// (half-pixel centers). src is HWC uint8.
[[nodiscard]] float sample_bilinear(luisa::span<const uint8_t> src, uint32_t width, uint32_t height,
                                    uint32_t channel, float x, float y) noexcept {
    const float x0f = std::floor(x);
    const float y0f = std::floor(y);
    const int32_t x0 = static_cast<int32_t>(x0f);
    const int32_t y0 = static_cast<int32_t>(y0f);
    const float fx = x - x0f;
    const float fy = y - y0f;
    const auto at = [&](int32_t px, int32_t py) -> float {
        const int32_t cx = std::clamp(px, 0, static_cast<int32_t>(width) - 1);
        const int32_t cy = std::clamp(py, 0, static_cast<int32_t>(height) - 1);
        return static_cast<float>(src[(static_cast<size_t>(cy) * width + cx) * 3u + channel]);
    };
    const float v00 = at(x0, y0);
    const float v10 = at(x0 + 1, y0);
    const float v01 = at(x0, y0 + 1);
    const float v11 = at(x0 + 1, y0 + 1);
    const float top = v00 + (v10 - v00) * fx;
    const float bottom = v01 + (v11 - v01) * fx;
    return top + (bottom - top) * fy;
}

// Box/area resampling: average the source pixels covered by the destination
// pixel's footprint.
[[nodiscard]] float sample_area(luisa::span<const uint8_t> src, uint32_t width, uint32_t height,
                                uint32_t channel, uint32_t dx, uint32_t dy,
                                uint32_t dst_width, uint32_t dst_height) noexcept {
    const float sx = static_cast<float>(width) / static_cast<float>(dst_width);
    const float sy = static_cast<float>(height) / static_cast<float>(dst_height);
    const float x_begin = static_cast<float>(dx) * sx;
    const float x_end = static_cast<float>(dx + 1) * sx;
    const float y_begin = static_cast<float>(dy) * sy;
    const float y_end = static_cast<float>(dy + 1) * sy;
    const uint32_t x0 = static_cast<uint32_t>(std::floor(x_begin));
    const uint32_t x1 = std::min(width, static_cast<uint32_t>(std::ceil(x_end)));
    const uint32_t y0 = static_cast<uint32_t>(std::floor(y_begin));
    const uint32_t y1 = std::min(height, static_cast<uint32_t>(std::ceil(y_end)));
    double sum = 0.0;
    double weight = 0.0;
    for (uint32_t y = y0; y < y1; y++) {
        const float wy = std::min(y_end, static_cast<float>(y + 1)) - std::max(y_begin, static_cast<float>(y));
        for (uint32_t x = x0; x < x1; x++) {
            const float wx = std::min(x_end, static_cast<float>(x + 1)) - std::max(x_begin, static_cast<float>(x));
            const float w = wx * wy;
            sum += static_cast<double>(src[(static_cast<size_t>(y) * width + x) * 3u + channel]) * w;
            weight += w;
        }
    }
    return weight > 0.0 ? static_cast<float>(sum / weight) : 0.0f;
}

}// namespace

[[nodiscard]] Result<PreprocessedImage> preprocess_image(const DecodedImage &image, const PreprocessSpec &spec) {
    using R = Result<PreprocessedImage>;
    if (image.width == 0 || image.height == 0 || image.pixels.size() != static_cast<size_t>(image.width) * image.height * 3u) {
        return R::failure(StatusCode::invalid_argument, "degenerate source image", "image");
    }
    if (!spec.mean.empty() && spec.mean.size() != 3u) {
        return R::failure(StatusCode::invalid_argument, "mean must have 3 channels", "mean");
    }
    if (!spec.std_dev.empty() && spec.std_dev.size() != 3u) {
        return R::failure(StatusCode::invalid_argument, "std_dev must have 3 channels", "std_dev");
    }
    for (auto s : spec.std_dev) {
        if (!std::isfinite(s) || s == 0.0f) {
            return R::failure(StatusCode::invalid_argument, "std_dev must be finite and nonzero", "std_dev");
        }
    }
    for (auto m : spec.mean) {
        if (!std::isfinite(m)) {
            return R::failure(StatusCode::invalid_argument, "mean must be finite", "mean");
        }
    }

    // Honor the requested channel order: convert the source pixels to
    // spec.order BEFORE resampling/normalization, so output channel c (and
    // mean[c]/std_dev[c]) corresponds to spec.order's channel c. When the
    // source is already in the requested order this is a no-op.
    luisa::vector<uint8_t> ordered_storage;
    luisa::span<const uint8_t> src_pixels{image.pixels.data(), image.pixels.size()};
    if (image.order != spec.order) {
        ordered_storage = image.pixels;
        const uint64_t pixel_count = static_cast<uint64_t>(image.width) * image.height;
        auto cs = convert_rgb_bgr(ordered_storage, pixel_count);
        if (!cs.ok()) { return R::failure(cs); }
        src_pixels = luisa::span<const uint8_t>{ordered_storage.data(), ordered_storage.size()};
    }

    // Resolve the target size.
    uint32_t target_w = spec.target_width;
    uint32_t target_h = spec.target_height;
    if (spec.max_long_edge > 0) {
        const uint32_t long_edge = std::max(image.width, image.height);
        if (long_edge > spec.max_long_edge) {
            const float scale = static_cast<float>(spec.max_long_edge) / static_cast<float>(long_edge);
            target_w = std::max(1u, static_cast<uint32_t>(std::lround(image.width * scale)));
            target_h = std::max(1u, static_cast<uint32_t>(std::lround(image.height * scale)));
        }
    }
    if (target_w == 0) { target_w = image.width; }
    if (target_h == 0) { target_h = image.height; }
    target_w = round_up_multiple(target_w, spec.pad_to_multiple);
    target_h = round_up_multiple(target_h, spec.pad_to_multiple);
    if (target_w == 0 || target_h == 0) {
        return R::failure(StatusCode::invalid_argument, "target dimensions are degenerate", "target");
    }
    const uint64_t target_pixels = static_cast<uint64_t>(target_w) * target_h;
    if (target_pixels > spec.max_pixels) {
        return R::failure(StatusCode::invalid_argument, "target pixel count exceeds the limit", "target");
    }

    PreprocessedImage out;
    out.width = target_w;
    out.height = target_h;
    out.channels = 3;
    out.patch_size = spec.patch_size;
    // Source-coordinate provenance: preprocessed -> original page pixels.
    out.transform.scale_x = static_cast<float>(image.width) / static_cast<float>(target_w);
    out.transform.scale_y = static_cast<float>(image.height) / static_cast<float>(target_h);
    out.transform.offset_x = 0.0f;
    out.transform.offset_y = 0.0f;

    // Resample into HWC float, then normalize and transpose to CHW.
    uint64_t chw_count = 0;
    if (!checked_mul(target_pixels, 3u, chw_count)) {
        return R::failure(StatusCode::invalid_argument, "pixel count overflow", "pixels");
    }
    out.pixels.resize(static_cast<size_t>(chw_count));
    const bool resize = target_w != image.width || target_h != image.height;
    for (uint32_t y = 0; y < target_h; y++) {
        for (uint32_t x = 0; x < target_w; x++) {
            for (uint32_t c = 0; c < 3u; c++) {
                float value = 0.0f;
                if (!resize) {
                    value = static_cast<float>(src_pixels[(static_cast<size_t>(y) * image.width + x) * 3u + c]);
                } else if (spec.resize == ResizeConvention::bilinear) {
                    const float sx = (static_cast<float>(x) + 0.5f) * out.transform.scale_x - 0.5f;
                    const float sy = (static_cast<float>(y) + 0.5f) * out.transform.scale_y - 0.5f;
                    value = sample_bilinear(src_pixels, image.width, image.height, c, sx, sy);
                } else {
                    value = sample_area(src_pixels, image.width, image.height, c, x, y, target_w, target_h);
                }
                // src_pixels is already in spec.order, so channel index c maps
                // directly to mean[c]/std_dev[c].
                float normalized = value / 255.0f;
                if (!spec.mean.empty()) { normalized = (normalized - spec.mean[c]) / spec.std_dev[c]; }
                const auto dst = static_cast<size_t>(c) * target_pixels + static_cast<size_t>(y) * target_w + x;
                out.pixels[dst] = normalized;
            }
        }
    }

    // Patch metadata.
    if (spec.patch_size > 0) {
        if (target_w % spec.patch_size != 0 || target_h % spec.patch_size != 0) {
            return R::failure(StatusCode::invalid_argument,
                              "target dimensions are not a multiple of patch_size", "patch_size");
        }
        out.patches_w = target_w / spec.patch_size;
        out.patches_h = target_h / spec.patch_size;
        if (!checked_mul(static_cast<uint64_t>(out.patches_w) * out.patches_h, 1u, out.patch_count)) {
            return R::failure(StatusCode::invalid_argument, "patch count overflow", "patch_size");
        }
    }
    return R::success(std::move(out));
}

[[nodiscard]] Result<RaggedImageBatch> pack_ragged_batch(
    luisa::span<const DecodedImage> images, const PreprocessSpec &spec,
    uint64_t max_total_patches) {
    using R = Result<RaggedImageBatch>;
    RaggedImageBatch batch;
    batch.patch_offsets.emplace_back(0);
    for (const auto &image : images) {
        auto processed = preprocess_image(image, spec);
        if (!processed.ok()) { return R::failure(processed.status); }
        uint64_t next_patches = 0, next_pixels = 0;
        if (!checked_add(batch.total_patches, processed.value->patch_count, next_patches) ||
            next_patches > max_total_patches) {
            return R::failure(StatusCode::invalid_argument, "ragged batch exceeds the patch budget", "patches");
        }
        if (!checked_add(batch.total_pixels, static_cast<uint64_t>(processed.value->width) * processed.value->height, next_pixels)) {
            return R::failure(StatusCode::invalid_argument, "ragged batch pixel count overflow", "pixels");
        }
        batch.total_patches = next_patches;
        batch.total_pixels = next_pixels;
        batch.patch_offsets.emplace_back(next_patches);
        batch.images.emplace_back(std::move(*processed.value));
    }
    return R::success(std::move(batch));
}

}// namespace luisa::ocr