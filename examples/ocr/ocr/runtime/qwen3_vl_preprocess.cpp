// OCR engine -- Qwen3-VL host preprocessing bridge (Phase 4). Host-only TU.
// The ONLY place the Qwen3-VL graph touches the Phase 1 preprocess reference;
// keeps api.h/model_recipe.h out of every device TU. See the header.
#include "ocr/runtime/qwen3_vl_preprocess.h"

#include <utility>

#include "ocr/preprocess.h"

namespace luisa::ocr {

namespace {

[[nodiscard]] PreprocessSpec qwen3_vl_preprocess_spec(const Qwen3VlPackedImage &packed,
                                                      luisa::span<const float> mean,
                                                      luisa::span<const float> std_dev) {
    PreprocessSpec spec;
    spec.resize = ResizeConvention::bilinear;
    spec.target_width = packed.resized_width;
    spec.target_height = packed.resized_height;
    spec.pad_to_multiple = 1;
    spec.order = PixelOrder::rgb;
    spec.mean.assign(mean.begin(), mean.end());
    spec.std_dev.assign(std_dev.begin(), std_dev.end());
    spec.patch_size = 0;// patch metadata comes from the packing contract
    spec.max_pixels = static_cast<uint64_t>(packed.resized_width) * packed.resized_height;
    return spec;
}

}// namespace

[[nodiscard]] Result<Qwen3VlPreprocessedImage> qwen3_vl_preprocess_image_host(
    const Qwen3VlImageInput &image, luisa::span<const float> mean, luisa::span<const float> std_dev) {
    using R = Result<Qwen3VlPreprocessedImage>;
    if (image.pixels.size() != static_cast<size_t>(image.source_width) * image.source_height * 3u) {
        return R::failure(StatusCode::invalid_argument, "image pixel buffer size mismatch", "pixels");
    }
    DecodedImage decoded;
    decoded.width = image.source_width;
    decoded.height = image.source_height;
    decoded.channels = 3;
    decoded.row_stride = static_cast<uint64_t>(image.source_width) * 3u;
    decoded.order = PixelOrder::rgb;
    decoded.pixels = image.pixels;
    auto spec = qwen3_vl_preprocess_spec(image.packed, mean, std_dev);
    auto pre = preprocess_image(decoded, spec);
    if (!pre.ok()) { return R::failure(pre.status); }
    Qwen3VlPreprocessedImage out;
    out.width = pre.value->width;
    out.height = pre.value->height;
    out.pixels = std::move(pre.value->pixels);
    return R::success(std::move(out));
}

}// namespace luisa::ocr
