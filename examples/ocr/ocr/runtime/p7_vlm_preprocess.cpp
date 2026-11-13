// OCR engine -- Phase 7 host preprocessing bridge. Host-only TU.
// The ONLY place the Phase 7 graph touches the Phase 1 preprocess reference;
// keeps api.h/model_recipe.h out of every device TU. The executor .cpp
// forward-declares p7_preprocess_image_host over the plain P7ImageInput type.
#include <utility>

#include "ocr/preprocess.h"
#include "ocr/runtime/p7_slice_config.h"

namespace luisa::ocr {

[[nodiscard]] Result<luisa::vector<float>> p7_preprocess_image_host(
    const P7ImageInput &image, luisa::span<const float> mean, luisa::span<const float> std_dev) {
    using R = Result<luisa::vector<float>>;
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
    PreprocessSpec spec;
    spec.resize = ResizeConvention::bilinear;
    spec.target_width = image.packed.resized_width;
    spec.target_height = image.packed.resized_height;
    spec.pad_to_multiple = 1;
    spec.order = PixelOrder::rgb;
    spec.mean.assign(mean.begin(), mean.end());
    spec.std_dev.assign(std_dev.begin(), std_dev.end());
    spec.patch_size = 0;
    spec.max_pixels = static_cast<uint64_t>(image.packed.resized_width) * image.packed.resized_height;
    auto pre = preprocess_image(decoded, spec);
    if (!pre.ok()) { return R::failure(pre.status); }
    if (pre.value->width != image.packed.resized_width || pre.value->height != image.packed.resized_height) {
        return R::failure(StatusCode::internal_error, "preprocessed size does not match the packing contract", "preprocess");
    }
    return R::success(std::move(pre.value->pixels));
}

}// namespace luisa::ocr
