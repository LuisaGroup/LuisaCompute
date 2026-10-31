// embedding_mm engine -- Image Processor implementation.
#include "embedding_mm/embed/image_processor.h"

#include <cmath>
#include <cstdio>
#include <cstring>

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#define STB_IMAGE_RESIZE_STATIC
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb/stb_image_resize2.h>

#include <luisa/core/logging.h>

namespace luisa::embed_mm {

std::pair<uint32_t, uint32_t> ImageProcessor::get_target_size(
    uint32_t width, uint32_t height,
    uint32_t patch_size,
    uint32_t max_patches,
    uint32_t pooling_kernel_size) {

    double total_px = static_cast<double>(width) * height;
    double target_px = static_cast<double>(max_patches) * (patch_size * patch_size);
    double factor = std::sqrt(target_px / total_px);
    double ideal_height = factor * height;
    double ideal_width = factor * width;
    uint32_t side_mult = pooling_kernel_size * patch_size; // 48

    uint32_t target_height = static_cast<uint32_t>(std::floor(ideal_height / side_mult)) * side_mult;
    uint32_t target_width = static_cast<uint32_t>(std::floor(ideal_width / side_mult)) * side_mult;

    uint32_t max_side_length = (max_patches / (pooling_kernel_size * pooling_kernel_size)) * side_mult;
    if (target_height == 0) {
        target_height = side_mult;
        target_width = std::min(static_cast<uint32_t>(std::floor(static_cast<double>(width) / height)) * side_mult, max_side_length);
    } else if (target_width == 0) {
        target_width = side_mult;
        target_height = std::min(static_cast<uint32_t>(std::floor(static_cast<double>(height) / width)) * side_mult, max_side_length);
    }

    if (static_cast<double>(target_width) * target_height > target_px) {
        // Fallback safety to never exceed patch budget
        target_width = (target_width > side_mult) ? (target_width - side_mult) : target_width;
    }

    return {target_width, target_height};
}

ProcessedImage ImageProcessor::process_memory(const uint8_t *rgb_bytes, int width, int height, int channels) {
    if (!rgb_bytes || width <= 0 || height <= 0) {
        LUISA_ERROR("ImageProcessor::process_memory: invalid image buffer (w={}, h={})", width, height);
    }

    constexpr uint32_t kPatchSize = 16u;
    constexpr uint32_t kPoolingKernel = 3u;
    constexpr uint32_t kMaxPatches = 2520u;
    constexpr uint32_t kPatchFloats = kPatchSize * kPatchSize * 3u; // 768

    auto [target_w, target_h] = get_target_size(static_cast<uint32_t>(width), static_cast<uint32_t>(height),
                                                kPatchSize, kMaxPatches, kPoolingKernel);

    // Resize image to target dimensions (RGB 3 channels) using Catmull-Rom bicubic interpolation (matching Gemma4ImageProcessor / torchvision bicubic)
    luisa::vector<uint8_t> resized(target_w * target_h * 3u);
    void *res = stbir_resize(
        rgb_bytes, width, height, width * channels,
        resized.data(), target_w, target_h, target_w * 3,
        STBIR_RGB, STBIR_TYPE_UINT8, STBIR_EDGE_CLAMP, STBIR_FILTER_CATMULLROM);
    if (!res) {
        LUISA_ERROR("ImageProcessor::process_memory: stbir_resize failed");
    }

    uint32_t num_patches_w = target_w / kPatchSize;
    uint32_t num_patches_h = target_h / kPatchSize;
    uint32_t num_patches = num_patches_w * num_patches_h;
    uint32_t num_soft_tokens = num_patches / (kPoolingKernel * kPoolingKernel);

    ProcessedImage img{};
    img.target_width = target_w;
    img.target_height = target_h;
    img.num_patches_w = num_patches_w;
    img.num_patches_h = num_patches_h;
    img.num_patches = num_patches;
    img.num_soft_tokens = num_soft_tokens;
    img.max_patches = kMaxPatches;

    img.pixel_values.assign(kMaxPatches * kPatchFloats, 0.0f);
    img.position_ids.assign(kMaxPatches * 2u, -1);

    // Patchify in row-major order: yp in [0, num_patches_h), xp in [0, num_patches_w)
    // Within patch: dy in [0, 16), dx in [0, 16), c in [0, 3)
    constexpr float kInv255 = 1.0f / 255.0f;
    for (uint32_t yp = 0; yp < num_patches_h; yp++) {
        for (uint32_t xp = 0; xp < num_patches_w; xp++) {
            uint32_t p_idx = yp * num_patches_w + xp;
            img.position_ids[p_idx * 2u + 0u] = static_cast<int32_t>(xp);
            img.position_ids[p_idx * 2u + 1u] = static_cast<int32_t>(yp);

            float *dst_patch = &img.pixel_values[p_idx * kPatchFloats];
            uint32_t dst_i = 0;
            for (uint32_t dy = 0; dy < kPatchSize; dy++) {
                for (uint32_t dx = 0; dx < kPatchSize; dx++) {
                    uint32_t Y = yp * kPatchSize + dy;
                    uint32_t X = xp * kPatchSize + dx;
                    uint32_t src_idx = (Y * target_w + X) * 3u;
                    dst_patch[dst_i + 0] = static_cast<float>(resized[src_idx + 0]) * kInv255;
                    dst_patch[dst_i + 1] = static_cast<float>(resized[src_idx + 1]) * kInv255;
                    dst_patch[dst_i + 2] = static_cast<float>(resized[src_idx + 2]) * kInv255;
                    dst_i += 3u;
                }
            }
        }
    }

    return img;
}

ProcessedImage ImageProcessor::process_file(const std::filesystem::path &image_path) {
    int w = 0, h = 0, channels = 0;
    stbi_uc *data = stbi_load(image_path.string().c_str(), &w, &h, &channels, 3);
    if (!data) {
        LUISA_ERROR("ImageProcessor::process_file: failed to load '{}' ({})",
                    image_path.string(), stbi_failure_reason());
    }
    auto result = process_memory(data, w, h, 3);
    stbi_image_free(data);
    return result;
}

}// namespace luisa::embed_mm
