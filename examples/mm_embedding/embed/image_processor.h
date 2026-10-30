#pragma once
// mm_embedding engine -- C++ Image Processor matching Gemma4ImageProcessor.
//
// Operations:
// 1. Aspect-ratio-preserving resize to multiples of 48 pixels (16 * 3).
// 2. Rescale pixel values [0, 255] -> [0.0, 1.0].
// 3. Patchify into [num_patches, 16 * 16 * 3 = 768] tensors.
// 4. Generate 2D patch position IDs (x, y).
// 5. Pad to max_patches (2520) with 0.0 floats and -1 position IDs.

#include <cstdint>
#include <filesystem>

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/vector.h>

namespace luisa::mm_embed {

struct ProcessedImage {
    luisa::vector<float> pixel_values;   // [max_patches * 768] in [0, 1]
    luisa::vector<int32_t> position_ids; // [max_patches * 2] (x, y)
    uint32_t target_width{768u};
    uint32_t target_height{768u};
    uint32_t num_patches_w{48u};
    uint32_t num_patches_h{48u};
    uint32_t num_patches{2304u};
    uint32_t num_soft_tokens{256u};
    uint32_t max_patches{2520u};
};

class ImageProcessor {
public:
    static ProcessedImage process_file(const std::filesystem::path &image_path);
    static ProcessedImage process_memory(const uint8_t *rgb_bytes, int width, int height, int channels = 3);

    // Compute target dimensions preserving aspect ratio according to Gemma 4 constraints
    static std::pair<uint32_t, uint32_t> get_target_size(
        uint32_t width, uint32_t height,
        uint32_t patch_size = 16u,
        uint32_t max_patches = 2520u,
        uint32_t pooling_kernel_size = 3u);
};

}// namespace luisa::mm_embed
