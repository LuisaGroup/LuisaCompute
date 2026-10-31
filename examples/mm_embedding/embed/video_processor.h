#pragma once
// mm_embedding engine -- C++ Video Processor matching EmbeddingGemma2VideoProcessor.
//
// Operations:
// 1. Video frame decoding (MP4 / H.264 via Media Foundation or image frame sequences via stb_image).
// 2. Temporal sampling (fps-based uniform sampling + overflow strategy).
// 3. Aspect-ratio-preserving resize to multiples of 48 pixels (16 * 3) fitting max_soft_tokens (140) budget.
// 4. Rescale pixel values [0, 255] -> [0.0, 1.0] using Catmull-Rom bicubic interpolation.
// 5. Patchify into [num_frames, num_patches, 16 * 16 * 3 = 768] tensors.
// 6. Generate 2D patch position IDs (x, y) padded to max_patches (1260) with -1.
// 7. Extract soft tokens count per frame (num_soft_tokens <= 140).

#include <cstdint>
#include <filesystem>
#include <utility>

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

namespace luisa::mm_embed {

enum class VideoOverflowStrategy : uint32_t {
    Uniform,
    Truncate
};

struct VideoConfig {
    float fps{1.0f};                                                        // Target sample FPS (1.0 default)
    uint32_t max_frames{32u};                                               // Max frames budget (32 default)
    VideoOverflowStrategy overflow_strategy{VideoOverflowStrategy::Uniform};// Overflow reduction strategy
};

struct RawVideoFrame {
    const uint8_t *rgb_bytes{nullptr};
    int width{0};
    int height{0};
    int channels{3};
    double timestamp_sec{0.0};
};

struct ProcessedVideoFrame {
    luisa::vector<float> pixel_values;  // [max_patches * 768] in [0, 1]
    luisa::vector<int32_t> position_ids;// [max_patches * 2] (x, y)
    uint32_t target_width{0u};
    uint32_t target_height{0u};
    uint32_t num_patches_w{0u};
    uint32_t num_patches_h{0u};
    uint32_t num_patches{0u};
    uint32_t num_soft_tokens{0u};
    uint32_t max_patches{1260u};
};

struct ProcessedVideo {
    luisa::vector<ProcessedVideoFrame> frames;
    uint32_t num_frames{0u};
    uint32_t num_soft_tokens_per_frame{0u};
    uint32_t total_soft_tokens{0u};
    uint32_t target_width{0u};
    uint32_t target_height{0u};
    uint32_t num_patches_w{0u};
    uint32_t num_patches_h{0u};
    uint32_t num_patches{0u};
    uint32_t max_patches{1260u};
};

class VideoProcessor {
public:
    // Process a standalone video file (.mp4) or directory of frames
    static ProcessedVideo process_file(
        const std::filesystem::path &video_path,
        const VideoConfig &config = {});

    // Process a sequence of image file paths
    static ProcessedVideo process_frames(
        luisa::span<const std::filesystem::path> frame_paths,
        const VideoConfig &config = {});

    // Process a sequence of in-memory raw RGB frame buffers
    static ProcessedVideo process_memory(
        luisa::span<const RawVideoFrame> frames,
        const VideoConfig &config = {});

    // Temporal frame sampling logic matching EmbeddingGemma2VideoProcessor
    static luisa::vector<uint32_t> sample_frame_indices(
        uint32_t total_frames,
        double duration_sec,
        double native_fps,
        const VideoConfig &config);

    // Compute target dimensions preserving aspect ratio for video budget (max_soft_tokens=140, max_patches=1260)
    static std::pair<uint32_t, uint32_t> get_target_size(
        uint32_t width, uint32_t height,
        uint32_t patch_size = 16u,
        uint32_t max_patches = 1260u,
        uint32_t pooling_kernel_size = 3u);
};

}// namespace luisa::mm_embed
