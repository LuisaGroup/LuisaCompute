// mm_embedding engine -- Video Processor implementation.
#include "mm_embedding/embed/video_processor.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include <stb/stb_image.h>
#include <stb/stb_image_resize2.h>

#include <luisa/core/logging.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mferror.h>

#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mfuuid.lib")
#pragma comment(lib, "ole32.lib")
#endif

namespace luisa::mm_embed {

namespace {

constexpr uint32_t kPatchSize = 16u;
constexpr uint32_t kPoolingKernel = 3u;
constexpr uint32_t kMaxVideoPatches = 1260u; // 140 * 3 * 3
constexpr uint32_t kPatchFloats = kPatchSize * kPatchSize * 3u; // 768

struct DecodedFrameData {
    luisa::vector<uint8_t> rgb_bytes;
    int width{0};
    int height{0};
    double timestamp_sec{0.0};
};

struct DecodedVideoResult {
    luisa::vector<DecodedFrameData> frames;
    double duration_sec{0.0};
    double native_fps{0.0};
};

#ifdef _WIN32
class MFInitializer {
public:
    MFInitializer() {
        HRESULT hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
        _com_ok = SUCCEEDED(hr);
        hr = MFStartup(MF_VERSION);
        _mf_ok = SUCCEEDED(hr);
    }
    ~MFInitializer() {
        if (_mf_ok) {
            MFShutdown();
        }
        if (_com_ok) {
            CoUninitialize();
        }
    }
    [[nodiscard]] bool is_ok() const noexcept { return _com_ok && _mf_ok; }

private:
    bool _com_ok{false};
    bool _mf_ok{false};
};

DecodedVideoResult decode_mp4_media_foundation(const std::filesystem::path &video_path) {
    static MFInitializer s_mf_init;
    if (!s_mf_init.is_ok()) {
        LUISA_ERROR("decode_mp4_media_foundation: Media Foundation initialization failed.");
    }

    IMFAttributes *pAttributes = nullptr;
    HRESULT hr = MFCreateAttributes(&pAttributes, 1);
    if (FAILED(hr)) {
        LUISA_ERROR("decode_mp4_media_foundation: MFCreateAttributes failed: 0x{:08X}", hr);
    }
    pAttributes->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, TRUE);

    IMFSourceReader *pReader = nullptr;
    std::wstring wpath = video_path.wstring();
    hr = MFCreateSourceReaderFromURL(wpath.c_str(), pAttributes, &pReader);
    pAttributes->Release();
    if (FAILED(hr)) {
        LUISA_ERROR("decode_mp4_media_foundation: MFCreateSourceReaderFromURL failed for '{}': 0x{:08X}",
                    video_path.string(), hr);
    }

    // Configure uncompressed RGB32 output format
    IMFMediaType *pMediaType = nullptr;
    hr = MFCreateMediaType(&pMediaType);
    if (FAILED(hr) || !pMediaType) {
        pReader->Release();
        LUISA_ERROR("decode_mp4_media_foundation: MFCreateMediaType failed: 0x{:08X}", hr);
    }
    pMediaType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
    pMediaType->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32);
    hr = pReader->SetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM, NULL, pMediaType);
    pMediaType->Release();
    if (FAILED(hr)) {
        pReader->Release();
        LUISA_ERROR("decode_mp4_media_foundation: SetCurrentMediaType(RGB32) failed: 0x{:08X}", hr);
    }

    IMFMediaType *pCurrentType = nullptr;
    hr = pReader->GetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM, &pCurrentType);
    if (FAILED(hr) || !pCurrentType) {
        pReader->Release();
        LUISA_ERROR("decode_mp4_media_foundation: GetCurrentMediaType failed: 0x{:08X}", hr);
    }

    UINT32 width = 0, height = 0;
    MFGetAttributeSize(pCurrentType, MF_MT_FRAME_SIZE, &width, &height);
    if (width == 0 || height == 0) {
        pCurrentType->Release();
        pReader->Release();
        LUISA_ERROR("decode_mp4_media_foundation: invalid frame dimensions (0x0) in '{}'", video_path.string());
    }

    UINT32 fps_num = 0, fps_den = 1;
    MFGetAttributeRatio(pCurrentType, MF_MT_FRAME_RATE, &fps_num, &fps_den);
    double native_fps = (fps_den > 0) ? (static_cast<double>(fps_num) / fps_den) : 0.0;

    INT32 stride = 0;
    if (FAILED(pCurrentType->GetUINT32(MF_MT_DEFAULT_STRIDE, reinterpret_cast<UINT32 *>(&stride)))) {
        stride = static_cast<INT32>(width * 4u);
    }
    pCurrentType->Release();

    PROPVARIANT var;
    PropVariantInit(&var);
    double duration_sec = 0.0;
    if (SUCCEEDED(pReader->GetPresentationAttribute(MF_SOURCE_READER_MEDIASOURCE, MF_PD_DURATION, &var))) {
        LONGLONG hnsDuration = var.hVal.QuadPart;
        duration_sec = static_cast<double>(hnsDuration) / 10000000.0;
        PropVariantClear(&var);
    }

    DecodedVideoResult result{};
    result.native_fps = native_fps;
    result.duration_sec = duration_sec;

    int stride_abs = std::abs(stride);
    size_t frame_bytes = static_cast<size_t>(width) * height * 3u;

    while (true) {
        DWORD streamIndex = 0, flags = 0;
        LONGLONG llTimeStamp = 0;
        IMFSample *pSample = nullptr;

        hr = pReader->ReadSample(MF_SOURCE_READER_FIRST_VIDEO_STREAM, 0, &streamIndex, &flags, &llTimeStamp, &pSample);
        if (FAILED(hr) || (flags & MF_SOURCE_READERF_ENDOFSTREAM)) {
            if (pSample) {
                pSample->Release();
            }
            break;
        }

        if (pSample) {
            IMFMediaBuffer *pBuffer = nullptr;
            hr = pSample->ConvertToContiguousBuffer(&pBuffer);
            if (SUCCEEDED(hr) && pBuffer) {
                BYTE *pData = nullptr;
                DWORD cbLen = 0;
                if (SUCCEEDED(pBuffer->Lock(&pData, NULL, &cbLen))) {
                    if (cbLen >= static_cast<DWORD>(height * stride_abs)) {
                        DecodedFrameData f{};
                        f.width = static_cast<int>(width);
                        f.height = static_cast<int>(height);
                        f.timestamp_sec = static_cast<double>(llTimeStamp) / 10000000.0;
                        f.rgb_bytes.resize(frame_bytes);

                        // Convert RGB32 (BGRX) to RGB24, handling top-down vs bottom-up stride
                        for (UINT32 y = 0; y < height; y++) {
                            const BYTE *src_row = (stride > 0) ? (pData + y * stride_abs)
                                                               : (pData + (height - 1u - y) * stride_abs);
                            uint8_t *dst_row = f.rgb_bytes.data() + y * width * 3u;
                            for (UINT32 x = 0; x < width; x++) {
                                dst_row[x * 3u + 0u] = src_row[x * 4u + 2u]; // R
                                dst_row[x * 3u + 1u] = src_row[x * 4u + 1u]; // G
                                dst_row[x * 3u + 2u] = src_row[x * 4u + 0u]; // B
                            }
                        }
                        result.frames.push_back(std::move(f));
                    }
                    pBuffer->Unlock();
                }
                pBuffer->Release();
            }
            pSample->Release();
        }
    }

    pReader->Release();

    if (result.duration_sec <= 0.0 && native_fps > 0.0 && !result.frames.empty()) {
        result.duration_sec = static_cast<double>(result.frames.size()) / native_fps;
    }

    return result;
}
#endif

ProcessedVideoFrame process_single_frame(
    const uint8_t *rgb_bytes, int width, int height, int channels,
    uint32_t target_w, uint32_t target_h,
    uint32_t num_patches_w, uint32_t num_patches_h,
    uint32_t num_patches, uint32_t num_soft_tokens) {

    luisa::vector<uint8_t> resized(target_w * target_h * 3u);
    void *res = stbir_resize(
        rgb_bytes, width, height, width * channels,
        resized.data(), target_w, target_h, target_w * 3,
        STBIR_RGB, STBIR_TYPE_UINT8, STBIR_EDGE_CLAMP, STBIR_FILTER_CATMULLROM);
    if (!res) {
        LUISA_ERROR("VideoProcessor: stbir_resize failed");
    }

    ProcessedVideoFrame f{};
    f.target_width = target_w;
    f.target_height = target_h;
    f.num_patches_w = num_patches_w;
    f.num_patches_h = num_patches_h;
    f.num_patches = num_patches;
    f.num_soft_tokens = num_soft_tokens;
    f.max_patches = kMaxVideoPatches;

    f.pixel_values.assign(kMaxVideoPatches * kPatchFloats, 0.0f);
    f.position_ids.assign(kMaxVideoPatches * 2u, -1);

    constexpr float kInv255 = 1.0f / 255.0f;
    for (uint32_t yp = 0; yp < num_patches_h; yp++) {
        for (uint32_t xp = 0; xp < num_patches_w; xp++) {
            uint32_t p_idx = yp * num_patches_w + xp;
            f.position_ids[p_idx * 2u + 0u] = static_cast<int32_t>(xp);
            f.position_ids[p_idx * 2u + 1u] = static_cast<int32_t>(yp);

            float *dst_patch = &f.pixel_values[p_idx * kPatchFloats];
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

    return f;
}

} // namespace

std::pair<uint32_t, uint32_t> VideoProcessor::get_target_size(
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
        target_width = (target_width > side_mult) ? (target_width - side_mult) : target_width;
    }

    return {target_width, target_height};
}

luisa::vector<uint32_t> VideoProcessor::sample_frame_indices(
    uint32_t total_frames,
    double duration_sec,
    double native_fps,
    const VideoConfig &config) {

    if (total_frames == 0) {
        return {};
    }

    luisa::vector<uint32_t> indices;

    // 1. Rate-based sampling matching EmbeddingGemma2VideoProcessor
    if (config.fps > 0.0f && native_fps > 0.0 && duration_sec > 0.0) {
        double step = native_fps / config.fps;
        int num_sampled = std::max(1, static_cast<int>(std::floor(duration_sec * config.fps)));
        indices.reserve(num_sampled);
        for (int i = 0; i < num_sampled; i++) {
            uint32_t idx = std::min(total_frames - 1u, static_cast<uint32_t>(std::floor(i * step)));
            indices.push_back(idx);
        }
    } else {
        indices.resize(total_frames);
        for (uint32_t i = 0; i < total_frames; i++) {
            indices[i] = i;
        }
    }

    // 2. Budget capping according to overflow_strategy
    if (indices.size() > config.max_frames && config.max_frames > 0) {
        if (config.overflow_strategy == VideoOverflowStrategy::Truncate) {
            indices.resize(config.max_frames);
        } else {
            // Uniform resampling matching np.linspace(0, len(indices)-1, max_frames, dtype=int)
            luisa::vector<uint32_t> sampled;
            sampled.reserve(config.max_frames);
            if (config.max_frames == 1u) {
                sampled.push_back(indices[0]);
            } else {
                double step = static_cast<double>(indices.size() - 1u) / (config.max_frames - 1u);
                for (uint32_t k = 0; k < config.max_frames; k++) {
                    uint32_t lin_i = (k == config.max_frames - 1u)
                                         ? static_cast<uint32_t>(indices.size() - 1u)
                                         : std::min(static_cast<uint32_t>(indices.size() - 1u), static_cast<uint32_t>(std::floor(k * step)));
                    sampled.push_back(indices[lin_i]);
                }
            }
            indices = std::move(sampled);
        }
    }

    return indices;
}

ProcessedVideo VideoProcessor::process_memory(
    luisa::span<const RawVideoFrame> frames,
    const VideoConfig &config) {

    if (frames.empty()) {
        LUISA_ERROR("VideoProcessor::process_memory: empty frame list");
    }

    int base_w = frames[0].width;
    int base_h = frames[0].height;
    if (base_w <= 0 || base_h <= 0) {
        LUISA_ERROR("VideoProcessor::process_memory: invalid frame resolution ({}x{})", base_w, base_h);
    }

    auto [target_w, target_h] = get_target_size(
        static_cast<uint32_t>(base_w), static_cast<uint32_t>(base_h),
        kPatchSize, kMaxVideoPatches, kPoolingKernel);

    uint32_t num_patches_w = target_w / kPatchSize;
    uint32_t num_patches_h = target_h / kPatchSize;
    uint32_t num_patches = num_patches_w * num_patches_h;
    uint32_t num_soft_tokens = num_patches / (kPoolingKernel * kPoolingKernel);

    double native_fps = 0.0;
    double duration_sec = 0.0;
    if (frames.size() > 1 && frames.back().timestamp_sec > frames.front().timestamp_sec) {
        native_fps = static_cast<double>(frames.size() - 1) / (frames.back().timestamp_sec - frames.front().timestamp_sec);
        duration_sec = static_cast<double>(frames.size()) / native_fps;
    }

    auto sampled_indices = sample_frame_indices(
        static_cast<uint32_t>(frames.size()), duration_sec, native_fps, config);

    ProcessedVideo result{};
    result.num_frames = static_cast<uint32_t>(sampled_indices.size());
    result.num_soft_tokens_per_frame = num_soft_tokens;
    result.total_soft_tokens = result.num_frames * num_soft_tokens;
    result.target_width = target_w;
    result.target_height = target_h;
    result.num_patches_w = num_patches_w;
    result.num_patches_h = num_patches_h;
    result.num_patches = num_patches;
    result.max_patches = kMaxVideoPatches;
    result.frames.reserve(result.num_frames);

    for (uint32_t idx : sampled_indices) {
        const auto &raw_frame = frames[idx];
        result.frames.push_back(process_single_frame(
            raw_frame.rgb_bytes, raw_frame.width, raw_frame.height, raw_frame.channels,
            target_w, target_h, num_patches_w, num_patches_h, num_patches, num_soft_tokens));
    }

    return result;
}

ProcessedVideo VideoProcessor::process_frames(
    luisa::span<const std::filesystem::path> frame_paths,
    const VideoConfig &config) {

    if (frame_paths.empty()) {
        LUISA_ERROR("VideoProcessor::process_frames: empty frame path list");
    }

    auto sampled_indices = sample_frame_indices(
        static_cast<uint32_t>(frame_paths.size()), 0.0, 0.0, config);

    // Read first sampled frame to determine resolution and patch layout
    int w = 0, h = 0, channels = 0;
    stbi_uc *first_data = stbi_load(frame_paths[sampled_indices[0]].string().c_str(), &w, &h, &channels, 3);
    if (!first_data) {
        LUISA_ERROR("VideoProcessor::process_frames: failed to load '{}' ({})",
                    frame_paths[sampled_indices[0]].string(), stbi_failure_reason());
    }

    auto [target_w, target_h] = get_target_size(
        static_cast<uint32_t>(w), static_cast<uint32_t>(h),
        kPatchSize, kMaxVideoPatches, kPoolingKernel);

    uint32_t num_patches_w = target_w / kPatchSize;
    uint32_t num_patches_h = target_h / kPatchSize;
    uint32_t num_patches = num_patches_w * num_patches_h;
    uint32_t num_soft_tokens = num_patches / (kPoolingKernel * kPoolingKernel);

    ProcessedVideo result{};
    result.num_frames = static_cast<uint32_t>(sampled_indices.size());
    result.num_soft_tokens_per_frame = num_soft_tokens;
    result.total_soft_tokens = result.num_frames * num_soft_tokens;
    result.target_width = target_w;
    result.target_height = target_h;
    result.num_patches_w = num_patches_w;
    result.num_patches_h = num_patches_h;
    result.num_patches = num_patches;
    result.max_patches = kMaxVideoPatches;
    result.frames.reserve(result.num_frames);

    result.frames.push_back(process_single_frame(
        first_data, w, h, 3,
        target_w, target_h, num_patches_w, num_patches_h, num_patches, num_soft_tokens));
    stbi_image_free(first_data);

    for (size_t i = 1; i < sampled_indices.size(); i++) {
        uint32_t idx = sampled_indices[i];
        int fw = 0, fh = 0, fc = 0;
        stbi_uc *fdata = stbi_load(frame_paths[idx].string().c_str(), &fw, &fh, &fc, 3);
        if (!fdata) {
            LUISA_ERROR("VideoProcessor::process_frames: failed to load '{}'", frame_paths[idx].string());
        }
        result.frames.push_back(process_single_frame(
            fdata, fw, fh, 3,
            target_w, target_h, num_patches_w, num_patches_h, num_patches, num_soft_tokens));
        stbi_image_free(fdata);
    }

    return result;
}

ProcessedVideo VideoProcessor::process_file(
    const std::filesystem::path &video_path,
    const VideoConfig &config) {

    if (!std::filesystem::exists(video_path)) {
        LUISA_ERROR("VideoProcessor::process_file: path does not exist '{}'", video_path.string());
    }

    // If video_path is a directory of frames, collect and process
    if (std::filesystem::is_directory(video_path)) {
        luisa::vector<std::filesystem::path> frame_paths;
        for (const auto &entry : std::filesystem::directory_iterator(video_path)) {
            if (entry.is_regular_file()) {
                auto ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".webp") {
                    frame_paths.push_back(entry.path());
                }
            }
        }
        std::sort(frame_paths.begin(), frame_paths.end());
        if (frame_paths.empty()) {
            LUISA_ERROR("VideoProcessor::process_file: no image frames found in directory '{}'", video_path.string());
        }
        return process_frames(frame_paths, config);
    }

    // Check for MP4 or video file
    auto ext = video_path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

#ifdef _WIN32
    if (ext == ".mp4" || ext == ".m4v" || ext == ".mov" || ext == ".wmv" || ext == ".avi") {
        auto decoded = decode_mp4_media_foundation(video_path);
        if (decoded.frames.empty()) {
            LUISA_ERROR("VideoProcessor::process_file: decoded 0 frames from '{}'", video_path.string());
        }

        luisa::vector<RawVideoFrame> raw_frames;
        raw_frames.reserve(decoded.frames.size());
        for (const auto &f : decoded.frames) {
            RawVideoFrame r{};
            r.rgb_bytes = f.rgb_bytes.data();
            r.width = f.width;
            r.height = f.height;
            r.channels = 3;
            r.timestamp_sec = f.timestamp_sec;
            raw_frames.push_back(r);
        }

        auto sampled_indices = sample_frame_indices(
            static_cast<uint32_t>(raw_frames.size()), decoded.duration_sec, decoded.native_fps, config);

        int base_w = decoded.frames[0].width;
        int base_h = decoded.frames[0].height;
        auto [target_w, target_h] = get_target_size(
            static_cast<uint32_t>(base_w), static_cast<uint32_t>(base_h),
            kPatchSize, kMaxVideoPatches, kPoolingKernel);

        uint32_t num_patches_w = target_w / kPatchSize;
        uint32_t num_patches_h = target_h / kPatchSize;
        uint32_t num_patches = num_patches_w * num_patches_h;
        uint32_t num_soft_tokens = num_patches / (kPoolingKernel * kPoolingKernel);

        ProcessedVideo result{};
        result.num_frames = static_cast<uint32_t>(sampled_indices.size());
        result.num_soft_tokens_per_frame = num_soft_tokens;
        result.total_soft_tokens = result.num_frames * num_soft_tokens;
        result.target_width = target_w;
        result.target_height = target_h;
        result.num_patches_w = num_patches_w;
        result.num_patches_h = num_patches_h;
        result.num_patches = num_patches;
        result.max_patches = kMaxVideoPatches;
        result.frames.reserve(result.num_frames);

        for (uint32_t idx : sampled_indices) {
            const auto &f = decoded.frames[idx];
            result.frames.push_back(process_single_frame(
                f.rgb_bytes.data(), f.width, f.height, 3,
                target_w, target_h, num_patches_w, num_patches_h, num_patches, num_soft_tokens));
        }
        return result;
    }
#endif

    // Fallback if not Windows or unsupported format: check if directory with same stem exists
    auto fallback_dir = video_path.parent_path() / video_path.stem();
    if (std::filesystem::exists(fallback_dir) && std::filesystem::is_directory(fallback_dir)) {
        return process_file(fallback_dir, config);
    }

    LUISA_ERROR("VideoProcessor::process_file: unsupported video format '{}'", video_path.string());
}

}// namespace luisa::mm_embed
