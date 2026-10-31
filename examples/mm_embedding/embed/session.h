#pragma once
// mm_embedding engine -- public Session API.
//
// Composes Tokenizer + Package loader + Device WeightStore + Executor into
// a clean, unified embedding generation session.
//
// Supports:
// 1. Text embedding (encode / encode_batch)
// 2. Image embedding (encode_image / encode_image_memory)
// 3. Multimodal image+text embedding (encode_multimodal)
// 4. Selective Encoder Loading (enable_vision config)

#include <cstdint>
#include <filesystem>

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

namespace luisa::mm_embed {

class Package;
class Tokenizer;
class Executor;

struct SessionConfig {
    bool enable_vision{true};// Selective Encoder Loading (toggle vision ~170M params)
    bool enable_audio{true}; // Selective Encoder Loading (toggle audio ~300M params)
};

struct EncodeOptions {
    uint32_t max_length{512u};
    uint32_t target_dimension{768u};// 768 (native), or MRL: 512, 256, 128
};

struct Embedding {
    luisa::vector<float> values;
    uint32_t dimension{768u};
};

class Session {
public:
    static luisa::unique_ptr<Session> create(
        compute::Device &device,
        const std::filesystem::path &egm_package,
        const std::filesystem::path &tokenizer_json,
        const SessionConfig &config = {});

    ~Session() noexcept;

    Session(const Session &) = delete;
    Session(Session &&) = delete;
    Session &operator=(const Session &) = delete;
    Session &operator=(Session &&) = delete;

    [[nodiscard]] bool has_vision() const noexcept;
    [[nodiscard]] bool has_audio() const noexcept;

    // Text encoding
    [[nodiscard]] Embedding encode(
        luisa::string_view text,
        const EncodeOptions &opt = {});

    [[nodiscard]] luisa::vector<Embedding> encode_batch(
        luisa::span<const luisa::string_view> texts,
        const EncodeOptions &opt = {});

    // Image encoding
    [[nodiscard]] Embedding encode_image(
        const std::filesystem::path &image_path,
        const EncodeOptions &opt = {});

    [[nodiscard]] Embedding encode_image_memory(
        const uint8_t *rgb_bytes,
        int width,
        int height,
        int channels = 3,
        const EncodeOptions &opt = {});

    // Multimodal encoding (image + text)
    [[nodiscard]] Embedding encode_multimodal(
        const std::filesystem::path &image_path,
        luisa::string_view prompt_text,
        const EncodeOptions &opt = {});

    // Audio encoding
    [[nodiscard]] Embedding encode_audio(
        const std::filesystem::path &audio_path,
        const EncodeOptions &opt = {});

    [[nodiscard]] Embedding encode_audio_memory(
        const float *samples,
        size_t count,
        int sample_rate = 16000,
        const EncodeOptions &opt = {});

    // Multimodal audio encoding (audio + text)
    [[nodiscard]] Embedding encode_multimodal_audio(
        const std::filesystem::path &audio_path,
        luisa::string_view prompt_text,
        const EncodeOptions &opt = {});

    [[nodiscard]] const Tokenizer &tokenizer() const noexcept;
    [[nodiscard]] const Package &package() const noexcept;

    Session() noexcept;

private:
    compute::Device *_device{};
    compute::Stream _stream;
    luisa::unique_ptr<Package> _package;
    luisa::unique_ptr<Tokenizer> _tokenizer;
    luisa::unique_ptr<Executor> _executor;
    bool _has_vision{false};
    bool _has_audio{false};

    // Persistent workspace buffers to eliminate per-query driver allocation overhead
    compute::Buffer<uint32_t> _dev_ids;
    compute::Buffer<float> _dev_mask;
    compute::Buffer<float> _dev_out;
    luisa::vector<uint32_t> _host_ids;
    luisa::vector<float> _host_mask;
    luisa::vector<float> _host_out;

    // Persistent vision buffers
    compute::Buffer<float> _dev_pixel_values;
    compute::Buffer<int32_t> _dev_position_ids;
    compute::Buffer<float> _dev_soft_tokens;

    // Persistent audio buffers
    compute::Buffer<float> _dev_audio_features;
    compute::Buffer<float> _dev_audio_soft_tokens;
};

}// namespace luisa::mm_embed
