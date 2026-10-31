#pragma once
// embedding_mm engine -- public Session API.
//
// Composes Tokenizer + Package loader + FeatureLoader + Executor into
// a clean, unified multimodal embedding generation session matching
// Google's SentenceTransformers API experience.
//
// Supports:
// 1. Text embedding (encode / encode_batch / encode_query / encode_document)
// 2. Image embedding (encode_image from .mmfeat tensor bundle or sidecar)
// 3. Audio embedding (encode_audio from .mmfeat tensor bundle or sidecar)
// 4. Video embedding (encode_video from .mmfeat tensor bundle or sidecar)
// 5. Interleaved Multimodal embedding (encode_multimodal with <|image|>, <|audio|>, <|video|>)
// 6. Cosine similarity & matrix computation (similarity / similarity_matrix)
// 7. Selective Encoder Loading (enable_vision / enable_audio config)

#include <cstdint>
#include <filesystem>

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include "embedding_mm/embed/feature_loader.h"

namespace luisa::embed_mm {

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

    // 1. Text encoding (general / symmetric)
    [[nodiscard]] Embedding encode(
        luisa::string_view text,
        const EncodeOptions &opt = {});

    [[nodiscard]] luisa::vector<Embedding> encode_batch(
        luisa::span<const luisa::string_view> texts,
        const EncodeOptions &opt = {});

    // Asymmetric retrieval helpers matching SentenceTransformers (task: search result | query: ... / title: none | text: ...)
    [[nodiscard]] Embedding encode_query(
        luisa::string_view query,
        const EncodeOptions &opt = {});

    [[nodiscard]] Embedding encode_document(
        luisa::string_view document,
        luisa::string_view title = "none",
        const EncodeOptions &opt = {});

    [[nodiscard]] luisa::vector<Embedding> encode_documents(
        luisa::span<const luisa::string_view> documents,
        const EncodeOptions &opt = {});

    // 2. Image encoding (from .mmfeat or path with .mmfeat sidecar)
    [[nodiscard]] Embedding encode_image(
        const std::filesystem::path &image_feat_path,
        const EncodeOptions &opt = {});

    [[nodiscard]] Embedding encode_image(
        const ImageFeatures &features,
        const EncodeOptions &opt = {});

    // 3. Audio encoding (from .mmfeat or path with .mmfeat sidecar)
    [[nodiscard]] Embedding encode_audio(
        const std::filesystem::path &audio_feat_path,
        const EncodeOptions &opt = {});

    [[nodiscard]] Embedding encode_audio(
        const AudioFeatures &features,
        const EncodeOptions &opt = {});

    // 4. Video encoding (from .mmfeat or path with .mmfeat sidecar)
    [[nodiscard]] Embedding encode_video(
        const std::filesystem::path &video_feat_path,
        const EncodeOptions &opt = {});

    [[nodiscard]] Embedding encode_video(
        const VideoFeatures &features,
        const EncodeOptions &opt = {});

    // 5. Interleaved Multimodal encoding
    [[nodiscard]] Embedding encode_multimodal(
        const std::filesystem::path &bundle_path,
        const EncodeOptions &opt = {});

    [[nodiscard]] Embedding encode_multimodal(
        const MultimodalFeatures &bundle,
        const EncodeOptions &opt = {});

    [[nodiscard]] Embedding encode_multimodal(
        const std::filesystem::path &media_feat_path,
        luisa::string_view prompt_text,
        const EncodeOptions &opt = {});

    [[nodiscard]] Embedding encode_multimodal_image(
        const ImageFeatures &features,
        luisa::string_view prompt_text,
        const EncodeOptions &opt = {});

    [[nodiscard]] Embedding encode_multimodal_audio(
        const std::filesystem::path &audio_feat_path,
        luisa::string_view prompt_text,
        const EncodeOptions &opt = {});

    [[nodiscard]] Embedding encode_multimodal_audio(
        const AudioFeatures &features,
        luisa::string_view prompt_text,
        const EncodeOptions &opt = {});

    [[nodiscard]] Embedding encode_multimodal_video(
        const std::filesystem::path &video_feat_path,
        luisa::string_view prompt_text,
        const EncodeOptions &opt = {});

    [[nodiscard]] Embedding encode_multimodal_video(
        const VideoFeatures &features,
        luisa::string_view prompt_text,
        const EncodeOptions &opt = {});

    // 6. Similarity computation matching model.similarity(a, b)
    [[nodiscard]] static float similarity(const Embedding &a, const Embedding &b) noexcept;

    [[nodiscard]] static luisa::vector<luisa::vector<float>> similarity_matrix(
        luisa::span<const Embedding> queries,
        luisa::span<const Embedding> candidates) noexcept;

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
    compute::Buffer<uint32_t> _dev_audio_token_indices;
    luisa::vector<uint32_t> _host_audio_token_indices;

    // Persistent video & multi-image buffers
    compute::Buffer<float> _dev_video_pixel_values;
    compute::Buffer<int32_t> _dev_video_position_ids;
    compute::Buffer<float> _dev_video_soft_tokens;
    compute::Buffer<uint32_t> _dev_video_token_indices;
    luisa::vector<uint32_t> _host_video_token_indices;
};

}// namespace luisa::embed_mm
