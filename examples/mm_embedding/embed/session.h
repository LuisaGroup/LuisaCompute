#pragma once
// mm_embedding engine -- public Session API.
//
// Composes Tokenizer + Package loader + Device WeightStore + Executor into
// a clean, unified embedding generation session.

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
        const std::filesystem::path &tokenizer_json);

    ~Session() noexcept;

    Session(const Session &) = delete;
    Session(Session &&) = delete;
    Session &operator=(const Session &) = delete;
    Session &operator=(Session &&) = delete;

    [[nodiscard]] Embedding encode(
        luisa::string_view text,
        const EncodeOptions &opt = {});

    [[nodiscard]] luisa::vector<Embedding> encode_batch(
        luisa::span<const luisa::string_view> texts,
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

    // Persistent workspace buffers to eliminate per-query driver allocation overhead
    compute::Buffer<uint32_t> _dev_ids;
    compute::Buffer<float> _dev_mask;
    compute::Buffer<float> _dev_out;
    luisa::vector<uint32_t> _host_ids;
    luisa::vector<float> _host_mask;
    luisa::vector<float> _host_out;
};

}// namespace luisa::mm_embed
