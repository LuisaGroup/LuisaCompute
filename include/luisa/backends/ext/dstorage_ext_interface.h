#pragma once

#include <luisa/runtime/rhi/device_interface.h>

namespace lc::validation {
class DStorageExtImpl;
}// namespace lc::validation

namespace luisa::compute {

class Stream;
class DStorageFile;

enum class DStorageCompression : uint {
    None,
    GDeflate,
    Cascaded,
    LZ4,
    Snappy,
    Bitcomp,
    ANS,
    LZFSE,
    LZMA,
    LZBitmap
};

enum class DStorageCompressionQuality : uint {
    Fastest,
    Default,
    Best
};

enum class DStorageStreamSource : uint {
    MemorySource = 1,
    FileSource = 2,
    AnySource = MemorySource | FileSource
};

struct DStorageStreamOption {
    DStorageStreamSource source{DStorageStreamSource::FileSource};
    bool supports_hdd{false};
};

class DStorageExt : public DeviceExtension {

public:
    static constexpr luisa::string_view name = "DStorageExt";
    using Compression = DStorageCompression;
    using CompressionQuality = DStorageCompressionQuality;

protected:
    friend class DStorageFile;
    friend class lc::validation::DStorageExtImpl;
    struct FileCreationInfo : public ResourceCreationInfo {
        size_t size_bytes;
        [[nodiscard]] static auto make_invalid() noexcept {
            return FileCreationInfo{ResourceCreationInfo::make_invalid(), 0u};
        }
    };
    struct PinnedMemoryInfo : public ResourceCreationInfo {
        size_t size_bytes;
        [[nodiscard]] static auto make_invalid() noexcept {
            return PinnedMemoryInfo{ResourceCreationInfo::make_invalid(), 0u};
        }
    };

protected:
    ~DStorageExt() noexcept = default;
    [[nodiscard]] virtual DeviceInterface *device() const noexcept = 0;
    [[nodiscard]] virtual ResourceCreationInfo create_stream_handle(const DStorageStreamOption &option) noexcept = 0;
    [[nodiscard]] virtual FileCreationInfo open_file_handle(luisa::string_view path) noexcept = 0;
    virtual void close_file_handle(uint64_t handle) noexcept = 0;
    [[nodiscard]] virtual PinnedMemoryInfo pin_host_memory(void *ptr, size_t size_bytes) noexcept = 0;
    virtual void unpin_host_memory(uint64_t handle) noexcept = 0;

public:
    [[nodiscard]] Stream create_stream(const DStorageStreamOption &option = {}) noexcept;
    [[nodiscard]] DStorageFile open_file(luisa::string_view path) noexcept;
    [[nodiscard]] DStorageFile pin_memory(void *data, size_t size_bytes) noexcept;

    virtual void compress(const void *data, size_t size_bytes,
                          Compression algorithm, CompressionQuality quality,
                          luisa::vector<std::byte> &result) noexcept = 0;
};

}// namespace luisa::compute
