#pragma once

#include <luisa/runtime/rhi/device_interface.h>

namespace lc::validation {
class DStorageExtImpl;
}// namespace lc::validation

namespace luisa::compute {

class Stream;
class DStorageFile;

#ifdef None
#undef None
#endif

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
    // A queue that accepts both file- and memory-sourced requests.  A backend
    // may implement this with two native queues (DX: one per source type), or
    // with a single host path that treats both sources identically (VK staging
    // fallback).
    AnySource = MemorySource | FileSource
};

struct DStorageStreamOption {
    /// The process-wide default staging buffer size (64 MiB).  DirectStorage
    /// applies this at first use; see the note on `staging_buffer_size`.
    static constexpr size_t default_staging_buffer_size = 64ull * 1024ull * 1024ull;

    DStorageStreamSource source{DStorageStreamSource::FileSource};
    /// Desired staging buffer size.  On DX this is a *process-global* hint
    /// (`IDStorageFactory::SetStagingBufferSize`): it is applied once, before
    /// the first queue is created, and every stream created afterwards must
    /// request a consistent value.  A later, different request is ignored with
    /// a warning rather than failing.  DirectStorage further requires that a
    /// single request never exceeds the staging buffer size.
    size_t staging_buffer_size = default_staging_buffer_size;
    /// Hint for spinning/HDD storage.  Also process-global on DX and latched
    /// by the first stream that requests it (see `DStorageExtImpl::_set_config`).
    bool supports_hdd{false};
};

/// Direct storage extension.
///
/// Handle contracts
///   * `DStorageReadCommand::FileSource::handle` is the handle returned by
///     `open_file()` (i.e. `DStorageFile::handle()`).
///   * `DStorageReadCommand::MemorySource::handle` is the handle returned by
///     `pin_memory()` (i.e. a `DStorageFile` created from `pin_host_memory`).
///     The concrete value is **opaque and backend-private**: DX and the VK
///     fallback use the raw host pointer as the handle, while CUDA/Metal use a
///     pointer to an internal object.  Callers must only ever pass the value
///     back through the DStorage API.
///
/// Lifetime
///   A `DStorageReadCommand` carries only the opaque handle, so the backend
///   still dereferences the underlying file/pinned-memory object while the
///   transfer is in flight.  A `DStorageFile` must therefore **outlive every
///   read it produced** (`dstream << read << synchronize()`, or a wait on an
///   event ordered after it, before the file goes out of scope).
///   `DStorageFileView` is non-owning and lightweight: a view may be destroyed
///   as soon as the command has been constructed.
class DStorageExt : public DeviceExtension {

public:
    static constexpr luisa::string_view name = "DStorageExt";
    using Compression = DStorageCompression;
    using CompressionQuality = DStorageCompressionQuality;
    static constexpr size_t default_staging_buffer_size =
        DStorageStreamOption::default_staging_buffer_size;

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
    [[nodiscard]] DStorageFile pin_memory(void *ptr, size_t size_bytes) noexcept;

    /// Whether `compress()` accepts the given algorithm.  The default accepts
    /// only `Compression::None` (a plain copy).  Backends that ship a
    /// compressor override this so callers can probe capability up front
    /// instead of hitting a fatal error half-way through a stream.
    [[nodiscard]] virtual bool supports_compression(Compression algorithm) const noexcept {
        return algorithm == Compression::None;
    }

    virtual void compress(const void *data, size_t size_bytes,
                          Compression algorithm, CompressionQuality quality,
                          luisa::vector<std::byte> &result) noexcept = 0;
};

}// namespace luisa::compute
