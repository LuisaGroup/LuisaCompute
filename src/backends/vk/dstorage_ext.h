#pragma once
#include <cstdio>
#include <luisa/core/logging.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/backends/ext/dstorage_ext_interface.h>
#include <luisa/backends/ext/dstorage_cmd.h>
#include "device.h"
// `CustomStream` uses the complete `lc::vk::Stream`/`Event` types (the internal
// COPY stream is signalled/waited through them), not just the forward
// declarations `device.h` carries.
#include "stream.h"

namespace lc::vk {

class VkDStorageExt;

/// A file opened by the Vulkan direct-storage fallback.  The fallback reads
/// into host memory synchronously, so the file is a plain `std::FILE *` and the
/// reads are serialized by the owning stream's mutex.
class VkDStorageFile {
    std::FILE *_file{nullptr};
    size_t _size_bytes{0u};
    luisa::string _path;

public:
    explicit VkDStorageFile(luisa::string_view path) noexcept;
    ~VkDStorageFile() noexcept;
    VkDStorageFile(VkDStorageFile const &) noexcept = delete;
    VkDStorageFile(VkDStorageFile &&) noexcept = delete;
    VkDStorageFile &operator=(VkDStorageFile const &) noexcept = delete;
    VkDStorageFile &operator=(VkDStorageFile &&) noexcept = delete;
    [[nodiscard]] bool valid() const noexcept { return _file != nullptr; }
    [[nodiscard]] size_t size_bytes() const noexcept { return _size_bytes; }
    [[nodiscard]] luisa::string_view path() const noexcept { return _path; }
    /// Read `size` bytes at `offset` into `dst`.  Returns false (without
    /// touching `dst`) when the range is out of bounds or the read fails.
    [[nodiscard]] bool read(uint64_t offset, size_t size, void *dst) noexcept;
};

/// Host-staged direct-storage stream for the Vulkan backend.
///
/// The host reads the requested file/pinned-memory range into a reusable
/// scratch buffer, then the data is uploaded through ordinary Luisa commands
/// (`BufferUploadCommand` / `TextureUploadCommand`) recorded on an internal
/// COPY stream.  Raw-memory destinations are written directly on the host and
/// emit no GPU command at all.
class VkDStorageStream final : public CustomStream {
    Device *_device{nullptr};
    VkDStorageExt *_ext{nullptr};
    uint64_t _copy_stream_handle{0u};
    Stream *_copy_stream{nullptr};
    luisa::spin_mutex _mtx;

public:
    VkDStorageStream(Device *device, VkDStorageExt *ext) noexcept;
    ~VkDStorageStream() noexcept override;
    VkDStorageStream(VkDStorageStream const &) noexcept = delete;
    VkDStorageStream(VkDStorageStream &&) noexcept = delete;

    [[nodiscard]] bool valid() const noexcept { return _copy_stream != nullptr; }
    [[nodiscard]] VkQueue queue() const noexcept { return _copy_stream == nullptr ? VK_NULL_HANDLE : _copy_stream->queue(); }

    void dispatch(CommandList &&list) noexcept override;
    void synchronize() noexcept override;
    void signal(Event *event, uint64_t fence) noexcept override;
    void wait(Event *event, uint64_t fence) noexcept override;
    void set_log_callback(const DeviceInterface::StreamLogCallback &callback) noexcept override;
};

/// Vulkan `DStorageExt`: a host-staged fallback (no native DirectStorage).
///
/// Supported:
///   * file- and pinned-memory-sourced reads into buffers, textures and raw
///     memory, split by a staging buffer on the CPU side;
///   * `Compression::None` only (`supports_compression` reports exactly that).
/// Not supported (reported, never silently ignored): compressed reads, sparse
/// destinations and custom decompression queues.
class VkDStorageExt final : public DStorageExt {
    Device *_device{nullptr};
    std::mutex _mtx;
    // Pinned host ranges, keyed by the opaque handle (the raw host pointer).
    luisa::unordered_map<uint64_t, size_t> _pinned_ranges;

public:
    explicit VkDStorageExt(Device *device) noexcept : _device{device} {}
    ~VkDStorageExt() noexcept;

protected:
    [[nodiscard]] DeviceInterface *device() const noexcept override { return _device; }
    [[nodiscard]] ResourceCreationInfo create_stream_handle(const DStorageStreamOption &option) noexcept override;
    [[nodiscard]] FileCreationInfo open_file_handle(luisa::string_view path) noexcept override;
    void close_file_handle(uint64_t handle) noexcept override;
    [[nodiscard]] PinnedMemoryInfo pin_host_memory(void *ptr, size_t size_bytes) noexcept override;
    void unpin_host_memory(uint64_t handle) noexcept override;

public:
    /// Size of a previously pinned host range, or 0 when unknown.
    [[nodiscard]] size_t pinned_memory_size(uint64_t handle) noexcept;
    [[nodiscard]] bool supports_compression(Compression algorithm) const noexcept override {
        return algorithm == Compression::None;
    }
    void compress(const void *data, size_t size_bytes,
                  Compression algorithm, CompressionQuality quality,
                  luisa::vector<std::byte> &result) noexcept override;
};

}// namespace lc::vk
