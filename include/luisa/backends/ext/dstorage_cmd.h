#pragma once

#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/rhi/pixel.h>
#include <luisa/backends/ext/registry.h>
#include <luisa/backends/ext/dstorage_ext_interface.h>

namespace luisa::compute {

// ---------------------------------------------------------------------------
// Direct-storage source layout contract
// ---------------------------------------------------------------------------
// A `DStorageReadCommand` moves bytes from `source()` into `request()`.
//
// Source size
//   `Source::size_bytes` is the number of bytes owned by the source.  For a
//   `DStorageFileView` this is the view's `size_bytes()`; for a
//   `DStorageFile` (pinned host memory) it is the pinned range size.
//
// Buffer / Memory destinations
//   Source bytes are contiguous, starting at `Source::offset_bytes`.  The
//   transfer size is `effective_transfer_size()`, i.e.
//   `min(Source::size_bytes, destination size in bytes)`.  Requests built
//   through `DStorageFileView::copy_to` are clamped at construction, so a
//   short file view never over-reads: only `effective_transfer_size()` bytes
//   are moved.
//
// Texture destinations
//   The addressed region is `TextureRequest::offset` / `size` (in texels) at
//   mip `level`.  Source rows are padded to a 256-byte pitch
//   (`D3D12_TEXTURE_DATA_PITCH_ALIGNMENT`), i.e. the required source size is
//   `dstorage_texture_source_size(storage, size)` =
//   `dstorage_texture_row_pitch(storage, size.x) * size.y * size.z`.
//   Rows inside a plane are stored back-to-back, planes are stored
//   back-to-back, and the padded region starts at `Source::offset_bytes`.
//   `TextureRequest::storage` makes the request self-describing so a backend
//   can derive the pitch without querying the native texture, and so writers
//   and tests can produce a file that every backend accepts.
//
//   The VK fallback repacks the 256-byte-pitch source into the tightly packed
//   rows its upload path expects; the CUDA/Metal paths use the same padded
//   layout (see `dstorage_texture_row_pitch`).
// ---------------------------------------------------------------------------

/// Row pitch (in bytes) of a direct-storage texture source for `width` texels
/// of `storage`.  Rows are aligned up to `D3D12_TEXTURE_DATA_PITCH_ALIGNMENT`
/// (256 bytes).
[[nodiscard]] constexpr size_t dstorage_texture_row_pitch(
    PixelStorage storage, uint width) noexcept {
    auto tight = pixel_storage_size(storage, uint3{width, 1u, 1u});
    return (tight + 255u) & ~size_t{255u};
}

/// Total bytes a direct-storage texture source must provide for a region of
/// `size` texels (`dstorage_texture_row_pitch(storage, size.x) * y * z`).
[[nodiscard]] constexpr size_t dstorage_texture_source_size(
    PixelStorage storage, uint3 size) noexcept {
    return dstorage_texture_row_pitch(storage, size.x) * size.y * size.z;
}

class DStorageReadCommand : public CustomCommand {

public:
    struct FileSource {
        uint64_t handle;
        size_t offset_bytes;
        size_t size_bytes;
    };

    struct MemorySource {
        uint64_t handle;
        size_t offset_bytes;
        size_t size_bytes;
    };

    using Source = luisa::variant<
        FileSource,
        MemorySource>;

    struct BufferRequest {
        uint64_t handle;
        size_t offset_bytes;
        size_t size_bytes;
    };

    struct TextureRequest {
        uint64_t handle;
        uint32_t level;
        PixelStorage storage;
        uint32_t offset[3u];
        uint32_t size[3u];
    };

    struct MemoryRequest {
        void *data;
        size_t size_bytes;
    };

    using Request = luisa::variant<
        BufferRequest,
        TextureRequest,
        MemoryRequest>;

    using Compression = DStorageCompression;

private:
    Source _source;
    Request _request;
    Compression _compression;

public:
    DStorageReadCommand(Source source,
                        Request request,
                        Compression compression) noexcept
        : _source{source},
          _request{request},
          _compression{compression} {}
    [[nodiscard]] const auto &source() const noexcept { return _source; }
    [[nodiscard]] const auto &request() const noexcept { return _request; }
    [[nodiscard]] auto compression() const noexcept { return _compression; }
    [[nodiscard]] bool is_compressed() const noexcept { return _compression != Compression::None; }

    /// Number of bytes the source owns (the clamped/clamp-able upper bound of
    /// the transfer).
    [[nodiscard]] size_t source_size_bytes() const noexcept {
        return luisa::visit(
            [](auto const &s) noexcept { return s.size_bytes; },
            _source);
    }

    /// Destination capacity in bytes.  For a texture destination this is the
    /// tightly packed GPU footprint `pixel_storage_size(storage, size)`.
    [[nodiscard]] size_t destination_size_bytes() const noexcept {
        return luisa::visit(
            []<typename T>(T const &r) noexcept -> size_t {
                if constexpr (std::is_same_v<T, TextureRequest>) {
                    return pixel_storage_size(
                        r.storage, uint3{r.size[0], r.size[1], r.size[2]});
                } else {
                    return r.size_bytes;
                }
            },
            _request);
    }

    /// Source bytes this request consumes when fully satisfied.
    ///   * Buffer / Memory destination: the destination size in bytes.
    ///   * Texture destination: the pitch-aligned source layout size,
    ///     `dstorage_texture_source_size(storage, size)`.
    [[nodiscard]] size_t required_source_size_bytes() const noexcept {
        return luisa::visit(
            []<typename T>(T const &r) noexcept -> size_t {
                if constexpr (std::is_same_v<T, TextureRequest>) {
                    return dstorage_texture_source_size(
                        r.storage, uint3{r.size[0], r.size[1], r.size[2]});
                } else {
                    return r.size_bytes;
                }
            },
            _request);
    }

    /// Bytes actually moved from the source, clamped by both the source size
    /// and what the request needs.  This is the size every backend must
    /// transfer.
    [[nodiscard]] size_t effective_transfer_size() const noexcept {
        auto s = source_size_bytes();
        auto r = required_source_size_bytes();
        return s < r ? s : r;
    }

    [[nodiscard]] uint64_t custom_cmd_uuid() const noexcept override { return to_underlying(CustomCommandUUID::DSTORAGE_READ); }
    LUISA_MAKE_COMMAND_COMMON(StreamTag::CUSTOM)
};

}// namespace luisa::compute
