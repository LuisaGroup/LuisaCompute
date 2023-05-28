#pragma once

#include <core/logging.h>

#include <runtime/rhi/resource.h>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <runtime/image.h>
#include <runtime/volume.h>
#include <runtime/buffer.h>

#include <backends/ext/dstorage_ext_interface.h>
#include <backends/ext/dstorage_cmd.h>

namespace luisa::compute {

enum struct DecompressorID : uint {
    GDeflate = 4u,
};

struct GDeflateFileHeader {

    static constexpr auto default_chunk_size = 1u << 16u;
    static constexpr auto max_chunk_count = (1u << 16u) - 1u;

    uint8_t decompressor_id;
    uint8_t magic;
    uint16_t chunk_count;
    uint32_t chunk_size_index : 2;
    uint32_t last_chunk_size : 18;
    uint32_t reserved : 12;

    [[nodiscard]] static auto create(size_t uncompressed_size) noexcept {
        auto n = (uncompressed_size + default_chunk_size - 1u) / default_chunk_size;
        LUISA_ASSERT(n <= max_chunk_count,
                     "Too many chunks. ({} > {})",
                     n, max_chunk_count);
        GDeflateFileHeader header{};
        header.decompressor_id = to_underlying(DecompressorID::GDeflate);
        header.magic = header.decompressor_id ^ 0xffu;
        header.chunk_count = static_cast<uint16_t>(n);
        header.chunk_size_index = 1u;
        header.last_chunk_size = static_cast<uint32_t>(uncompressed_size % default_chunk_size);
        header.reserved = 0u;
        return header;
    }

    [[nodiscard]] auto valid() const noexcept {
        return decompressor_id == to_underlying(DecompressorID::GDeflate) &&
               decompressor_id == (magic ^ 0xffu) &&
               chunk_size_index == 1u;
    }

    [[nodiscard]] auto uncompressed_size() const noexcept {
        return chunk_count == 0u ?
                   static_cast<size_t>(0u) :
                   static_cast<size_t>(chunk_count - 1u) * default_chunk_size + last_chunk_size;
    }
};

inline void decode_gdeflate_stream(const GDeflateFileHeader *header,
                                   const void *input, void *output,
                                   const void **input_chunk_ptrs,
                                   size_t *input_chunk_sizes,
                                   void **output_chunk_ptrs) noexcept {
    auto chunk_offsets = reinterpret_cast<const uint32_t *>(header + 1u);
    LUISA_ASSERT(header->uncompressed_size() == chunk_offsets[0],
                 "Uncompressed size mismatch. ({} != {})",
                 header->uncompressed_size(), chunk_offsets[0]);
    auto chunk_count = header->chunk_count;
    if (chunk_count == 0u) { return; }
    auto chunk_data = reinterpret_cast<const std::byte *>(
        static_cast<const std::byte *>(input) +
        sizeof(GDeflateFileHeader) +
        sizeof(uint) * header->chunk_count);
    auto accum_chunk_size = 0u;
    auto output_ptr = static_cast<std::byte *>(output);
    for (auto i = 0u; i < chunk_count; i++) {
        auto chunk_size = chunk_offsets[(i + 1u) % chunk_count] - accum_chunk_size;
        input_chunk_ptrs[i] = chunk_data + accum_chunk_size;
        input_chunk_sizes[i] = chunk_size;
        output_chunk_ptrs[i] = output_ptr;
        accum_chunk_size += chunk_size;
        output_ptr += GDeflateFileHeader::default_chunk_size;
    }
}

class DStorageFileView {

private:
    uint64_t _handle;
    size_t _offset_bytes;
    size_t _size_bytes;
    bool _is_pinned_memory;

private:
    DStorageFileView(uint64_t handle,
                     size_t offset_bytes,
                     size_t size_bytes,
                     bool is_pinned_memory) noexcept
        : _handle{handle},
          _offset_bytes{offset_bytes},
          _size_bytes{size_bytes},
          _is_pinned_memory{is_pinned_memory} {}

public:
    DStorageFileView(const DStorageFile &file) noexcept;
    DStorageFileView(const DStorageFile &file, size_t offset_bytes, size_t size_bytes) noexcept;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto offset_bytes() const noexcept { return _offset_bytes; }
    [[nodiscard]] auto size_bytes() const noexcept { return _size_bytes; }

    [[nodiscard]] auto subview(size_t offset_bytes, size_t size_bytes) const noexcept {
        LUISA_ASSERT(offset_bytes < _size_bytes, "Offset out of range.");
        LUISA_ASSERT(size_bytes <= _size_bytes - offset_bytes, "Size out of range.");
        return DStorageFileView{_handle, _offset_bytes + offset_bytes,
                                size_bytes, _is_pinned_memory};
    }

private:
    [[nodiscard]] auto _dstorage_source() const noexcept {
        DStorageReadCommand::Source source;
        if (_is_pinned_memory) {
            source = DStorageReadCommand::MemorySource{
                .handle = handle(),
                .offset_bytes = offset_bytes(),
                .size_bytes = size_bytes(),
            };
        } else {
            source = DStorageReadCommand::FileSource{
                .handle = handle(),
                .offset_bytes = offset_bytes(),
                .size_bytes = size_bytes(),
            };
        }
        return source;
    }

public:
    template<typename BufferOrView>
        requires is_buffer_or_view_v<BufferOrView>
    [[nodiscard]] auto copy_to(BufferOrView &&buffer,
                               DStorageCompression compression = DStorageCompression::None) const noexcept {
        BufferView view{buffer};
        return luisa::make_unique<DStorageReadCommand>(
            _dstorage_source(),
            DStorageReadCommand::BufferRequest{
                .handle = view.handle(),
                .offset_bytes = view.offset_bytes(),
                .size_bytes = view.size_bytes()},
            compression);
    }

    template<typename ImageOrView>
        requires is_image_or_view_v<ImageOrView>
    [[nodiscard]] auto copy_to(ImageOrView &&image,
                               DStorageCompression compression = DStorageCompression::None) const noexcept {
        ImageView view{image};
        auto size = view.size();
        return luisa::make_unique<DStorageReadCommand>(
            _dstorage_source(),
            DStorageReadCommand::TextureRequest{
                .handle = view.handle(),
                .level = view.level(),
                .size = {size.x, size.y, 1u}},
            compression);
    }

    template<typename VolumeOrView>
        requires is_volume_or_view_v<VolumeOrView>
    [[nodiscard]] auto copy_to(VolumeOrView volume,
                               DStorageCompression compression = DStorageCompression::None) const noexcept {
        VolumeView view{volume};
        auto size = view.size();
        return luisa::make_unique<DStorageReadCommand>(
            _dstorage_source(),
            DStorageReadCommand::TextureRequest{
                .handle = view.handle(),
                .level = view.level(),
                .size = {size.x, size.y, size.z}},
            compression);
    }

    [[nodiscard]] auto copy_to(void *data, size_t size,
                               DStorageCompression compression = DStorageCompression::None) const noexcept {
        return luisa::make_unique<DStorageReadCommand>(
            _dstorage_source(),
            DStorageReadCommand::MemoryRequest{
                .data = data,
                .size_bytes = size},
            compression);
    }

    template<typename T>
    [[nodiscard]] auto copy_to(luisa::span<T> data,
                               DStorageCompression compression = DStorageCompression::None) const noexcept {
        return copy_to(data.data(), data.size_bytes(), compression);
    }

    template<typename ResourceOrView>
    [[nodiscard]] auto decompress_to(ResourceOrView &&resource,
                                     DStorageCompression method = DStorageCompression::GDeflate) const noexcept {
        LUISA_ASSERT(method != DStorageCompression::None, "Cannot decompress with no method specified.");
        return copy_to(std::forward<ResourceOrView>(resource), method);
    }

    [[nodiscard]] auto decompress_to(void *data, size_t size,
                                     DStorageCompression method = DStorageCompression::GDeflate) const noexcept {
        LUISA_ASSERT(method != DStorageCompression::None, "Cannot decompress with no method specified.");
        return copy_to(data, size, method);
    }
};

class DStorageFile : public Resource {

private:
    DStorageExt *_ext;
    size_t _size_bytes;

private:
    // proxy move constructor to eliminate the use-after-move problem
    DStorageFile(DStorageFile &&file,
                 DStorageExt *ext,
                 size_t size_bytes) noexcept
        : Resource{std::move(file)},
          _ext{ext}, _size_bytes{size_bytes} {}

public:
    explicit DStorageFile(DStorageExt *ext,
                          const DStorageExt::FileCreationInfo &info) noexcept
        : Resource{ext->device(), Tag::DSTORAGE_FILE, info},
          _ext{ext}, _size_bytes{info.size_bytes} {}

    explicit DStorageFile(DStorageExt *ext,
                          const DStorageExt::PinnedMemoryInfo &info) noexcept
        : Resource{ext->device(), Tag::DSTORAGE_PINNED_MEMORY, info},
          _ext{ext}, _size_bytes{info.size_bytes} {}

    DStorageFile(const DStorageFile &) noexcept = delete;
    DStorageFile(DStorageFile &&rhs) noexcept
        : DStorageFile{std::move(rhs), rhs._ext, rhs._size_bytes} {}

    DStorageFile &operator=(DStorageFile const &) noexcept = delete;
    DStorageFile &operator=(DStorageFile &&rhs) noexcept {
        _move_from(std::move(rhs));
        return *this;
    }
    using Resource::operator bool;

    ~DStorageFile() noexcept override {
        if (handle() != invalid_resource_handle) {
            if (tag() == Tag::DSTORAGE_FILE) {
                _ext->close_file_handle(handle());
            } else {
                _ext->unpin_host_memory(handle());
            }
        }
    }

    [[nodiscard]] size_t size_bytes() const noexcept { return _size_bytes; }

    [[nodiscard]] DStorageFileView view(size_t offset_bytes = 0u) const noexcept {
        LUISA_ASSERT(offset_bytes < _size_bytes, "Offset exceeds file size.");
        return DStorageFileView{*this}.subview(offset_bytes, _size_bytes - offset_bytes);
    }
    [[nodiscard]] DStorageFileView view(size_t offset_bytes, size_t size_bytes) const noexcept {
        return DStorageFileView{*this}.subview(offset_bytes, size_bytes);
    }

    template<typename ResourceOrView>
    [[nodiscard]] auto copy_to(ResourceOrView &&resource,
                               DStorageCompression compression = DStorageCompression::None) const noexcept {
        return view().copy_to(std::forward<ResourceOrView>(resource), compression);
    }

    [[nodiscard]] auto copy_to(void *data, size_t size,
                               DStorageCompression compression = DStorageCompression::None) const noexcept {
        return view().copy_to(data, size, compression);
    }

    template<typename ResourceOrView>
    [[nodiscard]] auto decompress_to(ResourceOrView &&resource,
                                     DStorageCompression method = DStorageCompression::GDeflate) const noexcept {
        return view().decompress_to(std::forward<ResourceOrView>(resource), method);
    }

    [[nodiscard]] auto decompress_to(void *data, size_t size,
                                     DStorageCompression method = DStorageCompression::GDeflate) const noexcept {
        return view().decompress_to(data, size, method);
    }
};

inline DStorageFileView::DStorageFileView(const DStorageFile &file) noexcept
    : DStorageFileView{file.handle(), 0u, file.size_bytes(),
                       file.tag() == Resource::Tag::DSTORAGE_PINNED_MEMORY} {}

inline DStorageFileView::DStorageFileView(const DStorageFile &file, size_t offset_bytes, size_t size_bytes) noexcept
    : DStorageFileView{file.handle(), offset_bytes, size_bytes,
                       file.tag() == Resource::Tag::DSTORAGE_PINNED_MEMORY} {
    LUISA_ASSERT(offset_bytes < file.size_bytes() &&
                     size_bytes <= file.size_bytes() - offset_bytes,
                 "Offset exceeds file size.");
}

inline DStorageFile DStorageExt::open_file(luisa::string_view path) noexcept {
    return DStorageFile{this, this->open_file_handle(path)};
}

inline DStorageFile DStorageExt::pin_memory(void *ptr, size_t size_bytes) noexcept {
    return DStorageFile{this, this->pin_host_memory(ptr, size_bytes)};
}

inline Stream DStorageExt::create_stream(const DStorageStreamOption &option) noexcept {
    return Stream{this->device(), StreamTag::CUSTOM, this->create_stream_handle(option)};
}

}// namespace luisa::compute
