#pragma once

#include <luisa/core/logging.h>

#include <luisa/runtime/rhi/resource.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/sparse_image.h>
#include <luisa/runtime/sparse_volume.h>
#include <luisa/runtime/buffer.h>

#include <luisa/backends/ext/dstorage_ext_interface.h>
#include <luisa/backends/ext/dstorage_cmd.h>

namespace luisa::compute {

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

    template<typename SparseImage>
        requires is_sparse_image_v<SparseImage>
    [[nodiscard]] auto copy_to(
        SparseImage &&image,
        uint2 start_tile, uint2 tile_count, uint mip_level,
        DStorageCompression compression = DStorageCompression::None) const noexcept {
        auto size = tile_count * image.tile_size();
        auto offset = start_tile * image.tile_size();
        return luisa::make_unique<DStorageReadCommand>(
            _dstorage_source(),
            DStorageReadCommand::TextureRequest{
                .handle = image.handle(),
                .level = mip_level,
                .offset = {offset.x, offset.y, 0u},
                .size = {size.x, size.y, 1u}},
            compression);
    }

    template<typename SparseVolume>
        requires is_sparse_volume_v<SparseVolume>
    [[nodiscard]] auto copy_to(
        SparseVolume &&volume,
        uint3 start_tile, uint3 tile_count, uint mip_level,
        DStorageCompression compression = DStorageCompression::None) const noexcept {
        auto size = tile_count * volume.tile_size();
        auto offset = start_tile * volume.tile_size();
        return luisa::make_unique<DStorageReadCommand>(
            _dstorage_source(),
            DStorageReadCommand::TextureRequest{
                .handle = volume.handle(),
                .level = mip_level,
                .offset = {offset.x, offset.y, offset.z},
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

    template<typename SparseImage>
        requires is_sparse_image_v<SparseImage>
    [[nodiscard]] auto copy_to(
        SparseImage &&image,
        uint2 start_tile, uint2 tile_count, uint mip_level,
        DStorageCompression compression = DStorageCompression::None) const noexcept {
        return view().copy_to(std::forward<SparseImage>(image), start_tile, tile_count, mip_level, compression);
    }

    template<typename SparseVolume>
        requires is_sparse_volume_v<SparseVolume>
    [[nodiscard]] auto copy_to(
        SparseVolume &&volume,
        uint3 start_tile, uint3 tile_count, uint mip_level,
        DStorageCompression compression = DStorageCompression::None) const noexcept {
        return view().copy_to(std::forward<SparseVolume>(volume), start_tile, tile_count, mip_level, compression);
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

