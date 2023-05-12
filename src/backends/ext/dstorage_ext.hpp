#pragma once
#include <backends/ext/dstorage_ext_interface.h>
#include <runtime/rhi/resource.h>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <runtime/rhi/command.h>

namespace luisa::compute {
class DStorageCommandList;
class DStorageFile : public Resource {
    DStorageExt *_ext;
    size_t _size_bytes;

public:
    explicit DStorageFile(DStorageExt *ext, DStorageExt::File const &file) noexcept
        : Resource{
              ext->device(),
              Tag::DSTORAGE_FILE,
              file},
          _ext{ext}, _size_bytes{file.size_bytes} {
    }
    DStorageFile(DStorageFile const &) noexcept = delete;
    DStorageFile(DStorageFile &&rhs) noexcept
        : Resource{std::move(rhs)} {
        _ext = rhs._ext;
        _size_bytes = rhs._size_bytes;
    }
    DStorageFile &operator=(DStorageFile const &) = delete;
    DStorageFile &operator=(DStorageFile &&rhs) noexcept {
        this->~DStorageFile();
        new (std::launder(this)) DStorageFile{std::move(rhs)};
        return *this;
    }
    using Resource::operator bool;
    ~DStorageFile() noexcept {
        if (handle() != invalid_resource_handle) {
            _ext->close_file_handle(handle());
        }
    }
    [[nodiscard]] size_t size_bytes() const noexcept {
        return _size_bytes;
    }
    template<typename T>
    luisa::unique_ptr<Command> read_to(Buffer<T> const &buffer, size_t file_offset = 0) noexcept {
        return luisa::make_unique<DStorageReadCommand>(DStorageReadCommand::BufferEnqueue{
            .file_handle = handle(),
            .file_offset = file_offset,
            .buffer_handle = buffer.handle(),
            .buffer_offset = 0,
            .size_bytes = buffer.size_bytes()});
    }
    template<typename T>
    luisa::unique_ptr<Command> read_to(BufferView<T> const &buffer, size_t file_offset = 0) noexcept {
        return luisa::make_unique<DStorageReadCommand>(DStorageReadCommand::BufferEnqueue{
            .file_handle = handle(),
            .file_offset = file_offset,
            .buffer_handle = buffer.handle(),
            .buffer_offset = buffer.offset_bytes(),
            .size_bytes = buffer.size_bytes()});
    }
    template<typename T>
    luisa::unique_ptr<Command> read_to(Image<T> const &image, size_t file_offset = 0) noexcept {
        return luisa::make_unique<DStorageReadCommand>(DStorageReadCommand::ImageEnqueue{
            .file_handle = handle(),
            .file_offset = file_offset,
            .image_handle = image.handle(),
            .pixel_size = image.size_bytes(),
            .mip_level = 0});
    }
    template<typename T>
    luisa::unique_ptr<Command> read_to(ImageView<T> const &image, size_t file_offset = 0) noexcept {
        return luisa::make_unique<DStorageReadCommand>(DStorageReadCommand::ImageEnqueue{
            .file_handle = handle(),
            .file_offset = file_offset,
            .image_handle = image.handle(),
            .pixel_size = image.size_bytes(),
            .mip_level = image.level()});
    }
    template<typename T>
    luisa::unique_ptr<Command> read_to(Volume<T> const &image, size_t file_offset = 0) noexcept {
        return luisa::make_unique<DStorageReadCommand>(DStorageReadCommand::ImageEnqueue{
            .file_handle = handle(),
            .file_offset = file_offset,
            .image_handle = image.handle(),
            .pixel_size = image.size_bytes(),
            .mip_level = 0});
    }
    template<typename T>
    luisa::unique_ptr<Command> read_to(VolumeView<T> const &image, size_t file_offset = 0) noexcept {
        return luisa::make_unique<DStorageReadCommand>(DStorageReadCommand::ImageEnqueue{
            .file_handle = handle(),
            .file_offset = file_offset,
            .image_handle = image.handle(),
            .pixel_size = image.size_bytes(),
            .mip_level = image.level()});
    }
    luisa::unique_ptr<Command> read_to(void *dst_ptr, size_t size_bytes, size_t file_offset = 0) noexcept {
        return luisa::make_unique<DStorageReadCommand>(DStorageReadCommand::MemoryEnqueue{
            .file_handle = handle(),
            .file_offset = file_offset,
            .dst_ptr = dst_ptr,
            .size_bytes = size_bytes});
    }
    template<typename T>
        requires(std::is_trivial_v<T>)
    luisa::unique_ptr<Command> read_to(luisa::span<T> dst, size_t file_offset = 0) noexcept {
        return luisa::make_unique<DStorageReadCommand>(DStorageReadCommand::MemoryEnqueue{
            .file_handle = handle(),
            .file_offset = file_offset,
            .dst_ptr = dst.data(),
            .size_bytes = dst.size_bytes()});
    }
    template<typename T>
        requires(std::is_trivial_v<T>)
    luisa::unique_ptr<Command> read_to(T *data, size_t size_bytes, size_t file_offset = 0) noexcept {
        return luisa::make_unique<DStorageReadCommand>(DStorageReadCommand::MemoryEnqueue{
            .file_handle = handle(),
            .file_offset = file_offset,
            .dst_ptr = data,
            .size_bytes = size_bytes});
    }
};

inline DStorageFile DStorageExt::open_file(luisa::string_view path) noexcept {
    return DStorageFile{this, open_file_handle(path)};
}
inline Stream DStorageExt::create_stream() noexcept {
    return Stream{device(), StreamTag::CUSTOM, create_stream_handle()};
}
}// namespace luisa::compute