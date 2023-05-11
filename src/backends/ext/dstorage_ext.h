#pragma once
#include <runtime/device.h>
#include <runtime/event.h>
#include <runtime/stream.h>
#include <runtime/rhi/command.h>

namespace luisa::compute {
class DStorageCommandList;
class DStorageFile;

class DStorageExt : public DeviceExtension {
public:
    static constexpr luisa::string_view name = "DStorageExt";
    struct File : public ResourceCreationInfo {
        size_t size_bytes;
    };

    virtual File open_file_handle(luisa::string_view path) noexcept = 0;
    virtual void close_file_handle(uint64_t handle) noexcept = 0;
    virtual std::pair<DeviceInterface *, ResourceCreationInfo> create_stream_handle() noexcept = 0;
    [[nodiscard]] Stream create_stream() noexcept {
        auto handle = create_stream_handle();
        return Stream{handle.first, StreamTag::CUSTOM, handle.second};
    }
    [[nodiscard]] DStorageFile open_file(luisa::string_view path) noexcept;
};
class DStorageFile {
    DStorageExt *_ext;
    DStorageExt::File _file;

public:
    explicit DStorageFile(DStorageExt *ext, DStorageExt::File const &file) noexcept : _ext{ext}, _file{file} {}
    DStorageFile(DStorageFile const &) noexcept = delete;
    DStorageFile(DStorageFile &&rhs) noexcept {
        _ext = rhs._ext;
        _file = rhs._file;
        rhs._file.invalidate();
    }
    DStorageFile &operator=(DStorageFile const &) = delete;
    DStorageFile &operator=(DStorageFile &&rhs) noexcept {
        this->~DStorageFile();
        new (std::launder(this)) DStorageFile{std::move(rhs)};
        return *this;
    }
    ~DStorageFile() noexcept {
        if (_file.valid()) {
            _ext->close_file_handle(_file.handle);
        }
    }
    [[nodiscard]] auto valid() const noexcept {
        return _file.valid();
    }
    [[nodiscard]] auto handle() const noexcept {
        return _file.handle;
    }
    [[nodiscard]] size_t size_bytes() const noexcept {
        return _file.size_bytes;
    }
    template<typename T>
    luisa::unique_ptr<DStorageReadCommand> read_to(size_t file_offset, Buffer<T> const &buffer) noexcept {
        return luisa::make_unique<DStorageReadCommand>(DStorageReadCommand::BufferEnqueue{
            .file_handle = _file.handle,
            .file_offset = file_offset,
            .buffer_handle = buffer.handle(),
            .buffer_offset = 0,
            .size_bytes = buffer.size_bytes()});
    }
    template<typename T>
    luisa::unique_ptr<DStorageReadCommand> read_to(size_t file_offset, BufferView<T> const &buffer) noexcept {
        return luisa::make_unique<DStorageReadCommand>(DStorageReadCommand::BufferEnqueue{
            .file_handle = _file.handle,
            .file_offset = file_offset,
            .buffer_handle = buffer.handle(),
            .buffer_offset = buffer.offset_bytes(),
            .size_bytes = buffer.size_bytes()});
    }
    template<typename T>
    luisa::unique_ptr<DStorageReadCommand> read_to(size_t file_offset, Image<T> const &image) noexcept {
        return luisa::make_unique<DStorageReadCommand>(DStorageReadCommand::ImageEnqueue{
            .file_handle = _file.handle,
            .file_offset = file_offset,
            .image_handle = image.handle(),
            .pixel_size = image.size_bytes(),
            .mip_level = 0});
    }
    template<typename T>
    luisa::unique_ptr<DStorageReadCommand> read_to(size_t file_offset, ImageView<T> const &image) noexcept {
        return luisa::make_unique<DStorageReadCommand>(DStorageReadCommand::ImageEnqueue{
            .file_handle = _file.handle,
            .file_offset = file_offset,
            .image_handle = image.handle(),
            .pixel_size = image.size_bytes(),
            .mip_level = image.level()});
    }
    template<typename T>
    luisa::unique_ptr<DStorageReadCommand> read_to(size_t file_offset, Volume<T> const &image) noexcept {
        return luisa::make_unique<DStorageReadCommand>(DStorageReadCommand::ImageEnqueue{
            .file_handle = _file.handle,
            .file_offset = file_offset,
            .image_handle = image.handle(),
            .pixel_size = image.size_bytes(),
            .mip_level = 0});
    }
    template<typename T>
    luisa::unique_ptr<DStorageReadCommand> read_to(size_t file_offset, VolumeView<T> const &image) noexcept {
        return luisa::make_unique<DStorageReadCommand>(DStorageReadCommand::ImageEnqueue{
            .file_handle = _file.handle,
            .file_offset = file_offset,
            .image_handle = image.handle(),
            .pixel_size = image.size_bytes(),
            .mip_level = image.level()});
    }
    luisa::unique_ptr<DStorageReadCommand> read_to(size_t file_offset, void *dst_ptr, size_t size_bytes) noexcept {
        return luisa::make_unique<DStorageReadCommand>(DStorageReadCommand::MemoryEnqueue{
            .file_handle = _file.handle,
            .file_offset = file_offset,
            .dst_ptr = dst_ptr,
            .size_bytes = size_bytes});
    }
    template<typename T>
        requires(std::is_trivial_v<T>)
    luisa::unique_ptr<DStorageReadCommand> read_to(size_t file_offset, luisa::span<T> dst) noexcept {
        return luisa::make_unique<DStorageReadCommand>(DStorageReadCommand::MemoryEnqueue{
            .file_handle = _file.handle,
            .file_offset = file_offset,
            .dst_ptr = dst.data(),
            .size_bytes = dst.size_bytes()});
    }
};

inline DStorageFile DStorageExt::open_file(luisa::string_view path) noexcept {
    return DStorageFile{this, open_file_handle(path)};
}

}// namespace luisa::compute