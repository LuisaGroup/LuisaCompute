#pragma once
#include <backends/ext/dstorage_ext_interface.h>
#include <runtime/rhi/resource.h>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <runtime/rhi/command.h>

namespace luisa::compute {
class DStorageCommandList;
enum class DStorageDecompressType : uint {
    GDeflate
};
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
    luisa::unique_ptr<Command> read_to(BufferView<T> const &buffer, size_t file_offset = 0) noexcept {
        return luisa::make_unique<DStorageReadCommand>(
            handle(), file_offset,
            buffer.size_bytes(),
            luisa::monostate{},
            DStorageReadCommand::BufferEnqueue{
                .buffer_handle = buffer.handle(),
                .buffer_offset = buffer.offset_bytes()});
    }
    template<typename T>
    luisa::unique_ptr<Command> read_to(Buffer<T> const &buffer, size_t file_offset = 0) noexcept {
        return read_to(buffer.view(), file_offset);
    }
    template<typename T>
    luisa::unique_ptr<Command> decompress_to(BufferView<T> const &buffer, size_t uncompressed_size, size_t file_offset = 0, DStorageDecompressType compress_type = DStorageDecompressType::GDeflate) noexcept {
        return luisa::make_unique<DStorageReadCommand>(
            handle(), file_offset,
            buffer.size_bytes(),
            DStorageReadCommand::GDeflateCompression{uncompressed_size},
            DStorageReadCommand::BufferEnqueue{
                .buffer_handle = buffer.handle(),
                .buffer_offset = buffer.offset_bytes()});
    }
    template<typename T>
    luisa::unique_ptr<Command> decompress_to(Buffer<T> const &buffer, size_t uncompressed_size, size_t file_offset = 0, DStorageDecompressType compress_type = DStorageDecompressType::GDeflate) noexcept {
        return decompress_to(buffer.view(), uncompressed_size, file_offset, compress_type);
    }
    template<typename T>
    luisa::unique_ptr<Command> read_to(
        ImageView<T> const &image, size_t file_offset = 0) noexcept {
        return luisa::make_unique<DStorageReadCommand>(
            handle(), file_offset,
            image.size_bytes(),
            luisa::monostate{},
            DStorageReadCommand::ImageEnqueue{
                .image_handle = image.handle(),
                .mip_level = image.level()});
    }
    template<typename T>
    luisa::unique_ptr<Command> read_to(
        Image<T> const &image, size_t file_offset = 0) noexcept {
        return read_to(image.view(0), file_offset);
    }
    template<typename T>
    luisa::unique_ptr<Command> decompress_to(
        ImageView<T> const &image, size_t uncompressed_size, size_t file_offset = 0, DStorageDecompressType compress_type = DStorageDecompressType::GDeflate) noexcept {
        return luisa::make_unique<DStorageReadCommand>(
            handle(), file_offset,
            image.size_bytes(),
            DStorageReadCommand::GDeflateCompression{uncompressed_size},
            DStorageReadCommand::ImageEnqueue{
                .image_handle = image.handle(),
                .mip_level = image.level()});
    }
    template<typename T>
    luisa::unique_ptr<Command> decompress_to(
        Image<T> const &image, size_t uncompressed_size, size_t file_offset = 0, DStorageDecompressType compress_type = DStorageDecompressType::GDeflate) noexcept {
        return decompress_to(image.view(0), uncompressed_size, file_offset, compress_type);
    }
    template<typename T>
    luisa::unique_ptr<Command> read_to(
        VolumeView<T> const &volume, size_t file_offset = 0) noexcept {
        return luisa::make_unique<DStorageReadCommand>(
            handle(), file_offset,
            volume.size_bytes(),
            luisa::monostate{},
            DStorageReadCommand::ImageEnqueue{
                .volume_handle = volume.handle(),
                .mip_level = volume.level()});
    }
    template<typename T>
    luisa::unique_ptr<Command> read_to(
        Volume<T> const &volume, size_t file_offset = 0) noexcept {
        return read_to(volume.view(0), file_offset);
    }
    template<typename T>
    luisa::unique_ptr<Command> decompress_to(
        VolumeView<T> const &volume, size_t uncompressed_size, size_t file_offset = 0, DStorageDecompressType compress_type = DStorageDecompressType::GDeflate) noexcept {
        return luisa::make_unique<DStorageReadCommand>(
            handle(), file_offset,
            volume.size_bytes(),
            DStorageReadCommand::GDeflateCompression{uncompressed_size},
            DStorageReadCommand::ImageEnqueue{
                .volume_handle = volume.handle(),
                .mip_level = volume.level()});
    }
    template<typename T>
    luisa::unique_ptr<Command> decompress_to(
        Volume<T> const &volume, size_t uncompressed_size, size_t file_offset = 0, DStorageDecompressType compress_type = DStorageDecompressType::GDeflate) noexcept {
        return decompress_to(volume.view(0), uncompressed_size, file_offset, compress_type);
    }
    template<typename T>
        requires(std::is_trivial_v<T>)
    luisa::unique_ptr<Command> read_to(
        luisa::span<T> dst, size_t file_offset = 0) noexcept {
        return luisa::make_unique<DStorageReadCommand>(
            handle(), file_offset,
            dst.size_bytes(),
            luisa::monostate{},
            DStorageReadCommand::MemoryEnqueue{
                .dst_ptr = dst.data()});
    }
    template<typename T>
        requires(std::is_trivial_v<T>)
    luisa::unique_ptr<Command> read_to(
        T *data, size_t size, size_t file_offset = 0) noexcept {
        return luisa::make_unique<DStorageReadCommand>(
            handle(), file_offset,
            size * sizeof(T),
            luisa::monostate{},
            DStorageReadCommand::MemoryEnqueue{
                .dst_ptr = data});
    }
};

inline DStorageFile DStorageExt::open_file(luisa::string_view path) noexcept {
    return DStorageFile{this, open_file_handle(path)};
}
inline Stream DStorageExt::create_stream() noexcept {
    return Stream{device(), StreamTag::CUSTOM, create_stream_handle()};
}
}// namespace luisa::compute