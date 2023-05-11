#pragma once
#include <runtime/device.h>
#include <runtime/event.h>

namespace luisa::compute {
class DStorageStream;
class DStorageFile;

class DStorageExt : public DeviceExtension {
public:
    static constexpr luisa::string_view name = "DStorageExt";
    struct File : public ResourceCreationInfo {
        size_t size_bytes;
    };
    virtual ResourceCreationInfo create_stream_handle() noexcept = 0;
    virtual File open_file_handle(luisa::string_view path) noexcept = 0;
    virtual void close_file_handle(uint64_t handle) noexcept = 0;
    virtual void destroy_stream_handle(uint64_t handle) noexcept = 0;
    virtual void enqueue_buffer(uint64_t stream_handle, uint64_t file, size_t file_offset, uint64_t buffer_handle, size_t offset, size_t size_bytes) noexcept = 0;
    virtual void enqueue_image(uint64_t stream_handle, uint64_t file, size_t file_offset, uint64_t image_handle, size_t pixel_size, uint32_t mip) noexcept = 0;
    virtual void signal(uint64_t stream_handle, uint64_t event_handle) noexcept = 0;
    virtual void commit(uint64_t stream_handle) noexcept = 0;
    [[nodiscard]] DStorageStream create_stream() noexcept;
    [[nodiscard]] DStorageFile open_file(luisa::string_view path) noexcept;
};
class DStorageFile {
    DStorageExt *_ext;
    DStorageExt::File _file;

public:
    struct BufferEnqueue {
        uint64_t file_handle;
        size_t file_offset;
        uint64_t buffer_handle;
        size_t buffer_offset;
        size_t size_bytes;
    };
    struct ImageEnqueue {
        uint64_t file_handle;
        size_t file_offset;
        uint64_t image_handle;
        size_t pixel_size;
        uint32_t mip_level;
    };
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
    BufferEnqueue read_to(size_t file_offset, Buffer<T> const &buffer) noexcept {
        return BufferEnqueue{
            .file_handle = _file.handle,
            .file_offset = file_offset,
            .buffer_handle = buffer.handle(),
            .buffer_offset = 0,
            .size_bytes = buffer.size_bytes()};
    }
    template<typename T>
    BufferEnqueue read_to(size_t file_offset, BufferView<T> const &buffer) noexcept {
        return BufferEnqueue{
            .file_handle = _file.handle,
            .file_offset = file_offset,
            .buffer_handle = buffer.handle(),
            .buffer_offset = buffer.offset_bytes(),
            .size_bytes = buffer.size_bytes()};
    }
    template<typename T>
    ImageEnqueue read_to(size_t file_offset, Image<T> const &image) noexcept {
        return ImageEnqueue{
            .file_handle = _file.handle,
            .file_offset = file_offset,
            .image_handle = image.handle(),
            .pixel_size = image.size_bytes(),
            .mip_level = 0};
    }
    template<typename T>
    ImageEnqueue read_to(size_t file_offset, ImageView<T> const &image) noexcept {
        return ImageEnqueue{
            .file_handle = _file.handle,
            .file_offset = file_offset,
            .image_handle = image.handle(),
            .pixel_size = image.size_bytes(),
            .mip_level = image.level()};
    }
    template<typename T>
    ImageEnqueue read_to(size_t file_offset, Volume<T> const &image) noexcept {
        return ImageEnqueue{
            .file_handle = _file.handle,
            .file_offset = file_offset,
            .image_handle = image.handle(),
            .pixel_size = image.size_bytes(),
            .mip_level = 0};
    }
    template<typename T>
    ImageEnqueue read_to(size_t file_offset, VolumeView<T> const &image) noexcept {
        return ImageEnqueue{
            .file_handle = _file.handle,
            .file_offset = file_offset,
            .image_handle = image.handle(),
            .pixel_size = image.size_bytes(),
            .mip_level = image.level()};
    }
};

class DStorageStream {
private:
    DStorageExt *_ext;
    uint64_t _handle;

public:
    explicit DStorageStream(DStorageExt *ext, uint64_t handle) noexcept
        : _ext{ext}, _handle{handle} {}
    class Delegate {
        friend class DStorageStream;
        DStorageStream *_stream;
        Delegate(DStorageStream *stream) noexcept
            : _stream{stream} {}

    public:
        Delegate(Delegate const &) = delete;
        Delegate(Delegate &&rhs) noexcept {
            _stream = rhs._stream;
            rhs._stream = nullptr;
        }
        Delegate &operator=(Delegate const &) = delete;
        Delegate &operator=(Delegate &&rhs) noexcept {
            this->~Delegate();
            new (std::launder(this)) Delegate{std::move(rhs)};
            return *this;
        }
        Delegate &&operator<<(DStorageFile::BufferEnqueue const &cmd) noexcept {
            _stream->_ext->enqueue_buffer(_stream->_handle, cmd.file_handle, cmd.file_offset, cmd.buffer_handle, cmd.buffer_offset, cmd.size_bytes);
            return std::move(*this);
        }
        Delegate &&operator<<(DStorageFile::ImageEnqueue const &cmd) noexcept {
            _stream->_ext->enqueue_image(_stream->_handle, cmd.file_handle, cmd.file_offset, cmd.image_handle, cmd.pixel_size, cmd.mip_level);
            return std::move(*this);
        }
        DStorageStream &operator<<(Event::Signal &&signal) noexcept {
            _stream->_ext->commit(_stream->_handle);
            _stream->_ext->signal(_stream->_handle, signal.handle);
            return *_stream;
        }
        ~Delegate() noexcept {
            if (_stream)
                _stream->_ext->commit(_stream->_handle);
        }
    };
    DStorageStream(DStorageStream &&rhs) noexcept {
        _ext = rhs._ext;
        _handle = rhs._handle;
        rhs._handle = invalid_resource_handle;
    }
    DStorageStream &operator=(DStorageStream &&rhs) noexcept {
        this->~DStorageStream();
        new (std::launder(this)) DStorageStream{std::move(rhs)};
        return *this;
    }
    ~DStorageStream() noexcept {
        if (_handle != invalid_resource_handle) {
            _ext->destroy_stream_handle(_handle);
        }
    }
    DStorageStream &operator<<(Event::Signal &&signal) noexcept {
        _ext->signal(_handle, signal.handle);
        return *this;
    }
    Delegate operator<<(DStorageFile::BufferEnqueue const &cmd) noexcept {
        return Delegate{this} << cmd;
    }
    Delegate operator<<(DStorageFile::ImageEnqueue const &cmd) noexcept {
        return Delegate{this} << cmd;
    }
};
inline DStorageStream DStorageExt::create_stream() noexcept {
    return DStorageStream{this, create_stream_handle().handle};
}
inline DStorageFile DStorageExt::open_file(luisa::string_view path) noexcept {
    return DStorageFile{this, open_file_handle(path)};
}
}// namespace luisa::compute