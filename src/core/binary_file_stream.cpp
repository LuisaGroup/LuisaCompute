#include <luisa/core/logging.h>
#include <luisa/core/binary_file_stream.h>

namespace luisa {

#ifdef LUISA_PLATFORM_WINDOWS
#define LUISA_FSEEK _fseeki64
#define LUISA_FTELL _ftelli64
#else
#define LUISA_FSEEK fseeko
#define LUISA_FTELL ftello
#endif

namespace detail {

[[nodiscard]] LC_CORE_API size_t get_c_file_length(::FILE *file) noexcept {
    LUISA_FSEEK(file, 0, SEEK_END);
    auto length = LUISA_FTELL(file);
    LUISA_FSEEK(file, 0, SEEK_SET);
    return length;
}

}// namespace detail

BinaryFileStream::BinaryFileStream(::FILE *file, size_t length) noexcept
    : _file{file},
      _length{length} {
}

BinaryFileStream::BinaryFileStream(const luisa::string &path) noexcept {
    _file = std::fopen(path.c_str(), "rb");
    if (_file) {
        _length = detail::get_c_file_length(_file);
    } else {
        LUISA_VERBOSE("Read file {} failed.", path);
    }
}

BinaryFileStream::~BinaryFileStream() noexcept { close(); }

BinaryFileStream::BinaryFileStream(BinaryFileStream &&another) noexcept
    : _file{another._file},
      _length{another._length},
      _pos{another._pos} {
    another._file = nullptr;
    another._length = 0;
    another._pos = 0;
}

BinaryFileStream &BinaryFileStream::operator=(BinaryFileStream &&rhs) noexcept {
    if (&rhs != this) [[likely]] {
        close();
        _file = rhs._file;
        _length = rhs._length;
        _pos = rhs._pos;
        rhs._file = nullptr;
        rhs._length = 0;
        rhs._pos = 0;
    }
    return *this;
}

void BinaryFileStream::read(luisa::span<std::byte> dst) noexcept {
    if (!_file) [[unlikely]] {
        return;
    }
    auto size = std::min(dst.size(), _length - _pos);
    fread(dst.data(), 1, size, _file);
    _pos += size;
}

void BinaryFileStream::close() noexcept {
    if (_file) [[likely]] { fclose(_file); }
    _length = 0;
    _pos = 0;
}

void BinaryFileStream::set_pos(size_t pos) noexcept {
    LUISA_ASSERT(pos <= _length, "Set pos {} out of range {}.", pos, _length);
    LUISA_FSEEK(_file, static_cast<int64_t>(pos), SEEK_SET);
    _pos = pos;
}

#undef LUISA_FSEEK
#undef LUISA_FTELL

}// namespace luisa
