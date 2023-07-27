#include <luisa/core/logging.h>
#include <luisa/core/binary_file_stream.h>

namespace luisa {

#ifdef _WIN32
#define LUISA_FSEEK _fseeki64_nolock
#define LUISA_FTELL _ftelli64_nolock
#define LUISA_FREAD _fread_nolock
#define LUISA_FCLOSE _fclose_nolock
#else
#define LUISA_FSEEK fseeko
#define LUISA_FTELL ftello
#define LUISA_FREAD fread
#define LUISA_FCLOSE fclose
#endif
size_t BinaryFileStream::seek_len(::FILE *file) noexcept {
    LUISA_FSEEK(file, 0, SEEK_END);
    auto length = LUISA_FTELL(file);
    LUISA_FSEEK(file, 0, SEEK_SET);
    return length;
}
BinaryFileStream::BinaryFileStream(::FILE *file, size_t length) noexcept
    : _file{file},
      _length{length} {
}
BinaryFileStream::BinaryFileStream(const luisa::string &path) noexcept {
    _file = std::fopen(path.c_str(), "rb");
    if (_file) {
        _length = seek_len(_file);
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
    LUISA_FREAD(dst.data(), 1, size, _file);
    _pos += size;
}

void BinaryFileStream::close() noexcept {
    if (_file) [[likely]] { LUISA_FCLOSE(_file); }
    _length = 0;
    _pos = 0;
}

#undef LUISA_FSEEK
#undef LUISA_FTELL
#undef LUISA_FREAD
#undef LUISA_FCLOSE

}// namespace luisa

