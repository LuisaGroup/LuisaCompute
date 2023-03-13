//
// Created by Mike on 3/13/2023.
//

#include <core/logging.h>
#include <core/binary_file_stream.h>

namespace luisa {

#ifdef _WIN32
#define LUISA_FOPEN fopen
#define LUISA_FSEEK _fseeki64_nolock
#define LUISA_FTELL _ftelli64_nolock
#define LUISA_FREAD _fread_nolock
#define LUISA_FCLOSE _fclose_nolock
#else
#define LUISA_FOPEN fopen
#define LUISA_FSEEK fseeko
#define LUISA_FTELL ftello
#define LUISA_FREAD fread
#define LUISA_FCLOSE fclose
#endif

BinaryFileStream::BinaryFileStream(const luisa::string &path) noexcept {
    _file = LUISA_FOPEN(path.c_str(), "rb");
    if (_file != nullptr) {
        LUISA_FSEEK(_file, 0, SEEK_END);
        _length = LUISA_FTELL(_file);
        LUISA_FSEEK(_file, 0, SEEK_SET);
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
    if (&rhs != this) {
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
    if (!valid()) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to read from invalid file stream.");
        return;
    }
    auto size = std::min(dst.size(), _length - _pos);
    LUISA_FREAD(dst.data(), 1, size, _file);
    _pos += size;
}

void BinaryFileStream::close() noexcept {
    if (_file) { LUISA_FCLOSE(_file); }
    _length = 0;
    _pos = 0;
}

#undef LUISA_FOPEN
#undef LUISA_FSEEK
#undef LUISA_FTELL
#undef LUISA_FREAD
#undef LUISA_FCLOSE

}// namespace luisa
