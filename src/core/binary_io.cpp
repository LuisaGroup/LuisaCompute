//
// Created by Mike on 3/13/2023.
//

#include <core/logging.h>
#include <core/binary_io.h>

namespace luisa {

void BinaryStringStream::read(luisa::span<std::byte> dst) {
    auto size = std::min(dst.size(), _data.size() - _pos);
    std::memcpy(dst.data(), _data.data() + _pos, size);
    _pos += size;
}

#ifdef _WIN32
#define LUISA_FOPEN fopen
#define LUISA_FSEEK _fseeki64_nolock
#define LUISA_FTELL _ftelli64_nolock
#define LUISA_FREAD _fread_nolock
#define LUISA_FWRITE _fwrite_nolock
#define LUISA_FCLOSE _fclose_nolock
#else
#define LUISA_FOPEN fopen
#define LUISA_FSEEK fseek
#define LUISA_FTELL ftell
#define LUISA_FREAD fread
#define LUISA_FWRITE fwrite
#define LUISA_FCLOSE fclose
#endif

BinaryFileStream::BinaryFileStream(luisa::string_view path) noexcept {
    _file = LUISA_FOPEN(path.data(), "rb");
    if (_file == nullptr) { LUISA_WARNING_WITH_LOCATION(
        "Failed to create file stream for file '{}'.", path); }
}

BinaryFileStream::~BinaryFileStream() noexcept {
    if (_file) { LUISA_FCLOSE(_file); }
}

size_t BinaryFileStream::length() const {
    auto pos = LUISA_FTELL(_file);
    LUISA_FSEEK(_file, 0, SEEK_END);
    auto length = LUISA_FTELL(_file);
    LUISA_FSEEK(_file, pos, SEEK_SET);
    return length;
}

size_t BinaryFileStream::pos() const { return LUISA_FTELL(_file); }

void BinaryFileStream::read(luisa::span<std::byte> dst) {
    LUISA_FREAD(dst.data(), 1, dst.size(), _file);
}

#undef LUISA_FOPEN
#undef LUISA_FSEEK
#undef LUISA_FTELL
#undef LUISA_FREAD
#undef LUISA_FWRITE
#undef LUISA_FCLOSE

}// namespace luisa
