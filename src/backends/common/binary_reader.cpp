#include <backends/common/binary_reader.h>
namespace luisa::compute {
#ifdef _WIN32
#define LC_FSEEK _fseeki64_nolock
#define LC_FTELL _ftelli64_nolock
#define LC_FREAD _fread_nolock
#define LC_FCLOSE _fclose_nolock
#else
#define LC_FSEEK fseek
#define LC_FTELL ftell
#define LC_FREAD fread
#define LC_FCLOSE fclose
#endif
BinaryReader::BinaryReader(luisa::string const &path) noexcept {
    _pos = 0;
    _ifs = fopen(path.c_str(), "rb");
    if (_ifs) {
        LC_FSEEK(_ifs, 0, SEEK_END);
        _length = LC_FTELL(_ifs);
        LC_FSEEK(_ifs, 0, SEEK_SET);
    } else {
        _length = 0;
    }
}
void BinaryReader::read(luisa::span<std::byte> dst) noexcept {
    if (!_ifs) return;
    auto len = std::min<size_t>(_length - _pos, dst.size());
    if (len == 0) return;
    LC_FREAD(dst.data(), len, 1, _ifs);
    _pos += len;
}
BinaryReader::~BinaryReader() noexcept {
    if (!_ifs) return;
    LC_FCLOSE(_ifs);
}
BinaryReader::BinaryReader(BinaryReader &&rhs) noexcept
    : _ifs(rhs._ifs),
      _length(rhs._length),
      _pos(rhs._pos) {
    rhs._ifs = nullptr;
}
#undef LC_FSEEK
#undef LC_FTELL
#undef LC_FREAD
#undef LC_FCLOSE
}// namespace luisa::compute