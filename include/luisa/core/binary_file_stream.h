#pragma once

#include <luisa/core/binary_io.h>
#include <luisa/core/platform.h>
#include <cstdio>

namespace luisa {

namespace detail {

#if defined(LUISA_PLATFORM_WINDOWS) || defined(_WIN32)
#define LUISA_FSEEK _fseeki64
#define LUISA_FTELL _ftelli64
#else
#define LUISA_FSEEK fseeko
#define LUISA_FTELL ftello
#endif

/// Byte length of an open C file stream, or 0 on failure.
///
/// Inlined on purpose instead of exported from luisa-core: every module of the
/// runtime links the MSVC CRT statically (lc_win_runtime=MT), and a FILE* - plus
/// the file descriptor behind it - only means something inside the CRT instance
/// that created it. Asking another module to fseek/ftell a foreign stream looks the
/// descriptor up in the wrong table, which trips the UCRT invalid-parameter handler
/// (a fastfail, 0xC0000409, in debug builds) and silently returns a bogus length in
/// release ones. Inline keeps the operations in the caller's own runtime, i.e. in
/// the module that opened the file.
[[nodiscard]] inline size_t get_c_file_length(::FILE *file) noexcept {
    LUISA_FSEEK(file, 0, SEEK_END);
    auto length = LUISA_FTELL(file);
    LUISA_FSEEK(file, 0, SEEK_SET);
    return length < 0 ? 0u : static_cast<size_t>(length);
}

#undef LUISA_FSEEK
#undef LUISA_FTELL

}// namespace detail

/// Read-only binary stream over a C file stream.
///
/// Open it from a path: the FILE* is then created and consumed inside luisa-core,
/// so a stream handed to another module never carries a foreign descriptor.
class LUISA_CORE_API BinaryFileStream : public BinaryStream {
private:
    ::FILE *_file{nullptr};
    size_t _length{0u};
    size_t _pos{0u};

public:
    explicit BinaryFileStream(const luisa::string &path) noexcept;
    /// Wraps an already-open stream. The FILE* and the length must come from the
    /// SAME module that will construct this object: `read`/`close` below run inside
    /// luisa-core, and on Windows every module links the MSVC runtime statically
    /// (`lc_win_runtime`), so a descriptor opened elsewhere is not valid there.
    /// Prefer the path constructor.
    explicit BinaryFileStream(::FILE *file, size_t length) noexcept;
    ~BinaryFileStream() noexcept override;
    BinaryFileStream(BinaryFileStream &&another) noexcept;
    BinaryFileStream &operator=(BinaryFileStream &&rhs) noexcept;
    BinaryFileStream(const BinaryFileStream &) noexcept = delete;
    BinaryFileStream &operator=(const BinaryFileStream &) noexcept = delete;
    [[nodiscard]] auto valid() const noexcept { return _file != nullptr; }
    [[nodiscard]] explicit operator bool() const noexcept { return valid(); }
    [[nodiscard]] size_t length() const noexcept override { return _length; }
    [[nodiscard]] size_t pos() const noexcept override { return _pos; }
    void read(luisa::span<std::byte> dst) noexcept override;
    using BinaryStream::read;//猜猜为啥需要这一行？
    void set_pos(size_t pos) noexcept;
    void close() noexcept;
};

}// namespace luisa
