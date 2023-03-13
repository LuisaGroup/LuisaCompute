#pragma once

#include <core/binary_io.h>

namespace luisa {

class LC_CORE_API BinaryFileStream : public BinaryStream {

private:
    ::FILE *_file{nullptr};
    size_t _length{0u};
    size_t _pos{0u};

public:
    explicit BinaryFileStream(const luisa::string &path) noexcept;
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
    void close() noexcept;
};

}// namespace luisa
