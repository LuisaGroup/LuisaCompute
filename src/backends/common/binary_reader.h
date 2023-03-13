#pragma once
#include <core/stl/vector.h>
#include <core/binary_io.h>
namespace luisa::compute {
class BinaryReader : public IBinaryStream {
private:
    FILE *_ifs;
    size_t _length;
    size_t _pos;

public:
    BinaryReader(luisa::string const &path) noexcept;
    void read(luisa::span<std::byte> dst) noexcept;
    [[nodiscard]] operator bool() const noexcept {
        return _ifs;
    }
    [[nodiscard]] size_t pos() const noexcept override { return _pos; }
    [[nodiscard]] size_t length() const noexcept override { return _length; }
    ~BinaryReader() noexcept;
    BinaryReader(BinaryReader const &) = delete;
    BinaryReader(BinaryReader &&rhs) noexcept;
    BinaryReader &operator=(BinaryReader const &) = delete;
    BinaryReader &operator=(BinaryReader &&rhs) noexcept {
        this->~BinaryReader();
        new (std::launder(this)) BinaryReader(std::move(rhs));
        return *this;
    }
};
}// namespace luisa::compute