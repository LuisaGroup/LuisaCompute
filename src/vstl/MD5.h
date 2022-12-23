#pragma once
#include <vstl/common.h>
namespace vstd {
static constexpr size_t MD5_SIZE = 16;

LC_VSTL_API std::array<uint8_t, MD5_SIZE> GetMD5FromString(string const &str);
LC_VSTL_API std::array<uint8_t, MD5_SIZE> GetMD5FromArray(eastl::span<uint8_t> data);
//Used for unity
class LC_VSTL_API MD5 {
public:
    struct MD5Data {
        uint64 data0;
        uint64 data1;
    };

private:
    MD5Data data;

public:
    MD5Data const &ToBinary() const { return data; }
    MD5() {}
    MD5(string const &str);
    MD5(vstd::string_view str);
    MD5(vstd::span<uint8_t const> bin);
    MD5(MD5 const &) = default;
    MD5(MD5 &&) = default;
    MD5(MD5Data const &data);
    string ToString(bool upper = true) const;
    template<typename T>
    MD5 &operator=(T &&t) {
        this->~MD5();
        new (this) MD5(std::forward<T>(t));
        return *this;
    }
    ~MD5() = default;
    MD5 &operator=(MD5 const &) = default;
    MD5 &operator=(MD5 &&) = default;
    bool operator==(MD5 const &m) const;
    bool operator!=(MD5 const &m) const;
};
template<>
struct hash<MD5::MD5Data> {
    size_t operator()(MD5::MD5Data const &m) const {
        return Hash::CharArrayHash(&m.data0, sizeof(MD5::MD5Data));
    }
};
template<>
struct hash<MD5> {
    size_t operator()(MD5 const &m) const {
        static hash<MD5::MD5Data> dataHasher;
        return dataHasher(m.ToBinary());
    }
};

}// namespace vstd