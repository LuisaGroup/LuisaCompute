#pragma vengine_package vengine_dll

#include <util/VGuid.h>

#ifdef _WIN32
#include <objbase.h>
#elif defined(__linux__) || defined(__unix__)
#include <uuid/uuid.h>
#elif defined(__APPLE__)
#include <CoreFoundation/CFUUID.h>
#endif

namespace vstd {
Guid::Guid(bool generate) {
    if (generate) {
        ReGenerate();
    } else {
        memset(&data, 0, sizeof(GuidData));
    }
}

Guid::Guid(GuidData const &d) {
    memcpy(&data, &d, sizeof(GuidData));
}
Guid::Guid(std::string_view strv) {
    if (strv.size() != sizeof(GuidData) * 2) {
        vstl_log("Wrong guid string length!\n");
        VSTL_ABORT();
    }
    char const *ptr = &*strv.begin();
    auto toHex = [&]() {
        uint64 v = 0;
        auto GetNumber = [](char c) {
            switch (c) {
                case '0': return 0;
                case '1': return 1;
                case '2': return 2;
                case '3': return 3;
                case '4': return 4;
                case '5': return 5;
                case '6': return 6;
                case '7': return 7;
                case '8': return 8;
                case '9': return 9;
                case 'a': return 10;
                case 'b': return 11;
                case 'c': return 12;
                case 'd': return 13;
                case 'e': return 14;
                case 'f': return 15;
                case 'A': return 10;
                case 'B': return 11;
                case 'C': return 12;
                case 'D': return 13;
                case 'E': return 14;
                case 'F': return 15;
                default: VSTL_ABORT();
            }
        };
        auto endPtr = ptr + sizeof(uint64) * 2;
        int index = 0;
        while (ptr != endPtr) {
            v <<= 4;
            v |= GetNumber(*ptr);
            ++ptr;
        }
        ptr = endPtr;
        return v;
    };
    data.data0 = toHex();
    data.data1 = toHex();
}

Guid::Guid(std::span<uint8_t> data) {
    if (data.size() != sizeof(GuidData) * 2) {
        vstl_log("Wrong guid string length!\n");
        VSTL_ABORT();
    }
    memcpy(&this->data, data.data(), sizeof(GuidData));
}
Guid::Guid(std::array<uint8_t, sizeof(GuidData)> const &data) {
    memcpy(&this->data, data.data(), sizeof(GuidData));
}

std::string Guid::ToCompressedString() const {
    std::string result;
    result.resize(20);
    ToCompressedString(result.data());
    return result;
}
void Guid::ToCompressedString(char *result) const {
    static std::string_view strv = "`1234567890-=qwertyuiop[]asdfghjkl;zxcvbnm,.~!@#$%^&*()_+QWERTYUIOP{}|ASDFGHJKL:ZXCVBNM<>?";
    size_t index = 0;
    auto AddName = [&](uint64 v, uint64 tarByte) {
        while (v > 0) {
            auto bit = v % strv.size();
            result[index] = strv[bit];
            index++;
            v /= strv.size();
        }
        for (; index < tarByte; ++index) {
            result[index] = strv[0];
        }
    };
    AddName(data.data0, 10);
    AddName(data.data1, 20);
}

void Guid::ReGenerate() {
#ifdef _WIN32
    static_assert(sizeof(data) == sizeof(_GUID), "Size mismatch");
    HRESULT h = ::CoCreateGuid(reinterpret_cast<_GUID *>(&data));
#ifdef VSTL_DEBUG
    if (h != S_OK) {
        vstl_log("GUID Generate Failed!\n");
        VSTL_ABORT();
    }
#endif
#elif defined(__linux__) || defined(__unix__)
    static_assert(sizeof(data) == sizeof(uuid_t), "Size mismatch");
    uuid_generate(reinterpret_cast<uuid_t &>(data));
#elif defined(__APPLE__)
    auto newId = CFUUIDCreate(NULL);
    auto bytes = CFUUIDGetUUIDBytes(newId);
    static_assert(sizeof(data) == sizeof(bytes), "Size mismatch");
    memcpy(&data, &bytes, sizeof(data));
    CFRelease(newId);
#endif
}
std::array<uint8_t, sizeof(Guid::GuidData)> Guid::ToArray() const {
    std::array<uint8_t, sizeof(GuidData)> arr;
    memcpy(arr.data(), &data, sizeof(GuidData));
    return arr;
}
namespace vguid_detail {
void toHex(uint64 data, char *&sPtr, bool upper) {
    char const *hexUpperStr = upper ? "0123456789ABCDEF" : "0123456789abcdef";
    constexpr size_t hexSize = sizeof(data) * 2;
    auto ptrEnd = sPtr - hexSize;
    while (sPtr != ptrEnd) {
        *sPtr = hexUpperStr[data & 15];
        data >>= 4;
        sPtr--;
    }
}
}// namespace vguid_detail
std::string Guid::ToString(bool upper) const {
    std::string s;
    s.resize(sizeof(GuidData) * 2);
    auto sPtr = s.data() + sizeof(GuidData) * 2 - 1;
    vguid_detail::toHex(data.data1, sPtr, upper);
    vguid_detail::toHex(data.data0, sPtr, upper);
    return s;
}
void Guid::ToString(char *result, bool upper) const {
    auto sPtr = result + sizeof(GuidData) * 2 - 1;
    vguid_detail::toHex(data.data1, sPtr, upper);
    vguid_detail::toHex(data.data0, sPtr, upper);
}
std::ostream &operator<<(std::ostream &out, const Guid &obj) noexcept {
    out << obj.ToString();
    return out;
}

}// namespace vstd