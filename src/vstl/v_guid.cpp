#include <vstl/v_guid.h>
#include <vstl/string_utility.h>

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
        remake();
    } else {
        memset(&data, 0, sizeof(GuidData));
    }
}

Guid::Guid(GuidData const &d) {
    memcpy(&data, &d, sizeof(GuidData));
}
namespace VGuid_Detail {
int32 GetNumber(char c) {
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
        default: return 15;
    }
};
void ParseHex(std::string_view strv, Guid::GuidData &data) {
    char const *ptr = strv.data();
    auto toHex = [&]() {
        uint64 v = 0;
        auto endPtr = ptr + sizeof(uint64) * 2;
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
}// namespace VGuid_Detail
optional<Guid> Guid::TryParseGuid(std::string_view strv) {
    using namespace VGuid_Detail;
    switch (strv.size()) {
        case 22: {
            Guid opt;
            StringUtil::from_base64(strv, reinterpret_cast<uint8_t *>(&opt.data));
            return {opt};
        }
        case 32: {
            Guid opt;
            ParseHex(strv, opt.data);
            return {opt};
        }
    }
    return {};
}
Guid::Guid(std::string_view strv) {
    using namespace VGuid_Detail;
    switch (strv.size()) {
        case 22:
            StringUtil::from_base64(strv, reinterpret_cast<uint8_t *>(&data));
            break;
        case 32: {
            ParseHex(strv, data);
        } break;
        default:
            vengine_log("Wrong guid string length!\n");
            VENGINE_EXIT;
    }
}

Guid::Guid(span<uint8_t> data) {
    if (data.size() != sizeof(GuidData) * 2) {
        vengine_log("Wrong guid string length!\n");
        VENGINE_EXIT;
    }
    memcpy(&this->data, data.data(), sizeof(GuidData));
}
Guid::Guid(std::array<uint8_t, sizeof(GuidData)> const &data) {
    memcpy(&this->data, data.data(), sizeof(GuidData));
}

string Guid::to_base64() const {
    string result;
    StringUtil::to_base64({reinterpret_cast<uint8_t const *>(&data), sizeof(data)}, result);
    result.resize(result.size() - 2);
    return result;
}

void Guid::to_base64(char *result) const {
    StringUtil::to_base64({reinterpret_cast<uint8_t const *>(&data), sizeof(data)}, result);
}

void Guid::remake() {
#ifdef _WIN32
    static_assert(sizeof(data) == sizeof(_GUID), "Size mismatch");
#ifndef NDEBUG
    HRESULT h = ::CoCreateGuid(reinterpret_cast<_GUID *>(&data));
    if (h != S_OK) {
        vengine_log("GUID Generate Failed!\n"_sv);
        VENGINE_EXIT;
    }
#else
    ::CoCreateGuid(reinterpret_cast<_GUID *>(&data));
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
string Guid::to_string(bool upper) const {
    string s;
    s.resize(sizeof(GuidData) * 2);
    auto sPtr = s.data() + sizeof(GuidData) * 2 - 1;
    vguid_detail::toHex(data.data1, sPtr, upper);
    vguid_detail::toHex(data.data0, sPtr, upper);
    return s;
}
void Guid::to_string(char *result, bool upper) const {
    auto sPtr = result + sizeof(GuidData) * 2 - 1;
    vguid_detail::toHex(data.data1, sPtr, upper);
    vguid_detail::toHex(data.data0, sPtr, upper);
}

#ifdef EXPORT_UNITY_FUNCTION
VENGINE_UNITY_EXTERN void vguid_get_new(
    Guid *guidData) {
    *guidData = Guid(true).to_binary();
}
VENGINE_UNITY_EXTERN void vguid_get_from_std::string(
    char const *str,
    int32 strLen,
    Guid *guidData) {
    *guidData = Guid(std::string_view(str, strLen)).to_binary();
}
VENGINE_UNITY_EXTERN void vguid_to_std::string(
    Guid const *guidData,
    char *result,
    bool upper) {
    guidData->to_string(result, upper);
}
#endif
}// namespace vstd
