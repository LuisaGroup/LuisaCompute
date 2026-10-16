#include <luisa/vstl/string_utility.h>
#include <luisa/core/stl/string.h>
namespace vstd {

char StringUtil::to_lower(char c) {
    if ((c >= 'A') && (c <= 'Z'))
        return static_cast<char>(c + ('a' - 'A'));
    return c;
}
char StringUtil::to_upper(char c) {
    if ((c >= 'a') && (c <= 'z'))
        return static_cast<char>(c + ('A' - 'a'));
    return c;
}

void StringUtil::to_lower(string &str) {
    char *c = str.data();
    const uint size = str.length();
    for (uint i = 0; i < size; ++i) {
        c[i] = to_lower(c[i]);
    }
}
void StringUtil::to_upper(string &str) {
    char *c = str.data();
    const uint size = str.length();
    for (uint i = 0; i < size; ++i) {
        c[i] = to_upper(c[i]);
    }
}

string StringUtil::to_lower(luisa::string_view str) {
    string s;
    s.resize(str.size());
    for (size_t i = 0; i < str.size(); ++i) {
        auto &&v = s[i];
        v = str[i];
        v = to_lower(v);
    }
    return s;
}
string StringUtil::to_upper(luisa::string_view str) {
    string s;
    s.resize(str.size());
    for (size_t i = 0; i < str.size(); ++i) {
        auto &&v = s[i];
        v = str[i];
        v = to_upper(v);
    }
    return s;
}
luisa::string_view CharSplitIterator::operator*() const {
    return result;
}
void CharSplitIterator::operator++() {
    char const *start = curPtr;
    while (curPtr != endPtr) {
        if (*curPtr == sign) {
            if (start == curPtr) {
                ++curPtr;
                start = curPtr;
                continue;
            }
            result = luisa::string_view(start, curPtr - start);
            ++curPtr;
            return;
        }
        ++curPtr;
    }
    if (endPtr == start) {
        result = luisa::string_view(nullptr, 0);
    } else {
        result = luisa::string_view(start, endPtr - start);
    }
}
bool CharSplitIterator::operator==(IteEndTag) const {
    return result.size() == 0;
}

luisa::string_view StrVSplitIterator::operator*() const {
    return result;
}
void StrVSplitIterator::operator++() {
    auto IsSame = [&](char const *ptr) {
        auto sz = endPtr - ptr;
        if (sz < sign.size()) return false;
        luisa::string_view value(ptr, sign.size());
        return value == sign;
    };
    char const *start = curPtr;
    while (curPtr < endPtr) {
        if (IsSame(curPtr)) {
            if (start == curPtr) {
                curPtr += sign.size();
                start = curPtr;
                continue;
            }
            result = luisa::string_view(start, curPtr - start);
            curPtr += sign.size();
            return;
        }
        ++curPtr;
    }
    if (endPtr == start) {
        result = luisa::string_view(nullptr, 0);
    } else {
        result = luisa::string_view(start, endPtr - start);
    }
}
bool StrVSplitIterator::operator==(IteEndTag) const {
    return result.size() == 0;
}
namespace strutil_detail {

// constexpr causes error on ubuntu-24.04-arm
static constexpr auto NA = static_cast<char>(-1);
static const char tab[] = {
    NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,//   0NA5
    NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,//  16-31
    NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, 62, NA, NA, NA, 63,//  32-47
    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, NA, NA, NA, NA, NA, NA,//  48-63
    NA, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,          //  64-79
    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, NA, NA, NA, NA, NA,//  80-95
    NA, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,//  96NA11
    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, NA, NA, NA, NA, NA,// 112NA27
    NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,// 128NA43
    NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,// 144NA59
    NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,// 160NA75
    NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,// 176NA91
    NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,// 192-207
    NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,// 208-223
    NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,// 224-239
    NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA // 240-255
};

// constexpr causes error on ubuntu-24.04-arm
static const char tab1[] = {
    "ABCDEFGHIJKLMNOP"
    "QRSTUVWXYZabcdef"
    "ghijklmnopqrstuv"
    "wxyz0123456789+/"};

char const *get_inverse() {
    return &tab[0];
}

char const *get_alphabet() {
    return &tab1[0];
}

size_t encode(void *dest, void const *src, size_t len) {
    char *out = static_cast<char *>(dest);
    char const *in = static_cast<char const *>(src);
    auto const tab = get_alphabet();

    for (auto n = len / 3; n--;) {
        *out++ = tab[(in[0] & 0xfc) >> 2];
        *out++ = tab[((in[0] & 0x03) << 4) + ((in[1] & 0xf0) >> 4)];
        *out++ = tab[((in[2] & 0xc0) >> 6) + ((in[1] & 0x0f) << 2)];
        *out++ = tab[in[2] & 0x3f];
        in += 3;
    }

    switch (len % 3) {
        case 2:
            *out++ = tab[(in[0] & 0xfc) >> 2];
            *out++ = tab[((in[0] & 0x03) << 4) + ((in[1] & 0xf0) >> 4)];
            *out++ = tab[(in[1] & 0x0f) << 2];
            *out++ = '=';
            break;

        case 1:
            *out++ = tab[(in[0] & 0xfc) >> 2];
            *out++ = tab[((in[0] & 0x03) << 4)];
            *out++ = '=';
            *out++ = '=';
            break;

        case 0:
            break;
    }

    return out - static_cast<char *>(dest);
}

std::pair<size_t, size_t> decode(void *dest, char const *src, size_t len) {
    char *out = static_cast<char *>(dest);
    auto in = reinterpret_cast<unsigned char const *>(src);
    unsigned char c3[3], c4[4];
    int i = 0;
    int j = 0;

    auto const inverse = get_inverse();

    while (len-- && *in != '=') {
        auto const v = inverse[*in];
        if (v == -1)
            break;
        ++in;
        c4[i] = v;
        if (++i == 4) {
            c3[0] = (c4[0] << 2) + ((c4[1] & 0x30) >> 4);
            c3[1] = ((c4[1] & 0xf) << 4) + ((c4[2] & 0x3c) >> 2);
            c3[2] = ((c4[2] & 0x3) << 6) + c4[3];

            for (i = 0; i < 3; i++)
                *out++ = static_cast<char>(c3[i]);
            i = 0;
        }
    }

    if (i) {
        c3[0] = (c4[0] << 2) + ((c4[1] & 0x30) >> 4);
        c3[1] = ((c4[1] & 0xf) << 4) + ((c4[2] & 0x3c) >> 2);
        c3[2] = ((c4[2] & 0x3) << 6) + c4[3];

        for (j = 0; j < i - 1; j++)
            *out++ = static_cast<char>(c3[j]);
    }

    return {out - static_cast<char *>(dest),
            in - reinterpret_cast<unsigned char const *>(src)};
}
size_t constexpr encoded_size(size_t n) {
    return 4 * ((n + 2) / 3);
}

/// Returns max bytes needed to decode a base64 string
size_t constexpr decoded_size(size_t n) {
    return n / 4 * 3;// requires n&3==0, smaller
}

}// namespace strutil_detail
void StringUtil::to_base64(span<uint8_t const> binary, string &result) {
    using namespace strutil_detail;
    size_t oriSize = result.size();
    result.resize(oriSize + encoded_size(binary.size()));
    encode(result.data() + oriSize, binary.data(), binary.size());
}
void StringUtil::to_base64(span<uint8_t const> binary, char *result) {
    using namespace strutil_detail;
    encode(result, binary.data(), binary.size());
}

void StringUtil::from_base64(luisa::string_view str, vector<uint8_t> &result) {
    using namespace strutil_detail;
    size_t oriSize = result.size();
    result.reserve(oriSize + decoded_size(str.size()));
    auto destAndSrcSize = decode(result.data() + oriSize, str.data(), str.size());
    result.resize(oriSize + destAndSrcSize.first);
}

void StringUtil::from_base64(luisa::string_view str, uint8_t *size) {
    using namespace strutil_detail;
    decode(size, str.data(), str.size());
}
void StringUtil::to_hex_string(span<uint8_t const> binary, string &result, bool upper) {
    result.clear();
    result.resize(binary.size() * 2);
    to_hex_string(binary, result.data(), upper);
}
void StringUtil::to_hex_string(span<uint8_t const> binary, char *result, bool upper) {
    static char const *const hexUpperStr = upper ? "0123456789ABCDEF" : "0123456789abcdef";
    for (auto i : binary) {
        result[0] = hexUpperStr[(i >> 4) & 15];
        result[1] = hexUpperStr[i & 15];
        result += 2;
    }
}
}// namespace vstd
