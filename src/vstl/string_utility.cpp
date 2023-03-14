#include <vstl/string_utility.h>
namespace vstd {

char StringUtil::to_lower(char c) {
    if ((c >= 'A') && (c <= 'Z'))
        return c + ('a' - 'A');
    return c;
}
char StringUtil::to_upper(char c) {
    if ((c >= 'a') && (c <= 'z'))
        return c + ('A' - 'a');
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

string StringUtil::to_lower(std::string_view str) {
    string s;
    s.resize(str.size());
    for (auto i : range(str.size())) {
        auto &&v = s[i];
        v = str[i];
        v = to_lower(v);
    }
    return s;
}
string StringUtil::to_upper(std::string_view str) {
    string s;
    s.resize(str.size());
    for (auto i : range(str.size())) {
        auto &&v = s[i];
        v = str[i];
        v = to_upper(v);
    }
    return s;
}
std::string_view CharSplitIterator::operator*() const {
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
            result = std::string_view(start, curPtr - start);
            ++curPtr;
            return;
        }
        ++curPtr;
    }
    if (endPtr == start) {
        result = std::string_view(nullptr, 0);
    } else {
        result = std::string_view(start, endPtr - start);
    }
}
bool CharSplitIterator::operator==(IteEndTag) const {
    return result.size() == 0;
}

std::string_view StrVSplitIterator::operator*() const {
    return result;
}
void StrVSplitIterator::operator++() {
    auto IsSame = [&](char const *ptr) {
        auto sz = endPtr - ptr;
        if (sz < sign.size()) return false;
        std::string_view value(ptr, sign.size());
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
            result = std::string_view(start, curPtr - start);
            curPtr += sign.size();
            return;
        }
        ++curPtr;
    }
    if (endPtr == start) {
        result = std::string_view(nullptr, 0);
    } else {
        result = std::string_view(start, endPtr - start);
    }
}
bool StrVSplitIterator::operator==(IteEndTag) const {
    return result.size() == 0;
}
namespace strutil_detail {
static char constexpr tab[] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,//   0-15
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,//  16-31
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 62, -1, -1, -1, 63,//  32-47
    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, -1, -1, -1, -1, -1, -1,//  48-63
    -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,          //  64-79
    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, -1, -1, -1, -1, -1,//  80-95
    -1, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,//  96-111
    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, -1, -1, -1, -1, -1,// 112-127
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,// 128-143
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,// 144-159
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,// 160-175
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,// 176-191
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,// 192-207
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,// 208-223
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,// 224-239
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 // 240-255
};
static char constexpr tab1[] = {
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
                *out++ = c3[i];
            i = 0;
        }
    }

    if (i) {
        c3[0] = (c4[0] << 2) + ((c4[1] & 0x30) >> 4);
        c3[1] = ((c4[1] & 0xf) << 4) + ((c4[2] & 0x3c) >> 2);
        c3[2] = ((c4[2] & 0x3) << 6) + c4[3];

        for (j = 0; j < i - 1; j++)
            *out++ = c3[j];
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
void StringUtil::to_base64(span<uint8_t const> binary, string &str) {
    using namespace strutil_detail;
    size_t oriSize = str.size();
    str.resize(oriSize + encoded_size(binary.size()));
    encode(str.data() + oriSize, binary.data(), binary.size());
}
void StringUtil::to_base64(span<uint8_t const> binary, char *result) {
    using namespace strutil_detail;
    encode(result, binary.data(), binary.size());
}

void StringUtil::from_base64(std::string_view str, vector<uint8_t> &bin) {
    using namespace strutil_detail;
    size_t oriSize = bin.size();
    bin.reserve(oriSize + decoded_size(str.size()));
    auto destAndSrcSize = decode(bin.data() + oriSize, str.data(), str.size());
    bin.resize(oriSize + destAndSrcSize.first);
}

void StringUtil::from_base64(std::string_view str, uint8_t *size) {
    using namespace strutil_detail;
    decode(size, str.data(), str.size());
}

}// namespace vstd
