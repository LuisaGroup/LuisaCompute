#pragma once
#include <vstl/common.h>
#include <span>
namespace vstd {

struct LC_VSTL_API CharSplitIterator {
    char const *curPtr;
    char const *endPtr;
    char sign;
    std::string_view result;
    std::string_view operator*() const;
    void operator++();
    bool operator==(IteEndTag) const;
};
struct LC_VSTL_API StrVSplitIterator {
    char const *curPtr;
    char const *endPtr;
    std::string_view sign;
    std::string_view result;
    std::string_view operator*() const;
    void operator++();
    bool operator==(IteEndTag) const;
};
template<typename SignT, typename IteratorType>
struct StrvIEnumerator {
    char const *curPtr;
    char const *endPtr;
    SignT sign;
    IteratorType begin() const {
        IteratorType c{curPtr, endPtr, sign};
        ++c;
        return c;
    }
    IteEndTag end() const {
        return {};
    }
};
class LC_VSTL_API StringUtil {
private:
    StringUtil() = delete;
    KILL_COPY_CONSTRUCT(StringUtil)
public:
    static StrvIEnumerator<char, CharSplitIterator> Split(std::string_view str, char sign) {
        return {str.data(), str.data() + str.size(), sign};
    }
    static StrvIEnumerator<vstd::string_view, StrVSplitIterator> Split(std::string_view str, vstd::string_view signs) {
        return {str.data(), str.data() + str.size(), signs};
    }
    static int64 GetFirstIndexOf(std::string_view str, char sign);

    static variant<int64, double> StringToNumber(std::string_view numStr);
    static void ToLower(string &str);
    static void ToUpper(string &str);

    static string ToLower(std::string_view str);
    static string ToUpper(std::string_view str);
    static void EncodeToBase64(std::span<uint8_t const> binary, string &result);
    static void EncodeToBase64(std::span<uint8_t const> binary, char *result);
    static void DecodeFromBase64(std::string_view str, vector<uint8_t> &result);
    static void DecodeFromBase64(std::string_view str, uint8_t *size);
    static void TransformWCharToChar(
        wchar_t const *src,
        char *dst,
        size_t sz);
};
}// namespace vstd