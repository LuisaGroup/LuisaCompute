#pragma once
#include <vstl/common.h>
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
    static StrvIEnumerator<char, CharSplitIterator> split(std::string_view str, char sign) {
        return {str.data(), str.data() + str.size(), sign};
    }
    static StrvIEnumerator<vstd::string_view, StrVSplitIterator> split(std::string_view str, vstd::string_view signs) {
        return {str.data(), str.data() + str.size(), signs};
    }

    static void to_lower(string &str);
    static void to_upper(string &str);
    static char to_upper(char c);
    static char to_lower(char c);
    static string to_lower(std::string_view str);
    static string to_upper(std::string_view str);
    static void to_base64(span<uint8_t const> binary, string &result);
    static void to_base64(span<uint8_t const> binary, char *result);
    static void from_base64(std::string_view str, vector<uint8_t> &result);
    static void from_base64(std::string_view str, uint8_t *size);
};
}// namespace vstd