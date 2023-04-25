#include "string_builder.h"
namespace vstd {
StringBuilder::~StringBuilder() {}
StringBuilder &StringBuilder::append(vstd::string_view str) {
    vstd::push_back_all(vec, str.data(), str.size());
    return *this;
}
StringBuilder &StringBuilder::append(char str) {
    vec.push_back(str);
    return *this;
}
StringBuilder &StringBuilder::append(vstd::string const &str) {
    vstd::push_back_all(vec, str.data(), str.size());
    return *this;
}
StringBuilder::StringBuilder() {}
inline void _float_str_resize(size_t lastSize, StringBuilder &str) noexcept {
    for (int64_t i = str.size() - 1; i >= lastSize; --i) {
        if (str[i] == '.') [[unlikely]] {
            auto end = i + 2;
            int64_t j = str.size() - 1;
            for (; j >= end; --j) {
                if (str[j] != '0') {
                    break;
                }
            }
            str.resize(j + 1);
            return;
        }
    }
    str << ".0"sv;
}
void to_string(float Val, StringBuilder &str) noexcept {
    const size_t len = snprintf(nullptr, 0, "%a", Val);
    auto lastLen = str.size();
    str.push_back(len + 1);
    auto iter = str.end() - 1;
    *iter = 0;
    snprintf(str.data() + lastLen, len + 1, "%a", Val);
    *iter = 'f';
}
void to_string(double Val, StringBuilder &str) noexcept {
    const size_t len = snprintf(nullptr, 0, "%a", Val);
    auto lastLen = str.size();
    str.push_back(len + 1);
    auto iter = str.end() - 1;
    *iter = 0;
    snprintf(str.data() + lastLen, len + 1, "%a", Val);
    str.erase(str.end() - 1);
}

}// namespace vstd
