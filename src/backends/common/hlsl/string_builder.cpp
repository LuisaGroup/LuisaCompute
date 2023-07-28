#include "string_builder.h"
namespace vstd {
StringBuilder::~StringBuilder() = default;
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
StringBuilder::StringBuilder() = default;

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

