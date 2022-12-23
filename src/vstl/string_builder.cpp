#include "string_builder.h"
namespace vstd {
StringBuilder::StringBuilder(vstd::string *target)
    : target(target) {
}
StringBuilder::~StringBuilder() {
    auto lastSize = target->size();
    target->resize(lastSize + size);
    auto ptr = target->data() + lastSize;
    for (auto &&i : views) {
        i.visit(
            [&]<typename T>(T const &c) {
                if constexpr (std::is_same_v<T, char>) {
                    *ptr = c;
                    ++ptr;
                } else {
                    memcpy(ptr, c.data(), c.size());
                    ptr += c.size();
                }
            });
    }
}
StringBuilder &StringBuilder::operator<<(vstd::string_view str) {
    size += str.size();
    views.emplace_back(str);
    return *this;
}
StringBuilder &StringBuilder::operator<<(char str) {
    size += 1;
    views.emplace_back(str);
    return *this;
}
StringBuilder &StringBuilder::operator<<(vstd::string &&str) {
    size += str.size();
    views.emplace_back(std::move(str));
    return *this;
}
StringBuilder &StringBuilder::operator<<(vstd::string const &str) {
    size += str.size();
    views.emplace_back(vstd::string_view(str));
    return *this;
}
}// namespace vstd