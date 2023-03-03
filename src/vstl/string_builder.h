#pragma once
#include "common.h"
#include <vstl/small_vector.h>
namespace vstd {
class LC_VSTL_API StringBuilder final {
    vstd::fixed_vector<char, 32> vec;

public:
    void clear() { vec.clear(); }
    bool operator==(StringBuilder const &rhs) const {
        return view() == rhs.view();
    }
    bool operator!=(StringBuilder const &rhs) const {
        return view() != rhs.view();
    }
    bool operator==(string_view rhs) const {
        return view() == rhs;
    }
    bool operator!=(string_view rhs) const {
        return view() != rhs;
    }
    char const *data() const { return vec.data(); }
    char *data() { return vec.data(); }
    char const &operator[](size_t i) const { return vec[i]; }
    char &operator[](size_t i) { return vec[i]; }
    void resize(size_t size) { vec.resize_uninitialized(size); }
    void push_back(size_t size) { vec.push_back_uninitialized(size); }
    void reserve(size_t capacity) { vec.reserve(capacity); }
    size_t size() const { return vec.size(); }

    vstd::string_view view() const { return {vec.data(), vec.size()}; }
    StringBuilder &append(vstd::string_view str);
    StringBuilder &append(char const *str) {
        return append(string_view{str});
    }
    StringBuilder &append(char str);
    StringBuilder &append(vstd::string const &str);
    StringBuilder &append(StringBuilder const &str) {
        return append(str.view());
    }
    template<typename T>
    StringBuilder &operator<<(T &&t) {
        return append(std::forward<T>(t));
    }
    template<typename T>
    void operator+=(T &&t) {
        append(std::forward<T>(t));
    }
    StringBuilder();
    ~StringBuilder();
    StringBuilder(StringBuilder const &) = delete;
    StringBuilder(StringBuilder &&) = default;
    StringBuilder &operator=(StringBuilder const &) = delete;
    StringBuilder &operator=(StringBuilder &&) = default;
    decltype(auto) begin() { return vec.begin(); }
    decltype(auto) end() { return vec.end(); }
    decltype(auto) begin() const { return vec.begin(); }
    decltype(auto) end() const { return vec.end(); }
    void erase(auto &&iter) { vec.erase(iter); }
};
LC_VSTL_API void to_string(double val, StringBuilder &builder) noexcept;
LC_VSTL_API void to_string(float val, StringBuilder &builder) noexcept;
template<class Ty>
inline void IntegerToString(const Ty Val, StringBuilder &str) noexcept {// convert Val to string
    static_assert(std::is_integral_v<Ty>, "_Ty must be integral");
    using UTy = std::make_unsigned_t<Ty>;
    char Buff[21];// can hold -2^63 and 2^64 - 1, plus NUL
    char *const Buff_end = std::end(Buff);
    char *RNext = Buff_end;
    const auto UVal = static_cast<UTy>(Val);
    if (Val < 0) {
        RNext = UIntegral_to_buff(RNext, static_cast<UTy>(0 - UVal));
        *--RNext = '-';
    } else {
        RNext = UIntegral_to_buff(RNext, UVal);
    }
    str << vstd::string_view{RNext, static_cast<size_t>(Buff_end - RNext)};
}
template<typename T>
    requires std::is_integral_v<T>
[[nodiscard]] inline auto to_string(T val, StringBuilder &builder) noexcept {
    return IntegerToString(static_cast<canonical_integer_t<T>>(val), builder);
}
template<>
struct hash<StringBuilder> {
    using is_transparent = void;
    size_t operator()(string_view s) const noexcept {
        return hash<string_view>{}(s);
    }
    size_t operator()(StringBuilder const &s) const noexcept {
        return hash<string_view>{}(s.view());
    }
};
template<>
struct compare<StringBuilder> {
    int32_t operator()(StringBuilder const &lhs, StringBuilder const &rhs) const noexcept {
        return compare<string_view>{}(lhs.view(), rhs.view());
    }
};
}// namespace vstd
namespace luisa {
template<>
struct hash<vstd::StringBuilder> {
    using is_transparent = void;
    size_t operator()(vstd::StringBuilder const &s) const noexcept {
        return hash<string_view>{}(s.view());
    }
    size_t operator()(string_view s) const noexcept {
        return hash<string_view>{}(s);
    }
};
}// namespace luisa