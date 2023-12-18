#pragma once
#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/string.h>
namespace luisa::compute {
class SerDe {
public:
    template<typename T>
    static void ser_value(T const &t, luisa::vector<std::byte> &vec) noexcept;
    template<typename T>
    static T deser_value(std::byte const *&ptr) noexcept;

    template<typename T>
    static void ser_array(span<const T> t, luisa::vector<std::byte> &vec) noexcept;
    template<typename T>
    static vector<T> deser_array(std::byte const *&ptr) noexcept;
};

template<typename T>
inline void SerDe::ser_value(T const &t, luisa::vector<std::byte> &vec) noexcept {
    static_assert(std::is_trivially_destructible_v<T> && !std::is_pointer_v<T>);
    auto last_len = vec.size();
    vec.push_back_uninitialized(sizeof(T));
    memcpy(vec.data() + last_len, &t, sizeof(T));
}
template<typename T>
inline T SerDe::deser_value(std::byte const *&ptr) noexcept {
    static_assert(std::is_trivially_destructible_v<T> && !std::is_pointer_v<T>);
    T t;
    memcpy(&t, ptr, sizeof(T));
    ptr += sizeof(T);
    return t;
}
template<>
inline void SerDe::ser_value(luisa::string_view const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(t.size(), vec);
    auto last_len = vec.size();
    vec.push_back_uninitialized(t.size());
    memcpy(vec.data() + last_len, t.data(), t.size());
}
template<>
inline void SerDe::ser_value(luisa::string const &t, luisa::vector<std::byte> &vec) noexcept {
    ser_value(luisa::string_view{t}, vec);
}
template<>
inline luisa::string SerDe::deser_value(std::byte const *&ptr) noexcept {
    luisa::string t;
    auto size = deser_value<size_t>(ptr);
    t.clear();
    t.resize(size);
    memcpy(t.data(), ptr, size);
    ptr += size;
    return t;
}
template<typename T>
inline void SerDe::ser_array(span<const T> t, luisa::vector<std::byte> &vec) noexcept {
    ser_value<size_t>(t.size(), vec);
    for (auto &i : t) {
        ser_value<T>(i, vec);
    }
}
template<typename T>
inline vector<T> SerDe::deser_array(std::byte const *&ptr) noexcept {
    vector<T> r;
    auto size = deser_value<size_t>(ptr);
    r.push_back_uninitialized(size);
    for (size_t i = 0; i < size; ++i) {
        new (std::launder(r.data() + i)) T{deser_value<T>(ptr)};
    }
    return r;
}
}// namespace luisa::compute