#pragma once
#include "../attributes.hpp"
#include "../type_traits.hpp"

namespace luisa::shader {

template<typename Type, uint32 size, uint32 Flags = ArrayFlags::None>
struct [[builtin("array")]] Array {
    static constexpr uint32 N = size;

    template<typename... Args>
    [[noignore]] constexpr Array(Args... args) {
        set<0>(args...);
    }
    constexpr Array() = default;

    template<uint32 start, typename... Args>
    [[noignore]] constexpr void set(Type v, Args... args) {
        set(start, v);
        set<start + 1>(args...);
    }
    template<uint32 start>
    [[noignore]] constexpr void set(Type v) { set(start, v); }
    [[noignore]] constexpr void set(uint32 loc, Type v) { access_(loc) = v; }
    [[access]] constexpr Type &access_(uint32 loc) { return v[loc]; }
    [[access]] constexpr Type &operator[](uint32 loc) { return v[loc]; }

    [[noignore]] constexpr Type get(uint32 loc) const { return access_(loc); }
    [[access]] constexpr Type access_(uint32 loc) const { return v[loc]; }
    [[access]] constexpr Type operator[](uint32 loc) const { return v[loc]; }

private:
    // DONT EDIT THIS FIELD LAYOUT
    Type v[size];
};

template<uint32 size>
struct [[builtin("array")]] Array<int32, size, ArrayFlags::Shared> {
    static constexpr uint32 N = size;

    template<typename... Args>
    [[noignore]] constexpr Array(Args... args) {
        set<0>(args...);
    }
    constexpr Array() = default;

    template<uint32 start, typename... Args>
    [[noignore]] constexpr void set(int32 v, Args... args) {
        set(start, v);
        set<start + 1>(args...);
    }
    template<uint32 start>
    [[noignore]] constexpr void set(int32 v) { set(start, v); }
    [[noignore]] constexpr void set(uint32 loc, int32 v) { access_(loc) = v; }
    [[access]] constexpr int32 &access_(uint32 loc) { return v[loc]; }
    [[access]] constexpr int32 &operator[](uint32 loc) { return v[loc]; }

    [[noignore]] constexpr int32 get(uint32 loc) const { return access_(loc); }
    [[access]] constexpr int32 access_(uint32 loc) const { return v[loc]; }
    [[access]] constexpr int32 operator[](uint32 loc) const { return v[loc]; }
    [[callop("ATOMIC_EXCHANGE")]] int32 atomic_exchange(uint32 loc, int32 desired);
    [[callop("ATOMIC_COMPARE_EXCHANGE")]] int32 atomic_compare_exchange(uint32 loc, int32 expected, int32 desired);
    [[callop("ATOMIC_FETCH_ADD")]] int32 atomic_fetch_add(uint32 loc, int32 val);
    [[callop("ATOMIC_FETCH_SUB")]] int32 atomic_fetch_sub(uint32 loc, int32 val);
    [[callop("ATOMIC_FETCH_AND")]] int32 atomic_fetch_and(uint32 loc, int32 val);
    [[callop("ATOMIC_FETCH_OR")]] int32 atomic_fetch_or(uint32 loc, int32 val);
    [[callop("ATOMIC_FETCH_XOR")]] int32 atomic_fetch_xor(uint32 loc, int32 val);
    [[callop("ATOMIC_FETCH_MIN")]] int32 atomic_fetch_min(uint32 loc, int32 val);
    [[callop("ATOMIC_FETCH_MAX")]] int32 atomic_fetch_max(uint32 loc, int32 val);

private:
    // DONT EDIT THIS FIELD LAYOUT
    int32 v[size];
};
template<uint32 size>
struct [[builtin("array")]] Array<uint32, size, ArrayFlags::Shared> {
    static constexpr uint32 N = size;

    template<typename... Args>
    [[noignore]] constexpr Array(Args... args) {
        set<0>(args...);
    }
    constexpr Array() = default;

    template<uint32 start, typename... Args>
    [[noignore]] constexpr void set(uint32 v, Args... args) {
        set(start, v);
        set<start + 1>(args...);
    }
    template<uint32 start>
    [[noignore]] constexpr void set(uint32 v) { set(start, v); }
    [[noignore]] constexpr void set(uint32 loc, uint32 v) { access_(loc) = v; }
    [[access]] constexpr uint32 &access_(uint32 loc) { return v[loc]; }
    [[access]] constexpr uint32 &operator[](uint32 loc) { return v[loc]; }

    [[noignore]] constexpr uint32 get(uint32 loc) const { return access_(loc); }
    [[access]] constexpr uint32 access_(uint32 loc) const { return v[loc]; }
    [[access]] constexpr uint32 operator[](uint32 loc) const { return v[loc]; }
    [[callop("ATOMIC_EXCHANGE")]] uint32 atomic_exchange(uint32 loc, int32 desired);
    [[callop("ATOMIC_COMPARE_EXCHANGE")]] uint32 atomic_compare_exchange(uint32 loc, int32 expected, int32 desired);
    [[callop("ATOMIC_FETCH_ADD")]] uint32 atomic_fetch_add(uint32 loc, int32 val);
    [[callop("ATOMIC_FETCH_SUB")]] uint32 atomic_fetch_sub(uint32 loc, int32 val);
    [[callop("ATOMIC_FETCH_AND")]] uint32 atomic_fetch_and(uint32 loc, int32 val);
    [[callop("ATOMIC_FETCH_OR")]] uint32 atomic_fetch_or(uint32 loc, int32 val);
    [[callop("ATOMIC_FETCH_XOR")]] uint32 atomic_fetch_xor(uint32 loc, int32 val);
    [[callop("ATOMIC_FETCH_MIN")]] uint32 atomic_fetch_min(uint32 loc, int32 val);
    [[callop("ATOMIC_FETCH_MAX")]] uint32 atomic_fetch_max(uint32 loc, int32 val);

private:
    // DONT EDIT THIS FIELD LAYOUT
    uint32 v[size];
};
template<typename Type, uint32 size>
using SharedArray = Array<Type, size, ArrayFlags::Shared>;

}// namespace luisa::shader