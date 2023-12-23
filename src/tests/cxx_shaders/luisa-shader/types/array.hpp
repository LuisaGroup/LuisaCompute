#pragma once
#include "../attributes.hpp"
#include "../type_traits.hpp"

namespace luisa::shader {

template<typename Type, uint32 size, uint32 CacheFlags = 0 /*AUTO*/>
struct [[builtin("array")]] Array {
    template <typename...Args>
    [[noignore]] constexpr Array(Args... args)
    {
        set<0>(args...);
    }
    constexpr Array() = default;

    template <uint32 start, typename...Args>
    [[noignore]] constexpr void set(Type v, Args... args)
    {
        set(start, v);
        set<start + 1>(args...);
    }
    template <uint32 start>
    [[noignore]] constexpr void set(Type v) { set(start, v); }
    [[noignore]] constexpr void set(uint32 loc, Type v) { access_(loc) = v; }
    [[access]] constexpr Type& access_(uint32 loc) { return v[loc]; }
    [[access]] constexpr Type& operator[](uint32 loc) { return v[loc]; }
    
    [[noignore]] constexpr Type get(uint32 loc) const { return access_(loc); }
    [[access]] constexpr Type access_(uint32 loc) const { return v[loc]; }
    [[access]] constexpr Type operator[](uint32 loc) const { return v[loc]; }
    
private:
    Type v[size];
};

}// namespace luisa::shader