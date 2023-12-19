#pragma once
#include "attributes.hpp"
#include "type_traits.hpp"

namespace luisa::shader {

template<typename Type, uint64 size, uint32 CacheFlags = 0 /*AUTO*/>
struct [[builtin("array")]] Array {
    template <typename...Args>
    [[noignore]] Array(Args... args)
    {
        set<0>(args...);
    }
    [[ignore]] Array() = default;

    template <uint32 start, typename...Args>
    [[noignore]] void set(Type v, Args... args)
    {
        set(start, v);
        set<start + 1>(args...);
    }
    template <uint32 start>
    [[noignore]] void set(Type v) { set(start, v); }
    [[noignore]] void set(uint32 loc, Type v) { access_(loc) = v; }
    [[access]] Type& access_(uint32 loc);
    [[access]] Type& operator[](uint32 loc);
    
    [[noignore]] Type get(uint32 loc) const { return access_(loc); }
    [[access]] Type access_(uint32 loc) const;
    [[access]] Type operator[](uint32 loc) const; 
    
private:
    Type v[size];
};

}// namespace luisa::shader