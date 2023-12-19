#pragma once
#include "attributes.hpp"
#include "type_traits.hpp"

namespace luisa::shader {

template<typename Type, uint64 size, uint32 CacheFlags = 0 /*AUTO*/>
struct [[builtin("array")]] Array {
    [[ignore]] Array() = default;
     
    template <typename...Args>
    [[scope]] Array(Args... args)
    {
        set<0>(args...);
    }
    
    template <uint32 start>
    [[scope]] void set(Type v)
    {
        set(start, v);
    }
    template <uint32 start, typename...Args>
    [[scope]] void set(Type v, Args... args)
    {
        set(start, v);
        set<start + 1>(args...);
    }
    [[scope]] void set(uint32 loc, Type v) { acess_(loc) = v; }

    [[access]] Type& acess_(uint32 loc);
    [[access]] Type& operator[](uint32 loc);
    [[access]] Type operator[](uint32 loc) const; 
private:
    Type v[size];
};

}// namespace luisa::shader