#pragma once
#include "attributes.hpp"
#include "type_traits.hpp"


namespace luisa::shader {

template<typename Type, uint32 CacheFlags = 0 /*AUTO*/>
struct [[builtin("buffer")]] Buffer {
    [[callop("BUFFER_READ")]] Type load(uint32 loc);
    [[callop("BUFFER_WRITE")]] void store(uint32 loc, Type value);
};

template<uint32 CacheFlags>
struct [[builtin("buffer")]] Buffer<int32, CacheFlags> {
    [[callop("BUFFER_READ")]] int32 load(uint32 loc);
    [[callop("BUFFER_WRITE")]] void store(uint32 loc, int32 value);
    [[callop("ATOMIC_EXCHANGE")]] int32 atomic_exchange(uint32 loc, int32 desired);
    [[callop("ATOMIC_COMPARE_EXCHANGE")]] int32 atomic_compare_exchange(uint32 loc, int32 expected, int32 desired);
    [[callop("ATOMIC_FETCH_ADD")]] int32 atomic_fetch_add(uint32 loc, int32 val);
    [[callop("ATOMIC_FETCH_sub")]] int32 atomic_fetch_sub(uint32 loc, int32 val);
    [[callop("ATOMIC_FETCH_sub")]] int32 atomic_fetch_and(uint32 loc, int32 val);
    [[callop("ATOMIC_FETCH_sub")]] int32 atomic_fetch_or(uint32 loc, int32 val);
    [[callop("ATOMIC_FETCH_sub")]] int32 atomic_fetch_xor(uint32 loc, int32 val);
    [[callop("ATOMIC_FETCH_sub")]] int32 atomic_fetch_min(uint32 loc, int32 val);
    [[callop("ATOMIC_FETCH_sub")]] int32 atomic_fetch_max(uint32 loc, int32 val);
};
template<uint32 CacheFlags>
struct [[builtin("buffer")]] Buffer<uint32, CacheFlags> {
    [[callop("BUFFER_READ")]] uint32 load(uint32 loc);
    [[callop("BUFFER_WRITE")]] void store(uint32 loc, uint32 value);
    [[callop("ATOMIC_EXCHANGE")]] uint32 atomic_exchange(uint32 loc, uint32 desired);
    [[callop("ATOMIC_COMPARE_EXCHANGE")]] uint32 atomic_compare_exchange(uint32 loc, uint32 expected, uint32 desired);
    [[callop("ATOMIC_FETCH_ADD")]] uint32 atomic_fetch_add(uint32 loc, uint32 val);
    [[callop("ATOMIC_FETCH_sub")]] uint32 atomic_fetch_sub(uint32 loc, uint32 val);
    [[callop("ATOMIC_FETCH_sub")]] uint32 atomic_fetch_and(uint32 loc, uint32 val);
    [[callop("ATOMIC_FETCH_sub")]] uint32 atomic_fetch_or(uint32 loc, uint32 val);
    [[callop("ATOMIC_FETCH_sub")]] uint32 atomic_fetch_xor(uint32 loc, uint32 val);
    [[callop("ATOMIC_FETCH_sub")]] uint32 atomic_fetch_min(uint32 loc, uint32 val);
    [[callop("ATOMIC_FETCH_sub")]] uint32 atomic_fetch_max(uint32 loc, uint32 val);
};
template<uint32 CacheFlags>
struct [[builtin("buffer")]] Buffer<float, CacheFlags> {
    [[callop("BUFFER_READ")]] float load(uint32 loc);
    [[callop("BUFFER_WRITE")]] void store(uint32 loc, float value);
    [[callop("ATOMIC_EXCHANGE")]] float atomic_exchange(uint32 loc, float desired);
    [[callop("ATOMIC_COMPARE_EXCHANGE")]] float atomic_compare_exchange(uint32 loc, float expected, float desired);
    [[callop("ATOMIC_FETCH_ADD")]] float atomic_fetch_add(uint32 loc, float val);
    [[callop("ATOMIC_FETCH_sub")]] float atomic_fetch_sub(uint32 loc, float val);
    [[callop("ATOMIC_FETCH_sub")]] float atomic_fetch_min(uint32 loc, float val);
    [[callop("ATOMIC_FETCH_sub")]] float atomic_fetch_max(uint32 loc, float val);
};
}// namespace luisa::shader