#pragma once
#include "attributes.hpp"
#include "type_traits.hpp"

namespace luisa::shader {

template<typename Type, uint64 size, uint32 CacheFlags = 0 /*AUTO*/>
struct [[builtin("array")]] Array {
    [[expr("ACCESS")]] Type &operator[](uint32 loc);
    [[ignore]] Array() = default;
    [[ignore]] Array(Array const &) = delete;
    [[ignore]] Array(Array &&) = delete;
    [[ignore]] Array &operator=(Array const &) = delete;
    [[ignore]] Array &operator=(Array &&) = delete;
private:
    Type v[size];
};

}// namespace luisa::shader