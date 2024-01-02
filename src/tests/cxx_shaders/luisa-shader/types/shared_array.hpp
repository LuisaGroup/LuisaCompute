#pragma once
#include "../attributes.hpp"
#include "../type_traits.hpp"

namespace luisa::shader {

template<typename Type, uint32 size>
struct [[builtin("shared_array")]] SharedArray {
    [[expr("ACCESS")]] Type &operator[](uint32 loc);
    [[ignore]] SharedArray() = default;
    [[ignore]] SharedArray(SharedArray const &) = delete;
    [[ignore]] SharedArray(SharedArray &&) = delete;
    [[ignore]] SharedArray &operator=(SharedArray const &) = delete;
    [[ignore]] SharedArray &operator=(SharedArray &&) = delete;
private:
    Type v[size];
};

}// namespace luisa::shader