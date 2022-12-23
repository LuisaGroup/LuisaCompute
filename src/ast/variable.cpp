//
// Created by Mike Smith on 2022/12/19.
//

#include <core/stl/hash.h>
#include <ast/variable.h>

namespace luisa::compute {

uint64_t Variable::hash() const noexcept {
    using namespace std::string_view_literals;
    static thread_local auto seed = hash_value("__hash_variable"sv);
    auto u0 = static_cast<uint64_t>(_uid);
    auto u1 = static_cast<uint64_t>(_tag);
    std::array a{u0 | (u1 << 32u), _type->hash()};
    return hash64(&a, sizeof(a), seed);
}

}// namespace luisa::compute
