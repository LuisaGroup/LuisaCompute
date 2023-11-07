#include <luisa/core/stl/hash.h>
#include <luisa/ast/variable.h>

namespace luisa::compute {

uint64_t Variable::hash() const noexcept {
    using namespace std::string_view_literals;
    static auto seed = hash_value("__hash_variable"sv);
    auto u0 = static_cast<uint64_t>(_uid);
    auto u1 = static_cast<uint64_t>(_tag);
    return hash_combine({u0 | (u1 << 32u), _type->hash()}, seed);
}

Variable::Variable(const Type *type,
                   Variable::Tag tag,
                   uint32_t uid) noexcept
        : _type{type}, _uid{uid}, _tag{tag} {}

}// namespace luisa::compute

