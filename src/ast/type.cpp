//
// Created by Mike Smith on 2021/2/6.
//

#include <charconv>
#include <bit>

#include <core/logging.h>
#include <ast/type.h>
#include <ast/type_registry.h>

namespace luisa::compute {

luisa::span<const Type *const> Type::members() const noexcept {
    assert(is_structure());
    return _members;
}

const Type *Type::element() const noexcept {
    assert(is_array() || is_vector() || is_matrix() || is_buffer() || is_texture());
    return _members.front();
}

const Type *Type::from(std::string_view description) noexcept { return detail::TypeRegistry::instance().type_from(description); }
const Type *Type::find(uint64_t hash) noexcept { return detail::TypeRegistry::instance().type_from(hash); }
const Type *Type::at(uint32_t uid) noexcept { return detail::TypeRegistry::instance().type_at(uid); }
size_t Type::count() noexcept { return detail::TypeRegistry::instance().type_count(); }
void Type::traverse(TypeVisitor &visitor) noexcept { detail::TypeRegistry::instance().traverse(visitor); }

}// namespace luisa::compute
