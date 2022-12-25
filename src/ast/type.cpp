//
// Created by Mike Smith on 2021/2/6.
//

#include <bit>
#include <utility>

#include <ast/type.h>
#include <ast/type_registry.h>

namespace luisa::compute {

luisa::span<const Type *const> Type::members() const noexcept {
    assert(is_structure());
    return _members;
}

const Type *Type::element() const noexcept {
    if (is_scalar()) { return this; }
    assert(is_array() || is_vector() || is_matrix() || is_buffer() || is_texture());
    return _members.front();
}

const Type *Type::from(std::string_view description) noexcept { return detail::TypeRegistry::instance().type_from(description); }
const Type *Type::find(uint64_t hash) noexcept { return detail::TypeRegistry::instance().type_from(hash); }
const Type *Type::at(uint32_t uid) noexcept { return detail::TypeRegistry::instance().type_at(uid); }
size_t Type::count() noexcept { return detail::TypeRegistry::instance().type_count(); }
void Type::traverse(TypeVisitor &visitor) noexcept { detail::TypeRegistry::instance().traverse(visitor); }

class TypeVisitorAdapter final : public TypeVisitor {

private:
    luisa::function<void(const Type *)> _visitor;

public:
    explicit TypeVisitorAdapter(luisa::function<void(const Type *)> visitor) noexcept
        : _visitor(std::move(visitor)) {}
    void visit(const Type *type) noexcept override { _visitor(type); }
};

void Type::traverse(const function<void(const Type *)> &visitor) noexcept {
    TypeVisitorAdapter adapter{visitor};
    traverse(adapter);
}

bool Type::operator==(const Type &rhs) const noexcept { return _hash == rhs._hash; }
bool Type::operator<(const Type &rhs) const noexcept { return _index < rhs._index; }

uint Type::index() const noexcept { return _index; }
uint64_t Type::hash() const noexcept { return _hash; }
size_t Type::size() const noexcept { return _size; }
size_t Type::alignment() const noexcept { return _alignment; }
Type::Tag Type::tag() const noexcept { return _tag; }
luisa::string_view Type::description() const noexcept { return _description; }

uint Type::dimension() const noexcept {
    assert(is_array() || is_vector() || is_matrix() || is_texture());
    return _dimension;
}

bool Type::is_scalar() const noexcept {
    return _tag == Tag::BOOL ||
           _tag == Tag::FLOAT ||
           _tag == Tag::INT ||
           _tag == Tag::UINT;
}

bool Type::is_basic() const noexcept {
    return is_scalar() || is_vector() || is_matrix();
}

bool Type::is_array() const noexcept { return _tag == Tag::ARRAY; }
bool Type::is_vector() const noexcept { return _tag == Tag::VECTOR; }
bool Type::is_matrix() const noexcept { return _tag == Tag::MATRIX; }
bool Type::is_structure() const noexcept { return _tag == Tag::STRUCTURE; }
bool Type::is_buffer() const noexcept { return _tag == Tag::BUFFER; }
bool Type::is_texture() const noexcept { return _tag == Tag::TEXTURE; }
bool Type::is_bindless_array() const noexcept { return _tag == Tag::BINDLESS_ARRAY; }
bool Type::is_accel() const noexcept { return _tag == Tag::ACCEL; }
bool Type::is_custom() const noexcept { return _tag == Tag::CUSTOM; }

}// namespace luisa::compute
