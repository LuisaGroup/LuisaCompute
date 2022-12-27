//
// Created by Mike Smith on 2021/2/6.
//

#include <utility>

#include <core/logging.h>
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

const Type *Type::array(const Type *elem, size_t n) noexcept {
    return from(luisa::format("array<{},{}>", elem->description(), n));
}

const Type *Type::vector(const Type *elem, size_t n) noexcept {
    LUISA_ASSERT(n >= 2 && n <= 4, "Invalid vector dimension.");
    LUISA_ASSERT(elem->is_scalar(), "Vector element must be a scalar.");
    return from(luisa::format("vector<{},{}>", elem->description(), n));
}

const Type *Type::matrix(size_t n) noexcept {
    LUISA_ASSERT(n >= 2 && n <= 4, "Invalid matrix dimension.");
    return from(luisa::format("matrix<{}>", n));
}

const Type *Type::structure(size_t alignment, std::initializer_list<const Type *> members) noexcept {
    LUISA_ASSERT(alignment == 4u || alignment == 8u || alignment == 16u,
                 "Invalid structure alignment {} (must be 4, 8, or 16).",
                 alignment);
    auto desc = luisa::format("struct<{}", alignment);
    for (auto member : members) {
        desc.append(",").append(member->description());
    }
    desc.append(">");
    return from(desc);
}

const Type *Type::structure(std::initializer_list<const Type *> members) noexcept {
    auto alignment = 4u;
    for (auto m : members) { alignment = std::max<size_t>(m->alignment(), alignment); }
    return structure(alignment, members);
}

}// namespace luisa::compute
