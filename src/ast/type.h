//
// Created by Mike Smith on 2021/2/6.
//

#pragma once

#include <cassert>
#include <string>
#include <vector>
#include <string_view>
#include <span>
#include <memory>

namespace luisa::compute {

class Type;
class TypeRegistry;

struct TypeVisitor {
    virtual void visit(const Type *) noexcept = 0;
};

struct TypeData {
    std::string description;
    std::vector<const Type *> members;
};

class Type {

public:
    enum struct Tag : uint32_t {

        BOOL,
        FLOAT,
        INT,
        UINT,

        VECTOR,
        MATRIX,

        ARRAY,

        ATOMIC,
        STRUCTURE,
        
        BUFFER,
        TEXTURE,
        TEXTURE_HEAP
    };

private:
    uint64_t _hash;
    size_t _size;
    size_t _index;
    size_t _alignment;
    uint32_t _dimension;
    Tag _tag;
    std::unique_ptr<TypeData> _data;

    [[nodiscard]] static TypeRegistry &_registry() noexcept;

public:
    template<typename T>
    [[nodiscard]] static const Type *of() noexcept;
    template<typename T>
    [[nodiscard]] static auto of(T &&) noexcept { return of<std::remove_cvref_t<T>>(); }
    [[nodiscard]] static const Type *from(std::string_view description) noexcept;
    [[nodiscard]] static const Type *at(uint32_t uid) noexcept;
    [[nodiscard]] static size_t count() noexcept;
    static void traverse(TypeVisitor &visitor) noexcept;

    [[nodiscard]] bool operator==(const Type &rhs) const noexcept { return _hash == rhs._hash; }
    [[nodiscard]] bool operator!=(const Type &rhs) const noexcept { return !(*this == rhs); }
    [[nodiscard]] bool operator<(const Type &rhs) const noexcept { return _index < rhs._index; }

    [[nodiscard]] constexpr auto hash() const noexcept { return _hash; }
    [[nodiscard]] constexpr auto index() const noexcept { return _index; }
    [[nodiscard]] constexpr auto size() const noexcept { return _size; }
    [[nodiscard]] constexpr auto alignment() const noexcept { return _alignment; }
    [[nodiscard]] constexpr auto tag() const noexcept { return _tag; }
    [[nodiscard]] std::string_view description() const noexcept;
    [[nodiscard]] constexpr size_t dimension() const noexcept {
        assert(is_array() || is_vector() || is_matrix() || is_texture());
        return _dimension;
    }

    [[nodiscard]] std::span<const Type *const> members() const noexcept;
    [[nodiscard]] const Type *element() const noexcept;

    [[nodiscard]] constexpr bool is_scalar() const noexcept {
        return _tag == Tag::BOOL
               || _tag == Tag::FLOAT
               || _tag == Tag::INT
               || _tag == Tag::UINT;
    }

    [[nodiscard]] constexpr bool is_array() const noexcept { return _tag == Tag::ARRAY; }
    [[nodiscard]] constexpr bool is_vector() const noexcept { return _tag == Tag::VECTOR; }
    [[nodiscard]] constexpr bool is_matrix() const noexcept { return _tag == Tag::MATRIX; }
    [[nodiscard]] constexpr bool is_structure() const noexcept { return _tag == Tag::STRUCTURE; }
    [[nodiscard]] constexpr bool is_atomic() const noexcept { return _tag == Tag::ATOMIC; }
    [[nodiscard]] constexpr bool is_buffer() const noexcept { return _tag == Tag::BUFFER; }
    [[nodiscard]] constexpr bool is_texture() const noexcept { return _tag == Tag::TEXTURE; }
    [[nodiscard]] constexpr bool is_texture_heap() const noexcept { return _tag == Tag::TEXTURE_HEAP; }
};

}// namespace luisa::compute
