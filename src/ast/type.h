//
// Created by Mike Smith on 2021/2/6.
//

#pragma once

#include <span>
#include <string_view>

namespace luisa::compute {

class TypeRegistry;

class Type {

public:
    enum struct Tag : uint32_t {
        
        BOOL,
        
        FLOAT,
        INT8,
        UINT8,
        INT16,
        UINT16,
        INT32,
        UINT32,
        
        VECTOR,
        MATRIX,
        
        ARRAY,
        
        ATOMIC,
        STRUCTURE,
        
        BUFFER,
        // TODO: TEXTURE
    };

private:
    uint64_t _hash;
    size_t _size;
    size_t _index;
    size_t _alignment;
    uint32_t _element_count;
    Tag _tag;
    std::string_view _description;
    std::span<const Type *> _members;

public:
    template<typename T>
    [[nodiscard]] static const Type *of() noexcept;
    template<typename T>
    [[nodiscard]] static auto of(T &&) noexcept { return of<std::remove_cvref_t<T>>(); }
    [[nodiscard]] static const Type *from(std::string_view description) noexcept;

    [[nodiscard]] bool operator==(const Type &rhs) const noexcept { return _hash == rhs._hash; }
    [[nodiscard]] bool operator!=(const Type &rhs) const noexcept { return !(*this == rhs); }
    [[nodiscard]] bool operator<(const Type &rhs) const noexcept { return _index < rhs._index; }

    [[nodiscard]] constexpr auto hash() const noexcept { return _hash; }
    [[nodiscard]] constexpr auto index() const noexcept { return _index; }
    [[nodiscard]] constexpr auto size() const noexcept { return _size; }
    [[nodiscard]] constexpr auto alignment() const noexcept { return _alignment; }
    [[nodiscard]] constexpr auto tag() const noexcept { return _tag; }
    [[nodiscard]] std::string_view description() const noexcept { return _description; }

    [[nodiscard]] constexpr size_t element_count() const noexcept {
        assert(is_array() || is_vector() || is_matrix());
        return _element_count;
    }

    [[nodiscard]] const auto &members() const noexcept {
        assert(is_structure());
        return _members;
    }

    [[nodiscard]] auto element() const noexcept {
        assert(is_array() || is_atomic() || is_vector() || is_matrix());
        return _members.front();
    }

    [[nodiscard]] constexpr bool is_scalar() const noexcept {
        return static_cast<uint16_t>(_tag) >= static_cast<uint16_t>(Tag::BOOL) && static_cast<uint16_t>(_tag) <= static_cast<uint16_t>(Tag::UINT32);
    }

    [[nodiscard]] constexpr bool is_array() const noexcept { return _tag == Tag::ARRAY; }
    [[nodiscard]] constexpr bool is_vector() const noexcept { return _tag == Tag::VECTOR; }
    [[nodiscard]] constexpr bool is_matrix() const noexcept { return _tag == Tag::MATRIX; }
    [[nodiscard]] constexpr bool is_structure() const noexcept { return _tag == Tag::STRUCTURE; }
    [[nodiscard]] constexpr bool is_atomic() const noexcept { return _tag == Tag::ATOMIC; }
    [[nodiscard]] constexpr bool is_buffer() const noexcept { return _tag == Tag::BUFFER; }
};

}// namespace luisa::compute
