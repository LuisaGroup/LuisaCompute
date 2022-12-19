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

#include <core/concepts.h>
#include <core/stl.h>

namespace luisa::compute {

template<typename T>
struct array_dimension {
    static constexpr size_t value = 0u;
};

template<typename T, size_t N>
struct array_dimension<T[N]> {
    static constexpr auto value = N;
};

template<typename T, size_t N>
struct array_dimension<std::array<T, N>> {
    static constexpr auto value = N;
};

template<typename T>
constexpr auto array_dimension_v = array_dimension<T>::value;

template<typename T>
struct array_element {
    using type = T;
};

template<typename T, size_t N>
struct array_element<T[N]> {
    using type = T;
};

template<typename T, size_t N>
struct array_element<std::array<T, N>> {
    using type = T;
};

template<typename T>
using array_element_t = typename array_element<T>::type;

template<typename T>
struct is_array : std::false_type {};

template<typename T, size_t N>
struct is_array<T[N]> : std::true_type {};

template<typename T, size_t N>
struct is_array<std::array<T, N>> : std::true_type {};

template<typename T>
constexpr auto is_array_v = is_array<T>::value;

template<typename T>
struct is_tuple : std::false_type {};

template<typename... T>
struct is_tuple<std::tuple<T...>> : std::true_type {};

template<typename T>
constexpr auto is_tuple_v = is_tuple<T>::value;

template<typename T>
struct is_struct : std::false_type {};

template<typename... T>
struct is_struct<std::tuple<T...>> : std::true_type {};

template<typename T>
constexpr auto is_struct_v = is_struct<T>::value;

namespace detail {

template<typename T, size_t>
using array_to_tuple_element_t = T;

template<typename T, size_t... i>
[[nodiscard]] constexpr auto array_to_tuple_impl(std::index_sequence<i...>) noexcept {
    return static_cast<std::tuple<array_to_tuple_element_t<T, i>...> *>(nullptr);
}

}// namespace detail

template<typename T>
struct struct_member_tuple {
    using type = std::tuple<T>;
};

template<typename... T>
struct struct_member_tuple<std::tuple<T...>> {
    using type = std::tuple<T...>;
};

template<typename T, size_t N>
struct struct_member_tuple<std::array<T, N>> {
    using type = std::remove_pointer_t<
        decltype(detail::array_to_tuple_impl<T>(std::make_index_sequence<N>{}))>;
};

template<typename T, size_t N>
struct struct_member_tuple<T[N]> {
    using type = typename struct_member_tuple<std::array<T, N>>::type;
};

template<typename T, size_t N>
struct struct_member_tuple<Vector<T, N>> {
    using type = typename struct_member_tuple<std::array<T, N>>::type;
};

template<size_t N>
struct struct_member_tuple<Matrix<N>> {
    using type = typename struct_member_tuple<std::array<Vector<float, N>, N>>::type;
};

template<typename T>
using struct_member_tuple_t = typename struct_member_tuple<T>::type;

template<typename T>
struct canonical_layout {
    using type = typename canonical_layout<struct_member_tuple_t<T>>::type;
};

template<>
struct canonical_layout<float> {
    using type = std::tuple<float>;
};

template<>
struct canonical_layout<bool> {
    using type = std::tuple<bool>;
};

template<>
struct canonical_layout<int> {
    using type = std::tuple<int>;
};

template<>
struct canonical_layout<uint> {
    using type = std::tuple<uint>;
};

template<typename T>
struct canonical_layout<std::tuple<T>> {
    using type = typename canonical_layout<T>::type;
};

template<typename... T>
struct canonical_layout<std::tuple<T...>> {
    using type = std::tuple<typename canonical_layout<T>::type...>;
};

template<typename T>
using canonical_layout_t = typename canonical_layout<T>::type;

template<typename... T>
struct tuple_join {
    static_assert(always_false_v<T...>);
};

template<typename... A, typename... B, typename... C>
struct tuple_join<std::tuple<A...>, std::tuple<B...>, C...> {
    using type = typename tuple_join<std::tuple<A..., B...>, C...>::type;
};

template<typename... A>
struct tuple_join<std::tuple<A...>> {
    using type = std::tuple<A...>;
};

template<typename... T>
using tuple_join_t = typename tuple_join<T...>::type;

namespace detail {

template<typename L, typename T>
struct linear_layout_impl {
    using type = std::tuple<T>;
};

template<typename... L, typename... T>
struct linear_layout_impl<std::tuple<L...>, std::tuple<T...>> {
    using type = tuple_join_t<std::tuple<L...>, typename linear_layout_impl<std::tuple<>, T>::type...>;
};

}// namespace detail

template<typename T>
using linear_layout = detail::linear_layout_impl<std::tuple<>, canonical_layout_t<T>>;

template<typename T>
using linear_layout_t = typename linear_layout<T>::type;

namespace detail {

template<typename T>
struct dimension_impl {
    static constexpr auto value = dimension_impl<canonical_layout_t<T>>::value;
};

template<typename T, size_t N>
struct dimension_impl<T[N]> {
    static constexpr auto value = N;
};

template<typename T, size_t N>
struct dimension_impl<std::array<T, N>> {
    static constexpr auto value = N;
};

template<typename T, size_t N>
struct dimension_impl<Vector<T, N>> {
    static constexpr auto value = N;
};

template<size_t N>
struct dimension_impl<Matrix<N>> {
    static constexpr auto value = N;
};

template<typename... T>
struct dimension_impl<std::tuple<T...>> {
    static constexpr auto value = sizeof...(T);
};

class TypeRegistry;

}// namespace detail

template<typename T>
using dimension = detail::dimension_impl<std::remove_cvref_t<T>>;

template<typename T>
constexpr auto dimension_v = dimension<T>::value;

class Type;

struct TypeVisitor {
    virtual void visit(const Type *) noexcept = 0;
};

/// Type class
class LC_AST_API Type {

public:
        friend class detail::TypeRegistry;

    /// Type tags
    enum struct Tag : uint32_t {

        BOOL,
        FLOAT,
        INT,
        UINT,

        VECTOR,
        MATRIX,

        ARRAY,
        STRUCTURE,

        BUFFER,
        TEXTURE,
        BINDLESS_ARRAY,
        ACCEL
    };

private:
    uint64_t _hash;
    size_t _size;
    size_t _index;
    size_t _alignment;
    uint32_t _dimension;
    Tag _tag;
    luisa::string _description;
    luisa::vector<const Type *> _members;

public:
    /// Return Type object of type T
    template<typename T>
    [[nodiscard]] static const Type *of() noexcept;
    /// Return Type object of type T
    template<typename T>
    [[nodiscard]] static auto of(T &&) noexcept { return of<std::remove_cvref_t<T>>(); }
    /// Construct Type object from description
    [[nodiscard]] static const Type *from(std::string_view description) noexcept;
    /// Construct Type object from hash
    [[nodiscard]] static const Type *find(uint64_t hash) noexcept;
    /// Construct Type object from uid
    [[nodiscard]] static const Type *at(uint32_t uid) noexcept;
    /// Return type count
    [[nodiscard]] static size_t count() noexcept;
    /// Traverse TypeVisitor
    static void traverse(TypeVisitor &visitor) noexcept;
    static void traverse(const luisa::function<void(const Type *)> &visitor) noexcept;

    /// Compare by hash
    [[nodiscard]] bool operator==(const Type &rhs) const noexcept { return _hash == rhs._hash; }
    /// Compare by hash
    [[nodiscard]] bool operator!=(const Type &rhs) const noexcept { return !(*this == rhs); }
    /// Compare by index
    [[nodiscard]] bool operator<(const Type &rhs) const noexcept { return _index < rhs._index; }
    [[nodiscard]] constexpr auto index() const noexcept { return _index; }
    [[nodiscard]] constexpr auto hash() const noexcept { return _hash; }
    [[nodiscard]] constexpr auto size() const noexcept { return _size; }
    [[nodiscard]] constexpr auto alignment() const noexcept { return _alignment; }
    [[nodiscard]] constexpr auto tag() const noexcept { return _tag; }
    [[nodiscard]] auto description() const noexcept { return luisa::string_view{_description}; }
    [[nodiscard]] constexpr size_t dimension() const noexcept {
        assert(is_array() || is_vector() || is_matrix() || is_texture());
        return _dimension;
    }

    [[nodiscard]] luisa::span<const Type *const> members() const noexcept;
    /// Return pointer to first element
    [[nodiscard]] const Type *element() const noexcept;

    /// Scalar = bool || float || int || uint
    [[nodiscard]] constexpr bool is_scalar() const noexcept {
        return _tag == Tag::BOOL
               || _tag == Tag::FLOAT
               || _tag == Tag::INT
               || _tag == Tag::UINT;
    }

    /// Basic = scalar || vector || matrix
    [[nodiscard]] constexpr auto is_basic() const noexcept {
        return is_scalar() || is_vector() || is_matrix();
    }

    [[nodiscard]] constexpr bool is_array() const noexcept { return _tag == Tag::ARRAY; }
    [[nodiscard]] constexpr bool is_vector() const noexcept { return _tag == Tag::VECTOR; }
    [[nodiscard]] constexpr bool is_matrix() const noexcept { return _tag == Tag::MATRIX; }
    [[nodiscard]] constexpr bool is_structure() const noexcept { return _tag == Tag::STRUCTURE; }
    [[nodiscard]] constexpr bool is_buffer() const noexcept { return _tag == Tag::BUFFER; }
    [[nodiscard]] constexpr bool is_texture() const noexcept { return _tag == Tag::TEXTURE; }
    [[nodiscard]] constexpr bool is_bindless_array() const noexcept { return _tag == Tag::BINDLESS_ARRAY; }
    [[nodiscard]] constexpr bool is_accel() const noexcept { return _tag == Tag::ACCEL; }
};

}// namespace luisa::compute
