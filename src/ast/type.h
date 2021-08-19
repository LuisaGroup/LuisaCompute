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

template<typename T>
struct struct_member_tuple {
    using type = std::tuple<>;
};

template<typename... T>
struct struct_member_tuple<std::tuple<T...>> {
    using type = std::tuple<T...>;
};

template<typename T>
using struct_member_tuple_t = typename struct_member_tuple<T>::type;

namespace detail {

template<typename T, size_t>
using array_to_tuple_element_t = T;

template<typename T, size_t... i>
[[nodiscard]] constexpr auto array_to_tuple_impl(std::index_sequence<i...>) noexcept {
    return static_cast<std::tuple<array_to_tuple_element_t<T, i>...> *>(nullptr);
}

}// namespace detail

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

template<typename T, size_t N>
struct canonical_layout<std::array<T, N>> {
    using type = std::remove_pointer_t<
        decltype(detail::array_to_tuple_impl<
                 typename canonical_layout<T>::type>(std::make_index_sequence<N>{}))>;
};

template<typename T, size_t N>
struct canonical_layout<T[N]> {
    using type = typename canonical_layout<std::array<T, N>>::type;
};

template<typename T, size_t N>
struct canonical_layout<Vector<T, N>> {
    using type = typename canonical_layout<std::array<T, N>>::type;
};

template<size_t N>
struct canonical_layout<Matrix<N>> {
    using type = typename canonical_layout<std::array<Vector<float, N>, N>>::type;
};

template<typename T>
using canonical_layout_t = typename canonical_layout<T>::type;

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

}// namespace detail

template<typename T>
using dimension = detail::dimension_impl<std::remove_cvref_t<T>>;

template<typename T>
constexpr auto dimension_v = dimension<T>::value;

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
        STRUCTURE,

        BUFFER,
        TEXTURE,
        HEAP,
        ACCEL
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
    [[nodiscard]] constexpr bool is_buffer() const noexcept { return _tag == Tag::BUFFER; }
    [[nodiscard]] constexpr bool is_texture() const noexcept { return _tag == Tag::TEXTURE; }
    [[nodiscard]] constexpr bool is_heap() const noexcept { return _tag == Tag::HEAP; }
    [[nodiscard]] constexpr bool is_accel() const noexcept { return _tag == Tag::ACCEL; }
};

}// namespace luisa::compute
